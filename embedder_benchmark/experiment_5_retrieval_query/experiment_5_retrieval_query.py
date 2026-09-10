#!/usr/bin/env python3
"""
Experiment 5: Retrieval Querying — Dual-Modality (5 CDS records)
Tests all three query types against in-memory dual indices built from the
5 CDS records of Experiment 1, using the confirmed dual-modality strategy:
  - Index A: DNA sequences embedded with DNABERT-S (768-dim)
  - Index B: Metadata text embedded with all-MiniLM-L6-v2 (384-dim)

Query types:
  1. DNA→DNA  : DNA query → search Index A → return ranked DNA records
  2. Text→DNA : text query → search Index B → ranked IDs → return DNA records
  3. DNA→Text : DNA query → search Index A → ranked IDs → return metadata records

Evaluation: Precision@k (k=1,2,3) and MRR using leave-self-out ranking.
Ground truth: same species label = relevant document.

Run from repo root:
    python embedder_benchmark/experiment_5_retrieval_query/experiment_5_retrieval_query.py
"""
import os
import re
import time
import logging
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_DIR = os.path.join(REPO_ROOT, "rag_corpus_uniform")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SEQ_MIN_LEN = 300
SEQ_MAX_LEN = 900
TEXT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

SEQUENCE_SOURCES = [
    ("seq1_TT", "GCF_000008125.1_ASM812v1_cds_from_genomic.fna", 1, 0, "Thermus thermophilus"),
    ("seq2_TT", "GCF_000008125.1_ASM812v1_cds_from_genomic.fna", 2, 0, "Thermus thermophilus"),
    ("seq3_CT", "GCF_000008725.1_ASM872v1_cds_from_genomic.fna", 1, 1, "Chlamydia trachomatis"),
    ("seq4_SA", "GCF_000009005.1_ASM900v1_cds_from_genomic.fna", 1, 2, "Staphylococcus aureus"),
    ("seq5_DM", "GCF_000009025.1_ASM902v1_cds_from_genomic.fna", 1, 3, "Dehalococcoides mccartyi"),
]

AXIS_LABELS = [
    "T.thermophilus_1",
    "T.thermophilus_2",
    "C.trachomatis",
    "S.aureus",
    "D.mccartyi",
]
SPECIES_LABELS = [0, 0, 1, 2, 3]


# ── Data loading ────────────────────────────────────────────────────────────

def build_metadata_text(description: str, organism: str) -> str:
    fields = re.findall(r'\[([^\]]+)\]', description)
    clean = [f for f in fields if not f.startswith("gbkey=") and not f.startswith("location=")]
    return f"organism={organism} " + " ".join(clean)


def select_sequences() -> list[dict]:
    results = []
    for tag, fname, nth, label, organism in SEQUENCE_SOURCES:
        fpath = os.path.join(CORPUS_DIR, fname)
        count = 0
        for rec in SeqIO.parse(fpath, "fasta"):
            seq = str(rec.seq).upper().strip()
            if SEQ_MIN_LEN <= len(seq) <= SEQ_MAX_LEN:
                count += 1
                if count == nth:
                    results.append({
                        "id":       tag,
                        "tag":      AXIS_LABELS[len(results)],
                        "sequence": seq,
                        "metadata": build_metadata_text(rec.description, organism),
                        "label":    label,
                        "organism": organism,
                    })
                    break
    return results


# ── Embeddings ──────────────────────────────────────────────────────────────

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    embedding = (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return embedding.squeeze(0).cpu().numpy()


def embed_dnabert_s(inputs: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    model_name = "zhihan1996/DNABERT-S"
    log.info(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.use_flash_attn = False
    model = AutoModel.from_pretrained(
        model_name, config=config, trust_remote_code=True, low_cpu_mem_usage=False
    )
    model.eval()
    embeddings = []
    with torch.no_grad():
        for inp in inputs:
            enc = tokenizer(inp, return_tensors="pt", padding=True, truncation=True)
            out = model(**enc)
            last_hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
            embeddings.append(mean_pool(last_hidden, enc["attention_mask"]))
    del model, tokenizer
    return np.stack(embeddings)


def embed_text(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    log.info(f"Loading {TEXT_MODEL_ID} ...")
    model = SentenceTransformer(TEXT_MODEL_ID)
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    del model
    return np.array(embs)


# ── Retrieval engine ────────────────────────────────────────────────────────

def retrieve(query_emb: np.ndarray, index_embs: np.ndarray,
             records: list[dict], query_idx: int, k: int = 4) -> list[dict]:
    """
    Rank all index records by cosine similarity to query_emb,
    excluding the query record itself. Return top-k with scores.
    """
    sims = cosine_similarity(query_emb.reshape(1, -1), index_embs)[0]
    ranked = np.argsort(sims)[::-1]
    results = []
    for idx in ranked:
        if idx == query_idx:
            continue
        results.append({
            "rank":      len(results) + 1,
            "id":        records[idx]["id"],
            "tag":       records[idx]["tag"],
            "label":     records[idx]["label"],
            "organism":  records[idx]["organism"],
            "score":     float(sims[idx]),
            "sequence":  records[idx]["sequence"][:60] + "...",
            "metadata":  records[idx]["metadata"],
        })
        if len(results) == k:
            break
    return results


# ── Evaluation metrics ──────────────────────────────────────────────────────

def precision_at_k(results: list[dict], query_label: int, k: int) -> float:
    top_k = results[:k]
    relevant = sum(1 for r in top_k if r["label"] == query_label)
    return relevant / k


def reciprocal_rank(results: list[dict], query_label: int) -> float:
    for r in results:
        if r["label"] == query_label:
            return 1.0 / r["rank"]
    return 0.0


def has_same_species(records: list[dict], query_label: int, query_idx: int) -> bool:
    """Check if any other record shares the query's species label."""
    return any(r["label"] == query_label for i, r in enumerate(records) if i != query_idx)


# ── Visualisation ───────────────────────────────────────────────────────────

def plot_retrieval_heatmap(sim_matrix: np.ndarray, title: str,
                           row_labels: list[str], col_labels: list[str],
                           filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(sim_matrix, annot=True, fmt=".3f", cmap="viridis",
                vmin=0, vmax=1, xticklabels=col_labels, yticklabels=row_labels,
                ax=ax, linewidths=0.5, linecolor="white")
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=== Loading 5 CDS records ===")
    records = select_sequences()
    tags    = [r["tag"]      for r in records]
    seqs    = [r["sequence"] for r in records]
    metas   = [r["metadata"] for r in records]
    labels  = [r["label"]    for r in records]
    N = len(records)

    print("\nRecords loaded:")
    for r in records:
        print(f"  {r['id']:10s} | {r['organism']:30s} | {r['metadata'][:80]}...")

    # ── Build indices ───────────────────────────────────────────────────────
    log.info("=== Building Index A: DNA embeddings (DNABERT-S) ===")
    t0 = time.perf_counter()
    dna_embs = embed_dnabert_s(seqs)
    dna_build_time = time.perf_counter() - t0

    log.info("=== Building Index B: Metadata embeddings (all-MiniLM-L6-v2) ===")
    t0 = time.perf_counter()
    txt_embs = embed_text(metas)
    txt_build_time = time.perf_counter() - t0

    log.info(f"Index A built: {dna_embs.shape} in {dna_build_time:.2f}s")
    log.info(f"Index B built: {txt_embs.shape} in {txt_build_time:.2f}s")

    # ── Full similarity matrices ────────────────────────────────────────────
    dna_dna_sim = cosine_similarity(dna_embs)
    txt_txt_sim = cosine_similarity(txt_embs)

    plot_retrieval_heatmap(dna_dna_sim, "DNA–DNA Similarity (DNABERT-S)",
                           tags, tags, "experiment_5_dna_dna_sim.png")
    plot_retrieval_heatmap(txt_txt_sim, f"Text–Text Similarity (MiniLM)",
                           tags, tags, "experiment_5_txt_txt_sim.png")

    # ── Query evaluation ────────────────────────────────────────────────────
    query_types = [
        ("DNA→DNA",  dna_embs, dna_embs, "DNA query → DNA index → ranked DNA records"),
        ("Text→DNA", txt_embs, txt_embs, "text query → text index → ranked IDs → DNA records"),
        ("DNA→Text", dna_embs, dna_embs, "DNA query → DNA index → ranked IDs → metadata records"),
    ]

    all_eval_rows = []
    K_MAX = 3

    for qtype, query_embs, index_embs, description in query_types:
        print(f"\n{'='*65}")
        print(f"Query type: {qtype}")
        print(f"Description: {description}")
        print(f"{'='*65}")

        p_at = {k: [] for k in range(1, K_MAX + 1)}
        rr_list = []

        for qi in range(N):
            query_label = labels[qi]
            if not has_same_species(records, query_label, qi):
                # Singleton species — skip P@k/MRR (no correct answer exists)
                print(f"\n  Query [{tags[qi]}]: singleton species — P@k N/A")
                continue

            results = retrieve(query_embs[qi], index_embs, records, qi, k=N - 1)

            # For DNA→Text: override displayed field to metadata
            display_field = "metadata" if qtype == "DNA→Text" else "tag"

            print(f"\n  Query [{tags[qi]}] (label={query_label}):")
            for r in results:
                match = "✓" if r["label"] == query_label else " "
                val = r[display_field] if display_field == "metadata" else r["tag"]
                print(f"    Rank {r['rank']}: {match} {val[:70]}  (score={r['score']:.4f})")

            for k in range(1, K_MAX + 1):
                p = precision_at_k(results, query_label, k)
                p_at[k].append(p)

            rr = reciprocal_rank(results, query_label)
            rr_list.append(rr)

            row = {
                "query_type": qtype,
                "query_tag":  tags[qi],
                "query_label": query_label,
            }
            for k in range(1, K_MAX + 1):
                row[f"P@{k}"] = round(precision_at_k(results, query_label, k), 4)
            row["RR"] = round(rr, 4)
            all_eval_rows.append(row)

        # Aggregate
        print(f"\n  --- Aggregate ({qtype}) ---")
        for k in range(1, K_MAX + 1):
            vals = p_at[k]
            if vals:
                print(f"  Mean P@{k}: {np.mean(vals):.4f}  (n={len(vals)} queries with ground truth)")
        if rr_list:
            print(f"  MRR:       {np.mean(rr_list):.4f}")

    # ── Save results ────────────────────────────────────────────────────────
    df = pd.DataFrame(all_eval_rows)
    print("\n\n=== FULL EVALUATION TABLE ===")
    print(df.to_string(index=False))

    csv_path = os.path.join(OUTPUT_DIR, "experiment_5_results.csv")
    df.to_csv(csv_path, index=False)

    # Summary by query type
    print("\n=== SUMMARY BY QUERY TYPE ===")
    summary = df.groupby("query_type")[[f"P@{k}" for k in range(1, K_MAX+1)] + ["RR"]].mean()
    print(summary.round(4).to_string())
    summary_path = os.path.join(OUTPUT_DIR, "experiment_5_summary.csv")
    summary.to_csv(summary_path)

    print(f"\nResults saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")

    # ── Index stats ─────────────────────────────────────────────────────────
    print(f"\n=== INDEX STATISTICS ===")
    print(f"  Index A (DNA/DNABERT-S):   {dna_embs.shape}  build time: {dna_build_time:.2f}s")
    print(f"  Index B (Text/MiniLM):     {txt_embs.shape}  build time: {txt_build_time:.2f}s")
    print(f"  Memory (A): {dna_embs.nbytes / 1024:.1f} KB")
    print(f"  Memory (B): {txt_embs.nbytes / 1024:.1f} KB")


if __name__ == "__main__":
    main()
