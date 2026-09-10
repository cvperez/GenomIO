#!/usr/bin/env python3
"""
Experiment 6: Retrieval Querying — Scaled (50 CDS × 20 species = 1,000 records)
Scales the Experiment 5 retrieval evaluation to 1,000 records across 20 species.
Each species contributes 50 records, giving 49 relevant documents per query.

Query types:
  1. DNA→DNA  : DNA query → Index A (DNABERT-S, 768-dim) → ranked DNA records
  2. Text→DNA : text query → Index B (MiniLM, 384-dim) → ranked IDs → DNA records
  3. DNA→Text : DNA query → Index A → ranked IDs → metadata records

Metrics: P@1, P@5, P@10, P@20, Recall@10, Recall@20, MRR (leave-self-out).

Run from repo root:
    python embedder_benchmark/experiment_6_retrieval_scale/experiment_6_retrieval_scale.py
"""
import glob
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

SEQS_PER_SPECIES = 50
SEQ_MIN_LEN = 300
SEQ_MAX_LEN = 900
TEXT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Organism names come from rag_corpus_uniform/organisms.tsv, which
# scripts/download_genomes_uniform.sh writes as it resolves each species. Earlier
# revisions of these experiments hardcoded the table here and several entries were wrong
# (GCF_000013045 is Salinibacter ruber, not Streptococcus pneumoniae; GCF_000020225 is
# Akkermansia muciniphila, not Mycobacterium tuberculosis). The labels were internally
# consistent per species, so every silhouette, P@k and Recall number is unaffected, but
# the organism strings printed beside them were not real.
def _load_organism_names() -> dict:
    path = os.path.join(CORPUS_DIR, "organisms.tsv")
    names = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) >= 2:
                names[fields[0]] = fields[1]
    return names


ORGANISM_NAMES = _load_organism_names()

K_VALUES = [1, 5, 10, 20]
RECALL_K = [10, 20]


# ── Data loading ────────────────────────────────────────────────────────────

def build_metadata_text(description: str, organism: str) -> str:
    fields = re.findall(r'\[([^\]]+)\]', description)
    clean = [f for f in fields if not f.startswith("gbkey=") and not f.startswith("location=")]
    return f"organism={organism} " + " ".join(clean)


def select_sequences() -> tuple[list[dict], list[str]]:
    fna_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.fna")))
    records, species_tags = [], []
    for label, fpath in enumerate(fna_files):
        basename = os.path.basename(fpath)
        accession = "_".join(basename.split("_")[:2])
        organism = ORGANISM_NAMES.get(accession, accession)
        species_tags.append(accession)
        count = 0
        for rec in SeqIO.parse(fpath, "fasta"):
            seq = str(rec.seq).upper().strip()
            if SEQ_MIN_LEN <= len(seq) <= SEQ_MAX_LEN:
                records.append({
                    "sequence": seq,
                    "metadata": build_metadata_text(rec.description, organism),
                    "label":    label,
                    "organism": organism,
                    "accession": accession,
                })
                count += 1
                if count == SEQS_PER_SPECIES:
                    break
        log.info(f"  [{label:02d}] {accession} ({organism}): {count} records")
        species_tags[-1] = accession
    return records, species_tags


# ── Embeddings ──────────────────────────────────────────────────────────────

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    embedding = (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return embedding.squeeze(0).cpu().numpy()


def embed_dnabert_s(sequences: list[str]) -> np.ndarray:
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
        for i, seq in enumerate(sequences):
            enc = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            out = model(**enc)
            lh = out[0] if isinstance(out, tuple) else out.last_hidden_state
            embeddings.append(mean_pool(lh, enc["attention_mask"]))
            if (i + 1) % 100 == 0:
                log.info(f"  DNABERT-S: {i+1}/{len(sequences)} embedded")
    del model, tokenizer
    return np.stack(embeddings)


def embed_text(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    log.info(f"Loading {TEXT_MODEL_ID} ...")
    model = SentenceTransformer(TEXT_MODEL_ID)
    embs = []
    bs = 64
    for i in range(0, len(texts), bs):
        embs.append(model.encode(texts[i:i+bs], normalize_embeddings=True,
                                 show_progress_bar=False))
        if (i + bs) % 200 == 0 or i + bs >= len(texts):
            log.info(f"  MiniLM: {min(i+bs, len(texts))}/{len(texts)} embedded")
    del model
    return np.vstack(embs)


# ── Retrieval metrics ───────────────────────────────────────────────────────

def evaluate_retrieval(sim_matrix: np.ndarray, labels: np.ndarray,
                       query_type: str) -> pd.DataFrame:
    """
    Leave-self-out retrieval evaluation.
    For each query i, rank all other records by sim_matrix[i],
    compute P@k, Recall@k, and RR against same-species ground truth.
    """
    N = len(labels)
    n_relevant_per_query = np.array([
        np.sum(labels == labels[i]) - 1 for i in range(N)
    ])

    rows = []
    for i in range(N):
        sims = sim_matrix[i].copy()
        sims[i] = -2.0  # exclude self

        ranked_idx = np.argsort(sims)[::-1]
        relevant = labels[ranked_idx] == labels[i]
        n_rel = n_relevant_per_query[i]

        if n_rel == 0:
            continue

        row = {"query_type": query_type, "query_label": int(labels[i]),
               "n_relevant": int(n_rel)}

        for k in K_VALUES:
            top_k_rel = relevant[:k].sum()
            row[f"P@{k}"]      = float(top_k_rel / k)
            row[f"Recall@{k}"] = float(top_k_rel / n_rel)

        # MRR
        first_hit = np.argmax(relevant)
        row["RR"] = float(1.0 / (first_hit + 1)) if relevant[first_hit] else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


# ── Visualisation ───────────────────────────────────────────────────────────

def plot_metric_by_species(df: pd.DataFrame, metric: str, query_type: str,
                           species_tags: list[str], filename: str) -> None:
    means = df.groupby("query_label")[metric].mean().reset_index()
    means["species"] = means["query_label"].apply(
        lambda l: species_tags[l].split("_")[0] + "_" + species_tags[l].split("_")[1]
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(means["species"], means[metric], color="steelblue", alpha=0.8)
    ax.set_title(f"{metric} by Species — {query_type}", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric)
    ax.set_xticklabels(means["species"], rotation=45, ha="right", fontsize=7)
    ax.axhline(means[metric].mean(), color="red", linestyle="--", linewidth=1,
               label=f"Mean={means[metric].mean():.3f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_summary_bar(summary_df: pd.DataFrame, filename: str) -> None:
    metrics = [f"P@{k}" for k in K_VALUES] + ["MRR"]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (_, row) in enumerate(summary_df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=row.name, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Performance by Query Type", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=== Selecting sequences ===")
    records, species_tags = select_sequences()
    seqs   = [r["sequence"] for r in records]
    metas  = [r["metadata"] for r in records]
    labels = np.array([r["label"] for r in records])
    N = len(records)
    print(f"\n{N} records across {len(species_tags)} species "
          f"({SEQS_PER_SPECIES}/species, {SEQ_MIN_LEN}–{SEQ_MAX_LEN} bp)")

    # ── Build Index A: DNA ─────────────────────────────────────────────────
    log.info("=== Building Index A: DNA (DNABERT-S) ===")
    t0 = time.perf_counter()
    dna_embs = embed_dnabert_s(seqs)
    dna_build = time.perf_counter() - t0
    log.info(f"Index A: {dna_embs.shape} in {dna_build:.1f}s")

    # ── Build Index B: Text ────────────────────────────────────────────────
    log.info("=== Building Index B: Metadata (all-MiniLM-L6-v2) ===")
    t0 = time.perf_counter()
    txt_embs = embed_text(metas)
    txt_build = time.perf_counter() - t0
    log.info(f"Index B: {txt_embs.shape} in {txt_build:.1f}s")

    # ── Similarity matrices ────────────────────────────────────────────────
    log.info("Computing DNA–DNA similarity matrix ...")
    dna_sim = cosine_similarity(dna_embs)

    log.info("Computing Text–Text similarity matrix ...")
    txt_sim = cosine_similarity(txt_embs)

    # ── Evaluate all 3 query types ─────────────────────────────────────────
    query_configs = [
        ("DNA→DNA",  dna_sim, "DNA query → DNA index"),
        ("Text→DNA", txt_sim, "text query → text index → IDs → DNA records"),
        ("DNA→Text", dna_sim, "DNA query → DNA index → IDs → metadata records"),
    ]

    all_dfs = []
    for qtype, sim_matrix, description in query_configs:
        print(f"\n{'='*60}\n{qtype}: {description}\n{'='*60}")
        df = evaluate_retrieval(sim_matrix, labels, qtype)
        all_dfs.append(df)

        # Per-query-type aggregate
        agg = df[[f"P@{k}" for k in K_VALUES] + [f"Recall@{k}" for k in RECALL_K] + ["RR"]].mean()
        print(f"  n_queries: {len(df)}  (all species, leave-self-out)")
        for k in K_VALUES:
            print(f"  P@{k:2d}:       {agg[f'P@{k}']:.4f}")
        for k in RECALL_K:
            print(f"  Recall@{k:2d}:  {agg[f'Recall@{k}']:.4f}")
        print(f"  MRR:       {agg['RR']:.4f}")

        plot_metric_by_species(df, "P@1",  qtype, species_tags,
                               f"experiment_6_{qtype.replace('→','_')}_p1_by_species.png")
        plot_metric_by_species(df, "P@10", qtype, species_tags,
                               f"experiment_6_{qtype.replace('→','_')}_p10_by_species.png")

    all_eval = pd.concat(all_dfs, ignore_index=True)

    # ── Summary table ──────────────────────────────────────────────────────
    metric_cols = [f"P@{k}" for k in K_VALUES] + [f"Recall@{k}" for k in RECALL_K] + ["RR"]
    summary = all_eval.groupby("query_type")[metric_cols].mean().round(4)
    summary = summary.rename(columns={"RR": "MRR"})

    print("\n\n=== SUMMARY TABLE ===")
    print(summary.to_string())

    plot_summary_bar(summary, "experiment_6_summary_bar.png")

    # ── Save ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "experiment_6_results.csv")
    all_eval.to_csv(csv_path, index=False)
    summary_path = os.path.join(OUTPUT_DIR, "experiment_6_summary.csv")
    summary.to_csv(summary_path)

    print(f"\n=== INDEX STATISTICS ===")
    print(f"  Index A (DNA/DNABERT-S):  {dna_embs.shape}  build: {dna_build:.1f}s"
          f"  mem: {dna_embs.nbytes/1024:.0f} KB")
    print(f"  Index B (Text/MiniLM):    {txt_embs.shape}  build: {txt_build:.1f}s"
          f"  mem: {txt_embs.nbytes/1024:.0f} KB")

    print(f"\nResults saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
