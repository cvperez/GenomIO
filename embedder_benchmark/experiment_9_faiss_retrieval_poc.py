#!/usr/bin/env python3
"""
Experiment 9: FAISS-Based Retrieval — Proof of Concept (5 CDS records)
Re-runs the Experiment 5 retrieval evaluation routing queries through FAISS
indices instead of brute-force cosine similarity, confirming the deployed
system matches exact cosine search quality.

Index types (per Experiment 8 recommendation for n=5):
  - Index A: IndexFlatIP (DNA, 768-dim) — exact
  - Index B: IndexFlatIP (Text, 384-dim) — exact

Query types:
  1. DNA→DNA  : DNA query → FAISS Index A → ranked DNA records
  2. Text→DNA : text query → FAISS Index B → ID bridge → DNA records
  3. DNA→Text : DNA query → FAISS Index A → ID bridge → metadata records

Metrics: P@1, P@2, P@3, MRR (leave-self-out, same as Exp 5)
Additional: FAISS build time, query latency, accuracy vs brute-force, memory.

Run from repo root:
    python experiments/experiment_9_faiss_retrieval_poc.py
"""
import os
import re
import time
import logging
import warnings

import faiss
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(REPO_ROOT, "rag_corpus_uniform")
OUTPUT_DIR = os.path.join(REPO_ROOT, "embedder_benchmark")

TEXT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
N_QUERY_REPS  = 1000
K_VALUES      = [1, 2, 3]
SEQ_MIN_LEN   = 300
SEQ_MAX_LEN   = 900

SEQUENCE_SOURCES = [
    ("seq1_TT", "GCF_000008125.1_ASM812v1_cds_from_genomic.fna", 1, 0, "Thermus thermophilus"),
    ("seq2_TT", "GCF_000008125.1_ASM812v1_cds_from_genomic.fna", 2, 0, "Thermus thermophilus"),
    ("seq3_CT", "GCF_000008725.1_ASM872v1_cds_from_genomic.fna", 1, 1, "Chlamydia trachomatis"),
    ("seq4_SA", "GCF_000009005.1_ASM900v1_cds_from_genomic.fna", 1, 2, "Staphylococcus aureus"),
    ("seq5_DM", "GCF_000009025.1_ASM902v1_cds_from_genomic.fna", 1, 3, "Dehalococcoides mccartyi"),
]

AXIS_LABELS = [
    "T.thermophilus_1", "T.thermophilus_2",
    "C.trachomatis", "S.aureus", "D.mccartyi",
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
        for seq in sequences:
            enc = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            out = model(**enc)
            lh = out[0] if isinstance(out, tuple) else out.last_hidden_state
            embeddings.append(mean_pool(lh, enc["attention_mask"]))
    del model, tokenizer
    return np.stack(embeddings).astype(np.float32)


def embed_text(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    log.info(f"Loading {TEXT_MODEL_ID} ...")
    model = SentenceTransformer(TEXT_MODEL_ID)
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    del model
    return np.array(embs, dtype=np.float32)


# ── FAISS helpers ───────────────────────────────────────────────────────────

def normalise(v: np.ndarray) -> np.ndarray:
    u = v.copy()
    faiss.normalize_L2(u)
    return u


def build_flat_ip(vectors: np.ndarray) -> tuple:
    d = vectors.shape[1]
    v = normalise(vectors)
    idx = faiss.IndexFlatIP(d)
    t0 = time.perf_counter()
    idx.add(v)
    build_ms = (time.perf_counter() - t0) * 1000
    return idx, v, build_ms


def index_memory_bytes(index: faiss.Index) -> int:
    return len(faiss.serialize_index(index))


def benchmark_latency(index: faiss.Index, query_vecs: np.ndarray,
                      k: int, n_reps: int) -> float:
    for _ in range(10):
        index.search(query_vecs, k)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        index.search(query_vecs, k)
    return (time.perf_counter() - t0) / n_reps / len(query_vecs) * 1e6


def faiss_accuracy(index: faiss.Index, norm_vecs: np.ndarray,
                   baseline_sim: np.ndarray, k: int) -> float:
    n = len(norm_vecs)
    _, ids = index.search(norm_vecs, k + 1)
    correct = 0
    for i in range(n):
        bf = baseline_sim[i].copy()
        bf[i] = -2.0
        bf_topk = set(np.argsort(bf)[::-1][:k])
        fa_topk = set(int(x) for x in ids[i] if x != i and x >= 0)
        if bf_topk == set(list(fa_topk)[:k]):
            correct += 1
    return correct / n


# ── FAISS retrieval ─────────────────────────────────────────────────────────

def faiss_retrieve(index: faiss.Index, query_vec: np.ndarray,
                   records: list[dict], query_idx: int, k: int) -> list[dict]:
    scores, ids = index.search(query_vec.reshape(1, -1), k + 1)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == query_idx or idx < 0:
            continue
        results.append({
            "rank":     len(results) + 1,
            "id":       records[idx]["tag"],
            "label":    records[idx]["label"],
            "organism": records[idx]["organism"],
            "score":    float(score),
            "metadata": records[idx]["metadata"],
        })
        if len(results) == k:
            break
    return results


# ── Metrics ─────────────────────────────────────────────────────────────────

def precision_at_k(results: list[dict], query_label: int, k: int) -> float:
    return sum(1 for r in results[:k] if r["label"] == query_label) / k


def reciprocal_rank(results: list[dict], query_label: int) -> float:
    for r in results:
        if r["label"] == query_label:
            return 1.0 / r["rank"]
    return 0.0


def has_same_species(records: list[dict], query_label: int, query_idx: int) -> bool:
    return any(r["label"] == query_label for i, r in enumerate(records) if i != query_idx)


# ── Visualisation ────────────────────────────────────────────────────────────

def plot_heatmap(matrix: np.ndarray, title: str, labels: list[str],
                 filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="viridis",
                vmin=0, vmax=1, xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5, linecolor="white")
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=== Loading 5 CDS records ===")
    records = select_sequences()
    tags   = [r["tag"]      for r in records]
    seqs   = [r["sequence"] for r in records]
    metas  = [r["metadata"] for r in records]
    labels = [r["label"]    for r in records]
    N = len(records)

    print(f"\n{N} records: {tags}")
    for r in records:
        print(f"  {r['tag']:22s}: {r['metadata'][:80]}...")

    # ── Embed ────────────────────────────────────────────────────────────────
    log.info("=== Embedding DNA sequences (DNABERT-S) ===")
    dna_raw = embed_dnabert_s(seqs)

    log.info("=== Embedding metadata texts (MiniLM) ===")
    txt_raw = embed_text(metas)

    # Brute-force baseline
    dna_norm = normalise(dna_raw)
    txt_norm = normalise(txt_raw)
    dna_baseline = sklearn_cosine(dna_norm)
    txt_baseline = sklearn_cosine(txt_norm)

    # ── Build FAISS indices ──────────────────────────────────────────────────
    log.info("=== Building FAISS indices (IndexFlatIP, n=5) ===")
    dna_idx, dna_norm_v, dna_build_ms = build_flat_ip(dna_raw)
    txt_idx, txt_norm_v, txt_build_ms = build_flat_ip(txt_raw)

    dna_mem = index_memory_bytes(dna_idx)
    txt_mem = index_memory_bytes(txt_idx)
    dna_lat = benchmark_latency(dna_idx, dna_norm_v, k=1, n_reps=N_QUERY_REPS)
    txt_lat = benchmark_latency(txt_idx, txt_norm_v, k=1, n_reps=N_QUERY_REPS)
    dna_acc = faiss_accuracy(dna_idx, dna_norm_v, dna_baseline, k=1)
    txt_acc = faiss_accuracy(txt_idx, txt_norm_v, txt_baseline, k=1)

    print(f"\n=== FAISS INDEX STATS ===")
    print(f"  Index A (DNA  768-dim): build={dna_build_ms:.3f}ms  "
          f"lat={dna_lat:.2f}µs  mem={dna_mem:,}B  acc={dna_acc:.4f}")
    print(f"  Index B (Text 384-dim): build={txt_build_ms:.3f}ms  "
          f"lat={txt_lat:.2f}µs  mem={txt_mem:,}B  acc={txt_acc:.4f}")

    # Serialize indices
    faiss.write_index(dna_idx, os.path.join(OUTPUT_DIR, "experiment_9_index_dna.faiss"))
    faiss.write_index(txt_idx, os.path.join(OUTPUT_DIR, "experiment_9_index_text.faiss"))

    # ── Similarity heatmaps ──────────────────────────────────────────────────
    plot_heatmap(dna_baseline, "DNA–DNA Similarity (DNABERT-S)",
                 tags, "experiment_9_dna_dna_sim.png")
    plot_heatmap(txt_baseline, f"Text–Text Similarity (MiniLM)",
                 tags, "experiment_9_txt_txt_sim.png")

    # ── Query evaluation ─────────────────────────────────────────────────────
    query_configs = [
        ("DNA→DNA",  dna_idx, dna_norm_v, "DNA query → FAISS Index A → ranked DNA records"),
        ("Text→DNA", txt_idx, txt_norm_v, "text query → FAISS Index B → ID bridge → DNA records"),
        ("DNA→Text", dna_idx, dna_norm_v, "DNA query → FAISS Index A → ID bridge → metadata"),
    ]

    eval_rows = []

    for qtype, index, norm_vecs, description in query_configs:
        print(f"\n{'='*65}")
        print(f"Query type: {qtype}")
        print(f"Description: {description}")
        print(f"{'='*65}")

        p_at = {k: [] for k in K_VALUES}
        rr_list = []

        for qi in range(N):
            if not has_same_species(records, labels[qi], qi):
                print(f"\n  Query [{tags[qi]}]: singleton species — P@k N/A")
                continue

            results = faiss_retrieve(index, norm_vecs[qi], records, qi, k=N - 1)
            display = "metadata" if qtype == "DNA→Text" else "id"

            print(f"\n  Query [{tags[qi]}] (label={labels[qi]}):")
            for r in results:
                match = "✓" if r["label"] == labels[qi] else " "
                val = r[display][:70] if display == "metadata" else r["id"]
                print(f"    Rank {r['rank']}: {match} {val}  (score={r['score']:.4f})")

            for k in K_VALUES:
                p_at[k].append(precision_at_k(results, labels[qi], k))
            rr_list.append(reciprocal_rank(results, labels[qi]))

            row = {"query_type": qtype, "query_tag": tags[qi], "query_label": labels[qi]}
            for k in K_VALUES:
                row[f"P@{k}"] = round(precision_at_k(results, labels[qi], k), 4)
            row["RR"] = round(reciprocal_rank(results, labels[qi]), 4)
            eval_rows.append(row)

        print(f"\n  --- Aggregate ({qtype}) ---")
        for k in K_VALUES:
            if p_at[k]:
                print(f"  Mean P@{k}: {np.mean(p_at[k]):.4f}  (n={len(p_at[k])})")
        if rr_list:
            print(f"  MRR:       {np.mean(rr_list):.4f}")

    # ── Save results ─────────────────────────────────────────────────────────
    df = pd.DataFrame(eval_rows)
    print("\n\n=== FULL EVALUATION TABLE ===")
    print(df.to_string(index=False))

    metric_cols = [f"P@{k}" for k in K_VALUES] + ["RR"]
    summary = df.groupby("query_type")[metric_cols].mean().rename(columns={"RR": "MRR"})
    print("\n=== SUMMARY BY QUERY TYPE ===")
    print(summary.round(4).to_string())

    csv_path = os.path.join(OUTPUT_DIR, "experiment_9_results.csv")
    df.to_csv(csv_path, index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "experiment_9_summary.csv"))

    stats_rows = [
        {"index": "A (DNA)",  "model": "DNABERT-S",   "dim": 768, "n": N,
         "index_type": "IndexFlatIP",
         "build_ms": round(dna_build_ms, 4), "latency_us": round(dna_lat, 3),
         "memory_bytes": dna_mem, "top1_accuracy": round(dna_acc, 4)},
        {"index": "B (Text)", "model": "MiniLM-L6-v2", "dim": 384, "n": N,
         "index_type": "IndexFlatIP",
         "build_ms": round(txt_build_ms, 4), "latency_us": round(txt_lat, 3),
         "memory_bytes": txt_mem, "top1_accuracy": round(txt_acc, 4)},
    ]
    pd.DataFrame(stats_rows).to_csv(
        os.path.join(OUTPUT_DIR, "experiment_9_faiss_stats.csv"), index=False
    )

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
