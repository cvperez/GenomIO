#!/usr/bin/env python3
"""
Experiment 7: FAISS Index Construction — Proof of Concept (5 CDS records)
Builds and benchmarks FAISS indices for the dual-modality strategy
using the 5 CDS records from Experiment 1.

For each modality (DNA=768-dim, Text=384-dim), tests:
  - IndexFlatIP   : exact inner-product search (cosine for L2-normalised vectors)
  - IndexHNSWFlat : approximate HNSW graph search

Measures per index:
  - Build time (ms)
  - Query latency (µs, averaged over 1000 repetitions)
  - Top-k accuracy vs brute-force cosine baseline (k=1,2,3)
  - Serialised memory footprint (bytes)

Run from repo root:
    python experiments/experiment_7_faiss_index.py
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
from Bio import SeqIO
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(REPO_ROOT, "rag_corpus_uniform")
OUTPUT_DIR = os.path.join(REPO_ROOT, "embedder_benchmark")

TEXT_MODEL_ID  = "sentence-transformers/all-MiniLM-L6-v2"
N_QUERY_REPS   = 1000   # repetitions for latency measurement
K_VALUES       = [1, 2, 3]
HNSW_M         = 32     # HNSW connections per node

SEQ_MIN_LEN = 300
SEQ_MAX_LEN = 900

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
                        "tag":      tag,
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


# ── FAISS index builders ────────────────────────────────────────────────────

def build_flat_ip(vectors: np.ndarray) -> faiss.IndexFlatIP:
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    v = vectors.copy()
    faiss.normalize_L2(v)
    t0 = time.perf_counter()
    index.add(v)
    build_ms = (time.perf_counter() - t0) * 1000
    return index, v, build_ms


def build_hnsw(vectors: np.ndarray, M: int = HNSW_M) -> faiss.IndexHNSWFlat:
    d = vectors.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 16
    v = vectors.copy()
    faiss.normalize_L2(v)
    t0 = time.perf_counter()
    index.add(v)
    build_ms = (time.perf_counter() - t0) * 1000
    return index, v, build_ms


def index_memory_bytes(index: faiss.Index) -> int:
    buf = faiss.serialize_index(index)
    return len(buf)


# ── Benchmarking ────────────────────────────────────────────────────────────

def benchmark_queries(index: faiss.Index, query_vectors: np.ndarray,
                      k: int, n_reps: int) -> float:
    """Return mean query latency in microseconds over n_reps repetitions."""
    # Warm-up
    for _ in range(10):
        index.search(query_vectors, k)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        index.search(query_vectors, k)
    elapsed = time.perf_counter() - t0
    return (elapsed / n_reps / len(query_vectors)) * 1e6  # µs per query


def top_k_accuracy(index: faiss.Index, query_vectors: np.ndarray,
                   baseline_sim: np.ndarray, k: int) -> float:
    """
    Fraction of queries where FAISS top-k matches the brute-force top-k exactly.
    baseline_sim: (n, n) cosine similarity matrix (from sklearn).
    """
    n = len(query_vectors)
    _, faiss_ids = index.search(query_vectors, k + 1)  # +1 to account for self

    correct = 0
    for i in range(n):
        # Brute-force top-k excluding self
        bf_row = baseline_sim[i].copy()
        bf_row[i] = -2.0
        bf_topk = set(np.argsort(bf_row)[::-1][:k])
        # FAISS top-k excluding self
        fa_topk = set(int(x) for x in faiss_ids[i] if x != i and x >= 0)
        fa_topk = set(list(fa_topk)[:k])
        if bf_topk == fa_topk:
            correct += 1

    return correct / n


# ── Visualisation ───────────────────────────────────────────────────────────

def plot_comparison(rows: list[dict], filename: str) -> None:
    df = pd.DataFrame(rows)
    metrics = ["build_time_ms", "latency_us_per_query", "memory_bytes"]
    labels  = ["Build time (ms)", "Query latency (µs/query)", "Memory (bytes)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    modalities = df["modality"].unique()
    colors = {"DNA (768-dim)": "#1f77b4", "Text (384-dim)": "#ff7f0e"}

    for ax, metric, label in zip(axes, metrics, labels):
        for mod in modalities:
            sub = df[df["modality"] == mod]
            ax.bar(sub["index_type"], sub[metric],
                   label=mod, color=[colors[mod]] * len(sub), alpha=0.8, width=0.35,
                   align="center")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(label)
        ax.legend(fontsize=7)

    plt.suptitle("FAISS Index Comparison — IndexFlatIP vs IndexHNSWFlat", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=== Loading 5 CDS records (Experiment 1 set) ===")
    records = select_sequences()
    seqs  = [r["sequence"] for r in records]
    metas = [r["metadata"] for r in records]
    N = len(records)
    print(f"\n{N} records: {[r['tag'] for r in records]}")

    # ── Embed ───────────────────────────────────────────────────────────────
    log.info("=== Embedding DNA sequences (DNABERT-S) ===")
    dna_raw = embed_dnabert_s(seqs)        # (5, 768) float32

    log.info("=== Embedding metadata texts (MiniLM) ===")
    txt_raw = embed_text(metas)            # (5, 384) float32

    # Brute-force cosine baselines
    dna_norm = dna_raw / np.linalg.norm(dna_raw, axis=1, keepdims=True)
    txt_norm = txt_raw / np.linalg.norm(txt_raw, axis=1, keepdims=True)
    dna_baseline = sklearn_cosine(dna_norm)
    txt_baseline = sklearn_cosine(txt_norm)

    modality_configs = [
        ("DNA (768-dim)",  dna_raw, dna_baseline),
        ("Text (384-dim)", txt_raw, txt_baseline),
    ]

    all_rows = []

    for mod_name, raw_vecs, baseline_sim in modality_configs:
        print(f"\n{'='*60}\nModality: {mod_name}\n{'='*60}")

        # ── Flat IP ─────────────────────────────────────────────────────────
        flat_idx, flat_vecs, flat_build_ms = build_flat_ip(raw_vecs)
        flat_lat  = benchmark_queries(flat_idx, flat_vecs, k=1, n_reps=N_QUERY_REPS)
        flat_mem  = index_memory_bytes(flat_idx)
        flat_accs = {k: top_k_accuracy(flat_idx, flat_vecs, baseline_sim, k)
                     for k in K_VALUES}

        print(f"  IndexFlatIP:")
        print(f"    build_time  : {flat_build_ms:.3f} ms")
        print(f"    latency     : {flat_lat:.2f} µs/query")
        print(f"    memory      : {flat_mem:,} bytes")
        for k in K_VALUES:
            print(f"    top-{k} acc   : {flat_accs[k]:.4f}")

        all_rows.append({
            "modality":             mod_name,
            "index_type":           "IndexFlatIP",
            "dim":                  raw_vecs.shape[1],
            "n_vectors":            N,
            "build_time_ms":        round(flat_build_ms, 4),
            "latency_us_per_query": round(flat_lat, 3),
            "memory_bytes":         flat_mem,
            **{f"top{k}_accuracy": round(flat_accs[k], 4) for k in K_VALUES},
        })

        # ── HNSW ────────────────────────────────────────────────────────────
        hnsw_idx, hnsw_vecs, hnsw_build_ms = build_hnsw(raw_vecs)
        hnsw_lat  = benchmark_queries(hnsw_idx, hnsw_vecs, k=1, n_reps=N_QUERY_REPS)
        hnsw_mem  = index_memory_bytes(hnsw_idx)
        hnsw_accs = {k: top_k_accuracy(hnsw_idx, hnsw_vecs, baseline_sim, k)
                     for k in K_VALUES}

        print(f"  IndexHNSWFlat (M={HNSW_M}):")
        print(f"    build_time  : {hnsw_build_ms:.3f} ms")
        print(f"    latency     : {hnsw_lat:.2f} µs/query")
        print(f"    memory      : {hnsw_mem:,} bytes")
        for k in K_VALUES:
            print(f"    top-{k} acc   : {hnsw_accs[k]:.4f}")

        all_rows.append({
            "modality":             mod_name,
            "index_type":           f"IndexHNSWFlat(M={HNSW_M})",
            "dim":                  raw_vecs.shape[1],
            "n_vectors":            N,
            "build_time_ms":        round(hnsw_build_ms, 4),
            "latency_us_per_query": round(hnsw_lat, 3),
            "memory_bytes":         hnsw_mem,
            **{f"top{k}_accuracy": round(hnsw_accs[k], 4) for k in K_VALUES},
        })

        # ── Save index files ────────────────────────────────────────────────
        for idx_obj, name in [(flat_idx, "flat"), (hnsw_idx, "hnsw")]:
            mod_slug = "dna" if "DNA" in mod_name else "text"
            path = os.path.join(OUTPUT_DIR,
                                f"experiment_7_index_{mod_slug}_{name}.faiss")
            faiss.write_index(idx_obj, path)
            log.info(f"  Index saved: {path}")

    # ── Summary ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    print("\n\n=== COMPARISON TABLE ===")
    print(df[["modality", "index_type", "build_time_ms",
              "latency_us_per_query", "memory_bytes",
              "top1_accuracy", "top2_accuracy", "top3_accuracy"]].to_string(index=False))

    csv_path = os.path.join(OUTPUT_DIR, "experiment_7_results.csv")
    df.to_csv(csv_path, index=False)

    plot_comparison(all_rows, "experiment_7_comparison.png")

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
