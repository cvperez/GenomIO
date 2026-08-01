#!/usr/bin/env python3
"""
Experiment 10: FAISS-Based Retrieval — Scaled (1,000 CDS records)
Re-runs the Experiment 6 retrieval evaluation routing queries through FAISS
indices (per Experiment 8 recommendations), and compares results to Exp 6
brute-force baseline to quantify the accuracy cost of approximation.

Index types (per Experiment 8 recommendation for n=1,000):
  - Index A: IndexHNSWFlat(M=32, efConstruction=200, efSearch=64) — DNA, 768-dim
  - Index B: IndexFlatIP — Text, 384-dim (HNSW crossover at n≈2,000, flat still optimal)

Also fixes the organism name bug from Experiment 6 (accession ID version suffix).

Query types:
  1. DNA→DNA  : DNA query → FAISS Index A → ranked DNA records
  2. Text→DNA : text query → FAISS Index B → ID bridge → DNA records
  3. DNA→Text : DNA query → FAISS Index A → ID bridge → metadata records

Metrics: P@1, P@5, P@10, P@20, Recall@10, Recall@20, MRR (leave-self-out)
Additional: FAISS accuracy vs brute-force, build time, latency, memory.

Run from repo root:
    python experiments/experiment_10_faiss_retrieval_scale.py
"""
import glob
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

TEXT_MODEL_ID    = "sentence-transformers/all-MiniLM-L6-v2"
SEQS_PER_SPECIES = 50
SEQ_MIN_LEN      = 300
SEQ_MAX_LEN      = 900
HNSW_M           = 32
N_QUERY_REPS     = 100
K_VALUES         = [1, 5, 10, 20]
RECALL_K         = [10, 20]

# Experiment 6 brute-force results for comparison
EXP6_RESULTS = {
    "DNA→DNA":  {"P@1": 0.655, "P@5": 0.6158, "P@10": 0.5746, "P@20": 0.5282,
                 "Recall@10": 0.1173, "Recall@20": 0.2156, "MRR": 0.7666},
    "Text→DNA": {"P@1": 0.601, "P@5": 0.5624, "P@10": 0.5310, "P@20": 0.4704,
                 "Recall@10": 0.1084, "Recall@20": 0.1920, "MRR": 0.7036},
    "DNA→Text": {"P@1": 0.655, "P@5": 0.6158, "P@10": 0.5746, "P@20": 0.5282,
                 "Recall@10": 0.1173, "Recall@20": 0.2156, "MRR": 0.7666},
}

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


# ── Data loading ─────────────────────────────────────────────────────────────

def build_metadata_text(description: str, organism: str) -> str:
    fields = re.findall(r'\[([^\]]+)\]', description)
    clean = [f for f in fields if not f.startswith("gbkey=") and not f.startswith("location=")]
    return f"organism={organism} " + " ".join(clean)


def select_sequences() -> tuple[list[dict], list[str]]:
    fna_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.fna")))
    records, species_tags = [], []
    for label, fpath in enumerate(fna_files):
        basename = os.path.basename(fpath)
        parts = basename.split("_")
        # Fix: strip .1 version suffix from accession
        accession_key = parts[0] + "_" + parts[1].split(".")[0]
        organism = ORGANISM_NAMES.get(accession_key, accession_key)
        species_tags.append(accession_key)
        count = 0
        for rec in SeqIO.parse(fpath, "fasta"):
            seq = str(rec.seq).upper().strip()
            if SEQ_MIN_LEN <= len(seq) <= SEQ_MAX_LEN:
                records.append({
                    "sequence": seq,
                    "metadata": build_metadata_text(rec.description, organism),
                    "label":    label,
                    "organism": organism,
                    "accession": accession_key,
                })
                count += 1
                if count == SEQS_PER_SPECIES:
                    break
        log.info(f"  [{label:02d}] {accession_key} ({organism}): {count} records")
    return records, species_tags


# ── Embeddings ────────────────────────────────────────────────────────────────

def mean_pool(lhs, am):
    mask = am.unsqueeze(-1).expand(lhs.size()).float()
    emb = (lhs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return emb.squeeze(0).cpu().numpy()


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
                log.info(f"  DNABERT-S: {i+1}/{len(sequences)}")
    del model, tokenizer
    return np.stack(embeddings).astype(np.float32)


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
            log.info(f"  MiniLM: {min(i+bs, len(texts))}/{len(texts)}")
    del model
    return np.vstack(embs).astype(np.float32)


# ── FAISS helpers ────────────────────────────────────────────────────────────

def normalise(v: np.ndarray) -> np.ndarray:
    u = v.copy()
    faiss.normalize_L2(u)
    return u


def build_hnsw_index(vectors: np.ndarray) -> tuple:
    d = vectors.shape[1]
    v = normalise(vectors)
    idx = faiss.IndexHNSWFlat(d, HNSW_M)
    idx.hnsw.efConstruction = 200
    idx.hnsw.efSearch = 64
    t0 = time.perf_counter()
    idx.add(v)
    build_ms = (time.perf_counter() - t0) * 1000
    return idx, v, build_ms


def build_flat_index(vectors: np.ndarray) -> tuple:
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
    sample = query_vecs[:4]
    for _ in range(5):
        index.search(sample, k)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        index.search(sample, k)
    return (time.perf_counter() - t0) / n_reps / len(sample) * 1e6


def faiss_topk_accuracy(index: faiss.Index, norm_vecs: np.ndarray,
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


# ── Retrieval evaluation ──────────────────────────────────────────────────────

def evaluate_faiss_retrieval(index: faiss.Index, norm_vecs: np.ndarray,
                              labels: np.ndarray, query_type: str) -> pd.DataFrame:
    N = len(labels)
    n_relevant = np.array([np.sum(labels == labels[i]) - 1 for i in range(N)])

    # Query all vectors at once
    max_k = max(K_VALUES) + max(RECALL_K) + 1
    _, all_ids = index.search(norm_vecs, max_k)

    rows = []
    for i in range(N):
        if n_relevant[i] == 0:
            continue
        # Ranked indices excluding self
        ranked = [int(x) for x in all_ids[i] if x != i and x >= 0]
        relevant = np.array([labels[j] == labels[i] for j in ranked])
        n_rel = int(n_relevant[i])

        row = {"query_type": query_type, "query_label": int(labels[i]),
               "n_relevant": n_rel}
        for k in K_VALUES:
            row[f"P@{k}"]      = float(relevant[:k].sum() / k)
            row[f"Recall@{k}"] = float(relevant[:k].sum() / n_rel)
        first_hit = int(np.argmax(relevant)) if relevant.any() else len(relevant)
        row["RR"] = float(1.0 / (first_hit + 1)) if relevant.any() else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_metric_by_species(df: pd.DataFrame, metric: str, query_type: str,
                           species_tags: list[str], filename: str) -> None:
    means = df.groupby("query_label")[metric].mean().reset_index()
    means["species"] = means["query_label"].apply(
        lambda l: species_tags[l] if l < len(species_tags) else str(l)
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(means["species"], means[metric], color="steelblue", alpha=0.8)
    ax.set_title(f"{metric} by Species — {query_type} (FAISS)", fontsize=11)
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
    for i, (qt, row) in enumerate(summary_df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=qt, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("FAISS Retrieval Performance by Query Type (n=1,000)", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_comparison_vs_exp6(faiss_summary: pd.DataFrame, filename: str) -> None:
    query_types = list(EXP6_RESULTS.keys())
    metrics = ["P@1", "P@5", "P@10", "MRR"]
    x = np.arange(len(metrics))
    width = 0.15
    n_qt = len(query_types)
    colors = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(12, 5))
    for qi, qt in enumerate(query_types):
        exp6_vals = [EXP6_RESULTS[qt][m] for m in metrics]
        faiss_vals = [faiss_summary.loc[qt, m] if m in faiss_summary.columns
                      else faiss_summary.loc[qt, "MRR"] for m in metrics]

        offset = (qi * 2) * width
        ax.bar(x + offset,         exp6_vals,   width, label=f"{qt} Exp6 (Exact)",
               color=colors(qi / n_qt), alpha=0.5, hatch="//")
        ax.bar(x + offset + width, faiss_vals,  width, label=f"{qt} Exp10 (FAISS)",
               color=colors(qi / n_qt), alpha=0.9)

    ax.set_xticks(x + width * n_qt)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("FAISS (Exp 10) vs Brute-Force (Exp 6) Retrieval Quality", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=== Selecting sequences ===")
    records, species_tags = select_sequences()
    seqs   = [r["sequence"] for r in records]
    metas  = [r["metadata"] for r in records]
    labels = np.array([r["label"] for r in records])
    N = len(records)
    print(f"\n{N} records across {len(species_tags)} species "
          f"({SEQS_PER_SPECIES}/species)")

    # ── Embed ──────────────────────────────────────────────────────────────
    log.info("=== Embedding DNA sequences (DNABERT-S) ===")
    t_dna = time.perf_counter()
    dna_raw = embed_dnabert_s(seqs)
    dna_embed_s = time.perf_counter() - t_dna

    log.info("=== Embedding metadata texts (MiniLM) ===")
    t_txt = time.perf_counter()
    txt_raw = embed_text(metas)
    txt_embed_s = time.perf_counter() - t_txt

    dna_norm = normalise(dna_raw)
    txt_norm = normalise(txt_raw)

    # ── Build FAISS indices ────────────────────────────────────────────────
    log.info("=== Building Index A: HNSW (DNA, 768-dim) ===")
    dna_idx, dna_norm_v, dna_build_ms = build_hnsw_index(dna_raw)

    log.info("=== Building Index B: Flat (Text, 384-dim) ===")
    txt_idx, txt_norm_v, txt_build_ms = build_flat_index(txt_raw)

    dna_mem  = index_memory_bytes(dna_idx)
    txt_mem  = index_memory_bytes(txt_idx)
    dna_lat  = benchmark_latency(dna_idx, dna_norm_v, k=1, n_reps=N_QUERY_REPS)
    txt_lat  = benchmark_latency(txt_idx, txt_norm_v, k=1, n_reps=N_QUERY_REPS)

    # Brute-force accuracy check (compare FAISS top-1 vs sklearn)
    log.info("Computing brute-force baseline for accuracy check ...")
    dna_bf = sklearn_cosine(dna_norm)
    txt_bf = sklearn_cosine(txt_norm)
    dna_acc = faiss_topk_accuracy(dna_idx, dna_norm_v, dna_bf, k=1)
    txt_acc = faiss_topk_accuracy(txt_idx, txt_norm_v, txt_bf, k=1)

    print(f"\n=== FAISS INDEX STATS ===")
    print(f"  Index A (DNA  768-dim, HNSW M={HNSW_M}): "
          f"build={dna_build_ms:.1f}ms  lat={dna_lat:.1f}µs  "
          f"mem={dna_mem/1024:.0f}KB  acc={dna_acc:.4f}")
    print(f"  Index B (Text 384-dim, Flat):            "
          f"build={txt_build_ms:.1f}ms  lat={txt_lat:.1f}µs  "
          f"mem={txt_mem/1024:.0f}KB  acc={txt_acc:.4f}")

    faiss.write_index(dna_idx, os.path.join(OUTPUT_DIR, "experiment_10_index_dna.faiss"))
    faiss.write_index(txt_idx, os.path.join(OUTPUT_DIR, "experiment_10_index_text.faiss"))

    # ── Evaluate all 3 query types ─────────────────────────────────────────
    query_configs = [
        ("DNA→DNA",  dna_idx, dna_norm_v, "DNA query → FAISS HNSW Index A → ranked DNA records"),
        ("Text→DNA", txt_idx, txt_norm_v, "text query → FAISS Flat Index B → ID bridge → DNA"),
        ("DNA→Text", dna_idx, dna_norm_v, "DNA query → FAISS HNSW Index A → ID bridge → metadata"),
    ]

    all_dfs = []
    for qtype, index, norm_vecs, description in query_configs:
        print(f"\n{'='*60}\n{qtype}: {description}\n{'='*60}")
        df = evaluate_faiss_retrieval(index, norm_vecs, labels, qtype)
        all_dfs.append(df)

        agg = df[[f"P@{k}" for k in K_VALUES] +
                  [f"Recall@{k}" for k in RECALL_K] + ["RR"]].mean()
        print(f"  n_queries: {len(df)}")
        for k in K_VALUES:
            print(f"  P@{k:2d}:       {agg[f'P@{k}']:.4f}")
        for k in RECALL_K:
            print(f"  Recall@{k:2d}:  {agg[f'Recall@{k}']:.4f}")
        print(f"  MRR:       {agg['RR']:.4f}")

        slug = qtype.replace("→", "_")
        plot_metric_by_species(df, "P@1",  qtype, species_tags,
                               f"experiment_10_{slug}_p1_by_species.png")
        plot_metric_by_species(df, "P@10", qtype, species_tags,
                               f"experiment_10_{slug}_p10_by_species.png")

    all_eval = pd.concat(all_dfs, ignore_index=True)

    metric_cols = [f"P@{k}" for k in K_VALUES] + \
                  [f"Recall@{k}" for k in RECALL_K] + ["RR"]
    summary = all_eval.groupby("query_type")[metric_cols].mean().round(4)
    summary = summary.rename(columns={"RR": "MRR"})

    print("\n\n=== FAISS SUMMARY TABLE ===")
    print(summary.to_string())

    print("\n=== COMPARISON: FAISS (Exp 10) vs BRUTE-FORCE (Exp 6) ===")
    for qt in EXP6_RESULTS:
        if qt not in summary.index:
            continue
        print(f"\n  {qt}:")
        for m in ["P@1", "P@5", "P@10", "MRR"]:
            exp6_val  = EXP6_RESULTS[qt].get(m, float("nan"))
            faiss_val = summary.loc[qt, m]
            delta = faiss_val - exp6_val
            print(f"    {m:8s}: Exp6={exp6_val:.4f}  FAISS={faiss_val:.4f}  "
                  f"Δ={delta:+.4f}")

    plot_summary_bar(summary, "experiment_10_summary_bar.png")
    plot_comparison_vs_exp6(summary, "experiment_10_comparison_vs_exp6.png")

    # Save CSVs
    all_eval.to_csv(os.path.join(OUTPUT_DIR, "experiment_10_results.csv"), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "experiment_10_summary.csv"))

    stats_rows = [
        {"index": "A (DNA)",  "index_type": f"IndexHNSWFlat(M={HNSW_M})",
         "dim": 768, "n": N, "embed_time_s": round(dna_embed_s, 1),
         "build_ms": round(dna_build_ms, 1), "latency_us": round(dna_lat, 1),
         "memory_kb": round(dna_mem/1024, 0), "top1_accuracy": round(dna_acc, 4)},
        {"index": "B (Text)", "index_type": "IndexFlatIP",
         "dim": 384, "n": N, "embed_time_s": round(txt_embed_s, 1),
         "build_ms": round(txt_build_ms, 1), "latency_us": round(txt_lat, 1),
         "memory_kb": round(txt_mem/1024, 0), "top1_accuracy": round(txt_acc, 4)},
    ]
    pd.DataFrame(stats_rows).to_csv(
        os.path.join(OUTPUT_DIR, "experiment_10_faiss_stats.csv"), index=False
    )

    print(f"\nResults saved to: {os.path.join(OUTPUT_DIR, 'experiment_10_results.csv')}")


if __name__ == "__main__":
    main()
