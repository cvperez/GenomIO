#!/usr/bin/env python3
"""
Experiment 8: FAISS Index Construction — Scaled (1,000 CDS records)
Scales Experiment 7 to 1,000 records (50/species × 20 species) to determine
whether IndexHNSWFlat becomes competitive with IndexFlatIP at corpus sizes
realistic for this RAG system.

Additionally tests two supplementary scales (100 and 5,000 records) to map
the latency crossover point between flat and HNSW search.

Metrics: build time (ms), query latency (µs/query), top-k accuracy, memory (bytes).

Run from repo root:
    python embedder_benchmark/experiment_8_faiss_scale/experiment_8_faiss_scale.py
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
from Bio import SeqIO
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_DIR = os.path.join(REPO_ROOT, "rag_corpus_uniform")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

TEXT_MODEL_ID  = "sentence-transformers/all-MiniLM-L6-v2"
SEQS_PER_SPECIES = 150       # → 3,000 total (safe max: smallest species has ~204 records)
SEQ_MIN_LEN = 300
SEQ_MAX_LEN = 900
N_QUERY_REPS = 200           # repetitions for latency (fewer due to larger n)
K_VALUES     = [1, 5, 10, 20]
HNSW_M       = 32

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


# ── Data loading ────────────────────────────────────────────────────────────

def build_metadata_text(description: str, organism: str) -> str:
    fields = re.findall(r'\[([^\]]+)\]', description)
    clean = [f for f in fields if not f.startswith("gbkey=") and not f.startswith("location=")]
    return f"organism={organism} " + " ".join(clean)


def select_sequences(n_per_species: int) -> list[dict]:
    fna_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.fna")))
    records = []
    for label, fpath in enumerate(fna_files):
        basename = os.path.basename(fpath)
        parts = basename.split("_")
        accession_key = parts[0] + "_" + parts[1].split(".")[0]   # strip .1 suffix
        organism = ORGANISM_NAMES.get(accession_key, accession_key)
        count = 0
        for rec in SeqIO.parse(fpath, "fasta"):
            seq = str(rec.seq).upper().strip()
            if SEQ_MIN_LEN <= len(seq) <= SEQ_MAX_LEN:
                records.append({
                    "sequence": seq,
                    "metadata": build_metadata_text(rec.description, organism),
                    "label":    label,
                    "organism": organism,
                })
                count += 1
                if count == n_per_species:
                    break
    return records


# ── Embeddings ──────────────────────────────────────────────────────────────

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
    del model
    return np.vstack(embs).astype(np.float32)


# ── FAISS helpers ───────────────────────────────────────────────────────────

def normalise(v: np.ndarray) -> np.ndarray:
    u = v.copy()
    faiss.normalize_L2(u)
    return u


def build_and_bench(vectors: np.ndarray, index_type: str,
                    k: int, n_reps: int, baseline_sim: np.ndarray) -> dict:
    d = vectors.shape[1]
    v = normalise(vectors)

    if index_type == "flat":
        idx = faiss.IndexFlatIP(d)
    else:
        idx = faiss.IndexHNSWFlat(d, HNSW_M)
        idx.hnsw.efConstruction = 200
        idx.hnsw.efSearch = 64

    t0 = time.perf_counter()
    idx.add(v)
    build_ms = (time.perf_counter() - t0) * 1000

    # Warm-up
    for _ in range(5):
        idx.search(v[:4], k)

    t0 = time.perf_counter()
    for _ in range(n_reps):
        idx.search(v[:4], k)   # query with first 4 vectors
    lat_us = (time.perf_counter() - t0) / n_reps / 4 * 1e6

    mem = len(faiss.serialize_index(idx))

    # Accuracy: fraction of queries where top-k matches brute-force (excluding self)
    n = len(v)
    _, faiss_ids = idx.search(v, k + 1)
    correct = 0
    for i in range(n):
        bf = baseline_sim[i].copy()
        bf[i] = -2.0
        bf_topk = set(np.argsort(bf)[::-1][:k])
        fa_topk = set(int(x) for x in faiss_ids[i] if x != i and x >= 0)
        if bf_topk == set(list(fa_topk)[:k]):
            correct += 1
    accuracy = correct / n

    return {
        "build_time_ms":        round(build_ms, 3),
        "latency_us_per_query": round(lat_us, 3),
        "memory_bytes":         mem,
        "accuracy":             round(accuracy, 4),
    }


# ── Plots ───────────────────────────────────────────────────────────────────

def plot_latency_vs_n(rows: list[dict], filename: str) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, mod_filter, title in [
        (axes[0], "DNA",  "Query Latency — DNA Index (768-dim)"),
        (axes[1], "Text", "Query Latency — Text Index (384-dim)"),
    ]:
        sub = df[df["modality"] == mod_filter]
        for itype, grp in sub.groupby("index_type"):
            grp_sorted = grp.sort_values("n_vectors")
            ax.plot(grp_sorted["n_vectors"], grp_sorted["latency_us_per_query"],
                    marker="o", label=itype)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Corpus size (n vectors)")
        ax.set_ylabel("Latency (µs/query)")
        ax.legend(fontsize=8)
        ax.set_xscale("log")

    plt.suptitle("FAISS Latency: Flat vs HNSW across corpus sizes", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_accuracy_bar(rows: list[dict], filename: str) -> None:
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    colors = ["steelblue" if "Flat" in r else "coral" for r in df["index_type"]]
    bars = ax.bar(x, df["accuracy"], color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['modality']}\n{r['index_type']}\nn={r['n_vectors']}"
         for _, r in df.iterrows()],
        fontsize=7, rotation=15, ha="right"
    )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Top-k Accuracy vs Brute-Force")
    ax.set_title(f"FAISS Retrieval Accuracy (k={1})", fontsize=11)
    ax.axhline(1.0, color="green", linestyle="--", linewidth=1, label="Perfect")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info(f"=== Loading {SEQS_PER_SPECIES*20} sequences ({SEQS_PER_SPECIES}/species × 20 species) ===")
    records_1000 = select_sequences(SEQS_PER_SPECIES)
    print(f"\n{len(records_1000)} records loaded")

    log.info("=== Embedding DNA sequences (DNABERT-S) ===")
    dna_all = embed_dnabert_s([r["sequence"] for r in records_1000])

    log.info("=== Embedding metadata texts (MiniLM) ===")
    txt_all = embed_text([r["metadata"] for r in records_1000])

    # Test sizes spanning the expected Flat→HNSW crossover range
    test_sizes = [100, 250, 500, 1000, 2000, 3000]

    all_rows = []

    for modality, all_vecs in [("DNA", dna_all), ("Text", txt_all)]:
        print(f"\n{'='*60}\nModality: {modality} (dim={all_vecs.shape[1]})\n{'='*60}")

        for n in test_sizes:
            vecs = all_vecs[:n]
            baseline = sklearn_cosine(normalise(vecs))

            for itype in ["flat", "hnsw"]:
                res = build_and_bench(vecs, itype, k=1,
                                      n_reps=N_QUERY_REPS, baseline_sim=baseline)
                label = "IndexFlatIP" if itype == "flat" else f"IndexHNSWFlat(M={HNSW_M})"

                print(f"  n={n:5d}  {label:25s}  "
                      f"build={res['build_time_ms']:7.2f}ms  "
                      f"lat={res['latency_us_per_query']:8.2f}µs  "
                      f"mem={res['memory_bytes']:8,}B  "
                      f"acc={res['accuracy']:.4f}")

                all_rows.append({
                    "modality":             modality,
                    "index_type":           label,
                    "dim":                  all_vecs.shape[1],
                    "n_vectors":            n,
                    **res,
                })

    df = pd.DataFrame(all_rows)

    print("\n\n=== FULL RESULTS TABLE ===")
    print(df[["modality", "index_type", "n_vectors", "build_time_ms",
              "latency_us_per_query", "memory_bytes", "accuracy"]].to_string(index=False))

    csv_path = os.path.join(OUTPUT_DIR, "experiment_8_results.csv")
    df.to_csv(csv_path, index=False)

    plot_latency_vs_n(all_rows, "experiment_8_latency_vs_n.png")
    plot_accuracy_bar(all_rows, "experiment_8_accuracy.png")

    # Crossover analysis
    print("\n=== LATENCY CROSSOVER ANALYSIS ===")
    for mod in ["DNA", "Text"]:
        sub = df[df["modality"] == mod]
        for n in test_sizes:
            row = sub[sub["n_vectors"] == n]
            flat_lat = row[row["index_type"] == "IndexFlatIP"]["latency_us_per_query"].values[0]
            hnsw_lat = row[row["index_type"].str.startswith("IndexHNSW")]["latency_us_per_query"].values[0]
            winner = "Flat" if flat_lat < hnsw_lat else "HNSW"
            ratio = max(flat_lat, hnsw_lat) / min(flat_lat, hnsw_lat)
            print(f"  {mod} n={n:5d}: Flat={flat_lat:7.2f}µs  HNSW={hnsw_lat:7.2f}µs  "
                  f"→ {winner} wins by {ratio:.1f}×")

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
