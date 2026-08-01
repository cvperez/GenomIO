#!/usr/bin/env python3
"""
Experiment 3: Dual-Modality Embedding Strategy
Tests whether DNABERT-S can handle metadata text, or whether a separate
general-purpose text model is required for a dual-modality RAG index.

Run from repo root:
    python experiments/experiment_3_dual_modality.py
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
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(REPO_ROOT, "rag_corpus_uniform")
OUTPUT_DIR = os.path.join(REPO_ROOT, "embedder_benchmark")

# Same 5 records as Experiment 1
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

SEQ_MIN_LEN = 300
SEQ_MAX_LEN = 900

TEXT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


# ── Data loading ────────────────────────────────────────────────────────────

def build_metadata_text(description: str, organism: str) -> str:
    """Extract bracket-enclosed fields from FASTA description and prepend organism."""
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
                    metadata = build_metadata_text(rec.description, organism)
                    results.append({
                        "tag":      tag,
                        "sequence": seq,
                        "metadata": metadata,
                        "label":    label,
                        "organism": organism,
                    })
                    log.info(f"  {tag}: {rec.id}  ({len(seq)} bp)")
                    log.info(f"        metadata: {metadata}")
                    break
    return results


# ── Embedding: DNABERT-S ────────────────────────────────────────────────────

def mean_pool(last_hidden_state: torch.Tensor,
              attention_mask: torch.Tensor) -> np.ndarray:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    embedding = (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return embedding.squeeze(0).cpu().numpy()


def embed_dnabert_s(inputs: list[str], label: str = "input") -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    model_name = "zhihan1996/DNABERT-S"
    log.info(f"Loading {model_name} for {label} ...")
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


# ── Embedding: general-purpose text model ──────────────────────────────────

def embed_text_model(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    log.info(f"Loading {TEXT_MODEL_ID} ...")
    model = SentenceTransformer(TEXT_MODEL_ID)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    del model
    return np.array(embeddings)


# ── Visualisation ───────────────────────────────────────────────────────────

def plot_heatmap(matrix: np.ndarray, title: str, filename: str,
                 row_labels: list[str], col_labels: list[str],
                 vmin: float = -1.0, vmax: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


# ── Cross-modal rank analysis ───────────────────────────────────────────────

def cross_modal_ranks(dna_embs: np.ndarray, meta_embs: np.ndarray,
                      labels: list[str]) -> dict:
    """
    For each DNA sequence, compute cosine similarity against all metadata embeddings.
    Report rank of the matching metadata record (rank 1 = best match).
    Embeddings must be in the same space (same model, same dim).
    """
    sim = cosine_similarity(dna_embs, meta_embs)  # (5, 5)
    ranks = {}
    for i, lbl in enumerate(labels):
        row = sim[i]
        sorted_idx = np.argsort(row)[::-1]
        rank = int(np.where(sorted_idx == i)[0][0]) + 1
        ranks[lbl] = {"rank": rank, "similarity": float(row[i])}
    return ranks


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=== Loading 5 CDS records (same as Experiment 1) ===")
    records = select_sequences()
    sequences     = [r["sequence"] for r in records]
    metadata_texts = [r["metadata"] for r in records]
    labels        = [r["label"]    for r in records]

    print("\nMetadata texts extracted:")
    for r in records:
        print(f"  {r['tag']:12s}: {r['metadata']}")

    # ── 1. DNA embeddings via DNABERT-S ────────────────────────────────────
    t0 = time.perf_counter()
    dna_embs_ds = embed_dnabert_s(sequences, label="DNA sequences")
    dna_time = time.perf_counter() - t0
    log.info(f"DNA embeddings (DNABERT-S): {dna_embs_ds.shape}, {dna_time:.2f}s")

    # ── 2. Metadata embeddings via DNABERT-S ───────────────────────────────
    t0 = time.perf_counter()
    meta_embs_ds = embed_dnabert_s(metadata_texts, label="metadata (DNABERT-S)")
    meta_ds_time = time.perf_counter() - t0
    log.info(f"Metadata embeddings (DNABERT-S): {meta_embs_ds.shape}, {meta_ds_time:.2f}s")

    # ── 3. Metadata embeddings via text model ──────────────────────────────
    t0 = time.perf_counter()
    meta_embs_txt = embed_text_model(metadata_texts)
    meta_txt_time = time.perf_counter() - t0
    log.info(f"Metadata embeddings (text model): {meta_embs_txt.shape}, {meta_txt_time:.2f}s")

    # ── Similarity matrices ────────────────────────────────────────────────
    dna_dna_sim        = cosine_similarity(dna_embs_ds)
    meta_meta_ds_sim   = cosine_similarity(meta_embs_ds)
    meta_meta_txt_sim  = cosine_similarity(meta_embs_txt)
    dna_meta_ds_sim    = cosine_similarity(dna_embs_ds, meta_embs_ds)

    # ── Silhouette scores ──────────────────────────────────────────────────
    sil_dna_ds  = float(silhouette_score(dna_embs_ds,  labels))
    sil_meta_ds = float(silhouette_score(meta_embs_ds, labels))
    sil_meta_txt = float(silhouette_score(meta_embs_txt, labels))

    # ── Cross-modal ranks (DNABERT-S space only, same dim) ─────────────────
    ranks = cross_modal_ranks(dna_embs_ds, meta_embs_ds, AXIS_LABELS)

    # ── Embedding space compatibility ──────────────────────────────────────
    dna_dim  = dna_embs_ds.shape[1]
    txt_dim  = meta_embs_txt.shape[1]
    compatible = dna_dim == txt_dim

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_heatmap(dna_dna_sim,
                 "DNA–DNA Cosine Similarity (DNABERT-S)",
                 "experiment_3_dna_dna_heatmap.png",
                 AXIS_LABELS, AXIS_LABELS)

    plot_heatmap(meta_meta_ds_sim,
                 "Metadata–Metadata Cosine Similarity (DNABERT-S on text)",
                 "experiment_3_meta_meta_dnaberts_heatmap.png",
                 AXIS_LABELS, AXIS_LABELS)

    plot_heatmap(meta_meta_txt_sim,
                 f"Metadata–Metadata Cosine Similarity ({TEXT_MODEL_ID})",
                 "experiment_3_meta_meta_textmodel_heatmap.png",
                 AXIS_LABELS, AXIS_LABELS)

    plot_heatmap(dna_meta_ds_sim,
                 "DNA–Metadata Cross-Modal Similarity (DNABERT-S space)",
                 "experiment_3_dna_meta_crossmodal_heatmap.png",
                 AXIS_LABELS, AXIS_LABELS, vmin=0.0, vmax=1.0)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n=== SILHOUETTE SCORES ===")
    print(f"  DNA (DNABERT-S)           : {sil_dna_ds:.4f}")
    print(f"  Metadata (DNABERT-S)      : {sil_meta_ds:.4f}")
    print(f"  Metadata (text model)     : {sil_meta_txt:.4f}")

    print("\n=== CROSS-MODAL RANKS (DNA query → metadata index, DNABERT-S space) ===")
    for lbl, info in ranks.items():
        print(f"  {lbl:20s}: rank {info['rank']}/5  (sim={info['similarity']:.4f})")

    print(f"\n=== EMBEDDING DIMENSIONS ===")
    print(f"  DNABERT-S (DNA + metadata) : {dna_dim}")
    print(f"  Text model (metadata)      : {txt_dim}")
    print(f"  Dimensions compatible      : {compatible}")
    print(f"  → Single FAISS index possible: {'YES' if compatible else 'NO — dual index required'}")

    # ── Save CSV ───────────────────────────────────────────────────────────
    rows = [
        {"modality": "DNA",      "model": "DNABERT-S",    "dim": dna_dim,  "silhouette": round(sil_dna_ds, 4),   "time_s": round(dna_time, 3)},
        {"modality": "Metadata", "model": "DNABERT-S",    "dim": dna_dim,  "silhouette": round(sil_meta_ds, 4),  "time_s": round(meta_ds_time, 3)},
        {"modality": "Metadata", "model": TEXT_MODEL_ID,  "dim": txt_dim,  "silhouette": round(sil_meta_txt, 4), "time_s": round(meta_txt_time, 3)},
    ]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "experiment_3_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # ── Print similarity matrices for logging ──────────────────────────────
    print("\n=== DNA-DNA SIM MATRIX (DNABERT-S) ===")
    print(np.round(dna_dna_sim, 3))
    print("\n=== META-META SIM MATRIX (DNABERT-S on text) ===")
    print(np.round(meta_meta_ds_sim, 3))
    print("\n=== META-META SIM MATRIX (text model) ===")
    print(np.round(meta_meta_txt_sim, 3))
    print("\n=== DNA-META CROSS-MODAL SIM MATRIX (DNABERT-S space) ===")
    print(np.round(dna_meta_ds_sim, 3))


if __name__ == "__main__":
    main()
