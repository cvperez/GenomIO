#!/usr/bin/env python3
"""
Experiment 4: Dual-Modality Embedding Strategy — Scaled Validation
Confirms the dual-embedding strategy (DNABERT-S for DNA, all-MiniLM-L6-v2 for metadata)
at scale: 50 CDS records × 20 species = 1,000 sequences.

Run from repo root:
    python experiments/experiment_4_dual_modality_scale.py
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
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(REPO_ROOT, "rag_corpus_uniform")
OUTPUT_DIR = os.path.join(REPO_ROOT, "embedder_benchmark")

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


# ── Data loading ────────────────────────────────────────────────────────────

def build_metadata_text(description: str, organism: str) -> str:
    fields = re.findall(r'\[([^\]]+)\]', description)
    clean = [f for f in fields if not f.startswith("gbkey=") and not f.startswith("location=")]
    return f"organism={organism} " + " ".join(clean)


def select_sequences() -> tuple[list[dict], list[str]]:
    fna_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.fna")))
    records = []
    species_tags = []

    for label, fpath in enumerate(fna_files):
        basename = os.path.basename(fpath)
        accession = "_".join(basename.split("_")[:2])
        organism = ORGANISM_NAMES.get(accession, accession)
        tag = accession
        species_tags.append(tag)
        count = 0
        for rec in SeqIO.parse(fpath, "fasta"):
            seq = str(rec.seq).upper().strip()
            if SEQ_MIN_LEN <= len(seq) <= SEQ_MAX_LEN:
                metadata = build_metadata_text(rec.description, organism)
                records.append({
                    "sequence": seq,
                    "metadata": metadata,
                    "label":    label,
                    "species":  tag,
                    "organism": organism,
                })
                count += 1
                if count == SEQS_PER_SPECIES:
                    break
        if count < SEQS_PER_SPECIES:
            log.warning(f"  {tag}: only {count} qualifying records")
        else:
            log.info(f"  [{label:02d}] {tag} ({organism}): {count} records")

    return records, species_tags


# ── Embeddings ──────────────────────────────────────────────────────────────

def mean_pool(last_hidden_state: torch.Tensor,
              attention_mask: torch.Tensor) -> np.ndarray:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    embedding = (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return embedding.squeeze(0).cpu().numpy()


def embed_dnabert_s(sequences: list[str], desc: str = "") -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    model_name = "zhihan1996/DNABERT-S"
    log.info(f"Loading {model_name} ({desc}) ...")
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
            last_hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
            embeddings.append(mean_pool(last_hidden, enc["attention_mask"]))
            if (i + 1) % 100 == 0:
                log.info(f"  {desc}: {i+1}/{len(sequences)} embedded")
    del model, tokenizer
    return np.stack(embeddings)


def embed_text_model(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    log.info(f"Loading {TEXT_MODEL_ID} ...")
    model = SentenceTransformer(TEXT_MODEL_ID)
    batch_size = 64
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings.append(model.encode(batch, normalize_embeddings=True, show_progress_bar=False))
        if (i + batch_size) % 200 == 0 or i + batch_size >= len(texts):
            log.info(f"  text model: {min(i+batch_size, len(texts))}/{len(texts)} embedded")
    del model
    return np.vstack(embeddings)


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(embeddings: np.ndarray, labels: list[int],
                    n_species: int) -> dict:
    N = len(labels)
    labels_arr = np.array(labels)

    log.info("  Computing cosine similarity matrix ...")
    sim_matrix = cosine_similarity(embeddings)

    i_idx, j_idx = np.triu_indices(N, k=1)
    sim_vals = sim_matrix[i_idx, j_idx]
    intra_mask = labels_arr[i_idx] == labels_arr[j_idx]

    intra_sim  = float(sim_vals[intra_mask].mean()) if intra_mask.any() else float("nan")
    inter_dist = float(1.0 - sim_vals[~intra_mask].mean())

    log.info("  Computing silhouette score ...")
    sil = float(silhouette_score(embeddings, labels))

    species_sim = np.zeros((n_species, n_species))
    for si in range(n_species):
        for sj in range(n_species):
            idx_i = np.where(labels_arr == si)[0]
            idx_j = np.where(labels_arr == sj)[0]
            species_sim[si, sj] = sim_matrix[np.ix_(idx_i, idx_j)].mean()

    return {
        "sim_matrix":  sim_matrix,
        "species_sim": species_sim,
        "intra_sim":   intra_sim,
        "inter_dist":  inter_dist,
        "silhouette":  sil,
    }


# ── Visualizations ──────────────────────────────────────────────────────────

def plot_species_heatmap(species_sim: np.ndarray, title: str,
                         species_tags: list[str], filename: str) -> None:
    short = [t.split("_")[0] + "_" + t.split("_")[1] for t in species_tags]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(species_sim, cmap="viridis", vmin=0, vmax=1,
                xticklabels=short, yticklabels=short, ax=ax,
                linewidths=0.3, linecolor="white")
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_pca(embeddings: np.ndarray, labels: list[int],
             species_tags: list[str], title: str, filename: str) -> None:
    coords = PCA(n_components=2).fit_transform(embeddings)
    labels_arr = np.array(labels)
    n_species = len(species_tags)
    short = [t.split("_")[0] + "_" + t.split("_")[1] for t in species_tags]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(12, 8))
    for sp in range(n_species):
        mask = labels_arr == sp
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=20, alpha=0.6, color=cmap(sp / n_species), label=short[sp])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="upper right", fontsize=6, ncol=2, markerscale=1.5)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=== Selecting sequences from rag_corpus_uniform ===")
    records, species_tags = select_sequences()
    sequences      = [r["sequence"] for r in records]
    metadata_texts = [r["metadata"] for r in records]
    labels         = [r["label"]    for r in records]
    n_species      = len(species_tags)

    print(f"\n{len(sequences)} sequences across {n_species} species "
          f"({SEQS_PER_SPECIES}/species, {SEQ_MIN_LEN}–{SEQ_MAX_LEN} bp)")

    all_results = []

    # ── A. DNA embeddings — DNABERT-S ──────────────────────────────────────
    print(f"\n{'='*60}\nModality: DNA  |  Model: DNABERT-S\n{'='*60}")
    t0 = time.perf_counter()
    dna_embs = embed_dnabert_s(sequences, desc="DNA")
    dna_time = time.perf_counter() - t0
    dna_metrics = compute_metrics(dna_embs, labels, n_species)
    plot_species_heatmap(dna_metrics["species_sim"],
                         "Species-Mean DNA Similarity (DNABERT-S)",
                         species_tags, "experiment_4_dna_species_heatmap.png")
    plot_pca(dna_embs, labels, species_tags,
             "PCA of DNA Embeddings (DNABERT-S)",
             "experiment_4_dna_pca.png")
    all_results.append({
        "modality": "DNA", "model": "DNABERT-S", "dim": dna_embs.shape[1],
        "intra_species_sim":  round(dna_metrics["intra_sim"],  4),
        "inter_species_dist": round(dna_metrics["inter_dist"], 4),
        "silhouette_score":   round(dna_metrics["silhouette"], 4),
        "inference_time_per_seq": round(dna_time / len(sequences), 3),
        "total_time_s": round(dna_time, 1),
    })
    print(f"  silhouette: {dna_metrics['silhouette']:.4f}  |  "
          f"intra_sim: {dna_metrics['intra_sim']:.4f}  |  "
          f"inter_dist: {dna_metrics['inter_dist']:.4f}  |  "
          f"time: {dna_time:.1f}s")

    # ── B. Metadata embeddings — DNABERT-S (control) ───────────────────────
    print(f"\n{'='*60}\nModality: Metadata  |  Model: DNABERT-S (control)\n{'='*60}")
    t0 = time.perf_counter()
    meta_embs_ds = embed_dnabert_s(metadata_texts, desc="Metadata/DNABERT-S")
    meta_ds_time = time.perf_counter() - t0
    meta_ds_metrics = compute_metrics(meta_embs_ds, labels, n_species)
    plot_species_heatmap(meta_ds_metrics["species_sim"],
                         "Species-Mean Metadata Similarity (DNABERT-S on text — control)",
                         species_tags, "experiment_4_meta_dnaberts_species_heatmap.png")
    plot_pca(meta_embs_ds, labels, species_tags,
             "PCA of Metadata Embeddings (DNABERT-S on text — control)",
             "experiment_4_meta_dnaberts_pca.png")
    all_results.append({
        "modality": "Metadata", "model": "DNABERT-S (control)", "dim": meta_embs_ds.shape[1],
        "intra_species_sim":  round(meta_ds_metrics["intra_sim"],  4),
        "inter_species_dist": round(meta_ds_metrics["inter_dist"], 4),
        "silhouette_score":   round(meta_ds_metrics["silhouette"], 4),
        "inference_time_per_seq": round(meta_ds_time / len(sequences), 3),
        "total_time_s": round(meta_ds_time, 1),
    })
    print(f"  silhouette: {meta_ds_metrics['silhouette']:.4f}  |  "
          f"intra_sim: {meta_ds_metrics['intra_sim']:.4f}  |  "
          f"inter_dist: {meta_ds_metrics['inter_dist']:.4f}  |  "
          f"time: {meta_ds_time:.1f}s")

    # ── C. Metadata embeddings — text model (confirmed strategy) ───────────
    print(f"\n{'='*60}\nModality: Metadata  |  Model: {TEXT_MODEL_ID}\n{'='*60}")
    t0 = time.perf_counter()
    meta_embs_txt = embed_text_model(metadata_texts)
    meta_txt_time = time.perf_counter() - t0
    meta_txt_metrics = compute_metrics(meta_embs_txt, labels, n_species)
    plot_species_heatmap(meta_txt_metrics["species_sim"],
                         f"Species-Mean Metadata Similarity ({TEXT_MODEL_ID})",
                         species_tags, "experiment_4_meta_textmodel_species_heatmap.png")
    plot_pca(meta_embs_txt, labels, species_tags,
             f"PCA of Metadata Embeddings ({TEXT_MODEL_ID})",
             "experiment_4_meta_textmodel_pca.png")
    all_results.append({
        "modality": "Metadata", "model": TEXT_MODEL_ID, "dim": meta_embs_txt.shape[1],
        "intra_species_sim":  round(meta_txt_metrics["intra_sim"],  4),
        "inter_species_dist": round(meta_txt_metrics["inter_dist"], 4),
        "silhouette_score":   round(meta_txt_metrics["silhouette"], 4),
        "inference_time_per_seq": round(meta_txt_time / len(sequences), 3),
        "total_time_s": round(meta_txt_time, 1),
    })
    print(f"  silhouette: {meta_txt_metrics['silhouette']:.4f}  |  "
          f"intra_sim: {meta_txt_metrics['intra_sim']:.4f}  |  "
          f"inter_dist: {meta_txt_metrics['inter_dist']:.4f}  |  "
          f"time: {meta_txt_time:.1f}s")

    # ── Summary ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    print("\n\n=== COMPARISON TABLE ===")
    print(df.to_string(index=False))

    csv_path = os.path.join(OUTPUT_DIR, "experiment_4_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
