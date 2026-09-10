#!/usr/bin/env python3
"""
Experiment 12: Recall@K Curves and the Hybrid Fusion Decision
=============================================================

Closes the threads Experiment 10 left open:
  (1) it reported only two Recall points (Recall@10, Recall@20) per query type —
      never the full Recall@K curve;
  (2) it never tested whether Index A (DNA) and Index B (text) should be *fused*.

Scope is retrieval only. We score ranked id-lists against same-species relevance.
The generator is not run, no gap is filled, no assembled sequence is scored —
that is the separate, downstream level-2 evaluation, explicitly out of scope.

Index A : DNABERT-S DNA embeddings, IndexHNSWFlat(M=32, efSearch=64), 768-dim (Exp 10).
Index B : the text encoder selected in Experiment 11 — SapBERT (768-dim, CLS-pooled),
          IndexFlatIP. SapBERT was chosen over the marginally-higher-P@1 e5-base-v2
          because Experiment 12 is built around Recall@K and SapBERT leads there
          (Recall@49 0.893 vs 0.859) while sitting within 0.4 pp of e5 on P@1.

Three retrievers on the same K axes:
  1. A alone   (DNA query  → Index A)
  2. B alone   (text query → Index B)
  3. A (+) B   reciprocal rank fusion: each query contributes its DNA to A and its
               metadata to B simultaneously; the two ranked id-lists are merged.

Two relevance lenses, both reported:
  - fraction-recall (primary): |relevant ∩ topK| / |relevant|
  - hit-rate (RAG-facing):     1 if ≥1 relevant in topK

Run from repo root:
    python embedder_benchmark/experiment_12_recall_fusion/experiment_12_recall_fusion.py
"""
import glob
import logging
import os
import re
import sys
import time
import warnings

import faiss
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_DIR = os.path.join(REPO_ROOT, "rag_corpus_uniform")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pick the log file by entry point so the `ablation` sub-command never truncates the
# committed full-experiment log (FileHandler(mode="w") truncates at handler creation).
_ABLATION = len(sys.argv) > 1 and sys.argv[1] == "ablation"
_LOGFILE = "experiment_12_ablation_run.log" if _ABLATION else "experiment_12_run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, _LOGFILE), mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers",
              "filelock", "transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

SEQS_PER_SPECIES = 50
SEQ_MIN_LEN      = 300
SEQ_MAX_LEN      = 900
HNSW_M           = 32

K_SWEEP     = [1, 3, 5, 10, 15, 20, 30, 40, 49]
MAX_K       = 50      # 49 relevant + self → 49 ranked candidates after leave-self-out
RRF_C       = 60      # standard RRF constant; needs no tuning (ranks, not raw scores)
RRF_WEIGHTS = [1.0, 2.0]   # text up-weight sweep (Index B is the stronger ranker)

# Index B encoder selected by Experiment 11 (winner handoff — edit here if Exp 11 rerun
# changes the choice). SapBERT uses CLS pooling via the transformers_cls loader.
SELECTED_ENCODER = {
    "name": "SapBERT", "hf_id": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "dim": 768, "loader": "transformers_cls", "prefix": "",
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


# ── Data loading (copied from experiment_10, incl. the .1-suffix fix) ──────────

def build_metadata_text(description: str, organism: str) -> str:
    fields = re.findall(r"\[([^\]]+)\]", description)
    clean = [f for f in fields if not f.startswith("gbkey=") and not f.startswith("location=")]
    return f"organism={organism} " + " ".join(clean)


def select_sequences() -> tuple[list[dict], list[str]]:
    from Bio import SeqIO
    fna_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.fna")))
    records, species_tags = [], []
    for label, fpath in enumerate(fna_files):
        basename = os.path.basename(fpath)
        parts = basename.split("_")
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

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    emb = (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return emb.squeeze(0).cpu().numpy()


def embed_dnabert_s(sequences: list[str]) -> np.ndarray:
    from transformers import AutoConfig, AutoModel, AutoTokenizer
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


def load_text_encoder(spec: dict):
    """Return embed(texts) -> np.ndarray[float32] (raw, un-normalised).
    Same factory as Experiment 11 — supports the selected encoder's loader.
    """
    loader = spec["loader"]
    hf     = spec["hf_id"]
    prefix = spec.get("prefix", "")

    if loader in ("st", "st_prefix"):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(hf)

        def embed(texts):
            t = [prefix + x for x in texts] if prefix else texts
            out = []
            for i in range(0, len(t), 64):
                out.append(model.encode(t[i:i + 64], normalize_embeddings=False,
                                        show_progress_bar=False, convert_to_numpy=True))
                if (i + 64) % 256 == 0 or i + 64 >= len(t):
                    log.info(f"  {spec['name']}: {min(i + 64, len(t))}/{len(t)}")
            return np.vstack(out).astype(np.float32)

        return embed

    if loader == "transformers_cls":            # SapBERT → CLS token (not mean pool)
        from transformers import AutoModel, AutoTokenizer
        tok   = AutoTokenizer.from_pretrained(hf)
        model = AutoModel.from_pretrained(hf)
        model.eval()

        def embed(texts):
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), 32):
                    enc = tok(texts[i:i + 32], return_tensors="pt",
                              padding=True, truncation=True, max_length=64)
                    hs = model(**enc).last_hidden_state
                    out.append(hs[:, 0, :].cpu().numpy())       # CLS
                    if (i + 32) % 256 == 0 or i + 32 >= len(texts):
                        log.info(f"  {spec['name']}: {min(i + 32, len(texts))}/{len(texts)}")
            return np.vstack(out).astype(np.float32)

        return embed

    raise ValueError(f"Unknown loader: {loader}")


# ── FAISS helpers (copied from experiment_10) ─────────────────────────────────

def normalise(v: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(v.astype(np.float32))
    faiss.normalize_L2(u)
    return u


def build_hnsw_index(vectors: np.ndarray) -> tuple:
    v = normalise(vectors)
    idx = faiss.IndexHNSWFlat(v.shape[1], HNSW_M)
    idx.hnsw.efConstruction = 200
    idx.hnsw.efSearch = 64
    t0 = time.perf_counter()
    idx.add(v)
    return idx, v, (time.perf_counter() - t0) * 1000


def build_flat_index(vectors: np.ndarray) -> tuple:
    v = normalise(vectors)
    idx = faiss.IndexFlatIP(v.shape[1])
    t0 = time.perf_counter()
    idx.add(v)
    return idx, v, (time.perf_counter() - t0) * 1000


def index_memory_bytes(index: faiss.Index) -> int:
    return len(faiss.serialize_index(index))


# ── Reciprocal rank fusion ────────────────────────────────────────────────────

def rrf_fuse(ids_A: np.ndarray, ids_B: np.ndarray,
             max_k: int = MAX_K, c: int = RRF_C, w: float = 1.0) -> np.ndarray:
    """Merge two per-query neighbour-id lists by reciprocal rank fusion.

    RRF(d) = 1/(c + rankA(d)) + w * 1/(c + rankB(d))

    Ranks are 1-based over the *self-excluded* lists: the query's own id is
    skipped without consuming a rank slot, so both lists share the same 1-based
    footing (leave-self-out consistent). A doc absent from a list contributes 0
    from that list. Ties keep insertion order (A inserted first → deterministic).

    ids_A, ids_B : (N, max_k) from index.search (self may be present, -1 = padding).
    Returns fused (N, max_k) id matrix, self-excluded, -1 padded.
    """
    N = ids_A.shape[0]
    fused = np.full((N, max_k), -1, dtype=np.int64)
    for i in range(N):
        score = {}
        for ids, weight in ((ids_A, 1.0), (ids_B, w)):
            rank = 0
            for d in ids[i]:
                d = int(d)
                if d == i or d < 0:        # skip self / padding, don't consume a rank
                    continue
                rank += 1
                score[d] = score.get(d, 0.0) + weight * (1.0 / (c + rank))
        ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
        for j, (d, _) in enumerate(ranked[:max_k]):
            fused[i, j] = d
    return fused


# ── Retrieval evaluation (shared with Exp 11) ─────────────────────────────────

def evaluate_retrieval(all_ids: np.ndarray, labels: np.ndarray,
                       query_type: str, k_values: list[int]) -> pd.DataFrame:
    N = len(labels)
    n_relevant = np.array([(labels == labels[i]).sum() - 1 for i in range(N)])
    rows = []
    for i in range(N):
        n_rel = int(n_relevant[i])
        if n_rel == 0:
            continue
        ranked = [int(x) for x in all_ids[i] if x != i and x >= 0]
        rel = np.array([labels[j] == labels[i] for j in ranked], dtype=bool)
        row = {"query_type": query_type, "query_label": int(labels[i]), "n_relevant": n_rel}
        for k in k_values:
            hits = int(rel[:k].sum())
            row[f"P@{k}"]      = hits / k
            row[f"Recall@{k}"] = hits / n_rel
            row[f"Hit@{k}"]    = 1.0 if hits >= 1 else 0.0
        first = int(np.argmax(rel)) if rel.any() else len(rel)
        row["RR"] = 1.0 / (first + 1) if rel.any() else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def curve_and_scalars(df: pd.DataFrame, retriever: str) -> tuple[pd.DataFrame, dict]:
    """Per-K fraction-recall & hit-rate curve + scalar P@1/MRR for one retriever."""
    curve = pd.DataFrame({
        "retriever": retriever,
        "K": K_SWEEP,
        "fraction_recall": [float(df[f"Recall@{k}"].mean()) for k in K_SWEEP],
        "hit_rate":        [float(df[f"Hit@{k}"].mean())    for k in K_SWEEP],
    })
    scal = {"retriever": retriever, "P@1": float(df["P@1"].mean()),
            "MRR": float(df["RR"].mean())}
    return curve, scal


def find_knee(hit_rate: list[float], k_values: list[int], frac: float = 0.99) -> int:
    """Operating K* = smallest K whose hit-rate reaches `frac` of the K=max value."""
    target = frac * hit_rate[-1]
    for k, hr in zip(k_values, hit_rate):
        if hr >= target:
            return k
    return k_values[-1]


# ── Visualisation ─────────────────────────────────────────────────────────────

def _curve_plot(curves: pd.DataFrame, metric: str, title: str, ylabel: str,
                filename: str, kstar: int | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, rt in enumerate(curves["retriever"].unique()):
        sub = curves[curves["retriever"] == rt]
        ax.plot(sub["K"], sub[metric], marker="o", markersize=4, linewidth=1.9,
                color=cmap(i % 10), label=rt)
    if kstar is not None:
        ax.axvline(kstar, color="gray", linestyle=":", linewidth=1, label=f"K*={kstar}")
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(K_SWEEP)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_fusion_delta(curves: pd.DataFrame, filename: str) -> None:
    singles = curves[curves["retriever"].isin(["A (DNA)", "B (Text)"])]
    best_single = singles.groupby("K")[["fraction_recall", "hit_rate"]].max()
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    fused = [rt for rt in curves["retriever"].unique() if rt.startswith("A(+)B")]
    for i, rt in enumerate(fused):
        sub = curves[curves["retriever"] == rt].set_index("K")
        ax.plot(K_SWEEP, [sub.loc[k, "fraction_recall"] - best_single.loc[k, "fraction_recall"]
                          for k in K_SWEEP], marker="o", markersize=4,
                color=cmap(i % 10), label=f"{rt} − best single (fraction-recall)")
        ax.plot(K_SWEEP, [sub.loc[k, "hit_rate"] - best_single.loc[k, "hit_rate"]
                          for k in K_SWEEP], marker="s", markersize=4, linestyle="--",
                color=cmap(i % 10), label=f"{rt} − best single (hit-rate)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("K")
    ax.set_ylabel("Δ vs best single index")
    ax.set_title("Figure 12.3 — Fusion delta (A(+)B minus best single)", fontsize=11)
    ax.set_xticks(K_SWEEP)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


# ── Worked example (real numbers for the write-up) ────────────────────────────

def _fuse_single(row_a, row_b, qi, w):
    """Single-query RRF that correctly excludes the true query id `qi`.
    (rrf_fuse uses d == row-index for self-exclusion, which only works on the
    full matrix; here we pass the global qi explicitly.)"""
    score = {}
    for ids, weight in ((row_a, 1.0), (row_b, w)):
        rank = 0
        for d in ids:
            d = int(d)
            if d == qi or d < 0:
                continue
            rank += 1
            score[d] = score.get(d, 0.0) + weight * (1.0 / (RRF_C + rank))
    return [d for d, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def worked_example(records, labels, idsA, idsB, species_tags) -> None:
    """Print real per-list ranks for a query that *demonstrates* fusion's value:
    DNA ranks a wrong-species record at #1, text ranks a same-species record at
    #1. Falls back to the first Thermus thermophilus record if none is found.
    Numbers are verifiable from this run and feed the results.md worked example.
    """
    N = len(labels)

    def first_nonself(row, qi):
        for d in row:
            d = int(d)
            if d != qi and d >= 0:
                return d
        return -1

    qi = None
    for cand in range(N):
        a1, b1 = first_nonself(idsA[cand], cand), first_nonself(idsB[cand], cand)
        if a1 < 0 or b1 < 0:
            continue
        if labels[a1] != labels[cand] and labels[b1] == labels[cand]:
            qi = cand
            break
    if qi is None:                                   # fallback: a T. thermophilus query
        qi = int(np.where(labels == species_tags.index("GCF_000008125"))[0][0])

    org = records[qi]["organism"]
    print(f"\n{'=' * 64}\nWORKED EXAMPLE — query record {qi} ({org})\n{'=' * 64}")
    print(f"  metadata: {records[qi]['metadata'][:90]}")

    def ranks_of(ids_row, ids_other):
        rrow = [int(x) for x in ids_row if x != qi and x >= 0]
        rother = [int(x) for x in ids_other if x != qi and x >= 0]
        pos_other = {d: p + 1 for p, d in enumerate(rother)}
        out = []
        for p, d in enumerate(rrow[:5], start=1):
            same = "same-sp" if labels[d] == labels[qi] else f"OTHER:{species_tags[labels[d]]}"
            out.append((p, d, pos_other.get(d, "—"), same))
        return out

    print("  Index A (DNA) top-5  [rankA, id, rankB, species]:")
    for p, d, rb, s in ranks_of(idsA[qi], idsB[qi]):
        print(f"    A{p}: id={d:4d}  B-rank={rb!s:>3}  {s}")
    print("  Index B (Text) top-5 [rankB, id, rankA, species]:")
    for p, d, ra, s in ranks_of(idsB[qi], idsA[qi]):
        print(f"    B{p}: id={d:4d}  A-rank={ra!s:>3}  {s}")
    for w in RRF_WEIGHTS:
        top = _fuse_single(idsA[qi], idsB[qi], qi, w)[:5]
        tags = ["same-sp" if labels[d] == labels[qi]
                else f"OTHER:{species_tags[labels[d]]}" for d in top]
        print(f"  Fused (w={w}) top-5: " +
              ", ".join(f"id={d}({t})" for d, t in zip(top, tags)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=== Selecting sequences ===")
    records, species_tags = select_sequences()
    seqs   = [r["sequence"] for r in records]
    metas  = [r["metadata"] for r in records]
    labels = np.array([r["label"] for r in records])
    N = len(records)
    print(f"\n{N} records across {len(species_tags)} species. "
          f"Index B encoder = {SELECTED_ENCODER['name']}.")

    # ── Embed ────────────────────────────────────────────────────────────────
    log.info("=== Embedding DNA (DNABERT-S) for Index A ===")
    t0 = time.perf_counter()
    dna_raw = embed_dnabert_s(seqs)
    dna_s = time.perf_counter() - t0

    log.info(f"=== Embedding metadata ({SELECTED_ENCODER['name']}) for Index B ===")
    t0 = time.perf_counter()
    txt_raw = load_text_encoder(SELECTED_ENCODER)(metas)
    txt_s = time.perf_counter() - t0

    # ── Build indices ─────────────────────────────────────────────────────────
    log.info("=== Building Index A (HNSW, DNA) ===")
    idxA, dnaA, buildA = build_hnsw_index(dna_raw)
    log.info("=== Building Index B (Flat, Text) ===")
    idxB, txtB, buildB = build_flat_index(txt_raw)

    print(f"\n=== INDEX STATS ===")
    print(f"  A (DNA  768, HNSW M={HNSW_M}): build={buildA:.1f}ms  "
          f"mem={index_memory_bytes(idxA)/1024:.0f}KB  embed={dna_s:.1f}s")
    print(f"  B (Text {SELECTED_ENCODER['dim']}, Flat):       build={buildB:.1f}ms  "
          f"mem={index_memory_bytes(idxB)/1024:.0f}KB  embed={txt_s:.1f}s")

    faiss.write_index(idxA, os.path.join(OUTPUT_DIR, "experiment_12_index_dna.faiss"))
    faiss.write_index(idxB, os.path.join(OUTPUT_DIR, "experiment_12_index_text.faiss"))

    # ── Search each index once (self included; filtered downstream) ───────────
    _, idsA = idxA.search(dnaA, MAX_K)
    _, idsB = idxB.search(txtB, MAX_K)

    # ── Retrievers ─────────────────────────────────────────────────────────────
    retrievers = {"A (DNA)": idsA, "B (Text)": idsB}
    for w in RRF_WEIGHTS:
        retrievers[f"A(+)B (w={w})"] = rrf_fuse(idsA, idsB, w=w)

    curves, scalars, per_query = [], [], []
    for name, ids in retrievers.items():
        df = evaluate_retrieval(ids, labels, name, K_SWEEP)
        df["retriever"] = name
        per_query.append(df)
        c, s = curve_and_scalars(df, name)
        curves.append(c)
        scalars.append(s)
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        print(f"  P@1={s['P@1']:.4f}  MRR={s['MRR']:.4f}")
        for k in K_SWEEP:
            cr = c[c["K"] == k].iloc[0]
            print(f"  K={k:2d}: fraction-recall={cr['fraction_recall']:.4f}  "
                  f"hit-rate={cr['hit_rate']:.4f}")

    curves_df  = pd.concat(curves, ignore_index=True)
    scalars_df = pd.DataFrame(scalars)

    # ── Operating K* (knee of best hit-rate curve) ────────────────────────────
    best_rt = curves_df.loc[curves_df[curves_df["K"] == 10]["hit_rate"].idxmax(), "retriever"]
    best_hr = curves_df[curves_df["retriever"] == best_rt]["hit_rate"].tolist()
    kstar = find_knee(best_hr, K_SWEEP)
    print(f"\nOperating K* = {kstar} (knee of {best_rt} hit-rate curve)")

    # ── Figures ───────────────────────────────────────────────────────────────
    _curve_plot(curves_df, "fraction_recall",
                "Figure 12.1 — Fraction-recall vs K (same-species)",
                "Fraction-recall", "experiment_12_fraction_recall.png", kstar)
    _curve_plot(curves_df, "hit_rate",
                "Figure 12.2 — Hit-rate vs K (≥1 relevant in top-K)",
                "Hit-rate", "experiment_12_hitrate.png", kstar)
    plot_fusion_delta(curves_df, "experiment_12_fusion_delta.png")

    # ── Gate analysis ──────────────────────────────────────────────────────────
    def at(rt, k, metric):
        return float(curves_df[(curves_df["retriever"] == rt) &
                               (curves_df["K"] == k)][metric].iloc[0])

    best_single_fr = max(at("A (DNA)", kstar, "fraction_recall"),
                         at("B (Text)", kstar, "fraction_recall"))
    best_fused_rt = max([r for r in retrievers if r.startswith("A(+)B")],
                        key=lambda r: at(r, kstar, "fraction_recall"))
    fused_fr = at(best_fused_rt, kstar, "fraction_recall")
    print(f"\n{'=' * 60}\nGATE ANALYSIS @ K*={kstar}\n{'=' * 60}")
    print(f"  Gate 1 (fusion): best single FR={best_single_fr:.4f}  "
          f"best fused ({best_fused_rt}) FR={fused_fr:.4f}  "
          f"Δ={fused_fr - best_single_fr:+.4f}  "
          f"→ {'IMPLEMENT RRF' if fused_fr - best_single_fr >= 0.05 else 'fusion not material'}")
    print(f"  Gate 2 (DNA sufficiency): A FR={at('A (DNA)', kstar, 'fraction_recall'):.4f}  "
          f"B FR={at('B (Text)', kstar, 'fraction_recall'):.4f}")
    print(f"  Gate 3 (bottleneck): best hit-rate@K*="
          f"{max(at(r, kstar, 'hit_rate') for r in retrievers):.4f}")

    # ── Worked example (real numbers) ──────────────────────────────────────────
    worked_example(records, labels, idsA, idsB, species_tags)

    # ── Save ────────────────────────────────────────────────────────────────────
    merged = curves_df.merge(scalars_df, on="retriever")
    merged.to_csv(os.path.join(OUTPUT_DIR, "experiment_12_summary.csv"), index=False)
    pd.concat(per_query, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "experiment_12_results.csv"), index=False)
    print(f"\nResults saved to: {os.path.join(OUTPUT_DIR, 'experiment_12_summary.csv')}")

    # Mirror the key results into run.log (the FileHandler only sees log.* calls)
    log.info("RESULTS (retriever × K):\n" + merged.to_string(index=False))
    log.info(f"Operating K*={kstar}. "
             f"Gate 1 fusion Δ(fraction-recall)@K*={fused_fr - best_single_fr:+.4f} "
             f"({'IMPLEMENT RRF' if fused_fr - best_single_fr >= 0.05 else 'not material'}); "
             f"Gate 3 best hit-rate@K*={max(at(r, kstar, 'hit_rate') for r in retrievers):.4f}.")


# ── Follow-up ablation: organism-only text query ─────────────────────────────
#
# Realistic-inference regime. Experiment 12's headline Text->DNA numbers use a
# full-metadata query (organism + protein) and are an UPPER BOUND. At real
# inference the text query collapses to the assembly organism alone (the species
# is derivable from the contig accession; the missing gene's function is not).
# This ablation adds exactly one query level — Q_organism — and re-tests the
# Gate-1 fusion verdict in that regime. It reuses the committed Exp 12 indices and
# metric code; the ONLY new computation is the organism-only text query. The
# generator is still not run (scope: retrieval only).

ABLATION_KSTAR = 3   # operating K* carried from Experiment 12 (knee of hit-rate)
_RETR_MAP = {"A (DNA)": "DNA", "B (Text)": "Text-only", "A(+)B (w=2.0)": "Fusion w=2"}
_ABL_COLS = ["query_level", "retriever", "K", "fraction_recall", "hit_rate", "P@1", "MRR"]


def _rows_from_eval(df_perquery: pd.DataFrame, retriever: str, level: str) -> pd.DataFrame:
    curve, scal = curve_and_scalars(df_perquery, retriever)
    curve["P@1"] = scal["P@1"]
    curve["MRR"] = scal["MRR"]
    curve.insert(0, "query_level", level)
    return curve[_ABL_COLS]


def _load_qfull_rows() -> pd.DataFrame:
    """Q_full rows reused verbatim from the committed Exp 12 summary (NOT recomputed)."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "experiment_12_summary.csv"))
    df = df[df["retriever"].isin(_RETR_MAP)].copy()
    df["retriever"] = df["retriever"].map(_RETR_MAP)
    df.insert(0, "query_level", "Q_full")
    return df[_ABL_COLS]


def plot_ablation_recall(ablation: pd.DataFrame, kstar: int, filename: str) -> None:
    """Fig 12.4 — Recall@K* and Recall@49, Q_full vs Q_organism, two retriever curves."""
    levels = ["Q_full", "Q_organism"]
    retrievers = ["Text-only", "Fusion w=2"]
    colors = {"Text-only": "steelblue", "Fusion w=2": "darkorange"}

    def val(level, rt, k):
        m = ablation[(ablation.query_level == level) & (ablation.retriever == rt) & (ablation.K == k)]
        return float(m["fraction_recall"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for ax, k in zip(axes, [kstar, 49]):
        x = np.arange(len(levels))
        for rt in retrievers:
            ys = [val(lv, rt, k) for lv in levels]
            ax.plot(x, ys, marker="o", markersize=7, linewidth=2, color=colors[rt], label=rt)
            for xi, y in zip(x, ys):
                ax.annotate(f"{y:.3f}", (xi, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.set_title(f"Fraction-recall @ K={k}", fontsize=11)
        ax.set_ylabel("Fraction-recall (same-species)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Figure 12.4 — Text-side recall: full-metadata vs organism-only query",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_ablation_curves(ablation: pd.DataFrame, filename: str) -> None:
    """Fig 12.5 — full Recall@K curves, small-multiples by query level."""
    levels = ["Q_full", "Q_organism"]
    retrievers = ["DNA", "Text-only", "Fusion w=2"]
    colors = {"DNA": "seagreen", "Text-only": "steelblue", "Fusion w=2": "darkorange"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, lv in zip(axes, levels):
        for rt in retrievers:
            sub = ablation[(ablation.query_level == lv) & (ablation.retriever == rt)].sort_values("K")
            ax.plot(sub["K"], sub["fraction_recall"], marker="o", markersize=4,
                    linewidth=1.8, color=colors[rt], label=rt)
        ax.set_title(lv, fontsize=11)
        ax.set_xlabel("K")
        ax.set_xticks(K_SWEEP)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Fraction-recall (same-species)")
    fig.suptitle("Figure 12.5 — Recall@K by query level (DNA unchanged; text/fusion re-queried)",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_encoder_compare(enc: pd.DataFrame, filename: str) -> None:
    """Fig 12.6 — Index B encoder (SapBERT vs MiniLM), full vs organism-only query."""
    levels = ["Q_full", "Q_organism"]
    encoders = ["SapBERT", "MiniLM-L6-v2"]
    colors = {"SapBERT": "darkorange", "MiniLM-L6-v2": "slategray"}

    def p1(e, lv):
        m = enc[(enc.encoder == e) & (enc.query_level == lv) & (enc.K == 1)]
        return float(m["P@1"].iloc[0])

    def r49(e, lv):
        m = enc[(enc.encoder == e) & (enc.query_level == lv) & (enc.K == 49)]
        return float(m["fraction_recall"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (title, fn) in zip(axes, [("P@1 (Text-only)", p1),
                                      ("Recall@49 (Text-only)", r49)]):
        x = np.arange(len(levels))
        for e in encoders:
            ys = [fn(e, lv) for lv in levels]
            ax.plot(x, ys, marker="o", markersize=7, linewidth=2, color=colors[e], label=e)
            for xi, y in zip(x, ys):
                ax.annotate(f"{y:.3f}", (xi, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Figure 12.6 — Index B encoder: SapBERT vs MiniLM, full vs organism-only",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def run_ablation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info("=== Experiment 12 follow-up ablation: organism-only text query ===")

    records, species_tags = select_sequences()
    labels = np.array([r["label"] for r in records])
    N = len(records)

    # ── Reuse the committed Exp 12 indices (no corpus re-embed) ────────────────
    idxA = faiss.read_index(os.path.join(OUTPUT_DIR, "experiment_12_index_dna.faiss"))
    idxB = faiss.read_index(os.path.join(OUTPUT_DIR, "experiment_12_index_text.faiss"))
    assert idxA.ntotal == N and idxB.ntotal == N, \
        f"index size mismatch: A={idxA.ntotal} B={idxB.ntotal} records={N}"

    # DNA ranks are unchanged across query levels: reconstruct the stored DNA
    # vectors and re-search Index A. No DNABERT re-embed.
    try:
        dnaA = idxA.reconstruct_n(0, N)
    except Exception:                                   # HNSW fallback via flat storage
        dnaA = faiss.downcast_index(idxA.storage).reconstruct_n(0, N)
    _, idsA = idxA.search(dnaA, MAX_K)
    dna_df = evaluate_retrieval(idsA, labels, "DNA", K_SWEEP)

    # ── Harness validation: recomputed DNA == committed Exp 12 A (DNA) ─────────
    committed = pd.read_csv(os.path.join(OUTPUT_DIR, "experiment_12_summary.csv"))
    comm_dna = committed[committed["retriever"] == "A (DNA)"]
    dna_curve, _ = curve_and_scalars(dna_df, "DNA")
    for k in K_SWEEP:
        got_fr = float(dna_curve[dna_curve.K == k]["fraction_recall"].iloc[0])
        got_hr = float(dna_curve[dna_curve.K == k]["hit_rate"].iloc[0])
        ref_fr = float(comm_dna[comm_dna.K == k]["fraction_recall"].iloc[0])
        ref_hr = float(comm_dna[comm_dna.K == k]["hit_rate"].iloc[0])
        assert abs(got_fr - ref_fr) < 1e-9 and abs(got_hr - ref_hr) < 1e-9, (
            f"HARNESS MISMATCH at K={k}: recomputed DNA "
            f"(FR={got_fr}, HR={got_hr}) != committed (FR={ref_fr}, HR={ref_hr}). "
            f"Saved indices or harness changed — aborting.")
    log.info("Harness OK: recomputed DNA curve is byte-identical to committed Exp 12 A (DNA).")

    # ── Q_organism: organism-only query (drop protein & all other fields) ──────
    # Faithful realistic query: species known (from accession), gene function not.
    q_org = [f"organism={r['organism']}" for r in records]
    log.info(f"Embedding {N} organism-only queries with {SELECTED_ENCODER['name']} ...")
    qorg_vecs = normalise(load_text_encoder(SELECTED_ENCODER)(q_org))
    _, idsB_org = idxB.search(qorg_vecs, MAX_K)        # asymmetric: thin query vs full corpus

    text_df  = evaluate_retrieval(idsB_org, labels, "Text-only", K_SWEEP)
    fused_df = evaluate_retrieval(rrf_fuse(idsA, idsB_org, w=2.0), labels, "Fusion w=2", K_SWEEP)

    # ── Assemble level × retriever × K table ───────────────────────────────────
    qfull = _load_qfull_rows()
    qorg = pd.concat([
        _rows_from_eval(dna_df,   "DNA",        "Q_organism"),
        _rows_from_eval(text_df,  "Text-only",  "Q_organism"),
        _rows_from_eval(fused_df, "Fusion w=2", "Q_organism"),
    ], ignore_index=True)
    ablation = pd.concat([qfull, qorg], ignore_index=True)
    ablation.to_csv(os.path.join(OUTPUT_DIR, "experiment_12_ablation_summary.csv"), index=False)

    # ── Print per-level/retriever metrics ───────────────────────────────────────
    for lv in ["Q_full", "Q_organism"]:
        print(f"\n{'=' * 60}\n{lv}\n{'=' * 60}")
        for rt in ["DNA", "Text-only", "Fusion w=2"]:
            s = ablation[(ablation.query_level == lv) & (ablation.retriever == rt)].sort_values("K")
            p1 = float(s["P@1"].iloc[0]); mrr = float(s["MRR"].iloc[0])
            r3 = float(s[s.K == ABLATION_KSTAR]["fraction_recall"].iloc[0])
            r49 = float(s[s.K == 49]["fraction_recall"].iloc[0])
            print(f"  {rt:11s}  P@1={p1:.3f}  MRR={mrr:.3f}  "
                  f"FR@{ABLATION_KSTAR}={r3:.4f}  FR@49={r49:.4f}")

    # ── Gates ────────────────────────────────────────────────────────────────────
    def fr(level, rt, k):
        m = ablation[(ablation.query_level == level) & (ablation.retriever == rt) & (ablation.K == k)]
        return float(m["fraction_recall"].iloc[0])

    gateA_kstar = fr("Q_full", "Text-only", ABLATION_KSTAR) - fr("Q_organism", "Text-only", ABLATION_KSTAR)
    gateA_49 = fr("Q_full", "Text-only", 49) - fr("Q_organism", "Text-only", 49)
    gateB = fr("Q_organism", "Fusion w=2", ABLATION_KSTAR) - fr("Q_organism", "Text-only", ABLATION_KSTAR)
    gateB_49 = fr("Q_organism", "Fusion w=2", 49) - fr("Q_organism", "Text-only", 49)

    print(f"\n{'=' * 60}\nGATES (K*={ABLATION_KSTAR})\n{'=' * 60}")
    print(f"  Gate A (protein-field contribution, Text-only drop Q_full→Q_organism):")
    print(f"    fraction-recall@K*: {gateA_kstar:+.4f}    @49: {gateA_49:+.4f}")
    print(f"  Gate B (fusion re-test under Q_organism, Fusion w=2 − Text-only):")
    print(f"    fraction-recall@K*: {gateB:+.4f} "
          f"→ {'fusion RECOMMENDED (>5pp)' if gateB > 0.05 else 'verdict holds (≤5pp): fusion optional'}"
          f"    @49: {gateB_49:+.4f}")

    # ── Figures ──────────────────────────────────────────────────────────────────
    plot_ablation_recall(ablation, ABLATION_KSTAR, "experiment_12_ablation_recall.png")
    plot_ablation_curves(ablation, "experiment_12_ablation_curves.png")

    # ── Encoder comparison: does SapBERT's Index-B win survive organism-only? ──
    # Reproducible MiniLM baseline (the prior Index B encoder). Build a fresh MiniLM
    # corpus index and query it with full metadata and organism-only, alongside the
    # SapBERT numbers above. Answers: is SapBERT still the better Index B encoder once
    # the query collapses to a bare species name? (Exp 11 only compared encoders with
    # full metadata; this re-tests in the realistic regime.)
    log.info("=== Encoder comparison: MiniLM vs SapBERT (full vs organism) ===")
    metas = [r["metadata"] for r in records]
    MINILM = {"name": "MiniLM-L6-v2", "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
              "dim": 384, "loader": "st", "prefix": ""}
    mini_embed = load_text_encoder(MINILM)
    mini_corp = normalise(mini_embed(metas))
    mini_idx = faiss.IndexFlatIP(mini_corp.shape[1])
    mini_idx.add(mini_corp)
    _, mini_full_ids = mini_idx.search(mini_corp, MAX_K)              # symmetric full-metadata
    _, mini_org_ids  = mini_idx.search(normalise(mini_embed(q_org)), MAX_K)
    mini_full_df = evaluate_retrieval(mini_full_ids, labels, "Text-only", K_SWEEP)
    mini_org_df  = evaluate_retrieval(mini_org_ids,  labels, "Text-only", K_SWEEP)

    def _enc_rows(df, encoder, level):
        c, s = curve_and_scalars(df, "Text-only")
        c["P@1"] = s["P@1"]; c["MRR"] = s["MRR"]
        c.insert(0, "encoder", encoder); c.insert(1, "query_level", level)
        return c[["encoder", "query_level", "K", "fraction_recall", "hit_rate", "P@1", "MRR"]]

    sap_full = ablation[(ablation.query_level == "Q_full") & (ablation.retriever == "Text-only")]
    sap_full = sap_full.assign(encoder="SapBERT")[["encoder", "query_level", "K",
                                                   "fraction_recall", "hit_rate", "P@1", "MRR"]]
    enc_cmp = pd.concat([
        sap_full,
        _enc_rows(text_df,      "SapBERT",      "Q_organism"),
        _enc_rows(mini_full_df, "MiniLM-L6-v2", "Q_full"),
        _enc_rows(mini_org_df,  "MiniLM-L6-v2", "Q_organism"),
    ], ignore_index=True)
    enc_cmp.to_csv(os.path.join(OUTPUT_DIR, "experiment_12_encoder_compare.csv"), index=False)
    plot_encoder_compare(enc_cmp, "experiment_12_ablation_encoder.png")

    def _ep1(e, lv):
        m = enc_cmp[(enc_cmp.encoder == e) & (enc_cmp.query_level == lv) & (enc_cmp.K == 1)]
        return float(m["P@1"].iloc[0])

    def _er49(e, lv):
        m = enc_cmp[(enc_cmp.encoder == e) & (enc_cmp.query_level == lv) & (enc_cmp.K == 49)]
        return float(m["fraction_recall"].iloc[0])

    print(f"\n{'=' * 60}\nENCODER COMPARISON — Index B, Text-only\n{'=' * 60}")
    for e in ["SapBERT", "MiniLM-L6-v2"]:
        print(f"  {e:13s}  P@1: full={_ep1(e, 'Q_full'):.3f} organism={_ep1(e, 'Q_organism'):.3f}"
              f"   R@49: full={_er49(e, 'Q_full'):.3f} organism={_er49(e, 'Q_organism'):.3f}")
    print(f"  SapBERT−MiniLM P@1 gap: full={_ep1('SapBERT','Q_full')-_ep1('MiniLM-L6-v2','Q_full'):+.3f}"
          f"  organism={_ep1('SapBERT','Q_organism')-_ep1('MiniLM-L6-v2','Q_organism'):+.3f}")
    log.info("Encoder comparison (Text-only):\n" + enc_cmp.to_string(index=False))

    log.info("ABLATION RESULTS (level × retriever × K):\n" + ablation.to_string(index=False))
    log.info(f"Gate A Text-only drop Q_full→Q_organism: FR@{ABLATION_KSTAR}={gateA_kstar:+.4f}, "
             f"FR@49={gateA_49:+.4f}. Gate B Fusion w=2 − Text-only @K*={gateB:+.4f} "
             f"({'RECOMMENDED' if gateB > 0.05 else 'optional/holds'}).")
    print(f"\nResults saved to: {os.path.join(OUTPUT_DIR, 'experiment_12_ablation_summary.csv')}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ablation":
        run_ablation()
    else:
        main()
