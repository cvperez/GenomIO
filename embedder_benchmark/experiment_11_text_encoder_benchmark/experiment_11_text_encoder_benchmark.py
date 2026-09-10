#!/usr/bin/env python3
"""
Experiment 11: Text-Encoder Benchmark for the Metadata Index (Index B)
======================================================================

Single-variable ablation. Swap ONLY the text encoder feeding Index B and
re-measure same-species retrieval on the deployed n=1,000 / 20-species corpus.
Everything else is held constant: corpus, same-species relevance definition,
query set (each record's metadata string — symmetric, same encoder on both
sides), the K sweep, and the FAISS index type (IndexFlatIP, so cosine in every
case). Index A (DNA) is NOT built here; this experiment is the text side only.

Why: Index B currently uses all-MiniLM-L6-v2 and already drives strong retrieval
(Text->DNA P@1 = 0.906 in Experiment 10). The open question — settled before the
fusion experiment (Exp 12) — is whether a stronger *open-weights* text embedder
raises metadata-side retrieval on short structured CDS metadata. Paid encoders
(OpenAI/GenePT, Cohere, Voyage, Gemini) are excluded by design; only free,
locally runnable models are tested.

Candidates (all open-weights, free, CPU-runnable):
  - all-MiniLM-L6-v2          384   baseline (current Index B encoder)
  - bge-small-en-v1.5         384   same-size drop-in upgrade
  - bge-large-en-v1.5        1024   larger general retriever (MIT, no remote code)
  - SapBERT-from-PubMedBERT   768   biomedical entity-name specialist (CLS pooling)
  - MedEmbed-base-v0.1        768   biomedical retrieval (BGE fine-tuned)
  - e5-base-v2 (optional)     768   prefix-paradigm general check ("query: " both sides)

Metrics (mean over 1,000 leave-self-out queries): P@1, MRR, Recall@K for
K in {1,3,5,10,15,20,30,40,49}.

Sanity gate: the MiniLM row must reproduce Experiment 10's Text->DNA numbers
(P@1 ~ 0.906, Recall@10 ~ 0.182, Recall@20 ~ 0.360). If it does not, the harness
has diverged from Exp 10 and the other rows cannot be trusted.

Run from repo root:
    python embedder_benchmark/experiment_11_text_encoder_benchmark/experiment_11_text_encoder_benchmark.py
"""
import glob
import logging
import os
import re
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

# Log to both console and experiment_11_run.log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "experiment_11_run.log"), mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Silence the very chatty HTTP / model-download loggers so run.log stays readable
for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers",
              "filelock", "transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

SEQS_PER_SPECIES = 50
SEQ_MIN_LEN      = 300
SEQ_MAX_LEN      = 900

K_SWEEP             = [1, 3, 5, 10, 15, 20, 30, 40, 49]
MAX_K               = 50      # 49 relevant + self -> 49 ranked candidates after leave-self-out
BASELINE_NAME       = "MiniLM-L6-v2"
BASELINE_P1         = 0.906   # Exp 10 Text->DNA P@1 (line to beat)
SELECTION_TOLERANCE = 0.02    # "keep MiniLM unless beaten by > this on P@1"

# Exp 10 Text->DNA reference (for the MiniLM sanity gate)
EXP10_TEXT_REF = {"P@1": 0.906, "Recall@10": 0.1822, "Recall@20": 0.3599, "MRR": 0.9377}

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

# Single variable: the text encoder feeding Index B. `loader` dispatches loading.
TEXT_ENCODER_REGISTRY = [
    {"name": "MiniLM-L6-v2", "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
     "dim": 384, "loader": "st", "prefix": "", "baseline": True},
    {"name": "bge-small-en-v1.5", "hf_id": "BAAI/bge-small-en-v1.5",
     "dim": 384, "loader": "st", "prefix": ""},
    {"name": "bge-large-en-v1.5", "hf_id": "BAAI/bge-large-en-v1.5",
     "dim": 1024, "loader": "st", "prefix": ""},
    {"name": "SapBERT", "hf_id": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
     "dim": 768, "loader": "transformers_cls", "prefix": ""},
    {"name": "MedEmbed-base", "hf_id": "abhinand/MedEmbed-base-v0.1",
     "dim": 768, "loader": "st", "prefix": ""},
    {"name": "e5-base-v2", "hf_id": "intfloat/e5-base-v2",
     "dim": 768, "loader": "st_prefix", "prefix": "query: ", "optional": True},
]


# ── Data loading (copied verbatim from experiment_10, incl. the .1-suffix fix) ──

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


# ── Encoder loading ─────────────────────────────────────────────────────────

def load_encoder(spec: dict):
    """Return embed(texts) -> np.ndarray[float32] (raw, un-normalised).

    Normalisation is owned by the harness (normalise()), applied once to the
    stacked matrix, so IndexFlatIP == cosine uniformly across encoders.
    """
    loader = spec["loader"]
    hf     = spec["hf_id"]
    prefix = spec.get("prefix", "")

    if loader in ("st", "st_prefix"):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(hf)

        def embed(texts: list[str]) -> np.ndarray:
            t = [prefix + x for x in texts] if prefix else texts
            out = []
            bs = 64
            for i in range(0, len(t), bs):
                out.append(model.encode(t[i:i + bs], normalize_embeddings=False,
                                        show_progress_bar=False, convert_to_numpy=True))
                if (i + bs) % 256 == 0 or i + bs >= len(t):
                    log.info(f"  {spec['name']}: {min(i + bs, len(t))}/{len(t)}")
            return np.vstack(out).astype(np.float32)

        return embed

    if loader == "transformers_cls":          # SapBERT -> CLS token, NOT mean pool
        from transformers import AutoModel, AutoTokenizer
        tok   = AutoTokenizer.from_pretrained(hf)
        model = AutoModel.from_pretrained(hf)
        model.eval()

        def embed(texts: list[str]) -> np.ndarray:
            out = []
            bs = 32
            with torch.no_grad():
                for i in range(0, len(texts), bs):
                    enc = tok(texts[i:i + bs], return_tensors="pt",
                              padding=True, truncation=True, max_length=64)
                    hs = model(**enc).last_hidden_state      # (B, T, 768)
                    out.append(hs[:, 0, :].cpu().numpy())     # CLS = token 0
                    if (i + bs) % 256 == 0 or i + bs >= len(texts):
                        log.info(f"  {spec['name']}: {min(i + bs, len(texts))}/{len(texts)}")
            return np.vstack(out).astype(np.float32)

        return embed

    raise ValueError(f"Unknown loader: {loader}")


# ── FAISS helpers (copied from experiment_10) ─────────────────────────────────

def normalise(v: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(v.astype(np.float32))
    faiss.normalize_L2(u)
    return u


def build_flat_index(vectors: np.ndarray) -> tuple:
    v = normalise(vectors)
    idx = faiss.IndexFlatIP(v.shape[1])
    t0 = time.perf_counter()
    idx.add(v)
    build_ms = (time.perf_counter() - t0) * 1000
    return idx, v, build_ms


def index_memory_bytes(index: faiss.Index) -> int:
    return len(faiss.serialize_index(index))


# ── Generalised retrieval evaluation (shared with Exp 12) ─────────────────────

def evaluate_retrieval(all_ids: np.ndarray, labels: np.ndarray,
                       query_type: str, k_values: list[int]) -> pd.DataFrame:
    """Leave-self-out retrieval metrics from a precomputed neighbour-id matrix.

    all_ids : (N, >=max(k)+1) int — neighbour ids per query (self may be present;
              filtered out by `x != i`). Works identically for a single FAISS
              index or a fused id-list, so Exp 12 reuses it unchanged.

    Per query (n_relevant > 0): P@k, Recall@k, Hit@k for each k, and RR (-> MRR).
    Returns one row per evaluable query.
    """
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


def summarise(df: pd.DataFrame, k_values: list[int]) -> dict:
    """Mean over queries -> {P@1, MRR, Recall@k for all k}."""
    out = {"P@1": float(df["P@1"].mean()), "MRR": float(df["RR"].mean())}
    for k in k_values:
        out[f"Recall@{k}"] = float(df[f"Recall@{k}"].mean())
    return out


# ── Per-encoder runner (graceful skip) ────────────────────────────────────────

def run_encoder(spec: dict, metas: list[str], labels: np.ndarray) -> dict | None:
    name = spec["name"]
    print(f"\n{'=' * 64}\n{name}  ({spec['hf_id']})\n{'=' * 64}")
    try:
        embed = load_encoder(spec)
        t0 = time.perf_counter()
        vecs = embed(metas)
        embed_s = time.perf_counter() - t0

        actual_dim = vecs.shape[1]
        if actual_dim != spec["dim"]:
            log.warning(f"  {name}: expected dim {spec['dim']}, got {actual_dim}")

        idx, norm_vecs, _ = build_flat_index(vecs)
        mem_kb = index_memory_bytes(idx) / 1024
        _, all_ids = idx.search(norm_vecs, MAX_K)

        df = evaluate_retrieval(all_ids, labels, name, K_SWEEP)
        agg = summarise(df, K_SWEEP)

        print(f"  n_queries: {len(df)}   embed: {embed_s:.1f}s   dim: {actual_dim}")
        print(f"  P@1: {agg['P@1']:.4f}   MRR: {agg['MRR']:.4f}")
        for k in K_SWEEP:
            print(f"  Recall@{k:2d}: {agg[f'Recall@{k}']:.4f}")

        row = {"encoder": name, "hf_id": spec["hf_id"], "dim": actual_dim,
               "status": "OK", "embed_time_s": round(embed_s, 1),
               "index_mem_kb": round(mem_kb, 0),
               "baseline": bool(spec.get("baseline", False)),
               "optional": bool(spec.get("optional", False)), **agg}
        return {"summary": row, "per_query": df}

    except Exception as e:                       # noqa: BLE001 — graceful skip
        log.warning(f"SKIPPING {name}: {type(e).__name__}: {e}")
        row = {"encoder": name, "hf_id": spec["hf_id"], "dim": spec["dim"],
               "status": "FAILED", "embed_time_s": np.nan, "index_mem_kb": np.nan,
               "baseline": bool(spec.get("baseline", False)),
               "optional": bool(spec.get("optional", False)),
               "P@1": np.nan, "MRR": np.nan,
               **{f"Recall@{k}": np.nan for k in K_SWEEP}}
        return {"summary": row, "per_query": None}


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_recall_curves(summary: pd.DataFrame, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    ok = summary[summary["status"] == "OK"]
    for i, (_, r) in enumerate(ok.iterrows()):
        vals = [r[f"Recall@{k}"] for k in K_SWEEP]
        style = "--" if r["baseline"] else "-"
        lw = 2.5 if r["baseline"] else 1.8
        ax.plot(K_SWEEP, vals, style, marker="o", markersize=4, linewidth=lw,
                color=cmap(i % 10), label=r["encoder"])
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K (same-species)")
    ax.set_title("Figure 11.1 — Recall@K by text encoder (Index B, n=1,000)", fontsize=11)
    ax.set_xticks(K_SWEEP)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_p1_mrr_bars(summary: pd.DataFrame, filename: str) -> None:
    ok = summary[summary["status"] == "OK"]
    x = np.arange(len(ok))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, ok["P@1"], width, label="P@1", color="steelblue", alpha=0.9)
    ax.bar(x + width / 2, ok["MRR"], width, label="MRR", color="darkorange", alpha=0.9)
    ax.axhline(BASELINE_P1, color="red", linestyle="--", linewidth=1,
               label=f"MiniLM P@1 baseline = {BASELINE_P1}")
    ax.set_xticks(x)
    ax.set_xticklabels(ok["encoder"], rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Figure 11.2 — P@1 and MRR by text encoder", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_recall10_vs_dim(summary: pd.DataFrame, filename: str) -> None:
    ok = summary[summary["status"] == "OK"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ok["dim"], ok["Recall@10"], s=80, color="steelblue", zorder=3)
    for _, r in ok.iterrows():
        ax.annotate(r["encoder"], (r["dim"], r["Recall@10"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Embedding dimension (index-cost proxy)")
    ax.set_ylabel("Recall@10 (same-species)")
    ax.set_title("Figure 11.3 — Recall@10 vs index cost", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


# ── Winner selection (decision gates) ─────────────────────────────────────────

def select_winner(summary: pd.DataFrame) -> dict:
    """argmax P@1 among successful, non-optional-unless-winning encoders;
    keep MiniLM unless beaten by more than SELECTION_TOLERANCE on P@1."""
    ok = summary[summary["status"] == "OK"].copy()
    base = ok[ok["encoder"] == BASELINE_NAME]
    base_p1 = float(base["P@1"].iloc[0]) if len(base) else BASELINE_P1

    top = ok.loc[ok["P@1"].idxmax()]
    if float(top["P@1"]) - base_p1 <= SELECTION_TOLERANCE:
        winner = base.iloc[0] if len(base) else top
        reason = (f"no candidate beats MiniLM by > {SELECTION_TOLERANCE} P@1 "
                  f"(best={top['encoder']} @ {top['P@1']:.4f} vs baseline {base_p1:.4f})")
    else:
        winner = top
        reason = (f"{top['encoder']} beats MiniLM by "
                  f"{float(top['P@1']) - base_p1:+.4f} P@1")
    return {"name": winner["encoder"], "hf_id": winner["hf_id"],
            "dim": int(winner["dim"]), "p1": float(winner["P@1"]),
            "mrr": float(winner["MRR"]), "reason": reason, "base_p1": base_p1}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=== Selecting sequences ===")
    records, species_tags = select_sequences()
    metas  = [r["metadata"] for r in records]
    labels = np.array([r["label"] for r in records])
    N = len(records)
    print(f"\n{N} records across {len(species_tags)} species "
          f"({SEQS_PER_SPECIES}/species). Query = metadata string (symmetric).")

    rows, per_query_dfs = [], []
    for spec in TEXT_ENCODER_REGISTRY:
        res = run_encoder(spec, metas, labels)
        rows.append(res["summary"])
        if res["per_query"] is not None:
            res["per_query"]["encoder"] = spec["name"]
            per_query_dfs.append(res["per_query"])

    summary = pd.DataFrame(rows)

    # ── Sanity gate: MiniLM must reproduce Exp 10 Text->DNA ──────────────────
    base = summary[summary["encoder"] == BASELINE_NAME]
    print(f"\n{'=' * 64}\nSANITY GATE — MiniLM vs Exp 10 Text->DNA\n{'=' * 64}")
    if len(base) and base["status"].iloc[0] == "OK":
        b = base.iloc[0]
        for m, ref in EXP10_TEXT_REF.items():
            got = float(b[m]) if m in b else float("nan")
            ok = abs(got - ref) <= 0.01
            print(f"  {m:10s}: got {got:.4f}  ref {ref:.4f}  "
                  f"{'OK' if ok else 'MISMATCH (>0.01)'}")
    else:
        print("  MiniLM row FAILED — cannot validate harness.")

    # ── Summary table ─────────────────────────────────────────────────────────
    show_cols = ["encoder", "dim", "status", "P@1", "MRR",
                 "Recall@10", "Recall@20", "Recall@49", "embed_time_s"]
    print(f"\n{'=' * 64}\nSUMMARY (sorted by P@1)\n{'=' * 64}")
    print(summary.sort_values("P@1", ascending=False)[show_cols].to_string(index=False))

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_recall_curves(summary, "experiment_11_recall_at_k.png")
    plot_p1_mrr_bars(summary, "experiment_11_p1_mrr_bars.png")
    plot_recall10_vs_dim(summary, "experiment_11_recall10_vs_dim.png")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    summary.to_csv(os.path.join(OUTPUT_DIR, "experiment_11_summary.csv"), index=False)
    if per_query_dfs:
        pd.concat(per_query_dfs, ignore_index=True).to_csv(
            os.path.join(OUTPUT_DIR, "experiment_11_results.csv"), index=False)

    # ── Winner + handoff block ────────────────────────────────────────────────
    winner = select_winner(summary)
    print(f"\n{'=' * 64}\nWINNER: {winner['name']}\n{'=' * 64}")
    print(f"  reason: {winner['reason']}")
    print("\n=== HANDOFF TO EXP 12 ===")
    print(f"Winner: {winner['name']} (dim={winner['dim']}, P@1={winner['p1']:.4f})")
    print("Paste into experiment_12_recall_fusion.py:")
    # Recover the loader recipe from the registry
    wspec = next(s for s in TEXT_ENCODER_REGISTRY if s["name"] == winner["name"])
    print(f'SELECTED_ENCODER = {{"name": "{wspec["name"]}", '
          f'"hf_id": "{wspec["hf_id"]}", "dim": {winner["dim"]}, '
          f'"loader": "{wspec["loader"]}", "prefix": "{wspec.get("prefix", "")}"}}')

    print(f"\nResults saved to: {os.path.join(OUTPUT_DIR, 'experiment_11_summary.csv')}")

    # Mirror the key results into run.log (the FileHandler only sees log.* calls)
    log.info("RESULTS (sorted by P@1):\n" +
             summary.sort_values("P@1", ascending=False)[show_cols].to_string(index=False))
    log.info(f"WINNER: {winner['name']} (dim={winner['dim']}, P@1={winner['p1']:.4f}) "
             f"— {winner['reason']}")


if __name__ == "__main__":
    main()
