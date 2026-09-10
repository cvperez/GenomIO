# Experiment 11: Text-Encoder Benchmark for the Metadata Index (Index B)

The current reproduction notes below describe the checked-in script. The historical record retains the original measurements; raw artifacts mentioned there are not included.

## Current Reproduction Guide

### Objective

Compare six text encoders for same-organism retrieval while retaining the metadata-side evaluation procedure.

### Experimental Setup

The script uses 1,000 records in the reported setup (50 per corpus file, 300–900 bp). Queries and indexed documents use each record's metadata; the query itself is excluded. MiniLM, bge-small, bge-large, SapBERT, MedEmbed-base, and e5-base-v2 are compared with normalized `IndexFlatIP` search. SapBERT uses CLS pooling; e5 uses the `query: ` prefix on both sides. K is swept over 1,3,5,10,15,20,30,40,49. No DNA index or generator is run.

### Inputs

The checked-in [CDS corpus](../../../rag_corpus_uniform) supplies the FASTA records and headers. Selection and model settings are defined in the script, not in the root YAML configuration. The script also reads [organisms.tsv](../../../rag_corpus_uniform/organisms.tsv).

### How to Run

From the repository root, with the [benchmark dependencies](../../README.md#reproduction) installed:

```bash
python3 embedder_benchmark/experiment_11_text_encoder_benchmark/experiment_11_text_encoder_benchmark.py
```

These commands load model weights and may take substantial time. They were not executed for the repository cleanup. Outputs go to the `results/` directory beside the script; rerunning can overwrite generated artifacts.

### Results

The checked-in [historical result report](#historical-result-record) contains the recorded measurements. The raw CSVs, logs, plots, and binary artifacts described below are not included in this checkout.

`experiment_11_summary.csv` contains encoder metrics across k; `experiment_11_results.csv` contains query-level measurements. Recall and encoder-comparison plots plus `experiment_11_run.log` are generated.

### Main Findings

The stored table gives e5-base-v2 the highest P@1 (0.978) and SapBERT the highest Recall@49 (0.893). Experiment 12 uses SapBERT. Because the queries contain organism and protein metadata, these scores do not represent reconstruction from DNA flanks alone.

These findings summarize stored evidence, not a newly repeated experiment. Consult the [evidence limits](../../README.md#reproducibility-and-evidence-limits) before comparing historical numbers with a new run.

## Historical Result Record

> Historical evidence: numerical tables are retained. Raw CSVs, logs, and figures mentioned below are not included in this checkout. Earlier organism names, model explanations, and deployment recommendations may not describe the current implementation. Use the current reproduction guide above and [evidence limits](../../README.md#reproducibility-and-evidence-limits) for the current scope and reproduction constraints.

**Date**: 2026-06-03
**Hardware**: CPU-only; Python 3.10; torch 2.11.0; transformers 5.5.4;
sentence-transformers 5.4.1; faiss 1.13.2
**Script**: `embedder_benchmark/experiment_11_text_encoder_benchmark/experiment_11_text_encoder_benchmark.py`
**Corpus**: `rag_corpus_uniform/` : 1,000 CDS records (50/species × 20 species)
**Depends on**: Experiment 10 (deployed corpus, Index B harness, same-species relevance)
**Feeds**: Experiment 12 (Recall@K curves + fusion), which builds Index B from the
encoder selected here.

---

## 1. Objective

Index B (the metadata-text index) shipped in Experiment 10 with `all-MiniLM-L6-v2`
(384-dim), which already drives strong text retrieval (Text→DNA P@1 = 0.906). The open
question (which must be settled *before* the fusion experiment, because a stronger
Index B changes what fusion has to beat) is whether a stronger **open-weights** text
embedder raises metadata-side retrieval on short, structured CDS metadata.

This is a clean single-variable ablation. Only the text encoder feeding Index B changes;
the corpus, the same-species relevance definition, the query set (each record's metadata
string, searched symmetrically : same encoder on both sides), the K sweep, and the FAISS
index type (`IndexFlatIP`, so cosine in every case) are all held constant. Index A (DNA)
is not built here.

Paid/closed encoders (OpenAI/GenePT, Cohere, Voyage, Gemini) are **excluded by design**:
the GenePT-style description-embedding idea motivated this experiment, but GenePT relies
on a paid OpenAI model and is not reproducible for a thesis. Every candidate here is
open-weights, free, and runs locally on CPU.

## 2. Setup

| Parameter | Value |
|---|---|
| Records | 1,000 (50 per species) |
| Species | 20 bacterial genomes |
| Query | each record's metadata string (organism, gene, locus_tag, protein, …) |
| Matching | symmetric (metadata vs metadata), leave-self-out |
| Relevant set | other records of the same species (up to 49) |
| Index type | `IndexFlatIP` (L2-normalised → exact cosine) for **every** encoder |
| K sweep | {1, 3, 5, 10, 15, 20, 30, 40, 49} |
| Metrics | P@1, MRR, Recall@K (averaged over 1,000 queries) |

## 3. Candidate encoders

All open-weights, free, locally runnable. The single variable across runs.

| Encoder | HF id | Dim | Pooling | Prefix | Role |
|---|---|---|---|---|---|
| MiniLM-L6-v2 (baseline) | sentence-transformers/all-MiniLM-L6-v2 | 384 | mean | none | current Index B |
| bge-small-en-v1.5 | BAAI/bge-small-en-v1.5 | 384 | ST default | none | same-size drop-in |
| bge-large-en-v1.5 | BAAI/bge-large-en-v1.5 | 1024 | ST default | none | larger general (MIT) |
| SapBERT | cambridgeltl/SapBERT-from-PubMedBERT-fulltext | 768 | **CLS** | none | biomedical entity-name specialist |
| MedEmbed-base | abhinand/MedEmbed-base-v0.1 | 768 | ST default | none | biomedical retrieval (BGE fine-tuned) |
| e5-base-v2 | intfloat/e5-base-v2 | 768 | mean | `query: ` (both sides) | prefix-paradigm general check |

SapBERT is the one correctness-sensitive case: its metric-learning objective places the
sentence representation in the **[CLS]** token, so it is pooled as `last_hidden_state[:,0,:]`,
not mean-pooled. e5 requires the `query: ` prefix, applied identically to both sides since
matching is symmetric.

## 4. Pipeline

```
metadata string ──► encoder (variable) ──► L2-normalise ──► IndexFlatIP
                                                                  │
                                          index.search(all vecs, k=50)
                                                                  │
                                       leave-self-out ranking vs same-species labels
                                                                  │
                                          P@1 · MRR · Recall@K (mean over 1,000)
```

`MAX_K = 50` is requested so that, after dropping the query's own row, exactly 49 ranked
candidates remain : making Recall@49 well-posed for a 49-relevant ground truth.

## 5. Sanity gate (harness fidelity)

The MiniLM row must reproduce Experiment 10's Text→DNA numbers; otherwise the harness has
diverged and no other row can be trusted. It reproduces them **to four decimals**:

| Metric | Exp 10 ref | Exp 11 MiniLM | |
|---|---|---|---|
| P@1 | 0.9060 | 0.9060 | ✅ |
| MRR | 0.9377 | 0.9377 | ✅ |
| Recall@10 | 0.1822 | 0.1822 | ✅ |
| Recall@20 | 0.3599 | 0.3599 | ✅ |

## 6. Results

Sorted by P@1. Recall shown at representative K; the full curve is in
`experiment_11_summary.csv`.

| Encoder | dim | P@1 | MRR | R@5 | R@10 | R@20 | R@30 | R@49 | embed (s) |
|---|---|---|---|---|---|---|---|---|---|
| **e5-base-v2** | 768 | **0.978** | **0.988** | 0.100 | 0.200 | 0.397 | 0.574 | 0.859 | 60 |
| **SapBERT** | 768 | 0.974 | 0.985 | **0.100** | **0.201** | **0.402** | **0.594** | **0.893** | 49 |
| MiniLM-L6-v2 (baseline) | 384 | 0.906 | 0.938 | 0.091 | 0.182 | 0.360 | 0.520 | 0.764 | 13 |
| bge-large-en-v1.5 | 1024 | 0.900 | 0.935 | 0.092 | 0.185 | 0.367 | 0.525 | 0.761 | 185 |
| bge-small-en-v1.5 | 384 | 0.901 | 0.934 | 0.091 | 0.181 | 0.356 | 0.505 | 0.733 | 22 |
| MedEmbed-base | 768 | 0.880 | 0.920 | 0.091 | 0.183 | 0.368 | 0.520 | 0.728 | 59 |

The retrieval metrics (P@1, MRR, Recall@K) are deterministic and reproduce exactly across
runs. The `embed (s)` column is the one-off CPU index-build cost and is **indicative only**:
it varies with machine load (the values above are from an uncontended run; the committed
`experiment_11_summary.csv` was produced under parallel load and shows larger absolutes). The
*ordering* is stable: MiniLM and bge-small are fastest, bge-large is by far the slowest.

### Figures
- **Figure 11.1** `experiment_11_recall_at_k.png`: Recall@K vs K, one curve per encoder.
  SapBERT and e5 sit clearly above the MiniLM/bge/MedEmbed band at every K; the gap
  *widens* with K (SapBERT pulls ahead of e5 only in the deep-recall tail, K ≥ 30).
- **Figure 11.2** `experiment_11_p1_mrr_bars.png`: P@1 and MRR per encoder, with the
  MiniLM P@1 = 0.906 baseline line. Only SapBERT and e5 clear it.
- **Figure 11.3** `experiment_11_recall10_vs_dim.png`: Recall@10 vs embedding dimension.
  Shows that dimension/size does **not** predict quality: bge-large (1024-dim) ties the
  384-dim MiniLM, while the 768-dim SapBERT/e5 win.

## 7. Analysis: decision gates

**Gate 1 (free same-size drop-in (bge-small).** *Not cleared.* bge-small-en-v1.5
(384-dim, no pipeline change) lands at P@1 0.901 / MRR 0.934) within noise of MiniLM and
fractionally *below* it on every recall point past K=5. There is no free win here; bge-small
does not justify replacing MiniLM.

**Gate 2 : larger general model (bge-large).** *Not cleared.* The 1024-dim bge-large
matches MiniLM almost exactly (P@1 0.900, Recall@20 0.367 vs 0.360) at roughly **an
order of magnitude more embedding cost** (≈185 s vs ≈13 s uncontended) and a 2.7× larger
index. Scaling a general retriever buys nothing on this metadata. This is the clearest
negative result of the experiment.

**Gate 3 : domain specialists (SapBERT, MedEmbed).** *Split, and the informative result.*
The two biomedical models disagree sharply:
- **SapBERT wins decisively** : P@1 0.974 (+6.8 pp over MiniLM), and it leads *every* recall
  point, most strongly in the tail (Recall@49 0.893 vs 0.764, +12.9 pp). Its entity-name
  contrastive pretraining is well matched to CDS metadata, which is dominated by organism
  and protein names.
- **MedEmbed is the *worst* encoder tested** : P@1 0.880, below the MiniLM baseline.
  Fine-tuning BGE on clinical retrieval data did not transfer to bacterial CDS metadata.

So "biomedical" is not a property that predicts success: one biomedical model is best, the
other is worst. What separates them is the pretraining task (entity-name alignment vs
clinical passage retrieval), not the domain label.

**The prefix-paradigm general model (e5)** is the top P@1 scorer (0.978), edging SapBERT by
0.4 pp at rank 1 but trailing it in the recall tail. e5 and SapBERT are effectively
co-winners that trade off: e5 marginally better at rank 1, SapBERT better at recovering the
full species cluster.

### What this contradicts
The pre-registered expectation (from the mixed domain-vs-general literature) was a *modest*
gain from a general upgrade and a *smaller-than-hoped* gain from biomedical specialists. The
data inverts both halves: the general upgrades (bge-small/large) gave **no** gain, and a
biomedical specialist (SapBERT) gave the **largest** recall gain. MTEB-style general
leaderboard rank did not predict rank on this genomic-metadata task: consistent with the
finding that domain rank must be measured directly, not inferred from general benchmarks.

## 8. Conclusion

| Question | Answer |
|---|---|
| Free same-size drop-in (bge-small)? | **No**: ties MiniLM, no reason to switch. |
| Larger general (bge-large)? | **No**: matches MiniLM at 14× cost. Rejected. |
| Domain specialist? | **Yes, but model-specific**: SapBERT wins big; MedEmbed loses. |
| Best overall | e5-base-v2 (P@1) ≈ SapBERT (recall); both +~7 pp P@1 over MiniLM. |

> **A stronger open text encoder does materially help Index B, but not via the obvious
> route.** General-purpose upgrades (bge-small/large) do not beat the MiniLM baseline; the
> gains come from SapBERT (biomedical entity-name pretraining) and e5 (prefix paradigm),
> which are statistically tied at rank-1 and beat MiniLM by ~7 pp P@1.
>
> **Handoff to Experiment 12: SapBERT.** Although e5 edges P@1 by 0.4 pp, Experiment 12 is
> built around Recall@K, and SapBERT leads on exactly those metrics (Recall@20 0.402,
> Recall@49 0.893) while sitting within noise of e5 on P@1. SapBERT (768-dim, CLS-pooled)
> becomes the Index B encoder carried into the fusion experiment. The paid GenePT route is
> documented as considered and excluded.

### Reproduce
```bash
python embedder_benchmark/experiment_11_text_encoder_benchmark/experiment_11_text_encoder_benchmark.py
```
Outputs: `experiment_11_summary.csv` (per-encoder), `experiment_11_results.csv`
(per-query), the three figures above, and `experiment_11_run.log`.
