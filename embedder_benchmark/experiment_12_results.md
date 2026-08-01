# Experiment 12 — Recall@K Curves and the Hybrid Fusion Decision

**Date**: 2026-06-05
**Hardware**: CPU-only; Python 3.10; torch 2.11.0; transformers 5.5.4; faiss 1.13.2
**Script**: `experiments/experiment_12_recall_fusion.py`
**Corpus**: `rag_corpus_uniform/` — 1,000 CDS records (50/species × 20 species)
**Depends on**: Experiment 10 (deployed Index A + Index B) and Experiment 11 (selected
Index B text encoder — **SapBERT**, CLS-pooled, 768-dim).
**Scope**: retrieval only. We score ranked id-lists against same-species relevance. The
generator is not run, no gap is filled, no assembled sequence is scored — that is the
separate, downstream level-2 evaluation, explicitly out of scope here.

---

## 1. Objective

Experiment 10 ranked query modes by P@1 (DNA→DNA 0.655, Text→DNA 0.906) and reported only
two Recall points (Recall@10, Recall@20). It never produced the full Recall@K curve and
never tested whether the two indices should be **fused**. This experiment closes both:

1. **The Recall@K curves** for Index A (DNA), Index B (text), and their fusion, across
   K ∈ {1,3,5,10,15,20,30,40,49}, under two relevance lenses.
2. **The fusion decision** — reciprocal rank fusion (RRF) across A and B, which the bridge
   architecture already supports at zero architectural cost (both indices return ranked
   ids over the same row space).

Index B is built from the encoder selected in Experiment 11. e5-base-v2 edged SapBERT on
P@1 by 0.4 pp, but this experiment is built around Recall@K and SapBERT leads there
(Recall@49 0.893 vs 0.859); SapBERT was therefore carried forward. As a consistency check,
the B (Text) row below reproduces Experiment 11's SapBERT numbers exactly (P@1 0.974,
Recall@49 0.893).

## 2. Setup

| Parameter | Value |
|---|---|
| Records | 1,000 (50 per species, 20 species) |
| Index A | DNABERT-S DNA, `IndexHNSWFlat(M=32, efConstruction=200, efSearch=64)`, 768-dim |
| Index B | SapBERT metadata text (CLS-pooled), `IndexFlatIP`, 768-dim |
| Relevance | same-species (leave-self-out), up to 49 relevant per query |
| K sweep | {1, 3, 5, 10, 15, 20, 30, 40, 49}; `MAX_K=50` so Recall@49 is well-posed |
| RRF | `RRF(d) = 1/(c+rankA) + w·1/(c+rankB)`, c=60, w ∈ {1, 2} |
| Lenses | fraction-recall `|rel∩topK|/|rel|` (primary); hit-rate `1[≥1 rel in topK]` |

**Two relevance lenses, both reported.** Fraction-recall asks *what fraction of the species
cluster is surfaced* (continuous with Exp 6/10/11). Hit-rate asks *whether at least one
same-species record reaches the top-K* — the RAG-facing question, since for gap-filling a
single strong relevant record in the context is often what matters. The two can diverge
sharply, and here they do.

## 3. The double bridge — how fusion is applied in gap-filling

A gap produces **two different queries from the same event**, each searched in its own
index. Because both indices share a row space (the bridge, see `bridge_architecture.md`
§10, which names RRF across the two indices as the intended extension), every returned id
resolves to the same CDS record.

```
              Gap between two contigs
                 /                \
   DNA query: left+right flanks    text query: organism (+ flank function)
        |                                 |
   DNABERT-S, Index A (HNSW)         SapBERT, Index B (Flat)
        |                                 |
   ranked list A (by DNA)           ranked list B (by text)
                 \                /
                  RRF merge (text up-weighted, w≈2)
                          |
            top fused CDS, fetched via the bridge → context for the gap-filling model
```

The two queries are distinct objects, not the same string searched twice. RRF combines
**ranks**, not raw DNA/text cosines (which live on different scales), so the two encoders
stay fully decoupled even under fusion — c=60 needs no tuning.

## 4. Index statistics

| Index | Type | Build | Memory | Embed time (indicative) |
|---|---|---|---|---|
| A (DNA, 768) | IndexHNSWFlat(M=32) | 122 ms | 3,265 KB | ~1.5–6 min (CPU, DNABERT-S) |
| B (Text, 768) | IndexFlatIP | 82 ms | 3,000 KB | ~1 min (CPU, SapBERT) |

Embed time is a one-off, CPU- and load-dependent index-build cost (DNABERT-S over 1,000
sequences ranges from ~96 s uncontended, per Exp 10, to several minutes under parallel load);
the retrieval metrics below are deterministic and load-independent.

## 5. Results

### 5.1 Fraction-recall (primary lens)

| Retriever | P@1 | MRR | R@5 | R@10 | R@20 | R@30 | R@40 | R@49 |
|---|---|---|---|---|---|---|---|---|
| A (DNA) | 0.655 | 0.767 | 0.063 | 0.117 | 0.216 | 0.302 | 0.376 | 0.428 |
| B (Text, SapBERT) | 0.974 | 0.985 | 0.100 | 0.201 | 0.402 | 0.594 | 0.774 | **0.893** |
| A(+)B (w=1) | 0.989 | 0.994 | 0.100 | 0.193 | 0.358 | 0.496 | 0.619 | 0.714 |
| **A(+)B (w=2)** | **0.996** | **0.998** | 0.101 | 0.202 | 0.401 | 0.596 | 0.777 | **0.893** |

### 5.2 Hit-rate (RAG-facing lens)

| Retriever | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | HR@49 |
|---|---|---|---|---|---|---|
| A (DNA) | 0.655 | 0.855 | 0.909 | 0.958 | 0.985 | 0.996 |
| B (Text, SapBERT) | 0.974 | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 |
| A(+)B (w=1) | 0.989 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **A(+)B (w=2)** | **0.996** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### 5.3 Figures
- **Figure 12.1** `experiment_12_fraction_recall.png` — fraction-recall vs K. Text dominates
  DNA at every K; the gap *widens* with K (DNA never converges toward text).
- **Figure 12.2** `experiment_12_hitrate.png` — hit-rate vs K. Text saturates to 1.0 by
  K=5; DNA reaches ~0.96 by K=10. The K*=3 line marks the operating knee.
- **Figure 12.3** `experiment_12_fusion_delta.png` — fusion minus best single. Shows the
  w=1 vs w=2 split (below) at a glance.

## 6. The two faces of the result

**Hit-rate is essentially solved by text alone.** SapBERT retrieves ≥1 same-species record
in the top-1 for 97.4% of queries and in the top-3 for 99.6%; from K=5 it never misses. DNA
alone is weaker at rank 1 (65.5%) but catches up to 95.8% by K=10. "Is there a relevant
record in the context?" is a solved problem at K≥3.

**Fraction-recall is *not* solved — and DNA is the limiter.** Recovering the *full*
species cluster is much harder: DNA reaches only 0.428 of the cluster even at K=49, against
text's 0.893, and the gap grows with K rather than closing. The conclusion is structural:
the HNSW index is 100% accurate vs brute-force (Exp 10), so the DNA shortfall is the
**encoder**, not the index. Same-species CDS records span diverse gene families that
DNABERT-S does not pull tightly together.

## 7. Worked example — fusion correcting a DNA species error (real, from this run)

Query record 2, *Lactobacillus* sp. 001819145, metadata
`organism=Lactobacillus sp. 001819145 … protein=hypothetical protein`. The corpus contains
seven closely-related *Lactobacillus* species, which are genuinely hard to separate by raw
sequence — so this is exactly the case where DNA errs and text rescues.

| List | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
|---|---|---|---|---|---|
| **A (DNA)** | id 53 — *L.* sp. **013426185** (wrong) | id 221 — *L.* sp. **016699145** (wrong) | id 1 (same-sp) | id 46 (same-sp) | id 37 (same-sp) |
| **B (Text)** | id 1 (same-sp) | id 7 (same-sp) | id 45 (same-sp) | id 6 (same-sp) | id 10 (same-sp) |
| **Fused (w=1)** | id 1 | id 38 | id 46 | id 45 | id 17 — all same-sp |
| **Fused (w=2)** | id 1 | id 45 | id 38 | id 46 | id 17 — all same-sp |

DNA-alone would feed two **wrong-species** *Lactobacillus* records (ids 53, 221) as the top
context. Neither appears in text's list at all (B-rank "—"), so after fusion they fall out
of the top-5 entirely — the text ranker vetoes the DNA ranker's species errors. Note the
protein here is just "hypothetical protein", so text discriminates purely by the **organism
name**; even a thin, organism-only text query is enough to suppress cross-species DNA noise.

## 8. Weighting and its tradeoff

The text retriever is far stronger (P@1 0.974 vs 0.655), so the text term is up-weighted.
The weight is a real dial, not a free win — the two weights behave very differently on the
deep tail:

| | P@1 | Recall@49 |
|---|---|---|
| Text alone | 0.974 | 0.893 |
| Fused w=1 (equal) | 0.989 | **0.714** ← worse than text |
| Fused w=2 (text-weighted) | 0.996 | 0.893 ← matches text |

**Equal-weight (w=1) RRF actively degrades deep recall** (0.893 → 0.714): the weak DNA
ranking dilutes text's strong tail, dragging in DNA's cross-species neighbours at higher K.
**Text-weighted (w=2) RRF recovers text's full recall** (identical 0.893, and fractionally
ahead at K=40: 0.777 vs 0.774) *while* lifting rank-1. So if fusion is used at all, it must
be text-weighted; naive equal-weight fusion would have quietly hurt the system. A confidence
gate still wraps the step: if both lists' top cosines are weak (no genuinely similar corpus
record exists), pass no context rather than a confident guess on a distant match.

## 9. Operating K\*

K\* = **3**, the knee of the hit-rate curve (smallest K reaching 99% of the K=49 plateau).
At K=3, text retrieves ≥1 same-species record for 99.6% of queries. K\* is a retrieval-side
proxy read from the curves; whether 3 is also the right K to feed a generator is a question
for the separate generation evaluation, which is not run here.

## 10. Decision gates

**Gate 1 — fusion. Tested; not mandated by the primary lens.** At K\*=3 the fraction-recall
gain of the best fused retriever over the best single index is **+0.001** (0.0610 vs
0.0599) — far below the 5 pp bar. Fusion does, however, give a small, consistent **rank-1**
improvement: P@1 0.974 → 0.996 and hit-rate@1 0.974 → 0.996 (both +2.2 pp), at near-zero
cost since the bridge already returns both ranked lists. Honest verdict: **fusion is
optional, not required.** If implemented, it must be **w=2** (w=1 degrades recall, §8); the
benefit is a modest rank-1 lift, not better cluster recall. The recommended primary path
remains single-mode **Text→DNA (SapBERT)**.

**Gate 2 — DNA sufficiency. DNA is the limiting component.** DNA fraction-recall stays far
below text at K\* (0.039 vs 0.060) and at every K, and the gap widens rather than converging.
Because the HNSW index is loss-free (Exp 10), the next DNA-side experiment should target the
**encoder** — fine-tune DNABERT-S on this corpus, or revisit chunking — not the index.

**Gate 3 — bottleneck attribution. Retrieval is not the ceiling.** Best hit-rate@K\*=3 is
**1.000**: a relevant record is essentially always retrieved into a small top-K. The natural
next experiment is therefore the level-2 **generation** evaluation in the Gan et al.
framework — running the generator and scoring assembled sequences — which is out of scope here.

## 11. Limitations

- **Text numbers are an upper bound for the gap-filling use case.** In this benchmark the
  text query is the target record's own metadata, which *includes the organism name* — so
  species identity partly leaks into the query. At inference the text query is constructed
  from the draft assembly's organism (+ optional flank-gene function), which is thinner. The
  fusion *mechanism* (text correcting DNA species errors, §7) holds regardless, but absolute
  Text→DNA recall would be lower with a realistic thinner query.
- **Generation is not evaluated.** All metrics here are properties of the retrieved id-set
  alone (Gate 3 hands off to the generation evaluation).
- **Single-species-per-record relevance.** Genus/family-level similarity is not modelled;
  the seven *Lactobacillus* species are treated as fully distinct, which makes the
  same-species lens conservative for closely-related genomes.

## 12. Conclusion

| Question | Answer |
|---|---|
| Full Recall@K curve? | Produced (Fig 12.1/12.2). Text dominates DNA at every K. |
| Does DNA close the gap at high K? | **No** — DNA is the limiting encoder (Gate 2). |
| Implement RRF fusion? | **Optional, not required.** +0.001 fraction-recall at K\*; +2.2 pp rank-1. If used, **w=2 only**. |
| Operating K\* | **3** (hit-rate knee); a relevant record is essentially always present by K=3. |
| Where next? | Hit-rate is solved → retrieval is not the ceiling → **generation evaluation** (Gate 3). |

> **Recommendation: keep Text→DNA (SapBERT) as the primary single-index mode.** RRF fusion
> is a defensible, low-cost add-on that buys a small rank-1 improvement (P@1 0.974 → 0.996)
> but no better cluster recall, and only when text-weighted; equal-weight fusion is harmful.
> The DNA side is encoder-limited and is the right target for future retrieval work, but the
> more impactful next step is the downstream generation evaluation, since retrieval already
> places a relevant record in the top-3 for ~99.6% of queries.

## 13. Follow-up ablation — organism-only text query (realistic inference regime)

### Why
§1–§12 use a **full-metadata** text query (`organism=… protein=… …`), so the headline
Text→DNA numbers are an **upper bound**: they include a protein field that will not exist at
inference. The legacy/production retrieval paths never issue a text query at all — they query
with DNA flanks only (`src/core/gap_filler_rag.py:209` `query = left_seq + right_seq`;
`src/rag/retriever.py` regex-extracts `[ACGTN-]+`); Text→DNA is a **new bridge-side construct**.
At real inference the text query collapses to the **assembly organism alone** (the species is
derivable from the contig accession; the missing gene's function is unknowable). This ablation
adds exactly one realistic query level and re-tests the Gate-1 fusion verdict in that regime.

### What we test
Two query levels on the same 1,000-record corpus, same same-species/leave-self-out relevance,
the same saved indices (no corpus re-embed, no DNABERT re-run, **no generation**):
- **Q_full** — exact target metadata (organism + protein). Upper bound. **Reused verbatim** from
  the committed `experiment_12_summary.csv`; the full-metadata query is *not* re-run.
- **Q_organism** — keep only the organism field, drop the protein (and all other) fields, taken
  from each record's own corpus metadata (`organism=… protein=… → organism=…`). Models "species
  known, missing-gene function unknown." SapBERT-embedded fresh and searched against the saved
  full-metadata Index B (realistic *asymmetric* case: thin query vs richly-annotated corpus).

Q_organism keeps the true species (organism field) and removes only the protein — the one signal
that genuinely disappears at inference — so it measures the protein field's contribution *on top
of* the organism, not species identity. Retrievers: **Text-only** (Index B) and **Fusion w=2**
(RRF, c=60). **DNA-only** is reported but is identical across levels (the DNA query never changes).

*Harness validation (validation-first):* the DNA-only curve recomputed from the reused Index A is
**byte-identical** to the committed §5 `A (DNA)` rows — proving indices and harness are intact
without re-running the full-metadata text query.

### What we measure
P@1, MRR, Recall@K over the same sweep {1,3,5,10,15,20,30,40,49}, both lenses, per level ×
retriever. Primary readout: fraction-recall at K\*=3 (carried from §9) and at K=49.

### Figures
- **Figure 12.4** `experiment_12_ablation_recall.png` — fraction-recall at K\*=3 and K=49,
  Q_full vs Q_organism, Text-only and Fusion w=2. The key plot.
- **Figure 12.5** `experiment_12_ablation_curves.png` — full Recall@K curves, small-multiples by
  query level (DNA unchanged; text/fusion re-queried).
- **Figure 12.6** `experiment_12_ablation_encoder.png` — Index B encoder check: SapBERT vs MiniLM,
  full vs organism-only (P@1 and Recall@49).

### Decision gates
**Gate A — protein-field contribution (Text-only, Q_full → Q_organism).** Dropping the protein
field costs text substantially: **P@1 0.974 → 0.800 (−17.4 pp)**, MRR 0.985 → 0.831, and
**Recall@49 0.893 → 0.739 (−15.4 pp)**; the K\*=3 drop is small in absolute terms (0.060 → 0.050,
−0.99 pp) only because fraction-recall at K=3 is tiny. So the protein field was a real
contributor, especially at rank-1 and in the deep tail. Even so, organism-only text remains the
**strongest single retriever** — well above DNA (P@1 0.800 vs 0.655; Recall@49 0.739 vs 0.428).

**Gate B — fusion re-test under Q_organism (Fusion w=2 − Text-only).** On the pre-registered
metric (fraction-recall at K\*, 5 pp bar, kept identical to Gate 1 for consistency) the delta is
**+0.4 pp** and at K=49 it is **0.0** (both 0.739) — so the **"fusion optional" verdict holds**,
and DNA still cannot extend text's recall tail. *Caveat recorded inline:* at K\*=3 absolute
fraction-recall is only ~0.05, so a 5 pp **absolute** bar is very stringent there; the
operationally relevant top-of-list tells a different story. Fusion's **rank-1** value roughly
**quintuples** in the realistic regime: Fusion w=2's advantage over Text-only grows from
**+2.2 pp P@1** (Q_full) to **+10.7 pp** (Q_organism: 0.800 → 0.907), MRR +9.2 pp, and hit-rate@3
from +0.4 pp to **+8.8 pp** (0.849 → 0.937). As text weakens, DNA's corroboration stops being
redundant at the head of the list.

### Outcome
Both predicted outcomes partially materialised, and honestly so. Organism-only text degrades but
stays strong, which **keeps Text→DNA as the primary mode** — and it does *not* separate from
fusion on cluster recall (Gate B holds on fraction-recall; DNA adds nothing to the tail). But the
realistic regime makes fusion meaningfully more valuable **at rank-1** (P@1 +10.7 pp, hit-rate@3
+8.8 pp vs ~+2 pp and +0.4 pp under Q_full). Updated recommendation: keep Text→DNA (SapBERT)
primary; **enabling RRF fusion (w=2) is a stronger call in the realistic organism-only regime than
the full-metadata result suggested** — its benefit is concentrated at the top of the list (P@1,
hit-rate), not in deep recall. The deferred generation evaluation (Gate 3) remains the larger
next step.

### Encoder check — does SapBERT's Index-B win survive organism-only?
Experiment 11 selected SapBERT over MiniLM using **full-metadata** queries; this re-tests the
choice in the realistic regime by re-running the prior MiniLM encoder as an Index B baseline
(fresh MiniLM corpus index; full and organism-only queries; same leave-self-out harness — MiniLM's
full-metadata P@1 reproduces Experiment 11's 0.906, confirming the baseline is faithful).

**SapBERT stays ahead — and the margin widens.** Under organism-only, SapBERT reaches P@1 0.800 /
Recall@49 0.739 vs MiniLM's 0.699 / 0.670. SapBERT's P@1 lead over MiniLM grows from **+6.8 pp**
with full metadata (0.974 vs 0.906) to **+10.1 pp** organism-only (0.800 vs 0.699), with the same
direction on MRR (+8.6 pp) and Recall@49 (+6.9 pp). Both encoders lose the protein signal, but
SapBERT degrades
*less* at rank-1 (−17.4 pp vs MiniLM's −20.7 pp). The concern that SapBERT's entity-name
pretraining might choke on the seven near-identical `Lactobacillus sp. <numeric-id>` names does not
dominate: organism-name matching is exactly what SapBERT is good at, so the Index B choice is
empirically backed in the realistic regime, not only on the full-metadata benchmark (Fig 12.6).

**Limitation.** Q_organism assumes the assembly species is one of the 20 corpus species, so
organism-only matching is clean; if a real assembly's species is absent from the corpus,
organism-only retrieval degrades to genus/family matching — a corpus-coverage question outside
this ablation's scope.

### Reproduce
```bash
python experiments/experiment_12_recall_fusion.py            # §1–§12 (full experiment)
python experiments/experiment_12_recall_fusion.py ablation   # §13 organism-only ablation
```
§1–§12 outputs: `experiment_12_summary.csv` (retriever × K × lens), `experiment_12_results.csv`
(per-query), `experiment_12_index_dna.faiss` + `experiment_12_index_text.faiss`, Figs 12.1–12.3,
and `experiment_12_run.log`. The `ablation` sub-command reuses those indices (no corpus re-embed)
and writes `experiment_12_ablation_summary.csv` (level × retriever × K × lens), Figs 12.4–12.5,
`experiment_12_encoder_compare.csv` + Fig 12.6 (SapBERT vs MiniLM, full vs organism), and
`experiment_12_ablation_run.log`. The Index B encoder is set in `SELECTED_ENCODER`
(Experiment 11 handoff).
