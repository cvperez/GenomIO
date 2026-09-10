# Experiment 9: FAISS-Based Retrieval: Proof of Concept (5 CDS records): Results

The current reproduction notes below describe the checked-in script. The historical record retains the original measurements; raw artifacts mentioned there are not included.

## Current Reproduction Guide

### Objective

Repeat the small retrieval example through FAISS rather than direct cosine ranking.

### Experimental Setup

Both DNA/DNABERT-S and metadata/MiniLM use `IndexFlatIP` with normalized vectors. The script exercises DNA-to-DNA, text-to-DNA, and DNA-to-text routes, excludes self matches, and measures P@1/2/3 and MRR on queries with a same-label partner. It also measures FAISS timing, size, and agreement.

### Inputs

The checked-in [CDS corpus](../../../rag_corpus_uniform) supplies the FASTA records and headers. Selection and model settings are defined in the script, not in the root YAML configuration.

### How to Run

From the repository root, with the [benchmark dependencies](../../README.md#reproduction) installed:

```bash
python3 embedder_benchmark/experiment_9_faiss_retrieval_poc/experiment_9_faiss_retrieval_poc.py
```

These commands load model weights and may take substantial time. They were not executed for the repository cleanup. Outputs go to the `results/` directory beside the script; rerunning can overwrite generated artifacts.

### Results

The checked-in [historical result report](#historical-result-record) contains the recorded measurements. The raw CSVs, logs, plots, and binary artifacts described below are not included in this checkout.

`experiment_9_results.csv` and `experiment_9_summary.csv` contain retrieval metrics; `experiment_9_faiss_stats.csv` records index measurements. DNA/text indices and similarity plots are generated.

### Main Findings

The two evaluable queries have P@1 and MRR of 1.000 for each route, matching Experiment 5. The comparison establishes consistency of the small example; it does not demonstrate gap filling.

These findings summarize stored evidence, not a newly repeated experiment. Consult the [evidence limits](../../README.md#reproducibility-and-evidence-limits) before comparing historical numbers with a new run.

## Historical Result Record

> Historical evidence: numerical tables are retained. Raw CSVs, logs, and figures mentioned below are not included in this checkout. Earlier organism names, model explanations, and deployment recommendations may not describe the current implementation. Use the current reproduction guide above and [evidence limits](../../README.md#reproducibility-and-evidence-limits) for the current scope and reproduction constraints.

**Date**: 2026-05-01  
**Hardware**: CPU-only; Python 3.10; faiss 1.13.2; transformers 5.5.4; sentence-transformers 5.4.1  
**Script**: `embedder_benchmark/experiment_9_faiss_retrieval_poc/experiment_9_faiss_retrieval_poc.py`

---

## 1. Objective

Re-run the Experiment 5 retrieval evaluation routing all queries through FAISS indices
instead of brute-force cosine similarity (sklearn). Validates that the FAISS pipeline is
correctly wired end-to-end and that IndexFlatIP (exact cosine, the Experiment 8 recommendation
for n=5) reproduces Experiment 5 results exactly.

---

## 2. Setup

| Component | Value |
|-----------|-------|
| Records | 5 (same CDS as Experiments 1 and 5) |
| Index A | IndexFlatIP, DNA/DNABERT-S, 768-dim |
| Index B | IndexFlatIP, Text/MiniLM, 384-dim |
| Index type rationale | Per Exp 8: Flat is optimal for n < 1,000 |
| Query types | DNA→DNA, Text→DNA, DNA→Text |
| Evaluation | Leave-self-out; ground truth = same species label |

---

## 3. Query Pipeline Architecture (FAISS)

Experiment 9 re-routes all three query types through FAISS indices instead of sklearn cosine similarity. The **ID bridge** mechanism is unchanged: both FAISS indices (A and B) share the same integer row ordering as the `records[]` array, so `id_k` returned by FAISS is directly usable to fetch the opposite-modality payload.

### DNA→DNA (Index A: FAISS IndexFlatIP)

```
Query DNA  →  DNABERT-S  →  768-dim L2-normalised vector
   →  index_A.search(query_vec, k+1)       ← +1 to allow self-exclusion
   →  filter out query_idx from returned ids
   →  records[id_k].dna_sequence           ← payload
```

`IndexFlatIP` on L2-normalised vectors computes exact inner product = exact cosine similarity. Result is identical to sklearn brute-force cosine search.

### Text→DNA (Index B → ID bridge)

```
Query text  →  all-MiniLM-L6-v2  →  384-dim L2-normalised vector
   →  index_B.search(query_vec, k+1)
   →  filter out self → ranked (score, id) pairs
   →  ID bridge: records[id_k].dna_sequence   ← DNA payload
```

The text FAISS index (Index B) scores records by metadata similarity; the bridge then looks up the corresponding DNA sequence by row index. The DNA content is never searched in this path.

### DNA→Text (Index A → ID bridge)

```
Query DNA  →  DNABERT-S  →  768-dim L2-normalised vector
   →  index_A.search(query_vec, k+1)
   →  filter out self → ranked (score, id) pairs
   →  ID bridge: records[id_k].metadata_text  ← metadata payload
```

Identical search to DNA→DNA; only the payload changes. At n=5 all three query types produce P@1=1.0, confirming FAISS wiring is correct end-to-end before scaling.

---

## 4. FAISS Index Statistics

| Index | Model | Type | Build (ms) | Latency (µs/q) | Memory (bytes) | Top-1 accuracy |
|-------|-------|------|-----------|----------------|----------------|----------------|
| A (DNA) | DNABERT-S | IndexFlatIP | 0.153 | 3.47 | 15,405 | **1.000** |
| B (Text) | MiniLM-L6-v2 | IndexFlatIP | 0.016 | 1.94 | 7,725 | **1.000** |

Both indices achieve 100% top-1 accuracy relative to brute-force cosine baseline : confirming
that IndexFlatIP produces exactly the same ranking as sklearn cosine similarity.

---

## 5. Ranked Retrieval Results

Only the *T. thermophilus* pair has same-species ground truth. Singleton species (CT, SA, DM)
are excluded from P@k/MRR evaluation (no relevant document exists in the index).

### 5.1 DNA→DNA (FAISS Index A)

| Query | Rank 1 | Score | Relevant? |
|-------|--------|-------|-----------|
| T.thermophilus_1 | T.thermophilus_2 | 0.8125 | ✓ |
| T.thermophilus_2 | T.thermophilus_1 | 0.8125 | ✓ |

### 5.2 Text→DNA (FAISS Index B → ID bridge)

| Query | Rank 1 | Score | Relevant? |
|-------|--------|-------|-----------|
| T.thermophilus_1 | T.thermophilus_2 | 0.9437 | ✓ |
| T.thermophilus_2 | T.thermophilus_1 | 0.9437 | ✓ |

### 5.3 DNA→Text (FAISS Index A → ID bridge)

| Query | Rank 1 | Metadata returned | Relevant? |
|-------|--------|-------------------|-----------|
| T.thermophilus_1 | `organism=Thermus thermophilus locus_tag=TT_RS00015 protein=NAD(P)/FAD-dependent oxidoreductase...` | ✓ |
| T.thermophilus_2 | `organism=Thermus thermophilus locus_tag=TT_RS00010 protein=SDR family NAD(P)-dependent oxidoreductase...` | ✓ |

---

## 6. Evaluation Metrics

| Query type | P@1 | P@2 | P@3 | MRR | n (with GT) |
|------------|-----|-----|-----|-----|-------------|
| DNA→DNA | **1.000** | 0.500 | 0.333 | **1.000** | 2 |
| Text→DNA | **1.000** | 0.500 | 0.333 | **1.000** | 2 |
| DNA→Text | **1.000** | 0.500 | 0.333 | **1.000** | 2 |

> P@2=0.5 and P@3=0.333 are mathematically optimal: only 1 relevant document exists per
> query, so the denominator penalises beyond the single correct answer.

---

## 7. Comparison with Experiment 5 (Brute-Force Cosine)

| Metric | Exp 5 (sklearn cosine) | Exp 9 (FAISS IndexFlatIP) | Match |
|--------|----------------------|--------------------------|-------|
| DNA→DNA P@1 | 1.000 | 1.000 | ✓ |
| Text→DNA P@1 | 1.000 | 1.000 | ✓ |
| DNA→Text P@1 | 1.000 | 1.000 | ✓ |
| DNA→DNA MRR | 1.000 | 1.000 | ✓ |
| Text→DNA MRR | 1.000 | 1.000 | ✓ |
| DNA→Text MRR | 1.000 | 1.000 | ✓ |
| Top-1 FAISS accuracy vs BF | n/a | 1.000 | ✓ |

Exact match across all metrics. IndexFlatIP is a lossless replacement for brute-force cosine
at n=5.

---

## 8. Visualizations

| File | Contents |
|------|----------|
| `experiment_9_dna_dna_sim.png` | DNA–DNA cosine similarity heatmap (DNABERT-S) |
| `experiment_9_txt_txt_sim.png` | Text–Text cosine similarity heatmap (MiniLM) |
| `experiment_9_index_dna.faiss` | Serialised DNA FAISS index |
| `experiment_9_index_text.faiss` | Serialised Text FAISS index |

---

## 9. Conclusion

**FAISS IndexFlatIP produces identical results to brute-force cosine search at n=5.**
P@1=1.0 and MRR=1.0 for all 3 query types, consistent with Experiment 5. Both indices are
100% accurate, sub-millisecond build time, and sub-4 µs query latency. The FAISS pipeline
is correctly wired for all three query modes (DNA→DNA, Text→DNA, DNA→Text via ID bridge).

Experiment 10 scales this to n=1,000 using the Experiment 8-recommended HNSW index for
DNA and verifies that the approximation cost is negligible.
