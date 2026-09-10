# Experiment 5: Retrieval Querying: Dual-Modality (5 CDS): Results

The current reproduction notes below describe the checked-in script. The historical record retains the original measurements; raw artifacts mentioned there are not included.

## Current Reproduction Guide

### Objective

Exercise DNA-to-DNA, text-to-DNA, and DNA-to-text retrieval through a shared record-ID mapping.

### Experimental Setup

DNABERT-S DNA vectors and MiniLM metadata vectors are searched with in-memory cosine similarity. The five-record sample is the same small setup used in Experiment 1. Ranking excludes the query itself; another record with the same label is relevant. P@1, P@2, P@3, and reciprocal rank are evaluated on the two queries with a relevant partner.

### Inputs

The checked-in [CDS corpus](../../../rag_corpus_uniform) supplies the FASTA records and headers. Selection and model settings are defined in the script, not in the root YAML configuration.

### How to Run

From the repository root, with the [benchmark dependencies](../../README.md#reproduction) installed:

```bash
python3 embedder_benchmark/experiment_5_retrieval_query/experiment_5_retrieval_query.py
```

These commands load model weights and may take substantial time. They were not executed for the repository cleanup. Outputs go to the `results/` directory beside the script; rerunning can overwrite generated artifacts.

### Results

The checked-in [historical result report](#historical-result-record) contains the recorded measurements. The raw CSVs, logs, plots, and binary artifacts described below are not included in this checkout.

`experiment_5_results.csv` contains query metrics; `experiment_5_summary.csv` aggregates them by query type. DNA and text similarity heatmaps are generated.

### Main Findings

All three query types report P@1 and MRR of 1.000 on the two evaluable queries. The other three records have no same-label partner. This validates a small retrieval example, not general retrieval performance.

These findings summarize stored evidence, not a newly repeated experiment. Consult the [evidence limits](../../README.md#reproducibility-and-evidence-limits) before comparing historical numbers with a new run.

## Historical Result Record

> Historical evidence: numerical tables are retained. Raw CSVs, logs, and figures mentioned below are not included in this checkout. Earlier organism names, model explanations, and deployment recommendations may not describe the current implementation. Use the current reproduction guide above and [evidence limits](../../README.md#reproducibility-and-evidence-limits) for the current scope and reproduction constraints.

**Date**: 2026-04-28  
**Hardware**: CPU-only; Python 3.10; transformers 5.5.4; sentence-transformers 5.4.1  
**Script**: `embedder_benchmark/experiment_5_retrieval_query/experiment_5_retrieval_query.py`

---

## 1. Objective

Validate that the confirmed dual-modality strategy (DNABERT-S for DNA, all-MiniLM-L6-v2 for
metadata) enables all three retrieval query types:
- **DNA→DNA**: DNA sequence query → ranked DNA records (by sequence similarity)
- **Text→DNA**: metadata text query → text index → ranked IDs → corresponding DNA records
- **DNA→Text**: DNA sequence query → DNA index → ranked IDs → corresponding metadata records

Evaluation uses the 5 CDS records from Experiment 1 as both index and query set
(leave-self-out ranking). Ground truth: same species label = relevant document.

---

## 2. Index Statistics

| Index | Model | Shape | Build time | Memory |
|-------|-------|-------|-----------|--------|
| A: DNA | DNABERT-S | (5, 768) | 4.50 s | 15.0 KB |
| B: Metadata | all-MiniLM-L6-v2 | (5, 384) | 1.38 s | 7.5 KB |

---

## 3. Query Pipeline Architecture

The dual-modality system maintains **two separate indices** : Index A for DNA (DNABERT-S, 768-dim) and Index B for metadata text (MiniLM, 384-dim). Neither index is projected into the other's space. Queries are routed by input modality, and the **ID bridge** crosses the modality boundary using the integer row index shared by both indices: `id_k` returned by either index is the position of that record in the `records[]` array, so the opposite-modality payload is retrieved as `records[id_k].dna_sequence` or `records[id_k].metadata_text`.

### Why metadata is embedded rather than filtered

Metadata text (`organism=Thermus thermophilus locus_tag=TT_RS00015 protein=NAD(P)/FAD-dependent oxidoreductase`) is free-form and semantically variable. Exact filtering requires a precise string match and cannot rank records by relevance or handle paraphrases: `"NAD(P)/FAD-dependent oxidoreductase"` and `"SDR family NAD(P)-dependent oxidoreductase"` share meaning but not tokens. Embedding with all-MiniLM-L6-v2 maps metadata into a 384-dim vector space where semantically similar descriptions are geometrically close : enabling similarity-ranked retrieval. DNABERT-S, trained solely on nucleotide k-mers, cannot meaningfully encode natural language text; a separate text encoder is mandatory for Index B.

Filters (exact match on structured fields) remain useful as a post-processing step (retrieve top-N from either index, then discard records that fail a hard constraint) but they cannot replace embedding for ranked retrieval over free-text descriptions.

### Query type 1: DNA→DNA (Index A only)

```
Query: DNA sequence
      │
      ▼  DNABERT-S → attention-masked mean pool → 768-dim L2-normalised vector
      ▼
 Index A  (cosine similarity, sklearn)
      │  returns: [(score_0, id_0), (score_1, id_1), ...]
      │  id_k = integer row index in records[]
      ▼
 records[id_k].dna_sequence            ← DNA payload delivered, ranked by sequence similarity
```

No bridging: query and payload are both in the DNA modality.

### Query type 2: Text→DNA (Index B → ID bridge → DNA payload)

```
Query: metadata text string
      │
      ▼  all-MiniLM-L6-v2 → 384-dim L2-normalised vector
      ▼
 Index B  (cosine similarity, sklearn)
      │  returns: [(score_0, id_0), (score_1, id_1), ...]
      │  id_k = same integer row index in records[]   ← shared ordering
      ▼
 ID BRIDGE: records[id_k].dna_sequence  ← DNA payload delivered, ranked by metadata similarity
```

Index B ranks records by metadata similarity; the bridge delivers their DNA sequences. The DNA sequences are never searched : they are the payload retrieved by ID.

### Query type 3: DNA→Text (Index A → ID bridge → metadata payload)

```
Query: DNA sequence
      │
      ▼  DNABERT-S → 768-dim L2-normalised vector
      ▼
 Index A  (cosine similarity, sklearn)
      │  returns: [(score_0, id_0), (score_1, id_1), ...]
      │  id_k = integer row index in records[]   ← shared ordering
      ▼
 ID BRIDGE: records[id_k].metadata_text ← metadata payload delivered, ranked by DNA similarity
```

This is the exact reverse of Text→DNA: the DNA index ranks records by sequence similarity, and the bridge delivers their metadata strings. A DNA query for an oxidoreductase gene returns the metadata of the most similar DNA records : organism name, locus tag, protein description.

---

## 4. Similarity Matrices

### DNA–DNA (Index A cosine similarities)

|  | TT_1 | TT_2 | CT | SA | DM |
|--|------|------|----|----|----|
| **TT_1** | 1.000 | **0.813** | 0.136 | 0.154 | 0.314 |
| **TT_2** | 0.813 | 1.000 | 0.153 | 0.201 | 0.349 |
| **CT** | 0.136 | 0.153 | 1.000 | 0.346 | 0.106 |
| **SA** | 0.154 | 0.201 | 0.346 | 1.000 | 0.196 |
| **DM** | 0.314 | 0.349 | 0.106 | 0.196 | 1.000 |

### Text–Text (Index B cosine similarities)

|  | TT_1 | TT_2 | CT | SA | DM |
|--|------|------|----|----|----|
| **TT_1** | 1.000 | **0.944** | 0.487 | 0.748 | 0.645 |
| **TT_2** | 0.944 | 1.000 | 0.497 | 0.745 | 0.655 |
| **CT** | 0.487 | 0.497 | 1.000 | 0.559 | 0.544 |
| **SA** | 0.748 | 0.745 | 0.559 | 1.000 | 0.681 |
| **DM** | 0.645 | 0.655 | 0.544 | 0.681 | 1.000 |

---

## 5. Ranked Retrieval Results

> Only the *T. thermophilus* pair has ground truth (same species). Singleton species (CT, SA, DM) have no same-species record in the index and are excluded from P@k/MRR computation.

### 5.1 DNA→DNA (query: DNA embedding → search Index A)

**Query: T.thermophilus_1**

| Rank | Record | Score | Relevant? |
|------|--------|-------|-----------|
| 1 | T.thermophilus_2 | 0.8125 | ✓ |
| 2 | D.mccartyi | 0.3141 | No |
| 3 | S.aureus | 0.1543 | No |
| 4 | C.trachomatis | 0.1357 | No |

**Query: T.thermophilus_2**

| Rank | Record | Score | Relevant? |
|------|--------|-------|-----------|
| 1 | T.thermophilus_1 | 0.8125 | ✓ |
| 2 | D.mccartyi | 0.3487 | No |
| 3 | S.aureus | 0.2009 | No |
| 4 | C.trachomatis | 0.1532 | No |

### 5.2 Text→DNA (query: text embedding → search Index B → return DNA by ID)

**Query: T.thermophilus_1 metadata**

| Rank | Record | Text score | DNA record returned | Relevant? |
|------|--------|-----------|---------------------|-----------|
| 1 | T.thermophilus_2 | 0.9437 | TT_2 DNA sequence | ✓ |
| 2 | S.aureus | 0.7481 | SA DNA sequence | No |
| 3 | D.mccartyi | 0.6453 | DM DNA sequence | No |
| 4 | C.trachomatis | 0.4865 | CT DNA sequence | No |

**Query: T.thermophilus_2 metadata**

| Rank | Record | Text score | DNA record returned | Relevant? |
|------|--------|-----------|---------------------|-----------|
| 1 | T.thermophilus_1 | 0.9437 | TT_1 DNA sequence | ✓ |
| 2 | S.aureus | 0.7448 | SA DNA sequence | No |
| 3 | D.mccartyi | 0.6552 | DM DNA sequence | No |
| 4 | C.trachomatis | 0.4965 | CT DNA sequence | No |

### 5.3 DNA→Text (query: DNA embedding → search Index A → return metadata by ID)

**Query: T.thermophilus_1 DNA**

| Rank | Score | Metadata returned | Relevant? |
|------|-------|-------------------|-----------|
| 1 | 0.8125 | `organism=Thermus thermophilus locus_tag=TT_RS00015 protein=NAD(P)/FAD-dependent oxidoreductase...` | ✓ |
| 2 | 0.3141 | `organism=Dehalococcoides mccartyi gene=nadD...` | No |
| 3 | 0.1543 | `organism=Staphylococcus aureus locus_tag=SAB_RS00035...` | No |
| 4 | 0.1357 | `organism=Chlamydia trachomatis gene=gatC...` | No |

**Query: T.thermophilus_2 DNA**

| Rank | Score | Metadata returned | Relevant? |
|------|-------|-------------------|-----------|
| 1 | 0.8125 | `organism=Thermus thermophilus locus_tag=TT_RS00010 protein=SDR family NAD(P)-dependent oxidoreductase...` | ✓ |
| 2 | 0.3487 | `organism=Dehalococcoides mccartyi gene=nadD...` | No |
| 3 | 0.2009 | `organism=Staphylococcus aureus locus_tag=SAB_RS00035...` | No |
| 4 | 0.1532 | `organism=Chlamydia trachomatis gene=gatC...` | No |

---

## 6. Evaluation Metrics

| Query type | P@1 | P@2 | P@3 | MRR | n (queries with GT) |
|------------|-----|-----|-----|-----|---------------------|
| DNA→DNA | **1.000** | 0.500 | 0.333 | **1.000** | 2 |
| Text→DNA | **1.000** | 0.500 | 0.333 | **1.000** | 2 |
| DNA→Text | **1.000** | 0.500 | 0.333 | **1.000** | 2 |

> P@2 = 0.5 and P@3 = 0.333 reflect the corpus structure: only 1 relevant document exists per query
> (the other TT record), so the denominator k penalises results beyond the single relevant document.
> These values are mathematically optimal given n_relevant=1 per query.

---

## 7. Analysis

### All three query types work correctly

Every query type retrieves the same-species record at rank 1 (P@1=1.0, MRR=1.0):
- **DNA→DNA**: DNABERT-S correctly places TT_1 and TT_2 closer to each other (0.813) than to
  any other species (max 0.349), matching Experiment 1.
- **Text→DNA**: The text index retrieves TT records as each other's nearest neighbor (0.9437),
  well separated from SA (0.748) and DM (0.645). The ID bridge correctly returns the corresponding
  DNA record.
- **DNA→Text**: The DNA index retrieves same-species DNA first (0.8125), and the ID bridge
  returns the correct same-species metadata. The returned metadata is immediately interpretable:
  both TT records share organism name and oxidoreductase protein family.

### Cross-modal retrieval via ID bridge is transparent and correct

The dual-index architecture with ID bridging works without any cross-space projection. A DNA
query retrieves DNA records by sequence similarity, and the metadata of those records is returned
by ID lookup. A text query retrieves the most semantically similar metadata records, and their
DNA sequences are returned by the same ID lookup. The two modalities remain in separate
geometric spaces; only record IDs cross the boundary.

### Score gap confirms retrieval reliability

For DNA→DNA and DNA→Text, the gap between rank-1 (0.813) and rank-2 (0.314–0.349) scores is
large : the correct answer is unambiguous. For Text→DNA, the gap between rank-1 (0.944) and
rank-2 (0.745–0.748) is also clear, though smaller because text similarities are more uniformly
high (organism names share vocabulary with other records in the SA/DM metadata).

### Limitation: only 2 evaluable queries

Only the TT pair provides same-species ground truth among 5 records. The P@k values are
statistically insufficient for general conclusions. Experiment 6 scales to 1,000 records
across 20 species (49 relevant documents per query) for robust evaluation.

---

## 8. Conclusion

The dual-modality strategy (DNABERT-S Index A + MiniLM Index B) **successfully supports all
three query types** in a proof-of-concept setting. P@1 = 1.0 and MRR = 1.0 for all query
types. The ID bridge mechanism correctly routes cross-modal queries without requiring a shared
embedding space. Experiment 6 validates these results at scale with statistically meaningful
sample sizes.
