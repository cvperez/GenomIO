# Experiment 10: FAISS-Based Retrieval — Scaled (1,000 CDS records) — Results

**Date**: 2026-05-01  
**Hardware**: CPU-only; Python 3.10; faiss 1.13.2; transformers 5.5.4; sentence-transformers 5.4.1  
**Script**: `experiments/experiment_10_faiss_retrieval_scale.py`

---

## 1. Objective

Scale the Experiment 9 FAISS retrieval proof-of-concept to 1,000 records (50/species × 20
species) using the Experiment 8-recommended index types, and compare retrieval quality to the
Experiment 6 brute-force cosine baseline. Also fixes the organism name bug from Experiment 6
(accession ID version suffix not stripped) to measure its impact on text retrieval quality.

---

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Records | 1,000 (50/species × 20 species, 300–900 bp) |
| Index A | IndexHNSWFlat(M=32, efConstruction=200, efSearch=64) — DNA, 768-dim |
| Index B | IndexFlatIP — Text, 384-dim |
| Index rationale | Per Exp 8: HNSW optimal for DNA at n≥1,000; Flat still optimal for Text at n=1,000 |
| Query types | DNA→DNA, Text→DNA, DNA→Text |
| Evaluation | Leave-self-out; ground truth = same species label; 49 relevant docs/query |
| Bug fix | Organism name now resolved correctly (stripped `.1` version suffix) |

---

## 3. Query Pipeline Architecture (FAISS, n=1,000)

The routing logic is identical to Experiment 9 but uses the Experiment 8-recommended index types at n=1,000. Both indices share the same integer row ordering as `records[]`, so every `id_k` returned by FAISS is directly usable to fetch the opposite-modality payload via the **ID bridge**.

### DNA→DNA (Index A — IndexHNSWFlat M=32)

```
Query DNA  →  DNABERT-S  →  768-dim L2-normalised vector
   →  index_A.search(query_vec, k+1)    ← HNSW approximate search
   →  filter out query_idx from returned ids
   →  records[id_k].dna_sequence        ← DNA payload
```

At n=1,000, IndexHNSWFlat (efSearch=64) achieves 100% top-1 accuracy vs exact cosine — the approximation introduces zero retrieval error at this corpus size (validated in Experiment 8).

### Text→DNA (Index B → ID bridge — IndexFlatIP)

```
Query text  →  all-MiniLM-L6-v2  →  384-dim L2-normalised vector
   →  index_B.search(query_vec, k+1)    ← exact search (Flat optimal at n=1,000)
   →  filter out self → ranked (score, id) pairs
   →  ID bridge: records[id_k].dna_sequence   ← DNA payload
```

Index B uses IndexFlatIP (exact cosine) because the Experiment 8 crossover for the text index is at n≈2,000; HNSW is not yet beneficial here. The **organism name bug fix** applied in this experiment (`accession_key = parts[0] + "_" + parts[1].split(".")[0]`) ensures metadata text carries real species names (`organism=Thermus thermophilus`) rather than accession IDs. This raises Text→DNA P@1 from 0.601 (Exp 6) to 0.906 — a +30 pp gain driven entirely by metadata quality, not model or index change.

### DNA→Text (Index A → ID bridge — IndexHNSWFlat M=32)

```
Query DNA  →  DNABERT-S  →  768-dim L2-normalised vector
   →  index_A.search(query_vec, k+1)    ← same HNSW search as DNA→DNA
   →  filter out self → ranked (score, id) pairs
   →  ID bridge: records[id_k].metadata_text  ← metadata payload
```

DNA→Text and DNA→DNA share the same Index A search and produce identical metrics. The bridge selects which payload (DNA sequence vs metadata string) to return.

---

## 4. FAISS Index Statistics

| Index | Model | Type | Embed time | Build (ms) | Latency (µs/q) | Memory (KB) | Top-1 accuracy |
|-------|-------|------|-----------|-----------|----------------|-------------|----------------|
| A (DNA) | DNABERT-S | IndexHNSWFlat(M=32) | 96.3 s | 42.0 | 63.6 | 3,265 | **1.000** |
| B (Text) | MiniLM-L6-v2 | IndexFlatIP | 5.5 s | 1.0 | 22.9 | 1,500 | **1.000** |

Both indices achieve 100% top-1 accuracy relative to brute-force cosine — HNSW approximation
introduces zero retrieval error at n=1,000 with efSearch=64.

---

## 5. Retrieval Metrics

| Query type | P@1 | P@5 | P@10 | P@20 | Recall@10 | Recall@20 | MRR |
|------------|-----|-----|------|------|-----------|-----------|-----|
| **DNA→DNA** | **0.655** | **0.616** | **0.575** | **0.528** | **0.117** | **0.216** | **0.766** |
| **DNA→Text** | 0.655 | 0.616 | 0.575 | 0.528 | 0.117 | 0.216 | 0.766 |
| **Text→DNA** | **0.906** | **0.895** | **0.893** | **0.882** | **0.182** | **0.360** | **0.938** |

---

## 6. Comparison: FAISS (Exp 10) vs Brute-Force (Exp 6)

| Query type | Metric | Exp 6 (Brute-force, sklearn) | Exp 10 (FAISS) | Δ |
|------------|--------|-----------------------------|-----------------|----|
| DNA→DNA | P@1 | 0.655 | 0.655 | 0.000 |
| DNA→DNA | P@10 | 0.575 | 0.575 | 0.000 |
| DNA→DNA | MRR | 0.767 | 0.766 | −0.001 |
| Text→DNA | P@1 | 0.601 | **0.906** | **+0.305** |
| Text→DNA | P@10 | 0.531 | **0.893** | **+0.362** |
| Text→DNA | MRR | 0.704 | **0.938** | **+0.234** |
| DNA→Text | P@1 | 0.655 | 0.655 | 0.000 |
| DNA→Text | P@10 | 0.575 | 0.575 | 0.000 |
| DNA→Text | MRR | 0.767 | 0.766 | −0.001 |

---

## 7. Visualizations

| File | Contents |
|------|----------|
| `experiment_10_DNA_DNA_p1_by_species.png` | P@1 per species — DNA→DNA |
| `experiment_10_DNA_DNA_p10_by_species.png` | P@10 per species — DNA→DNA |
| `experiment_10_Text_DNA_p1_by_species.png` | P@1 per species — Text→DNA |
| `experiment_10_Text_DNA_p10_by_species.png` | P@10 per species — Text→DNA |
| `experiment_10_DNA_Text_p1_by_species.png` | P@1 per species — DNA→Text |
| `experiment_10_DNA_Text_p10_by_species.png` | P@10 per species — DNA→Text |
| `experiment_10_summary_bar.png` | Summary bar chart across all 3 query types |
| `experiment_10_comparison_vs_exp6.png` | FAISS vs brute-force side-by-side bar chart |
| `experiment_10_index_dna.faiss` | Serialised DNA HNSW index |
| `experiment_10_index_text.faiss` | Serialised Text Flat index |

---

## 8. Analysis

### DNA→DNA and DNA→Text — identical to Experiment 6

Both query modes search Index A and produce P@1=0.655 and MRR=0.766 — essentially identical
to Experiment 6 (P@1=0.655, MRR=0.767). The −0.001 MRR difference is floating-point rounding.

This confirms that **IndexHNSWFlat(M=32) with efSearch=64 is lossless at n=1,000** for the
DNA index: 100% top-1 accuracy vs brute-force, and no measurable degradation in retrieval
quality metrics. HNSW is a safe replacement for exact cosine search at this corpus size.

### Text→DNA — dramatic improvement (+0.305 P@1) from organism name fix

The Text→DNA P@1 jumped from 0.601 (Exp 6) to **0.906** (Exp 10), and MRR from 0.704 to
**0.938**. This improvement is **not caused by FAISS** — both Exp 6 and Exp 10 use IndexFlatIP
for the text index (exact search). The improvement is entirely due to the **organism name bug fix**.

In Experiment 6, metadata text defaulted to accession IDs
(e.g., `organism=GCF_000008125.1 locus_tag=...`) because the `.1` version suffix prevented
the organism name lookup from matching. With the fix applied in Experiment 10, same-species
records share the correct organism name (e.g., `organism=Thermus thermophilus`), which the
MiniLM model correctly identifies as a strong similarity signal. This raises same-species
metadata similarity significantly, improving the ranking of relevant documents.

The corrected Text→DNA performance (P@1=0.906, MRR=0.938) is now **substantially better than
DNA→DNA** (P@1=0.655, MRR=0.766), making text queries the highest-performing retrieval mode.
This makes intuitive sense: organism names are highly discriminative for species-level retrieval,
whereas DNA sequences can share functional similarity across species (conserved genes).

### FAISS accuracy — no approximation loss at n=1,000

Both indices achieve **100% top-1 accuracy** vs brute-force cosine. The HNSW approximation
(efSearch=64) introduces zero retrieval errors at n=1,000. This is consistent with Experiment 8
findings where HNSW accuracy was ≥99.7% at n=1,000 (the sub-100% values there were due to
float32/float64 tie-breaking, not genuine errors).

### Latency — HNSW competitive at n=1,000

- DNA Index A (HNSW): 63.6 µs/query — matches Experiment 8 measurement of ~52 µs at n=1,000
  (slight difference due to N_QUERY_REPS=100 and batch size differences).
- Text Index B (Flat): 22.9 µs/query — consistent with Experiment 8.

At n=1,000, HNSW is essentially tied with Flat for the DNA index (Experiment 8 crossover
at n≈1,000), confirming the transition point. For n>1,000, HNSW will be strictly faster.

---

## 9. Conclusion

**FAISS-based retrieval is fully validated at n=1,000 with zero quality degradation:**

| Query type | P@1 | MRR | vs Exp 6 | Cause of change |
|------------|-----|-----|----------|-----------------|
| DNA→DNA | 0.655 | 0.766 | No change | FAISS exact match |
| DNA→Text | 0.655 | 0.766 | No change | FAISS exact match |
| Text→DNA | **0.906** | **0.938** | **+0.305** | **Organism name bug fix** |

The deployed system (HNSW for DNA, Flat for Text at n=1,000) achieves:
- **DNA retrieval**: P@1=0.655, MRR=0.766 — unchanged from brute-force
- **Text retrieval**: P@1=0.906, MRR=0.938 — substantially improved after metadata fix
- **FAISS accuracy**: 100% for both indices — no approximation cost at this scale
- **Query latency**: 64 µs (DNA HNSW) + 23 µs (Text Flat) — well within real-time budget

The organism name fix is a critical finding: correcting the metadata text alone improves
Text→DNA P@1 by 30 percentage points. The final system should ensure organism names are
always included in metadata text for maximum retrieval performance.
