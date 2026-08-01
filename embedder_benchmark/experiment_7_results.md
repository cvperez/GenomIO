# Experiment 7: FAISS Index Construction — Proof of Concept (5 CDS) — Results

**Date**: 2026-04-29  
**Hardware**: CPU-only; Python 3.10; faiss 1.13.2  
**Script**: `experiments/experiment_7_faiss_index.py`

---

## 1. Objective

Validate FAISS index construction for the confirmed dual-modality strategy using the 5 CDS
records from Experiment 1. Test two index types per modality and measure build time, query
latency, top-k accuracy vs brute-force cosine baseline, and serialised memory footprint.

---

## 2. Index Configurations

| Index | Modality | Dim | Type | Description |
|-------|----------|-----|------|-------------|
| A-Flat | DNA | 768 | IndexFlatIP | Exact inner-product search on L2-normalised vectors |
| A-HNSW | DNA | 768 | IndexHNSWFlat(M=32) | Approximate HNSW graph, efSearch=16 |
| B-Flat | Text | 384 | IndexFlatIP | Exact inner-product search on L2-normalised vectors |
| B-HNSW | Text | 384 | IndexHNSWFlat(M=32) | Approximate HNSW graph, efSearch=16 |

> Vectors are L2-normalised before insertion. Inner product on unit vectors equals cosine
> similarity, so IndexFlatIP is the correct exact cosine search index in FAISS.

---

## 3. Results

| Modality | Index type | Build (ms) | Latency (µs/query) | Memory (bytes) | Top-1 acc | Top-2 acc | Top-3 acc |
|----------|-----------|-----------|-------------------|----------------|-----------|-----------|-----------|
| DNA (768-dim) | IndexFlatIP | 0.094 | **1.94** | 15,405 | 1.000 | 1.000 | 1.000 |
| DNA (768-dim) | IndexHNSWFlat(M=32) | 0.091 | 50.89 | 16,926 | 1.000 | 1.000 | 1.000 |
| Text (384-dim) | IndexFlatIP | 0.023 | **2.09** | 7,725 | 1.000 | 1.000 | 1.000 |
| Text (384-dim) | IndexHNSWFlat(M=32) | 0.045 | 25.56 | 9,246 | 1.000 | 1.000 | 1.000 |

---

## 4. Saved Index Files

| File | Size |
|------|------|
| `experiment_7_index_dna_flat.faiss` | 15,405 bytes |
| `experiment_7_index_dna_hnsw.faiss` | 16,926 bytes |
| `experiment_7_index_text_flat.faiss` | 7,725 bytes |
| `experiment_7_index_text_hnsw.faiss` | 9,246 bytes |

---

## 5. Analysis

### Accuracy — both index types correct at n=5

All four index configurations achieve 100% accuracy (top-1, top-2, top-3) relative to the
brute-force cosine baseline. At n=5, approximation is irrelevant — HNSW with M=32 still
builds an exact graph over all 5 nodes, so both index types return identical results.

### Latency — IndexFlatIP is faster than HNSW at small n

**IndexFlatIP is 26× faster than IndexHNSWFlat for DNA queries (1.94 µs vs 50.89 µs)** and
12× faster for text queries (2.09 µs vs 25.56 µs). This is a well-known HNSW characteristic:
at small corpus sizes, the graph traversal overhead (pointer chasing, efSearch candidate
evaluation) dominates over the brute-force scan cost. The crossover where HNSW becomes
competitive typically occurs between n=1,000 and n=10,000 depending on dimension and M.

At n=5, a flat scan requires only 5 dot products. HNSW must traverse its graph even when
the answer is trivially the nearest of 5 nodes.

### Memory — HNSW overhead is ~10%

HNSW uses ~10% more memory than flat at n=5 (graph adjacency list overhead: M=32 links
per node × 4 bytes × 5 nodes = 640 bytes additional). This overhead is expected to remain
modest as n grows (O(n × M × 4 bytes) for the graph vs O(n × d × 4 bytes) for vectors).

### Build time — sub-millisecond for both at n=5

Both index types build in under 0.1 ms. Build time is dominated by vector insertion, which
is O(n) for flat and O(n log n) for HNSW. At n=5, both are negligible.

---

## 6. Conclusion

FAISS integration is validated. Both index types correctly embed the dual-modality strategy:
- **Index A (DNA, DNABERT-S)**: 768-dim, IndexFlatIP or IndexHNSWFlat
- **Index B (Text, MiniLM)**: 384-dim, IndexFlatIP or IndexHNSWFlat

At n=5, **IndexFlatIP is the correct choice**: exact, faster, and smaller. HNSW's
approximation advantage does not materialise until the corpus grows to thousands of vectors.
Experiment 8 scales to n=1,000 to determine whether HNSW becomes competitive at the corpus
sizes typical for this RAG system.
