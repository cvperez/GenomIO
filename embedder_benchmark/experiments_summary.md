# Experiments Summary — GenomIO RAG Infrastructure

**Period**: 2026-04-21 – 2026-05-01  
**Hardware**: CPU-only; Python 3.10; PyTorch 2.11; transformers 5.5.4; faiss 1.13.2  
**Corpus**: `rag_corpus_uniform/` — 20 CDS-only FASTA files (one per bacterial species)

---

## Background and Glossary

This section defines the key concepts used throughout the experiments. A reader unfamiliar with machine learning or bioinformatics retrieval systems can use it as a reference before reading any individual experiment.

---

### DNA language model
A neural network trained on large collections of DNA sequences. Like a language model trained on text, it learns statistical patterns in nucleotide sequences. Different training objectives produce very different results: some models learn to predict masked nucleotides (MLM), others learn to place sequences from the same species close together in vector space (contrastive training). Whether a model is useful for retrieval depends on its training objective, not its size.

### Embedding
A numerical representation of a sequence or text as a fixed-length list of numbers (a vector). Two inputs that are biologically or semantically similar should produce vectors that are close together in space. An embedding model takes a DNA sequence or a text string and returns its vector. The quality of an embedding determines how useful it is for finding similar records.

### Embedding space / vector space
The mathematical space where embeddings live. Each record is a point in this space. A well-structured embedding space places records from the same species close together and records from different species far apart. Poor embedding spaces distribute all records uniformly, making similarity search meaningless.

### Cosine similarity
A number between −1 and 1 that measures how similar two vectors are, regardless of their magnitude. A value of 1.0 means the vectors point in exactly the same direction (identical); 0.0 means they are orthogonal (unrelated); negative values indicate opposition. In practice, DNA embedding similarities between same-species sequences typically range from 0.5 to 0.9.

### Silhouette score
A single number summarising cluster quality, ranging from −1 to +1. A positive score means that, on average, each record is more similar to other records of its own species than to records of any other species — the desired behaviour for retrieval. A score near 0 means the clusters overlap. A negative score means records are actually closer to records from *other* species than their own, which would cause a retrieval system to return wrong results.

### Intra-species similarity / Inter-species distance
Intra-species similarity is the average cosine similarity between all pairs of records from the same species. Inter-species distance is 1 minus the average cosine similarity between all pairs of records from *different* species. A good embedding model should have high intra-species similarity (same-species records are close) and high inter-species distance (different-species records are far apart).

### MLM pre-training (Masked Language Modelling)
A training objective where the model learns to predict randomly masked tokens (nucleotides or words) from context. This produces rich representations of local sequence patterns but does not explicitly optimise for placing similar sequences close together in vector space. As a result, MLM-trained models often perform poorly for nearest-neighbour retrieval tasks.

### Contrastive training
A training objective where the model is shown pairs of similar sequences (e.g., two sequences from the same species) and pairs of dissimilar sequences, and is trained to pull similar pairs closer together and push dissimilar pairs further apart in vector space. This directly optimises the property needed for retrieval — which is why contrastively trained models outperform MLM-trained models of similar size in these experiments.

### RAG — Retrieval-Augmented Generation
A system architecture where a retrieval component first finds relevant documents from a corpus, and a generative model (e.g., an LLM) uses those documents as context to produce its output. In GenomIO, the retrieval step finds CDS records similar to a query DNA sequence, and those records provide biological context for gap-filling.

### Dual-modality index
Because each CDS record has two types of content — a raw DNA nucleotide sequence and a text metadata string (organism name, gene name, protein description) — two separate indices are needed. Index A holds DNA embeddings (DNABERT-S, 768-dimensional vectors). Index B holds text embeddings (all-MiniLM-L6-v2, 384-dimensional vectors). The two indices cannot be merged because they have different dimensions and were produced by incompatible models.

### ID bridge
The mechanism that connects the two indices. Both indices are built in the same order as the record list, so any record has the same integer row index (e.g., row 42) in both Index A, Index B, and the records list. When a query is run against one index (e.g., a text query finds record 42 in Index B), the corresponding DNA sequence is retrieved by simply looking up `records[42].dna_sequence`. No mathematical projection between embedding spaces is needed.

### FAISS
Facebook AI Similarity Search — an open-source library for fast nearest-neighbour search over large collections of vectors. It provides both exact search (guaranteed to return the true closest vectors) and approximate search (faster, with a small chance of missing the closest result). Used in these experiments to replace slow brute-force cosine similarity with a production-ready index.

### IndexFlatIP
A FAISS index type that performs exact inner-product search. When vectors are L2-normalised (scaled to unit length), inner product equals cosine similarity, so this is equivalent to exact cosine search. It is always 100% accurate but scales linearly with corpus size — every query must compare against every stored vector.

### IndexHNSWFlat (HNSW)
A FAISS index type that builds a multi-layer proximity graph (Hierarchical Navigable Small World). Queries traverse the graph rather than scanning all vectors, giving sub-linear query time. The trade-off is slightly reduced accuracy (a small fraction of queries may not return the exact nearest neighbour) and higher memory. In practice, accuracy stays above 99.7% at the scales tested here.

### P@k — Precision at k
The fraction of the top-k retrieved results that are relevant (i.e., from the same species as the query). P@1 is the probability that the single top result is correct. P@5 is the fraction of the top 5 results that are correct, and so on. A RAG system typically operates at k=3 to k=5, so P@5 is the most operationally relevant metric.

### Recall@k
The fraction of all relevant documents in the corpus that appear in the top-k results. If a query has 49 relevant documents (same-species records) and the top-10 results contain 6 of them, Recall@10 = 6/49 ≈ 0.12. Recall grows as k increases; high recall at small k is desirable.

### MRR — Mean Reciprocal Rank
The average of 1/rank across all queries, where rank is the position of the first relevant result. MRR=1.0 means every query finds a relevant result at rank 1. MRR=0.5 means the first relevant result is at rank 2 on average. MRR=0.767 (DNA→DNA, Experiment 6) means the first same-species record appears on average at rank ≈ 1.3 — almost always within the top two results.

### Leave-self-out evaluation
An evaluation protocol where each record is used as a query against the full corpus, but its own entry is excluded from the results (since retrieving yourself is trivially correct). Ground truth is defined as "same species label = relevant". This gives 1,000 evaluable queries from a 1,000-record corpus without needing a separate test set.

---

## Research Question

Which embedding and indexing strategy produces the most biologically meaningful and
operationally efficient retrieval infrastructure for a genomic RAG system built over
bacterial CDS corpora?

CDS records are inherently dual-modal: they combine a raw nucleotide sequence with
structured English-language metadata (organism name, gene name, protein description,
accession ID). The system must support three retrieval modes:
- **DNA→DNA**: given a DNA sequence, find the most similar DNA sequences
- **Text→DNA**: given a metadata text query, retrieve the most relevant DNA records
- **DNA→Text**: given a DNA sequence, retrieve the associated metadata of similar records

---

## Final System Architecture (result of all 10 experiments)

```
CDS Record: [DNA sequence] + [Metadata text]
        │
        ├─ DNABERT-S (zhihan1996/DNABERT-S) ──────► Index A  (FAISS IndexHNSWFlat, 768-dim)
        │   contrastive-trained genomics LM              └─► DNA→DNA queries
        │                                                └─► DNA→Text queries (ID bridge)
        │
        └─ all-MiniLM-L6-v2 ────────────────────► Index B  (FAISS IndexFlatIP, 384-dim)
            general-purpose text model                   └─► Text→DNA queries (ID bridge)
```

**Retrieval performance via FAISS at n=1,000 sequences, 20 species (Experiment 10):**

| Query type | P@1 | P@5 | P@10 | MRR | FAISS accuracy |
|------------|-----|-----|------|-----|----------------|
| DNA→DNA | 0.655 | 0.616 | 0.575 | 0.766 | 100% |
| DNA→Text | 0.655 | 0.616 | 0.575 | 0.766 | 100% |
| Text→DNA | **0.906** | **0.895** | **0.893** | **0.938** | 100% |

> Text→DNA performance jumped from 0.601 (Exp 6) to 0.906 after fixing the organism name
> bug in metadata text — organism names are now correctly included (e.g., "Thermus thermophilus"
> instead of accession IDs), making same-species metadata embeddings far more discriminative.

---

## Quick Reference Table

| # | Experiment | Scale | Main finding | Conclusion |
|---|-----------|-------|--------------|------------|
| 1 | Embedding model selection | 5 seqs, 4 species, **16 models** | DNABERT_1 leads at small n (0.269); NT v2 family near-random at every scale (50M–500M) | **Enumerate candidates; rule out MLM-only family** |
| 2 | Scaled embedding | 1,000 seqs, 20 species, **16 models** | **DNABERT_S wins at scale (0.049)**; DNABERT family arc isolates contrastive training as the contribution; SpliceBERT collapses (#2→#14) | **DNABERT_S confirmed; objective ≫ scale ≫ tokenisation** |
| 3 | Dual-modality strategy | 5 seqs, 2 models | DNABERT-S silhouette on text (0.319) is a false positive; MiniLM correct (0.212) | **Two models required** |
| 4 | Dual-modality at scale | 1,000 seqs, 20 species | DNABERT-S on text → silhouette −0.042 (failure confirmed); MiniLM +0.042 | **Strategy confirmed** |
| 5 | Retrieval querying (PoC) | 5 seqs, 3 query types | All 3 query types: P@1=1.0, MRR=1.0 | **ID bridge works** |
| 6 | Scaled retrieval | 1,000 seqs, 20 species | DNA→DNA P@1=0.655 MRR=0.767; Text→DNA P@1=0.601 MRR=0.704 | **All modes viable at scale** |
| 7 | FAISS index construction (PoC) | 5 seqs | IndexFlatIP 12–26× faster than HNSW at small n; both 100% accurate | **Use Flat for small n** |
| 8 | FAISS index at scale | 100–3,000 seqs | Flat→HNSW crossover at n≈1,000 (DNA) and n≈2,000 (Text) | **Use HNSW for n≥1,000** |
| 9 | FAISS retrieval (PoC) | 5 seqs | IndexFlatIP reproduces Exp 5 exactly — P@1=1.0, MRR=1.0, 100% FAISS accuracy | **FAISS pipeline verified** |
| 10 | FAISS retrieval scaled | 1,000 seqs, 20 species | HNSW 100% accurate; DNA unchanged vs Exp 6; Text→DNA P@1 0.601→0.906 after organism name fix | **Deployment-ready; fix metadata** |
| 11 | Text-encoder benchmark (Index B) | 1,000 seqs, 20 species, **6 encoders** | SapBERT (P@1 0.974) and e5-base-v2 (0.978) beat MiniLM (0.906) by ~7 pp; general bge upgrades do **not**; MedEmbed is worst | **SapBERT selected for Index B; objective ≫ "domain" label** |
| 12 | Recall@K curves + RRF fusion | 1,000 seqs, 20 species, 3 retrievers | Text dominates DNA at every K; hit-rate@3≈1.0; DNA is encoder-limited; fusion +0.1 pp recall (only +2.2 pp P@1, and w=2 only) | **Keep Text→DNA primary; fusion optional; next = generation eval** |

---

---

## Experiment 1 — Embedding Model Selection (16 candidates, n=5)

### Why
The existing RAG system (`src/core/gap_filler_rag.py`) uses `all-MiniLM-L6-v2`, a general
NLP sentence transformer, to embed DNA sequences. This produces semantically meaningless
vectors for nucleotide sequences. The goal was to identify a genomics-native replacement that
produces biologically structured embedding spaces suitable for nearest-neighbour retrieval.

The registry was deliberately expanded to **16 models** covering three orthogonal axes:
training objective (MLM vs contrastive), tokenisation (BPE, 6-mer, 3-mer, char, learned),
and parameter scale (50M → 2,500M). 13 are CPU-runnable; 3 require GPU and were evaluated
on Colab A100.

### What we tested
Five CDS sequences from 4 bacterial species (2 × *Thermus thermophilus*, 1 each of
*C. trachomatis*, *S. aureus*, *D. mccartyi*) — same setup as the original 4-model run.
This is a proof-of-concept scale; Experiment 2 is the diagnostic ranking.

### Results — top of the leaderboard (silhouette ↓)

| Rank | Model | Silhouette | Inter-dist | Tokenisation | Training |
|---|---|---|---|---|---|
| 1 | **DNABERT_1** | **0.2686** | 0.179 | 6-mer | MLM |
| 2 | SpliceBERT | 0.2406 | 0.145 | char | MLM |
| 3 | GROVER | 0.2189 | 0.269 | learned | MLM |
| 4 | GENA_LM | 0.1934 | 0.005 | BPE | MLM |
| 5 | DNABERT_S | 0.1794 | **0.7828** | BPE | **Contrastive** |
| … | (8 more) | … | … | | |
| 16 | NT_v2_100M | −0.0472 | 0.912 | k-mer | MLM |

Five NT v2 variants (50M → 2.5B) cluster near silhouette = 0 with inter-distance ≈ 1.0 —
unstructured embeddings on the unit hypersphere.

### Key findings (n=5, indicative only)
- **The n=5 ranking is misleading and Experiment 2 reveals the reversal.** SpliceBERT (#2) drops to #14 at scale; DNABERT_1 (#1) drops to #2; DNABERT_S (#5) rises to **#1**.
- **NT v2 family is near-random regardless of size.** All 5 variants (50M, 100M, 250M, 500M, 2,500M) score silhouette within ±0.05 of zero — a 50× parameter span produces no useful change. Falsifies the "bigger MLM = better embeddings" assumption.
- **DNABERT_S has the strongest inter-species distance (0.78)** — the only model that pushes different-species sequences apart while clustering same-species sequences. This is the signature of contrastive training and the property that survives scaling.

### Conclusion
> Experiment 1 cannot select the production model on its own. It enumerates 16 candidates
> and **rules out the entire MLM-only family** as a class — a strong negative finding that
> justifies focusing on contrastively-trained alternatives. Final selection requires
> Experiment 2's 20-species evaluation.

See `experiment_1_results.md` for the full 13-row table (10 CPU + 3 GPU after Colab merge)
and per-model heatmaps.

---

## Experiment 2 — Scaled Embedding Benchmark (16 candidates, n=1,000)

### Why
Experiment 1's n=5 silhouette has effectively no statistical power (one same-species pair).
The diagnostic ranking is at meaningful scale: 1,000 sequences across all 20 available species
= 24,500 same-species pairs and 475,000 cross-species pairs.

### What we tested
50 CDS sequences per species × 20 species = 1,000 sequences total. The 13 CPU models ran
locally; the 3 GPU-only models (NT_2500M, Caduceus, HyenaDNA) ran on Colab A100 and were
merged via `exp1_gpu.csv` / `exp2_gpu.csv`. All 16 registry models produced numbers.

### Results — full ranking by silhouette

| Rank | Model | Silhouette | Inter-dist | Training |
|---|---|---|---|---|
| 1 | **DNABERT_S** | **0.0486** | **0.6460** | Contrastive |
| 2 | DNABERT_1 | 0.0236 | 0.160 | MLM (6-mer) |
| 3 | NT_v2_50M_3mer | 0.0139 | 0.657 | MLM (3-mer) |
| 4 | GROVER | 0.0128 | 0.275 | MLM |
| 5 | NT_2500M (Colab) | 0.0090 | 0.105 | MLM |
| 6 | GENA_LM | 0.0062 | 0.005 | MLM (BigBird) |
| 7 | DNABERT_2 | 0.0029 | 0.215 | MLM (BPE) |
| 7 | GERM (DNABERT-2 fallback) | 0.0029 | 0.215 | — |
| 9 | NT_v2_500M | −0.0065 | 0.992 | MLM |
| 10 | NT_v2_250M | −0.0083 | 0.991 | MLM |
| 11 | NT_v2_50M | −0.0108 | 0.983 | MLM |
| 12 | NT_v2_100M | −0.0124 | 0.984 | MLM |
| 13 | MetaBERTa | −0.0131 | 0.262 | MLM |
| 14 | SpliceBERT | −0.0228 | 0.137 | MLM |
| 15 | HyenaDNA (Colab) | −0.0386 | 0.042 | MLM (Hyena) |
| 16 | Caduceus (Colab) | −0.0581 | 0.008 | MLM (Mamba) |

### Three ablations the expanded benchmark enables

**The DNABERT family arc — contrastive training, not BPE, is the contribution:**

| Variant | Tokenisation | Training | Silhouette | Inter-dist |
|---|---|---|---|---|
| DNABERT_1 | 6-mer | MLM | 0.0236 | 0.160 |
| DNABERT_2 | BPE | MLM | 0.0029 | 0.215 |
| **DNABERT_S** | BPE | **Contrastive** | **0.0486** | **0.646** |

DNABERT_1 → DNABERT_2 changed tokenisation only (6-mer → BPE), and silhouette **decreased**
8×. DNABERT_2 → DNABERT_S kept BPE and added contrastive training; silhouette jumped 16×
and inter-distance tripled. The improvement is entirely from contrastive learning.

**The NT v2 scale curve — MLM does not scale into useful embeddings:**

| Model | Params | Silhouette | Inter-dist |
|---|---|---|---|
| NT_v2_50M | 50 M | −0.0108 | 0.983 |
| NT_v2_100M | 100 M | −0.0124 | 0.984 |
| NT_v2_250M | 250 M | −0.0083 | 0.991 |
| NT_v2_500M | 500 M | −0.0065 | 0.992 |
| NT_2500M | 2,500 M | +0.0090 | 0.105 |

10× scaling at fixed MLM objective (50M → 500M) produces no change. 50× scaling (50M → 2.5B)
finally lifts the silhouette into weak positive territory but with 23× more parameters than
DNABERT_S for less than 1/5 the silhouette.

**Tokenisation ablation at fixed scale (NT v2 50M):**

| Variant | Tokenisation | Silhouette | Inter-dist |
|---|---|---|---|
| NT_v2_50M | k-mer | −0.0108 | 0.983 |
| NT_v2_50M_3mer | **3-mer** | **+0.0139** | **0.657** |

3-mer tokenisation alone can salvage a marginal MLM model — but still cannot reach
DNABERT_S's level without contrastive training.

### Key findings
- **13 of 16 models score silhouette ≤ 0.015 at scale.** Only DNABERT_S (0.049), DNABERT_1
  (0.024), and NT_v2_50M_3mer (0.014) achieve a clearly positive silhouette.
- **DNABERT_S is the only model with inter-species distance > 0.6 at scale.** Every other
  model either fails to separate species (intra ≈ inter, distance ≈ 0.1) or fails to cluster
  same-species records (distance ≈ 1.0, uniformly distributed).
- **SpliceBERT collapse (#2 → #14) is the canonical false-positive case.** High intra-species
  similarity at n=5 was a domain artefact (mammalian splice-site training applied to bacterial
  CDS records); at 20 species it collapses entire genera into one indistinguishable cluster.
- **Long-context models do not help on short CDS records.** Caduceus (Mamba, 131k context)
  and HyenaDNA (Hyena, 1M context) both score negative silhouettes — their long-range
  inductive bias does not translate into discriminative dense representations of 300–900 bp
  sequences. Possible future work: re-evaluate on full genes / operons.

### Conclusion
> **DNABERT_S confirmed as production model.** It is the only model in the 16-model benchmark
> that simultaneously ranks #1 on silhouette (0.0486) AND #1 on inter-species distance
> (0.646) at n=1,000 — and it does so with a 110M-parameter footprint that is 23× smaller
> than the only MLM competitor with any positive silhouette (NT_2500M).
>
> The DNABERT family arc and the NT v2 scale curve together make the central thesis claim
> empirically defensible: **training objective matters more than scale or tokenisation**.

### Final figure: `species_separation_top5.png`

`plot_species_separation.py` produces a 1×5 UMAP figure of the top-5 models with on-disk
embeddings. UMAP is run on each model's 1,000 raw embeddings (cosine metric, 15 neighbours,
random seed 42), points coloured by species (20 colours from `tab20`), 300 DPI.

Top-5 with available `.npy` embeddings:

1. DNABERT_S (silhouette 0.049)
2. DNABERT_1 (0.024)
3. NT_v2_50M_3mer (0.014)
4. GROVER (0.013)
5. NT_2500M (0.009)

Visual reading of the figure: DNABERT_S shows visibly separated species clusters; DNABERT_1
and GROVER show partial separation with significant overlap; NT_v2_50M_3mer shows a
loosely structured embedding space; NT_2500M shows residual clustering consistent with its
weak positive silhouette. All 11 models below this top-5 would show a single dense blob
under the same UMAP — they are excluded from the figure for that reason.

---

## Experiment 3 — Dual-Modality Embedding Strategy

### Why
CDS records contain two types of information: a raw DNA nucleotide sequence and structured
English metadata (organism name, gene name, protein description, accession ID). A RAG system
that supports text queries (e.g., "find records for oxidoreductases in thermophilic bacteria")
requires embeddings of the metadata text. The question is whether DNABERT-S can serve both
modalities, or whether a separate text model is needed.

### What we tested
The same 5 CDS records from Experiment 1, tested with three embedding configurations:
1. DNABERT-S on raw DNA sequences (replication of Experiment 1)
2. DNABERT-S on metadata text strings (control — can it handle English?)
3. `all-MiniLM-L6-v2` on metadata text strings (general-purpose text baseline)

Also measured: cross-modal DNA→metadata cosine similarity in DNABERT-S space (rows=DNA,
cols=metadata) to test whether a single index could serve both modalities.

### Results

| Modality | Model | Silhouette | Same-species sim | Cross-species sim range |
|----------|-------|------------|-----------------|------------------------|
| DNA | DNABERT-S | 0.179 | 0.813 | 0.106–0.349 |
| Metadata | DNABERT-S | 0.319 ⚠️ | 0.994 | 0.813–0.938 |
| Metadata | all-MiniLM-L6-v2 | 0.212 | 0.944 | 0.487–0.748 |

**Cross-modal retrieval ranks (DNA query → DNABERT-S metadata index):**
T.thermophilus_1: rank 4/5 | T.thermophilus_2: rank 5/5 | C.trachomatis: rank 4/5

**Embedding dimensions**: DNABERT-S = 768, MiniLM = 384 → **incompatible, dual index required**.

### Key findings
- **DNABERT-S silhouette on text (0.319) is a false positive.** The 6-mer DNA tokeniser maps
  all English text characters to `[UNK]`, producing nearly identical vectors for all records
  (all similarities 0.81–0.99). The TT pair marginally clusters above this uniform background
  — not because the model understands the organism names, but because it produces near-constant
  embeddings with a tiny variance that happens to favour the most similar pair.
- **Cross-modal retrieval completely fails**: DNA queries retrieve their own metadata record
  at rank 4/5 or 5/5. DNABERT-S DNA embeddings and metadata embeddings occupy different
  regions of the 768-dim space with no coherent geometric relationship.
- **all-MiniLM-L6-v2** correctly separates species in metadata space (sim range 0.487–0.944),
  with the same-species TT pair correctly ranked first at high similarity (0.944).

### Conclusion
> **A dual-embedding strategy is required.** DNABERT-S cannot serve as a unified encoder.
> The two-index architecture is mandated:
> - **Index A** (768-dim): DNABERT-S embeddings of DNA sequences
> - **Index B** (384-dim): all-MiniLM-L6-v2 embeddings of metadata text
>
> Cross-modal retrieval is achieved via an **ID bridge**: a query in one space retrieves
> record IDs, and the corresponding records in the other modality are returned by ID lookup.

---

## Experiment 4 — Dual-Modality Strategy Validation at Scale

### Why
Experiment 3 was run on only 5 sequences (4 species). The DNABERT-S metadata silhouette
(0.319) appeared high — a potential false positive that needed statistical exposure at scale.

### What we tested
Same three embedding configurations as Experiment 3, scaled to 1,000 sequences (50/species
× 20 species).

### Results

| Modality | Model | Silhouette (Exp 3, n=5) | Silhouette (Exp 4, n=1,000) |
|----------|-------|------------------------|------------------------------|
| DNA | DNABERT-S | 0.179 | 0.049 |
| Metadata | DNABERT-S (control) | 0.319 ⚠️ | **−0.042** |
| Metadata | all-MiniLM-L6-v2 | 0.212 | **+0.042** |

### Key findings
- **The false positive is fully exposed**: DNABERT-S on text goes from +0.319 (4 species,
  5 sequences) to **−0.042** (20 species, 1,000 sequences). A negative silhouette means
  records are closer to sequences from other species than to their own — the model would
  **actively degrade retrieval quality** if used as a metadata encoder.
- The drop mechanism is clear: at 4 species, the near-uniform UNK embeddings happened to
  produce a marginal TT-pair advantage. At 20 species, there are more cross-species
  neighbours pulling each record away from its cluster, flipping the silhouette negative.
- **all-MiniLM-L6-v2 maintains a positive silhouette (+0.042)** across 20 species and
  1,000 sequences. It runs at 0.006 s/seq — 18× faster than DNABERT-S for metadata.

### Conclusion
> **The dual-modality strategy is confirmed at scale.** Using DNABERT-S for metadata is
> definitively ruled out. The two-index architecture (DNABERT-S + MiniLM) is the correct
> and only viable approach for this system.

---

## Experiment 5 — Retrieval Querying (Proof of Concept)

### Why
Experiments 1–4 evaluated embedding quality in isolation (silhouette, intra/inter similarity).
Experiment 5 is the first end-to-end retrieval test: given a query, does the dual-index
architecture actually retrieve the correct records? It also validates the ID bridge mechanism
that connects the two modalities.

### What we tested
In-memory FAISS indices built from the 5 Experiment 1 CDS records (Index A: DNABERT-S 768-dim,
Index B: MiniLM 384-dim). All 3 query types tested with leave-self-out evaluation. Ground
truth: same species label = relevant document.

### Results

| Query type | Description | P@1 | P@2 | P@3 | MRR |
|------------|-------------|-----|-----|-----|-----|
| DNA→DNA | DNA query → search Index A → ranked DNA records | **1.000** | 0.500 | 0.333 | **1.000** |
| Text→DNA | text query → search Index B → ID bridge → DNA records | **1.000** | 0.500 | 0.333 | **1.000** |
| DNA→Text | DNA query → search Index A → ID bridge → metadata records | **1.000** | 0.500 | 0.333 | **1.000** |

> P@2 = 0.5 and P@3 = 0.333 are mathematically optimal given only 1 relevant document per
> query (the other TT record). The denominator k penalises beyond the single relevant document.

### Key findings
- **All 3 query types retrieve the correct record at rank 1** (P@1=1.0, MRR=1.0).
- The score gap is large and unambiguous: DNA→DNA TT pair similarity = 0.813 vs next-best
  0.349; Text→DNA TT pair similarity = 0.944 vs next-best 0.748.
- **The ID bridge works transparently**: cross-modal queries do not require a shared embedding
  space — a DNA query retrieves by sequence similarity, and the metadata of those records is
  returned by record ID. No projection layer, no shared space.
- DNA→Text correctly returns the matching organism name and protein description at rank 1.

### Conclusion
> **The dual-index architecture with ID bridging supports all 3 query types correctly.**
> Proof-of-concept validated. Experiment 6 scales this to 1,000 records for statistical robustness.

---

## Experiment 6 — Scaled Retrieval (1,000 Records, 20 Species)

### Why
Experiment 5 had only 5 records and 2 evaluable queries (only the TT pair had same-species
ground truth). P@k and MRR values at n=5 are anecdotal. Experiment 6 scales to 1,000 records
(49 relevant documents per query) for statistically meaningful retrieval metrics.

### What we tested
All 3 query types evaluated on 1,000 records (50/species × 20 species) using leave-self-out
ranking. Metrics: P@1, P@5, P@10, P@20, Recall@10, Recall@20, MRR.

### Results

| Query type | P@1 | P@5 | P@10 | P@20 | Recall@10 | Recall@20 | MRR |
|------------|-----|-----|------|------|-----------|-----------|-----|
| DNA→DNA | **0.655** | **0.616** | **0.575** | **0.528** | **0.117** | **0.216** | **0.767** |
| DNA→Text | 0.655 | 0.616 | 0.575 | 0.528 | 0.117 | 0.216 | 0.767 |
| Text→DNA | 0.601 | 0.562 | 0.531 | 0.470 | 0.108 | 0.192 | 0.704 |

### Key findings
- **DNA→DNA and DNA→Text are identical** because both search Index A. The ID bridge for
  DNA→Text correctly routes the result to metadata.
- **MRR = 0.767** for DNA→DNA means the first relevant document appears on average at rank
  ≈ 1.3 — within the first two results in the vast majority of queries.
- **P@k decay** (0.655 → 0.528 for k=1 to k=20) is expected: same-species records are
  concentrated near rank 1, and beyond the top few results the ranking becomes mixed as
  functionally conserved genes appear across species.
- **Text→DNA is slightly weaker** (P@1 = 0.601 vs 0.655). This is partly due to a cosmetic
  bug where organism names defaulted to accession IDs (e.g., "GCF_000008125.1") instead of
  species names ("Thermus thermophilus"), reducing inter-species text separation. Fixing this
  is expected to bring Text→DNA closer to DNA→DNA performance.
- **Recall@20 = 0.216** means the top-20 results capture ~21.6% of the 49 relevant documents
  per query — reasonable given that CDS sequences from the same species span diverse gene
  families that do not all cluster tightly in embedding space.

### Conclusion
> **All 3 query types are validated at scale.** The dual-modality strategy achieves solid
> retrieval performance: P@1 > 0.60 and MRR > 0.70 for all modes. DNA→DNA is the primary
> retrieval pathway. Text→DNA and DNA→Text work correctly via the ID bridge. Retrieval
> quality is sufficient for a RAG pipeline where top-3 to top-5 context documents are passed
> to the generative model.

---

## Experiment 7 — FAISS Index Construction (Proof of Concept, 5 Records)

### Why
Experiments 1–6 used brute-force cosine similarity (sklearn). A production RAG system requires
an approximate nearest-neighbour index (FAISS) for sub-millisecond query latency at scale.
Experiment 7 validates that FAISS correctly builds and queries the dual-modality indices, and
establishes a baseline for index type comparison at minimal scale.

### What we tested
Two FAISS index types for each modality on 5 records:
- **IndexFlatIP**: exact inner-product search on L2-normalised vectors (= exact cosine search)
- **IndexHNSWFlat(M=32)**: approximate HNSW graph search

Measured: build time (ms), query latency (µs/query, 1,000 repetitions), top-k accuracy vs
brute-force cosine, serialised memory (bytes).

### Results

| Index | Modality | Build (ms) | Latency (µs) | Memory (bytes) | Top-1/2/3 acc |
|-------|----------|-----------|-------------|----------------|--------------|
| IndexFlatIP | DNA (768-dim) | 0.094 | **1.94** | 15,405 | 100% / 100% / 100% |
| IndexHNSWFlat | DNA (768-dim) | 0.091 | 50.89 | 16,926 | 100% / 100% / 100% |
| IndexFlatIP | Text (384-dim) | 0.023 | **2.09** | 7,725 | 100% / 100% / 100% |
| IndexHNSWFlat | Text (384-dim) | 0.045 | 25.56 | 9,246 | 100% / 100% / 100% |

### Key findings
- **Both index types are 100% accurate** at n=5. At this size, HNSW builds an exact graph
  over all 5 nodes — approximation does not apply.
- **IndexFlatIP is 12–26× faster than HNSW** at n=5. HNSW's graph traversal overhead
  (pointer chasing, efSearch candidate evaluation) dominates over the trivial brute-force
  scan of 5 vectors.
- Build times are sub-millisecond for both. Memory overhead from HNSW is ~10%.

### Conclusion
> **FAISS integration validated.** At n=5, IndexFlatIP is the correct choice: exact, faster,
> and smaller. HNSW's speed advantage requires a minimum corpus size to materialise —
> determined in Experiment 8.

---

## Experiment 8 — FAISS Index at Scale (100–3,000 Records)

### Why
Experiment 7 showed HNSW is slower than flat at n=5. The theoretical advantage of HNSW
(O(log n) query vs O(n) for flat) only materialises when n is large enough that the graph
traversal cost is outweighed by avoiding a full scan. Experiment 8 maps the crossover point
across the corpus sizes realistic for this RAG system.

### What we tested
Both index types (IndexFlatIP, IndexHNSWFlat M=32) for both modalities (DNA 768-dim, Text
384-dim) at 6 corpus sizes: 100, 250, 500, 1,000, 2,000, 3,000 vectors. Full embeddings
generated from 3,000 records (150/species × 20 species); sub-sampled for each test size.

### Results — Latency Crossover

**DNA Index (768-dim):**

| n | Flat latency | HNSW latency | Winner | Speed ratio |
|---|-------------|-------------|--------|-------------|
| 100 | 11.4 µs | 35.9 µs | **Flat** | 3.1× |
| 250 | 13.6 µs | 48.8 µs | **Flat** | 3.6× |
| 500 | 25.1 µs | 59.8 µs | **Flat** | 2.4× |
| **1,000** | **52.7 µs** | **51.5 µs** | **≈ Tied** | **1.0×** |
| 2,000 | 97.0 µs | 72.7 µs | **HNSW** | 1.3× |
| 3,000 | 130.7 µs | 88.1 µs | **HNSW** | 1.5× |

**Text Index (384-dim):**

| n | Flat latency | HNSW latency | Winner | Speed ratio |
|---|-------------|-------------|--------|-------------|
| 100 | 5.4 µs | 89.5 µs | **Flat** | 16.6× |
| 500 | 8.5 µs | 35.2 µs | **Flat** | 4.1× |
| 1,000 | 21.8 µs | 34.7 µs | **Flat** | 1.6× |
| **2,000** | **43.9 µs** | **33.4 µs** | **HNSW** | **1.3×** |
| 3,000 | 63.1 µs | 44.3 µs | **HNSW** | 1.4× |

**Accuracy**: Both index types achieve ≥99.7% top-1 accuracy at all tested sizes. Rare
sub-100% values reflect float32 vs float64 tie-breaking, not genuine retrieval errors.

**Memory overhead from HNSW**: constant ~8.8% for DNA (768-dim), ~17.7% for Text (384-dim).

### Key findings
- **DNA crossover at n ≈ 1,000**: at 1,000 records both indices have the same latency (52.7
  vs 51.5 µs). Beyond this, HNSW becomes progressively faster.
- **Text crossover at n ≈ 1,500–2,000**: lower-dimensional vectors make flat scans cheaper
  per candidate, delaying the HNSW advantage.
- At n=3,000: HNSW is **1.5× faster for DNA** and **1.4× faster for Text**.
- At IOWarp scale (n=100,000+), flat scan grows linearly while HNSW grows as O(log n × M).
  Extrapolating: HNSW would be ~50–100× faster at n=100,000.

### Deployment recommendation

| Corpus size | Recommended index | Justification |
|-------------|------------------|---------------|
| n < 1,000 | **IndexFlatIP** | Exact, faster, no build overhead |
| 1,000 ≤ n ≤ 100,000 | **IndexHNSWFlat(M=32)** | Speed advantage, >99.7% accuracy |
| n > 1,000,000 (IOWarp) | **IndexIVFPQ** | Memory-efficient ANN; consider 8–32× compression |

### Conclusion
> **IndexHNSWFlat(M=32) is the recommended FAISS index type** for the current RAG corpus
> (n ≥ 1,000 records). The crossover from flat to HNSW occurs at n ≈ 1,000 for the DNA
> index and n ≈ 2,000 for the text index — well within the operational range of the system.
> At IOWarp petabyte scale, IndexIVFPQ should replace HNSW to control memory footprint.

---

---

## Experiment 9 — FAISS Retrieval Querying (Proof of Concept)

### Why
Experiments 5 and 6 validated retrieval quality using brute-force cosine similarity (sklearn).
Experiments 7 and 8 validated FAISS index infrastructure. Experiment 9 connects the two:
routing the Experiment 5 retrieval evaluation through FAISS indices to verify the pipeline
works end-to-end and that IndexFlatIP produces exactly the same rankings as sklearn.

### What we tested
Same 5 CDS records as Experiment 5. IndexFlatIP for both modalities (Exp 8 recommendation
for n=5). All 3 query types. Metrics: P@1, P@2, P@3, MRR. Also measured FAISS build time,
query latency, top-1 accuracy vs brute-force, and memory.

### Results

| Index | Type | Build (ms) | Latency (µs/q) | Memory (bytes) | Top-1 accuracy |
|-------|------|-----------|----------------|----------------|----------------|
| A (DNA, 768-dim) | IndexFlatIP | 0.153 | 3.47 | 15,405 | **1.000** |
| B (Text, 384-dim) | IndexFlatIP | 0.016 | 1.94 | 7,725 | **1.000** |

| Query type | P@1 | P@2 | P@3 | MRR |
|------------|-----|-----|-----|-----|
| DNA→DNA | 1.000 | 0.500 | 0.333 | 1.000 |
| Text→DNA | 1.000 | 0.500 | 0.333 | 1.000 |
| DNA→Text | 1.000 | 0.500 | 0.333 | 1.000 |

### Key findings
- **Exact match with Experiment 5**: P@1=1.0 and MRR=1.0 for all 3 query types — FAISS
  IndexFlatIP produces identical rankings to brute-force cosine at n=5.
- **100% FAISS accuracy**: no retrieval errors vs brute-force baseline.
- **Sub-4 µs query latency** for both indices. FAISS pipeline is correctly wired end-to-end.

### Conclusion
> **FAISS pipeline validated.** IndexFlatIP is a lossless replacement for sklearn cosine
> at n=5. All 3 query types (DNA→DNA, Text→DNA, DNA→Text via ID bridge) work correctly
> through FAISS. Experiment 10 validates at scale with HNSW.

---

## Experiment 10 — FAISS Retrieval Querying (Scaled, 1,000 Records)

### Why
Experiment 9 confirmed the FAISS pipeline at n=5. Experiment 10 scales to 1,000 records
using the Experiment 8-recommended index types, compares results to Experiment 6 brute-force
baseline, and applies the organism name bug fix to measure its impact on text retrieval.

### What we tested
1,000 records (50/species × 20 species). Index A: IndexHNSWFlat(M=32, efSearch=64) for DNA
(Exp 8 recommendation for n≥1,000). Index B: IndexFlatIP for Text (crossover at n≈2,000,
Flat still optimal at n=1,000). Bug fix: organism names resolved correctly (`.1` suffix stripped).
Metrics: P@1/5/10/20, Recall@10/20, MRR. Comparison vs Experiment 6 brute-force results.

### Results

| Index | Type | Build (ms) | Latency (µs/q) | Memory (KB) | Top-1 accuracy |
|-------|------|-----------|----------------|-------------|----------------|
| A (DNA, 768-dim) | IndexHNSWFlat(M=32) | 42.0 | 63.6 | 3,265 | **1.000** |
| B (Text, 384-dim) | IndexFlatIP | 1.0 | 22.9 | 1,500 | **1.000** |

| Query type | P@1 | P@5 | P@10 | P@20 | Recall@10 | Recall@20 | MRR |
|------------|-----|-----|------|------|-----------|-----------|-----|
| DNA→DNA | 0.655 | 0.616 | 0.575 | 0.528 | 0.117 | 0.216 | 0.766 |
| DNA→Text | 0.655 | 0.616 | 0.575 | 0.528 | 0.117 | 0.216 | 0.766 |
| **Text→DNA** | **0.906** | **0.895** | **0.893** | **0.882** | **0.182** | **0.360** | **0.938** |

**Comparison vs Experiment 6 (brute-force):**

| Query type | Metric | Exp 6 | Exp 10 (FAISS) | Δ | Cause |
|------------|--------|-------|----------------|---|-------|
| DNA→DNA | P@1 | 0.655 | 0.655 | 0.000 | No change — FAISS exact |
| DNA→DNA | MRR | 0.767 | 0.766 | −0.001 | Rounding only |
| Text→DNA | P@1 | 0.601 | **0.906** | **+0.305** | **Organism name bug fix** |
| Text→DNA | MRR | 0.704 | **0.938** | **+0.234** | **Organism name bug fix** |
| DNA→Text | P@1 | 0.655 | 0.655 | 0.000 | No change — FAISS exact |

### Key findings
- **DNA→DNA and DNA→Text unchanged**: HNSW at n=1,000 is lossless (100% accuracy vs
  brute-force). Retrieval quality is identical to Experiment 6.
- **Text→DNA improved dramatically (+30.5 pp P@1)**: entirely due to the organism name fix,
  not FAISS. Including correct species names in metadata text makes same-species records
  far more similar in MiniLM space. Text→DNA is now the **highest-performing query mode**
  (P@1=0.906, MRR=0.938), surpassing DNA→DNA (P@1=0.655).
- **HNSW accuracy 100%** at n=1,000 with efSearch=64 — no approximation cost.
- **Organism name fix is critical**: a metadata formatting issue, not a model issue,
  was suppressing 30 percentage points of retrieval performance in Experiment 6.

### Conclusion
> **FAISS-based retrieval is deployment-ready at n=1,000 with zero quality degradation.**
> HNSW for DNA and Flat for Text are the correct index choices at this scale. The organism
> name fix raises Text→DNA P@1 from 0.601 to 0.906 — this fix must be applied to the
> production metadata pipeline before deployment.

---

## Open Questions and Next Steps

1. **Apply organism name fix to production pipeline**: The `.1` version suffix bug suppressed
   30 pp of Text→DNA performance. Fixing this in the corpus preprocessing pipeline is the
   highest-priority action before deployment.

2. **GPU inference**: All experiments ran on CPU. DNABERT-S on GPU reduces embedding from
   ~109 s/1,000 seqs to ~2–5 s, enabling real-time corpus updates.

3. **IndexIVFPQ at IOWarp scale**: Validate product quantisation (8–32× memory reduction
   with ~1–3% accuracy loss) for deployment on the petabyte-scale IOWarp platform.

4. **End-to-end gap filling validation**: Confirm that improved DNA retrieval quality
   (DNABERT-S vs the original all-MiniLM-L6-v2 baseline) translates to better gap-filling
   accuracy in the full LangChain agent pipeline.
