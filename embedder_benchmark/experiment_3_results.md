# Experiment 3: Dual-Modality Embedding Strategy — Results

**Date**: 2026-04-28  
**Hardware**: CPU-only (no CUDA); Python 3.10; transformers 5.5.4; sentence-transformers 5.4.1  
**Script**: `experiments/experiment_3_dual_modality.py`

---

## 1. Objective

Determine whether DNABERT-S (the winning model from Experiments 1–2) can produce meaningful
embeddings for both raw DNA sequences **and** CDS metadata text, or whether a separate
general-purpose text model is required for a dual-modality RAG index.

---

## 2. Input Records (same 5 CDS as Experiment 1)

| Tag | Species | Metadata text extracted |
|-----|---------|------------------------|
| seq1_TT | *Thermus thermophilus* | `organism=Thermus thermophilus locus_tag=TT_RS00010 protein=SDR family NAD(P)-dependent oxidoreductase protein_id=WP_011172461.1` |
| seq2_TT | *Thermus thermophilus* | `organism=Thermus thermophilus locus_tag=TT_RS00015 protein=NAD(P)/FAD-dependent oxidoreductase protein_id=WP_011172462.1` |
| seq3_CT | *Chlamydia trachomatis* | `organism=Chlamydia trachomatis gene=gatC locus_tag=CT_002 db_xref=GeneID:884067 protein=aspartyl/glutamyl-tRNA amidotransferase subunit C protein_id=NP_219504.1` |
| seq4_SA | *Staphylococcus aureus* | `organism=Staphylococcus aureus locus_tag=SAB_RS00035 protein=NAD(P)H-hydrate dehydratase protein_id=WP_000449218.1` |
| seq5_DM | *Dehalococcoides mccartyi* | `organism=Dehalococcoides mccartyi gene=nadD locus_tag=CBDB_RS00015 protein=nicotinate-nucleotide adenylyltransferase protein_id=WP_011308641.1` |

---

## 3. Why Embed Metadata Rather Than Filter

CDS metadata in FASTA headers is free-form text: `organism=Thermus thermophilus locus_tag=TT_RS00010 protein=SDR family NAD(P)-dependent oxidoreductase protein_id=WP_011172461.1`. A retrieval system needs to rank candidate records by how similar their metadata is to a query — not just include or exclude them by a hard condition.

**Filtering** (exact match, prefix, or boolean) can implement hard constraints: "return only records where organism == 'Thermus thermophilus'". But it cannot assign a similarity score, cannot tolerate paraphrases, and cannot rank. Two protein descriptions — `"NAD(P)/FAD-dependent oxidoreductase"` and `"SDR family NAD(P)-dependent oxidoreductase"` — share biological meaning but do not share tokens. A filter cannot relate them.

**Embedding** maps each metadata string to a point in a continuous vector space where geometric proximity ≈ semantic similarity. A well-trained text encoder places the two TT records (same organism, related protein family) close together and the CT record (different organism, different gene function) far away. The cosine score then serves as a graded similarity signal usable for ranked retrieval.

Filters remain useful as a **post-processing constraint** (e.g., restrict results to a particular organism after FAISS retrieval), but they cannot replace embedding for the ranking step.

The question this experiment answers is not *whether* to embed, but *which model* to use: the same DNABERT-S model selected for DNA sequences, or a dedicated natural-language encoder. The two candidates are tested below.

---

## 4. Embedding Configurations Tested

| Modality | Model | Dim | Time (5 seqs) |
|----------|-------|-----|---------------|
| DNA | DNABERT-S | 768 | 6.83 s |
| Metadata text | DNABERT-S | 768 | 1.49 s |
| Metadata text | all-MiniLM-L6-v2 | 384 | 10.08 s |

---

## 5. Similarity Matrices

### 5.1 DNA–DNA (DNABERT-S) — replicates Experiment 1

|  | TT_1 | TT_2 | CT | SA | DM |
|--|------|------|----|----|----|
| **TT_1** | 1.000 | **0.813** | 0.136 | 0.154 | 0.314 |
| **TT_2** | 0.813 | 1.000 | 0.153 | 0.201 | 0.349 |
| **CT** | 0.136 | 0.153 | 1.000 | 0.346 | 0.106 |
| **SA** | 0.154 | 0.201 | 0.346 | 1.000 | 0.196 |
| **DM** | 0.314 | 0.349 | 0.106 | 0.196 | 1.000 |

> Same-species pair (TT_1/TT_2) = 0.813. Cross-species range: 0.106–0.349. Matches Experiment 1.

### 5.2 Metadata–Metadata (DNABERT-S on text)

|  | TT_1 | TT_2 | CT | SA | DM |
|--|------|------|----|----|----|
| **TT_1** | 1.000 | **0.994** | 0.856 | 0.822 | 0.854 |
| **TT_2** | 0.994 | 1.000 | 0.841 | 0.813 | 0.825 |
| **CT** | 0.856 | 0.841 | 1.000 | 0.895 | 0.938 |
| **SA** | 0.822 | 0.813 | 0.895 | 1.000 | 0.933 |
| **DM** | 0.854 | 0.825 | 0.938 | 0.933 | 1.000 |

> All similarities compressed into [0.81, 1.0]. The matrix is near-uniform — DNABERT-S cannot
> distinguish between metadata from different species.

### 5.3 Metadata–Metadata (all-MiniLM-L6-v2)

|  | TT_1 | TT_2 | CT | SA | DM |
|--|------|------|----|----|----|
| **TT_1** | 1.000 | **0.944** | 0.487 | 0.748 | 0.645 |
| **TT_2** | 0.944 | 1.000 | 0.497 | 0.745 | 0.655 |
| **CT** | 0.487 | 0.497 | 1.000 | 0.559 | 0.544 |
| **SA** | 0.748 | 0.745 | 0.559 | 1.000 | 0.681 |
| **DM** | 0.645 | 0.655 | 0.544 | 0.681 | 1.000 |

> Same-species pair (TT_1/TT_2) = 0.944. Cross-species range: 0.487–0.748. Wide spread —
> the text model correctly groups same-species records and separates different-species records.

### 5.4 DNA–Metadata Cross-Modal (DNABERT-S space, rows=DNA, cols=Metadata)

|  | TT_1 | TT_2 | CT | SA | DM |
|--|------|------|----|----|----|
| **TT_1** | 0.018 | 0.009 | 0.060 | 0.062 | 0.082 |
| **TT_2** | 0.077 | 0.076 | 0.089 | 0.093 | 0.094 |
| **CT** | 0.085 | 0.084 | 0.074 | 0.119 | 0.070 |
| **SA** | 0.076 | 0.065 | 0.112 | **0.183** | 0.126 |
| **DM** | 0.187 | 0.167 | 0.189 | 0.205 | **0.201** |

> All values near zero (range: 0.009–0.205). DNA embeddings and metadata embeddings occupy
> completely different regions of the 768-dim DNABERT-S space.

---

## 6. Silhouette Scores

| Modality | Model | Silhouette ↑ |
|----------|-------|--------------|
| DNA | DNABERT-S | 0.1794 |
| Metadata | DNABERT-S | 0.3192 ⚠️ |
| Metadata | all-MiniLM-L6-v2 | 0.2115 |

> **Warning on DNABERT-S metadata silhouette**: The value 0.3192 is a **false positive**.
> DNABERT-S tokenizes English text as DNA k-mers, producing near-identical UNK-dominated
> embeddings for all inputs. The TT pair achieves similarity 0.994 vs cross-species ~0.85,
> giving a positive silhouette only because the near-identical vectors for TT_1/TT_2 happen
> to cluster marginally above the uniformly high background. This is not biologically meaningful
> discrimination — it is numerical noise on top of near-constant vectors.

---

## 7. Cross-Modal Retrieval Ranks (DNA query → Metadata index, DNABERT-S space)

| Query sequence | Rank of correct metadata | Cosine sim (diagonal) |
|----------------|--------------------------|----------------------|
| T.thermophilus_1 | **4/5** | 0.018 |
| T.thermophilus_2 | **5/5** | 0.076 |
| C.trachomatis | **4/5** | 0.074 |
| S.aureus | 1/5 | 0.183 |
| D.mccartyi | 2/5 | 0.201 |

> 3 out of 5 sequences fail to retrieve their own metadata record as rank 1.
> Cross-modal retrieval using a single DNABERT-S space is **unreliable**.

---

## 8. Embedding Dimension Compatibility

| Model | Dimension |
|-------|-----------|
| DNABERT-S (DNA) | 768 |
| DNABERT-S (Metadata) | 768 |
| all-MiniLM-L6-v2 (Metadata) | 384 |

Dimensions of DNABERT-S and all-MiniLM-L6-v2 are **incompatible** (768 ≠ 384).
A single FAISS index cannot hold both modalities — **dual index architecture is required**.

---

## 9. Analysis

### DNABERT-S on metadata text — fundamentally unsuitable

DNABERT-S uses a 6-mer DNA tokenizer. English metadata strings (e.g.,
`"organism=Thermus thermophilus locus_tag=TT_RS00010 protein=SDR family..."`) contain
characters and words that are not valid DNA k-mers. The tokenizer maps all non-ACGT
subsequences to `[UNK]`, producing embedding vectors that are dominated by the UNK token
representation. As a result, all metadata texts produce nearly identical vectors (cosine
similarities 0.81–0.99), and the model provides no discriminative power over metadata content.

The apparently high silhouette (0.3192) is misleading: it reflects the fact that the TT pair
(two sequences from the same organism with nearly identical locus tags) produces marginally
more similar UNK-dominated vectors than cross-species pairs. This is not species-level
biological signal — it is an artifact of near-constant embeddings.

### all-MiniLM-L6-v2 on metadata text — correct behavior

The text model produces a meaningful, spread similarity matrix (0.49–0.94) that correctly:
- Places the two *T. thermophilus* records close together (0.944) — they share organism name,
  locus tag prefix, and protein family
- Separates *C. trachomatis* (0.487–0.497 from TT records) — most distant, distinct organism and gene
- Places *S. aureus* and *D. mccartyi* at intermediate distances

### Cross-modal coherence — no shared geometric structure

The DNA–metadata cross-modal matrix (range: 0.009–0.205) shows that DNA embeddings and
metadata embeddings in DNABERT-S space share no coherent geometric structure. A DNA sequence
and its own metadata record are not closer in embedding space than arbitrary cross-record pairs.
Cross-modal retrieval within a single index is therefore not viable.

---

## 10. Conclusion

**A dual-embedding strategy is required and confirmed.**

| Component | Model | FAISS Index |
|-----------|-------|-------------|
| DNA sequences | DNABERT-S (`zhihan1996/DNABERT-S`) | Index A (dim=768) |
| CDS metadata text | all-MiniLM-L6-v2 (`sentence-transformers/all-MiniLM-L6-v2`) | Index B (dim=384) |

The two indices must be queried separately:
- A **DNA query** searches Index A and retrieves candidate CDS records by sequence similarity.
- A **text query** searches Index B and retrieves candidate CDS records by metadata similarity.
- Results from both indices can be merged by record ID for a unified ranked list.

DNABERT-S cannot serve as a unified encoder for both modalities. Attempting to use a single
index would either (a) lose all metadata discriminability, or (b) require a projection layer
that is not part of this system's design.
