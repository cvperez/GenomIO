# The Bridge Architecture: Dual-Modality Retrieval for Genomic CDS Records

> Historical discussion: retained for its tables, glossary, and design rationale. The [benchmark overview](README.md) and experiment reports describe the current files and evidence limits. Claims about isolated training effects, tokenizer mechanisms, or untested scales are interpretations, not additional measurements. Metadata retrieval and fusion are benchmark implementations, not integrated application features.

**Defined by**: Experiment 3 (proof-of-concept, n=5) and Experiment 4 (scale validation, n=1,000 across 20 species)
**Used by**: Experiments 5, 6, 9, 10 (retrieval querying and FAISS deployment)
**Production schema**: `Index A` (DNABERT-S, DNA, 768-dim) + `Index B` (all-MiniLM-L6-v2, metadata text, 384-dim) connected by a shared integer **record ID**.

---

## 1. What problem does the bridge architecture solve?

Each CDS record in `rag_corpus_uniform/` carries **two distinct types of information**:

| Modality | Example content | Vocabulary |
|---|---|---|
| **DNA sequence** | `ATGAGCGCCGTGCTGGTGACCGGCGGCAGCCG...` (300–900 bp) | 4 nucleotides (A, C, G, T) |
| **Metadata text** | `organism=Thermus thermophilus locus_tag=TT_RS00010 protein=SDR family NAD(P)-dependent oxidoreductase` | Free-form English |

A RAG system over this corpus must support three distinct query modes:

1. **DNA→DNA** : given a DNA sequence, find structurally similar DNA records.
2. **Text→DNA** : given a metadata description ("oxidoreductase in *Thermus thermophilus*"), retrieve the DNA records most likely to match.
3. **DNA→Text** : given a DNA sequence, recover the metadata of the most similar records.

The naive design uses **one model + one index** for both modalities. Experiment 3 demonstrates that this design is broken in a non-obvious way; Experiment 4 confirms the breakage at 20-species scale. The bridge architecture is the fix.

---

## 2. Why a single-encoder index does not work

### 2.1 The false-positive at n=5 (Experiment 3)

Experiment 3 ran DNABERT-S as a unified encoder on both DNA and metadata text. The metadata-on-DNABERT-S silhouette was **0.319** : superficially the highest score in the experiment. Inspection of the cosine similarity matrix revealed why:

| Pair type | DNABERT-S on metadata text: cosine similarity |
|---|---|
| Same-species (TT/TT) | 0.994 |
| Cross-species | 0.81 – 0.94 |

All metadata strings collapse into a narrow region of vector space (similarities in [0.81, 1.0]). DNABERT-S tokenises the metadata text with its 6-mer DNA tokeniser, so almost every character maps to `[UNK]`. The result is a near-constant vector for every record. A two-record same-species pair happens to be slightly closer than the rest, which lifts the silhouette into the positive range, but the embeddings carry no actual semantic signal.

### 2.2 The collapse at n=1,000 (Experiment 4)

Experiment 4 ran the same comparison on 50 records × 20 species:

| Configuration | Silhouette (n=5) | Silhouette (n=1,000) |
|---|---|---|
| **DNA / DNABERT-S** | 0.179 | 0.049 |
| **Metadata / DNABERT-S** | 0.319 ← false positive | **−0.042** |
| **Metadata / all-MiniLM-L6-v2** | 0.212 | **+0.042** |

DNABERT-S on metadata flips from +0.32 to −0.04. The negative sign is critical: it means records are on average *closer* to metadata of other species than to their own species. Using DNABERT-S as a unified encoder would actively degrade retrieval quality below random.

all-MiniLM-L6-v2 (the dedicated natural-language sentence encoder) maintains a positive silhouette and produces a meaningful inter-species spread (0.487 – 0.748 cross-species vs 0.944 same-species at n=5).

### 2.3 Cross-modal coherence: DNA and metadata occupy disjoint regions

Experiment 3 also computed the DNA-vs-metadata cosine similarity matrix inside DNABERT-S space. All values fell in the range 0.009 – 0.205 : DNA and metadata embeddings are nearly orthogonal under DNABERT-S. A DNA query for record A retrieves record A's metadata at rank 4 out of 5 in three of five queries.

This rules out the alternative of training a shared projection: there is no coherent geometric structure linking the two modalities under a single encoder.

### 2.4 The constraint that forces two indices

Even if DNABERT-S could embed text usefully (it cannot), the two vector spaces have **incompatible dimensions**:

| Encoder | Output dimension |
|---|---|
| DNABERT-S | 768 |
| all-MiniLM-L6-v2 | 384 |

A single FAISS index requires a fixed dimension. Concatenation, padding, or projection would change the geometry. The cleanest solution is a **dual index**, and once you have two indices, the question of how to cross-modal-query becomes a routing question rather than a representation question.

---

## 3. The bridge architecture in one diagram

```
                            CDS record (one row in records[])
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
     [DNA sequence]                                            [Metadata text string]
            │                                                           │
    DNABERT-S encoder                                  all-MiniLM-L6-v2 encoder
    (contrastive, 768-dim)                            (general NL, 384-dim)
            │                                                           │
            ▼                                                           ▼
   ┌────────────────────────┐                          ┌────────────────────────┐
   │  Index A (FAISS, 768)  │                          │  Index B (FAISS, 384)  │
   │  L2-normalised cosine  │                          │  L2-normalised cosine  │
   │  HNSW(M=32) for n≥1k   │                          │  IndexFlatIP for n<2k  │
   │  Flat for smaller n    │                          │                        │
   └────────────────────────┘                          └────────────────────────┘
            │                                                           │
            └──────────────────────┐         ┌──────────────────────────┘
                                   ▼         ▼
                            ┌─────────────────────┐
                            │   ID BRIDGE         │  ← integer row index
                            │   records[id_k]     │     shared between A and B
                            └─────────────────────┘
                                   │         │
                       ┌───────────┘         └──────────┐
                       ▼                                ▼
              records[id_k].dna           records[id_k].metadata_text
```

The architecture has three components: two encoder-index pairs and a trivial lookup table. The two indices are independent : they have different dimensions, different models, different optimal FAISS index types, and no shared parameters. The only thing they share is the **row order** of the records list. That is the bridge.

---

## 4. The ID bridge: what it actually is

The bridge is **not a learned projection**, not a hash map, not a join key. It is the implicit fact that both indices are built by iterating `records[]` in the same order:

```python
# at index-build time
records = [load_record(p) for p in record_paths]   # canonical row order

dna_vectors  = np.stack([DNABERT_S(r.dna)          for r in records])
text_vectors = np.stack([MiniLM(r.metadata_text)   for r in records])

index_A = faiss.IndexHNSWFlat(768, 32)
index_A.add(l2_normalise(dna_vectors))             # row k in records → vector k in A

index_B = faiss.IndexFlatIP(384)
index_B.add(l2_normalise(text_vectors))            # row k in records → vector k in B
```

After this, the same integer `k` references the same record everywhere:
- `records[k]` is the canonical row
- `index_A.reconstruct(k)` is its DNA embedding
- `index_B.reconstruct(k)` is its metadata embedding

A FAISS query against either index returns row indices, not vectors. The bridge is just `records[k]`. No projection layer, no parameters, no training.

This is the entire mechanism. The simplicity is the point : there is nothing to break, nothing to retrain, and no parameter mismatch when one encoder is swapped.

---

## 5. Query routing: three modes

### 5.1 DNA → DNA  (Index A, no bridge needed)

Both query and payload are DNA. The bridge is unused.

```
query DNA   ──► DNABERT-S ──► 768-dim L2-normalised vector
                                       │
                                       ▼
                          index_A.search(q, k+1)        ← +1 for leave-self-out
                                       │
                       returns: [(score, id), ...]
                                       │
                                       ▼
                          records[id].dna_sequence       ← payload
```

### 5.2 Text → DNA  (Index B → bridge → DNA payload)

The query is a natural language description; the payload must be DNA. Index B does the ranking, the bridge fetches the DNA.

```
query text  ──► all-MiniLM-L6-v2 ──► 384-dim L2-normalised vector
                                       │
                                       ▼
                          index_B.search(q, k+1)
                                       │
                       returns: [(score, id), ...]
                                       │
                                       ▼ ID BRIDGE
                          records[id].dna_sequence       ← payload
```

The DNA sequences are never compared to the text query directly. Index B alone determines the ranking; the DNA is just the payload that gets returned by row index.

### 5.3 DNA → Text  (Index A → bridge → metadata payload)

The mirror of 5.2. Query is DNA; payload is metadata. Index A determines the ranking; the bridge fetches the metadata.

```
query DNA   ──► DNABERT-S ──► 768-dim L2-normalised vector
                                       │
                                       ▼
                          index_A.search(q, k+1)
                                       │
                       returns: [(score, id), ...]
                                       │
                                       ▼ ID BRIDGE
                          records[id].metadata_text      ← payload
```

Because Index A is queried in both DNA→DNA and DNA→Text, the rankings are **identical** for the two query types. Only the payload differs. Experiments 6 and 10 show this directly: every DNA→DNA metric equals the corresponding DNA→Text metric.

---

## 6. Why a single-space projection is not the alternative

A reasonable-looking alternative would be to train a projection that maps text embeddings into the DNA space (or vice versa) so a single index could serve both modalities. The bridge architecture deliberately rejects this design for three reasons:

1. **No training data for the alignment.** A projection that maps "Thermus thermophilus oxidoreductase" near the DNA embedding of TT_RS00015 would need supervised pairs. The available CDS records are unlabelled at that granularity.
2. **The cross-modal coherence test failed.** Experiment 3 measured the DNA–metadata cosine in DNABERT-S space and got values in [0.009, 0.205] (no shared structure exists for a linear projection to recover.
3. **Operational fragility.** Swapping the DNA encoder (say, fine-tuning DNABERT-S on a new corpus) would invalidate the projection. The bridge architecture decouples the two encoders entirely) they can be retrained, replaced, or version-bumped independently.

---

## 7. Why embedding metadata at all (vs filtering)

A second alternative would be to skip Index B and treat metadata as a structured filter applied to FAISS results. This is rejected because filters do **inclusion**, not **ranking**:

| Approach | Output | Use case |
|---|---|---|
| **Exact filter** | Boolean (include / exclude) | `organism == "Thermus thermophilus"` |
| **Embedding + cosine** | Continuous similarity score | "oxidoreductase activity, thermophilic bacterium" |

Two protein descriptions (`"SDR family NAD(P)-dependent oxidoreductase"` and `"NAD(P)/FAD-dependent oxidoreductase"`) share meaning but not tokens. A filter cannot relate them; an embedding can. The bridge architecture uses both: embeddings for ranking, optional post-filters for hard constraints (e.g., genus-level restriction applied after FAISS retrieval).

---

## 8. What experiments 3 and 4 jointly establish

| Claim | Evidence (Exp 3) | Evidence (Exp 4) |
|---|---|---|
| Single-encoder fails for metadata | DNABERT-S all metadata cosines in [0.81, 1.0]; UNK-dominated | Silhouette flips from +0.32 to −0.04 at 20 species |
| Dedicated text encoder works | MiniLM intra-pair 0.944, cross-pair 0.487–0.748 | MiniLM silhouette +0.042 at 20 species |
| DNA and metadata spaces are disjoint | Cross-modal cosines in [0.009, 0.205] | (Not re-measured; result is structural) |
| Dimensions are incompatible | DNABERT-S = 768, MiniLM = 384 | Same |

The combined finding mandates the dual-index design and rules out the alternatives. Everything downstream (the three query modes, the FAISS index choices in Experiments 7–8, the retrieval evaluations in Experiments 5/6/9/10) assumes this architecture as a given.

---

## 9. Concrete properties of the deployed bridge

From the FAISS deployment experiments (7–10) at the production scale of n = 1,000 records across 20 bacterial species:

| Property | Value | Source |
|---|---|---|
| Index A type | `IndexHNSWFlat(M=32, efConstruction=200, efSearch=64)` | Exp 8 (crossover at n≈1,000 for DNA) |
| Index B type | `IndexFlatIP` (exact) | Exp 8 (crossover at n≈2,000 for text) |
| Index A accuracy vs brute-force | 100% top-1 | Exp 10 |
| Index B accuracy vs brute-force | 100% top-1 | Exp 10 |
| DNA→DNA P@1 | 0.655 | Exp 10 |
| Text→DNA P@1 | 0.906 | Exp 10 |
| DNA→Text P@1 | 0.655 (identical ranking to DNA→DNA) | Exp 10 |
| End-to-end query latency | 64 µs (HNSW) + 23 µs (Flat) | Exp 10 |

Text→DNA outperforms DNA→DNA by 25 percentage points at rank 1. This is the inverse of the naive expectation: matching at the level of organism names and protein descriptions is more discriminative for species-level retrieval than matching at the level of raw nucleotide sequence, because functionally conserved genes share sequence across species while metadata strings do not.

---

## 10. Limitations and where the bridge could be extended

- **The two encoders are frozen.** Neither DNABERT-S nor MiniLM is fine-tuned on this corpus. If domain shift becomes severe, both can be replaced or fine-tuned independently (but the bridge mechanism itself does not change.
- **Metadata quality directly drives Text→DNA accuracy.** The +30 percentage-point lift in Text→DNA P@1 between Experiment 6 (0.601) and Experiment 10 (0.906) came from one fix: correctly resolving organism names from FASTA filename accessions instead of leaving them as `GCF_000008125.1`. The bridge architecture amplifies metadata quality.
- **No hybrid score.** The current design picks one of the two indices per query; it does not combine A and B scores into a hybrid ranking. A hybrid retriever (e.g., reciprocal rank fusion across the two indices) is a natural extension and would require no architectural change) both indices already return ranked ids on the same row space.
- **Single-species-per-record assumption.** Records have one organism label. The architecture does not currently model multi-species or hierarchical (genus, family) similarity. The bridge would still work if metadata were extended with taxonomic strings; only the encoder choice would need re-evaluation.

---

## 11. Summary

The bridge architecture is the operational answer to a single empirical finding: **a 6-mer DNA tokeniser cannot produce useful metadata embeddings, and the metadata embeddings need to come from a model trained on natural language.** Once you commit to two encoders with different dimensions, two indices are forced, and the cheapest way to connect them is to keep the row order in sync between both indices and use the integer row as a bridge.

The architecture has zero learned parameters in the bridge itself, supports all three query types with a uniform two-step pattern (search the matching-modality index, fetch payload from the opposite modality via row index), and decouples the two encoders so they can be upgraded independently. Experiments 3 and 4 motivate the design; experiments 5, 6, 9, and 10 verify it works at the production scale.
