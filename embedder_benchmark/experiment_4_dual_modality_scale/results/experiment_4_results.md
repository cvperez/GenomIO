# Experiment 4: Dual-Modality Embedding Strategy: Scaled Validation Results

The current reproduction notes below describe the checked-in script. The historical record retains the original measurements; raw artifacts mentioned there are not included.

## Current Reproduction Guide

### Objective

Repeat the three-way DNA/metadata embedding comparison with multiple records per organism group.

### Experimental Setup

The script selects up to 50 CDS records per corpus file using the 300–900 bp filter, giving 1,000 records in the stored evaluation. It compares DNA/DNABERT-S, metadata/DNABERT-S, and metadata/MiniLM. It reads organism names from `organisms.tsv` and computes cosine-based group metrics, silhouette, PCA, and species-mean heatmaps.

### Inputs

The checked-in [CDS corpus](../../../rag_corpus_uniform) supplies the FASTA records and headers. Selection and model settings are defined in the script, not in the root YAML configuration. The script also reads [organisms.tsv](../../../rag_corpus_uniform/organisms.tsv).

### How to Run

From the repository root, with the [benchmark dependencies](../../README.md#reproduction) installed:

```bash
python3 embedder_benchmark/experiment_4_dual_modality_scale/experiment_4_dual_modality_scale.py
```

These commands load model weights and may take substantial time. They were not executed for the repository cleanup. Outputs go to the `results/` directory beside the script; rerunning can overwrite generated artifacts.

### Results

The checked-in [historical result report](#historical-result-record) contains the recorded measurements. The raw CSVs, logs, plots, and binary artifacts described below are not included in this checkout.

`experiment_4_results.csv` contains the three embedding configurations. Each configuration generates a species heatmap and PCA plot.

### Main Findings

The stored silhouettes are 0.0486 for DNA/DNABERT-S, -0.0418 for metadata/DNABERT-S, and 0.0424 for metadata/MiniLM. The comparison supports using a separate text encoder for this metadata benchmark; it does not establish reconstruction quality.

These findings summarize stored evidence, not a newly repeated experiment. Consult the [evidence limits](../../README.md#reproducibility-and-evidence-limits) before comparing historical numbers with a new run.

## Historical Result Record

> Historical evidence: numerical tables are retained. Raw CSVs, logs, and figures mentioned below are not included in this checkout. Earlier organism names, model explanations, and deployment recommendations may not describe the current implementation. Use the current reproduction guide above and [evidence limits](../../README.md#reproducibility-and-evidence-limits) for the current scope and reproduction constraints.

**Date**: 2026-04-28  
**Hardware**: CPU-only; Python 3.10; transformers 5.5.4; sentence-transformers 5.4.1  
**Script**: `embedder_benchmark/experiment_4_dual_modality_scale/experiment_4_dual_modality_scale.py`

---

## 1. Objective

Validate at scale (50 CDS × 20 species = 1,000 sequences) the dual-modality embedding
strategy identified in Experiment 3: DNABERT-S for DNA sequences, all-MiniLM-L6-v2 for
metadata text. Confirm that the false-positive silhouette of DNABERT-S on text (0.3192 in
Experiment 3, n=5) collapses under more species and more sequences.

---

## 2. Scale

- **Sequences**: 1,000 total (50 per species × 20 species, 300–900 bp filter)
- **Species**: all 20 accessions in `rag_corpus_uniform/`
- **Embedding passes**: 3 (DNA/DNABERT-S, Metadata/DNABERT-S control, Metadata/MiniLM)

---

## 3. Why Embedding (Not Filtering) Must Work at Scale

Experiment 3 established that metadata must be embedded with a text model, not filtered by exact string match, to enable ranked retrieval. This experiment stress-tests that conclusion by scaling to 20 species and 1,000 sequences.

At this scale, a filter-based approach would require knowing in advance which organism label to match : it cannot generalise to unseen organisms or retrieve by functional similarity (e.g., "find CDS records encoding oxidoreductases similar to my query"). The embedding approach must produce a positive silhouette (intra-species similarity > inter-species similarity) to be useful for retrieval.

The critical question is whether the **MiniLM silhouette** (0.2115 at n=5, 4 species) survives the harder 20-species problem, and whether the **DNABERT-S metadata false positive** (0.3192 at n=5) is exposed as such at scale.

---

## 4. Results

| Modality | Model | Dim | Intra-sim ↑ | Inter-dist ↑ | Silhouette ↑ | Time/seq |
|----------|-------|-----|-------------|--------------|--------------|----------|
| **DNA** | DNABERT-S | 768 | 0.5969 | 0.6460 | **0.0486** | 0.109 s |
| Metadata | DNABERT-S (control) | 768 | 0.9513 | 0.1048 | **−0.0418** | 0.092 s |
| **Metadata** | all-MiniLM-L6-v2 | 384 | 0.8009 | 0.2513 | **0.0424** | 0.006 s |

---

## 5. Visualizations

| File | Contents |
|------|----------|
| `experiment_4_dna_species_heatmap.png` | 20×20 species-mean DNA similarity (DNABERT-S) |
| `experiment_4_dna_pca.png` | 2D PCA of DNA embeddings |
| `experiment_4_meta_dnaberts_species_heatmap.png` | 20×20 metadata similarity (DNABERT-S on text: control) |
| `experiment_4_meta_dnaberts_pca.png` | 2D PCA of DNABERT-S metadata embeddings |
| `experiment_4_meta_textmodel_species_heatmap.png` | 20×20 metadata similarity (all-MiniLM-L6-v2) |
| `experiment_4_meta_textmodel_pca.png` | 2D PCA of MiniLM metadata embeddings |

---

## 6. Analysis

### DNA/DNABERT-S: Identical to Experiment 2

All three metrics (intra=0.5969, inter=0.646, silhouette=0.0486) are numerically identical
to Experiment 2, confirming reproducibility. The DNA embedding quality is stable and independent
of what metadata model is used alongside it.

### Metadata/DNABERT-S: False positive exposed at scale

In Experiment 3 (n=5, 4 species), DNABERT-S on text produced silhouette=0.3192 : an
apparently strong result. At 1,000 sequences across 20 species, the silhouette collapses to
**−0.0418**, definitively negative.

The cause is clear from the metrics: intra-species similarity = 0.9513, inter-species
distance = 0.1048. All metadata embeddings are packed into a tiny region of the 768-dim space
(nearly uniform UNK-token representations), with intra-species pairs only marginally more
similar than cross-species pairs. At 4 species, this marginal difference happened to produce a
positive silhouette; at 20 species, the same pattern produces a negative one because there are
more cross-species neighbors pulling each point away from its cluster.

A negative silhouette means that DNABERT-S metadata embeddings are **actively misleading**:
in a nearest-neighbor retrieval, a query record would more likely retrieve records from other
species than from its own species. This confirms that DNABERT-S cannot be used as a metadata encoder.

### Metadata/all-MiniLM-L6-v2: Positive silhouette maintained at 20 species

The text model achieves silhouette=0.0424 across 20 species and 1,000 sequences : positive
and consistent with the 5-sequence result (0.2115 in Experiment 3, higher there due to only
4 species including a single same-species pair with very distinct organism names).

Intra-species similarity = 0.8009: same-species metadata records (same organism name, shared
locus tag prefixes, similar protein family descriptions) are consistently pulled together.
Inter-species distance = 0.2513: different-species metadata records are meaningfully separated.

The text model runs at 0.006 s/seq (18× faster than DNABERT-S for metadata) and requires
no GPU for inference on 1,000 records.

---

## 7. Comparison with Experiment 3

| Configuration | Silhouette (Exp 3, n=5, 4 species) | Silhouette (Exp 4, n=1000, 20 species) |
|---------------|-------------------------------------|----------------------------------------|
| DNA / DNABERT-S | 0.1794 | 0.0486 |
| Metadata / DNABERT-S | 0.3192 ⚠️ (false positive) | **−0.0418** |
| Metadata / MiniLM | 0.2115 | 0.0424 |

The Experiment 3 warning about the DNABERT-S metadata silhouette being a false positive is
fully confirmed: it collapses from +0.32 to −0.04 at scale. The MiniLM silhouette drops
moderately (from 0.21 to 0.04) as expected for a harder 20-class problem, but remains positive.

---

## 8. Conclusion

The dual-embedding strategy from Experiment 3 is **confirmed at scale**:

| Component | Model | FAISS Index | Dim |
|-----------|-------|-------------|-----|
| DNA sequences | DNABERT-S | Index A | 768 |
| CDS metadata text | all-MiniLM-L6-v2 | Index B | 384 |

Using DNABERT-S for both modalities is ruled out: at 20-species scale its metadata silhouette
is negative (−0.0418), meaning it would actively degrade retrieval quality compared to random.
The text model is the correct and only viable choice for metadata embedding.
