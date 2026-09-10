# Experiment 2: Scaled Embedding Benchmark: Results

The current reproduction notes below describe the checked-in script. The historical record retains the original measurements; raw artifacts mentioned there are not included.

## Current Reproduction Guide

### Objective

Evaluate how the embedding rankings change when the sample includes many records from each organism group.

### Experimental Setup

The script takes up to 50 qualifying CDS records per sorted corpus file, filtered to 300–900 bp. The stored report evaluates 1,000 records from 20 accession groups. It compares the 16 `MODEL_REGISTRY` entries using within-group cosine similarity, between-group distance, silhouette, and timing. GERM uses the documented DNABERT-2 fallback when a local checkpoint is absent.

### Inputs

The checked-in [CDS corpus](../../../rag_corpus_uniform) supplies the FASTA records and headers. Selection and model settings are defined in the script, not in the root YAML configuration.

### How to Run

From the repository root, with the [benchmark dependencies](../../README.md#reproduction) installed:

```bash
python3 embedder_benchmark/experiment_2_embedding_benchmark/experiment_2_embedding_benchmark.py
```

These commands load model weights and may take substantial time. They were not executed for the repository cleanup. Outputs go to the `results/` directory beside the script; rerunning can overwrite generated artifacts.

The script accepts `--only` with comma-separated registry keys and `--force` to rerun existing CSV entries. For example:

```bash
python3 embedder_benchmark/experiment_2_embedding_benchmark/experiment_2_embedding_benchmark.py --only DNABERT_S
```

The historical reports refer to separate GPU/Colab runs and merged GPU CSVs. Those notebooks and CSVs are absent from this checkout. The CPU script alone is not a verified reproduction of the combined historical table. See the overview's [reproducibility limits](../../README.md#reproducibility-and-evidence-limits).

### Results

The checked-in [historical result report](#historical-result-record) contains the recorded measurements. The raw CSVs, logs, plots, and binary artifacts described below are not included in this checkout.

`experiment_2_results.csv` contains model metrics. `experiment_2_labels.npy` and `experiment_2_embeddings_<MODEL>.npy` save labels and embeddings; species heatmaps and PCA plots are generated per model.

### Main Findings

DNABERT_S has the highest silhouette in the stored table (0.0486), followed by DNABERT_1 (0.0236). These are corpus-specific embedding results, not gap-reconstruction accuracy or a controlled isolation of training-objective effects.

These findings summarize stored evidence, not a newly repeated experiment. Consult the [evidence limits](../../README.md#reproducibility-and-evidence-limits) before comparing historical numbers with a new run.

## Historical Result Record

> Historical evidence: numerical tables are retained. Raw CSVs, logs, and figures mentioned below are not included in this checkout. Earlier organism names, model explanations, and deployment recommendations may not describe the current implementation. Use the current reproduction guide above and [evidence limits](../../README.md#reproducibility-and-evidence-limits) for the current scope and reproduction constraints.

**Date**: 2026-04-28 (extended 2026-05-19 / 2026-05-20 / 2026-05-21)
**Hardware**: CPU-only locally + A100 GPU (Colab Pro) for 3 models; Python 3.10; PyTorch 2.11; transformers 5.5.4
**Script**: `embedder_benchmark/experiment_2_embedding_benchmark/experiment_2_embedding_benchmark.py` + `experiments/embedding_benchmark_gpu.ipynb`

---

## 1. Objective

Validate the embedding model ranking from Experiment 1 at statistically meaningful scale:
50 CDS sequences × 20 species = **1,000 sequences total** (24,500 same-species pairs and 475,000 cross-species pairs). The n=5 silhouette in Experiment 1 has almost no statistical power (one same-species pair); the n=1,000 silhouette across 20 classes is the diagnostic ranking that drives the production model selection.

---

## 2. Input Sequences

**1,000 CDS records** drawn from all 20 species in `rag_corpus_uniform/`, 50 per species (300–900 bp filter).

| Label | Accession | Records |
|-------|-----------|---------|
| 0 | GCA_001819145.1 | 50 |
| 1 | GCA_013426185.1 | 50 |
| 2 | GCA_015709715.1 | 50 |
| 3 | GCA_016699085.1 | 50 |
| 4 | GCA_016699145.1 | 50 |
| 5 | GCA_016699385.1 | 50 |
| 6 | GCA_025999335.1 | 50 |
| 7 | GCF_000008125.1 (*Thermus thermophilus*) | 50 |
| 8 | GCF_000008725.1 (*Chlamydia trachomatis*) | 50 |
| 9 | GCF_000009005.1 (*Staphylococcus aureus*) | 50 |
| 10 | GCF_000009025.1 (*Dehalococcoides mccartyi*) | 50 |
| 11 | GCF_000013045.1 | 50 |
| 12 | GCF_000019205.1 | 50 |
| 13 | GCF_000020225.1 | 50 |
| 14 | GCF_000158275.2 | 50 |
| 15 | GCF_000204255.1 | 50 |
| 16 | GCF_001433955.1 | 50 |
| 17 | GCF_005697215.1 | 50 |
| 18 | GCF_016859125.1 | 50 |
| 19 | GCF_030623315.1 | 50 |

---

## 3. Model Registry: all 16 complete

| Model | Dim | Training | Tokenisation | Where | Status |
|-------|-----|----------|--------------|-------|--------|
| DNABERT_S | 768 | Contrastive | BPE | CPU local | ✅ |
| DNABERT_2 | 768 | MLM | BPE | CPU local | ✅ |
| DNABERT_1 | 768 | MLM | 6-mer | CPU local | ✅ |
| NT_v2_500M | 1024 | MLM | k-mer | CPU local | ✅ |
| NT_v2_250M | 768 | MLM | k-mer | CPU local | ✅ |
| NT_v2_100M | 512 | MLM | k-mer | CPU local | ✅ |
| NT_v2_50M | 512 | MLM | k-mer | CPU local | ✅ |
| NT_v2_50M_3mer | 512 | MLM | 3-mer | CPU local | ✅ |
| NT_2500M | 2560 | MLM | k-mer | **A100 (Colab)** | ✅ |
| MetaBERTa | 1024 | MLM | 6-mer | CPU local | ✅ |
| GENA_LM | 768 | MLM (BigBird) | BPE | CPU local | ✅ |
| GERM | 768 | (DNABERT-2 fallback) | BPE | CPU local | ✅* |
| SpliceBERT | 512 | MLM | character | CPU local | ✅ |
| GROVER | 768 | MLM | learned | CPU local | ✅ |
| Caduceus | 512 | MLM (Mamba SSM) | character | **A100 (Colab)** | ✅ |
| HyenaDNA | 256 | MLM (Hyena, long-context) | character | **A100 (Colab)** | ✅ |

*GERM has no local checkpoint and falls back to DNABERT-2 weights : identical metrics.

The 3 GPU-only models (Caduceus, HyenaDNA, NT_2500M) were evaluated in a Colab A100 session and merged via `exp1_gpu.csv` / `exp2_gpu.csv`.

---

## 4. Quantitative Results (sorted by silhouette ↓)

| Rank | Model | Dim | Intra-sim | Inter-dist | **Silhouette** | Δ vs Exp 1 |
|---|---|-----|-----------|------------|----------------|-----------|
| 1 | **DNABERT_S** | 768 | 0.5969 | **0.6460** | **0.0486** | −0.131 (5 → 1 ranking) |
| 2 | DNABERT_1 | 768 | 0.9529 | 0.1595 | 0.0236 | −0.245 (1 → 2 ranking) |
| 3 | NT_v2_50M_3mer | 512 | 0.5291 | 0.6571 | 0.0139 | −0.084 |
| 4 | GROVER | 768 | 0.8888 | 0.2745 | 0.0128 | −0.206 (3 → 4 ranking) |
| 5 | NT_2500M | 2560 | 0.9375 | 0.1050 | 0.0090 | new (GPU) |
| 6 | GENA_LM | 768 | 0.9966 | 0.0050 | 0.0062 | −0.187 (4 → 6 ranking) |
| 7 | DNABERT_2 | 768 | 0.8829 | 0.2153 | 0.0029 | −0.175 |
| 7 | GERM | 768 | 0.8829 | 0.2153 | 0.0029 | (DNABERT-2 fallback) |
| 9 | NT_v2_500M | 1024 | 0.0126 | 0.9924 | −0.0065 | unchanged |
| 10 | NT_v2_250M | 768 | 0.0137 | 0.9906 | −0.0083 | +0.018 |
| 11 | NT_v2_50M | 512 | 0.0262 | 0.9830 | −0.0108 | unchanged |
| 12 | NT_v2_100M | 512 | 0.0211 | 0.9843 | −0.0124 | +0.035 |
| 13 | MetaBERTa | 1024 | 0.8644 | 0.2624 | −0.0131 | −0.058 |
| 14 | SpliceBERT | 512 | 0.9489 | 0.1373 | −0.0228 | **−0.263 (2 → 14)** |
| 15 | HyenaDNA | 256 | 0.9877 | 0.0420 | −0.0386 | new (GPU) |
| 16 | Caduceus | 512 | 0.9979 | 0.0080 | −0.0581 | new (GPU) |

> **Metric definitions** : see `../../experiments_summary.md` glossary. *Intra-species sim* is the mean cosine similarity across all same-species pairs averaged over 20 species (24,500 pairs). *Inter-species dist* = 1 − mean cosine similarity across 475,000 cross-species pairs. *Silhouette* is the sklearn silhouette score over 1,000 samples with 20 labels.

---

## 5. Visualizations

Per-model artifacts (`embedder_benchmark\experiment_2_embedding_benchmark/results/experiment_2_{MODEL}_*.png`):
- 20×20 species-mean cosine similarity heatmap
- 2D PCA scatter coloured by species

Cross-model comparative figure:
- `experiments/species_separation_top5.png` / `.pdf` : UMAP (cosine, 1×5 grid) of the top-5 models with `.npy` embeddings on disk

---

## 6. Analysis

### 6.1 Top-2 reversal: DNABERT_S overtakes DNABERT_1 at scale

At n=5, DNABERT_1 had the highest silhouette (0.2686 vs DNABERT_S 0.1794). At n=1,000, DNABERT_S takes the top spot (0.0486 vs DNABERT_1 0.0236). The reason is visible in inter-species distance:

| Model | Inter-dist (n=5) | Inter-dist (n=1,000) | Inter-dist retention |
|---|---|---|---|
| DNABERT_S | 0.7828 | 0.6460 | 82.5% |
| DNABERT_1 | 0.1785 | 0.1595 | 89.4% |

DNABERT_1's intra-species similarity is uniformly high (0.95+) and its inter-species distance is uniformly low (~0.16) at both scales : different species are not pushed apart. Its high silhouette at n=5 came from the same-species TT pair being very close. At 20 species this is no longer enough: cross-species sequences are nearly as close as same-species ones.

DNABERT_S has the only embedding space in the benchmark where the **inter-species distance survives scaling**. Same-species sequences move apart (0.81 → 0.60) but different-species sequences move apart more (1−0.78 → 1−0.65), so the relative gap is preserved.

### 6.2 The DNABERT family arc: contribution comes from contrastive training, not BPE

Three models from the DNABERT lineage at fixed scale (n=1,000):

| Variant | Tokenisation | Training | Silhouette | Inter-dist |
|---|---|---|---|---|
| DNABERT_1 | 6-mer | MLM | 0.0236 | 0.1595 |
| DNABERT_2 | BPE | MLM | 0.0029 | 0.2153 |
| **DNABERT_S** | BPE | **Contrastive** | **0.0486** | **0.6460** |

The DNABERT_1 → DNABERT_2 step changed tokenisation (6-mer → BPE) while keeping MLM training. Result: silhouette **decreased** from 0.0236 to 0.0029. BPE alone made things worse.

The DNABERT_2 → DNABERT_S step kept BPE and added contrastive training. Result: silhouette **jumped** from 0.0029 to 0.0486 (16×), inter-distance from 0.2153 to 0.6460 (3×). The contrastive objective is what makes DNABERT_S retrieval-ready.

This isolates the contribution: the family's improvement is not from BPE tokenisation, it is from **species-aware contrastive learning during fine-tuning**.

### 6.3 NT v2 scale curve: MLM does not improve with scale

Five NT v2 models trained with the same MLM objective at five parameter scales:

| Model | Params | Silhouette | Inter-dist |
|---|---|---|---|
| NT_v2_50M | 50 M | −0.0108 | 0.9830 |
| NT_v2_100M | 100 M | −0.0124 | 0.9843 |
| NT_v2_250M | 250 M | −0.0083 | 0.9906 |
| NT_v2_500M | 500 M | −0.0065 | 0.9924 |
| NT_2500M | 2,500 M | **+0.0090** | 0.1050 |

Two findings:

1. **From 50M to 500M (10× scaling at fixed training objective): no useful change.** Silhouette stays at −0.01 ± 0.003. Inter-distance stays at 0.98–0.99 : embeddings are uniformly distributed on the unit hypersphere. This is direct empirical falsification of "bigger MLM model = better embeddings."

2. **At 2.5B (5× more): a flip to weakly positive.** NT_2500M is the only NT v2 variant with positive silhouette (0.009). It loses the uniform-hypersphere behaviour entirely (inter-distance crashes from 0.99 to 0.11), so embeddings now have *some* metric structure, but the silhouette is still nine times smaller than DNABERT_S's (0.009 vs 0.0486). The cost was 50× more parameters than DNABERT-2 : DNABERT-S delivers more separation per parameter by a wide margin.

### 6.4 Tokenisation ablation: 3-mer beats k-mer at fixed scale

Two 50M NT v2 models with the same training objective and dataset, differing only in tokenisation:

| Model | Tokenisation | Silhouette | Inter-dist |
|---|---|---|---|
| NT_v2_50M | k-mer (default) | −0.0108 | 0.9830 |
| **NT_v2_50M_3mer** | **3-mer** | **+0.0139** | **0.6571** |

Switching to 3-mer tokenisation lifts the silhouette into positive territory and recovers a meaningful inter-distance. This is a clean, isolated demonstration that tokenisation choice can salvage a marginal MLM model, but cannot reach DNABERT_S's performance (0.0486) without contrastive training.

### 6.5 SpliceBERT collapse: the n=5 ranking lies

SpliceBERT was rank 2 at n=5 (silhouette 0.2406). At n=1,000 it is **rank 14 with silhouette −0.0228** : a 26-position drop on a 16-model leaderboard. The mechanism is the same as DNABERT_1 but more extreme: very high intra-species similarity (0.9489) is paid for with very low inter-species distance (0.1373), so when 20 species are present they all collapse into one dense cluster.

SpliceBERT is trained on mammalian RNA splice-site sequences. Applying it to bacterial CDS records is a domain mismatch; the model produces nearly-identical embeddings for all bacterial sequences regardless of species. This is a useful negative result: pretrained-on-anything ≠ usable-for-anything.

### 6.6 Caduceus and HyenaDNA: long-context models, short-context corpus

Both Caduceus (Mamba SSM, 131k context) and HyenaDNA (Hyena, 1M context) score negative silhouettes (−0.058 and −0.039) at n=1,000. Their high intra-species similarity (>0.98) and near-zero inter-species distance (<0.05) mean they produce nearly-identical embeddings for everything. This is consistent with the design of these models: they are optimised for long-range dependency modelling, not for producing discriminative dense representations of short (300–900 bp) CDS records. Future work could revisit them on whole-genome or operon-length sequences.

### 6.7 The "all 13 of 16 fail at scale" pattern

13 of 16 models score silhouette ≤ 0.015 at n=1,000. Only 3 score above:

- **DNABERT_S: 0.0486** (contrastive training)
- DNABERT_1: 0.0236 (6-mer historical baseline)
- NT_v2_50M_3mer: 0.0139 (3-mer ablation)

Every model in this top tier either has a contrastive component (DNABERT_S) or a non-standard tokenisation that aligns better with the corpus statistics (DNABERT_1's 6-mer, NT_v2's 3-mer). Standard MLM with BPE at any size (DNABERT_2, all 5 NT v2 variants, GROVER, GENA_LM) saturates around silhouette = 0, regardless of parameter count.

---

## 7. Conclusion

| Criterion | Winner | Score |
|-----------|--------|-------|
| Overall cluster quality (silhouette) | **DNABERT_S** | **0.0486** |
| Inter-species separation | DNABERT_S | 0.6460 |
| Per-parameter efficiency | DNABERT_S (110 M params) | 23× better than NT_2500M |
| Inference speed (CPU) | MetaBERTa | 0.048 s/seq |

**Selected production model: DNABERT_S** (`zhihan1996/DNABERT-S`)

The 16-model benchmark fully validates the original 4-model selection. DNABERT_S is the only model that simultaneously: (1) ranks #1 on silhouette at scale; (2) ranks #1 on inter-species distance at scale; (3) achieves this with a 110 M parameter footprint that is 23× smaller than NT_2500M (the only MLM model with any positive silhouette at n=1,000). Contrastive training is the decisive design choice. Scaling parameters is not a substitute, and changing tokenisation alone is not a substitute.

The DNABERT family arc (1 → 2 → S) provides a clean controlled demonstration: at fixed training objective, the BPE tokenisation actually *hurt* performance (1 → 2). The improvement came entirely from contrastive fine-tuning (2 → S). The NT v2 scale curve (50M → 500M, all MLM) provides the orthogonal demonstration: scaling 10× at fixed objective does not help. Together these two ablations make the thesis claim (*training objective matters more than scale or tokenisation*) empirically defensible.

---

## 8. Limitations

- **Short sequences (300–900 bp).** This range was chosen to fit every model's tokeniser limits. HyenaDNA and Caduceus are optimised for much longer contexts and may behave differently on full genes / operons.
- **20 species, all bacterial.** All sequences are bacterial CDS records. The conclusion may not transfer to viral, archaeal, or eukaryotic embeddings. Of course, the GenomIO RAG pipeline targets bacterial gap filling, so this is the operationally relevant scope.
- **GERM fallback.** The published GERM checkpoint was not located locally; the registry entry falls back to DNABERT-2 weights and produces identical numbers. Counted in the registry for completeness, not as an independent data point.
- **CPU vs GPU timing not directly comparable.** The 13 CPU models were timed on a single-threaded CPU; the 3 GPU models on an A100. Inference-speed comparisons across the two groups are not apples-to-apples; see per-experiment CSV for raw values.
- **Silhouette absolute values are small.** At 20 classes, silhouette scores near 0 are expected even for well-clustered models. The **sign** (positive vs negative) and the **relative ranking** are the substantive signal, not the absolute number.
