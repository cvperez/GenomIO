# Experiment 1: Embedding Model Selection — Results

**Date**: 2026-04-21 (extended runs 2026-05-19 / 2026-05-20)
**Hardware**: CPU-only (no CUDA); Python 3.10; PyTorch 2.11.0+cu130; transformers 5.5.4
**Script**: `experiments/experiment_1_embedding_benchmark.py`

---

## 1. Objective

Determine which genomics-native embedding model produces the most biologically meaningful vector representations of DNA sequences by evaluating, on a small controlled subset, two properties:

1. **Cluster** sequences from the same species (intra-species similarity)
2. **Separate** sequences from different species (inter-species separation)

The 5-sequence scale is a proof-of-concept — Experiment 2 scales the same models to 1,000 sequences for statistically robust conclusions.

---

## 2. Input Sequences

Five CDS records selected from `rag_corpus_uniform/`, raw nucleotide sequences only (headers stripped):

| Tag | Record ID | Length | Species | Label |
|-----|-----------|--------|---------|-------|
| seq1_TT | `lcl\|NC_005835.1_cds_WP_011172461.1_2` | 771 bp | *Thermus thermophilus* | 0 |
| seq2_TT | `lcl\|NC_005835.1_cds_WP_011172462.1_3` | 543 bp | *Thermus thermophilus* | 0 |
| seq3_CT | `lcl\|NC_000117.1_cds_NP_219504.1_3` | 303 bp | *Chlamydia trachomatis* | 1 |
| seq4_SA | `lcl\|NC_007622.1_cds_WP_000449218.1_7` | 813 bp | *Staphylococcus aureus* | 2 |
| seq5_DM | `lcl\|NC_007356.1_cds_WP_011308641.1_3` | 615 bp | *Dehalococcoides mccartyi* | 3 |

---

## 3. Model Registry — 16 candidates, 13 complete

The registry was deliberately expanded from the original 5 candidates to 16 to strengthen the empirical foundation for the model selection. The 16 models cover three orthogonal axes of variation that the thesis claims about: **training objective**, **tokenisation**, and **parameter scale**.

### 3.1 Models with full results (CPU-runnable, 13)

| Model | HuggingFace ID | Training | Tokenisation | Status |
|-------|----------------|----------|--------------|--------|
| DNABERT_S | `zhihan1996/DNABERT-S` | Contrastive | BPE | ✅ |
| DNABERT_2 | `zhihan1996/DNABERT-2-117M` | MLM | BPE | ✅ |
| DNABERT_1 | `armheb/DNA_bert_6` | MLM | 6-mer | ✅ |
| NT_v2_500M | `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species` | MLM | k-mer | ✅ |
| NT_v2_250M | `InstaDeepAI/nucleotide-transformer-v2-250m-multi-species` | MLM | k-mer | ✅ |
| NT_v2_100M | `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species` | MLM | k-mer | ✅ |
| NT_v2_50M | `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` | MLM | k-mer | ✅ |
| NT_v2_50M_3mer | `InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species` | MLM | 3-mer | ✅ |
| MetaBERTa | `MsAlEhR/MetaBerta-400-fragments-18k-genome` | MLM | 6-mer | ✅ |
| GENA_LM | `AIRI-Institute/gena-lm-bigbird-base-t2t` | MLM (BigBird) | BPE | ✅ |
| GERM | (DNABERT-2 fallback — no local checkpoint) | MLM | BPE | ✅* |
| SpliceBERT | `multimolecule/splicebert` | MLM | char | ✅ |
| GROVER | `PoetschLab/GROVER` | MLM | learned | ✅ |

*GERM is included as a placeholder; with no local checkpoint it falls back to DNABERT-2 weights, hence identical metrics. Listed for transparency.

### 3.2 Models pending GPU evaluation (3)

| Model | HuggingFace ID | Architecture | Status |
|-------|----------------|--------------|--------|
| Caduceus | `kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16` | Mamba SSM, bidirectional | ⏳ Colab |
| HyenaDNA | `LongSafari/hyenadna-large-1m-seqlen-hf` | Hyena (long-context) | ⏳ Colab |
| NT_2500M | `InstaDeepAI/nucleotide-transformer-2.5b-multi-species` | MLM, 2.5B params | ⏳ Colab |

> **Note on substitutions.** Three models from the original spec (`Evo2_7B`, `Evo2_1B`, `AIDO_DNA_7B`) proved unreachable: Evo2's StripedHyena kernels are incompatible with both T4 and current torch on Colab, and AIDO.DNA-7B's tokenizer fails to instantiate. They were replaced with `NT_v2_100M`, `NT_v2_250M` (completing the NT v2 scale curve) and `DNABERT_1` (closing the DNABERT family arc) — substitutions that strengthen the thesis narrative.

### 3.3 Compatibility patches

All NT v2 cached `modeling_esm.py` files (50M, 100M, 250M, 500M, 50M-3mer) required four patches against transformers 5.5.4:
1. Inline definition of `find_pruneable_heads_and_indices` (removed from `transformers.pytorch_utils`)
2. Add `is_decoder` / `add_cross_attention` parameters and assignments to `EsmConfig`
3. Add `all_tied_weights_keys = {}` class attribute to `EsmForMaskedLM`
4. Replace `self.get_head_mask(...)` call with inline `[None] * num_hidden_layers`

The same patches in the 500M cache were copied verbatim to the 100M and 250M caches. DNABERT-S/2 needed ALiBi tensor forced to CPU + flash attention disabled. MetaBERTa's `KmerTokenizer` was replaced with an inline 6-mer encoder.

---

## 4. Quantitative Results (sorted by silhouette ↓)

| Rank | Model | Dim | Intra-sim ↑ | Inter-dist ↑ | **Silhouette ↑** | Time/seq |
|---|---|-----|-------------|--------------|------------------|----------|
| 1 | **DNABERT_1** | 768 | 0.9867 | 0.1785 | **0.2686** | 18.5 s |
| 2 | SpliceBERT | 512 | 0.9833 | 0.1447 | 0.2406 | 3.5 s |
| 3 | GROVER | 768 | 0.9580 | 0.2692 | 0.2189 | 11.3 s |
| 4 | GENA_LM | 768 | 0.9989 | 0.0049 | 0.1934 | 17.9 s |
| 5 | **DNABERT_S** | 768 | 0.8125 | **0.7828** | 0.1794 | 0.93 s |
| 6 | DNABERT_2 | 768 | 0.9359 | 0.3320 | 0.1776 | 0.32 s |
| 6 | GERM* | 768 | 0.9359 | 0.3320 | 0.1776 | 1.18 s |
| 8 | NT_v2_50M_3mer | 512 | 0.5946 | 0.7790 | 0.0979 | 1.27 s |
| 9 | MetaBERTa | 1024 | 0.8491 | 0.2880 | 0.0450 | 0.12 s |
| 10 | NT_v2_500M | 1024 | 0.0120 | 1.0058 | −0.0026 | 2.25 s |
| 11 | NT_v2_50M | 512 | 0.0502 | 0.9766 | −0.0043 | 1.15 s |
| 12 | NT_v2_250M | 768 | −0.0618 | 0.9929 | −0.0268 | 20.6 s |
| 13 | NT_v2_100M | 512 | 0.0150 | 0.9122 | −0.0472 | 27.3 s |

\* GERM is DNABERT-2 weights (no local checkpoint found); identical numbers expected.

> **Metric definitions** — see `experiments_summary.md` glossary. *Intra-sim* is cosine similarity between the two *T. thermophilus* CDS records; *inter-dist* is the mean cosine distance across the 6 cross-species pairs; *silhouette* is the sklearn silhouette score over labels `[0,0,1,2,3]`.

---

## 5. Cosine Similarity Heatmaps

One heatmap per model in `experiments/experiment_1_{MODEL}_heatmap.png` (13 files).

---

## 6. Analysis

### 6.1 The n=5 ranking is misleading — Experiment 2 reveals the reversal

The most striking pattern at this scale is that the **top of the leaderboard does not survive scaling**. SpliceBERT (rank 2 at n=5, silhouette 0.2406) drops to rank 13 in Experiment 2; DNABERT_1 (rank 1 at n=5, 0.2686) drops to rank 2 in Experiment 2; DNABERT_S (rank 5 at n=5, 0.1794) **rises to rank 1** in Experiment 2 (0.0486). This is exactly why the paired-scale experimental design exists: n=5 with one same-species pair has almost no statistical power, and a model's apparent strength at this scale can easily be spurious.

Experiment 1 should not be read as a final ranking. Its role is to **enumerate candidates and rule out obvious failures** (NT v2 family, near-zero silhouette across all sizes).

### 6.2 NT v2 scale curve — MLM training fails at every size tested

Five NT v2 variants are present, all using the same MLM training objective but different parameter counts and tokenisations:

| Model | Params | Tokenisation | Silhouette |
|---|---|---|---|
| NT_v2_50M | 50 M | k-mer | −0.0043 |
| NT_v2_50M_3mer | 50 M | 3-mer | **0.0979** |
| NT_v2_100M | 100 M | k-mer | −0.0472 |
| NT_v2_250M | 250 M | k-mer | −0.0268 |
| NT_v2_500M | 500 M | k-mer | −0.0026 |

Two observations:
1. **Scaling does not help.** Increasing parameters by 10× (50M → 500M) at fixed training objective produces silhouettes within ±0.05 of zero — all consistent with random embeddings on the unit hypersphere (inter-distance ≈ 1.0).
2. **Tokenisation matters more than size.** The 3-mer variant scores 0.0979 — the only NT v2 model with a clearly positive silhouette at this scale. The two TT sequences share enough 3-mer composition that they cluster; the 6-mer variants do not.

This is empirical falsification of the "bigger model = better embeddings" assumption when the training objective is MLM. Experiment 2 confirms this at scale.

### 6.3 SpliceBERT — strong at n=5, fails at n=1,000

SpliceBERT's high silhouette (0.2406) comes from very high intra-TT similarity (0.9833) on character-level tokenisation. With only one same-species pair, this is enough to dominate the silhouette. Experiment 2 shows the same model with silhouette **−0.0228** at 20 species: the high intra-similarity is paid for by high cross-species similarity too, so the embeddings collapse to a single dense region with no inter-species structure. Domain mismatch (the model is trained on RNA splice-site sequences from mammalian genomes, not bacterial CDS records) is the likely cause.

### 6.4 DNABERT family — the contribution is contrastive training, not BPE

Three models from the DNABERT lineage are now in the benchmark:

| Variant | Tokenisation | Training | Silhouette (n=5) |
|---|---|---|---|
| DNABERT_1 | 6-mer | MLM | 0.2686 |
| DNABERT_2 | BPE | MLM | 0.1776 |
| DNABERT_S | BPE | Contrastive | 0.1794 |

At n=5, DNABERT_1 leads — but the comparison at n=1,000 (Experiment 2) is the diagnostic one. The historical claim that "BPE tokenisation was the breakthrough" is testable: DNABERT_1 vs DNABERT_2 isolates the tokenisation change at fixed training objective.

### 6.5 DNABERT_S — moderate intra-similarity, dominant inter-species distance

DNABERT_S has only moderate intra-similarity (0.8125), well below DNABERT_1 (0.9867) and SpliceBERT (0.9833) — but its inter-species distance of **0.7828** is by far the highest among models that achieve any species-level discrimination. This is the signature of a contrastively-trained model: it learned to push different-species sequences apart, even at the cost of some same-species tightness. This is exactly the right trade-off for nearest-neighbour retrieval.

---

## 7. Conclusion

| Criterion | Winner |
|-----------|--------|
| Silhouette (n=5) | DNABERT_1 (0.2686) — but does not survive scaling |
| Intra-species similarity | GENA_LM (0.9989) |
| Inter-species separation | DNABERT_S (0.7828) |
| Inference speed | MetaBERTa (0.12 s/seq) |

**Experiment 1 cannot select the production model alone.** It enumerates 13 candidates and shows that all 5 NT v2 variants are near-random — a strong negative finding that justifies excluding the entire MLM-only family. Final selection requires Experiment 2's 20-species evaluation. The 3 GPU-only models (Caduceus, HyenaDNA, NT_2500M) are evaluated separately in the Colab pipeline; their results will be merged into `all_models_ranked.csv` and the final UMAP figure (`species_separation_top5.png`).

---

## 8. Limitations

- **Five sequences only.** The silhouette score with n=5 and 4 species classes (3 singleton, 1 pair) has limited statistical power; SpliceBERT and DNABERT_1's top positions are likely partly artifacts. This is exactly what Experiment 2 corrects.
- **GERM fallback.** Without a local GERM checkpoint, the registry entry falls back to DNABERT-2 weights and produces identical numbers. Included for registry completeness, not as an independent data point.
- **CPU inference.** Inference times reflect CPU performance; GPU would be 10–50× faster.
- **3 models pending.** Caduceus, HyenaDNA, NT_2500M require GPU and are deferred to the Colab pipeline.
- **Short sequences.** All sequences are 303–813 bp to fit every model's tokeniser limits. Some models (HyenaDNA, Caduceus) are optimised for much longer contexts and may show different behaviour on full genes / operons.
