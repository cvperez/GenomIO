# Retrieval and embedding experiments

## Overview

This directory contains 12 existing experiments evaluating DNA embeddings, metadata embeddings, retrieval routes, FAISS indices, and rank fusion. Each experiment keeps its script and corresponding result report, including reproduction notes, together. Experiment numbers retain the historical sequence; the organism-only ablation remains part of Experiment 12.

| Experiment | Evaluation and reproduction guide |
|---|---|
| 1 | [DNA embedding model comparison on five CDS records](experiment_1_embedding_benchmark/results/experiment_1_results.md) |
| 2 | [DNA embedding model comparison at 1,000 records](experiment_2_embedding_benchmark/results/experiment_2_results.md) |
| 3 | [DNA and metadata embedding comparison](experiment_3_dual_modality/results/experiment_3_results.md) |
| 4 | [DNA and metadata embeddings at 1,000 records](experiment_4_dual_modality_scale/results/experiment_4_results.md) |
| 5 | [Dual-modality retrieval on five CDS records](experiment_5_retrieval_query/results/experiment_5_results.md) |
| 6 | [Dual-modality retrieval at 1,000 records](experiment_6_retrieval_scale/results/experiment_6_results.md) |
| 7 | [FAISS index comparison on five records](experiment_7_faiss_index/results/experiment_7_results.md) |
| 8 | [FAISS index scaling](experiment_8_faiss_scale/results/experiment_8_results.md) |
| 9 | [FAISS retrieval on five CDS records](experiment_9_faiss_retrieval_poc/results/experiment_9_results.md) |
| 10 | [FAISS retrieval at 1,000 records](experiment_10_faiss_retrieval_scale/results/experiment_10_results.md) |
| 11 | [Metadata text encoder comparison](experiment_11_text_encoder_benchmark/results/experiment_11_results.md) |
| 12 | [Recall curves and DNA/metadata rank fusion](experiment_12_recall_fusion/results/experiment_12_results.md) |

Experiments 1–4 compare representations, 5–6 evaluate brute-force retrieval, 7–8 measure index behavior, 9–10 evaluate FAISS retrieval, and 11–12 compare metadata encoders and fusion. These are retrieval and embedding evaluations; none establishes the accuracy of reconstructed genomic gaps.

## Relationship to GenomIO

The branch's [DNA retrieval implementation](../src/rag/index.py) uses DNABERT-S and a cached FAISS index over the [CDS corpus](../rag_corpus_uniform). It is used by the standalone RAG comparison and the [multi-agent system](../multiagent_system/README.md). The metadata index and rank fusion evaluated here have not been integrated into those execution paths. The older LangChain planner still uses substring retrieval.

The [historical summary and glossary](experiments_summary.md) and [bridge design](bridge_architecture.md) preserve the broader experimental discussion. Treat proposed integrations in those documents as design work, not implemented application behavior.

## Reproduction

Run commands from the repository root. The scripts import dependencies listed in [requirements.txt](../requirements.txt), including PyTorch, Transformers, Biopython, NumPy, pandas, scikit-learn, matplotlib, seaborn, sentence-transformers, and FAISS where used:

```bash
pip install -r requirements.txt
```

Open the individual experiment report for its exact command, inputs, settings, and generated filenames. The scripts locate the repository relative to their own file and write to their own `results/` directories. They read the checked-in `rag_corpus_uniform/` directly; the application's cached index is not a prerequisite for these experiments.

The scripts can download model weights and run for minutes or hours. Reproduction is an explicit research run, not a lightweight installation check. Experiments 1–2 support selecting registry entries and resuming a results CSV. Experiment 12's ablation requires its full-run indices and summary first. Experiments 11–12 create file logs when the module loads; importing them is not a read-only validation procedure.

## Results

Each `experiment_<number>_<description>/results/` contains its existing `experiment_<number>_results.md`. All 12 reports are present. No raw benchmark CSVs, logs, plots, NumPy arrays, or FAISS artifacts are present in this checkout. Filenames described as generated outputs indicate what the scripts write, not additional evidence available for download.

## Reproducibility and Evidence Limits

- The reports describe historical runs; dependency lower bounds are not a locked reproduction environment. Several reports describe compatibility changes to downloaded model code. Those external cache changes are not fully packaged here.
- Experiments 1–2 reference a GPU notebook, `exp1_gpu.csv`, `exp2_gpu.csv`, merged rankings, and comparative figures that are absent. Their full GPU execution/merge procedure cannot be verified from this checkout. GERM's fallback duplicates DNABERT-2 rather than providing an independent checkpoint.
- Organism names were corrected in the scripts using `organisms.tsv`. Older reports retain inaccurate organism names and earlier metadata behavior. DNA-only group metrics do not depend on the spelling of metadata, but text embeddings and text-retrieval metrics can change when their input metadata changes. Do not assume all historical text scores are reproduced by the current scripts.
- Same-organism relevance is a proxy for retrieval evaluation, not proof of homology or useful reconstruction context. Full-metadata queries can contain information unavailable for a missing gene. The organism-only ablation still assumes a known organism present in the corpus.
- Model comparisons do not isolate every architectural or training difference. Historical causal explanations and extrapolations beyond tested sample sizes are interpretations, not additional measurements. Current reproduction notes summarize the recorded tables conservatively.
