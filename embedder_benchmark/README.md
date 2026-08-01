# Embedder benchmark

This folder holds the evidence for one decision: **which model should turn a DNA sequence
into a vector, so that similar sequences end up close together**. The answer the pipeline
now uses is `zhihan1996/DNABERT-S`.

Start with **`experiments_summary.md`**. It has a glossary written for someone who has not
done this before, a one line summary of each experiment, and the final architecture. The
per experiment write ups (`experiment_N_results.md`) have the full numbers, and the
matching `experiment_N_*.py` files reproduce them.

## The short version

The RAG system used to embed DNA with `all-MiniLM-L6-v2`, a model trained on English
sentences. Experiments 1 and 2 tested 16 DNA models against it. The ranking at n=5
(Experiment 1) turned out to be misleading, so Experiment 2 rebuilt it at 1,000 sequences
across 20 species, which is 24,500 same-species pairs instead of one:

| Rank | Model | Silhouette | Inter species distance | Training objective |
|---|---|---|---|---|
| 1 | **DNABERT_S** | **0.0486** | **0.646** | Contrastive |
| 2 | DNABERT_1 | 0.0236 | 0.160 | Masked language modelling |
| 4 | GROVER | 0.0128 | 0.275 | Masked language modelling |
| 7 | DNABERT_2 | 0.0029 | 0.215 | Masked language modelling |
| 9 to 12 | Nucleotide Transformer v2, 50M to 500M | negative | 0.98 to 0.99 | Masked language modelling |

Silhouette above zero means a record is on average closer to records of its own species
than to any other species, which is exactly the property retrieval needs. Thirteen of the
sixteen models score at or below 0.015.

The interesting part is *why* DNABERT-S wins, and the DNABERT family answers it. Going
from DNABERT_1 to DNABERT_2 changed only the tokenizer and made the score **8 times
worse**. Going from DNABERT_2 to DNABERT_S kept the tokenizer and added contrastive
training, and the score jumped **16 times** while inter species distance tripled. Scaling
a masked language model 10 times over (Nucleotide Transformer 50M to 500M) changes
nothing at all. What matters is the training objective, not the size and not the
tokenizer.

Experiments 5 to 10 then built retrieval on top and measured it. With FAISS at 1,000
records, DNA to DNA precision at rank 1 is 0.655, and the approximate HNSW index agrees
with exhaustive search on 100% of top-1 hits.

## What shipped and what did not

Experiments 3 and 4 show that DNABERT-S must **not** be used on the metadata text: its DNA
tokenizer maps English to unknown tokens, so every metadata string collapses to nearly the
same vector, and the silhouette goes negative at scale. That motivates a second index over
metadata with a text encoder, described in `bridge_architecture.md`, and Experiments 11
and 12 select and tune it.

Only the DNA index shipped into `src/rag/`. The metadata index and the rank fusion from
Experiments 11 and 12 score higher on retrieval but are a much larger change, so they are
documented here and left for later work.

## A correction

Earlier revisions of experiments 4, 6, 8, 10, 11 and 12 hardcoded a table of organism
names, and several entries in it were wrong: `GCF_000013045` is *Salinibacter ruber*, not
*Streptococcus pneumoniae*, and `GCF_000020225` is *Akkermansia muciniphila*, not
*Mycobacterium tuberculosis*. The seven `GCA_*` metagenome assembled genomes were labelled
*Lactobacillus sp.* when they are candidate phyla radiation bacteria.

The labels were internally consistent, one distinct label per species file, so **every
silhouette, precision and recall number in these documents is unaffected** by the error.
What was wrong is only the names printed beside them. The scripts now read the real names
from `rag_corpus_uniform/organisms.tsv`; the prose in the older `experiment_N_results.md`
files has not been rewritten, so treat organism names appearing there with suspicion and
the numbers as sound.

## Running them

From the repository root, with the corpus in place:

```bash
python3 embedder_benchmark/experiment_2_embedding_benchmark.py
python3 embedder_benchmark/experiment_12_recall_fusion.py
python3 embedder_benchmark/experiment_12_recall_fusion.py ablation
```

Each script writes its own CSVs, figures and log back into this folder. They are CPU only
and slow, from minutes to hours depending on how many models a given experiment loads,
and they download model weights from HuggingFace on first run. Three models in Experiment
1 and 2 (Nucleotide Transformer 2500M, Caduceus, HyenaDNA) need a GPU and were run
separately on Colab, then merged in from `exp1_gpu.csv` and `exp2_gpu.csv`.

Beyond the project requirements these need `matplotlib`, `seaborn`, `scikit-learn` and
`sentence-transformers`, all of which are already in `requirements.txt`.

Figures and the cached `.npy` and `.faiss` artifacts are not committed; the scripts
regenerate them.
