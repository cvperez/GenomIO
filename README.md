# GenomIO

## Overview

GenomIO is a research repository for filling gaps between genomic contigs with DNA language models. It contains baseline masked-language-model workflows, retrieval-augmented inference, embedding and retrieval experiments, and a separate multi-agent reconstruction system.

The implementation and stored reports support studying retrieval and workflow behavior. They do not establish reliable recovery of missing genomic sequence. Read the [known gap-filling and scoring limitations](docs/gap_filling_known_bugs.md) before interpreting reconstruction scores.

## Contributions in the `tfm-carmen-vázquez` Branch

The starting point, represented by `main`, includes model wrappers in `src/models/`, gap-filling and evaluation scripts in `src/core/`, and the LangChain planner in `src/agents/` with its harness in `gap-filler-agents/`. This branch adds the following work.

### Retrieval-Augmented Generation

- A uniform CDS corpus and preparation/inspection utilities in [rag_corpus_uniform/](rag_corpus_uniform) and [scripts/](scripts).
- DNA embedding with DNABERT-S, corpus loading, and a cached FAISS index in [src/rag/](src/rag), integrated into [src/core/gap_filler_rag.py](src/core/gap_filler_rag.py). The [retrieval notes](docs/dnabert_s_retrieval.md) explain the integration and its limits.
- Twelve [embedding and retrieval experiments](embedder_benchmark/README.md), covering DNA model comparisons, metadata encoders, retrieval routes, index scaling, recall curves, and DNA/metadata rank fusion. Each experiment contains its corresponding historical result report.

The metadata index and fusion are implemented in benchmark scripts, not in the application's retrieval path. The older LangChain planner retains its substring-based retriever.

### Multi-Agent System

[multiagent_system/](multiagent_system/README.md) adds Coordinator, Reconstruction, and Retrieval agents driven through the Claude Agent SDK. Their MCP tools communicate over the repository's A2A implementation. Reconstruction measures the model's available token budget, requests candidate batches, resolves retrieved row IDs locally, and records generation diagnostics. The Coordinator applies a length/alphabet/context acceptance gate.

The directory includes protocol and tool-contract tests, a local integration harness, and a SLURM deployment script. [Recorded validation results](multiagent_system/RESULTS.md) describe successful workflow checks but also limited reconstruction quality on the evaluated gap. Passing the gate does not prove biological correctness.

### Experimental Infrastructure

The branch adds corpus indexing utilities, experiment scripts and reports, and multi-agent state/trace instrumentation. The benchmark organization keeps scripts and outputs associated with their experiment. Raw benchmark artifacts and multi-agent runtime traces referenced by historical reports are not included in this checkout; the linked READMEs distinguish stored evidence from generated outputs.

## Repository Structure

```text
.
├── src/                    # Model wrappers, baseline workflows, planner, DNA retrieval
├── embedder_benchmark/     # 12 experiment folders, each with a script and documented results/
├── multiagent_system/      # Three-agent implementation, tests, deployment, and documentation
├── gap-filler-agents/      # Existing LangChain planner execution harness
├── rag_corpus_uniform/     # CDS corpus and organism-name mapping used by DNA retrieval
├── rag_corpus/             # Corpus retained for the legacy substring retriever
├── data/                   # Simulated contigs, gap tables, and CONTEXT/TARGET inputs
├── scripts/                # Corpus preparation, inspection, and index building
├── tests/                  # Original project tests
├── config/                 # Existing YAML configuration reference
├── environments/           # Existing environment specification files
├── notebooks/              # Existing model notebooks
├── docs/                   # Retrieval notes, known issues, and historical guides
├── results/                # Existing general output location
├── tech_report/            # Technical report source and archived PDFs
├── multi_agent_presentation.md
└── README.md
```

## Getting Started

Work from the repository root on `tfm-carmen-vázquez`. Install the project dependencies in your research environment:

```bash
pip install -r requirements.txt
pip install -e .
```

The historical multi-agent runs used Python 3.10.12 on Linux; deployment commands in its README use Bash and POSIX virtual-environment paths. Dependency lower bounds do not reproduce that environment exactly. Some older model code requires compatibility adjustments described in the experiment reports. The root YAML file is not the configuration source for the benchmark or multi-agent system; inspect their scripts and `multiagent_system/config.py` respectively.

Choose the workflow you need:

| Workflow | Starting point |
|---|---|
| Read or reproduce retrieval experiments | [Benchmark overview](embedder_benchmark/README.md) |
| Run the three-agent system | [Deployment and execution](multiagent_system/README.md#deployment-and-execution) |
| Build/check the application's DNA retrieval index | [Index builder](scripts/build_rag_index.py) |
| Inspect the original model wrappers | [src/models/](src/models) |

For the standalone RAG workflow, check the cache before building it:

```bash
python3 scripts/build_rag_index.py --check
python3 scripts/build_rag_index.py
```

The second command can download weights and embed the corpus; it is an expensive setup step. It writes `.cache/rag_index/`, which is not committed. After setup, the standalone comparison is invoked as:

```bash
python3 src/core/gap_filler_rag.py
```

It runs the configured non-RAG and RAG conditions and writes the CSV paths defined in the script. Review its settings first and read the [scoring limitations](docs/gap_filling_known_bugs.md); the stored legacy identity definition is not a reliable quality measure.

## Existing Workflows and Data

The original [gap filler](src/core/gap_filler.py) consumes contig FASTA files and gap TSV tables. The [batch evaluator](src/core/evaluation.py) reads text inputs with `CONTEXT:` and `TARGET:` sections and writes model/folder CSVs. Its entry point currently contains a Colab-specific input path that must be configured for another machine.

The older planner harness remains at [gap-filler-agents/test.py](gap-filler-agents/test.py). Its retrieval and generation tools are separate from the three-agent implementation; consult [known issues](docs/gap_filling_known_bugs.md) before using it as an experimental baseline. Model wrappers are present for DNABERT-2, GROVER, Gena-LM, and Nucleotide Transformer; their presence is not a claim that every model/version combination has been validated.

Corpus utilities include [uniform genome downloading](scripts/download_genomes_uniform.sh), [CDS extraction](scripts/extract_cds_from_gbff.py), and [CDS uniformity inspection](scripts/evaluate_cds_uniformity.py). The checked-in input data and historical evidence are retained.

## Testing and Documentation

The multi-agent README distinguishes lightweight tests from model-backed integration runs. To run its lightweight tests in an environment with the needed dependencies and without enabling `GENOMIO_HEAVY`:

```bash
python3 -m pytest multiagent_system/tests -c multiagent_system/pytest.ini
```

The original suite remains in [tests/](tests). It includes model-related code and should not be treated as a documentation-only smoke check.

- [Benchmark guides and evidence](embedder_benchmark/README.md)
- [Multi-agent architecture, tools, and execution](multiagent_system/README.md)
- [Historical multi-agent validation](multiagent_system/RESULTS.md)
- [DNA retrieval integration](docs/dnabert_s_retrieval.md)
- [Known gap-filling issues](docs/gap_filling_known_bugs.md)
- [Multi-agent technical report source](tech_report/multiagent_system.tex)
- [Multi-agent presentation notes](multi_agent_presentation.md)

The older [installation](docs/installation.md), [usage](docs/usage.md), and [API](docs/api.md) guides are retained for context. They contain examples and interface sketches that do not all match the current implementation; use the linked source and workflow-specific READMEs to verify commands and APIs.

## Contributing

Describe changes and validation in a pull request. Keep experimental settings and recorded results separate from documentation or organizational changes, and identify prerequisites that cannot be reproduced from the checkout.

## License, Contact, and Attribution

The repository is distributed under the [MIT License](LICENSE). Contact: Gnosis Research Center, grc@illinoistech.edu. Report issues through the [repository issue tracker](https://github.com/grc-iit/GenomIO/issues).

The existing project citation is retained to credit the starting work; it is not a citation for all additions on this branch:

```bibtex
@software{genomio2025,
  title = {GenIO: Leveraging LLM Advancements in the Detection, Analysis, and Filling of Gaps During DNA Sequencing},
  author = {Clara Aparicio Mendez},
  year = {2025},
  school = {Illinois Institute of Technology},
  institution = {Gnosis Research Center},
  url = {https://github.com/grc-iit/GenomIO}
}
```

Acknowledgments: the DNA model authors, LangChain contributors, and providers of the genomic datasets used by this repository.
