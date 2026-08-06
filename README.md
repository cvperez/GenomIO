# GenomIO

A comprehensive framework for genomic gap filling using Large Language Models (LLMs). GenomIO leverages state-of-the-art DNA language models to predict and fill gaps in genomic sequences, with support for multiple model architectures and RAG-enhanced inference.

## 🧬 Features

- **Multiple DNA Language Models**: Support for DNABERT-2, GROVER, Gena-LM, and Nucleotide Transformer
- **RAG-Enhanced Inference**: Retrieval-Augmented Generation using genomic corpus for context
- **Agent-Based Architecture**: LangChain-powered agents for intelligent gap filling
- **Comprehensive Evaluation**: Built-in evaluation metrics and benchmarking tools  
- **Modular Design**: Clean, extensible codebase ready for research and production
- **Easy Data Management**: Automated genomic data downloading and processing

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GenomIO.git
cd GenomIO

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Basic Usage
Run the agent pipeline directly from the command line:

```bash
cd gap-filler-agents
python test.py
```

**Payload passed to the agent (`planner`)**:

```json
{
  "sequence": "ATGCGT...---...GCTAGC", 
  "gap_length": 500,
  "meta": {
    "gap_id": "AP012051.1_gap1",
    "contig1_header": "AP012051.1_contig1",
    "contig2_header": "AP012051.1_contig2"
  }
}
```
**Example output**:
```bash
[DEBUG] Gap info: {'gap_id': 'AP012051.1_gap1', 'start': 1000, 'end': 1500, 'length': 500, 'gap_sequence': '...'}
[INFO] Using full contigs:
  contig1: AP012051.1_contig1 (len=45231)
  contig2: AP012051.1_contig2 (len=38942)
[INFO] Sequence passed to the agent:
  Total length (without dashes): 84173
  Start: ATGCGTACGATCGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA...
  End: ...GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
[DEBUG] Calling plan() with gap_length=500
[Final Output]: {'predicted_sequence': 'ATCG...GCTA', 'confidence': 0.87}
```

## 📁 Project Structure

```
GenomIO/
├── src/                          # Source code
│   ├── models/                   # DNA language model implementations
│   │   ├── dnabert2.py          # DNABERT-2 model
│   │   ├── grover.py            # GROVER model  
│   │   ├── gena_lm.py           # Gena-LM model
│   │   └── nucleotide_transformer.py
│   ├── agents/                   # LangChain agents
│   │   ├── planner.py           # Planning agent
│   │   └── tools/               # Agent tools
│   ├── rag/                      # RAG implementation
│   │   ├── dnabert_s.py         # DNABERT-S sequence embedder
│   │   ├── corpus.py            # Reads rag_corpus_uniform into records
│   │   ├── index.py             # FAISS index, built once and cached
│   │   ├── retriever.py         # Similarity search used by the agent
│   │   └── gap_filler.py        # RAG-enhanced gap filling
│   ├── core/                     # Core functionality
│   │   ├── gap_filler.py        # Main gap filling logic
│   │   ├── gap_filler_rag.py    # RAG-enhanced version
│   │   └── evaluation.py        # Model evaluation
│   └── utils/                    # Utilities
│       ├── data_download.py     # Data downloading
│       └── species_analysis.py  # Species analysis tools
├── data/                         # Data directory
│   ├── simulated_draft_genomes/ # Training/test data
│   └── test_sequences/          # Test sequences
├── rag_corpus_uniform/           # RAG corpus, one NCBI gene per record
├── rag_corpus/                   # Superseded corpus, kept for reference
├── embedder_benchmark/           # Why DNABERT-S: 12 experiments, 16 models
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Test suite
├── config/                       # Configuration files
├── docs/                         # Documentation
└── results/                      # Output results
```

## 🧪 Models Supported

### DNABERT-2
- State-of-the-art DNA language model
- 117M parameters
- Pre-trained on genomic sequences

### GROVER
- Graph-based molecular representation
- Optimized for small molecule and DNA sequences

### Gena-LM BigBird
- Long-sequence modeling capability
- 4096 token context length
- Attention mechanism optimized for genomics

### Nucleotide Transformer  
- Transformer architecture for nucleotide sequences
- 500M parameters
- Pre-trained on 1000 genomes

## 🔬 RAG Integration

GenomIO includes a Retrieval-Augmented Generation system that:

- Embeds genomic sequences with **DNABERT-S** (`zhihan1996/DNABERT-S`), selected over 15
  other DNA models by the benchmark in `embedder_benchmark/`
- Searches them with a FAISS index over `rag_corpus_uniform/`, a corpus in which every
  record is one NCBI annotated gene, so no chunking is applied anywhere
- Retrieves relevant genomic context for gap filling
- Supports custom genomic databases

The index is built once and cached, since it is too large to commit:

```bash
python3 scripts/build_rag_index.py
```

This applies to `src/core/gap_filler_rag.py`, the standalone comparison script. **The
LangChain agent is unchanged**: it still finds context by substring search over
`rag_corpus/` and does not use the index.

**See [`docs/dnabert_s_retrieval.md`](docs/dnabert_s_retrieval.md)** for what this replaced
and why, written for readers new to the project, including what switching the agent over
would involve.

## 🧪 Evaluation

> **Known bugs, not yet fixed.** Three problems in the gap filling and scoring code are
> documented in [`docs/gap_filling_known_bugs.md`](docs/gap_filling_known_bugs.md), with the
> evidence for each and the change that would fix it. The most important one is that the
> identity score currently rates random DNA at 64%, higher than any result the pipeline has
> produced, so **the numbers in `results_AP012051.1_*.csv` should not be used**. Read that
> document before drawing conclusions from any gap filling output.

The framework includes comprehensive evaluation tools:

- **Accuracy Metrics**: Per-nucleotide and sequence-level accuracy
- **Biological Validity**: Checks for valid DNA sequences and ORF preservation
- **Benchmark Datasets**: Standardized test sets for fair comparison
- **Model Comparison**: Side-by-side evaluation of different models

## 📊 Usage Examples

### 1. Basic Gap Filling

Run the standard gap filling pipeline:

```bash
cd src/core
python gap_filler.py
```

- Loads contigs and gaps (.fasta + .tsv)
- Predicts the missing sequence using [MASK] tokens
- Saves results to results_[accession-number].csv
### 2. RAG-Enhanced Gap Filling

Run the RAG pipeline (retrieves context from the .fna files in rag_corpus_uniform/):

```python
python src/core/gap_filler_rag.py
```
- Load the cached DNABERT-S FAISS index, building it first if it is missing
- Run without RAG → saves results_[accession-number]_no_rag.csv
- Run with RAG → saves results_[accession-number]_rag.csv

### 3. Batch Processing

Run the evaluation script to benchmark one or more test folders with the selected model(s).  
The script reads `.txt` files with the format:
```python
CONTEXT:
<left_context_sequence>

TARGET:
<target_gap_sequence>
```
#### Configure `evaluation.py`

Edit the **ENTRY POINT** at the bottom of `evaluation.py` to set your folders and models:

```python
# ENTRY POINT
if __name__ == "__main__":
    test_folders = [
        "data/test_sequences/5000bp"  # <-- your folder(s) with CONTEXT/TARGET .txt files
    ]

    # Choose one or more model names (HF repo ids)
    # Examples:
    # "AIRI-Institute/gena-lm-bigbird-base-t2t"
    # "zhihan1996/DNABERT-2-117M"
    # "PoetschLab/GROVER"
    # "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"
    model_names = ["AIRI-Institute/gena-lm-bigbird-base-t2t"]

    for model_name in model_names:
        main(test_folders, model_name)
```
Run
```python
cd src/core
python evaluation.py
``` 
Results are written to:
```bash
./results/<model_name>/<folder_name>_results.csv
```

### 4. Agent-based Pipeline


This example shows how the **planner agent** orchestrates two tools to retrieve context and fill a genomic gap.  
It uses your `test.py`, `rag/gap_filler.py`, `rag/retriever.py`, and `agents/planner.py`.

---

#### How it works

1. **Load inputs (test.py)**  
   - Reads the target gap from `simulated_draft_genomes/gaps/AP012051.1_gaps.tsv` (fields: `gap_id, start, end, length, sequence`).  
   - Reads the first two contigs from `simulated_draft_genomes/contigs/AP012051.1_contigs.fasta`.  
   - Builds an input sequence with a **gap marker** `---` between contig1 and contig2 flanks:
     ```
     <...contig1_tail>---<contig2_head>
     ```
   - Prepares the payload:
     ```json
     {
       "sequence": "ATGCGT...---...GCTAGC",
       "gap_length": 500,
       "meta": {
         "gap_id": "AP012051.1_gap1",
         "contig1_header": "AP012051.1_contig1",
         "contig2_header": "AP012051.1_contig2"
       }
     }
     ```

2. **Plan & Tools (agents/planner.py)**  
   - Creates a **tool-calling agent** with a **strict order**:
     1) `context_tool` → retrieves up to 3 matching sequences from `rag_corpus` (via `rag/retriever.py`).  
     2) `gap_filler_tool` → fills the `---` using a masked LM (**Gena-LM BigBird**) (via `rag/gap_filler.py`).
   - **Critical rules enforced in the system prompt**:
     - The agent **must** pass the user-provided `gap_length` **exactly** to the gap filler.  
     - `context_tool` → first; `gap_filler_tool` → second.

3. **Context retrieval (rag/retriever.py)**  
   - Extracts DNA-like text from the `sequence` (ACGTN and dashes).  
   - Splits by `---` and searches `.fna`/`.fasta` files in `rag_corpus` for **exact** or **partial** substring matches.  
   - Returns up to **3** best matches (metadata + sequence) as plain text for the gap filler.  
   - This path is unchanged. The DNABERT-S index is used by `src/core/gap_filler_rag.py`, not by the agent; see [`docs/dnabert_s_retrieval.md`](docs/dnabert_s_retrieval.md) for why switching the agent over is a larger change than it looks.

4. **Gap filling (rag/gap_filler.py)**  
   - Loads `AIRI-Institute/gena-lm-bigbird-base-t2t`.  
   - Builds a masked prompt with **`[MASKS]`** between left/right contexts (optionally augmented with retrieved text).  
   - Iteratively adjusts the number of `[MASK]` tokens to approach `gap_length` (±3 nt tolerance).

5. **Final output**  
   - The planner returns a concise result (e.g., predicted sequence and confidence/notes).  
   - `test.py` prints the final output.

Run

```bash
cd genomio/gap-filler-agents
python test.py
```

## 📚 Data

### Downloading Genomic Data

Use the built-in downloader to fetch genomic sequences:

```python
from src.utils.data_download import GenomeDownloader

downloader = GenomeDownloader()
downloader.download_species_genomes([
    "Escherichia coli",
    "Staphylococcus aureus"
])
```

### Data Format

- **Input**: FASTA files with gap markers (`---`)
- **Output**: Filled FASTA sequences
- **Metadata**: TSV files with gap information

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test categories  
pytest tests/test_models.py
pytest tests/test_rag.py
pytest tests/test_agents.py
```

## 📖 Documentation

- [Installation Guide](docs/installation.md)
- [API Documentation](docs/api.md)
- [Usage Examples](docs/usage.md)
- [Model Comparison](docs/model_comparison.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **GRC**: Gnosis Research Center)
- **Email**: grc@illinoistech.edu
- **Issues**: [GitHub Issues](https://github.com/grc-iit/GenomIO/issues)

## 🙏 Acknowledgments

- DNABERT-2 team for their excellent pre-trained models
- LangChain for the agent framework
- The bioinformatics community for genomic datasets

## 📈 Citation

If you use GenomIO in your research, please cite:

```bibtex
@software{genomio2025,
  title   = {GenIO: Leveraging LLM Advancements in the Detection, Analysis, and Filling of Gaps During DNA Sequencing},
  author  = {Clara Aparicio Mendez},
  year    = {2025},
  school  = {Illinois Institute of Technology},
  institution = {Gnosis Research Center},
  url={https://github.com/grc-iit/GenomIO}
}
```
