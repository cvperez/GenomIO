# rag/corpus.py
"""
Reads rag_corpus_uniform/ into a canonical, order stable list of CDS records.

Order matters: the FAISS index stores vectors by row number, and a search returns row
numbers rather than records. Row k of the index is record k of this list, so the two only
line up as long as this function walks the files in the same order every time. Files are
sorted by name and records are kept in file order.

Each record is a plain dict:

    sequence   the nucleotides, uppercased
    header     the original FASTA description line, without the leading '>'
    accession  assembly accession without version suffix, e.g. GCF_000008125
    organism   species name, resolved through rag_corpus_uniform/organisms.tsv
    metadata   'organism=... gene=... protein=...' built from the header fields

The metadata string is assembled here but is deliberately not fed to the gap filling
model. It exists so that retrieval results can be printed and inspected by a human, and
because a future metadata index would need it. See docs/dnabert_s_retrieval.md.
"""

import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORPUS_DIR = os.path.join(REPO_ROOT, "rag_corpus_uniform")
ORGANISM_TABLE = "organisms.tsv"

_BRACKETED = re.compile(r"\[([^\]]+)\]")

# Header fields that describe file layout rather than biology, dropped from the metadata
# string because they add length without adding meaning.
_SKIP_PREFIXES = ("gbkey=", "location=")


def accession_from_filename(filename):
    """
    GCF_000008125.1_ASM812v1_cds_from_genomic.fna -> GCF_000008125

    The version suffix is stripped so the key survives NCBI bumping an assembly from .1
    to .2 without the organism table going stale.
    """
    parts = os.path.basename(filename).split("_")
    if len(parts) < 2:
        return os.path.splitext(os.path.basename(filename))[0]
    return parts[0] + "_" + parts[1].split(".")[0]


def load_organisms(corpus_dir=None):
    """Read organisms.tsv into {accession: organism}. Missing file gives an empty map."""
    path = os.path.join(corpus_dir or CORPUS_DIR, ORGANISM_TABLE)
    organisms = {}
    if not os.path.exists(path):
        print(f"[WARNING] {path} not found, records will fall back to accession names")
        return organisms

    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) >= 2:
                organisms[fields[0]] = fields[1]
    return organisms


def build_metadata_text(description, organism):
    """'organism=Thermus thermophilus gene=dnaA protein=...' from a FASTA header."""
    fields = [f for f in _BRACKETED.findall(description)
              if not f.startswith(_SKIP_PREFIXES)]
    return f"organism={organism} " + " ".join(fields)


def corpus_files(corpus_dir=None):
    """Sorted list of .fna paths. Sorted, because index row order depends on it."""
    directory = corpus_dir or CORPUS_DIR
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"RAG corpus not found at {directory}. It ships with the repository; if it is "
            f"missing, rebuild it with scripts/download_genomes_uniform.sh"
        )
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".fna")
    )


def iter_fasta(path):
    """Yield (header, sequence) pairs from a FASTA file, in file order."""
    header = None
    chunks = []
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks).upper()


def load_records(corpus_dir=None, min_length=None, max_length=None):
    """
    Parse the whole corpus. Roughly 43,500 records and a few seconds.

    min_length / max_length filter by nucleotide count and default to no filtering. The
    benchmark in embedder_benchmark/ sampled 300 to 900 bp to keep species balanced;
    production retrieval keeps everything, since a record only competes on similarity.
    """
    organisms = load_organisms(corpus_dir)
    records = []

    for path in corpus_files(corpus_dir):
        accession = accession_from_filename(path)
        organism = organisms.get(accession, accession)

        for header, sequence in iter_fasta(path):
            if not sequence:
                continue
            if min_length is not None and len(sequence) < min_length:
                continue
            if max_length is not None and len(sequence) > max_length:
                continue
            records.append({
                "sequence": sequence,
                "header": header,
                "accession": accession,
                "organism": organism,
                "metadata": build_metadata_text(header, organism),
            })

    return records
