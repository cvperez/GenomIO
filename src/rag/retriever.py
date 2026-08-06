# rag/retriever.py

import os
import re
from pathlib import Path
from typing import List, Dict

SEQ_REGEX = re.compile(r"[ACGTN\-]+", re.IGNORECASE)

def _extract_sequence(text: str) -> str:
    """
    Extract the first long stretch that looks like a DNA sequence (ACGTN and dashes).
    Falls back to stripping spaces if nothing is found.
    """
    if not isinstance(text, str):
        return ""
    candidates = SEQ_REGEX.findall(text.upper())
    if not candidates:
        return text.strip().upper()
    # pick the longest candidate to be safe
    return max(candidates, key=len).upper()

# Load all sequences from .fna / .fasta files in the specified directory
def load_all_sequences(directory: str, exts: tuple = (".fna", ".fasta")) -> List[Dict]:
    sequences = []
    print(f"Loading sequences from directory: {directory}")
    for file in os.listdir(directory):
        if file.lower().endswith(exts):
            path = Path(directory) / file
            print(f"Reading file: {file}")
            with open(path, "r") as f:
                meta = ""
                sequence = ""
                for line in f:
                    if line.startswith(">"):
                        # save previous sequence before starting a new one
                        if sequence:
                            sequences.append({"meta": meta, "sequence": sequence})
                            sequence = ""
                        meta = line.strip()
                    else:
                        sequence += line.strip().upper()  # normalize sequence to uppercase
                if sequence:
                    sequences.append({"meta": meta, "sequence": sequence})
    print(f"Total sequences loaded: {len(sequences)}")
    return sequences

# Retrieve best 3 matching sequence contexts given an input that may include labels
def retrieve_context(user_input: str) -> str:
    print(f"\nStarting context retrieval. Raw input preview: {user_input[:120].replace(chr(10),' ')}...")
    raw_seq = _extract_sequence(user_input)
    if not raw_seq:
        print("No sequence detected in input.")
        return "No matching sequence found in rag_corpus."

    fragments = [frag for frag in raw_seq.split("---") if frag]
    print(f"Extracted fragments: {fragments}")

    all_seqs = load_all_sequences("rag/rag_corpus", exts=(".fna", ".fasta"))

    exact_matches = []
    partial_matches = []

    for entry in all_seqs:
        seq = entry["sequence"]
        # exact: all fragments present
        if all(f in seq for f in fragments):
            exact_matches.append(entry)
        # partial: at least one fragment present
        elif any(f in seq for f in fragments):
            partial_matches.append(entry)

        if len(exact_matches) >= 3:
            break

    # choose up to 3: exact first, then partial
    selected = exact_matches[:3]
    if len(selected) < 3:
        selected += partial_matches[: 3 - len(selected)]

    if not selected:
        print("No matching sequence found")
        return "No matching sequence found in rag_corpus."

    print(f"Returning {len(selected)} match(es)")
    result = ""
    for i, entry in enumerate(selected):
        result += f"\nMatch {i+1}:\n{entry['meta']}\n{entry['sequence']}\n"

    return result.strip()
