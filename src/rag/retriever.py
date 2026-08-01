# rag/retriever.py

import re

from .index import search_many

SEQ_REGEX = re.compile(r"[ACGTN\-]+", re.IGNORECASE)

# Flanks longer than this are trimmed before embedding. DNABERT-S truncates its input
# anyway, so a whole contig would silently reduce to its first slice; trimming makes the
# part that is actually compared the part nearest the gap, which is the informative one.
MAX_FLANK = 900


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


def _trim_flank(fragment: str, side: str) -> str:
    """Keep the MAX_FLANK bases closest to the gap."""
    if len(fragment) <= MAX_FLANK:
        return fragment
    return fragment[-MAX_FLANK:] if side == "left" else fragment[:MAX_FLANK]


def _flanks(sequence: str) -> list:
    """
    Split a gapped sequence into the flanks that sit either side of the gaps.

    'AAA---TTT---GGG' gives the left flank of gap 1, the piece between the two gaps, and
    the right flank of gap 2. Each is trimmed towards the gap it borders.
    """
    fragments = [f for f in sequence.split("---") if f]
    if not fragments:
        return []
    if len(fragments) == 1:
        return [_trim_flank(fragments[0], "left")]

    trimmed = [_trim_flank(fragments[0], "left")]
    trimmed += [f[:MAX_FLANK] for f in fragments[1:-1]]
    trimmed.append(_trim_flank(fragments[-1], "right"))
    return trimmed


# Retrieve best 3 matching sequence contexts given an input that may include labels
def retrieve_context(user_input: str, k: int = 3) -> str:
    """
    Find the corpus records most similar to the flanks of a gapped sequence.

    Similarity is cosine distance between DNABERT-S embeddings, not substring matching:
    a record comes back because it looks biologically like the query, not because it
    contains it character for character. The return format is unchanged from the
    substring version, so the agent tools that call this need no changes.
    """
    print(f"\nStarting context retrieval. Raw input preview: {user_input[:120].replace(chr(10),' ')}...")
    raw_seq = _extract_sequence(user_input)
    if not raw_seq:
        print("No sequence detected in input.")
        return "No matching sequence found in rag_corpus_uniform."

    flanks = _flanks(raw_seq)
    print(f"Querying with {len(flanks)} flank(s), lengths: {[len(f) for f in flanks]}")

    # Each flank is searched separately and the results merged, so a record that matches
    # either side of the gap can win. A record found by more than one flank is kept once,
    # at its best score.
    best = {}
    for hits in search_many(flanks, k=k):
        for score, record in hits:
            key = record["header"]
            if key not in best or score > best[key][0]:
                best[key] = (score, record)

    selected = sorted(best.values(), key=lambda pair: pair[0], reverse=True)[:k]

    if not selected:
        print("No matching sequence found")
        return "No matching sequence found in rag_corpus_uniform."

    print(f"Returning {len(selected)} match(es)")
    for score, record in selected:
        print(f"  {score:.3f}  {record['organism']}  ({len(record['sequence'])} bp)")

    result = ""
    for i, (score, record) in enumerate(selected):
        result += f"\nMatch {i+1}:\n>{record['header']}\n{record['sequence']}\n"

    return result.strip()
