"""Reading the simulated draft genomes, and picking the right contigs for a gap.

This module exists because the old harness got the second part wrong. ``test.py`` in
``gap-filler-agents/`` took ``contigs[0]`` and ``contigs[1]``, but the FASTA is sorted
longest-first, so those are ``contig31`` (190,651 bp, genome position 1298041-1488692)
and ``contig22`` (167,069 bp, 898138-1065207) -- two contigs that are not adjacent to
each other and have nothing to do with gap1. The harness then passed them alongside
gap1's 870 bp length, so the run was filling a gap that did not exist between those
sequences.

The headers carry what is needed to do it properly::

    >AP012051.1_contig1 length=5951  original_pos=0-5951
    >AP012051.1_contig2 length=13996 original_pos=6821-20817

and gap1 is ``start=5951, end=6821``. So the left flank is the contig whose ``end``
equals the gap's ``start``, and the right flank is the contig whose ``start`` equals the
gap's ``end``. Both match uniquely.
"""
from __future__ import annotations

import csv
import os
import re
from typing import Any, Iterator

from . import config

_POS = re.compile(r"original_pos=(\d+)-(\d+)")
_LEN = re.compile(r"length=(\d+)")


class DatasetError(RuntimeError):
    pass


def contigs_path(accession: str) -> str:
    return os.path.join(config.CONTIGS_DIR, f"{accession}_contigs.fasta")


def gaps_path(accession: str) -> str:
    return os.path.join(config.GAPS_DIR, f"{accession}_gaps.tsv")


def iter_fasta(path: str) -> Iterator[tuple[str, str]]:
    header: str | None = None
    chunks: list[str] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks).upper()


def parse_contigs(path: str) -> list[dict[str, Any]]:
    """Parse the contig FASTA into records carrying their genome coordinates.

    Order is file order, which is descending length -- deliberately preserved so a test
    can assert on the trap the old harness fell into.
    """
    contigs: list[dict[str, Any]] = []
    for header, sequence in iter_fasta(path):
        pos = _POS.search(header)
        declared = _LEN.search(header)
        contigs.append(
            {
                "id": header.split()[0],
                "header": header,
                "sequence": sequence,
                "length": len(sequence),
                "declared_length": int(declared.group(1)) if declared else len(sequence),
                "start": int(pos.group(1)) if pos else -1,
                "end": int(pos.group(2)) if pos else -1,
            }
        )
    if not contigs:
        raise DatasetError(f"no contigs parsed from {path}")
    return contigs


def parse_gaps(path: str) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    with open(path, "r") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            gaps.append(
                {
                    "gap_id": row["gap_id"],
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "length": int(row["length"]),
                    "sequence": (row.get("sequence") or "").strip().upper(),
                }
            )
    if not gaps:
        raise DatasetError(f"no gaps parsed from {path}")
    return gaps


def find_gap(gaps: list[dict[str, Any]], gap_id: str) -> dict[str, Any]:
    for gap in gaps:
        if gap["gap_id"] == gap_id:
            return gap
    raise DatasetError(f"{gap_id} not found among {len(gaps)} gaps")


def _exactly_one(matches: list[dict[str, Any]], what: str) -> dict[str, Any]:
    if not matches:
        raise DatasetError(f"no contig {what}")
    if len(matches) > 1:
        ids = ", ".join(c["id"] for c in matches)
        raise DatasetError(f"{len(matches)} contigs {what} ({ids}); expected exactly one")
    return matches[0]


def flanks_for_gap(
    contigs: list[dict[str, Any]], gap: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The contigs immediately left and right of ``gap``, by genome coordinate."""
    left = _exactly_one(
        [c for c in contigs if c["end"] == gap["start"]],
        f"ends at position {gap['start']} (the left edge of {gap['gap_id']})",
    )
    right = _exactly_one(
        [c for c in contigs if c["start"] == gap["end"]],
        f"starts at position {gap['end']} (the right edge of {gap['gap_id']})",
    )
    return left, right


def has_both_flanks(contigs: list[dict[str, Any]], gap: dict[str, Any]) -> bool:
    """Whether this gap is bordered by a contig on both sides.

    Not every gap is. In AP012051.1, ``gap33`` runs 1557603-1558103 while the assembly's
    last contig ends at 1557603, so it is a terminal gap with nothing to its right. It
    cannot be reconstructed by a model that conditions on both flanks, and the system
    declines it rather than inventing a right-hand context. 32 of the 33 gaps qualify.
    """
    ends = {c["end"] for c in contigs}
    starts = {c["start"] for c in contigs}
    return gap["start"] in ends and gap["end"] in starts


def reconstructable_gaps(
    contigs: list[dict[str, Any]], gaps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [g for g in gaps if has_both_flanks(contigs, g)]


def load_gap_case(accession: str, gap_id: str) -> dict[str, Any]:
    """Everything a run needs for one gap: both flanking contigs and the target length."""
    contigs = parse_contigs(contigs_path(accession))
    gap = find_gap(parse_gaps(gaps_path(accession)), gap_id)
    left, right = flanks_for_gap(contigs, gap)
    return {
        "accession": accession,
        "gap_id": gap["gap_id"],
        "gap_length": gap["length"],
        "gap_start": gap["start"],
        "gap_end": gap["end"],
        "true_sequence": gap["sequence"],
        "left_id": left["id"],
        "right_id": right["id"],
        "left": left["sequence"],
        "right": right["sequence"],
        "left_length": left["length"],
        "right_length": right["length"],
    }
