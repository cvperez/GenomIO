"""Organism -> FAISS row range, for the restricted retrieval path.

Each ``.fna`` in ``rag_corpus_uniform/`` holds exactly one organism, and
``src/rag/corpus.py`` appends records file by file in sorted filename order. So every
organism occupies one *contiguous* block of row IDs -- 20 organisms, 20 blocks. That
property is what makes restriction cheap.

A warning learned the hard way, and the reason this module post-filters by default:
the index is an ``IndexHNSWFlat``. Handing HNSW a sparse ``faiss.IDSelectorBatch``
returns all ``-1`` -- graph traversal starts from the entry point and only *accepts*
selected nodes, so with a restrictive filter the greedy walk never reaches anything.
Raising ``efSearch`` does not save it. The safe move is to search wide and drop what
does not belong; ``IDSelectorRange`` (which HNSW handles better, since the blocks are
contiguous) is available as a tagged fallback and its results are validated.
"""
from __future__ import annotations

import threading
from typing import Any

from .corpus_store import CorpusStore, get_store


class OrganismIndex:
    """Contiguous ``[lo, hi)`` row block per organism, built once at startup."""

    def __init__(self, blocks: dict[str, tuple[int, int]], total_rows: int) -> None:
        self.blocks = blocks
        self.total_rows = total_rows
        self._lower = {name.lower(): name for name in blocks}

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, store: CorpusStore | None = None) -> "OrganismIndex":
        store = store or get_store()
        total = len(store)
        blocks: dict[str, tuple[int, int]] = {}
        current: str | None = None
        start = 0
        for row in range(total):
            organism = store.organism(row)
            if organism != current:
                if current is not None:
                    blocks[current] = (start, row)
                current = organism
                start = row
        if current is not None:
            blocks[current] = (start, total)
        return cls(blocks, total)

    # ------------------------------------------------------------------ lookup
    def names(self) -> list[str]:
        return sorted(self.blocks)

    def resolve(self, organism: str) -> str | None:
        """Match an organism name case-insensitively, then by unique prefix.

        Prefix matching exists because callers write "Fusobacterium animalis" while the
        corpus may carry a longer NCBI name. Ambiguous prefixes resolve to nothing
        rather than to an arbitrary choice.
        """
        if not organism:
            return None
        key = organism.strip().lower()
        if key in self._lower:
            return self._lower[key]
        hits = [full for low, full in self._lower.items() if low.startswith(key)]
        if len(hits) == 1:
            return hits[0]
        hits = [full for low, full in self._lower.items() if key in low]
        return hits[0] if len(hits) == 1 else None

    def block(self, organism: str) -> tuple[int, int] | None:
        resolved = self.resolve(organism)
        return self.blocks.get(resolved) if resolved else None

    def size(self, organism: str) -> int:
        found = self.block(organism)
        return (found[1] - found[0]) if found else 0

    def contains(self, organism: str, row_id: int) -> bool:
        found = self.block(organism)
        return bool(found) and found[0] <= row_id < found[1]

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"organism": name, "lo": lo, "hi": hi, "rows": hi - lo}
            for name, (lo, hi) in sorted(self.blocks.items(), key=lambda kv: kv[1])
        ]


_INDEX: OrganismIndex | None = None
_LOCK = threading.Lock()


def get_organism_index() -> OrganismIndex:
    global _INDEX
    with _LOCK:
        if _INDEX is None:
            _INDEX = OrganismIndex.build()
        return _INDEX


def set_organism_index(index: OrganismIndex | None) -> None:
    global _INDEX
    with _LOCK:
        _INDEX = index
