"""Random access to the corpus records behind the FAISS index.

The retrieval index maps a query to *row numbers*. Turning a row number back into a
sequence is what this module does, and it is the only place corpus nucleotides enter the
system: the Retrieval agent returns row IDs and scores, the Reconstruction agent resolves
them here. That boundary is what stops corpus DNA from crossing an agent hop as an opaque
blob, which is how the old pipeline lost track of what the model was actually shown.

Why not just ``json.loads`` the whole file: ``.cache/rag_index/records.jsonl`` is 59 MB
and 43,575 records; held as Python dicts that is roughly 250 MB per process, and two
agents need it. Instead we scan once for line offsets (~1 s, a 350 kB int64 array) and
seek per row. A lookup is around 50 microseconds, which is nothing next to a forward pass.

The row -> record mapping is positional and guaranteed by ``src/rag/corpus.py``: files
sorted by name, records appended in file order. Line k of records.jsonl is FAISS row k.
If that ever drifts, every result is confidently wrong rather than obviously broken --
hence :func:`corpus_fingerprint`, which both agents publish in their agent card so the
Coordinator can compare them before delegating.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Iterable

import numpy as np

from .. import config


class CorpusStore:
    """Offset-indexed read-only view of ``records.jsonl``."""

    def __init__(self, records_path: str | None = None) -> None:
        self.records_path = records_path or config.RECORDS_PATH
        self._offsets: np.ndarray | None = None
        self._lock = threading.RLock()
        self._fh = None

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> "CorpusStore":
        """Build the offset table. Idempotent; safe to call at agent startup."""
        with self._lock:
            if self._offsets is not None:
                return self
            if not os.path.exists(self.records_path):
                raise FileNotFoundError(
                    f"{self.records_path} is missing. Build the retrieval index first: "
                    f"python3 scripts/build_rag_index.py"
                )
            offsets: list[int] = []
            with open(self.records_path, "rb") as handle:
                offset = 0
                for line in handle:
                    if line.strip():
                        offsets.append(offset)
                    offset += len(line)
            self._offsets = np.asarray(offsets, dtype=np.int64)
            self._fh = open(self.records_path, "rb")
            return self

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    @property
    def loaded(self) -> bool:
        return self._offsets is not None

    def __len__(self) -> int:
        if self._offsets is None:
            return 0
        return int(self._offsets.shape[0])

    # ------------------------------------------------------------------ access
    def get(self, row_id: int) -> dict[str, Any]:
        """Return the record at ``row_id``.

        Keys are exactly those written by ``src/rag/corpus.py``:
        ``sequence``, ``header``, ``accession``, ``organism``, ``metadata``.
        There is no ``product`` key -- the protein name lives inside ``header`` as
        ``[protein=...]``; use :func:`protein_of` to pull it out.
        """
        if self._offsets is None:
            raise RuntimeError("CorpusStore.load() has not been called")
        if not (0 <= row_id < len(self)):
            raise IndexError(f"row {row_id} outside [0, {len(self)})")
        with self._lock:
            self._fh.seek(int(self._offsets[row_id]))
            line = self._fh.readline()
        return json.loads(line)

    def get_many(self, row_ids: Iterable[int]) -> list[dict[str, Any]]:
        return [self.get(int(r)) for r in row_ids]

    def sequence(self, row_id: int) -> str:
        return str(self.get(row_id).get("sequence", ""))

    def organism(self, row_id: int) -> str:
        return str(self.get(row_id).get("organism", ""))

    def summary(self, row_id: int) -> dict[str, Any]:
        """Metadata only -- never the nucleotides. Backs the describe_rows tool."""
        record = self.get(row_id)
        return {
            "row_id": int(row_id),
            "organism": record.get("organism", ""),
            "accession": record.get("accession", ""),
            "protein": protein_of(record),
            "length": len(record.get("sequence", "")),
            "header": record.get("header", ""),
        }


def protein_of(record: dict[str, Any]) -> str:
    """Pull ``[protein=...]`` out of a record header. Empty string when absent."""
    header = str(record.get("header", ""))
    marker = "[protein="
    start = header.find(marker)
    if start < 0:
        return ""
    end = header.find("]", start)
    return header[start + len(marker): end] if end > start else ""


def corpus_fingerprint(
    manifest_path: str | None = None, records_path: str | None = None
) -> str:
    """A short hash identifying which corpus snapshot this process is reading.

    Derived from the index manifest plus the byte size of records.jsonl. Both the
    Retrieval and Reconstruction agents publish it; the Coordinator refuses to delegate
    when they disagree, because a mismatch silently changes which gene every row ID
    refers to.
    """
    manifest_path = manifest_path or config.MANIFEST_PATH
    records_path = records_path or config.RECORDS_PATH
    digest = hashlib.sha256()
    try:
        with open(manifest_path, "rb") as handle:
            digest.update(handle.read())
    except OSError:
        digest.update(b"no-manifest")
    try:
        digest.update(str(os.path.getsize(records_path)).encode("utf-8"))
    except OSError:
        digest.update(b"no-records")
    return digest.hexdigest()[:16]


_STORE: CorpusStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> CorpusStore:
    """Process-wide singleton, so two tools in one agent share one offset table."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = CorpusStore().load()
        return _STORE


def set_store(store: CorpusStore | None) -> None:
    """Injection point for tests, which use a fake store instead of the 59 MB file."""
    global _STORE
    with _STORE_LOCK:
        _STORE = store
