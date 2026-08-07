"""Shared fixtures.

Two fakes keep the contract tests fast and dependency-free: a corpus store that answers
from a dict instead of the 59 MB records file, and a tokenizer whose bases-per-token
ratio is fixed so token arithmetic is exactly predictable.

Anything needing the real index, the real corpus or torch is marked ``heavy`` and skipped
unless ``GENOMIO_HEAVY=1``.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

HEAVY = os.environ.get("GENOMIO_HEAVY", "") == "1"

# The protocol layer, the gate and the dataset helpers deliberately have no dependency on
# the Claude Agent SDK, so they can be tested with the plain system interpreter. The tool
# and naming tests do need it. Skip rather than error, so `python3 -m pytest` outside the
# venv still exercises everything it can.
try:
    import claude_agent_sdk  # noqa: F401

    HAS_SDK = True
except ImportError:  # pragma: no cover
    HAS_SDK = False

collect_ignore = (
    []
    if HAS_SDK
    else ["test_tool_contracts.py", "test_naming.py", "test_agent_runtime.py"]
)


def pytest_collection_modifyitems(config, items):
    if HEAVY:
        return
    skip = pytest.mark.skip(reason="needs the real corpus/index or torch; set GENOMIO_HEAVY=1")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip)


# --------------------------------------------------------------------------- fakes
class FakeCorpusStore:
    """Row -> record, from a list. Same surface as tools.corpus_store.CorpusStore."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self.loaded = True

    def load(self):
        return self

    def close(self) -> None:
        return None

    def __len__(self) -> int:
        return len(self._records)

    def get(self, row_id: int) -> dict[str, Any]:
        if not (0 <= row_id < len(self._records)):
            raise IndexError(f"row {row_id} outside [0, {len(self._records)})")
        return dict(self._records[row_id])

    def get_many(self, row_ids):
        return [self.get(int(r)) for r in row_ids]

    def sequence(self, row_id: int) -> str:
        return self.get(row_id)["sequence"]

    def organism(self, row_id: int) -> str:
        return self.get(row_id)["organism"]

    def summary(self, row_id: int) -> dict[str, Any]:
        from multiagent_system.tools.corpus_store import protein_of

        record = self.get(row_id)
        return {
            "row_id": int(row_id),
            "organism": record["organism"],
            "accession": record.get("accession", ""),
            "protein": protein_of(record),
            "length": len(record["sequence"]),
            "header": record.get("header", ""),
        }


class FakeTokenizer:
    """Deterministic stand-in for the GENA-LM tokenizer.

    ``bases_per_token`` characters map to one token, so a caller can compute exactly how
    many candidates should be admitted for a given budget.
    """

    mask_token = "[MASK]"
    mask_token_id = 4

    def __init__(self, bases_per_token: int = 6) -> None:
        self.bases_per_token = bases_per_token

    def _count(self, text: str) -> int:
        # +2 for the [CLS]/[SEP] pair the real tokenizer adds.
        return max(1, -(-len(text) // self.bases_per_token)) + 2

    def __call__(self, text, **kwargs):
        ids = list(range(self._count(text)))
        if kwargs.get("truncation") and kwargs.get("max_length"):
            ids = ids[: kwargs["max_length"]]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def encode(self, text, **kwargs):
        return self(text, **kwargs)["input_ids"]


def make_records(count: int = 12, organisms: tuple[str, ...] = ("Alpha", "Beta", "Gamma")):
    """Contiguous organism blocks, mirroring how the real corpus is laid out."""
    per = max(1, count // len(organisms))
    records: list[dict[str, Any]] = []
    for i in range(count):
        organism = organisms[min(i // per, len(organisms) - 1)]
        length = 120 + (i % 5) * 60
        records.append(
            {
                "sequence": "ACGT" * (length // 4),
                "header": f"lcl|rec_{i} [locus_tag=L{i}] [protein=protein {i}] [gbkey=CDS]",
                "accession": f"ACC_{organism}",
                "organism": organism,
                "metadata": f"organism={organism} protein=protein {i}",
            }
        )
    return records


@pytest.fixture
def fake_records():
    return make_records()


@pytest.fixture
def fake_store(fake_records):
    from multiagent_system.tools import corpus_store

    store = FakeCorpusStore(fake_records)
    corpus_store.set_store(store)
    yield store
    corpus_store.set_store(None)


@pytest.fixture
def fake_organisms(fake_store):
    from multiagent_system.tools import organism_index

    index = organism_index.OrganismIndex.build(fake_store)
    organism_index.set_organism_index(index)
    yield index
    organism_index.set_organism_index(None)


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()
