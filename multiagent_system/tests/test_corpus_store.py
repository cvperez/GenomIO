"""Row-ID resolution and the organism blocks that make restricted retrieval possible."""
from __future__ import annotations

import json

import pytest

from multiagent_system.tools import corpus_store
from multiagent_system.tools.corpus_store import CorpusStore, protein_of
from multiagent_system.tools.organism_index import OrganismIndex

from .conftest import make_records


@pytest.fixture
def records_file(tmp_path):
    records = make_records(count=9, organisms=("Alpha", "Beta", "Gamma"))
    path = tmp_path / "records.jsonl"
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return str(path), records


def test_offsets_round_trip_every_row(records_file):
    path, records = records_file
    store = CorpusStore(path).load()
    assert len(store) == len(records)
    for row, expected in enumerate(records):
        assert store.get(row) == expected
    store.close()


def test_rows_are_addressable_out_of_order(records_file):
    """Seeking backwards must work; retrieval hands back scores in rank order."""
    path, records = records_file
    store = CorpusStore(path).load()
    for row in (7, 0, 4, 8, 1):
        assert store.sequence(row) == records[row]["sequence"]
    store.close()


def test_out_of_range_row_raises(records_file):
    path, records = records_file
    store = CorpusStore(path).load()
    with pytest.raises(IndexError):
        store.get(len(records))
    with pytest.raises(IndexError):
        store.get(-1)
    store.close()


def test_summary_never_leaks_nucleotides(records_file):
    """describe_rows is diagnostic; corpus DNA must not cross an agent hop."""
    path, _ = records_file
    store = CorpusStore(path).load()
    summary = store.summary(3)
    assert "sequence" not in summary
    assert summary["organism"] == "Beta"
    assert summary["protein"] == "protein 3"
    assert summary["length"] > 0
    store.close()


def test_missing_records_file_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_rag_index"):
        CorpusStore(str(tmp_path / "absent.jsonl")).load()


def test_protein_is_parsed_out_of_the_header_because_there_is_no_product_key():
    assert protein_of({"header": "lcl|x [locus_tag=A] [protein=DNA gyrase] [gbkey=CDS]"}) == (
        "DNA gyrase"
    )
    assert protein_of({"header": "lcl|x [locus_tag=A]"}) == ""
    assert protein_of({}) == ""


def test_fingerprint_is_stable_and_changes_with_the_corpus(tmp_path):
    manifest = tmp_path / "manifest.json"
    records = tmp_path / "records.jsonl"
    manifest.write_text('{"records": 3}')
    records.write_text("a\nb\nc\n")

    first = corpus_store.corpus_fingerprint(str(manifest), str(records))
    assert first == corpus_store.corpus_fingerprint(str(manifest), str(records))

    manifest.write_text('{"records": 4}')
    assert corpus_store.corpus_fingerprint(str(manifest), str(records)) != first


def test_fingerprint_does_not_raise_when_the_cache_is_absent(tmp_path):
    assert corpus_store.corpus_fingerprint(str(tmp_path / "no"), str(tmp_path / "no"))


# --------------------------------------------------------------------------- organisms
def test_organism_blocks_are_contiguous_and_cover_every_row(fake_store):
    index = OrganismIndex.build(fake_store)
    blocks = sorted(index.blocks.values())
    assert blocks[0][0] == 0
    assert blocks[-1][1] == len(fake_store)
    for (_, end), (start, _) in zip(blocks, blocks[1:]):
        assert end == start  # no holes, no overlaps


def test_lookup_is_case_insensitive_and_accepts_a_unique_prefix(fake_store):
    index = OrganismIndex.build(fake_store)
    assert index.resolve("alpha") == "Alpha"
    assert index.resolve("Alph") == "Alpha"
    assert index.resolve("Nonexistent") is None
    assert index.resolve("") is None


def test_membership_matches_the_block(fake_store):
    index = OrganismIndex.build(fake_store)
    lo, hi = index.block("Beta")
    assert index.size("Beta") == hi - lo
    assert index.contains("Beta", lo) and index.contains("Beta", hi - 1)
    assert not index.contains("Beta", hi)
    assert not index.contains("Beta", lo - 1)


@pytest.mark.heavy
def test_real_corpus_has_twenty_contiguous_organism_blocks():
    """The property the restricted path depends on, checked against the real cache."""
    store = corpus_store.get_store()
    assert len(store) == 43575
    index = OrganismIndex.build(store)
    assert len(index.blocks) == 20
    described = index.describe()
    assert described[0]["lo"] == 0
    assert described[-1]["hi"] == 43575
    assert all(block["rows"] > 0 for block in described)
    assert index.size("Fusobacterium animalis") > 0
