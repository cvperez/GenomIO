"""Picking the contigs that actually border a gap.

This is a regression test for a specific defect in the old harness, so the defect itself
is asserted below rather than only described.
"""
from __future__ import annotations

import pytest

from multiagent_system import dataset

ACCESSION = "AP012051.1"
GAP1 = "AP012051.1_gap1"


@pytest.fixture(scope="module")
def contigs():
    return dataset.parse_contigs(dataset.contigs_path(ACCESSION))


@pytest.fixture(scope="module")
def gaps():
    return dataset.parse_gaps(dataset.gaps_path(ACCESSION))


def test_the_fasta_is_sorted_by_length_which_is_what_trapped_the_old_harness(contigs):
    """gap-filler-agents/test.py took contigs[0] and contigs[1].

    They are the two longest contigs, 190 kb and 167 kb, which are neither adjacent to
    each other nor anywhere near gap1. The old run therefore filled an 870 bp gap
    between two sequences that have no gap between them.
    """
    assert contigs[0]["id"] == "AP012051.1_contig31"
    assert contigs[1]["id"] == "AP012051.1_contig22"
    assert contigs[0]["length"] == 190651
    assert contigs[1]["length"] == 167069
    # Not adjacent: contig31 spans 1298041-1488692, contig22 spans 898138-1065207.
    assert contigs[0]["start"] != contigs[1]["end"]
    assert contigs[1]["start"] != contigs[0]["end"]


def test_headers_carry_usable_genome_coordinates(contigs):
    for contig in contigs:
        assert contig["start"] >= 0 and contig["end"] > contig["start"]
        assert contig["length"] == contig["declared_length"]
        assert contig["end"] - contig["start"] == contig["length"]


def test_gap1_flanks_resolve_to_contig1_and_contig2(contigs, gaps):
    gap = dataset.find_gap(gaps, GAP1)
    assert (gap["start"], gap["end"], gap["length"]) == (5951, 6821, 870)

    left, right = dataset.flanks_for_gap(contigs, gap)
    assert left["id"] == "AP012051.1_contig1"
    assert right["id"] == "AP012051.1_contig2"
    assert left["length"] == 5951
    assert right["length"] == 13996
    assert left["end"] == gap["start"]
    assert right["start"] == gap["end"]


def test_the_matcher_works_for_every_interior_gap_not_just_gap1(contigs, gaps):
    """32 of the 33 gaps, each resolving to exactly one contig on each side."""
    usable = dataset.reconstructable_gaps(contigs, gaps)
    assert len(usable) == 32
    for gap in usable:
        left, right = dataset.flanks_for_gap(contigs, gap)
        assert left["end"] == gap["start"]
        assert right["start"] == gap["end"]
        assert gap["length"] == gap["end"] - gap["start"]


def test_the_terminal_gap_is_recognised_rather_than_guessed_at(contigs, gaps):
    """gap33 runs past the end of the assembly, so it has no right-hand contig.

    The system declines it instead of inventing a right flank, which is the honest
    behaviour for a gap the model cannot condition on from both sides.
    """
    gap33 = dataset.find_gap(gaps, "AP012051.1_gap33")
    assert gap33["start"] == 1557603 and gap33["end"] == 1558103
    assert max(c["end"] for c in contigs) == 1557603
    assert not dataset.has_both_flanks(contigs, gap33)
    with pytest.raises(dataset.DatasetError, match="no contig starts at position 1558103"):
        dataset.flanks_for_gap(contigs, gap33)


def test_load_gap_case_bundles_what_a_run_needs():
    case = dataset.load_gap_case(ACCESSION, GAP1)
    assert case["gap_length"] == 870
    assert case["left_id"] == "AP012051.1_contig1"
    assert case["right_id"] == "AP012051.1_contig2"
    assert len(case["left"]) == 5951
    assert len(case["right"]) == 13996
    assert len(case["true_sequence"]) == 870
    assert set(case["left"]) <= set("ACGTN")


def test_an_unknown_gap_id_fails_loudly():
    gaps = dataset.parse_gaps(dataset.gaps_path(ACCESSION))
    with pytest.raises(dataset.DatasetError):
        dataset.find_gap(gaps, "AP012051.1_gap999")


def test_a_gap_with_no_matching_flank_fails_loudly(contigs):
    bogus = {"gap_id": "fake", "start": 123456789, "end": 123456999, "length": 210}
    with pytest.raises(dataset.DatasetError, match="no contig ends at"):
        dataset.flanks_for_gap(contigs, bogus)
