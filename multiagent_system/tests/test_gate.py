"""The acceptance gate. Pure functions, no models, no network."""
from __future__ import annotations

import pytest

from multiagent_system import gate

TARGET = 870


def test_tolerance_is_five_percent_with_a_floor():
    assert gate.tolerance_bases(870) == 44
    assert gate.tolerance_bases(10) == 3  # floor wins on tiny gaps
    assert gate.tolerance_bases(4867) == 244


def test_accepts_a_good_result():
    accepted, reasons = gate.evaluate("A" * 871, TARGET, context_tokens_admitted=1180)
    assert accepted
    assert reasons == []
    assert gate.verdict(accepted, reasons) == "ACCEPTED"


def test_accepts_at_the_tolerance_boundary():
    assert gate.evaluate("A" * (TARGET + 44), TARGET, 100)[0]
    assert not gate.evaluate("A" * (TARGET + 45), TARGET, 100)[0]


def test_rejects_wrong_length():
    accepted, reasons = gate.evaluate("A" * 500, TARGET, 1180)
    assert not accepted
    assert reasons == [gate.REASON_LENGTH]
    assert gate.verdict(accepted, reasons) == "REJECTED:LENGTH"


@pytest.mark.parametrize("bad", ["acgt" * 300, "ACGX" * 300, "ACGT " * 240])
def test_rejects_characters_outside_acgtn(bad):
    padded = bad[:TARGET]
    accepted, reasons = gate.evaluate(padded, TARGET, 1180)
    assert not accepted
    assert gate.REASON_ALPHABET in reasons


def test_rejects_a_perfect_sequence_when_no_context_was_admitted():
    """The check that did not exist before.

    A flawless reconstruction is still rejected when nothing retrieved reached the
    model, because calling that run RAG-enabled would be a lie.
    """
    accepted, reasons = gate.evaluate("ACGT" * 217 + "AC", TARGET, context_tokens_admitted=0)
    assert not accepted
    assert reasons == [gate.REASON_NO_CONTEXT]
    assert "truncation" in gate.explain(reasons)


def test_empty_output_reports_empty_not_length():
    accepted, reasons = gate.evaluate("", TARGET, 1180)
    assert not accepted
    assert reasons == [gate.REASON_EMPTY]


def test_all_applicable_reasons_are_reported_together():
    accepted, reasons = gate.evaluate("xyz", TARGET, context_tokens_admitted=0)
    assert not accepted
    assert set(reasons) == {gate.REASON_LENGTH, gate.REASON_ALPHABET, gate.REASON_NO_CONTEXT}
    assert gate.verdict(accepted, reasons).startswith("REJECTED:")


def test_verdicts_distinguish_plumbing_from_generator_quality():
    """A red test must say which of the two failed; they need different responses."""
    _, plumbing = gate.evaluate("A" * TARGET, TARGET, 0)
    _, quality = gate.evaluate("A" * 10, TARGET, 500)
    assert gate.verdict(False, plumbing) == "REJECTED:NO_CONTEXT"
    assert gate.verdict(False, quality) == "REJECTED:LENGTH"
    assert gate.explain(plumbing) != gate.explain(quality)
