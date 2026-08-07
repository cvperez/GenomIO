"""The acceptance gate.

Nothing leaves the Coordinator without passing this. It is a pure function with no
imports beyond the standard library so it can be reasoned about and tested on its own.

The three checks come from the design document:

  LENGTH      produced length within tolerance of the requested gap_length
  ALPHABET    every character in {A, C, G, T, N}
  NO_CONTEXT  context_tokens_admitted > 0

The third is the one that did not exist before. It is what makes "RAG-enabled" a
measured precondition rather than a label: under the old pipeline the retrieved context
was appended after the sequence and truncated away, and the row was still recorded as
RAG-enabled.

The verdicts are deliberately distinguishable. REJECTED:NO_CONTEXT means the plumbing in
*this* system is broken. REJECTED:LENGTH and REJECTED:ALPHABET mean the underlying
masked language model produced something poor, which this project does not claim to fix.
"""
from __future__ import annotations

import math

VALID_BASES = frozenset("ACGTN")

REASON_EMPTY = "EMPTY"
REASON_LENGTH = "LENGTH"
REASON_ALPHABET = "ALPHABET"
REASON_NO_CONTEXT = "NO_CONTEXT"

VERDICT_ACCEPTED = "ACCEPTED"


def tolerance_bases(gap_length: int, tolerance: float = 0.05) -> int:
    """How many bases the produced sequence may differ from the target by."""
    return max(3, math.ceil(tolerance * max(0, gap_length)))


def evaluate(
    produced: str,
    gap_length: int,
    context_tokens_admitted: int,
    tolerance: float = 0.05,
) -> tuple[bool, list[str]]:
    """Return ``(accepted, reasons)``.

    ``reasons`` is a subset of {EMPTY, LENGTH, ALPHABET, NO_CONTEXT} and is empty when
    the result is accepted. All applicable reasons are reported, not just the first, so
    a failing run says everything that is wrong with it in one pass.
    """
    reasons: list[str] = []

    if not produced:
        reasons.append(REASON_EMPTY)
    else:
        if abs(len(produced) - gap_length) > tolerance_bases(gap_length, tolerance):
            reasons.append(REASON_LENGTH)
        if not set(produced) <= VALID_BASES:
            reasons.append(REASON_ALPHABET)

    if context_tokens_admitted <= 0:
        reasons.append(REASON_NO_CONTEXT)

    return (not reasons), reasons


def verdict(accepted: bool, reasons: list[str]) -> str:
    """A single string for the trace record, e.g. ``REJECTED:LENGTH+NO_CONTEXT``."""
    if accepted:
        return VERDICT_ACCEPTED
    return "REJECTED:" + "+".join(reasons) if reasons else "REJECTED"


def explain(reasons: list[str]) -> str:
    """Human-readable diagnosis, used in the test's failure message and the trace."""
    parts = []
    for reason in reasons:
        if reason == REASON_EMPTY:
            parts.append("the generator produced nothing at all")
        elif reason == REASON_LENGTH:
            parts.append(
                "produced length is outside tolerance of the target -- a generator "
                "quality problem, not a retrieval one"
            )
        elif reason == REASON_ALPHABET:
            parts.append(
                "output contains characters outside ACGTN -- decoded special tokens "
                "survived, a generator problem"
            )
        elif reason == REASON_NO_CONTEXT:
            parts.append(
                "no retrieved context reached the model -- this is the truncation "
                "failure the system exists to detect"
            )
    return "; ".join(parts) if parts else "accepted"
