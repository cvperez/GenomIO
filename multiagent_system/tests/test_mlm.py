"""The generation core: prompt assembly, the measured seed, and a real end-to-end fill.

The light tests pin the four changes made to ``predict_until_length``. The heavy test at
the bottom runs the real DNABERT-S retrieval and the real GENA-LM fill against gap1, and
is where the project's central number first becomes observable:
``context_tokens_admitted``.
"""
from __future__ import annotations

import os

import pytest

from multiagent_system import config, dataset
from multiagent_system.tools import mlm

from .conftest import FakeTokenizer

HEAVY = os.environ.get("GENOMIO_HEAVY", "") == "1"


# --------------------------------------------------------------------------- prompt
def test_context_goes_in_front_of_the_sequence():
    """The single most consequential change.

    The original appended context after the flanks, and truncation removes from the end,
    so the retrieved evidence was always first to be deleted.
    """
    built = mlm.build_model_input("CTXCTX", "AAAA", "TTTT")
    assert built.startswith("CTXCTX")
    assert built.index("CTXCTX") < built.index("AAAA") < built.index(mlm.MASK_SLOT)
    assert built == "CTXCTX AAAA [MASKS] TTTT"


def test_no_english_label_is_handed_to_a_nucleotide_tokenizer():
    """The original wrote '\\n\\n# Context: ...' into a DNA-only vocabulary."""
    built = mlm.build_model_input("ACGT", "AAAA", "TTTT")
    assert "#" not in built and "Context" not in built


def test_empty_context_leaves_the_input_unchanged():
    assert mlm.build_model_input("", "AAAA", "TTTT") == "AAAA [MASKS] TTTT"
    assert mlm.build_model_input("   ", "AAAA", "TTTT") == "AAAA [MASKS] TTTT"


# --------------------------------------------------------------------------- seeding
def test_the_mask_seed_uses_a_measured_ratio_not_the_hardcoded_four():
    tokenizer = FakeTokenizer(bases_per_token=6)
    ratio = mlm.measure_bases_per_token(tokenizer, "ACGT" * 300, "TGCA" * 300)
    assert ratio == pytest.approx(6.0, abs=0.2)

    measured = mlm.seed_mask_count(870, ratio)
    original = int(870 / 4)
    assert measured == pytest.approx(145, abs=3)
    # The old guess starts roughly 45% high and spends attempts walking back down.
    assert original > measured * 1.4


def test_the_ratio_falls_back_to_the_original_guess_when_unmeasurable():
    assert mlm.measure_bases_per_token(FakeTokenizer(), "", "") == 4.0


def test_seed_is_never_zero():
    assert mlm.seed_mask_count(1, 100.0) == 1
    assert mlm.seed_mask_count(0, 6.0) == 1


# --------------------------------------------------------------------------- counters
def test_adjustment_matches_the_original_schedule():
    """Unchanged on purpose: output differences must come from the four changes only."""
    assert mlm._adjustment(300, 870) == max(5, 300 // 4)   # relative > 0.2
    assert mlm._adjustment(130, 870) == max(3, 130 // 6)   # relative > 0.1
    assert mlm._adjustment(40, 870) == max(1, 40 // 8)     # else


def test_fill_result_serialises_the_counters_the_old_code_printed_and_dropped():
    result = mlm.FillResult(target_length=870, attempts_used=4, final_mask_count=151)
    data = result.as_dict()
    for key in (
        "attempts_used",
        "final_mask_count",
        "seed_mask_count",
        "truncated",
        "alphabet_violations",
        "input_tokens",
        "converged",
    ):
        assert key in data


# --------------------------------------------------------------------------- heavy
@pytest.mark.heavy
def test_real_pipeline_admits_context_and_fills_gap1():
    """The whole scientific path, no agents and no LLM: retrieve, admit, generate.

    This is Phase 2's milestone. If ``context_tokens_admitted`` is zero here, nothing
    built on top of it can be honestly described as retrieval-augmented.
    """
    import torch
    from transformers import AutoTokenizer, BigBirdForMaskedLM

    from rag.index import search

    case = dataset.load_gap_case("AP012051.1", "AP012051.1_gap1")
    assert case["gap_length"] == 870

    # 1. retrieve, on the flanks that border the gap
    query = case["left"][-config.QUERY_FLANK:] + case["right"][: config.QUERY_FLANK]
    assert len(query) == 2 * config.QUERY_FLANK
    hits = search(query, k=8)
    assert hits, "DNABERT-S retrieval returned nothing for gap1"
    assert all(-1.01 <= score <= 1.01 for score, _ in hits)

    # 2. measure the window
    torch.manual_seed(config.SEED)
    tokenizer = AutoTokenizer.from_pretrained(config.GENA_LM_MODEL)
    model = BigBirdForMaskedLM.from_pretrained(config.GENA_LM_MODEL)
    model.eval()

    left = case["left"][-config.MODEL_FLANK:]
    right = case["right"][: config.MODEL_FLANK]
    base_tokens = mlm.count_tokens(tokenizer, mlm.build_model_input("", left, right))
    ratio = mlm.measure_bases_per_token(tokenizer, left, right)
    seed = mlm.seed_mask_count(870, ratio)
    free = config.MAX_LENGTH - base_tokens - seed - config.SAFETY_TOKENS
    assert free > 0, f"no room for context: base={base_tokens}, seed={seed}"

    # 3. admit what fits
    admitted, tokens = [], 0
    remaining = free
    for _score, record in hits:
        cost = mlm.count_tokens(tokenizer, record["sequence"])
        if cost <= remaining:
            admitted.append(record["sequence"])
            remaining -= cost
            tokens += cost
    assert tokens > 0, "nothing was admitted -- the context would never reach the model"

    # 4. generate
    result = mlm.fill_to_length(
        " ".join(admitted), left, right, 870, tokenizer, model,
        threshold=config.DEFAULT_THRESHOLD, seed_masks=seed,
        max_attempts=config.MAX_ATTEMPTS, max_length=config.MAX_LENGTH,
    )

    assert result.truncated is False, (
        f"input overran the window: {result.untruncated_tokens} > {config.MAX_LENGTH}"
    )
    assert result.produced_length > 0
    assert result.alphabet_violations == 0
    assert abs(result.produced_length - 870) <= 44
    print(
        f"\n  base={base_tokens} free={free} admitted={tokens} tokens "
        f"({len(admitted)} records) input={result.input_tokens} "
        f"produced={result.produced_length}/870 attempts={result.attempts_used}"
    )


@pytest.mark.heavy
def test_untrimmed_long_contigs_leave_no_room_which_is_the_bug_being_detected():
    """The old harness's contig pair, measured rather than argued about.

    contig31 (190 kb) and contig22 (167 kb) overrun the 4096-token window many times
    over, so with no trimming there is no budget for context at all -- and the old
    pipeline recorded such runs as RAG-enabled anyway.
    """
    from transformers import AutoTokenizer

    contigs = dataset.parse_contigs(dataset.contigs_path("AP012051.1"))
    tokenizer = AutoTokenizer.from_pretrained(config.GENA_LM_MODEL)

    base = mlm.build_model_input("", contigs[0]["sequence"], contigs[1]["sequence"])
    tokens = mlm.count_tokens(tokenizer, base)
    assert tokens > config.MAX_LENGTH * 10
    assert config.MAX_LENGTH - tokens < 0  # free budget is negative: nothing can fit
    print(f"\n  untrimmed contig31+contig22 = {tokens} tokens vs a {config.MAX_LENGTH} window")
