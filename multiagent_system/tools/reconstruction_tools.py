"""MCP tools for the Reconstruction agent.

This agent owns the GENA-LM tokenizer and model, so it is the only component that knows
the token budget -- and therefore the only one that can decide how much retrieved context
fits. In the original code that knowledge lived in the same function as the candidate
ranking, which is exactly why the constraint between them was never written down: there
was no boundary to write it across.

The negotiation edge runs from here. ``request_candidates`` and ``extend_candidates``
call the Retrieval agent over A2A, sized by a budget this agent measured. Neither agent
can answer "how many candidates fit" alone: retrieval knows the ranking but not the
tokenizer state, reconstruction knows the budget but not the corpus.

Nucleotides enter the system at exactly one place, ``admit_candidates``, which resolves
row IDs against the local corpus store. Corpus DNA never crosses an agent hop.
"""
from __future__ import annotations

import time
from typing import Any

from claude_agent_sdk import tool

from .. import config
from ..a2a import types as a2a_types
from ..a2a.client import A2AClient
from ..runtime import errors
from ..runtime.blocking import run_blocking
from ..runtime.state import Phase, ReconTask, guarded
from . import mlm
from .corpus_store import get_store

SERVER_KEY = config.SERVER_KEYS[config.RECONSTRUCTION]
TOOL_NAMES = (
    "measure_budget",
    "request_candidates",
    "extend_candidates",
    "admit_candidates",
    "generate",
    "report_state",
)
ALLOWED_TOOLS = [f"mcp__{SERVER_KEY}__{name}" for name in TOOL_NAMES]


class _Models:
    """Loaded once at startup. A per-request load would cost 4 s and thrash memory."""

    tokenizer: Any = None
    model: Any = None
    loaded: bool = False


MODELS = _Models()


def warm() -> dict[str, Any]:
    """Load GENA-LM and run one tiny forward pass so the first real call is not cold."""
    import torch
    from transformers import AutoTokenizer, BigBirdForMaskedLM

    started = time.monotonic()
    torch.manual_seed(config.SEED)
    MODELS.tokenizer = AutoTokenizer.from_pretrained(config.GENA_LM_MODEL)
    MODELS.model = BigBirdForMaskedLM.from_pretrained(config.GENA_LM_MODEL)
    MODELS.model.eval()

    primed = MODELS.tokenizer("ACGT ACGT", return_tensors="pt")
    with torch.no_grad():
        MODELS.model(**primed)

    get_store()  # build the record offset table too
    MODELS.loaded = True
    return {"model": config.GENA_LM_MODEL, "warm_s": round(time.monotonic() - started, 2)}


# --------------------------------------------------------------------------- helpers
def _trim(sequence: str, side: str) -> str:
    """Keep MODEL_FLANK bases nearest the gap. 0 disables trimming.

    Measured on gap1 (contigs 5,951 and 13,996 bp): untrimmed, the base input is 3,305 of
    4,096 tokens, leaving about 583 free once the mask block is added. At a 2,000 bp
    flank it is 682 tokens, leaving roughly 3,200. On a long contig pair -- the old
    harness used two contigs of 190 kb and 167 kb -- untrimmed leaves nothing at all,
    which is the truncation failure this system reports rather than absorbs.
    """
    flank = config.MODEL_FLANK
    if flank <= 0 or len(sequence) <= flank:
        return sequence
    return sequence[-flank:] if side == "left" else sequence[:flank]


def _retrieval_client() -> A2AClient:
    return A2AClient(config.PEER_URLS[config.RETRIEVAL], timeout=config.PEER_TIMEOUT_S)


async def _ask_retrieval(payload: dict[str, Any]) -> dict[str, Any]:
    """One A2A round trip to the Retrieval agent. This is the negotiation edge."""
    message = a2a_types.build_message(
        role="user",
        parts=[a2a_types.data_part(payload)],
        context_id=payload.get("context_id") or "",
    )
    task = await _retrieval_client().send(message)
    state = a2a_types.task_state(task)
    data = a2a_types.artifact_data(task)
    if state is not a2a_types.TaskState.COMPLETED:
        return {
            "ok": False,
            "peer_task_state": state.value,
            "detail": data or {"error": "retrieval agent returned no artifact"},
        }
    return {"ok": True, "peer_task_state": state.value, **data}


# --------------------------------------------------------------------------- tools
@tool(
    "measure_budget",
    "Tokenize the model input with the context slot empty and report how many of the "
    "4096 token positions are free for retrieved context. Must be called before "
    "requesting or admitting any candidates.",
    {"task_id": str},
)
@guarded("measure_budget", requires=None)
async def measure_budget(args: dict[str, Any], state: ReconTask) -> dict[str, Any]:
    if state.phase is Phase.GENERATED:
        return errors.tool_error(
            errors.ALREADY_GENERATED,
            remedy="This task has already produced a sequence. Finish your turn.",
            task_id=state.task_id,
        )

    tokenizer = MODELS.tokenizer
    state.model_left = _trim(state.left, "left")
    state.model_right = _trim(state.right, "right")

    base = mlm.build_model_input("", state.model_left, state.model_right)
    base_tokens = await run_blocking(mlm.count_tokens, tokenizer, base)
    bases_per_token = await run_blocking(
        mlm.measure_bases_per_token, tokenizer, state.model_left, state.model_right
    )
    seed = mlm.seed_mask_count(state.gap_length, bases_per_token)

    # The mask block occupies real positions too, so reserve them before offering the
    # rest as context budget.
    free = config.MAX_LENGTH - base_tokens - seed - config.SAFETY_TOKENS
    free = max(0, free)

    state.base_tokens = base_tokens
    state.bases_per_token = bases_per_token
    state.seed_mask_count = seed
    state.free_tokens_total = free
    state.free_tokens_remaining = free
    state.advance(Phase.MEASURED)

    state.budget = {
        "task_id": state.task_id,
        "model_flank": config.MODEL_FLANK,
        "left_len": len(state.left),
        "right_len": len(state.right),
        "left_used": len(state.model_left),
        "right_used": len(state.model_right),
        "base_tokens": base_tokens,
        "max_length": config.MAX_LENGTH,
        "reserved_for_masks": seed,
        "safety_tokens": config.SAFETY_TOKENS,
        "free_tokens": free,
        "bases_per_token": round(bases_per_token, 3),
        "seed_mask_count": seed,
        "gap_length": state.gap_length,
    }
    payload = dict(state.budget)
    payload["next_step"] = (
        f"Call request_candidates with task_id and a batch_size sized to {free} free "
        f"tokens."
        if free > 0
        else "No tokens are free for context; the flanks already fill the window."
    )
    return errors.tool_ok(payload)


@tool(
    "request_candidates",
    "Ask the retrieval agent for a batch of candidate corpus row IDs, sized to the "
    "free token budget. Opens the candidate pool. Requires measure_budget first.",
    {"task_id": str, "batch_size": int},
)
@guarded("request_candidates", requires=Phase.MEASURED)
async def request_candidates(args: dict[str, Any], state: ReconTask) -> dict[str, Any]:
    batch_size = _as_int(args.get("batch_size"), 0)
    if not (config.MIN_BATCH_SIZE <= batch_size <= config.MAX_BATCH_SIZE):
        return errors.tool_error(
            errors.BAD_BATCH_SIZE,
            remedy=f"batch_size must be between {config.MIN_BATCH_SIZE} and "
            f"{config.MAX_BATCH_SIZE}. Pick a size appropriate to "
            f"{state.free_tokens_remaining} free tokens.",
            task_id=state.task_id,
            given=args.get("batch_size"),
        )

    response = await _ask_retrieval(
        {
            "op": "retrieve",
            "request_id": state.retrieval_request_id,
            "context_id": state.context_id,
            "query_text": state.query_text,
            "batch_size": batch_size,
            "organism": state.organism,
        }
    )
    if not response.get("ok"):
        return errors.tool_error(
            errors.PEER_UNREACHABLE,
            remedy="The retrieval agent did not complete the request. Report the "
            "failure and finish your turn; do not invent candidates.",
            task_id=state.task_id,
            **{k: v for k, v in response.items() if k != "ok"},
        )

    rows = [int(r) for r in response.get("row_ids", [])]
    state.offered_row_ids.extend(rows)
    _record_peer_round(state, response, rows)
    return errors.tool_ok(_candidate_payload(state, response, rows))


@tool(
    "extend_candidates",
    "Ask the retrieval agent for the next batch from the pool already opened for this "
    "task. Use when free tokens remain after admitting the previous batch.",
    {"task_id": str, "batch_size": int},
)
@guarded("extend_candidates", requires=Phase.MEASURED)
async def extend_candidates(args: dict[str, Any], state: ReconTask) -> dict[str, Any]:
    batch_size = _as_int(args.get("batch_size"), 0)
    if not (config.MIN_BATCH_SIZE <= batch_size <= config.MAX_BATCH_SIZE):
        return errors.tool_error(
            errors.BAD_BATCH_SIZE,
            remedy=f"batch_size must be between {config.MIN_BATCH_SIZE} and "
            f"{config.MAX_BATCH_SIZE}.",
            task_id=state.task_id,
            given=args.get("batch_size"),
        )

    response = await _ask_retrieval(
        {
            "op": "extend",
            "request_id": state.retrieval_request_id,
            "context_id": state.context_id,
            "batch_size": batch_size,
        }
    )
    if not response.get("ok"):
        return errors.tool_error(
            errors.PEER_UNREACHABLE,
            remedy="The retrieval agent did not complete the request. Proceed with what "
            "has already been admitted, or finish your turn.",
            task_id=state.task_id,
            **{k: v for k, v in response.items() if k != "ok"},
        )

    rows = [int(r) for r in response.get("row_ids", [])]
    state.offered_row_ids.extend(rows)
    _record_peer_round(state, response, rows)
    return errors.tool_ok(_candidate_payload(state, response, rows))


def _record_peer_round(
    state: ReconTask, response: dict[str, Any], rows: list[int]
) -> None:
    """Keep what the peer reported, so the run's trace shows both sides of the edge."""
    state.retrieval_tool_calls.extend(response.get("llm_tool_calls", []) or [])
    state.retrieval_rounds.append(
        {
            "served": len(rows),
            "served_total": response.get("served_total", 0),
            "pool_size": response.get("pool_size", 0),
            "pool_exhausted": bool(response.get("pool_exhausted")),
            "path": response.get("path", ""),
        }
    )


def _candidate_payload(
    state: ReconTask, response: dict[str, Any], rows: list[int]
) -> dict[str, Any]:
    exhausted = bool(response.get("pool_exhausted"))
    return {
        "task_id": state.task_id,
        "row_ids": rows,
        "scores": response.get("scores", []),
        "path": response.get("path", ""),
        "pool_size": response.get("pool_size", 0),
        "served_total": response.get("served_total", 0),
        "pool_exhausted": exhausted,
        "restricted_set_size": response.get("restricted_set_size", 0),
        "free_tokens_remaining": state.free_tokens_remaining,
        "next_step": (
            "The pool is exhausted; admit what you have, then call generate."
            if exhausted and not rows
            else f"Call admit_candidates with these row_ids and "
            f"free_tokens={state.free_tokens_remaining}."
        ),
    }


@tool(
    "admit_candidates",
    "Resolve corpus row IDs against the local record store, tokenize each candidate and "
    "admit those that fit within the free token budget. free_tokens must equal the value "
    "this agent currently reports.",
    {"task_id": str, "row_ids": list, "free_tokens": int},
)
@guarded("admit_candidates", requires=Phase.MEASURED)
async def admit_candidates(args: dict[str, Any], state: ReconTask) -> dict[str, Any]:
    raw_rows = args.get("row_ids")
    if not isinstance(raw_rows, list) or not raw_rows:
        return errors.tool_error(
            errors.BAD_ROW_IDS,
            remedy="row_ids must be a non-empty list of integers returned by "
            "request_candidates or extend_candidates.",
            task_id=state.task_id,
            given=raw_rows,
        )

    store = get_store()
    rows: list[int] = []
    for item in raw_rows:
        try:
            row = int(item)
        except (TypeError, ValueError):
            return errors.tool_error(
                errors.BAD_ROW_IDS,
                remedy="Every element of row_ids must be an integer row number.",
                task_id=state.task_id,
                offending=item,
            )
        if not (0 <= row < len(store)):
            return errors.tool_error(
                errors.BAD_ROW_IDS,
                remedy=f"Row IDs must fall inside [0, {len(store)}). Use only IDs the "
                f"retrieval agent returned.",
                task_id=state.task_id,
                offending=row,
            )
        rows.append(row)

    declared = _as_int(args.get("free_tokens"), -1)
    if declared != state.free_tokens_remaining:
        # The mechanism that makes "batch size derived from a real budget" enforceable
        # rather than aspirational: the correct value is only obtainable by asking.
        return errors.tool_error(
            errors.BUDGET_MISMATCH,
            remedy=f"free_tokens must be {state.free_tokens_remaining}, the value this "
            f"agent currently reports. Call again with that number.",
            task_id=state.task_id,
            given=args.get("free_tokens"),
            free_tokens=state.free_tokens_remaining,
        )

    if state.free_tokens_remaining <= 0:
        return errors.tool_error(
            errors.WINDOW_FULL,
            remedy="The context window is full. Call generate now.",
            task_id=state.task_id,
            context_tokens_admitted=state.context_tokens_admitted,
        )

    tokenizer = MODELS.tokenizer
    admitted: list[int] = []
    rejected: list[int] = []
    duplicates = 0
    reasons: dict[str, str] = {}
    consumed = 0

    for row in rows:
        if row in state.admitted_row_ids:
            duplicates += 1
            continue
        sequence = await run_blocking(store.sequence, row)
        cost = await run_blocking(mlm.count_tokens, tokenizer, sequence)
        if cost <= state.free_tokens_remaining:
            state.admitted_row_ids.append(row)
            state.context_blocks.append(sequence)
            state.free_tokens_remaining -= cost
            state.context_tokens_admitted += cost
            consumed += cost
            admitted.append(row)
        else:
            # Keep going: a later, shorter record may still fit.
            rejected.append(row)
            state.rejected_row_ids.append(row)
            reasons[str(row)] = (
                f"{cost} tokens exceeds the {state.free_tokens_remaining} remaining"
            )

    if state.admitted_row_ids:
        state.advance(Phase.ADMITTING)

    window_full = state.free_tokens_remaining <= config.SAFETY_TOKENS
    return errors.tool_ok(
        {
            "task_id": state.task_id,
            "admitted": admitted,
            "rejected": rejected,
            "admitted_count": len(admitted),
            "rejected_count": len(rejected),
            "duplicates": duplicates,
            "tokens_consumed": consumed,
            "context_tokens_admitted": state.context_tokens_admitted,
            "candidates_consumed": len(state.admitted_row_ids),
            "free_tokens_remaining": state.free_tokens_remaining,
            "window_full": window_full,
            "reject_reason_by_row": reasons,
            "next_step": (
                "Call generate now."
                if window_full or not state.free_tokens_remaining
                else f"{state.free_tokens_remaining} tokens are still free; call "
                f"extend_candidates for more, or generate if the pool is exhausted."
            ),
        }
    )


@tool(
    "generate",
    "Run the masked language model over the assembled input and produce the missing "
    "sequence. Requires the budget to have been measured and at least one candidate "
    "admitted. gap_length must be exactly the value this task was created with.",
    {"task_id": str, "gap_length": int, "threshold": float},
)
@guarded("generate", requires=Phase.ADMITTING, max_calls=3)
async def generate(args: dict[str, Any], state: ReconTask) -> dict[str, Any]:
    if state.phase is Phase.GENERATED and state.result:
        return errors.tool_ok({**state.result, "cached": True})

    if state.context_tokens_admitted <= 0:
        return errors.tool_error(
            errors.EMPTY_CONTEXT,
            remedy="No retrieved context was admitted, so this run could not be called "
            "RAG-enabled. Admit at least one candidate first.",
            task_id=state.task_id,
        )

    requested = _as_int(args.get("gap_length"), -1)
    if requested != state.gap_length:
        # The bug from planner_tools.py, where an adjusted gap length was silently
        # substituted for the requested one and clamped to 1.
        return errors.tool_error(
            errors.GAP_LENGTH_MUTATED,
            remedy=f"gap_length must be exactly {state.gap_length}, the value this task "
            f"was created with. Do not recompute or adjust it.",
            task_id=state.task_id,
            given=args.get("gap_length"),
            gap_length=state.gap_length,
        )

    threshold = _as_float(args.get("threshold"), config.DEFAULT_THRESHOLD)
    if not (0.0 < threshold <= 1.0):
        return errors.tool_error(
            errors.BAD_THRESHOLD,
            remedy=f"threshold must be in (0, 1]; {config.DEFAULT_THRESHOLD} is the "
            f"default.",
            task_id=state.task_id,
            given=args.get("threshold"),
        )

    started = time.monotonic()
    fill = await run_blocking(
        mlm.fill_to_length,
        state.context,
        state.model_left,
        state.model_right,
        state.gap_length,
        MODELS.tokenizer,
        MODELS.model,
        threshold=threshold,
        seed_masks=state.seed_mask_count,
        max_attempts=config.MAX_ATTEMPTS,
        max_length=config.MAX_LENGTH,
    )

    state.advance(Phase.GENERATED)
    state.result = {
        "task_id": state.task_id,
        "sequence": fill.sequence,
        **fill.as_dict(),
        "context_tokens_admitted": state.context_tokens_admitted,
        "candidates_offered": len(state.offered_row_ids),
        "candidates_consumed": len(state.admitted_row_ids),
        "admitted_row_ids": list(state.admitted_row_ids),
        "retrieval_tool_calls": list(state.retrieval_tool_calls),
        "retrieval_rounds": list(state.retrieval_rounds),
        "tool_calls_by_name": dict(state.call_counts),
        "rejected_row_ids": list(state.rejected_row_ids),
        "model_flank": config.MODEL_FLANK,
        "budget": state.budget,
        "elapsed_s": round(time.monotonic() - started, 2),
    }
    # The sequence is large; keep it out of the model's context but in the artifact.
    summary = {k: v for k, v in state.result.items() if k not in ("sequence", "attempts")}
    summary["sequence_preview"] = fill.sequence[:60]
    summary["next_step"] = "Generation is complete. Summarise briefly and finish."
    return errors.tool_ok(summary)


@tool(
    "report_state",
    "Return this task's full state and call trace. Read-only diagnostic.",
    {"task_id": str},
)
@guarded("report_state", requires=None, max_calls=4)
async def report_state(args: dict[str, Any], state: ReconTask) -> dict[str, Any]:
    return errors.tool_ok(
        {
            "task_id": state.task_id,
            "phase": state.phase.value,
            "gap_length": state.gap_length,
            "free_tokens_remaining": state.free_tokens_remaining,
            "context_tokens_admitted": state.context_tokens_admitted,
            "candidates_offered": len(state.offered_row_ids),
            "candidates_consumed": len(state.admitted_row_ids),
            "calls": dict(state.call_counts),
            "seconds_left": round(state.seconds_left, 1),
            "trace": state.trace,
        }
    )


TOOLS = [
    measure_budget,
    request_candidates,
    extend_candidates,
    admit_candidates,
    generate,
    report_state,
]


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
