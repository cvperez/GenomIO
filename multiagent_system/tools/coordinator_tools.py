"""MCP tools for the Coordinator.

The Coordinator executes policy, not science. It wraps none of the RAG code: slicing the
query flanks is string arithmetic, and the acceptance gate is a set of assertions. It
loads no models, which is why it starts in about two seconds while its peers spend
thirty-eight and eight.

Its one irreplaceable job is the gate. The old pipeline had no acceptance check of any
kind -- whatever the generator produced was returned, and because the retrieved context
was being truncated away unnoticed, the entire RAG-versus-no-RAG comparison rested on a
label nothing had verified. ``evaluate_result`` turns that label into a measured
precondition.
"""
from __future__ import annotations

import time
from typing import Any

from claude_agent_sdk import tool

from .. import config, gate
from ..a2a import types as a2a_types
from ..a2a.client import A2AClient
from ..runtime import errors
from ..runtime.state import CoordTask, Phase, guarded

SERVER_KEY = config.SERVER_KEYS[config.COORDINATOR]
TOOL_NAMES = ("prepare_query", "delegate_reconstruction", "evaluate_result", "emit_trace")
ALLOWED_TOOLS = [f"mcp__{SERVER_KEY}__{name}" for name in TOOL_NAMES]

MODE_RESTRICTED = "restricted"
MODE_UNRESTRICTED = "unrestricted"


def build_query(left: str, right: str, flank: int | None = None) -> str:
    """The retrieval query: ``flank`` bases either side of the gap.

    Not the whole contigs. The original passed ``left_seq + right_seq`` -- often over
    100,000 bases -- to an embedder that truncates at roughly 2,000, so what was actually
    searched for was the first fragment of the left contig and nothing at all from the
    right one.
    """
    width = config.QUERY_FLANK if flank is None else flank
    return f"{left[-width:]}{right[:width]}"


@tool(
    "prepare_query",
    "Slice the retrieval query flanks from this task's contigs, check that both peer "
    "agents are healthy and reading the same corpus, and select restricted or "
    "unrestricted retrieval mode. Must be called first.",
    {"task_id": str},
)
@guarded("prepare_query", requires=None)
async def prepare_query(args: dict[str, Any], state: CoordTask) -> dict[str, Any]:
    state.query_text = build_query(state.left, state.right)
    state.mode = MODE_RESTRICTED if state.organism else MODE_UNRESTRICTED

    peers: dict[str, Any] = {}
    fingerprints: dict[str, str] = {}
    for role in (config.RETRIEVAL, config.RECONSTRUCTION):
        client = A2AClient(config.PEER_URLS[role], timeout=config.HEALTH_TIMEOUT_S)
        try:
            health = await client.health(timeout=config.HEALTH_TIMEOUT_S)
            card = await client.agent_card(timeout=config.HEALTH_TIMEOUT_S)
            peers[role] = bool(health.get("models_loaded"))
            fingerprints[role] = str(
                (card.get("metadata") or {}).get("x-corpus-fingerprint", "")
            )
        except Exception as exc:
            peers[role] = False
            fingerprints[role] = ""
            state.record("prepare_query", False, {"peer": role, "error": str(exc)})

    missing = [role for role, ok in peers.items() if not ok]
    if missing:
        return errors.tool_error(
            errors.PEER_UNREACHABLE,
            remedy=f"The {', '.join(missing)} agent is not ready. Report this and finish "
            f"your turn; do not attempt the reconstruction yourself.",
            task_id=state.task_id,
            peers=peers,
        )

    # A positional row ID means nothing unless both agents read the same corpus snapshot.
    # A mismatch would not error anywhere downstream -- every result would just refer to
    # a different gene than intended.
    if fingerprints[config.RETRIEVAL] != fingerprints[config.RECONSTRUCTION]:
        return errors.tool_error(
            errors.CORPUS_MISMATCH,
            remedy="The two peers are reading different corpus snapshots, so row IDs "
            "would not mean the same thing to both. Report this and finish your turn.",
            task_id=state.task_id,
            fingerprints=fingerprints,
        )

    state.corpus_fingerprint = fingerprints[config.RETRIEVAL]
    state.advance(Phase.PREPARED)
    return errors.tool_ok(
        {
            "task_id": state.task_id,
            "gap_id": state.gap_id,
            "mode": state.mode,
            "organism": state.organism or None,
            "query_flank": config.QUERY_FLANK,
            "query_len": len(state.query_text),
            "gap_length": state.gap_length,
            "left_len": len(state.left),
            "right_len": len(state.right),
            "retrieval_ok": peers[config.RETRIEVAL],
            "reconstruction_ok": peers[config.RECONSTRUCTION],
            "corpus_fingerprint": state.corpus_fingerprint,
            "next_step": "Call delegate_reconstruction with this task_id.",
        }
    )


@tool(
    "delegate_reconstruction",
    "Send the prepared task to the reconstruction agent over A2A and wait for its "
    "result. That agent negotiates candidate batches with the retrieval agent itself.",
    {"task_id": str},
)
@guarded("delegate_reconstruction", requires=Phase.PREPARED, max_calls=2)
async def delegate_reconstruction(args: dict[str, Any], state: CoordTask) -> dict[str, Any]:
    client = A2AClient(config.PEER_URLS[config.RECONSTRUCTION], timeout=config.PEER_TIMEOUT_S)
    message = a2a_types.build_message(
        role="user",
        parts=[
            a2a_types.data_part(
                {
                    "op": "reconstruct",
                    "gap_id": state.gap_id,
                    "gap_length": state.gap_length,
                    "left": state.left,
                    "right": state.right,
                    "organism": state.organism,
                    "query_text": state.query_text,
                    "context_id": state.context_id,
                }
            )
        ],
        context_id=state.context_id,
    )

    started = time.monotonic()
    try:
        task = await client.send(message)
    except Exception as exc:
        return errors.tool_error(
            errors.PEER_UNREACHABLE,
            remedy="The reconstruction agent could not be reached. Report the failure "
            "and finish your turn.",
            task_id=state.task_id,
            detail=f"{type(exc).__name__}: {exc}",
        )

    peer_state = a2a_types.task_state(task)
    data = a2a_types.artifact_data(task)
    if peer_state is not a2a_types.TaskState.COMPLETED or not data.get("sequence"):
        return errors.tool_error(
            errors.PEER_FAILED,
            remedy="The reconstruction agent did not produce a sequence. Report the "
            "failure and finish your turn; do not fabricate a result.",
            task_id=state.task_id,
            peer_task_state=peer_state.value,
            detail={k: v for k, v in data.items() if k != "sequence"},
        )

    state.peer_result = data
    state.advance(Phase.DELEGATED)

    # The sequence itself stays out of the model's context; it lives in state and is
    # written into the artifact by the harness.
    summary = {k: v for k, v in data.items() if k not in ("sequence", "attempts", "budget")}
    summary.update(
        {
            "task_id": state.task_id,
            "peer_task_state": peer_state.value,
            "sequence_preview": str(data.get("sequence", ""))[:60],
            "round_trip_s": round(time.monotonic() - started, 2),
            "next_step": "Call evaluate_result with this task_id.",
        }
    )
    return errors.tool_ok(summary)


@tool(
    "evaluate_result",
    "Apply the acceptance gate: produced length within tolerance of gap_length, alphabet "
    "a subset of ACGTN, and context tokens admitted greater than zero.",
    {"task_id": str},
)
@guarded("evaluate_result", requires=Phase.DELEGATED, max_calls=3)
async def evaluate_result(args: dict[str, Any], state: CoordTask) -> dict[str, Any]:
    result = state.peer_result or {}
    sequence = str(result.get("sequence", ""))
    admitted = int(result.get("context_tokens_admitted", 0) or 0)

    accepted, reasons = gate.evaluate(
        sequence, state.gap_length, admitted, tolerance=config.TOLERANCE
    )
    state.accepted = accepted
    state.reasons = reasons
    state.verdict = gate.verdict(accepted, reasons)
    state.advance(Phase.EVALUATED)

    return errors.tool_ok(
        {
            "task_id": state.task_id,
            "accepted": accepted,
            "verdict": state.verdict,
            "reasons": reasons,
            "diagnosis": gate.explain(reasons),
            "produced_length": len(sequence),
            "gap_length": state.gap_length,
            "tolerance_bases": gate.tolerance_bases(state.gap_length, config.TOLERANCE),
            "context_tokens_admitted": admitted,
            "truncated": bool(result.get("truncated")),
            "next_step": "Call emit_trace with this task_id to finish.",
        }
    )


@tool(
    "emit_trace",
    "Finalise and return the run trace: mode, candidates offered and consumed, context "
    "tokens admitted, generation attempts, final mask count and verdict.",
    {"task_id": str},
)
@guarded("emit_trace", requires=Phase.DELEGATED, max_calls=3)
async def emit_trace(args: dict[str, Any], state: CoordTask) -> dict[str, Any]:
    return errors.tool_ok(build_trace(state))


def build_trace(state: CoordTask) -> dict[str, Any]:
    """One record per run. Called by the tool and again by the harness at the end.

    The harness calls it unconditionally, so a model that stops early still leaves a
    trace saying so rather than leaving nothing behind.
    """
    result = state.peer_result or {}
    return {
        "task_id": state.task_id,
        "context_id": state.context_id,
        "accession": state.accession,
        "gap_id": state.gap_id,
        "gap_length": state.gap_length,
        "mode": state.mode or MODE_UNRESTRICTED,
        "organism": state.organism or None,
        "corpus_fingerprint": state.corpus_fingerprint,
        "query_flank": config.QUERY_FLANK,
        "model_flank": config.MODEL_FLANK,
        "candidates_offered": int(result.get("candidates_offered", 0) or 0),
        "candidates_consumed": int(result.get("candidates_consumed", 0) or 0),
        "context_tokens_admitted": int(result.get("context_tokens_admitted", 0) or 0),
        "free_tokens": int((result.get("budget") or {}).get("free_tokens", 0) or 0),
        "base_tokens": int((result.get("budget") or {}).get("base_tokens", 0) or 0),
        "input_tokens": int(result.get("input_tokens", 0) or 0),
        "truncated": bool(result.get("truncated")),
        "generation_attempts": int(result.get("attempts_used", 0) or 0),
        "seed_mask_count": int(result.get("seed_mask_count", 0) or 0),
        "final_mask_count": int(result.get("final_mask_count", 0) or 0),
        "converged": bool(result.get("converged")),
        "produced_length": int(result.get("produced_length", 0) or 0),
        "alphabet_violations": int(result.get("alphabet_violations", 0) or 0),
        "admitted_row_ids": list(result.get("admitted_row_ids", []) or []),
        # Both hops beyond the coordinator, so a test can assert tool confinement and
        # pool reuse end to end rather than only in the logs.
        "reconstruction_calls": dict(result.get("tool_calls_by_name", {}) or {}),
        "reconstruction_tool_calls": list(result.get("llm_tool_calls", []) or []),
        "retrieval_tool_calls": list(result.get("retrieval_tool_calls", []) or []),
        "retrieval_rounds": list(result.get("retrieval_rounds", []) or []),
        "verdict": state.verdict or "NO_RESULT",
        "accepted": bool(state.accepted),
        "reasons": list(state.reasons),
        "phase": state.phase.value,
        "calls": dict(state.call_counts),
        "elapsed_s": round(time.monotonic() - state.created_at, 2),
        "coordinator_trace": state.trace,
    }


TOOLS = [prepare_query, delegate_reconstruction, evaluate_result, emit_trace]
