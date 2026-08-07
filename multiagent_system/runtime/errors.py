"""Tool result envelopes and the error vocabulary.

Every tool returns the MCP content shape the Claude Agent SDK expects::

    {"content": [{"type": "text", "text": "<json>"}], "is_error": <bool>}

The text is always JSON, so the model reads structured fields rather than parsing prose,
and so the harness can log exactly what the model was told.

Error payloads carry a ``remedy``: a sentence naming the precise next tool to call. That
string is the only steering in the system that is emitted *at the moment of violation*
by code that knows the real state. A system prompt asking for an ordering is a request;
this is a constraint, because the tool refuses to do the work until it is satisfied.
"""
from __future__ import annotations

import json
from typing import Any

# --- lifecycle / budget ------------------------------------------------------
UNKNOWN_TASK = "UNKNOWN_TASK"
TASK_TERMINATED = "TASK_TERMINATED"
DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
CALL_BUDGET_EXHAUSTED = "CALL_BUDGET_EXHAUSTED"
PRECONDITION_FAILED = "PRECONDITION_FAILED"

# --- reconstruction ----------------------------------------------------------
BUDGET_NOT_MEASURED = "BUDGET_NOT_MEASURED"
BUDGET_MISMATCH = "BUDGET_MISMATCH"
BAD_ROW_IDS = "BAD_ROW_IDS"
WINDOW_FULL = "WINDOW_FULL"
NOTHING_ADMITTED = "NOTHING_ADMITTED"
EMPTY_CONTEXT = "EMPTY_CONTEXT"
GAP_LENGTH_MUTATED = "GAP_LENGTH_MUTATED"
BAD_THRESHOLD = "BAD_THRESHOLD"
ALREADY_GENERATED = "ALREADY_GENERATED"

# --- retrieval ---------------------------------------------------------------
BAD_BATCH_SIZE = "BAD_BATCH_SIZE"
QUERY_CHANGED = "QUERY_CHANGED"
NO_POOL = "NO_POOL"
RESTRICTED_POOL_EMPTY = "RESTRICTED_POOL_EMPTY"
UNKNOWN_ORGANISM = "UNKNOWN_ORGANISM"
EMPTY_QUERY = "EMPTY_QUERY"

# --- coordinator -------------------------------------------------------------
PEER_UNREACHABLE = "PEER_UNREACHABLE"
CORPUS_MISMATCH = "CORPUS_MISMATCH"
NOT_PREPARED = "NOT_PREPARED"
NO_RESULT = "NO_RESULT"
PEER_FAILED = "PEER_FAILED"

# Raised by the harness, never returned to the model: the agent finished its turns
# without ever producing a result, so there is nothing to report but the trace.
NO_RESULT_PRODUCED = "NO_RESULT_PRODUCED"


def tool_ok(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, separators=(",", ":"))}],
        "is_error": False,
    }


def tool_error(code: str, remedy: str = "", **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"error": PRECONDITION_FAILED, "code": code}
    payload.update(fields)
    if remedy:
        payload["remedy"] = remedy
    return {
        "content": [{"type": "text", "text": json.dumps(payload, separators=(",", ":"))}],
        "is_error": True,
    }


def read_result(result: dict[str, Any]) -> dict[str, Any]:
    """Parse a tool result back into a dict. Used by the harness and by tests."""
    for block in result.get("content", []) or []:
        text = block.get("text")
        if text:
            try:
                return json.loads(text)
            except ValueError:
                return {"text": text}
    return {}


def is_error(result: dict[str, Any]) -> bool:
    return bool(result.get("is_error"))


def error_code(result: dict[str, Any]) -> str:
    return str(read_result(result).get("code", "")) if is_error(result) else ""
