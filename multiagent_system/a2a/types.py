"""A2A protocol data types, JSON-RPC envelopes and builders.

Only what these three agents actually exchange is modelled: Messages made of TextParts
and DataParts, Tasks with the full nine-state lifecycle, and Artifacts. Streaming, push
notifications and FileParts are deliberately absent -- see ``server.py`` and the README
for what is declined and why.

The wire form is the JSON-RPC one from the A2A specification (``"submitted"``,
``"working"``, ...). The protobuf spelling (``TASK_STATE_SUBMITTED``) is available via
``TaskState.proto_name`` for anyone reading the gRPC side of the spec.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class TaskState(str, Enum):
    """The nine lifecycle states defined by A2A.

    Four are terminal (completed, failed, canceled, rejected), two are interrupted and
    resumable (input-required, auth-required), three are active (unknown, submitted,
    working). These agents traverse submitted -> working -> completed|failed, and use
    rejected for a malformed request or a corpus fingerprint mismatch. The rest exist so
    a client written against the specification does not break on an unknown value.
    """

    UNKNOWN = "unknown"
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    INPUT_REQUIRED = "input-required"
    REJECTED = "rejected"
    AUTH_REQUIRED = "auth-required"

    @property
    def proto_name(self) -> str:
        return "TASK_STATE_" + self.name

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL

    @property
    def is_interrupted(self) -> bool:
        return self in _INTERRUPTED


_TERMINAL = frozenset(
    {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED, TaskState.REJECTED}
)
_INTERRUPTED = frozenset({TaskState.INPUT_REQUIRED, TaskState.AUTH_REQUIRED})


# --------------------------------------------------------------------------- parts
def text_part(text: str) -> dict[str, Any]:
    return {"kind": "text", "text": text}


def data_part(data: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "data", "data": data}


def part_text(part: dict[str, Any]) -> str:
    return str(part.get("text", "")) if part.get("kind") == "text" else ""


def part_data(part: dict[str, Any]) -> dict[str, Any]:
    value = part.get("data") if part.get("kind") == "data" else None
    return value if isinstance(value, dict) else {}


def first_data(message: dict[str, Any]) -> dict[str, Any]:
    """The first DataPart of a message, or {}.

    Everything structured travels here. The model never sees or reproduces it, which is
    why gap_length and the contig strings survive a run intact.
    """
    for part in message.get("parts", []) or []:
        found = part_data(part)
        if found:
            return found
    return {}


def joined_text(message: dict[str, Any]) -> str:
    return "\n".join(
        t for t in (part_text(p) for p in message.get("parts", []) or []) if t
    ).strip()


# --------------------------------------------------------------------------- message
def build_message(
    *,
    role: str,
    parts: list[dict[str, Any]],
    context_id: str | None = None,
    task_id: str | None = None,
    reference_task_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "kind": "message",
        "role": role,
        "parts": parts,
        "messageId": new_id("msg"),
    }
    if context_id:
        message["contextId"] = context_id
    if task_id:
        message["taskId"] = task_id
    if reference_task_ids:
        message["referenceTaskIds"] = list(reference_task_ids)
    if metadata:
        message["metadata"] = metadata
    return message


# --------------------------------------------------------------------------- artifact
def build_artifact(
    *,
    name: str,
    parts: list[dict[str, Any]],
    description: str = "",
    artifact_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "artifactId": artifact_id or new_id("art"),
        "name": name,
        "parts": parts,
    }
    if description:
        artifact["description"] = description
    if metadata:
        artifact["metadata"] = metadata
    return artifact


def artifact_data(task: dict[str, Any]) -> dict[str, Any]:
    """The DataPart of a task's first artifact -- the machine-readable result.

    Callers read this. They never read the assistant's prose, which is carried in a
    sibling TextPart purely for the audit trail.
    """
    for artifact in task.get("artifacts", []) or []:
        for part in artifact.get("parts", []) or []:
            found = part_data(part)
            if found:
                return found
    return {}


# --------------------------------------------------------------------------- task
def build_task(
    *,
    context_id: str,
    state: TaskState = TaskState.SUBMITTED,
    task_id: str | None = None,
    message: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "kind": "task",
        "id": task_id or new_id("task"),
        "contextId": context_id,
        "status": build_status(state),
        "history": [message] if message else [],
        "artifacts": [],
        "metadata": metadata or {},
    }
    return task


def build_status(
    state: TaskState, message: dict[str, Any] | None = None
) -> dict[str, Any]:
    status: dict[str, Any] = {"state": state.value, "timestamp": utcnow()}
    if message:
        status["message"] = message
    return status


def task_state(task: dict[str, Any]) -> TaskState:
    raw = (task.get("status") or {}).get("state", TaskState.UNKNOWN.value)
    try:
        return TaskState(raw)
    except ValueError:
        return TaskState.UNKNOWN


# --------------------------------------------------------------------------- json-rpc
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

ERROR_MESSAGES = {
    PARSE_ERROR: "Parse error",
    INVALID_REQUEST: "Invalid Request",
    METHOD_NOT_FOUND: "Method not found",
    INVALID_PARAMS: "Invalid params",
    INTERNAL_ERROR: "Internal error",
}


def rpc_request(method: str, params: dict[str, Any], request_id: str | None = None):
    return {
        "jsonrpc": "2.0",
        "id": request_id or new_id("req"),
        "method": method,
        "params": params,
    }


def rpc_result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def rpc_error(
    request_id: Any, code: int, message: str = "", data: Any = None
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": code,
        "message": message or ERROR_MESSAGES.get(code, "Error"),
    }
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}
