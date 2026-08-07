"""Per-task state and the precondition guard.

This module is where the design's central claim lives. The old pipeline asked for an
ordering in a system prompt -- "You MUST use these tools in the correct order" -- and
nothing checked it: ``AgentExecutor`` imposed no ordering, set no iteration cap, and had
no failure handling. Here the ordering is a *data dependency* enforced before any work
happens: ``admit_candidates`` cannot run until ``measure_budget`` has, because it needs a
free-token count only ``measure_budget`` produces, and it rejects any other value.

Two structural choices support that:

**Tools take identifiers, not payloads.** Every tool's first parameter is a task or
request id. The 14 kb contig strings, the retrieved sequences and the assembled context
all live in the state object and never appear in a tool argument or a tool result. The
model cannot corrupt what it never handles -- which is the failure mode the old planner
had, where the nucleotide string round-tripped through token generation with no checksum.

**The answer is read from state, not from prose.** The harness serialises the A2A
response from ``state.result``. If the model narrates a beautiful sequence it invented,
that text is stored as an advisory artifact and read by nothing.
"""
from __future__ import annotations

import functools
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .. import config
from . import errors


class Phase(str, Enum):
    """Monotone lifecycle. A task only ever moves forward, or to TERMINATED."""

    CREATED = "CREATED"
    POOLED = "POOLED"          # retrieval: a ranked candidate pool exists
    PREPARED = "PREPARED"      # coordinator: query sliced, peers verified
    MEASURED = "MEASURED"      # reconstruction: token budget known
    ADMITTING = "ADMITTING"    # reconstruction: at least one candidate admitted
    DELEGATED = "DELEGATED"    # coordinator: peer returned a result
    GENERATED = "GENERATED"    # reconstruction: sequence produced
    EVALUATED = "EVALUATED"    # coordinator: gate applied
    TERMINATED = "TERMINATED"  # budget/deadline blown; nothing further is allowed


_ORDER = {
    Phase.CREATED: 0,
    Phase.POOLED: 1,
    Phase.PREPARED: 1,
    Phase.MEASURED: 1,
    Phase.ADMITTING: 2,
    Phase.DELEGATED: 2,
    Phase.GENERATED: 3,
    Phase.EVALUATED: 3,
    Phase.TERMINATED: 99,
}


def phase_rank(phase: Phase) -> int:
    return _ORDER[phase]


@dataclass
class BaseState:
    task_id: str
    context_id: str = ""
    phase: Phase = Phase.CREATED
    created_at: float = field(default_factory=time.monotonic)
    deadline: float = 0.0
    call_counts: Counter = field(default_factory=Counter)
    trace: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    narrative: str = ""

    def __post_init__(self) -> None:
        if not self.deadline:
            self.deadline = self.created_at + config.TASK_DEADLINE_S

    def advance(self, phase: Phase) -> None:
        """Move forward only. Re-entering an earlier phase is silently ignored."""
        if phase_rank(phase) >= phase_rank(self.phase) or phase is Phase.TERMINATED:
            self.phase = phase

    def record(self, tool: str, ok: bool, detail: dict[str, Any]) -> None:
        self.trace.append(
            {
                "seq": len(self.trace) + 1,
                "tool": tool,
                "ok": ok,
                "phase": self.phase.value,
                "elapsed_s": round(time.monotonic() - self.created_at, 3),
                **detail,
            }
        )

    @property
    def seconds_left(self) -> float:
        return self.deadline - time.monotonic()


@dataclass
class CoordTask(BaseState):
    """One gap, from request to verdict."""

    accession: str = ""
    gap_id: str = ""
    gap_length: int = 0
    organism: str = ""
    left: str = ""
    right: str = ""
    true_sequence: str = ""
    mode: str = ""
    query_text: str = ""
    corpus_fingerprint: str = ""
    peer_result: dict[str, Any] | None = None
    accepted: bool = False
    verdict: str = ""
    reasons: list[str] = field(default_factory=list)


@dataclass
class ReconTask(BaseState):
    """One reconstruction: measure the window, fill it, generate."""

    gap_length: int = 0
    left: str = ""
    right: str = ""
    organism: str = ""
    query_text: str = ""
    retrieval_request_id: str = ""

    # measure_budget
    budget: dict[str, Any] | None = None
    model_left: str = ""
    model_right: str = ""
    base_tokens: int = 0
    free_tokens_total: int = 0
    free_tokens_remaining: int = 0
    bases_per_token: float = 0.0
    seed_mask_count: int = 0

    # What the retrieval peer did, carried back so the coordinator's trace can show it.
    # Without this the end-to-end test cannot see two hops away, and the pool-reuse and
    # tool-confinement properties would only ever be observable in logs.
    retrieval_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieval_rounds: list[dict[str, Any]] = field(default_factory=list)

    # admit_candidates
    offered_row_ids: list[int] = field(default_factory=list)
    admitted_row_ids: list[int] = field(default_factory=list)
    rejected_row_ids: list[int] = field(default_factory=list)
    context_blocks: list[str] = field(default_factory=list)
    context_tokens_admitted: int = 0

    @property
    def context(self) -> str:
        """Assembled context. Agent-internal: never returned by any tool."""
        return " ".join(self.context_blocks)


@dataclass
class RetrievalSession(BaseState):
    """A ranked candidate pool, served incrementally.

    Replaces the old fixed ``k=2``, a number with no relationship to how much room the
    tokenizer actually had. The batch size now comes from the only component that can
    know it: the Reconstruction agent, which measured the window.
    """

    # The query nucleotides live here, put in place from the A2A DataPart. They are
    # never a tool argument and never appear in the prompt: the model decides *when* to
    # search, not *what* to search for, so it cannot paraphrase or truncate the query.
    query_text: str = ""
    query_fingerprint: str = ""
    organism: str = ""
    resolved_organism: str = ""
    path: str = ""
    ranked_row_ids: list[int] = field(default_factory=list)
    ranked_scores: list[float] = field(default_factory=list)
    cursor: int = 0
    restricted_set_size: int = 0
    served: list[int] = field(default_factory=list)
    # The batch served by the most recent call. The A2A artifact reports this, not the
    # cumulative list, because each request asks for one batch.
    last_batch: list[int] = field(default_factory=list)
    last_scores: list[float] = field(default_factory=list)

    @property
    def pool_size(self) -> int:
        return len(self.ranked_row_ids)

    @property
    def exhausted(self) -> bool:
        return self.cursor >= len(self.ranked_row_ids)

    def next_batch(self, size: int) -> tuple[list[int], list[float]]:
        start, end = self.cursor, min(self.cursor + size, len(self.ranked_row_ids))
        self.cursor = end
        rows = [int(r) for r in self.ranked_row_ids[start:end]]
        scores = [float(s) for s in self.ranked_scores[start:end]]
        self.served.extend(rows)
        self.last_batch, self.last_scores = rows, scores
        return rows, scores


class TaskRegistry:
    """Process-global task table, keyed by task/request id."""

    def __init__(self) -> None:
        self._items: dict[str, BaseState] = {}
        self._lock = threading.RLock()

    def put(self, state: BaseState) -> BaseState:
        with self._lock:
            self._items[state.task_id] = state
            return state

    def get(self, task_id: str) -> BaseState | None:
        with self._lock:
            return self._items.get(task_id)

    def drop(self, task_id: str) -> None:
        with self._lock:
            self._items.pop(task_id, None)

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


REGISTRY = TaskRegistry()


def guarded(
    tool_name: str,
    *,
    id_field: str = "task_id",
    requires: Phase | None = None,
    max_calls: int | None = None,
    remedy_when_missing: str = "",
):
    """Wrap a tool handler with the precondition prologue.

    Checks, in order, before the handler body runs:

      1. the task exists
      2. it has not been terminated
      3. the wall-clock deadline has not passed
      4. this tool's call budget is not exhausted
      5. the task has reached the required phase

    Failures 3 and 4 terminate the task, so a looping model cannot keep spending. Every
    outcome, pass or fail, is appended to the task's trace.
    """

    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(args: dict[str, Any]) -> dict[str, Any]:
            task_id = str(args.get(id_field, "") or "")
            state = REGISTRY.get(task_id)

            if state is None:
                return errors.tool_error(
                    errors.UNKNOWN_TASK,
                    remedy=remedy_when_missing
                    or f"No task {task_id!r} is open on this agent. Use the {id_field} "
                    f"given in the request.",
                    **{id_field: task_id},
                    known_ids=REGISTRY.ids()[:8],
                )

            if state.phase is Phase.TERMINATED:
                state.record(tool_name, False, {"code": errors.TASK_TERMINATED})
                return errors.tool_error(
                    errors.TASK_TERMINATED,
                    remedy="This task has been stopped and will accept no further "
                    "calls. Finish your turn.",
                    **{id_field: task_id},
                )

            if state.seconds_left <= 0:
                state.advance(Phase.TERMINATED)
                state.record(tool_name, False, {"code": errors.DEADLINE_EXCEEDED})
                return errors.tool_error(
                    errors.DEADLINE_EXCEEDED,
                    remedy="The time budget for this task is spent. Finish your turn.",
                    **{id_field: task_id},
                    deadline_s=round(config.TASK_DEADLINE_S, 1),
                )

            limit = config.MAX_CALLS_PER_TOOL if max_calls is None else max_calls
            if state.call_counts[tool_name] >= limit:
                state.advance(Phase.TERMINATED)
                state.record(tool_name, False, {"code": errors.CALL_BUDGET_EXHAUSTED})
                return errors.tool_error(
                    errors.CALL_BUDGET_EXHAUSTED,
                    remedy=f"{tool_name} has been called {limit} times, which is its "
                    f"limit. Finish your turn.",
                    **{id_field: task_id},
                    calls=limit,
                )

            if requires is not None and phase_rank(state.phase) < phase_rank(requires):
                code, remedy = _precondition(tool_name, requires)
                state.record(tool_name, False, {"code": code})
                return errors.tool_error(
                    code,
                    remedy=remedy,
                    **{id_field: task_id},
                    phase=state.phase.value,
                    required_phase=requires.value,
                )

            state.call_counts[tool_name] += 1
            result = await fn(args, state)
            state.record(
                tool_name,
                not errors.is_error(result),
                {"code": errors.error_code(result)} if errors.is_error(result) else {},
            )
            return result

        return wrapper

    return decorator


def _precondition(tool_name: str, requires: Phase) -> tuple[str, str]:
    """The error code and remedy for a phase that has not been reached yet."""
    if requires is Phase.MEASURED:
        return (
            errors.BUDGET_NOT_MEASURED,
            "Call mcp__reconstruction_agent__measure_budget with this task_id first; "
            "it reports how many tokens are free for retrieved context.",
        )
    if requires is Phase.ADMITTING:
        return (
            errors.NOTHING_ADMITTED,
            "Call mcp__reconstruction_agent__admit_candidates with row IDs from the "
            "retrieval agent first; nothing has been admitted for this task.",
        )
    if requires is Phase.POOLED:
        return (
            errors.NO_POOL,
            "Call mcp__retrieval_agent__retrieve_context with this request_id first to "
            "open a candidate pool.",
        )
    if requires is Phase.PREPARED:
        return (
            errors.NOT_PREPARED,
            "Call mcp__coordinator_agent__prepare_query with this task_id first.",
        )
    if requires is Phase.DELEGATED:
        return (
            errors.NO_RESULT,
            "Call mcp__coordinator_agent__delegate_reconstruction with this task_id "
            "first; there is no result to evaluate yet.",
        )
    return (errors.PRECONDITION_FAILED, f"{tool_name} cannot run in this phase yet.")
