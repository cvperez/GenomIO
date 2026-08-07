"""The LLM-driven agent: an A2A server, an MCP tool server and a Claude loop in one process.

Each inbound A2A ``message/send`` opens a task, runs one Claude Agent SDK conversation
whose only available tools are this agent's own, and answers with whatever the tool
handlers wrote into the task state.

The load-bearing detail is that last clause. **The reply is serialised from state, never
parsed out of the model's text.** The model's prose is attached to the artifact as a
TextPart and read by nothing. So a model that narrates an invented nucleotide sequence
changes no output, and a model that stops early yields a ``failed`` task carrying its
trace rather than a plausible-looking answer.

That is what makes an LLM control path safe here. The model chooses *when* to call the
tools; it cannot choose to skip a precondition, mutate a target length, or manufacture a
result, because every one of those is checked by code before any work happens.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import config
from ..a2a import types as a2a_types
from ..a2a.card import build_agent_card
from ..a2a.store import TaskStore
from . import errors
from .state import REGISTRY, BaseState, Phase


@dataclass
class AgentSpec:
    """Everything that differs between the three agents."""

    role: str
    name: str
    description: str
    system_prompt: str
    skills: list[dict[str, Any]]
    tools: list[Any]
    server_key: str
    allowed_tools: list[str]
    warm: Callable[[], dict[str, Any]]
    make_state: Callable[[dict[str, Any], str], BaseState]
    build_prompt: Callable[[BaseState, dict[str, Any]], str]
    finish: Callable[[BaseState], dict[str, Any]]
    artifact_name: str
    # Whether the run actually produced something, which decides completed vs failed.
    # Each agent knows this differently: reconstruction sets ``state.result``, retrieval
    # has served a batch from a pool, the coordinator has passed its gate.
    has_result: Callable[[BaseState], bool] = lambda state: state.result is not None
    max_turns: int = config.MAX_TURNS
    # Keep the task state after the request finishes. Only the retrieval agent needs
    # this, so a follow-up extend_context lands on the pool the first call ranked.
    persist_state: bool = False
    organisms: Callable[[], list[str]] | None = None
    fingerprint: Callable[[], str] | None = None
    extra_card_fields: dict[str, Any] = field(default_factory=dict)


class LLMAgent:
    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec
        self.store = TaskStore()
        self.started_at = time.monotonic()
        self.models_loaded = False
        self.warm_info: dict[str, Any] = {}
        self._sdk = None
        self._mcp_server = None
        self._turns = 0

        # The tool ids the model sees are mcp__<server_key>__<tool_name>. If the server
        # key here and the one passed to create_sdk_mcp_server ever disagree, the model
        # is simply told there are no tools -- it does not error, it just talks.
        expected = {f"mcp__{spec.server_key}__{t.name}" for t in spec.tools}
        if expected != set(spec.allowed_tools):
            raise RuntimeError(
                f"tool id mismatch for {spec.role}: tools give {sorted(expected)} but "
                f"allowed_tools is {sorted(spec.allowed_tools)}"
            )

    # ------------------------------------------------------------------ lifecycle
    def warm_up(self) -> dict[str, Any]:
        self.warm_info = self.spec.warm() or {}
        self.models_loaded = True
        return self.warm_info

    def build_mcp_server(self):
        from claude_agent_sdk import create_sdk_mcp_server

        self._mcp_server = create_sdk_mcp_server(
            name=self.spec.server_key, version=config.AGENT_VERSION, tools=self.spec.tools
        )
        return self._mcp_server

    # ------------------------------------------------------------------ endpoints
    def agent_card(self) -> dict[str, Any]:
        card = build_agent_card(
            name=self.spec.name,
            description=self.spec.description,
            url=config.PEER_URLS[self.spec.role],
            skills=self.spec.skills,
            corpus_fingerprint=self.spec.fingerprint() if self.spec.fingerprint else "",
            organisms=self.spec.organisms() if self.spec.organisms else None,
        )
        card["metadata"].update(self.spec.extra_card_fields)
        card["metadata"]["x-tools"] = [t.name for t in self.spec.tools]
        return card

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy" if self.models_loaded else "starting",
            "agent": self.spec.name,
            "role": self.spec.role,
            "models_loaded": self.models_loaded,
            "uptime_s": round(time.monotonic() - self.started_at, 1),
            "open_tasks": len(REGISTRY.ids()),
            "llm_turns": self._turns,
            "warm": self.warm_info,
        }

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return self.store.get(task_id)

    def cancel_task(self, task_id: str) -> dict[str, Any] | None:
        task = self.store.get(task_id)
        if task is None:
            return None
        state = REGISTRY.get(task_id)
        if state is not None:
            state.advance(Phase.TERMINATED)
        if a2a_types.task_state(task).is_terminal:
            return task
        return self.store.set_state(task_id, a2a_types.TaskState.CANCELED)

    # ------------------------------------------------------------------ the request
    async def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        data = a2a_types.first_data(message)
        context_id = message.get("contextId") or data.get("context_id") or a2a_types.new_id(
            "ctx"
        )

        task = self.store.create(
            a2a_types.build_task(context_id=context_id, message=message)
        )
        task_id = task["id"]

        try:
            state = self.spec.make_state(data, task_id)
        except Exception as exc:
            # A malformed request is rejected, not failed: the agent is fine, the ask
            # was not.
            self.store.set_state(task_id, a2a_types.TaskState.REJECTED)
            self.store.add_artifact(
                task_id,
                a2a_types.build_artifact(
                    name="error",
                    parts=[
                        a2a_types.data_part(
                            {"error": "BAD_REQUEST", "detail": f"{type(exc).__name__}: {exc}"}
                        )
                    ],
                ),
            )
            return self.store.get(task_id)

        state.context_id = context_id
        REGISTRY.put(state)
        self.store.set_state(task_id, a2a_types.TaskState.WORKING)

        prompt = self.spec.build_prompt(state, data)
        narrative, tool_calls, error = await self._run_llm(prompt, state)
        state.narrative = narrative

        payload = self.spec.finish(state)
        payload["llm_tool_calls"] = tool_calls
        if error:
            payload["llm_error"] = error

        produced = bool(self.spec.has_result(state))
        parts = [a2a_types.data_part(payload)]
        if narrative:
            # Advisory only. Nothing reads this.
            parts.append(a2a_types.text_part(narrative))

        self.store.add_artifact(
            task_id,
            a2a_types.build_artifact(
                name=self.spec.artifact_name,
                parts=parts,
                description=f"{self.spec.name} result for task {task_id}",
            ),
        )
        self.store.set_state(
            task_id,
            a2a_types.TaskState.COMPLETED if produced else a2a_types.TaskState.FAILED,
        )
        self._log_task(state, payload, produced)
        if not self.spec.persist_state:
            REGISTRY.drop(state.task_id)
        return self.store.get(task_id)

    # ------------------------------------------------------------------ the LLM loop
    def build_options_kwargs(self) -> dict[str, Any]:
        """The SDK options for this agent.

        Split out from :meth:`_run_llm` so it can be asserted on without starting a
        model -- see ``tests/test_agent_runtime.py``. Both ``tools`` and
        ``allowed_tools`` are set: ``allowed_tools`` alone still let the built-in
        ``ToolSearch`` through, which the first smoke run caught.
        """
        options_kwargs: dict[str, Any] = {
            "system_prompt": self.spec.system_prompt,
            "mcp_servers": {self.spec.server_key: self._mcp_server},
            # Only this agent's own tools. No Bash, no Read, no Write -- there is no
            # alternate route by which the model could manufacture an answer.
            "tools": list(self.spec.allowed_tools),
            "allowed_tools": list(self.spec.allowed_tools),
            "max_turns": self.spec.max_turns,
            "permission_mode": "bypassPermissions",
            "setting_sources": [],
            "model": config.AGENT_MODEL,
        }
        cli = os.path.expanduser("~/.local/bin/claude")
        if os.path.exists(cli):
            options_kwargs["cli_path"] = cli
        return options_kwargs

    async def _run_llm(
        self, prompt: str, state: BaseState
    ) -> tuple[str, list[dict[str, Any]], str]:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            query,
        )

        options_kwargs = self.build_options_kwargs()
        texts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        result_text = ""
        error = ""

        try:
            async for chunk in query(prompt=prompt, options=ClaudeAgentOptions(**options_kwargs)):
                if isinstance(chunk, AssistantMessage):
                    self._turns += 1
                    for block in chunk.content or []:
                        if isinstance(block, TextBlock):
                            texts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            tool_calls.append(
                                {"name": block.name, "args": _small(block.input)}
                            )
                elif isinstance(chunk, ResultMessage):
                    if getattr(chunk, "result", None):
                        result_text = str(chunk.result)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        narrative = (result_text or "\n".join(t for t in texts if t)).strip()
        return narrative, tool_calls, error

    # ------------------------------------------------------------------ logging
    def _log_task(self, state: BaseState, payload: dict[str, Any], produced: bool) -> None:
        try:
            os.makedirs(config.LOG_DIR, exist_ok=True)
            path = os.path.join(config.LOG_DIR, f"tasks-{self.spec.role}.jsonl")
            record = {
                "ts": a2a_types.utcnow(),
                "role": self.spec.role,
                "task_id": state.task_id,
                "context_id": state.context_id,
                "produced": produced,
                "phase": state.phase.value,
                "calls": dict(state.call_counts),
                "trace": state.trace,
                "narrative": state.narrative[:500],
                "payload": {
                    k: v for k, v in payload.items() if k not in ("sequence", "attempts")
                },
            }
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception:
            pass  # logging must never take down a run


def no_result_payload(state: BaseState, extra: dict[str, Any] | None = None):
    """What an agent returns when its model finished without producing anything.

    Loud and diagnostic rather than empty: the phase reached, which tools were called and
    how often, and the full precondition trace.
    """
    payload = {
        "error": errors.NO_RESULT_PRODUCED,
        "task_id": state.task_id,
        "phase": state.phase.value,
        "call_counts": dict(state.call_counts),
        "trace": state.trace,
    }
    if extra:
        payload.update(extra)
    return payload


def _small(value: Any, limit: int = 200) -> Any:
    """Truncate tool arguments before logging them."""
    if isinstance(value, dict):
        return {k: _small(v, limit) for k, v in value.items()}
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"...<{len(value)} chars>"
    if isinstance(value, list) and len(value) > 32:
        return value[:32] + [f"...<{len(value)} items>"]
    return value
