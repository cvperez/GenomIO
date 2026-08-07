"""Two runtime properties that previously only ever showed up in logs.

**Tool confinement.** `allowed_tools` alone did not stop the model reaching the built-in
`ToolSearch`; the first smoke run recorded a call to it. Setting `tools` as well is the
fix, and it is invisible in every passing run — nothing fails when a stray tool is
available, the model just has an extra option it should not have.

**Pool persistence.** A retrieval pool has to outlive the A2A request that opened it, or
`extend_context` either fails or silently re-ranks. The bug was that the agent dropped
task state when a request finished. A rebuilt pool would still return plausible row IDs,
so the negotiation would have looked fine while re-embedding the query every round.

Both are checked here without an LLM: the SDK options are built and asserted on directly,
and the agent's message handler is driven with a stubbed model turn.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from multiagent_system import config
from multiagent_system.a2a import types as a2a
from multiagent_system.agents import coordinator, reconstruction, retrieval
from multiagent_system.runtime.agent import LLMAgent
from multiagent_system.runtime.state import REGISTRY, Phase, RetrievalSession
from multiagent_system.tools import retrieval_tools as vt

SPECS = [
    ("coordinator", coordinator.build_spec),
    ("retrieval", retrieval.build_spec),
    ("reconstruction", reconstruction.build_spec),
]


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def clean_registry():
    REGISTRY.clear()
    yield
    REGISTRY.clear()


# ==================================================================== tool confinement
@pytest.mark.parametrize("name,factory", SPECS, ids=[s[0] for s in SPECS])
class TestToolConfinement:
    def test_both_tools_and_allowed_tools_are_set(self, name, factory):
        """`allowed_tools` alone let ToolSearch through in the first smoke run."""
        options = LLMAgent(factory()).build_options_kwargs()
        assert "tools" in options, (
            "only allowed_tools is set; the built-in tool set is then unrestricted"
        )
        assert options["tools"] == options["allowed_tools"]

    def test_only_this_agent_s_own_mcp_tools_are_offered(self, name, factory):
        spec = factory()
        options = LLMAgent(spec).build_options_kwargs()
        expected = {f"mcp__{spec.server_key}__{t.name}" for t in spec.tools}
        assert set(options["tools"]) == expected
        for tool_id in options["tools"]:
            assert tool_id.startswith(f"mcp__{spec.server_key}__")

    def test_no_filesystem_or_shell_tool_is_reachable(self, name, factory):
        """There must be no route by which a model could fabricate an answer."""
        options = LLMAgent(factory()).build_options_kwargs()
        offered = set(options["tools"]) | set(options["allowed_tools"])
        for builtin in (
            "Bash", "Read", "Write", "Edit", "ToolSearch", "WebFetch", "WebSearch", "Task",
        ):
            assert builtin not in offered

    def test_project_settings_cannot_widen_the_tool_set(self, name, factory):
        options = LLMAgent(factory()).build_options_kwargs()
        assert options["setting_sources"] == []

    def test_the_loop_is_bounded(self, name, factory):
        options = LLMAgent(factory()).build_options_kwargs()
        assert 0 < options["max_turns"] <= 64

    def test_the_model_is_pinned_rather_than_inherited(self, name, factory):
        """A run whose model is whatever the CLI happened to be set to is not a result."""
        assert LLMAgent(factory()).build_options_kwargs()["model"] == config.AGENT_MODEL


def test_a_trace_containing_a_non_mcp_tool_call_is_detectable():
    """The check test.py applies end to end, exercised on a synthetic trace."""
    from multiagent_system.test import find_foreign_tool_calls

    clean = {
        "reconstruction_tool_calls": [{"name": "mcp__reconstruction_agent__generate"}],
        "retrieval_tool_calls": [{"name": "mcp__retrieval_agent__retrieve_context"}],
    }
    assert find_foreign_tool_calls(clean, []) == []

    leaked = {
        "reconstruction_tool_calls": [{"name": "mcp__reconstruction_agent__generate"}],
        "retrieval_tool_calls": [{"name": "ToolSearch"}],
    }
    assert find_foreign_tool_calls(leaked, [{"name": "Bash"}]) == ["ToolSearch", "Bash"]


# ==================================================================== pool persistence
class StubbedRetrievalAgent(LLMAgent):
    """A retrieval agent whose "model" calls one tool with a fixed batch size."""

    def __init__(self, spec, batches):
        super().__init__(spec)
        self.batches = list(batches)
        self.calls: list[str] = []

    async def _run_llm(self, prompt: str, state):
        size = self.batches.pop(0)
        tool = "extend_context" if "Operation: extend" in prompt else "retrieve_context"
        self.calls.append(tool)
        handler = {"retrieve_context": vt.retrieve_context, "extend_context": vt.extend_context}[
            tool
        ].handler
        await handler({"request_id": state.task_id, "batch_size": size})
        return f"served a batch via {tool}", [{"name": f"mcp__retrieval_agent__{tool}"}], ""


@pytest.fixture
def stub_search(monkeypatch):
    """A 40-row ranking, and a counter proving how often the query was embedded."""
    embeds = {"n": 0}

    def fake(query_text, k):
        embeds["n"] += 1
        rows = list(range(min(k, 40)))
        return rows, [round(0.9 - 0.005 * i, 4) for i in rows]

    monkeypatch.setattr(vt, "_search_unrestricted", fake)
    return embeds


@pytest.fixture
def stub_agent(fake_store, fake_organisms, stub_search):
    def build(batches):
        return StubbedRetrievalAgent(retrieval.build_spec(), batches)

    return build


def request_for(request_id: str, op: str, batch_size: int, query: str = "ACGT" * 60):
    data: dict[str, Any] = {
        "op": op,
        "request_id": request_id,
        "context_id": "ctx-1",
        "batch_size": batch_size,
    }
    if op == "retrieve":
        data["query_text"] = query
        data["organism"] = ""
    return a2a.build_message(role="user", parts=[a2a.data_part(data)], context_id="ctx-1")


class TestRetrievalPoolSurvivesTheRequest:
    def test_a_second_request_continues_the_same_ranking(self, stub_agent, stub_search):
        agent = stub_agent([10, 5, 2])

        first = a2a.artifact_data(run(agent.handle_message(request_for("p-1", "retrieve", 10))))
        second = a2a.artifact_data(run(agent.handle_message(request_for("p-1", "extend", 5))))
        third = a2a.artifact_data(run(agent.handle_message(request_for("p-1", "extend", 2))))

        # The cursor advances; the ranking does not restart.
        assert first["row_ids"] == list(range(0, 10))
        assert second["row_ids"] == list(range(10, 15))
        assert third["row_ids"] == list(range(15, 17))
        assert [first, second, third][-1]["served_total"] == 17

        # One pool, one ranking, throughout.
        assert {r["pool_size"] for r in (first, second, third)} == {40}

    def test_extending_costs_no_embedding_work(self, stub_agent, stub_search):
        """If the pool were rebuilt each time, this would be 3 instead of 1."""
        agent = stub_agent([10, 5, 2])
        for op, size in (("retrieve", 10), ("extend", 5), ("extend", 2)):
            run(agent.handle_message(request_for("p-1", op, size)))
        assert stub_search["n"] == 1

    def test_no_row_is_ever_served_twice(self, stub_agent, stub_search):
        agent = stub_agent([10, 5, 2])
        served: list[int] = []
        for op, size in (("retrieve", 10), ("extend", 5), ("extend", 2)):
            served += a2a.artifact_data(run(agent.handle_message(request_for("p-1", op, size))))[
                "row_ids"
            ]
        assert len(served) == len(set(served)) == 17

    def test_a_different_request_id_gets_its_own_pool(self, stub_agent, stub_search):
        agent = stub_agent([4, 4])
        run(agent.handle_message(request_for("p-1", "retrieve", 4)))
        other = a2a.artifact_data(run(agent.handle_message(request_for("p-2", "retrieve", 4))))
        assert other["row_ids"] == list(range(0, 4))  # starts over, as it must
        assert stub_search["n"] == 2

    def test_the_retrieval_agent_declares_that_it_keeps_state(self):
        """The flag the whole property rests on; the other two agents must not set it."""
        assert retrieval.build_spec().persist_state is True
        assert reconstruction.build_spec().persist_state is False
        assert coordinator.build_spec().persist_state is False

    def test_a_session_that_was_dropped_reports_no_pool_rather_than_re_ranking(
        self, stub_agent, stub_search
    ):
        """What the bug looked like: extend arriving with the pool gone.

        It must surface as NO_POOL, not as a silently rebuilt ranking that happens to
        return plausible row IDs.
        """
        agent = stub_agent([10])
        run(agent.handle_message(request_for("p-1", "retrieve", 10)))
        REGISTRY.drop("p-1")

        session = RetrievalSession(task_id="p-1", context_id="ctx-1")
        REGISTRY.put(session)
        from multiagent_system.runtime import errors

        result = run(vt.extend_context.handler({"request_id": "p-1", "batch_size": 5}))
        assert errors.error_code(result) == errors.NO_POOL
        assert session.phase is not Phase.POOLED
