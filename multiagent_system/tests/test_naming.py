"""Tool-id naming, which fails silently rather than loudly when it drifts.

The model sees each tool as ``mcp__<server_key>__<tool_name>``, where ``server_key`` is
the key under which the MCP server is registered. If ``allowed_tools`` and the actual
tool names disagree, the SDK does not raise -- the model is simply told it has no tools
and starts talking instead of acting. These assertions turn that into a build error.
"""
from __future__ import annotations

import pytest

from multiagent_system import config
from multiagent_system.tools import coordinator_tools, reconstruction_tools, retrieval_tools

MODULES = [
    (config.COORDINATOR, coordinator_tools),
    (config.RETRIEVAL, retrieval_tools),
    (config.RECONSTRUCTION, reconstruction_tools),
]


@pytest.mark.parametrize("role,module", MODULES, ids=[m[0] for m in MODULES])
def test_allowed_tools_match_the_registered_tools(role, module):
    assert module.SERVER_KEY == config.SERVER_KEYS[role]
    assert {t.name for t in module.TOOLS} == set(module.TOOL_NAMES)
    assert module.ALLOWED_TOOLS == [
        f"mcp__{module.SERVER_KEY}__{name}" for name in module.TOOL_NAMES
    ]


@pytest.mark.parametrize("role,module", MODULES, ids=[m[0] for m in MODULES])
def test_every_tool_declares_a_schema_and_a_real_description(role, module):
    for handle in module.TOOLS:
        assert handle.description and len(handle.description) > 40
        assert isinstance(handle.input_schema, dict) and handle.input_schema
        assert callable(handle.handler)


@pytest.mark.parametrize("role,module", MODULES, ids=[m[0] for m in MODULES])
def test_every_tool_is_keyed_by_an_identifier_not_a_payload(role, module):
    """No tool takes a sequence.

    The old planner interpolated the nucleotide string into the prompt and made the model
    re-emit it as a tool argument, with no checksum and no length assertion. Here the
    only large value any tool accepts is ``query_text`` on the retrieval agent, which is
    passed through and never regenerated.
    """
    for handle in module.TOOLS:
        keys = set(handle.input_schema)
        assert keys & {"task_id", "request_id"}, f"{handle.name} has no identifier"
        assert not keys & {"sequence", "left", "right", "context", "rag_matches"}


def test_server_keys_are_distinct():
    assert len(set(config.SERVER_KEYS.values())) == len(config.SERVER_KEYS)


def test_agent_specs_build_and_validate_their_own_naming():
    """LLMAgent re-checks the coupling at construction, so a mismatch fails at startup."""
    from multiagent_system.agents import coordinator, reconstruction, retrieval
    from multiagent_system.runtime.agent import LLMAgent

    for factory in (coordinator.build_spec, retrieval.build_spec, reconstruction.build_spec):
        spec = factory()
        agent = LLMAgent(spec)  # raises if the ids disagree
        card = agent.agent_card()
        assert card["name"] and card["skills"]
        assert card["capabilities"]["streaming"] is False
        assert set(card["metadata"]["x-tools"]) == {t.name for t in spec.tools}
        assert agent.health()["models_loaded"] is False  # nothing warmed yet


def test_a_naming_mismatch_is_rejected_at_construction():
    from multiagent_system.agents import retrieval
    from multiagent_system.runtime.agent import LLMAgent

    spec = retrieval.build_spec()
    spec.allowed_tools = ["mcp__wrong_key__retrieve_context"]
    with pytest.raises(RuntimeError, match="tool id mismatch"):
        LLMAgent(spec)
