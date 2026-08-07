"""The A2A protocol layer, exercised against a real socket with a stub agent.

No models, no LLM, no SDK -- this runs under the system interpreter.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from multiagent_system.a2a import server as a2a_server
from multiagent_system.a2a import types as t
from multiagent_system.a2a.card import build_agent_card, skill
from multiagent_system.a2a.client import A2AError, SyncA2AClient
from multiagent_system.a2a.store import TaskStore, TerminalTaskError

CARD = build_agent_card(
    name="Stub Agent",
    description="a stub",
    url="http://127.0.0.1:0",
    skills=[skill(skill_id="s", name="S", description="d", tags=["t"])],
    corpus_fingerprint="deadbeefdeadbeef",
    organisms=["Fusobacterium animalis"],
)


@pytest.fixture(scope="module")
def fleet():
    """A live A2A server on an ephemeral port, driven by a background event loop."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    ready.wait(5)

    store = TaskStore()
    state = {"models_loaded": True}

    async def on_message_send(message):
        data = t.first_data(message)
        if data.get("boom"):
            raise RuntimeError("stub failure")
        task = t.build_task(
            context_id=message.get("contextId") or t.new_id("ctx"), message=message
        )
        store.create(task)
        store.set_state(task["id"], t.TaskState.WORKING)
        store.add_artifact(
            task["id"],
            t.build_artifact(
                name="echo",
                parts=[t.data_part({"echo": data}), t.text_part("narrative")],
            ),
        )
        store.set_state(task["id"], t.TaskState.COMPLETED)
        return store.get(task["id"])

    def cancel(task_id):
        try:
            return store.set_state(task_id, t.TaskState.CANCELED)
        except TerminalTaskError:
            return store.get(task_id)

    httpd = a2a_server.start_server(
        loop=loop,
        host="127.0.0.1",
        port=0,
        agent_card=lambda: CARD,
        health=lambda: {"status": "healthy", "models_loaded": state["models_loaded"]},
        on_message_send=on_message_send,
        get_task=store.get,
        cancel_task=cancel,
        request_timeout=30.0,
    )
    port = httpd.server_address[1]
    client = SyncA2AClient(f"http://127.0.0.1:{port}", timeout=30.0)

    for _ in range(50):
        try:
            client.health(timeout=1.0)
            break
        except Exception:
            time.sleep(0.1)

    yield client, store, state

    httpd.shutdown()
    loop.call_soon_threadsafe(loop.stop)


# --------------------------------------------------------------------------- discovery
def test_both_well_known_paths_return_the_identical_card(fleet):
    client, _, _ = fleet
    canonical = client.agent_card()
    alias = client._get("/.well-known/agent-configuration")
    assert canonical == alias
    assert canonical["name"] == "Stub Agent"
    assert canonical["protocolVersion"]
    assert canonical["skills"][0]["id"] == "s"


def test_card_declares_what_is_not_supported(fleet):
    client, _, _ = fleet
    caps = client.agent_card()["capabilities"]
    assert caps["streaming"] is False
    assert caps["pushNotifications"] is False


def test_card_carries_the_corpus_fingerprint(fleet):
    client, _, _ = fleet
    assert client.agent_card()["metadata"]["x-corpus-fingerprint"] == "deadbeefdeadbeef"


def test_health_reports_model_readiness(fleet):
    client, _, state = fleet
    assert client.is_ready()
    state["models_loaded"] = False
    assert not client.is_ready()
    state["models_loaded"] = True


# --------------------------------------------------------------------------- messaging
def test_message_send_round_trip_returns_a_completed_task(fleet):
    client, _, _ = fleet
    message = t.build_message(role="user", parts=[t.data_part({"gap_length": 870})])
    task = client.send(message)
    assert t.task_state(task) is t.TaskState.COMPLETED
    assert t.artifact_data(task)["echo"]["gap_length"] == 870
    assert task["contextId"]


def test_context_id_is_echoed_so_tasks_join_across_agents(fleet):
    client, store, _ = fleet
    ctx = "ctx-shared"
    for _ in range(2):
        client.send(t.build_message(role="user", parts=[t.data_part({})], context_id=ctx))
    assert len(store.by_context(ctx)) == 2


def test_tasks_get_retrieves_a_finished_task(fleet):
    client, _, _ = fleet
    task = client.send(t.build_message(role="user", parts=[t.data_part({"x": 1})]))
    assert client.get_task(task["id"])["id"] == task["id"]
    assert client._get(f"/tasks/{task['id']}")["id"] == task["id"]


# --------------------------------------------------------------------------- errors
def test_unknown_method_is_method_not_found(fleet):
    client, _, _ = fleet
    assert client.call("does/not/exist", {})["error"]["code"] == t.METHOD_NOT_FOUND


@pytest.mark.parametrize(
    "method",
    ["message/stream", "tasks/resubscribe", "tasks/pushNotificationConfig/set"],
)
def test_declined_methods_say_so_rather_than_pretending(fleet, method):
    client, _, _ = fleet
    error = client.call(method, {})["error"]
    assert error["code"] == t.METHOD_NOT_FOUND
    assert "not supported" in error["message"]


def test_malformed_json_is_a_parse_error(fleet):
    client, _, _ = fleet
    import httpx

    response = httpx.post(
        client.base_url + "/",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    assert response.json()["error"]["code"] == t.PARSE_ERROR


def test_message_without_parts_is_invalid_params(fleet):
    client, _, _ = fleet
    assert client.call("message/send", {"message": {"role": "user"}})["error"]["code"] == (
        t.INVALID_PARAMS
    )


def test_missing_jsonrpc_version_is_invalid_request(fleet):
    client, _, _ = fleet
    assert client._post({"id": 1, "method": "tasks/get"})["error"]["code"] == (
        t.INVALID_REQUEST
    )


def test_handler_exception_becomes_internal_error_not_a_dropped_connection(fleet):
    client, _, _ = fleet
    with pytest.raises(A2AError) as excinfo:
        client.send(t.build_message(role="user", parts=[t.data_part({"boom": True})]))
    assert excinfo.value.code == t.INTERNAL_ERROR


def test_unknown_task_id_is_reported(fleet):
    client, _, _ = fleet
    assert client.call("tasks/get", {"id": "nope"})["error"]["code"] == t.INVALID_PARAMS
    import httpx

    assert httpx.get(client.base_url + "/tasks/nope", timeout=10).status_code == 404


def test_responses_carry_content_length(fleet):
    client, _, _ = fleet
    import httpx

    response = httpx.get(client.base_url + "/health", timeout=10)
    assert response.headers["content-length"] == str(len(response.content))


# --------------------------------------------------------------------------- store
def test_terminal_tasks_are_immutable():
    """The specification's rule: refinement means a new task, same contextId."""
    store = TaskStore()
    task = store.create(t.build_task(context_id="ctx"))
    store.set_state(task["id"], t.TaskState.COMPLETED)
    with pytest.raises(TerminalTaskError):
        store.set_state(task["id"], t.TaskState.WORKING)


def test_store_evicts_oldest_beyond_its_bound():
    store = TaskStore(max_tasks=3)
    ids = [store.create(t.build_task(context_id="c"))["id"] for _ in range(5)]
    assert len(store) == 3
    assert store.get(ids[0]) is None
    assert store.get(ids[-1]) is not None


def test_task_state_enum_covers_the_nine_states_and_knows_which_are_terminal():
    assert len(list(t.TaskState)) == 9
    terminal = {s for s in t.TaskState if s.is_terminal}
    assert terminal == {
        t.TaskState.COMPLETED,
        t.TaskState.FAILED,
        t.TaskState.CANCELED,
        t.TaskState.REJECTED,
    }
    assert {s for s in t.TaskState if s.is_interrupted} == {
        t.TaskState.INPUT_REQUIRED,
        t.TaskState.AUTH_REQUIRED,
    }
    assert t.TaskState.SUBMITTED.proto_name == "TASK_STATE_SUBMITTED"
