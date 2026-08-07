"""A2A HTTP server on the Python standard library.

Deliberately not FastAPI: ``fastapi``/``uvicorn`` are not installed in this environment
and nothing here needs them. ``http.server`` covers five routes and keeps the protocol
layer importable by the system interpreter, so ``tests/test_a2a.py`` runs without the
Claude Agent SDK.

Threading contract, which matters:

  The asyncio event loop owns the main thread -- the Claude Agent SDK binds an anyio
  task group and a CLI subprocess transport to whichever loop created them, so it must
  not be built or torn down from a worker. The HTTP server therefore runs on a daemon
  thread and hands work back to the loop with ``run_coroutine_threadsafe``.

Implemented:  GET /.well-known/agent-card.json, GET /.well-known/agent-configuration,
              GET /health, GET /tasks/{id}, POST / (JSON-RPC 2.0: message/send,
              tasks/get, tasks/cancel)

Declined with -32601: message/stream, tasks/resubscribe,
              tasks/pushNotificationConfig/{set,get,list,delete}
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from .types import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    rpc_error,
    rpc_result,
    utcnow,
)

# Methods that exist in the specification but that these agents do not offer. Answering
# with "method not found" is honest; silently accepting and never delivering is not.
UNSUPPORTED_METHODS = {
    "message/stream": "streaming is not supported; use message/send (capabilities.streaming is false)",
    "tasks/resubscribe": "streaming is not supported, so there is no stream to resubscribe to",
    "tasks/pushNotificationConfig/set": "push notifications are not supported (capabilities.pushNotifications is false)",
    "tasks/pushNotificationConfig/get": "push notifications are not supported",
    "tasks/pushNotificationConfig/list": "push notifications are not supported",
    "tasks/pushNotificationConfig/delete": "push notifications are not supported",
}


class A2AHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_handler(
    *,
    loop: asyncio.AbstractEventLoop,
    agent_card: Callable[[], dict[str, Any]],
    health: Callable[[], dict[str, Any]],
    on_message_send: Callable[[dict[str, Any]], Any],
    get_task: Callable[[str], dict[str, Any] | None],
    cancel_task: Callable[[str], dict[str, Any] | None],
    request_timeout: float,
):
    """Build the request handler class, closed over the agent's callbacks."""

    class A2AHandler(BaseHTTPRequestHandler):
        # HTTP/1.1 with an explicit Content-Length on every response. Without both,
        # keep-alive clients (httpx defaults to them) hang waiting for a body end.
        protocol_version = "HTTP/1.1"
        server_version = "GenomIO-A2A/1.0"

        # ------------------------------------------------------------------ helpers
        def _send_json(self, code: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> bytes:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return b""
            return self.rfile.read(length) if length > 0 else b""

        def log_message(self, *_args) -> None:  # silence per-request stderr noise
            return

        # ------------------------------------------------------------------ routes
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            try:
                if path in ("/.well-known/agent-card.json", "/.well-known/agent-configuration"):
                    self._send_json(200, agent_card())
                elif path == "/health":
                    self._send_json(200, health())
                elif path.startswith("/tasks/"):
                    task = get_task(path[len("/tasks/"):])
                    if task is None:
                        self._send_json(404, {"error": "task not found"})
                    else:
                        self._send_json(200, task)
                else:
                    self._send_json(404, {"error": "unknown path", "path": path})
            except Exception as exc:  # never let a handler kill the server thread
                self._send_json(500, {"error": str(exc), "ts": utcnow()})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path not in ("/", "/rpc"):
                self._send_json(404, {"error": "unknown path", "path": path})
                return

            raw = self._read_body()
            try:
                request = json.loads(raw or b"{}")
            except (ValueError, UnicodeDecodeError):
                self._send_json(200, rpc_error(None, PARSE_ERROR))
                return

            if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                self._send_json(
                    200,
                    rpc_error(
                        request.get("id") if isinstance(request, dict) else None,
                        INVALID_REQUEST,
                        "expected a JSON-RPC 2.0 request object",
                    ),
                )
                return

            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params")
            if not isinstance(params, dict):
                params = {}

            if method in UNSUPPORTED_METHODS:
                self._send_json(
                    200,
                    rpc_error(request_id, METHOD_NOT_FOUND, UNSUPPORTED_METHODS[method]),
                )
                return

            try:
                if method == "message/send":
                    message = params.get("message")
                    if not isinstance(message, dict) or not message.get("parts"):
                        self._send_json(
                            200,
                            rpc_error(
                                request_id,
                                INVALID_PARAMS,
                                "params.message must be an object carrying parts",
                            ),
                        )
                        return
                    # Hand the work to the event loop, which owns the SDK client, and
                    # block this request thread until it finishes.
                    future = asyncio.run_coroutine_threadsafe(
                        _as_coro(on_message_send, message), loop
                    )
                    result = future.result(timeout=request_timeout)
                    self._send_json(200, rpc_result(request_id, result))

                elif method == "tasks/get":
                    task_id = str(params.get("id", ""))
                    task = get_task(task_id)
                    if task is None:
                        self._send_json(
                            200, rpc_error(request_id, INVALID_PARAMS, f"no task {task_id!r}")
                        )
                    else:
                        self._send_json(200, rpc_result(request_id, task))

                elif method == "tasks/cancel":
                    task_id = str(params.get("id", ""))
                    task = cancel_task(task_id)
                    if task is None:
                        self._send_json(
                            200, rpc_error(request_id, INVALID_PARAMS, f"no task {task_id!r}")
                        )
                    else:
                        self._send_json(200, rpc_result(request_id, task))

                else:
                    self._send_json(
                        200,
                        rpc_error(
                            request_id, METHOD_NOT_FOUND, f"unknown method {method!r}"
                        ),
                    )
            except Exception as exc:
                self._send_json(
                    200,
                    rpc_error(
                        request_id,
                        INTERNAL_ERROR,
                        f"{type(exc).__name__}: {exc}",
                    ),
                )

    return A2AHandler


async def _as_coro(fn: Callable[[dict[str, Any]], Any], message: dict[str, Any]):
    result = fn(message)
    if asyncio.iscoroutine(result):
        return await result
    return result


def start_server(
    *,
    loop: asyncio.AbstractEventLoop,
    host: str,
    port: int,
    agent_card: Callable[[], dict[str, Any]],
    health: Callable[[], dict[str, Any]],
    on_message_send: Callable[[dict[str, Any]], Any],
    get_task: Callable[[str], dict[str, Any] | None],
    cancel_task: Callable[[str], dict[str, Any] | None],
    request_timeout: float = 3600.0,
) -> A2AHTTPServer:
    """Bind and serve on a daemon thread. Returns the server; call .shutdown() to stop."""
    handler = make_handler(
        loop=loop,
        agent_card=agent_card,
        health=health,
        on_message_send=on_message_send,
        get_task=get_task,
        cancel_task=cancel_task,
        request_timeout=request_timeout,
    )
    httpd = A2AHTTPServer((host, port), handler)
    threading.Thread(
        target=httpd.serve_forever,
        kwargs={"poll_interval": 0.2},
        daemon=True,
        name=f"a2a-http:{port}",
    ).start()
    # Give the socket a moment so an immediate health poll does not race the bind.
    time.sleep(0.05)
    return httpd
