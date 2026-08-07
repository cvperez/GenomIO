"""A2A clients.

``A2AClient`` is async and used agent-to-agent: the Reconstruction agent is an A2A
client of the Retrieval agent, which is the negotiation edge in the design.
``SyncA2AClient`` is the same protocol over blocking I/O, for ``test.py``.

Both use ``httpx`` (already installed) when available and fall back to ``urllib`` so the
protocol layer stays importable with nothing but the standard library.

``send`` blocks until the task reaches a terminal state and returns the Task. Both
callers want the answer, so polling would add machinery for no benefit.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .types import rpc_request

try:  # pragma: no cover - exercised implicitly by whichever path is installed
    import httpx

    _HAS_HTTPX = True
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    _HAS_HTTPX = False


class A2AError(RuntimeError):
    """A JSON-RPC error response, or a transport failure."""

    def __init__(self, message: str, code: int | None = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


def _unwrap(payload: dict[str, Any]) -> Any:
    if "error" in payload:
        err = payload["error"] or {}
        raise A2AError(
            str(err.get("message", "unknown A2A error")),
            code=err.get("code"),
            data=err.get("data"),
        )
    return payload.get("result")


class SyncA2AClient:
    def __init__(self, base_url: str, timeout: float = 3600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------ transport
    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        if _HAS_HTTPX:
            response = httpx.post(
                self.base_url + "/",
                content=body,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        request = urllib.request.Request(
            self.base_url + "/", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str, timeout: float | None = None) -> dict[str, Any]:
        url = self.base_url + path
        wait = self.timeout if timeout is None else timeout
        if _HAS_HTTPX:
            response = httpx.get(url, timeout=wait)
            response.raise_for_status()
            return response.json()
        with urllib.request.urlopen(url, timeout=wait) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ------------------------------------------------------------------ protocol
    def agent_card(self, timeout: float = 10.0) -> dict[str, Any]:
        return self._get("/.well-known/agent-card.json", timeout=timeout)

    def health(self, timeout: float = 10.0) -> dict[str, Any]:
        return self._get("/health", timeout=timeout)

    def is_ready(self, timeout: float = 5.0) -> bool:
        """True once the agent has finished loading its models."""
        try:
            return bool(self.health(timeout=timeout).get("models_loaded"))
        except Exception:
            return False

    def send(self, message: dict[str, Any]) -> dict[str, Any]:
        return _unwrap(self._post(rpc_request("message/send", {"message": message})))

    def get_task(self, task_id: str) -> dict[str, Any]:
        return _unwrap(self._post(rpc_request("tasks/get", {"id": task_id})))

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        return _unwrap(self._post(rpc_request("tasks/cancel", {"id": task_id})))

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Raw call, so tests can exercise unsupported methods and error codes."""
        return self._post(rpc_request(method, params))


class A2AClient:
    """Async twin of :class:`SyncA2AClient`, used on the agent-to-agent edge."""

    def __init__(self, base_url: str, timeout: float = 600.0) -> None:
        if not _HAS_HTTPX:  # pragma: no cover
            raise RuntimeError("A2AClient needs httpx; use SyncA2AClient instead")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.base_url + "/",
                content=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()

    async def agent_card(self, timeout: float = 10.0) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(self.base_url + "/.well-known/agent-card.json")
            response.raise_for_status()
            return response.json()

    async def health(self, timeout: float = 10.0) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(self.base_url + "/health")
            response.raise_for_status()
            return response.json()

    async def send(self, message: dict[str, Any]) -> dict[str, Any]:
        return _unwrap(await self._post(rpc_request("message/send", {"message": message})))

    async def get_task(self, task_id: str) -> dict[str, Any]:
        return _unwrap(await self._post(rpc_request("tasks/get", {"id": task_id})))
