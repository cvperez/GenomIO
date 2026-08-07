"""Starting, finding and health-checking the three agents.

Shared by ``test.py`` and ``deploy.sh``. Local mode launches three subprocesses on one
host; SLURM mode reads endpoints that ``deploy.sh`` already placed. The two differ only
in how ``endpoints`` gets populated -- everything downstream is identical, which is how
one test body covers both.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from typing import Any

from . import config
from .a2a.client import SyncA2AClient

MODULES = {
    config.COORDINATOR: "multiagent_system.agents.coordinator",
    config.RETRIEVAL: "multiagent_system.agents.retrieval",
    config.RECONSTRUCTION: "multiagent_system.agents.reconstruction",
}


def python_executable() -> str:
    """The venv interpreter, which is the only one with claude_agent_sdk."""
    if os.path.exists(config.VENV_PYTHON):
        return config.VENV_PYTHON
    return sys.executable


def endpoints_file() -> str:
    return os.path.join(config.RUN_DIR, "endpoints.json")


def load_endpoints(path: str | None = None) -> dict[str, str]:
    with open(path or endpoints_file(), "r") as handle:
        data = json.load(handle)
    return {role: str(url).rstrip("/") for role, url in data.items() if role in MODULES}


def save_endpoints(endpoints: dict[str, str], path: str | None = None) -> str:
    target = path or endpoints_file()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as handle:
        json.dump(endpoints, handle, indent=2)
    return target


def parse_endpoints(spec: str) -> dict[str, str]:
    """``coordinator=http://h:8100,retrieval=...`` into a dict."""
    endpoints: dict[str, str] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        role, _, url = item.partition("=")
        role = role.strip()
        if role not in MODULES:
            raise ValueError(f"unknown role {role!r}; expected one of {sorted(MODULES)}")
        endpoints[role] = url.strip().rstrip("/")
    return endpoints


class LocalFleet:
    """Three agent processes on this host, on consecutive ports."""

    def __init__(self, base_port: int | None = None, threads: int = 8) -> None:
        base = base_port or config.BASE_PORT
        self.ports = {
            config.COORDINATOR: base,
            config.RETRIEVAL: base + 1,
            config.RECONSTRUCTION: base + 2,
        }
        self.endpoints = {
            role: f"http://127.0.0.1:{port}" for role, port in self.ports.items()
        }
        self.threads = threads
        self.processes: dict[str, subprocess.Popen] = {}
        self.log_paths: dict[str, str] = {}

    def start(self) -> dict[str, str]:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        env = os.environ.copy()
        # Subscription auth through the claude CLI, not API credits.
        env.pop("ANTHROPIC_API_KEY", None)
        env["OMP_NUM_THREADS"] = str(self.threads)
        env["PYTHONUNBUFFERED"] = "1"
        for role, url in self.endpoints.items():
            env[f"GENOMIO_URL_{role.upper()}"] = url

        for role, module in MODULES.items():
            log_path = os.path.join(config.LOG_DIR, f"agent-{role}.log")
            self.log_paths[role] = log_path
            command = [
                python_executable(),
                "-m",
                module,
                "--port",
                str(self.ports[role]),
                "--host",
                "127.0.0.1",
                "--threads",
                str(self.threads),
                "--retrieval-url",
                self.endpoints[config.RETRIEVAL],
                "--reconstruction-url",
                self.endpoints[config.RECONSTRUCTION],
            ]
            handle = open(log_path, "w")
            self.processes[role] = subprocess.Popen(
                command,
                cwd=config.REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # so a stray signal cannot take the fleet down
            )
        return self.endpoints

    def stop(self, timeout: float = 10.0) -> None:
        for role, process in self.processes.items():
            if process.poll() is not None:
                continue
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
        deadline = time.monotonic() + timeout
        for process in self.processes.values():
            remaining = max(0.1, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()

    def dead(self) -> list[str]:
        return [role for role, p in self.processes.items() if p.poll() is not None]

    def tail(self, role: str, lines: int = 40) -> str:
        path = self.log_paths.get(role)
        if not path or not os.path.exists(path):
            return "(no log)"
        with open(path, "r", errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])


def wait_until_ready(
    endpoints: dict[str, str],
    timeout: float = 300.0,
    poll: float = 3.0,
    on_tick=None,
    fleet: "LocalFleet | None" = None,
) -> tuple[bool, dict[str, Any]]:
    """Poll every agent's /health until all report ``models_loaded``.

    This gate matters: the retrieval agent spends about 32 seconds loading DNABERT-S, and
    work sent before that lands in a request that looks like a hang.
    """
    deadline = time.monotonic() + timeout
    status: dict[str, Any] = {role: None for role in endpoints}
    while time.monotonic() < deadline:
        if fleet is not None:
            gone = fleet.dead()
            if gone:
                return False, {"exited": gone, **status}
        for role, url in endpoints.items():
            if status.get(role):
                continue
            try:
                health = SyncA2AClient(url, timeout=5.0).health(timeout=5.0)
                if health.get("models_loaded"):
                    status[role] = health
            except Exception:
                pass
        if all(status.values()):
            return True, status
        if on_tick:
            on_tick({r: bool(v) for r, v in status.items()})
        time.sleep(poll)
    return False, status
