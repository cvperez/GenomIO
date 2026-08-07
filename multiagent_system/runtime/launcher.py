"""Process entry point shared by all three agents.

Ordering here is not incidental:

1. **asyncio owns the main thread.** The Claude Agent SDK binds an anyio task group and a
   subprocess transport to whichever loop created them. Building it on a worker and
   tearing it down elsewhere raises anyio's "cancel scope in a different task".
2. **Models load before the HTTP server reports ready.** ``/health`` only says
   ``models_loaded: true`` after the warm-up returns, so a caller cannot send work into
   a 32-second DNABERT-S load and watch it look like a hang.
3. **The HTTP server runs on a daemon thread** and hands requests back to the loop with
   ``run_coroutine_threadsafe``.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from typing import Any

from .. import config
from ..a2a.server import start_server
from .agent import AgentSpec, LLMAgent


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--host", default=config.BIND_HOST, help="bind address")
    parser.add_argument("--port", type=int, default=0, help="0 uses the role default")
    parser.add_argument(
        "--retrieval-url", default="", help="override the retrieval agent's URL"
    )
    parser.add_argument(
        "--reconstruction-url", default="", help="override the reconstruction agent's URL"
    )
    parser.add_argument("--model", default="", help="override the agent model")
    parser.add_argument("--threads", type=int, default=0, help="torch thread cap")
    return parser


def apply_common_args(args: argparse.Namespace, role: str) -> int:
    if args.retrieval_url:
        config.PEER_URLS[config.RETRIEVAL] = args.retrieval_url.rstrip("/")
    if args.reconstruction_url:
        config.PEER_URLS[config.RECONSTRUCTION] = args.reconstruction_url.rstrip("/")
    if args.model:
        config.AGENT_MODEL = args.model
    port = args.port or config.PORTS[role]
    config.PORTS[role] = port

    # The agent advertises where peers should reach it, which is not the bind address:
    # binding 0.0.0.0 is right, telling a peer to connect to 0.0.0.0 is not.
    if not os.environ.get(f"GENOMIO_URL_{role.upper()}"):
        import socket

        config.PEER_URLS[role] = f"http://{socket.gethostname()}:{port}"

    if args.threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        try:
            import torch

            torch.set_num_threads(args.threads)
        except ImportError:
            pass
    return port


async def serve(spec: AgentSpec, host: str, port: int) -> int:
    loop = asyncio.get_running_loop()
    agent = LLMAgent(spec)

    print(f"[{spec.role}] loading models...", flush=True)
    started = time.monotonic()
    warm = await asyncio.to_thread(agent.warm_up)
    print(
        f"[{spec.role}] ready in {time.monotonic() - started:.1f}s {warm}",
        flush=True,
    )

    agent.build_mcp_server()

    httpd = start_server(
        loop=loop,
        host=host,
        port=port,
        agent_card=agent.agent_card,
        health=agent.health,
        on_message_send=agent.handle_message,
        get_task=agent.get_task,
        cancel_task=agent.cancel_task,
        request_timeout=config.CLIENT_TIMEOUT_S,
    )

    print(
        f"[{spec.role}] serving on http://{host}:{port} "
        f"(peers: retrieval={config.PEER_URLS[config.RETRIEVAL]} "
        f"reconstruction={config.PEER_URLS[config.RECONSTRUCTION]})",
        flush=True,
    )
    print(f"[{spec.role}] tools: {', '.join(spec.allowed_tools)}", flush=True)

    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover
            pass

    try:
        await stop.wait()
    finally:
        print(f"[{spec.role}] shutting down", flush=True)
        httpd.shutdown()
    return 0


def run(spec_factory, role: str, argv: list[str] | None = None) -> int:
    """Parse arguments, build the spec and serve. Every agent's ``__main__`` calls this."""
    parser = add_common_args(
        argparse.ArgumentParser(description=f"GenomIO {role} agent (A2A + MCP)")
    )
    args = parser.parse_args(argv)
    port = apply_common_args(args, role)

    # Subscription auth via the claude CLI, not API credits. deploy.sh unsets this too;
    # doing it here as well means a stray key in a shell cannot silently change billing.
    if os.environ.pop("ANTHROPIC_API_KEY", None):
        print(
            f"[{role}] ANTHROPIC_API_KEY was set and has been ignored; using CLI "
            f"subscription auth",
            file=sys.stderr,
            flush=True,
        )

    try:
        spec = spec_factory()
    except Exception as exc:
        print(f"[{role}] FATAL: could not build agent spec: {exc}", file=sys.stderr)
        return 2

    try:
        return asyncio.run(serve(spec, args.host, port))
    except KeyboardInterrupt:
        return 0
