"""Running blocking model work without stalling the agent's event loop.

Two problems this solves, both of which produce confusing symptoms rather than errors:

**The event loop must stay responsive.** The Claude Agent SDK talks to the ``claude`` CLI
over a subprocess stdio pump driven by the same loop the tool handlers run on. A 4-second
GENA-LM forward pass on that loop delays the pump; the 32-second first DNABERT-S load
looks like a hang and can drop the connection. So every torch/faiss call goes to a worker
thread.

**The models are not thread-safe.** ``src/rag/dnabert_s.py`` keeps its tokenizer and
model in module globals with no lock, and ``asyncio.to_thread`` uses a multi-worker
executor. One process-global lock serialises all heavy work, which costs nothing here
because these agents handle one task at a time anyway.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# One lock for every torch/faiss call in the process. Deliberately coarse.
MODEL_LOCK = threading.RLock()


def _locked(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    with MODEL_LOCK:
        return fn(*args, **kwargs)


async def run_blocking(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Await a blocking call on a worker thread, holding the model lock."""
    return await asyncio.to_thread(_locked, fn, *args, **kwargs)


def run_blocking_sync(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Same lock, for startup warm-up where there is no loop to yield to yet."""
    return _locked(fn, *args, **kwargs)
