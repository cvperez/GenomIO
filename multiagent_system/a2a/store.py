"""In-memory A2A task store.

Bounded and thread-safe. The HTTP server runs on daemon threads while the agent logic
runs on the asyncio loop, so every mutation goes through one lock.

Two rules from the specification are enforced here rather than left to callers:

  * terminal tasks are immutable -- a completed/failed/canceled/rejected task cannot be
    moved back to working. Refinement means a *new* task carrying the same contextId.
  * contextId groups tasks. The Coordinator mints one per gap and threads it down to
    Reconstruction and Retrieval, so all of a run's tasks across three processes join on
    a single key afterwards.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from .types import TaskState, build_status, task_state


class TerminalTaskError(RuntimeError):
    """Raised when something tries to mutate a task that has already finished."""


class TaskStore:
    def __init__(self, max_tasks: int = 256) -> None:
        self._tasks: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._contexts: dict[str, list[str]] = {}
        self._max_tasks = max_tasks
        self._lock = threading.RLock()

    def create(self, task: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            task_id = task["id"]
            self._tasks[task_id] = task
            self._tasks.move_to_end(task_id)
            self._contexts.setdefault(task["contextId"], []).append(task_id)
            self._evict()
            return task

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                self._tasks.move_to_end(task_id)
            return task

    def set_state(
        self,
        task_id: str,
        state: TaskState,
        message: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task_state(task).is_terminal:
                raise TerminalTaskError(
                    f"task {task_id} is {task_state(task).value}; create a new task "
                    f"with the same contextId instead"
                )
            task["status"] = build_status(state, message)
            if message:
                task.setdefault("history", []).append(message)
            return task

    def add_artifact(self, task_id: str, artifact: dict[str, Any]):
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.setdefault("artifacts", []).append(artifact)
            return task

    def append_history(self, task_id: str, message: dict[str, Any]):
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.setdefault("history", []).append(message)
            return task

    def by_context(self, context_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._tasks[t] for t in self._contexts.get(context_id, []) if t in self._tasks
            ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)

    def _evict(self) -> None:
        while len(self._tasks) > self._max_tasks:
            old_id, old_task = self._tasks.popitem(last=False)
            siblings = self._contexts.get(old_task["contextId"])
            if siblings and old_id in siblings:
                siblings.remove(old_id)
                if not siblings:
                    self._contexts.pop(old_task["contextId"], None)
