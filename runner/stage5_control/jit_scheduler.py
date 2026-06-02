"""Thread-safe priority-queue scheduler for path-3 JIT warmups."""

from __future__ import annotations

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class WarmupTask:
    task_key: str
    fire_time: float
    action_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(order=True)
class _ScheduledEntry:
    fire_time: float
    sequence: int
    task_key: str = field(compare=False)
    task: WarmupTask = field(compare=False)


class JitScheduler:
    """Single-threaded JIT warmup scheduler with upsert and cancellation."""

    def __init__(
        self,
        fire_callback: Callable[[WarmupTask], None],
        clock: Callable[[], float] = time.monotonic,
    ):
        self._fire_callback = fire_callback
        self._clock = clock
        self._condition = threading.Condition()
        self._heap: list[_ScheduledEntry] = []
        self._index: dict[str, _ScheduledEntry] = {}
        self._sequence = 0
        self._stop_requested = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background scheduler thread if it is not already running."""

        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_requested = False
            self._thread = threading.Thread(
                target=self._run,
                name="jit-warmup-scheduler",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the scheduler thread and wait for it to terminate."""

        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()

        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def schedule(self, task: WarmupTask) -> None:
        """Insert or update a warmup task keyed by ``task.task_key``."""

        if not task.task_key:
            raise ValueError("task_key must be non-empty")
        if not task.action_name:
            raise ValueError("action_name must be non-empty")

        with self._condition:
            self._sequence += 1
            entry = _ScheduledEntry(
                fire_time=float(task.fire_time),
                sequence=self._sequence,
                task_key=task.task_key,
                task=task,
            )
            self._index[task.task_key] = entry
            heapq.heappush(self._heap, entry)
            self._condition.notify_all()

    def cancel(self, task_key: str) -> None:
        """Cancel a queued task if present."""

        with self._condition:
            removed = self._index.pop(task_key, None)
            if removed is not None:
                self._condition.notify_all()

    def pending_count(self) -> int:
        """Return the number of currently queued, non-fired tasks."""

        with self._condition:
            return len(self._index)

    def _peek_valid_locked(self) -> _ScheduledEntry | None:
        while self._heap:
            entry = self._heap[0]
            if self._index.get(entry.task_key) is entry:
                return entry
            heapq.heappop(self._heap)
        return None

    def _pop_due_locked(self, now: float) -> _ScheduledEntry | None:
        while self._heap:
            entry = self._heap[0]
            if self._index.get(entry.task_key) is not entry:
                heapq.heappop(self._heap)
                continue
            if entry.fire_time > now:
                return None
            heapq.heappop(self._heap)
            self._index.pop(entry.task_key, None)
            return entry
        return None

    def _run(self) -> None:
        while True:
            task: WarmupTask | None = None
            with self._condition:
                while task is None:
                    if self._stop_requested:
                        return

                    now = self._clock()
                    due_entry = self._pop_due_locked(now)
                    if due_entry is not None:
                        task = due_entry.task
                        break

                    next_entry = self._peek_valid_locked()
                    if next_entry is None:
                        self._condition.wait()
                    else:
                        timeout = max(0.0, next_entry.fire_time - now)
                        self._condition.wait(timeout=timeout)

            try:
                self._fire_callback(task)
            except Exception:
                LOG.exception("JIT warmup callback failed for task_key=%s", task.task_key)
