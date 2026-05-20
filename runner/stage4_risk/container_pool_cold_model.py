from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PoolContainer:
    next_free_ms: float
    expire_ms: float


class ContainerPoolColdModel:
    """Per-stage warm-container pool used by offline Monte Carlo risk simulation."""

    def __init__(self) -> None:
        self.containers: list[PoolContainer] = []

    def _evict_expired(self, now_ms: float) -> None:
        self.containers = [
            container for container in self.containers if container.expire_ms >= now_ms
        ]

    def ensure_warm_capacity(
        self,
        *,
        warm_count: float,
        window_start_ms: float,
        window_end_ms: float,
        keepalive_ms: float,
        now_ms: float,
    ) -> int:
        """Ensure the plan's desired warm capacity exists for this control window."""
        self._evict_expired(now_ms)
        target = max(0, int(math.ceil(float(warm_count))))
        added = 0
        expire_ms = max(float(window_end_ms), float(now_ms)) + max(0.0, float(keepalive_ms))
        while len(self.containers) < target:
            self.containers.append(
                PoolContainer(
                    next_free_ms=float(window_start_ms),
                    expire_ms=expire_ms,
                )
            )
            added += 1
        return added

    def reserve(self, ready_time_ms: float) -> tuple[int, bool]:
        """Reserve a container at ready_time_ms and return (index, cold_like)."""
        self._evict_expired(ready_time_ms)
        idle_indexes = [
            idx
            for idx, container in enumerate(self.containers)
            if container.next_free_ms <= ready_time_ms
        ]
        if idle_indexes:
            idx = min(
                idle_indexes,
                key=lambda item: (
                    self.containers[item].next_free_ms,
                    self.containers[item].expire_ms,
                    item,
                ),
            )
            return idx, False

        self.containers.append(
            PoolContainer(
                next_free_ms=float(ready_time_ms),
                expire_ms=float(ready_time_ms),
            )
        )
        return len(self.containers) - 1, True

    def complete(
        self,
        *,
        index: int,
        ready_time_ms: float,
        duration_ms: float,
        keepalive_ms: float,
    ) -> None:
        end_ms = float(ready_time_ms) + max(1.0, float(duration_ms))
        self.containers[index] = PoolContainer(
            next_free_ms=end_ms,
            expire_ms=end_ms + max(0.0, float(keepalive_ms)),
        )
