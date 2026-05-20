import math
import random
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class WorkloadEvent:
    index: int
    sleep_before_ms: int
    pattern: str


def generate_workload_events(
    pattern: str,
    count: int,
    base_interval_ms: int,
    seed: int,
    burst_every: int = 20,
    burst_size: int = 5,
    burst_interval_ms: int = 50,
    idle_interval_ms: int = 3000,
    period_steps: int = 30,
    amplitude: float = 0.6,
    sparse_probability: float = 0.25,
) -> List[WorkloadEvent]:
    rng = random.Random(seed)
    events: List[WorkloadEvent] = []

    for idx in range(count):
        if idx == 0:
            interval = 0
        elif pattern == "constant":
            interval = base_interval_ms
        elif pattern == "burst":
            position = idx % max(1, burst_every)
            interval = burst_interval_ms if 0 < position <= burst_size else idle_interval_ms
        elif pattern == "periodic":
            phase = 2.0 * math.pi * (idx % max(1, period_steps)) / max(1, period_steps)
            factor = 1.0 - amplitude * math.sin(phase)
            interval = max(20, int(base_interval_ms * factor))
        elif pattern == "sparse":
            interval = idle_interval_ms if rng.random() < sparse_probability else base_interval_ms
        elif pattern == "poisson":
            interval = max(1, int(rng.expovariate(1.0 / max(1, base_interval_ms))))
        else:
            raise ValueError(f"unsupported workload pattern: {pattern}")

        events.append(WorkloadEvent(index=idx, sleep_before_ms=int(interval), pattern=pattern))

    return events

