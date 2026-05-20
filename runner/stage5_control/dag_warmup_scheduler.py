from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from ..workflow import WorkflowSpec


@dataclass(frozen=True)
class WarmupTiming:
    warmup_start_ms: float
    ready_ms: float
    lead_time_ms: float


def cold_overhead_lead_times_ms(
    latency_samples: pd.DataFrame,
    workflow_name: str | None = None,
    *,
    default_ms: float = 1000.0,
) -> dict[str, float]:
    """Estimate per-stage JIT warmup lead time from cold-like platform overhead."""
    if latency_samples is None or latency_samples.empty:
        return {"*": float(default_ms)}

    rows = latency_samples.copy()
    if workflow_name is not None and "workflow_name" in rows.columns:
        rows = rows[rows["workflow_name"].astype(str) == str(workflow_name)].copy()
    if rows.empty:
        return {"*": float(default_ms)}

    rows["platform_overhead_ms"] = pd.to_numeric(
        rows.get("platform_overhead_ms"), errors="coerce"
    )
    rows = rows.dropna(subset=["platform_overhead_ms"])
    if rows.empty:
        return {"*": float(default_ms)}

    if "latency_class" in rows.columns:
        cold_rows = rows[
            rows["latency_class"].astype(str).str.startswith("cold_like")
        ].copy()
    else:
        cold_rows = pd.DataFrame()
    if cold_rows.empty:
        cold_rows = rows

    lead_times = (
        cold_rows.groupby("stage_name")["platform_overhead_ms"]
        .median()
        .clip(lower=1.0)
        .to_dict()
    )
    global_lead = float(cold_rows["platform_overhead_ms"].median())
    lead_times["*"] = max(1.0, global_lead)
    return {str(stage): float(value) for stage, value in lead_times.items()}


def stage_lead_time_ms(stage_name: str, lead_times_ms: Mapping[str, float] | None) -> float:
    if not lead_times_ms:
        return 1000.0
    return max(0.0, float(lead_times_ms.get(stage_name, lead_times_ms.get("*", 1000.0))))


def warmup_timing_for_stage(
    *,
    warmup_mode: str,
    is_root: bool,
    ready_abs_ms: float,
    window_start_ms: float,
    lead_time_ms: float,
) -> WarmupTiming:
    """Return when a planned warm container becomes usable for a stage."""
    if warmup_mode == "window" or is_root:
        return WarmupTiming(
            warmup_start_ms=float(window_start_ms),
            ready_ms=float(window_start_ms),
            lead_time_ms=0.0,
        )

    if warmup_mode != "dag_jit":
        raise ValueError(f"unknown warmup_mode={warmup_mode}")

    lead = max(0.0, float(lead_time_ms))
    needed_start = float(ready_abs_ms) - lead
    if needed_start < float(window_start_ms):
        # The parent path is too short to hide this stage's cold overhead after
        # the workflow starts. Treat the planned warm capacity as ready at the
        # control-window boundary; the controller would issue that warmup just
        # before the window begins.
        return WarmupTiming(
            warmup_start_ms=float(window_start_ms) - lead,
            ready_ms=float(window_start_ms),
            lead_time_ms=lead,
        )

    warmup_start = needed_start
    return WarmupTiming(
        warmup_start_ms=warmup_start,
        ready_ms=warmup_start + lead,
        lead_time_ms=lead,
    )


def warm_interval_start_sec(
    *,
    warmup_mode: str,
    workflow: WorkflowSpec | None,
    stage_name: str,
    window_start_sec: float,
    window_end_sec: float,
    lead_time_sec: float,
) -> float:
    """Coarse cost-model start time for planned warm capacity."""
    if warmup_mode == "window" or workflow is None:
        return float(window_start_sec)
    if warmup_mode != "dag_jit":
        raise ValueError(f"unknown warmup_mode={warmup_mode}")
    node = workflow.nodes.get(stage_name)
    if node is None or not node.parents:
        return float(window_start_sec)
    return max(float(window_start_sec), float(window_end_sec) - max(0.0, float(lead_time_sec)))
