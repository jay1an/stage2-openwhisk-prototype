"""Shared utility helpers for profile-driven workflow trace simulation.

The original module contained a calibration pipeline that read deleted
SEBS pilot traces and a CLI that generated traces for deleted SEBS
workflow YAMLs. Those entry points are gone; only the workflow-agnostic
data structures and helper functions remain, since they are imported by
`simulate_profiled_stage_trace.py`.
"""

from __future__ import annotations

import heapq
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..workflow import WorkflowSpec


TRACE_COLUMNS = [
    "workflow_name",
    "request_id",
    "stage_name",
    "parent_stages",
    "entry_ts_ms",
    "dispatch_start_ms",
    "dispatch_end_ms",
    "dispatch_latency_ms",
    "action_start_ns",
    "action_end_ns",
    "action_duration_ms",
    "platform_overhead_ms",
    "container_id",
    "cold_like",
    "status",
    "error",
]


@dataclass
class ContainerState:
    container_id: str
    next_free_ms: int
    expire_ms: int


@dataclass
class RequestState:
    workflow: WorkflowSpec
    request_id: str
    entry_ts_ms: int
    save_output: bool
    completed_end_ms: Dict[str, int]


def alloc_or_reuse_container(
    pools: Dict[str, List[ContainerState]],
    action_name: str,
    ready_time_ms: int,
    keepalive_ms: int,
) -> tuple[ContainerState, bool]:
    current = pools.setdefault(action_name, [])
    retained = [c for c in current if c.expire_ms >= ready_time_ms]
    pools[action_name] = retained
    idle = [c for c in retained if c.next_free_ms <= ready_time_ms]
    if idle:
        chosen = min(idle, key=lambda c: (c.next_free_ms, c.container_id))
        return chosen, False

    chosen = ContainerState(
        container_id=str(uuid.uuid4()),
        next_free_ms=ready_time_ms,
        expire_ms=ready_time_ms + keepalive_ms,
    )
    retained.append(chosen)
    return chosen, True


def add_entry_row(
    workflow_name: str,
    request_id: str,
    entry_ts_ms: int,
) -> dict:
    return {
        "workflow_name": workflow_name,
        "request_id": request_id,
        "stage_name": "__entry__",
        "parent_stages": "",
        "entry_ts_ms": entry_ts_ms,
        "dispatch_start_ms": entry_ts_ms,
        "dispatch_end_ms": entry_ts_ms,
        "dispatch_latency_ms": 0,
        "action_start_ns": "",
        "action_end_ns": "",
        "action_duration_ms": "",
        "platform_overhead_ms": "",
        "container_id": "",
        "cold_like": "",
        "status": "ok",
        "error": "",
    }


def generate_request_ids(count: int) -> list[str]:
    return [str(uuid.uuid4()) for _ in range(count)]


def build_entry_times(schedule: pd.DataFrame, base_entry_ts_ms: int) -> np.ndarray:
    offsets = schedule["target_offset_ms"].astype(int).to_numpy()
    return base_entry_ts_ms + offsets


def build_burnin_entry_times(
    schedule: pd.DataFrame,
    base_entry_ts_ms: int,
    copies: int,
) -> list[np.ndarray]:
    if copies <= 0:
        return []
    offsets = schedule["target_offset_ms"].astype(int).to_numpy()
    if len(offsets) <= 1:
        gap = 1000
    else:
        positive_gaps = np.diff(offsets)
        positive_gaps = positive_gaps[positive_gaps > 0]
        gap = int(np.median(positive_gaps)) if len(positive_gaps) else 1000
    span = int(offsets[-1]) + gap
    burnin = []
    for repeat in range(copies, 0, -1):
        burnin.append(base_entry_ts_ms + offsets - repeat * span)
    return burnin


def summarize_trace(trace: pd.DataFrame) -> pd.DataFrame:
    stages = trace[trace["stage_name"] != "__entry__"].copy()
    summary = (
        stages.groupby(["workflow_name", "stage_name", "cold_like"], dropna=False)[
            ["dispatch_latency_ms", "platform_overhead_ms", "action_duration_ms"]
        ]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_")
        for col in summary.columns.to_flat_index()
    ]
    return summary


def write_split_traces(
    trace: pd.DataFrame,
    out_dir: Path,
    file_prefix: str,
    train_ratio: float,
) -> dict:
    request_order = (
        trace[trace["stage_name"] == "__entry__"][["request_id", "entry_ts_ms"]]
        .sort_values(["entry_ts_ms", "request_id"])
        .reset_index(drop=True)
    )
    split_idx = int(math.floor(len(request_order) * train_ratio))
    train_ids = set(request_order.head(split_idx)["request_id"])
    test_ids = set(request_order.tail(len(request_order) - split_idx)["request_id"])

    train_trace = trace[trace["request_id"].isin(train_ids)].copy()
    test_trace = trace[trace["request_id"].isin(test_ids)].copy()
    split_map = request_order.copy()
    split_map["split"] = np.where(split_map["request_id"].isin(train_ids), "train", "test")

    train_path = out_dir / f"{file_prefix}_train.csv"
    test_path = out_dir / f"{file_prefix}_test.csv"
    split_path = out_dir / f"{file_prefix}_split.csv"
    train_trace.to_csv(train_path, index=False)
    test_trace.to_csv(test_path, index=False)
    split_map.to_csv(split_path, index=False)

    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "split_path": str(split_path),
        "train_requests": int(len(train_ids)),
        "test_requests": int(len(test_ids)),
    }
