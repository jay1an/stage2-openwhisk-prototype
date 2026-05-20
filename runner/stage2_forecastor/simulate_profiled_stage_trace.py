from __future__ import annotations

import argparse
import heapq
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .generate_synthetic_traces import (
    TRACE_COLUMNS,
    ContainerState,
    RequestState,
    add_entry_row,
    alloc_or_reuse_container,
    build_burnin_entry_times,
    build_entry_times,
    generate_request_ids,
    summarize_trace,
    write_split_traces,
)
from ..workflow import NodeSpec, WorkflowSpec, load_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate profiled synthetic stage traces from workflow CPU/memory profiles. "
            "Unlike the older SeBS trace simulator, action duration is derived from "
            "cpu_iters and memory_kb, not sleep_ms."
        )
    )
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--out-trace", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-split-dir", required=True)
    parser.add_argument("--out-metadata", required=True)
    parser.add_argument("--base-entry-ts-ms", type=int, default=1_900_000_000_000)
    parser.add_argument("--keepalive-ms", type=int, default=60_000)
    parser.add_argument("--burnin-copies", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-iters-per-ms", type=float, default=8_000.0)
    parser.add_argument("--memory-ops-per-ms", type=float, default=12_000.0)
    parser.add_argument("--action-sigma", type=float, default=0.10)
    parser.add_argument("--warm-overhead-sigma", type=float, default=0.18)
    parser.add_argument("--cold-overhead-sigma", type=float, default=0.28)
    parser.add_argument("--warm-slow-probability", type=float, default=0.07)
    parser.add_argument("--warm-slow-multiplier", type=float, default=7.0)
    parser.add_argument("--dispatch-jitter-ms-max", type=int, default=2)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def node_profile(node: NodeSpec) -> dict[str, float]:
    return {
        "cpu_iters": float(node.cpu_iters or 40_000),
        "memory_kb": float(node.memory_kb or 64),
        "memory_passes": float(node.memory_passes or 1),
        "memory_stride": float(node.memory_stride or 256),
        "warm_overhead_ms": float(node.warm_overhead_ms or 50.0),
        "cold_overhead_ms": float(node.cold_overhead_ms or 1500.0),
    }


def sample_action_duration_ms(
    node: NodeSpec,
    args: argparse.Namespace,
    rng: np.random.Generator,
    action_scale: float,
) -> float:
    profile = node_profile(node)
    cpu_ms = profile["cpu_iters"] / max(1.0, args.cpu_iters_per_ms)
    memory_stride = max(1.0, profile["memory_stride"])
    memory_ops = profile["memory_kb"] * 1024.0 / memory_stride * profile["memory_passes"]
    memory_ms = memory_ops / max(1.0, args.memory_ops_per_ms)
    base_ms = max(1.0, cpu_ms + memory_ms)
    noise = float(rng.lognormal(mean=0.0, sigma=args.action_sigma))
    return max(1.0, base_ms * action_scale * noise)


def sample_overhead_ms(
    node: NodeSpec,
    cold_like: bool,
    args: argparse.Namespace,
    rng: np.random.Generator,
    overhead_scale: float,
) -> tuple[float, str]:
    profile = node_profile(node)
    if cold_like:
        median = profile["cold_overhead_ms"]
        value = float(rng.lognormal(mean=math.log(max(1.0, median)), sigma=args.cold_overhead_sigma))
        return max(1.0, value * overhead_scale), "cold"

    median = profile["warm_overhead_ms"]
    if rng.random() < args.warm_slow_probability:
        median *= args.warm_slow_multiplier
        state = "warm_slow"
    else:
        state = "warm_fast"
    value = float(rng.lognormal(mean=math.log(max(1.0, median)), sigma=args.warm_overhead_sigma))
    return max(1.0, value * overhead_scale), state


def pre_overhead_fraction(overhead_state: str) -> float:
    if overhead_state == "cold":
        return 0.96
    if overhead_state == "warm_slow":
        return 0.88
    return 0.65


def simulate_workflow_trace(
    workflow: WorkflowSpec,
    schedule: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    request_states: dict[str, RequestState] = {}
    event_heap: list[tuple[int, int, str, str]] = []
    pools: dict[str, list[ContainerState]] = {}
    rows: list[dict] = []
    counter = 0

    action_scales = {
        node.action: float(rng.lognormal(mean=0.0, sigma=0.04))
        for node in workflow.nodes.values()
    }
    overhead_scales = {
        node.action: float(rng.lognormal(mean=0.0, sigma=0.06))
        for node in workflow.nodes.values()
    }

    def seed_request(entry_ts_ms: int, request_id: str, save_output: bool) -> None:
        nonlocal counter
        request_states[request_id] = RequestState(
            workflow=workflow,
            request_id=request_id,
            entry_ts_ms=int(entry_ts_ms),
            save_output=save_output,
            completed_end_ms={},
        )
        if save_output:
            rows.append(add_entry_row(workflow.workflow_name, request_id, int(entry_ts_ms)))
        for node in workflow.ready_nodes(completed=[], running=[]):
            heapq.heappush(event_heap, (int(entry_ts_ms), counter, request_id, node.name))
            counter += 1

    for burnin_times in build_burnin_entry_times(
        schedule,
        args.base_entry_ts_ms,
        args.burnin_copies,
    ):
        for entry_ts_ms in burnin_times:
            seed_request(int(entry_ts_ms), f"warmup-{uuid.uuid4()}", False)

    request_ids = generate_request_ids(len(schedule))
    for entry_ts_ms, request_id in zip(
        build_entry_times(schedule, args.base_entry_ts_ms),
        request_ids,
    ):
        seed_request(int(entry_ts_ms), request_id, True)

    while event_heap:
        ready_time_ms, _, request_id, node_name = heapq.heappop(event_heap)
        state = request_states[request_id]
        node = state.workflow.nodes[node_name]

        dispatch_start_ms = int(
            math.ceil(ready_time_ms + rng.integers(0, args.dispatch_jitter_ms_max + 1))
        )
        container, cold_like = alloc_or_reuse_container(
            pools,
            node.action,
            dispatch_start_ms,
            args.keepalive_ms,
        )
        action_duration_ms = sample_action_duration_ms(
            node,
            args,
            rng,
            action_scales[node.action],
        )
        overhead_ms, overhead_state = sample_overhead_ms(
            node,
            cold_like,
            args,
            rng,
            overhead_scales[node.action],
        )
        pre_fraction = pre_overhead_fraction(overhead_state)
        raw_dispatch_end_ms = dispatch_start_ms + overhead_ms + action_duration_ms
        dispatch_end_ms = int(math.ceil(raw_dispatch_end_ms))
        dispatch_latency_ms = dispatch_end_ms - dispatch_start_ms
        platform_overhead_ms = dispatch_latency_ms - action_duration_ms
        action_start_ns = int(round((dispatch_start_ms + overhead_ms * pre_fraction) * 1_000_000))
        action_end_ns = int(round(action_start_ns + action_duration_ms * 1_000_000))

        container.next_free_ms = dispatch_end_ms
        container.expire_ms = dispatch_end_ms + args.keepalive_ms
        state.completed_end_ms[node.name] = dispatch_end_ms

        if state.save_output:
            rows.append(
                {
                    "workflow_name": workflow.workflow_name,
                    "request_id": request_id,
                    "stage_name": node.name,
                    "parent_stages": ",".join(node.parents),
                    "entry_ts_ms": state.entry_ts_ms,
                    "dispatch_start_ms": dispatch_start_ms,
                    "dispatch_end_ms": dispatch_end_ms,
                    "dispatch_latency_ms": dispatch_latency_ms,
                    "action_start_ns": action_start_ns,
                    "action_end_ns": action_end_ns,
                    "action_duration_ms": round(float(action_duration_ms), 6),
                    "platform_overhead_ms": round(float(platform_overhead_ms), 6),
                    "container_id": container.container_id,
                    "cold_like": bool(cold_like),
                    "status": "ok",
                    "error": "",
                }
            )

        completed = state.completed_end_ms.keys()
        for child in workflow.ready_nodes(completed=completed, running=[]):
            if child.name in state.completed_end_ms:
                continue
            if any(item[2] == request_id and item[3] == child.name for item in event_heap):
                continue
            child_ready_ms = (
                max(state.completed_end_ms[parent] for parent in child.parents)
                if child.parents
                else state.entry_ts_ms
            )
            heapq.heappush(event_heap, (int(child_ready_ms), counter, request_id, child.name))
            counter += 1

    out = pd.DataFrame(rows, columns=TRACE_COLUMNS)
    return out.sort_values(["entry_ts_ms", "request_id", "dispatch_start_ms", "stage_name"]).reset_index(drop=True)


def workflow_latency_summary(trace: pd.DataFrame) -> dict:
    stage_rows = trace[trace["stage_name"] != "__entry__"].copy()
    if stage_rows.empty:
        return {}
    grouped = stage_rows.groupby("request_id").agg(
        entry_ts_ms=("entry_ts_ms", "first"),
        workflow_end_ms=("dispatch_end_ms", "max"),
    )
    latencies = grouped["workflow_end_ms"] - grouped["entry_ts_ms"]
    return {
        "requests": int(len(latencies)),
        "workflow_latency_p50_ms": float(latencies.quantile(0.50)),
        "workflow_latency_p90_ms": float(latencies.quantile(0.90)),
        "workflow_latency_p95_ms": float(latencies.quantile(0.95)),
        "workflow_latency_mean_ms": float(latencies.mean()),
        "cold_like_stage_rate": float(stage_rows["cold_like"].astype(bool).mean()),
    }


def main() -> None:
    args = parse_args()
    root = project_root()
    workflow = load_workflow(str(root / args.workflow_config))
    schedule = pd.read_csv(root / args.schedule).sort_values("index").reset_index(drop=True)
    rng = np.random.default_rng(args.seed)

    trace = simulate_workflow_trace(workflow, schedule, args, rng)

    out_trace = root / args.out_trace
    out_summary = root / args.out_summary
    out_split_dir = root / args.out_split_dir
    out_metadata = root / args.out_metadata
    out_trace.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_split_dir.mkdir(parents=True, exist_ok=True)
    out_metadata.parent.mkdir(parents=True, exist_ok=True)

    trace.to_csv(out_trace, index=False)
    summarize_trace(trace).to_csv(out_summary, index=False)
    split_info = write_split_traces(
        trace=trace,
        out_dir=out_split_dir,
        file_prefix=out_trace.stem,
        train_ratio=args.train_ratio,
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_name": workflow.workflow_name,
        "workflow_config": str(root / args.workflow_config),
        "schedule": str(root / args.schedule),
        "out_trace": str(out_trace),
        "out_summary": str(out_summary),
        "split_info": split_info,
        "base_entry_ts_ms": args.base_entry_ts_ms,
        "keepalive_ms": args.keepalive_ms,
        "burnin_copies": args.burnin_copies,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "rows": int(len(trace)),
        "entry_requests": int((trace["stage_name"] == "__entry__").sum()),
        "stage_rows": int((trace["stage_name"] != "__entry__").sum()),
        **workflow_latency_summary(trace),
        "notes": [
            "Action durations are generated from cpu_iters and memory_kb profiles, not sleep_ms.",
            "Platform overhead and cold-like paths are synthetic but stage-profiled.",
            "This is method-development data; replace with real OpenWhisk replay for final evidence.",
        ],
    }
    out_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote {out_trace}")
    print(f"wrote {out_summary}")
    print(f"wrote {out_metadata}")
    print(pd.DataFrame([metadata]).drop(columns=["notes", "split_info"]).to_string(index=False))
    print("split_info:")
    print(json.dumps(split_info, indent=2))


if __name__ == "__main__":
    main()
