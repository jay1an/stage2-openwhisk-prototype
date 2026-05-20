import argparse
import heapq
import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..workflow import WorkflowSpec, load_workflow


DEFAULT_DATASETS = [
    {
        "workflow_config": "configs/sebs_trip_booking.yaml",
        "schedule": "data/azure_schedules/schedule_sparse_1_sebs_trip_booking.csv",
    },
    {
        "workflow_config": "configs/sebs_video.yaml",
        "schedule": "data/azure_schedules/schedule_bursty_1_sebs_video.csv",
    },
    {
        "workflow_config": "configs/sebs_map_reduce.yaml",
        "schedule": "data/azure_schedules/schedule_periodic_0_sebs_map_reduce.csv",
    },
    {
        "workflow_config": "configs/sebs_ml.yaml",
        "schedule": "data/azure_schedules/schedule_mixed_0_sebs_ml.csv",
    },
]

DEFAULT_CALIBRATION_TRACES = [
    "data/real_traces/traces_sparse1_sebs_trip_booking_probe5.csv",
    "data/real_traces/traces_sparse1_sebs_trip_booking_l20.csv",
    "data/real_traces/traces_sparse1_sebs_trip_booking_l20_mi4.csv",
]

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
class Calibration:
    action_intercept_ms: float
    action_sleep_slope: float
    action_residual_log_pool: np.ndarray
    cold_overhead_pool: np.ndarray
    warm_fast_overhead_pool: np.ndarray
    warm_slow_overhead_pool: np.ndarray
    warm_slow_probability: float
    fast_warm_threshold_ms: float
    keepalive_ms: int
    duration_action_sigma: float
    overhead_action_sigma: float
    dispatch_jitter_ms_max: int


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate calibrated synthetic workflow traces from Azure entry schedules."
    )
    parser.add_argument(
        "--output-root",
        default="data/synthetic",
        help="directory where synthetic traces, manifests, and metadata are written",
    )
    parser.add_argument(
        "--base-entry-ts-ms",
        type=int,
        default=1_800_000_000_000,
        help="base timestamp used to anchor generated schedules",
    )
    parser.add_argument(
        "--keepalive-ms",
        type=int,
        default=60_000,
        help="synthetic keep-alive timeout for warm container reuse",
    )
    parser.add_argument(
        "--burnin-copies",
        type=int,
        default=1,
        help="number of hidden schedule copies replayed before the saved window",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="fraction of requests assigned to train split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducible synthetic generation",
    )
    return parser.parse_args()


def load_trip_booking_sleep_map(root: Path) -> dict[str, int]:
    workflow = load_workflow(str(root / "configs" / "sebs_trip_booking.yaml"))
    return {node.name: node.sleep_ms for node in workflow.nodes.values()}


def build_calibration(root: Path, keepalive_ms: int) -> Calibration:
    sleep_map = load_trip_booking_sleep_map(root)
    frames = []
    for rel in DEFAULT_CALIBRATION_TRACES:
        df = pd.read_csv(root / rel)
        df = df[df["stage_name"] != "__entry__"].copy()
        df["sleep_ms"] = df["stage_name"].map(sleep_map)
        frames.append(df)
    rows = pd.concat(frames, ignore_index=True)

    valid = rows.dropna(subset=["sleep_ms", "action_duration_ms", "platform_overhead_ms"]).copy()
    x = np.vstack([np.ones(len(valid)), valid["sleep_ms"].astype(float)]).T
    y = valid["action_duration_ms"].astype(float).to_numpy()
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    pred = np.maximum(1.0, coef[0] + coef[1] * valid["sleep_ms"].astype(float).to_numpy())
    residual_log_pool = np.log(np.maximum(1e-6, y / pred))

    warm = valid[valid["cold_like"] == False]["platform_overhead_ms"].astype(float)
    cold = valid[valid["cold_like"] == True]["platform_overhead_ms"].astype(float)
    fast_warm_threshold_ms = 200.0
    warm_fast = warm[warm <= fast_warm_threshold_ms]
    warm_slow = warm[warm > fast_warm_threshold_ms]

    if warm_fast.empty:
        raise ValueError("warm-fast overhead pool is empty; cannot build calibration")
    if warm_slow.empty:
        raise ValueError("warm-slow overhead pool is empty; cannot build calibration")
    if cold.empty:
        raise ValueError("cold overhead pool is empty; cannot build calibration")

    return Calibration(
        action_intercept_ms=float(coef[0]),
        action_sleep_slope=float(coef[1]),
        action_residual_log_pool=np.asarray(residual_log_pool, dtype=float),
        cold_overhead_pool=cold.to_numpy(dtype=float),
        warm_fast_overhead_pool=warm_fast.to_numpy(dtype=float),
        warm_slow_overhead_pool=warm_slow.to_numpy(dtype=float),
        warm_slow_probability=float(len(warm_slow) / max(1, len(warm))),
        fast_warm_threshold_ms=fast_warm_threshold_ms,
        keepalive_ms=int(keepalive_ms),
        duration_action_sigma=0.06,
        overhead_action_sigma=0.08,
        dispatch_jitter_ms_max=2,
    )


def sample_action_duration_ms(
    calibration: Calibration,
    sleep_ms: int,
    action_duration_scale: float,
    rng: np.random.Generator,
) -> float:
    base = calibration.action_intercept_ms + calibration.action_sleep_slope * float(sleep_ms)
    residual_log = float(rng.choice(calibration.action_residual_log_pool))
    return max(1.0, base * action_duration_scale * math.exp(residual_log))


def sample_overhead_ms(
    calibration: Calibration,
    cold_like: bool,
    overhead_scale: float,
    rng: np.random.Generator,
) -> tuple[float, str]:
    if cold_like:
        sampled = float(rng.choice(calibration.cold_overhead_pool))
        return max(1.0, sampled * overhead_scale), "cold"

    if rng.random() < calibration.warm_slow_probability:
        sampled = float(rng.choice(calibration.warm_slow_overhead_pool))
        return max(1.0, sampled * overhead_scale), "warm_slow"

    sampled = float(rng.choice(calibration.warm_fast_overhead_pool))
    return max(1.0, sampled * overhead_scale), "warm_fast"


def pre_overhead_fraction(overhead_state: str) -> float:
    if overhead_state == "cold":
        return 0.97
    if overhead_state == "warm_slow":
        return 0.92
    return 0.70


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


def simulate_workflow_trace(
    workflow: WorkflowSpec,
    schedule: pd.DataFrame,
    calibration: Calibration,
    rng: np.random.Generator,
    base_entry_ts_ms: int,
    burnin_copies: int,
) -> pd.DataFrame:
    request_states: dict[str, RequestState] = {}
    event_heap: list[tuple[int, int, str, str]] = []
    pools: Dict[str, List[ContainerState]] = {}
    rows: list[dict] = []
    counter = 0

    duration_scale_by_action: dict[str, float] = {}
    overhead_scale_by_action: dict[str, float] = {}
    for node in workflow.nodes.values():
        duration_scale_by_action.setdefault(
            node.action,
            float(rng.lognormal(mean=0.0, sigma=calibration.duration_action_sigma)),
        )
        overhead_scale_by_action.setdefault(
            node.action,
            float(rng.lognormal(mean=0.0, sigma=calibration.overhead_action_sigma)),
        )

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
        ready = workflow.ready_nodes(completed=[], running=[])
        for node in ready:
            heapq.heappush(event_heap, (int(entry_ts_ms), counter, request_id, node.name))
            counter += 1

    for burnin_times in build_burnin_entry_times(schedule, base_entry_ts_ms, burnin_copies):
        for entry_ts_ms in burnin_times:
            seed_request(int(entry_ts_ms), f"warmup-{uuid.uuid4()}", False)

    request_ids = generate_request_ids(len(schedule))
    for entry_ts_ms, request_id in zip(build_entry_times(schedule, base_entry_ts_ms), request_ids):
        seed_request(int(entry_ts_ms), request_id, True)

    while event_heap:
        ready_time_ms, _, request_id, node_name = heapq.heappop(event_heap)
        state = request_states[request_id]
        node = state.workflow.nodes[node_name]

        dispatch_start_ms = int(math.ceil(ready_time_ms + rng.integers(0, calibration.dispatch_jitter_ms_max + 1)))
        container, cold_like = alloc_or_reuse_container(
            pools,
            node.action,
            dispatch_start_ms,
            calibration.keepalive_ms,
        )
        action_duration_ms = sample_action_duration_ms(
            calibration,
            node.sleep_ms,
            duration_scale_by_action[node.action],
            rng,
        )
        overhead_ms, overhead_state = sample_overhead_ms(
            calibration,
            cold_like,
            overhead_scale_by_action[node.action],
            rng,
        )
        pre_fraction = pre_overhead_fraction(overhead_state)
        pre_overhead_ms = overhead_ms * pre_fraction
        raw_dispatch_end_ms = dispatch_start_ms + overhead_ms + action_duration_ms
        dispatch_end_ms = int(math.ceil(raw_dispatch_end_ms))
        dispatch_latency_ms = dispatch_end_ms - dispatch_start_ms
        platform_overhead_ms = dispatch_latency_ms - action_duration_ms
        action_start_ns = int(round((dispatch_start_ms + pre_overhead_ms) * 1_000_000))
        action_end_ns = int(round(action_start_ns + action_duration_ms * 1_000_000))

        container.next_free_ms = dispatch_end_ms
        container.expire_ms = dispatch_end_ms + calibration.keepalive_ms
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
        ready = workflow.ready_nodes(completed=completed, running=[])
        for child in ready:
            if child.name in state.completed_end_ms:
                continue
            if any(item[2] == request_id and item[3] == child.name for item in event_heap):
                continue
            child_ready_ms = max(state.completed_end_ms[parent] for parent in child.parents) if child.parents else state.entry_ts_ms
            heapq.heappush(event_heap, (int(child_ready_ms), counter, request_id, child.name))
            counter += 1

    out = pd.DataFrame(rows, columns=TRACE_COLUMNS)
    return out.sort_values(["entry_ts_ms", "request_id", "dispatch_start_ms", "stage_name"]).reset_index(drop=True)


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


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent.parent
    out_root = root / args.output_root
    traces_dir = out_root / "traces"
    summaries_dir = out_root / "summaries"
    splits_dir = out_root / "splits"
    metadata_dir = out_root / "metadata"
    for path in [traces_dir, summaries_dir, splits_dir, metadata_dir]:
        path.mkdir(parents=True, exist_ok=True)

    calibration = build_calibration(root, keepalive_ms=args.keepalive_ms)
    rng = np.random.default_rng(args.seed)

    manifest_rows = []
    for item in DEFAULT_DATASETS:
        workflow = load_workflow(str(root / item["workflow_config"]))
        schedule = pd.read_csv(root / item["schedule"]).sort_values("index").reset_index(drop=True)
        trace = simulate_workflow_trace(
            workflow=workflow,
            schedule=schedule,
            calibration=calibration,
            rng=rng,
            base_entry_ts_ms=args.base_entry_ts_ms,
            burnin_copies=args.burnin_copies,
        )

        prefix = f"{workflow.workflow_name}_synthetic"
        trace_path = traces_dir / f"{prefix}.csv"
        summary_path = summaries_dir / f"{prefix}_summary.csv"
        trace.to_csv(trace_path, index=False)
        summarize_trace(trace).to_csv(summary_path, index=False)
        split_info = write_split_traces(trace, splits_dir, prefix, args.train_ratio)

        saved_requests = int(trace[trace["stage_name"] == "__entry__"]["request_id"].nunique())
        manifest_rows.append(
            {
                "workflow_name": workflow.workflow_name,
                "workflow_config": item["workflow_config"],
                "schedule": item["schedule"],
                "trace_path": str(trace_path),
                "summary_path": str(summary_path),
                **split_info,
                "saved_requests": saved_requests,
                "saved_rows": int(len(trace)),
                "cold_rate": float(
                    trace[trace["stage_name"] != "__entry__"]["cold_like"].astype(bool).mean()
                ),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    metadata = {
        "description": "Calibrated synthetic traces seeded by Azure entry schedules and OpenWhisk pilot replay data.",
        "calibration_source_traces": DEFAULT_CALIBRATION_TRACES,
        "dataset_pairs": DEFAULT_DATASETS,
        "calibration": {
            "action_intercept_ms": calibration.action_intercept_ms,
            "action_sleep_slope": calibration.action_sleep_slope,
            "action_residual_log_samples": int(len(calibration.action_residual_log_pool)),
            "cold_overhead_samples": int(len(calibration.cold_overhead_pool)),
            "warm_fast_overhead_samples": int(len(calibration.warm_fast_overhead_pool)),
            "warm_slow_overhead_samples": int(len(calibration.warm_slow_overhead_pool)),
            "warm_slow_probability": calibration.warm_slow_probability,
            "fast_warm_threshold_ms": calibration.fast_warm_threshold_ms,
            "keepalive_ms": calibration.keepalive_ms,
            "duration_action_sigma": calibration.duration_action_sigma,
            "overhead_action_sigma": calibration.overhead_action_sigma,
            "dispatch_jitter_ms_max": calibration.dispatch_jitter_ms_max,
        },
        "generation": {
            "base_entry_ts_ms": args.base_entry_ts_ms,
            "burnin_copies": args.burnin_copies,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
        },
        "notes": [
            "Entry arrivals are copied directly from Azure-derived schedules.",
            "Stage execution durations are calibrated from trip-booking replay data via a positive linear sleep_ms model with bootstrap residuals.",
            "Platform overhead is modeled as an empirical mixture: cold, warm_fast, and warm_slow.",
            "Cold versus warm assignment is generated by a per-action container reuse simulator with explicit keepalive state.",
            "This dataset is suitable for method development and ablation, but not as the sole external-validity evidence of a final paper.",
        ],
    }
    metadata_path = metadata_dir / "generation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote manifest: {manifest_path}")
    print(manifest.to_string(index=False))
    print(f"wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()

