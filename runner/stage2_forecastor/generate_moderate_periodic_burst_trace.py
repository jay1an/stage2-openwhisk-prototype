"""Generate a synthetic moderate-periodic-burst entry trace for Stage-2 evaluation.

Designed per user spec:
- 2 hours (1440 windows at 5 sec each)
- Peak count per window ~ 12-14 (instantaneous QPS ~ 2.4-2.8)
- Strong main period (5 min) + weak subharmonic (1 min)
- Occasional moderate bursts every ~12 minutes (2-3 windows, peak ~10-12)
- Realistic for serverless workloads; not engineered to favor any single forecast method.

Outputs entry-only rows compatible with Stage-2 forecast_entry.py / compare_stage_forecasts.py
plus a split-map CSV that marks the first 70% of windows as warmup-history and the last 30%
as the evaluation horizon (Stage-2 is an online predictor; "split" only labels
where evaluation begins, not where any model gets trained).
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def build_lambda_grid(
    num_windows: int,
    window_ms: int,
    base: float,
    main_amp: float,
    main_period_windows: int,
    sub_amp: float,
    sub_period_windows: int,
    burst_period_windows: int,
    burst_width_windows: int,
    burst_amp: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the per-window Poisson rate vector lambda(t)."""

    t = np.arange(num_windows)
    main = main_amp * 0.5 * (1.0 + np.sin(2 * np.pi * t / main_period_windows - np.pi / 2))
    sub = sub_amp * 0.5 * (1.0 + np.sin(2 * np.pi * t / sub_period_windows))
    lam = base + main + sub

    # Inject sparse moderate bursts at jittered cadence.
    burst_centers = []
    cursor = burst_period_windows
    while cursor < num_windows:
        jitter = int(rng.integers(-burst_period_windows // 4, burst_period_windows // 4 + 1))
        center = int(np.clip(cursor + jitter, 0, num_windows - 1))
        burst_centers.append(center)
        cursor += burst_period_windows

    for center in burst_centers:
        for offset in range(-burst_width_windows, burst_width_windows + 1):
            idx = center + offset
            if 0 <= idx < num_windows:
                decay = burst_amp * (1.0 - abs(offset) / (burst_width_windows + 1))
                lam[idx] += decay

    return lam, burst_centers


def sample_counts(lam: np.ndarray, hard_cap: int, rng: np.random.Generator) -> np.ndarray:
    counts = rng.poisson(lam=lam)
    counts = np.minimum(counts, hard_cap)
    return counts


def spread_within_window(count: int, window_start_ms: int, window_ms: int, rng: np.random.Generator):
    """Distribute `count` request timestamps uniformly within one window."""
    if count <= 0:
        return np.array([], dtype=np.int64)
    offsets = rng.uniform(0.0, window_ms, size=count)
    return (window_start_ms + offsets).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-name", default="spoken_dialog_flow")
    parser.add_argument("--window-ms", type=int, default=5000)
    parser.add_argument("--num-windows", type=int, default=1440, help="default: 2 hours at 5 sec")
    parser.add_argument("--base-rate", type=float, default=1.4)
    parser.add_argument("--main-amp", type=float, default=3.4)
    parser.add_argument("--main-period-windows", type=int, default=60, help="5 min by default")
    parser.add_argument("--sub-amp", type=float, default=1.0)
    parser.add_argument("--sub-period-windows", type=int, default=12, help="1 min by default")
    parser.add_argument("--burst-period-windows", type=int, default=144, help="every 12 min")
    parser.add_argument("--burst-width-windows", type=int, default=1, help="half-width; total = 2*w+1")
    parser.add_argument("--burst-amp", type=float, default=6.0)
    parser.add_argument("--hard-cap", type=int, default=14)
    parser.add_argument("--base-entry-ts-ms", type=int, default=1900000000000)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument(
        "--out-dir",
        default="data/stage_synthetic/moderate_periodic_burst",
        help="root dir for the trace artifacts",
    )
    parser.add_argument("--tag", default="moderate_periodic_burst", help="filename infix")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    lam, burst_centers = build_lambda_grid(
        num_windows=args.num_windows,
        window_ms=args.window_ms,
        base=args.base_rate,
        main_amp=args.main_amp,
        main_period_windows=args.main_period_windows,
        sub_amp=args.sub_amp,
        sub_period_windows=args.sub_period_windows,
        burst_period_windows=args.burst_period_windows,
        burst_width_windows=args.burst_width_windows,
        burst_amp=args.burst_amp,
        rng=rng,
    )
    counts = sample_counts(lam, hard_cap=args.hard_cap, rng=rng)

    # Materialize entry rows.
    rows = []
    ts_offset = 0
    request_seq = 0
    for w_idx, c in enumerate(counts):
        window_start = args.base_entry_ts_ms + w_idx * args.window_ms
        offsets = spread_within_window(int(c), window_start, args.window_ms, rng)
        for ts in sorted(offsets):
            rows.append(
                {
                    "workflow_name": args.workflow_name,
                    "request_id": str(uuid.UUID(int=int.from_bytes(rng.bytes(16), "big"))),
                    "stage_name": "__entry__",
                    "parent_stages": "",
                    "entry_ts_ms": int(ts),
                    "dispatch_start_ms": int(ts),
                    "dispatch_end_ms": int(ts),
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
            )
            request_seq += 1

    df = pd.DataFrame(rows)

    # Designate eval start at train_ratio mark (per-window, not per-request, so it's deterministic).
    n_train_windows = int(math.ceil(args.num_windows * args.train_ratio))
    eval_start_window = n_train_windows
    base_window = args.base_entry_ts_ms // args.window_ms
    eval_start_ms = (base_window + eval_start_window) * args.window_ms

    df["fold"] = np.where(df["entry_ts_ms"] < eval_start_ms, "train", "test")

    # Persist artifacts.
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    file_base = f"{args.workflow_name}_{args.tag}"

    trace_path = out_root / f"{file_base}_stage_trace.csv"
    train_path = out_root / "splits" / f"{file_base}_stage_trace_train.csv"
    test_path = out_root / "splits" / f"{file_base}_stage_trace_test.csv"
    split_map_path = out_root / "splits" / f"{file_base}_stage_trace_split.csv"
    train_path.parent.mkdir(parents=True, exist_ok=True)

    trace_cols = [
        "workflow_name", "request_id", "stage_name", "parent_stages", "entry_ts_ms",
        "dispatch_start_ms", "dispatch_end_ms", "dispatch_latency_ms",
        "action_start_ns", "action_end_ns", "action_duration_ms", "platform_overhead_ms",
        "container_id", "cold_like", "status", "error",
    ]
    df[trace_cols].to_csv(trace_path, index=False)
    df[df["fold"] == "train"][trace_cols].to_csv(train_path, index=False)
    df[df["fold"] == "test"][trace_cols].to_csv(test_path, index=False)
    df[["request_id", "fold"]].to_csv(split_map_path, index=False)

    # Per-window summary CSV.
    per_window_df = pd.DataFrame(
        {
            "window_index": np.arange(args.num_windows),
            "window_start_ms": args.base_entry_ts_ms + np.arange(args.num_windows) * args.window_ms,
            "lambda": lam,
            "count": counts,
            "fold": np.where(np.arange(args.num_windows) < eval_start_window, "train", "test"),
        }
    )
    per_window_path = out_root / f"{file_base}_per_window.csv"
    per_window_df.to_csv(per_window_path, index=False)

    # Metadata.
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_name": args.workflow_name,
        "window_ms": args.window_ms,
        "num_windows": args.num_windows,
        "duration_minutes": args.num_windows * args.window_ms / 60000,
        "spec": {
            "base_rate": args.base_rate,
            "main_amp": args.main_amp,
            "main_period_windows": args.main_period_windows,
            "sub_amp": args.sub_amp,
            "sub_period_windows": args.sub_period_windows,
            "burst_period_windows": args.burst_period_windows,
            "burst_width_windows": args.burst_width_windows,
            "burst_amp": args.burst_amp,
            "hard_cap": args.hard_cap,
        },
        "burst_centers": burst_centers,
        "train_ratio": args.train_ratio,
        "eval_start_window_index": eval_start_window,
        "eval_start_ms": eval_start_ms,
        "base_entry_ts_ms": args.base_entry_ts_ms,
        "seed": args.seed,
        "summary": {
            "total_requests": int(counts.sum()),
            "train_requests": int(df[df["fold"] == "train"].shape[0]),
            "test_requests": int(df[df["fold"] == "test"].shape[0]),
            "active_windows": int((counts > 0).sum()),
            "active_rate": float((counts > 0).mean()),
            "peak": int(counts.max()),
            "mean_all": float(counts.mean()),
            "mean_active": float(counts[counts > 0].mean()) if (counts > 0).any() else 0.0,
            "p50": float(np.quantile(counts, 0.5)),
            "p90": float(np.quantile(counts, 0.9)),
            "p95": float(np.quantile(counts, 0.95)),
            "p99": float(np.quantile(counts, 0.99)),
        },
        "paths": {
            "trace": str(trace_path),
            "train_trace": str(train_path),
            "test_trace": str(test_path),
            "split_map": str(split_map_path),
            "per_window": str(per_window_path),
        },
        "notes": [
            "Entry-only rows; downstream stage rows omitted (Stage-2 forecast_entry.py only consumes __entry__).",
            "Stage-2 is an online predictor; the train/test split here just labels where evaluation begins.",
        ],
    }
    metadata_path = out_root / f"{file_base}_stage_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(json.dumps(metadata["summary"], indent=2))
    print(f"\nTrace written to {trace_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
