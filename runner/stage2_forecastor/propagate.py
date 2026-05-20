import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..workflow import load_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--entry-forecast", required=True)
    parser.add_argument("--window-sec", type=int, default=5)
    parser.add_argument(
        "--window-ms",
        type=int,
        default=None,
        help="override --window-sec with a millisecond-level window",
    )
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument(
        "--train-until-ms",
        type=int,
        default=None,
        help="build delay kernels from rows with entry_ts_ms <= this timestamp",
    )
    parser.add_argument(
        "--train-until-window",
        type=int,
        default=None,
        help="build delay kernels from rows whose entry window index is <= this value",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def resolve_window_ms(args: argparse.Namespace) -> int:
    if args.window_ms is not None:
        if args.window_ms <= 0:
            raise ValueError("--window-ms must be positive")
        return args.window_ms
    if args.window_sec <= 0:
        raise ValueError("--window-sec must be positive")
    return args.window_sec * 1000


def build_delay_kernel(stage_rows: pd.DataFrame, window_ms: int) -> dict:
    delay_ms = stage_rows["dispatch_start_ms"].astype(float) - stage_rows["entry_ts_ms"].astype(float)
    offsets = np.maximum(0, np.floor(delay_ms / window_ms).astype(int))
    counts = offsets.value_counts().sort_index()
    total = counts.sum()
    if total == 0:
        return {0: 1.0}
    return {int(offset): float(count / total) for offset, count in counts.items()}


def ceil_count(value: float) -> int:
    return int(math.ceil(max(0.0, value)))


def alloc_count(value: float, activation_threshold: float) -> int:
    if value < activation_threshold:
        return 0
    return ceil_count(value)


def apply_training_cutoff(
    stage_rows: pd.DataFrame,
    window_ms: int,
    train_until_ms: Optional[int],
    train_until_window: Optional[int],
) -> pd.DataFrame:
    if train_until_ms is not None:
        stage_rows = stage_rows[stage_rows["entry_ts_ms"] <= train_until_ms].copy()
    if train_until_window is not None:
        stage_rows = stage_rows[
            (stage_rows["entry_ts_ms"] // window_ms).astype(int)
            <= train_until_window
        ].copy()
    return stage_rows


def main() -> None:
    args = parse_args()
    workflow = load_workflow(args.workflow)
    workflow_name = workflow.workflow_name
    window_ms = resolve_window_ms(args)

    trace = pd.read_csv(args.trace)
    forecast = pd.read_csv(args.entry_forecast)

    rows = []
    for node_name in workflow.nodes:
        stage_rows = trace[
            (trace["workflow_name"] == workflow_name)
            & (trace["stage_name"] == node_name)
            & (trace["status"] == "ok")
        ].copy()
        if stage_rows.empty:
            continue

        stage_rows = apply_training_cutoff(
            stage_rows,
            window_ms,
            args.train_until_ms,
            args.train_until_window,
        )
        if stage_rows.empty:
            continue

        kernel = build_delay_kernel(stage_rows, window_ms)
        for _, entry_row in forecast.iterrows():
            for offset, probability in kernel.items():
                target_window = int(entry_row["window"]) + offset
                rows.append(
                    {
                        "workflow_name": workflow_name,
                        "method": entry_row.get("method", "unknown"),
                        "stage_name": node_name,
                        "window": target_window,
                        "window_start_ms": target_window * window_ms,
                        "p50_count": float(entry_row["p50_count"]) * probability,
                        "p90_count": float(entry_row["p90_count"]) * probability,
                        "p95_count": float(entry_row["p95_count"]) * probability,
                        "p99_count": float(entry_row.get("p99_count", entry_row["p95_count"])) * probability,
                        "delay_offset_windows": offset,
                        "kernel_probability": probability,
                    }
                )

    out_df = (
        pd.DataFrame(rows)
        .groupby(["workflow_name", "method", "stage_name", "window", "window_start_ms"], as_index=False)
        [["p50_count", "p90_count", "p95_count", "p99_count"]]
        .sum()
    )
    out_df["ceil_p50_count"] = out_df["p50_count"].map(ceil_count)
    out_df["ceil_p90_count"] = out_df["p90_count"].map(ceil_count)
    out_df["ceil_p95_count"] = out_df["p95_count"].map(ceil_count)
    out_df["ceil_p99_count"] = out_df["p99_count"].map(ceil_count)
    out_df["alloc_p50_count"] = out_df["p50_count"].map(
        lambda value: alloc_count(value, args.activation_threshold)
    )
    out_df["alloc_p90_count"] = out_df["p90_count"].map(
        lambda value: alloc_count(value, args.activation_threshold)
    )
    out_df["alloc_p95_count"] = out_df["p95_count"].map(
        lambda value: alloc_count(value, args.activation_threshold)
    )
    out_df["alloc_p99_count"] = out_df["p99_count"].map(
        lambda value: alloc_count(value, args.activation_threshold)
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

