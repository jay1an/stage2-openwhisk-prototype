import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .compare_stage_forecasts import (
    build_count_series,
    forecast_from_series,
    load_split,
    resolve_window_ms,
)
from ..workflow import load_workflow


POLICIES = ["p50", "p90", "p95"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling-origin comparison for workflow entry forecasting baselines."
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--split-map", default=None)
    parser.add_argument("--split-strategy", choices=["request-count", "time"], default="time")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--window-sec", type=int, default=5)
    parser.add_argument("--window-ms", type=int, default=None)
    parser.add_argument("--horizon-windows", type=int, default=12)
    parser.add_argument("--origin-step-windows", type=int, default=None)
    parser.add_argument(
        "--methods",
        default="ewma,burst-aware,hurdle-ewma,tsb,hazard-hurdle",
        help="comma-separated forecasting methods",
    )
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--residual-window", type=int, default=60)
    parser.add_argument("--history-window", type=int, default=30)
    parser.add_argument("--burst-threshold", type=float, default=2.0)
    parser.add_argument("--burst-period-windows", type=int, default=None)
    parser.add_argument("--burst-width-windows", type=int, default=0)
    parser.add_argument("--background-count", type=float, default=None)
    parser.add_argument("--idle-zero-ratio", type=float, default=0.8)
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--write-detail", action="store_true")
    return parser.parse_args()


def allocated_count(row: pd.Series, policy: str) -> int:
    alloc_col = f"alloc_{policy}_count"
    ceil_col = f"ceil_{policy}_count"
    raw_col = f"{policy}_count"
    if alloc_col in row and not pd.isna(row[alloc_col]):
        return int(row[alloc_col])
    if ceil_col in row and not pd.isna(row[ceil_col]):
        return int(row[ceil_col])
    return int(math.ceil(max(0.0, float(row[raw_col]))))


def build_actual_entry_counts(trace: pd.DataFrame, workflow_name: str, window_ms: int) -> dict[int, int]:
    rows = trace[
        (trace["workflow_name"] == workflow_name)
        & (trace["stage_name"] == "__entry__")
        & (trace["status"] == "ok")
    ].copy()
    rows["window"] = (rows["entry_ts_ms"] // window_ms).astype(int)
    return rows.groupby("window").size().astype(int).to_dict()


def summarize(detail: pd.DataFrame, window_ms: int) -> pd.DataFrame:
    rows = []
    for (workflow_name, method, policy), group in detail.groupby(["workflow_name", "method", "policy"]):
        actual = group["actual_count"].astype(float)
        forecast = group["forecast_count"].astype(float)
        allocated = group["allocated_count"].astype(float)
        active = group[group["actual_count"] > 0]
        predicted_active = group[group["allocated_count"] > 0]
        tp = int(((group["actual_count"] > 0) & (group["allocated_count"] > 0)).sum())
        fp = int(((group["actual_count"] == 0) & (group["allocated_count"] > 0)).sum())
        fn = int(((group["actual_count"] > 0) & (group["allocated_count"] == 0)).sum())
        tn = int(((group["actual_count"] == 0) & (group["allocated_count"] == 0)).sum())
        actual_total = float(actual.sum())
        allocated_total = float(allocated.sum())
        under_total = float(group["under_count"].sum())
        over_total = float(group["over_count"].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "workflow_name": workflow_name,
                "method": method,
                "policy": policy,
                "origins": int(group["origin_window"].nunique()),
                "forecast_rows": int(len(group)),
                "active_rows": int(len(active)),
                "predicted_active_rows": int(len(predicted_active)),
                "true_positive_active_rows": tp,
                "false_positive_active_rows": fp,
                "false_negative_active_rows": fn,
                "true_negative_active_rows": tn,
                "actual_total": int(actual_total),
                "allocated_replica_windows": int(allocated_total),
                "allocated_replica_seconds": float(allocated_total * window_ms / 1000.0),
                "under_total": int(under_total),
                "over_total": int(over_total),
                "coverage_rate": float((group["under_count"] == 0).mean()),
                "quantile_hit_rate": float((actual <= forecast).mean()),
                "active_quantile_hit_rate": (
                    float((active["actual_count"] <= active["forecast_count"]).mean()) if len(active) else 1.0
                ),
                "active_coverage_rate": float((active["under_count"] == 0).mean()) if len(active) else 1.0,
                "demand_coverage_rate": float(1.0 - under_total / actual_total) if actual_total > 0 else 1.0,
                "allocation_utilization": (
                    float((actual_total - under_total) / allocated_total)
                    if allocated_total > 0
                    else 0.0
                ),
                "over_allocation_ratio": (
                    float(over_total / allocated_total)
                    if allocated_total > 0
                    else 0.0
                ),
                "active_precision": float(precision),
                "active_recall": float(recall),
                "active_f1": float(f1),
                "active_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0,
                "mae": float(np.mean(np.abs(actual - forecast))),
                "rmse": float(np.sqrt(np.mean((actual - forecast) ** 2))),
                "max_actual": int(actual.max()) if len(actual) else 0,
                "max_allocated": int(allocated.max()) if len(allocated) else 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.horizon_windows <= 0:
        raise ValueError("--horizon-windows must be positive")
    step = args.origin_step_windows or args.horizon_windows
    if step <= 0:
        raise ValueError("--origin-step-windows must be positive")

    window_ms = resolve_window_ms(args)
    workflow = load_workflow(args.workflow_config)
    workflow_name = workflow.workflow_name
    trace = pd.read_csv(args.trace)
    train_ids, test_ids, split_map = load_split(
        trace,
        workflow_name,
        args.split_map,
        args.train_ratio,
        args.split_strategy,
    )
    split_map["entry_window"] = (split_map["entry_ts_ms"] // window_ms).astype(int)
    if "split_cutoff_ms" in split_map.columns and split_map["split_cutoff_ms"].notna().any():
        train_end_window = int(split_map["split_cutoff_ms"].dropna().iloc[0] // window_ms)
        eval_start_window = train_end_window + 1
    else:
        train_end_window = int(split_map[split_map["split"] == "train"]["entry_window"].max())
        eval_start_window = int(split_map[split_map["split"] == "test"]["entry_window"].min())
    eval_end_window = int(split_map[split_map["split"] == "test"]["entry_window"].max())

    entries = trace[
        (trace["workflow_name"] == workflow_name)
        & (trace["stage_name"] == "__entry__")
        & (trace["status"] == "ok")
    ].copy()
    entries["window"] = (entries["entry_ts_ms"] // window_ms).astype(int)
    first_window = int(entries["window"].min())
    actual_by_window = build_actual_entry_counts(trace, workflow_name, window_ms)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]

    rows = []
    last_origin = eval_end_window - 1
    for origin_window in range(train_end_window, last_origin + 1, step):
        history = entries[entries["window"] <= origin_window].copy()
        if history.empty:
            counts = pd.Series([0.0], index=[origin_window], dtype=float)
        else:
            counts = (
                history.groupby("window")
                .size()
                .reindex(range(first_window, origin_window + 1), fill_value=0)
                .astype(float)
            )

        horizon = min(args.horizon_windows, eval_end_window - origin_window)
        if horizon <= 0:
            continue
        for method in methods:
            forecast = forecast_from_series(
                counts=counts,
                method=method,
                alpha=args.alpha,
                residual_window=args.residual_window,
                history_window=args.history_window,
                burst_threshold=args.burst_threshold,
                burst_period_windows=args.burst_period_windows,
                burst_width_windows=args.burst_width_windows,
                background_count=args.background_count,
                idle_zero_ratio=args.idle_zero_ratio,
                activation_threshold=args.activation_threshold,
                horizon=horizon,
                method_label=method,
            )
            for _, forecast_row in forecast.iterrows():
                target_window = int(forecast_row["window"])
                if target_window < eval_start_window or target_window > eval_end_window:
                    continue
                actual = int(actual_by_window.get(target_window, 0))
                for policy in POLICIES:
                    forecast_count = float(forecast_row[f"{policy}_count"])
                    allocated = allocated_count(forecast_row, policy)
                    rows.append(
                        {
                            "workflow_name": workflow_name,
                            "method": method,
                            "policy": policy,
                            "origin_window": origin_window,
                            "target_window": target_window,
                            "horizon_step": target_window - origin_window,
                            "actual_count": actual,
                            "forecast_count": forecast_count,
                            "allocated_count": allocated,
                            "under_count": max(0, actual - allocated),
                            "over_count": max(0, allocated - actual),
                        }
                    )

    detail = pd.DataFrame(rows)
    summary = summarize(detail, window_ms)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{workflow_name}_rolling_entry_compare_summary.csv"
    detail_path = out_dir / f"{workflow_name}_rolling_entry_compare_detail.csv"
    metadata_path = out_dir / f"{workflow_name}_rolling_entry_compare_metadata.json"
    summary.to_csv(summary_path, index=False)
    if args.write_detail:
        detail.to_csv(detail_path, index=False)
    metadata = {
        "workflow_name": workflow_name,
        "trace": args.trace,
        "workflow_config": args.workflow_config,
        "split_map": args.split_map,
        "split_strategy": args.split_strategy if args.split_map is None else "provided-map",
        "window_ms": window_ms,
        "horizon_windows": args.horizon_windows,
        "origin_step_windows": step,
        "methods": methods,
        "train_requests": len(train_ids),
        "test_requests": len(test_ids),
        "train_end_window": train_end_window,
        "eval_start_window": eval_start_window,
        "eval_end_window": eval_end_window,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    if args.write_detail:
        print(f"wrote {detail_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

