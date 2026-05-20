import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .compare_stage_forecasts import (
    build_independent_stage_forecast,
    forecast_from_series,
    load_split,
    propagate_entry_forecast,
    resolve_window_ms,
)
from ..workflow import load_workflow


POLICIES = ["p50", "p90", "p95"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rolling-origin stage-level comparison between independent per-stage "
            "forecasting and workflow-entry forecasting plus DAG propagation."
        )
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--split-map", default=None)
    parser.add_argument("--split-strategy", choices=["request-count", "time"], default="time")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--window-sec", type=int, default=5)
    parser.add_argument("--window-ms", type=int, default=None)
    parser.add_argument("--horizon-windows", type=int, default=1)
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
    parser.add_argument("--write-forecast-csvs", action="store_true")
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


def window_series(
    rows: pd.DataFrame,
    timestamp_col: str,
    window_ms: int,
) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=int)
    return (rows[timestamp_col] // window_ms).astype(int)


def actual_stage_by_window(
    test_stage_rows: pd.DataFrame,
    window_ms: int,
) -> dict[tuple[str, int], int]:
    if test_stage_rows.empty:
        return {}
    rows = test_stage_rows.copy()
    rows["window"] = window_series(rows, "dispatch_start_ms", window_ms)
    grouped = rows.groupby(["stage_name", "window"]).size()
    return {(str(stage), int(window)): int(count) for (stage, window), count in grouped.items()}


def entry_count_history(
    entry_rows: pd.DataFrame,
    first_window: int,
    origin_window: int,
) -> pd.Series:
    history = entry_rows[entry_rows["window"] <= origin_window].copy()
    if history.empty:
        return pd.Series([0.0], index=[origin_window], dtype=float)
    return (
        history.groupby("window")
        .size()
        .reindex(range(first_window, origin_window + 1), fill_value=0)
        .astype(float)
    )


def add_alloc_columns(
    forecast: pd.DataFrame,
    activation_threshold: float,
) -> pd.DataFrame:
    out = forecast.copy()
    for policy in POLICIES:
        raw_col = f"{policy}_count"
        if raw_col not in out:
            continue
        out[f"alloc_{policy}_count"] = out[raw_col].map(
            lambda value: 0 if float(value) < activation_threshold else int(math.ceil(max(0.0, float(value))))
        )
    return out


def forecast_lookup(forecast: pd.DataFrame) -> dict[tuple[str, int], pd.Series]:
    if forecast.empty:
        return {}
    rows = (
        forecast.groupby(["stage_name", "window"], as_index=False)
        .max(numeric_only=True)
        .reset_index(drop=True)
    )
    return {
        (str(row["stage_name"]), int(row["window"])): row
        for _, row in rows.iterrows()
    }


def summarize(detail: pd.DataFrame, window_ms: int, by_stage: bool) -> pd.DataFrame:
    group_cols = ["workflow_name", "method", "policy"]
    if by_stage:
        group_cols.append("stage_name")

    rows = []
    for keys, group in detail.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        actual = group["actual_count"].astype(float)
        forecast = group["forecast_count"].astype(float)
        allocated = group["allocated_count"].astype(float)
        active = group[group["actual_count"] > 0]
        actual_total = float(actual.sum())
        allocated_total = float(allocated.sum())
        under_total = float(group["under_count"].sum())
        over_total = float(group["over_count"].sum())
        row = {
            **dict(zip(group_cols, keys)),
            "origins": int(group["origin_window"].nunique()),
            "forecast_rows": int(len(group)),
            "active_rows": int(len(active)),
            "actual_total": int(actual_total),
            "allocated_replica_windows": int(allocated_total),
            "allocated_replica_seconds": float(allocated_total * window_ms / 1000.0),
            "under_total": int(under_total),
            "over_total": int(over_total),
            "coverage_rate": float((group["under_count"] == 0).mean()),
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
            "mae": float(np.mean(np.abs(actual - forecast))),
            "rmse": float(np.sqrt(np.mean((actual - forecast) ** 2))),
            "max_actual": int(actual.max()) if len(actual) else 0,
            "max_allocated": int(allocated.max()) if len(allocated) else 0,
        }
        rows.append(row)
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

    workflow_rows = trace[
        (trace["workflow_name"] == workflow_name)
        & (trace["status"] == "ok")
    ].copy()
    entries = workflow_rows[workflow_rows["stage_name"] == "__entry__"].copy()
    entries["window"] = window_series(entries, "entry_ts_ms", window_ms)
    first_entry_window = int(entries["window"].min())

    split_map["entry_window"] = (split_map["entry_ts_ms"] // window_ms).astype(int)
    if "split_cutoff_ms" in split_map.columns and split_map["split_cutoff_ms"].notna().any():
        train_end_window = int(split_map["split_cutoff_ms"].dropna().iloc[0] // window_ms)
        eval_start_window = train_end_window + 1
    else:
        train_end_window = int(split_map[split_map["split"] == "train"]["entry_window"].max())
        eval_start_window = int(split_map[split_map["split"] == "test"]["entry_window"].min())

    stage_rows = workflow_rows[workflow_rows["stage_name"] != "__entry__"].copy()
    stage_rows["dispatch_window"] = window_series(stage_rows, "dispatch_start_ms", window_ms)
    test_stage_rows = stage_rows[stage_rows["request_id"].isin(test_ids)].copy()
    if test_stage_rows.empty:
        raise ValueError("no test stage rows found")
    eval_end_window = int(test_stage_rows["dispatch_window"].max())
    actual_by_stage_window = actual_stage_by_window(test_stage_rows, window_ms)

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    detail_rows = []
    forecast_frames = []
    last_origin = eval_end_window - 1

    for origin_window in range(train_end_window, last_origin + 1, step):
        horizon = min(args.horizon_windows, eval_end_window - origin_window)
        if horizon <= 0:
            continue

        history_entry_counts = entry_count_history(entries, first_entry_window, origin_window)
        history_stage_rows = stage_rows[stage_rows["dispatch_window"] <= origin_window].copy()

        for method in methods:
            entry_forecast = forecast_from_series(
                counts=history_entry_counts,
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
                method_label=f"dag-{method}",
            )
            entry_forecast["workflow_name"] = workflow_name
            entry_forecast["window_start_ms"] = entry_forecast["window"] * window_ms
            dag_forecast = propagate_entry_forecast(
                workflow_name=workflow_name,
                workflow=workflow,
                entry_forecast=entry_forecast,
                train_stage_rows=history_stage_rows,
                window_ms=window_ms,
            )
            dag_forecast = add_alloc_columns(dag_forecast, args.activation_threshold)
            dag_forecast["origin_window"] = origin_window

            independent_forecast = build_independent_stage_forecast(
                workflow_name=workflow_name,
                workflow=workflow,
                train_stage_rows=history_stage_rows,
                train_end_window=origin_window,
                horizon=horizon,
                window_ms=window_ms,
                args=argparse.Namespace(**{**vars(args), "method": method}),
            )
            independent_forecast["method"] = f"independent-{method}"
            independent_forecast["origin_window"] = origin_window

            for forecast in [dag_forecast, independent_forecast]:
                forecast_frames.append(forecast)
                lookup = forecast_lookup(forecast)
                method_name = str(forecast["method"].iloc[0]) if not forecast.empty else method
                for target_window in range(origin_window + 1, origin_window + horizon + 1):
                    if target_window < eval_start_window or target_window > eval_end_window:
                        continue
                    for stage_name in workflow.nodes:
                        forecast_row = lookup.get((stage_name, target_window))
                        actual = int(actual_by_stage_window.get((stage_name, target_window), 0))
                        for policy in POLICIES:
                            if forecast_row is None:
                                forecast_count = 0.0
                                allocated = 0
                            else:
                                forecast_count = float(forecast_row.get(f"{policy}_count", 0.0))
                                allocated = allocated_count(forecast_row, policy)
                            detail_rows.append(
                                {
                                    "workflow_name": workflow_name,
                                    "method": method_name,
                                    "policy": policy,
                                    "stage_name": stage_name,
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

    detail = pd.DataFrame(detail_rows)
    summary = summarize(detail, window_ms, by_stage=False)
    summary_by_stage = summarize(detail, window_ms, by_stage=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{workflow_name}_rolling_stage_compare_summary.csv"
    summary_by_stage_path = out_dir / f"{workflow_name}_rolling_stage_compare_by_stage.csv"
    detail_path = out_dir / f"{workflow_name}_rolling_stage_compare_detail.csv"
    forecast_path = out_dir / f"{workflow_name}_rolling_stage_forecasts.csv"
    metadata_path = out_dir / f"{workflow_name}_rolling_stage_compare_metadata.json"

    summary.to_csv(summary_path, index=False)
    summary_by_stage.to_csv(summary_by_stage_path, index=False)
    if args.write_detail:
        detail.to_csv(detail_path, index=False)
    if args.write_forecast_csvs:
        pd.concat(forecast_frames, ignore_index=True).to_csv(forecast_path, index=False)
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
    print(f"wrote {summary_by_stage_path}")
    if args.write_detail:
        print(f"wrote {detail_path}")
    if args.write_forecast_csvs:
        print(f"wrote {forecast_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

