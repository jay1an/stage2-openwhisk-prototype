"""Online 1-step-ahead evaluation harness for Stage-2 entry forecasting.

Walks every test window as an online origin, builds history up to that window,
runs each method's 1-step-ahead forecast, and records (actual, forecast, allocated)
per (method, policy, window). Adds two reference rows:

- oracle: knows the true count for the next window; allocated = actual (best possible)
- selector: pinball-loss weighted online expert selection over a recent window

Outputs:
- {workflow}_online_eval_detail.csv  (all rows)
- {workflow}_online_eval_summary.csv (per-method aggregates)
- {workflow}_online_eval_metadata.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .compare_stage_forecasts import build_count_series, forecast_from_series, load_split
from ..workflow import load_workflow


POLICIES = ["p50", "p90", "p95"]
POLICY_Q = {"p50": 0.5, "p90": 0.9, "p95": 0.95}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", required=True)
    p.add_argument("--workflow-config", required=True)
    p.add_argument("--split-map", required=True)
    p.add_argument("--window-ms", type=int, default=5000)
    p.add_argument(
        "--methods",
        default="ewma,burst-aware,hurdle-ewma,hazard-hurdle,burst-localized,fip-fourier",
    )
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--residual-window", type=int, default=60)
    p.add_argument("--history-window", type=int, default=30)
    p.add_argument("--burst-threshold", type=float, default=2.0)
    p.add_argument("--burst-period-windows", type=int, default=None)
    p.add_argument("--burst-width-windows", type=int, default=0)
    p.add_argument("--background-count", type=float, default=None)
    p.add_argument("--idle-zero-ratio", type=float, default=0.8)
    p.add_argument("--activation-threshold", type=float, default=0.1)
    p.add_argument(
        "--selector-window",
        type=int,
        default=48,
        help="number of recent windows used to score experts for the selector",
    )
    p.add_argument(
        "--selector-min-history",
        type=int,
        default=12,
        help="number of warmup windows before the selector can switch",
    )
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def pinball_loss(q: float, actual: float, predicted: float) -> float:
    delta = actual - predicted
    return max(q * delta, (q - 1.0) * delta)


def allocated_for_policy(forecast_row: pd.Series, policy: str) -> int:
    alloc_col = f"alloc_{policy}_count"
    if alloc_col in forecast_row and not pd.isna(forecast_row[alloc_col]):
        return int(forecast_row[alloc_col])
    ceil_col = f"ceil_{policy}_count"
    if ceil_col in forecast_row and not pd.isna(forecast_row[ceil_col]):
        return int(forecast_row[ceil_col])
    return int(math.ceil(max(0.0, float(forecast_row[f"{policy}_count"]))))


def main() -> None:
    args = parse_args()
    workflow = load_workflow(args.workflow_config)
    workflow_name = workflow.workflow_name
    trace = pd.read_csv(args.trace)

    train_ids, test_ids, split_map = load_split(
        trace,
        workflow_name,
        args.split_map,
        train_ratio=0.7,
        split_strategy="time",
    )
    split_map["entry_window"] = (split_map["entry_ts_ms"] // args.window_ms).astype(int)
    train_end_window = int(split_map[split_map["split"] == "train"]["entry_window"].max())
    eval_start_window = train_end_window + 1
    eval_end_window = int(split_map[split_map["split"] == "test"]["entry_window"].max())

    entries = trace[
        (trace["workflow_name"] == workflow_name) & (trace["stage_name"] == "__entry__")
    ].copy()
    entries["window"] = (entries["entry_ts_ms"] // args.window_ms).astype(int)
    first_window = int(entries["window"].min())
    actual_by_window = entries.groupby("window").size().astype(int).to_dict()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # Per-method per-policy rolling pinball loss buffer for selector.
    loss_buf: dict[str, dict[str, deque]] = {
        m: {p: deque(maxlen=args.selector_window) for p in POLICIES} for m in methods
    }

    detail_rows: list[dict] = []
    selector_choices: list[dict] = []

    for origin_window in range(train_end_window, eval_end_window):
        target_window = origin_window + 1
        actual = int(actual_by_window.get(target_window, 0))

        counts = (
            entries[entries["window"] <= origin_window]
            .groupby("window")
            .size()
            .reindex(range(first_window, origin_window + 1), fill_value=0)
            .astype(float)
        )

        per_method_forecast = {}
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
                horizon=1,
                method_label=method,
            )
            row = forecast.iloc[0]
            per_method_forecast[method] = row
            for policy in POLICIES:
                fc = float(row[f"{policy}_count"])
                alloc = allocated_for_policy(row, policy)
                detail_rows.append(
                    {
                        "workflow_name": workflow_name,
                        "method": method,
                        "policy": policy,
                        "origin_window": origin_window,
                        "target_window": target_window,
                        "actual_count": actual,
                        "forecast_count": fc,
                        "allocated_count": alloc,
                        "under_count": max(0, actual - alloc),
                        "over_count": max(0, alloc - actual),
                        "pinball_loss": pinball_loss(POLICY_Q[policy], actual, fc),
                    }
                )
                loss_buf[method][policy].append(pinball_loss(POLICY_Q[policy], actual, fc))

        # Oracle: knows actual.
        for policy in POLICIES:
            detail_rows.append(
                {
                    "workflow_name": workflow_name,
                    "method": "oracle",
                    "policy": policy,
                    "origin_window": origin_window,
                    "target_window": target_window,
                    "actual_count": actual,
                    "forecast_count": float(actual),
                    "allocated_count": actual,
                    "under_count": 0,
                    "over_count": 0,
                    "pinball_loss": 0.0,
                }
            )

        # Selector (per-policy): pick method with lowest recent mean pinball loss.
        windows_seen = origin_window - train_end_window + 1
        for policy in POLICIES:
            if windows_seen <= args.selector_min_history:
                chosen = "ewma"  # safe default during warmup
            else:
                scores = {
                    m: (np.mean(loss_buf[m][policy]) if loss_buf[m][policy] else float("inf"))
                    for m in methods
                }
                chosen = min(scores, key=scores.get)
            row = per_method_forecast[chosen]
            fc = float(row[f"{policy}_count"])
            alloc = allocated_for_policy(row, policy)
            detail_rows.append(
                {
                    "workflow_name": workflow_name,
                    "method": f"selector",
                    "policy": policy,
                    "origin_window": origin_window,
                    "target_window": target_window,
                    "actual_count": actual,
                    "forecast_count": fc,
                    "allocated_count": alloc,
                    "under_count": max(0, actual - alloc),
                    "over_count": max(0, alloc - actual),
                    "pinball_loss": pinball_loss(POLICY_Q[policy], actual, fc),
                }
            )
            selector_choices.append(
                {
                    "policy": policy,
                    "origin_window": origin_window,
                    "target_window": target_window,
                    "chosen_method": chosen,
                }
            )

    detail = pd.DataFrame(detail_rows)
    choices = pd.DataFrame(selector_choices)

    # Aggregate.
    summary_rows = []
    for (method, policy), grp in detail.groupby(["method", "policy"]):
        actual_arr = grp["actual_count"].astype(float).values
        alloc_arr = grp["allocated_count"].astype(float).values
        fc_arr = grp["forecast_count"].astype(float).values
        covered = (alloc_arr >= actual_arr).mean()
        active_mask = actual_arr > 0
        active_coverage = (alloc_arr[active_mask] >= actual_arr[active_mask]).mean() if active_mask.any() else 1.0
        mae = float(np.mean(np.abs(alloc_arr - actual_arr)))
        rmse = float(np.sqrt(np.mean((alloc_arr - actual_arr) ** 2)))
        pinball = float(grp["pinball_loss"].mean())
        summary_rows.append(
            {
                "method": method,
                "policy": policy,
                "windows": len(grp),
                "actual_total": int(actual_arr.sum()),
                "allocated_total": int(alloc_arr.sum()),
                "under_total": int(grp["under_count"].sum()),
                "over_total": int(grp["over_count"].sum()),
                "coverage_rate": float(covered),
                "active_coverage_rate": float(active_coverage),
                "mae": mae,
                "rmse": rmse,
                "pinball_loss_mean": pinball,
                "over_allocation_ratio": float(grp["over_count"].sum() / max(1, alloc_arr.sum())),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["policy", "mae"])

    # Selector usage breakdown.
    selector_usage = (
        choices.groupby(["policy", "chosen_method"]).size().reset_index(name="count")
    )
    selector_usage["pct"] = selector_usage.groupby("policy")["count"].transform(
        lambda x: x / x.sum() * 100
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"{workflow_name}_online_eval_detail.csv"
    summary_path = out_dir / f"{workflow_name}_online_eval_summary.csv"
    selector_path = out_dir / f"{workflow_name}_online_eval_selector_usage.csv"
    metadata_path = out_dir / f"{workflow_name}_online_eval_metadata.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    selector_usage.to_csv(selector_path, index=False)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace": args.trace,
        "split_map": args.split_map,
        "workflow_name": workflow_name,
        "window_ms": args.window_ms,
        "methods": methods,
        "eval_start_window": eval_start_window,
        "eval_end_window": eval_end_window,
        "selector_window": args.selector_window,
        "selector_min_history": args.selector_min_history,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("===== Summary (p95 only, sorted by MAE) =====")
    p95_summary = summary[summary["policy"] == "p95"].copy()
    print(p95_summary.to_string(index=False))
    print(f"\n===== Selector usage (p95) =====")
    print(selector_usage[selector_usage["policy"] == "p95"].to_string(index=False))
    print(f"\nWrote: {summary_path}\nWrote: {detail_path}")


if __name__ == "__main__":
    main()
