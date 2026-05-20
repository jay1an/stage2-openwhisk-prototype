import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


POLICIES = ["p50", "p90", "p95"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--forecast", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument(
        "--level",
        choices=["entry", "stage"],
        required=True,
        help="evaluate entry forecast or stage-level forecast",
    )
    parser.add_argument("--window-sec", type=int, default=5)
    parser.add_argument(
        "--window-ms",
        type=int,
        default=None,
        help="override --window-sec with a millisecond-level window",
    )
    parser.add_argument(
        "--eval-start-window",
        type=int,
        default=None,
        help="first forecast window to evaluate, inclusive",
    )
    parser.add_argument(
        "--eval-end-window",
        type=int,
        default=None,
        help="last forecast window to evaluate, inclusive",
    )
    parser.add_argument(
        "--actual-entry-start-window",
        type=int,
        default=None,
        help="only count actual rows from workflow entries at or after this entry window",
    )
    parser.add_argument(
        "--actual-entry-end-window",
        type=int,
        default=None,
        help="only count actual rows from workflow entries at or before this entry window",
    )
    parser.add_argument("--detail-out", required=True)
    parser.add_argument("--summary-out", required=True)
    return parser.parse_args()


def resolve_window_ms(args: argparse.Namespace) -> int:
    if args.window_ms is not None:
        if args.window_ms <= 0:
            raise ValueError("--window-ms must be positive")
        return args.window_ms
    if args.window_sec <= 0:
        raise ValueError("--window-sec must be positive")
    return args.window_sec * 1000


def ceil_count(value: float) -> int:
    return int(math.ceil(max(0.0, value)))


def apply_entry_window_filter(
    rows: pd.DataFrame,
    window_ms: int,
    start_window: int | None,
    end_window: int | None,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    entry_window = (rows["entry_ts_ms"] // window_ms).astype(int)
    if start_window is not None:
        rows = rows[entry_window >= start_window].copy()
        entry_window = entry_window.loc[rows.index]
    if end_window is not None:
        rows = rows[entry_window <= end_window].copy()
    return rows


def actual_entry_counts(
    trace: pd.DataFrame,
    workflow: str,
    window_ms: int,
    entry_start_window: int | None,
    entry_end_window: int | None,
) -> pd.DataFrame:
    rows = trace[
        (trace["workflow_name"] == workflow)
        & (trace["stage_name"] == "__entry__")
        & (trace["status"] == "ok")
    ].copy()
    rows = apply_entry_window_filter(
        rows,
        window_ms,
        entry_start_window,
        entry_end_window,
    )
    if rows.empty:
        return pd.DataFrame(columns=["window", "actual_count"])
    rows["window"] = (rows["entry_ts_ms"] // window_ms).astype(int)
    return (
        rows.groupby("window")
        .size()
        .reset_index(name="actual_count")
    )


def actual_stage_counts(
    trace: pd.DataFrame,
    workflow: str,
    window_ms: int,
    entry_start_window: int | None,
    entry_end_window: int | None,
) -> pd.DataFrame:
    rows = trace[
        (trace["workflow_name"] == workflow)
        & (trace["stage_name"] != "__entry__")
        & (trace["status"] == "ok")
    ].copy()
    rows = apply_entry_window_filter(
        rows,
        window_ms,
        entry_start_window,
        entry_end_window,
    )
    if rows.empty:
        return pd.DataFrame(columns=["stage_name", "window", "actual_count"])
    rows["window"] = (rows["dispatch_start_ms"] // window_ms).astype(int)
    return (
        rows.groupby(["stage_name", "window"])
        .size()
        .reset_index(name="actual_count")
    )


def method_name(forecast: pd.DataFrame) -> str:
    if "method" not in forecast:
        return "unknown"
    values = sorted(str(value) for value in forecast["method"].dropna().unique())
    return "+".join(values) if values else "unknown"


def resolve_eval_bounds(
    forecast: pd.DataFrame,
    eval_start_window: int | None,
    eval_end_window: int | None,
) -> tuple[int, int]:
    if forecast.empty and (eval_start_window is None or eval_end_window is None):
        raise ValueError("empty forecast requires explicit --eval-start-window and --eval-end-window")
    start = int(eval_start_window) if eval_start_window is not None else int(forecast["window"].min())
    end = int(eval_end_window) if eval_end_window is not None else int(forecast["window"].max())
    if end < start:
        raise ValueError("--eval-end-window must be >= --eval-start-window")
    return start, end


def filter_by_window(df: pd.DataFrame, start_window: int, end_window: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[(df["window"] >= start_window) & (df["window"] <= end_window)].copy()


def forecast_value_columns(forecast: pd.DataFrame) -> list[str]:
    columns = []
    if "p_active" in forecast:
        columns.append("p_active")
    for policy in POLICIES:
        columns.extend(
            [
                f"{policy}_count",
                f"ceil_{policy}_count",
                f"alloc_{policy}_count",
            ]
        )
    return [column for column in columns if column in forecast]


def complete_forecast_frame(
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
    workflow: str,
    level: str,
    window_ms: int,
    start_window: int,
    end_window: int,
) -> pd.DataFrame:
    method = method_name(forecast)
    value_cols = forecast_value_columns(forecast)
    if not value_cols:
        value_cols = [f"{policy}_count" for policy in POLICIES]
        forecast = forecast.copy()
        for column in value_cols:
            forecast[column] = 0.0

    windows = list(range(start_window, end_window + 1))
    if level == "entry":
        grid = pd.DataFrame({"window": windows})
        grid["workflow_name"] = workflow
        grid["method"] = method
        grid["window_start_ms"] = grid["window"] * window_ms
        keys = ["window"]
    else:
        forecast_stages = set(forecast.get("stage_name", pd.Series(dtype=str)).dropna().astype(str))
        actual_stages = set(actual.get("stage_name", pd.Series(dtype=str)).dropna().astype(str))
        stages = sorted(forecast_stages | actual_stages)
        if not stages:
            stages = sorted(forecast_stages) or ["unknown"]
        grid = pd.MultiIndex.from_product(
            [stages, windows],
            names=["stage_name", "window"],
        ).to_frame(index=False)
        grid["workflow_name"] = workflow
        grid["method"] = method
        grid["window_start_ms"] = grid["window"] * window_ms
        keys = ["stage_name", "window"]

    use_cols = keys + value_cols
    existing_cols = [column for column in use_cols if column in forecast]
    forecast_values = forecast[existing_cols].copy()
    if not forecast_values.empty:
        forecast_values = forecast_values.groupby(keys, as_index=False).max(numeric_only=True)

    completed = grid.merge(forecast_values, on=keys, how="left")
    for column in value_cols:
        if column not in completed:
            completed[column] = 0.0
        completed[column] = completed[column].fillna(0.0)
    return completed


def allocated_count(row: pd.Series, policy: str) -> int:
    alloc_col = f"alloc_{policy}_count"
    ceil_col = f"ceil_{policy}_count"
    raw_col = f"{policy}_count"
    if alloc_col in row and not pd.isna(row[alloc_col]):
        return int(row[alloc_col])
    if ceil_col in row and not pd.isna(row[ceil_col]):
        return int(row[ceil_col])
    return ceil_count(float(row[raw_col]))


def build_detail(forecast: pd.DataFrame, actual: pd.DataFrame, level: str) -> pd.DataFrame:
    if level == "entry":
        merged = forecast.merge(actual, on="window", how="left")
        merged["stage_name"] = "__entry__"
    else:
        merged = forecast.merge(actual, on=["stage_name", "window"], how="left")
    merged["actual_count"] = merged["actual_count"].fillna(0).astype(int)

    rows = []
    for _, row in merged.iterrows():
        base = {
            "workflow_name": row["workflow_name"],
            "method": row.get("method", "unknown"),
            "level": level,
            "stage_name": row["stage_name"],
            "window": int(row["window"]),
            "window_start_ms": int(row["window_start_ms"]),
            "actual_count": int(row["actual_count"]),
        }
        if "p_active" in row:
            base["p_active"] = float(row.get("p_active", 1.0))
        for policy in POLICIES:
            raw_col = f"{policy}_count"
            if raw_col not in row:
                continue
            forecast_count = float(row[raw_col])
            alloc = allocated_count(row, policy)
            actual_count = int(row["actual_count"])
            rows.append(
                {
                    **base,
                    "policy": policy,
                    "forecast_count": forecast_count,
                    "allocated_count": alloc,
                    "absolute_error": abs(actual_count - forecast_count),
                    "under_count": max(0, actual_count - alloc),
                    "over_count": max(0, alloc - actual_count),
                    "covered": actual_count <= alloc,
                }
            )
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["workflow_name", "method", "level", "stage_name", "policy"]
    rows = []
    for keys, group in detail.groupby(group_cols):
        actual = group["actual_count"].astype(float)
        forecast = group["forecast_count"].astype(float)
        allocated = group["allocated_count"].astype(float)
        active = group[group["actual_count"] > 0]
        actual_total = float(actual.sum())
        allocated_total = float(allocated.sum())
        under_total = float(group["under_count"].sum())
        over_total = float(group["over_count"].sum())
        ordered = group.sort_values("window")
        allocated_int = ordered["allocated_count"].astype(int)
        scale_events = int(
            (allocated_int.iloc[0] != 0 if len(allocated_int) else 0)
            + (allocated_int.diff().fillna(0) != 0).sum()
        )
        window_ms = int(group["window_ms"].iloc[0]) if "window_ms" in group else 0
        allocated_replica_seconds = (
            float(allocated_total * window_ms / 1000.0) if window_ms > 0 else float("nan")
        )
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "windows": int(len(group)),
                "active_windows": int(len(active)),
                "actual_total": int(actual_total),
                "allocated_replica_windows": int(allocated_total),
                "allocated_replica_seconds": allocated_replica_seconds,
                "allocated_total": int(allocated_total),
                "under_total": int(under_total),
                "over_total": int(over_total),
                "coverage_rate": float(group["covered"].mean()),
                "quantile_hit_rate": float((actual <= forecast).mean()),
                "active_quantile_hit_rate": (
                    float((active["actual_count"] <= active["forecast_count"]).mean()) if len(active) else 1.0
                ),
                "active_coverage_rate": (
                    float(active["covered"].mean()) if len(active) else 1.0
                ),
                "demand_coverage_rate": (
                    float(1.0 - under_total / actual_total)
                    if actual_total > 0
                    else 1.0
                ),
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
                "peak_allocated": int(allocated.max()) if len(allocated) else 0,
                "scale_events": scale_events,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    window_ms = resolve_window_ms(args)
    trace = pd.read_csv(args.trace)
    forecast = pd.read_csv(args.forecast)
    eval_start_window, eval_end_window = resolve_eval_bounds(
        forecast,
        args.eval_start_window,
        args.eval_end_window,
    )

    if args.level == "entry":
        actual = actual_entry_counts(
            trace,
            args.workflow,
            window_ms,
            args.actual_entry_start_window,
            args.actual_entry_end_window,
        )
    else:
        actual = actual_stage_counts(
            trace,
            args.workflow,
            window_ms,
            args.actual_entry_start_window,
            args.actual_entry_end_window,
        )
    actual = filter_by_window(actual, eval_start_window, eval_end_window)
    forecast = filter_by_window(forecast, eval_start_window, eval_end_window)

    forecast = complete_forecast_frame(
        forecast,
        actual,
        args.workflow,
        args.level,
        window_ms,
        eval_start_window,
        eval_end_window,
    )
    detail = build_detail(forecast, actual, args.level)
    detail["window_ms"] = window_ms
    summary = summarize(detail)

    detail_out = Path(args.detail_out)
    summary_out = Path(args.summary_out)
    detail_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_out, index=False)
    summary.to_csv(summary_out, index=False)

    print(f"wrote {detail_out}")
    print(f"wrote {summary_out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

