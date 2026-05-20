import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .evaluate_forecast import (
    actual_stage_counts,
    build_detail,
    complete_forecast_frame,
    filter_by_window,
    summarize,
)
from .forecast_entry import (
    alloc_count,
    burst_groups,
    burst_aware_forecast,
    burst_localized_forecast,
    ceil_count,
    ewma,
    fip_fourier_forecast,
    hurdle_ewma_forecast,
    hazard_hurdle_forecast,
    estimate_burst_period,
    is_predicted_burst_window,
    recent_residual_quantile,
    tsb_forecast,
)
from ..workflow import load_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare stage-independent forecasting against workflow-entry forecasting plus DAG propagation."
        )
    )
    parser.add_argument("--trace", required=True, help="workflow trace CSV")
    parser.add_argument("--workflow-config", required=True, help="workflow YAML config")
    parser.add_argument(
        "--split-map",
        default=None,
        help="optional split CSV generated with the synthetic traces",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="used only when --split-map is not provided",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["request-count", "time"],
        default="request-count",
        help="used only when --split-map is not provided",
    )
    parser.add_argument("--window-sec", type=int, default=5)
    parser.add_argument(
        "--window-ms",
        type=int,
        default=None,
        help="override --window-sec with a millisecond-level window",
    )
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument(
        "--method",
        choices=[
            "ewma",
            "burst-aware",
            "burst-localized",
            "hurdle-ewma",
            "tsb",
            "hazard-hurdle",
            "fip-fourier",
        ],
        default="ewma",
    )
    parser.add_argument("--residual-window", type=int, default=60)
    parser.add_argument("--history-window", type=int, default=30)
    parser.add_argument("--burst-threshold", type=float, default=2.0)
    parser.add_argument("--burst-period-windows", type=int, default=None)
    parser.add_argument("--burst-width-windows", type=int, default=0)
    parser.add_argument("--background-count", type=float, default=None)
    parser.add_argument("--idle-zero-ratio", type=float, default=0.8)
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument("--out-dir", required=True, help="directory for forecasts, summaries, and metadata")
    parser.add_argument(
        "--write-detail",
        action="store_true",
        help="also write the window-level comparison detail CSV",
    )
    parser.add_argument(
        "--write-forecast-csvs",
        action="store_true",
        help="also write intermediate entry/stage forecast CSVs",
    )
    return parser.parse_args()


def resolve_window_ms(args: argparse.Namespace) -> int:
    if args.window_ms is not None:
        if args.window_ms <= 0:
            raise ValueError("--window-ms must be positive")
        return args.window_ms
    if args.window_sec <= 0:
        raise ValueError("--window-sec must be positive")
    return args.window_sec * 1000


def load_split(
    trace: pd.DataFrame,
    workflow_name: str,
    split_map_path: str | None,
    train_ratio: float,
    split_strategy: str = "request-count",
) -> tuple[list[str], list[str], pd.DataFrame]:
    entry_rows = (
        trace[
            (trace["workflow_name"] == workflow_name)
            & (trace["stage_name"] == "__entry__")
            & (trace["status"] == "ok")
        ][["request_id", "entry_ts_ms"]]
        .drop_duplicates()
        .sort_values(["entry_ts_ms", "request_id"])
        .reset_index(drop=True)
    )
    if entry_rows.empty:
        raise ValueError(f"no entry rows found for workflow={workflow_name}")

    if split_map_path:
        split_map = pd.read_csv(split_map_path).copy()
        if "split" not in split_map.columns:
            raise ValueError("--split-map must contain a split column")
        keep_cols = ["request_id", "split"]
        if "split_cutoff_ms" in split_map.columns:
            keep_cols.append("split_cutoff_ms")
        if "split_strategy" in split_map.columns:
            keep_cols.append("split_strategy")
        split_map = entry_rows.merge(split_map[keep_cols], on="request_id", how="left")
        if split_map["split"].isna().any():
            missing = int(split_map["split"].isna().sum())
            raise ValueError(f"split map is missing {missing} request_ids from the trace")
    else:
        split_map = entry_rows.copy()
        split_map["split"] = "test"
        if split_strategy == "request-count":
            split_idx = int(math.floor(len(entry_rows) * train_ratio))
            split_map.loc[: max(-1, split_idx - 1), "split"] = "train"
        elif split_strategy == "time":
            start = int(split_map["entry_ts_ms"].min())
            end = int(split_map["entry_ts_ms"].max())
            cutoff = start + int(math.floor((end - start) * train_ratio))
            split_map.loc[split_map["entry_ts_ms"] <= cutoff, "split"] = "train"
            split_map["split_cutoff_ms"] = cutoff
            split_map["split_strategy"] = "time"
        else:
            raise ValueError(f"unsupported split strategy: {split_strategy}")

    train_ids = split_map[split_map["split"] == "train"]["request_id"].tolist()
    test_ids = split_map[split_map["split"] == "test"]["request_id"].tolist()
    if not train_ids or not test_ids:
        raise ValueError("split must leave at least one train request and one test request")
    return train_ids, test_ids, split_map


def build_count_series(rows: pd.DataFrame, window_col: str, end_window: int) -> pd.Series:
    if rows.empty:
        return pd.Series([0.0], index=[end_window], dtype=float)
    counts = (
        rows.groupby(window_col)
        .size()
        .reindex(range(int(rows[window_col].min()), end_window + 1), fill_value=0)
        .astype(float)
    )
    return counts


def forecast_from_series(
    counts: pd.Series,
    method: str,
    alpha: float,
    residual_window: int,
    history_window: int,
    burst_threshold: float,
    burst_period_windows: int | None,
    burst_width_windows: int,
    background_count: float | None,
    idle_zero_ratio: float,
    activation_threshold: float,
    horizon: int,
    method_label: str,
) -> pd.DataFrame:
    count_values = counts.to_numpy(dtype=float)
    last_window = int(counts.index.max())
    localized_background = None
    localized_peak = None
    localized_period = None
    localized_last_peak_window = None

    if method == "ewma":
        base = ewma(count_values, alpha)
        p90_pad = recent_residual_quantile(count_values, base, 0.90, residual_window)
        p95_pad = recent_residual_quantile(count_values, base, 0.95, residual_window)
        p99_pad = recent_residual_quantile(count_values, base, 0.99, residual_window)
        base_values = (
            max(0.0, base),
            max(0.0, base + p90_pad),
            max(0.0, base + p95_pad),
            max(0.0, base + p99_pad),
        )
    elif method == "burst-aware":
        base_values = burst_aware_forecast(
            count_values,
            alpha,
            history_window,
            burst_threshold,
            idle_zero_ratio,
        )
    elif method == "hurdle-ewma":
        base_values = hurdle_ewma_forecast(
            count_values,
            alpha,
            residual_window,
            history_window,
        )
    elif method == "tsb":
        base_values = tsb_forecast(
            count_values,
            alpha,
            history_window,
        )
    elif method == "fip-fourier":
        base_values = None
    elif method == "hazard-hurdle":
        base_values = None
    elif method == "burst-localized":
        base_values = None
        if background_count is None:
            recent = count_values[-max(1, min(len(count_values), history_window)) :]
            zero_ratio = float((recent <= 0).mean()) if len(recent) else 1.0
            localized_background = 0.0 if zero_ratio >= idle_zero_ratio else ewma(count_values, alpha)
        else:
            localized_background = background_count
        localized_background = max(0.0, float(localized_background))
        groups = burst_groups(counts, burst_threshold)
        if groups:
            localized_period = burst_period_windows or estimate_burst_period(groups)
            localized_last_peak_window = int(groups[-1]["peak_window"])
            localized_peak = max(float(group["peak_count"]) for group in groups)
    else:
        raise ValueError(f"unsupported method: {method}")

    rows = []
    for step in range(1, horizon + 1):
        target_window = last_window + step
        p_active = 1.0
        if method == "hazard-hurdle":
            p50, p90, p95, p99, p_active = hazard_hurdle_forecast(
                count_values,
                alpha,
                history_window,
                horizon_step=step,
            )
        elif method == "fip-fourier":
            p50, p90, p95, p99 = fip_fourier_forecast(
                count_values,
                horizon_step=step,
                local_window=max(60, history_window),
                harmonics=10,
                residual_window=residual_window,
            )
        elif method == "burst-localized":
            if localized_peak is None or localized_period is None or localized_period <= 0:
                p50 = p90 = p95 = p99 = localized_background
            elif is_predicted_burst_window(
                target_window,
                localized_last_peak_window,
                localized_period,
                max(0, burst_width_windows),
            ):
                nonzero = count_values[count_values > 0]
                nonzero_mean = float(nonzero.mean()) if len(nonzero) else localized_background
                p50 = max(localized_background, min(localized_peak, nonzero_mean))
                p90 = max(p50, localized_peak)
                p95 = p90
                p99 = p90
            else:
                p50 = p90 = p95 = p99 = localized_background
        else:
            p50, p90, p95, p99 = base_values

        rows.append(
            {
                "method": method_label,
                "window": target_window,
                "p_active": p_active,
                "p50_count": p50,
                "p90_count": p90,
                "p95_count": p95,
                "p99_count": p99,
                "ceil_p50_count": ceil_count(p50),
                "ceil_p90_count": ceil_count(p90),
                "ceil_p95_count": ceil_count(p95),
                "ceil_p99_count": ceil_count(p99),
                "alloc_p50_count": alloc_count(p50, activation_threshold),
                "alloc_p90_count": alloc_count(p90, activation_threshold),
                "alloc_p95_count": alloc_count(p95, activation_threshold),
                "alloc_p99_count": alloc_count(p99, activation_threshold),
            }
        )
    return pd.DataFrame(rows)


def build_delay_kernel(stage_rows: pd.DataFrame, window_ms: int) -> dict[int, float]:
    delay_ms = stage_rows["dispatch_start_ms"].astype(float) - stage_rows["entry_ts_ms"].astype(float)
    offsets = ((delay_ms.clip(lower=0)) // window_ms).astype(int)
    counts = offsets.value_counts().sort_index()
    total = int(counts.sum())
    if total <= 0:
        return {0: 1.0}
    return {int(offset): float(count / total) for offset, count in counts.items()}


def propagate_entry_forecast(
    workflow_name: str,
    workflow,
    entry_forecast: pd.DataFrame,
    train_stage_rows: pd.DataFrame,
    window_ms: int,
) -> pd.DataFrame:
    rows = []
    for stage_name in workflow.nodes:
        stage_rows = train_stage_rows[train_stage_rows["stage_name"] == stage_name].copy()
        if stage_rows.empty:
            kernel = {0: 1.0}
        else:
            kernel = build_delay_kernel(stage_rows, window_ms)

        for _, forecast_row in entry_forecast.iterrows():
            for offset, probability in kernel.items():
                target_window = int(forecast_row["window"]) + offset
                rows.append(
                    {
                        "workflow_name": workflow_name,
                        "method": str(forecast_row["method"]),
                        "stage_name": stage_name,
                        "window": target_window,
                        "window_start_ms": target_window * window_ms,
                        "p_active": float(forecast_row.get("p_active", 1.0)),
                        "p50_count": float(forecast_row["p50_count"]) * probability,
                        "p90_count": float(forecast_row["p90_count"]) * probability,
                        "p95_count": float(forecast_row["p95_count"]) * probability,
                        "p99_count": float(forecast_row["p99_count"]) * probability,
                    }
                )

    out = (
        pd.DataFrame(rows)
        .groupby(["workflow_name", "method", "stage_name", "window", "window_start_ms"], as_index=False)[
            [
                "p_active",
                "p50_count",
                "p90_count",
                "p95_count",
                "p99_count",
            ]
        ]
        .agg(
            {
                "p_active": "max",
                "p50_count": "sum",
                "p90_count": "sum",
                "p95_count": "sum",
                "p99_count": "sum",
            }
        )
    )
    for policy in ["p50", "p90", "p95", "p99"]:
        out[f"ceil_{policy}_count"] = out[f"{policy}_count"].map(ceil_count)
    return out


def build_independent_stage_forecast(
    workflow_name: str,
    workflow,
    train_stage_rows: pd.DataFrame,
    train_end_window: int,
    horizon: int,
    window_ms: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for stage_name in workflow.nodes:
        stage_rows = train_stage_rows[train_stage_rows["stage_name"] == stage_name].copy()
        if stage_rows.empty:
            counts = pd.Series([0.0], index=[train_end_window], dtype=float)
        else:
            stage_rows["window"] = (stage_rows["dispatch_start_ms"] // window_ms).astype(int)
            stage_rows = stage_rows[stage_rows["window"] <= train_end_window].copy()
            counts = build_count_series(stage_rows, "window", train_end_window)

        forecast = forecast_from_series(
            counts=counts,
            method=args.method,
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
            method_label=f"independent-{args.method}",
        )
        forecast["workflow_name"] = workflow_name
        forecast["stage_name"] = stage_name
        forecast["window_start_ms"] = forecast["window"] * window_ms
        rows.append(forecast)

    out = pd.concat(rows, ignore_index=True)
    return out[
        [
            "workflow_name",
            "method",
            "stage_name",
            "window",
            "window_start_ms",
            "p_active",
            "p50_count",
            "p90_count",
            "p95_count",
            "p99_count",
            "ceil_p50_count",
            "ceil_p90_count",
            "ceil_p95_count",
            "ceil_p99_count",
            "alloc_p50_count",
            "alloc_p90_count",
            "alloc_p95_count",
            "alloc_p99_count",
        ]
    ].reset_index(drop=True)


def evaluate_stage_forecast(
    trace: pd.DataFrame,
    workflow_name: str,
    forecast: pd.DataFrame,
    window_ms: int,
    test_start_window: int,
    test_end_window: int,
    eval_start_window: int,
    eval_end_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = actual_stage_counts(
        trace,
        workflow_name,
        window_ms,
        test_start_window,
        test_end_window,
    )
    actual = filter_by_window(actual, eval_start_window, eval_end_window)
    forecast = filter_by_window(forecast, eval_start_window, eval_end_window)
    completed = complete_forecast_frame(
        forecast,
        actual,
        workflow_name,
        "stage",
        window_ms,
        eval_start_window,
        eval_end_window,
    )
    detail = build_detail(completed, actual, "stage")
    detail["window_ms"] = window_ms
    summary = summarize(detail)
    return detail, summary


def main() -> None:
    args = parse_args()
    window_ms = resolve_window_ms(args)
    trace = pd.read_csv(args.trace)
    workflow = load_workflow(args.workflow_config)
    workflow_name = workflow.workflow_name

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
        test_start_window = train_end_window + 1
    else:
        train_end_window = int(split_map[split_map["split"] == "train"]["entry_window"].max())
        test_start_window = int(split_map[split_map["split"] == "test"]["entry_window"].min())
    test_end_window = int(split_map[split_map["split"] == "test"]["entry_window"].max())
    entry_horizon = test_end_window - train_end_window
    if entry_horizon <= 0:
        raise ValueError("test horizon must be positive")

    workflow_rows = trace[
        (trace["workflow_name"] == workflow_name)
        & (trace["status"] == "ok")
    ].copy()
    train_entry_rows = workflow_rows[
        (workflow_rows["stage_name"] == "__entry__")
        & (workflow_rows["request_id"].isin(train_ids))
    ].copy()
    train_entry_rows["window"] = (train_entry_rows["entry_ts_ms"] // window_ms).astype(int)
    train_entry_counts = build_count_series(train_entry_rows, "window", train_end_window)

    entry_forecast = forecast_from_series(
        counts=train_entry_counts,
        method=args.method,
        alpha=args.alpha,
        residual_window=args.residual_window,
        history_window=args.history_window,
        burst_threshold=args.burst_threshold,
        burst_period_windows=args.burst_period_windows,
        burst_width_windows=args.burst_width_windows,
        background_count=args.background_count,
        idle_zero_ratio=args.idle_zero_ratio,
        activation_threshold=args.activation_threshold,
        horizon=entry_horizon,
        method_label=f"dag-{args.method}",
    )
    entry_forecast["workflow_name"] = workflow_name
    entry_forecast["window_start_ms"] = entry_forecast["window"] * window_ms
    entry_forecast = entry_forecast[
        [
            "workflow_name",
            "method",
            "window",
            "window_start_ms",
            "p_active",
            "p50_count",
            "p90_count",
            "p95_count",
            "p99_count",
            "ceil_p50_count",
            "ceil_p90_count",
            "ceil_p95_count",
            "ceil_p99_count",
            "alloc_p50_count",
            "alloc_p90_count",
            "alloc_p95_count",
            "alloc_p99_count",
        ]
    ].reset_index(drop=True)

    train_stage_rows = workflow_rows[
        (workflow_rows["stage_name"] != "__entry__")
        & (workflow_rows["request_id"].isin(train_ids))
    ].copy()
    stage_forecast_dag = propagate_entry_forecast(
        workflow_name=workflow_name,
        workflow=workflow,
        entry_forecast=entry_forecast,
        train_stage_rows=train_stage_rows,
        window_ms=window_ms,
    )
    for policy in ["p50", "p90", "p95", "p99"]:
        stage_forecast_dag[f"alloc_{policy}_count"] = stage_forecast_dag[f"{policy}_count"].map(
            lambda value: alloc_count(value, args.activation_threshold)
        )

    test_stage_rows = workflow_rows[
        (workflow_rows["stage_name"] != "__entry__")
        & (workflow_rows["request_id"].isin(test_ids))
    ].copy()
    test_stage_rows["window"] = (test_stage_rows["dispatch_start_ms"] // window_ms).astype(int)
    stage_eval_end_window = int(test_stage_rows["window"].max())
    stage_horizon = stage_eval_end_window - train_end_window
    if stage_horizon <= 0:
        raise ValueError("stage horizon must be positive")

    stage_forecast_independent = build_independent_stage_forecast(
        workflow_name=workflow_name,
        workflow=workflow,
        train_stage_rows=train_stage_rows,
        train_end_window=train_end_window,
        horizon=stage_horizon,
        window_ms=window_ms,
        args=args,
    )

    eval_start_window = train_end_window + 1
    eval_end_window = stage_eval_end_window
    dag_detail, dag_summary = evaluate_stage_forecast(
        trace,
        workflow_name,
        stage_forecast_dag,
        window_ms,
        test_start_window,
        test_end_window,
        eval_start_window,
        eval_end_window,
    )
    ind_detail, ind_summary = evaluate_stage_forecast(
        trace,
        workflow_name,
        stage_forecast_independent,
        window_ms,
        test_start_window,
        test_end_window,
        eval_start_window,
        eval_end_window,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entry_forecast_path = out_dir / f"{workflow_name}_{args.method}_entry_forecast.csv"
    dag_forecast_path = out_dir / f"{workflow_name}_{args.method}_dag_stage_forecast.csv"
    independent_forecast_path = out_dir / f"{workflow_name}_{args.method}_independent_stage_forecast.csv"
    detail_path = out_dir / f"{workflow_name}_{args.method}_stage_compare_detail.csv"
    summary_path = out_dir / f"{workflow_name}_{args.method}_stage_compare_summary.csv"
    metadata_path = out_dir / f"{workflow_name}_{args.method}_compare_metadata.json"

    if args.write_forecast_csvs:
        entry_forecast.to_csv(entry_forecast_path, index=False)
        stage_forecast_dag.to_csv(dag_forecast_path, index=False)
        stage_forecast_independent.to_csv(independent_forecast_path, index=False)
    compare_detail = pd.concat([dag_detail, ind_detail], ignore_index=True)
    compare_detail["target_window"] = compare_detail["window"]
    compare_detail["origin_window"] = compare_detail["window"]
    compare_summary = pd.concat([dag_summary, ind_summary], ignore_index=True)
    if args.write_detail:
        compare_detail.to_csv(detail_path, index=False)
    compare_summary.to_csv(summary_path, index=False)

    metadata = {
        "workflow_name": workflow_name,
        "trace": args.trace,
        "workflow_config": args.workflow_config,
        "split_map": args.split_map,
        "split_strategy": args.split_strategy if args.split_map is None else "provided-map",
        "window_ms": window_ms,
        "method": args.method,
        "train_requests": len(train_ids),
        "test_requests": len(test_ids),
        "train_end_window": train_end_window,
        "test_start_window": test_start_window,
        "test_end_window": test_end_window,
        "eval_start_window": eval_start_window,
        "eval_end_window": eval_end_window,
        "stage_eval_end_window": stage_eval_end_window,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.write_forecast_csvs:
        print(f"wrote {entry_forecast_path}")
        print(f"wrote {dag_forecast_path}")
        print(f"wrote {independent_forecast_path}")
    if args.write_detail:
        print(f"wrote {detail_path}")
    print(f"wrote {summary_path}")
    print(compare_summary.to_string(index=False))


if __name__ == "__main__":
    main()

