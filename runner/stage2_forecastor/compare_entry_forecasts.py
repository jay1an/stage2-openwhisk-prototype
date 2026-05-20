import argparse
import json
from pathlib import Path

import pandas as pd

from .compare_stage_forecasts import (
    build_count_series,
    forecast_from_series,
    load_split,
    resolve_window_ms,
)
from .evaluate_forecast import (
    actual_entry_counts,
    build_detail,
    complete_forecast_frame,
    filter_by_window,
    summarize,
)
from ..workflow import load_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare entry forecasting baselines on a workflow trace."
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
    parser.add_argument(
        "--methods",
        default="ewma,burst-aware,hurdle-ewma,tsb,hazard-hurdle",
        help="comma-separated entry forecasting methods to compare",
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
    parser.add_argument("--out-dir", required=True, help="directory for summaries and metadata")
    parser.add_argument(
        "--write-forecast-csvs",
        action="store_true",
        help="also write per-method forecast CSVs",
    )
    parser.add_argument(
        "--write-detail",
        action="store_true",
        help="also write the window-level comparison detail CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        test_start_window = train_end_window + 1
    else:
        train_end_window = int(split_map[split_map["split"] == "train"]["entry_window"].max())
        test_start_window = int(split_map[split_map["split"] == "test"]["entry_window"].min())
    test_end_window = int(split_map[split_map["split"] == "test"]["entry_window"].max())
    horizon = test_end_window - train_end_window
    if horizon <= 0:
        raise ValueError("test horizon must be positive")

    entry_rows = trace[
        (trace["workflow_name"] == workflow_name)
        & (trace["stage_name"] == "__entry__")
        & (trace["status"] == "ok")
        & (trace["request_id"].isin(train_ids))
    ].copy()
    entry_rows["window"] = (entry_rows["entry_ts_ms"] // window_ms).astype(int)
    counts = build_count_series(entry_rows, "window", train_end_window)

    actual = actual_entry_counts(
        trace,
        workflow_name,
        window_ms,
        test_start_window,
        test_end_window,
    )
    eval_start_window = train_end_window + 1
    eval_end_window = test_end_window
    actual = filter_by_window(actual, eval_start_window, eval_end_window)

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    summaries = []
    all_detail = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        forecast["workflow_name"] = workflow_name
        forecast["window_start_ms"] = forecast["window"] * window_ms
        forecast = forecast[
            [
                "workflow_name",
                "method",
                "window",
                "window_start_ms",
                "p50_count",
                "p90_count",
                "p95_count",
                "ceil_p50_count",
                "ceil_p90_count",
                "ceil_p95_count",
                "alloc_p50_count",
                "alloc_p90_count",
                "alloc_p95_count",
            ]
        ].reset_index(drop=True)
        forecast = filter_by_window(forecast, eval_start_window, eval_end_window)
        completed = complete_forecast_frame(
            forecast,
            actual,
            workflow_name,
            "entry",
            window_ms,
            eval_start_window,
            eval_end_window,
        )
        detail = build_detail(completed, actual, "entry")
        detail["window_ms"] = window_ms
        summary = summarize(detail)
        summaries.append(summary)
        all_detail.append(detail)
        if args.write_forecast_csvs:
            forecast.to_csv(out_dir / f"{workflow_name}_{method}_entry_forecast.csv", index=False)

    compare_summary = pd.concat(summaries, ignore_index=True)
    compare_detail = pd.concat(all_detail, ignore_index=True)
    summary_path = out_dir / f"{workflow_name}_entry_compare_summary.csv"
    detail_path = out_dir / f"{workflow_name}_entry_compare_detail.csv"
    metadata_path = out_dir / f"{workflow_name}_entry_compare_metadata.json"

    compare_summary.to_csv(summary_path, index=False)
    if args.write_detail:
        compare_detail.to_csv(detail_path, index=False)
    metadata = {
        "workflow_name": workflow_name,
        "trace": args.trace,
        "workflow_config": args.workflow_config,
        "split_map": args.split_map,
        "split_strategy": args.split_strategy if args.split_map is None else "provided-map",
        "window_ms": window_ms,
        "methods": methods,
        "train_requests": len(train_ids),
        "test_requests": len(test_ids),
        "train_end_window": train_end_window,
        "test_start_window": test_start_window,
        "test_end_window": test_end_window,
        "eval_start_window": eval_start_window,
        "eval_end_window": eval_end_window,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote {summary_path}")
    if args.write_detail:
        print(f"wrote {detail_path}")
    print(compare_summary.to_string(index=False))


if __name__ == "__main__":
    main()

