import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from .compare_entry_twostage_forecasts import (
    POLICIES,
    add_forecast_columns,
    apply_count_calibration_shifts,
    build_entry_counts,
    build_prediction_frame,
    build_supervised_frame,
    choose_fit_calibration_windows,
    choose_gate_thresholds,
    compute_count_calibration_shifts,
    feature_columns,
    fit_active_model,
    fit_conditional_models,
    hazard_active_probability,
    predict_active_probability,
    predict_conditional_quantiles,
    recent_empirical_quantiles,
)
from .compare_stage_forecasts import (
    load_split,
    propagate_entry_forecast,
    resolve_window_ms,
)
from .compare_stage_rolling_forecasts import (
    actual_stage_by_window,
    add_alloc_columns,
    allocated_count,
    forecast_lookup,
    summarize,
    window_series,
)
from ..workflow import load_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rolling-origin stage comparison for two-stage entry forecasting "
            "followed by DAG propagation."
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
        default="twostage-gbdt,twostage-gbdt-calibrated,twostage-hazard,twostage-hybrid-calibrated",
    )
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument(
        "--calibration-mode",
        choices=["coverage-first", "cost-aware"],
        default="cost-aware",
    )
    parser.add_argument(
        "--count-calibration-mode",
        choices=["none", "conformal", "cost-aware"],
        default="none",
    )
    parser.add_argument("--under-cost", type=float, default=10.0)
    parser.add_argument("--over-cost", type=float, default=1.0)
    parser.add_argument("--calibration-grid-size", type=int, default=80)
    parser.add_argument("--count-calibration-grid-size", type=int, default=81)
    parser.add_argument("--max-positive-weight", type=float, default=30.0)
    parser.add_argument("--min-positive-regression-rows", type=int, default=8)
    parser.add_argument("--recent-positive-window", type=int, default=240)
    parser.add_argument("--hazard-alpha", type=float, default=0.20)
    parser.add_argument("--hazard-bandwidth-windows", type=int, default=2)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--write-detail", action="store_true")
    parser.add_argument("--write-forecast-csvs", action="store_true")
    return parser.parse_args()


def split_windows(split_map: pd.DataFrame, window_ms: int) -> tuple[int, int, int]:
    split_map = split_map.copy()
    split_map["entry_window"] = (split_map["entry_ts_ms"] // window_ms).astype(int)
    if "split_cutoff_ms" in split_map.columns and split_map["split_cutoff_ms"].notna().any():
        train_end_window = int(split_map["split_cutoff_ms"].dropna().iloc[0] // window_ms)
        eval_start_window = train_end_window + 1
    else:
        train_end_window = int(split_map[split_map["split"] == "train"]["entry_window"].max())
        eval_start_window = int(split_map[split_map["split"] == "test"]["entry_window"].min())
    eval_end_window = int(split_map[split_map["split"] == "test"]["entry_window"].max())
    return train_end_window, eval_start_window, eval_end_window


def build_observed_counts(
    full_counts: pd.Series,
    origin_window: int,
    eval_end_window: int,
) -> pd.Series:
    counts = full_counts.copy()
    counts.loc[counts.index > origin_window] = 0.0
    if counts.index.max() < eval_end_window:
        counts = counts.reindex(range(int(counts.index.min()), eval_end_window + 1), fill_value=0.0)
    return counts.astype(float)


def fit_and_predict_origin(
    counts_seen: pd.Series,
    first_window: int,
    origin_window: int,
    horizon: int,
    window_ms: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    train_frame = build_supervised_frame(
        counts_seen,
        first_window + 1,
        origin_window,
        first_window,
        window_ms,
    )
    if train_frame.empty:
        raise ValueError("not enough history to fit two-stage entry model")

    train_windows = train_frame["window"].to_numpy(dtype=int)
    fit_end_window, calibration_start_window = choose_fit_calibration_windows(
        train_windows, args.calibration_fraction
    )
    fit_frame = train_frame[train_frame["window"] <= fit_end_window].copy()
    calibration_frame = train_frame[train_frame["window"] >= calibration_start_window].copy()
    if fit_frame.empty:
        fit_frame = train_frame.copy()
    if calibration_frame.empty:
        calibration_frame = train_frame.copy()

    features = feature_columns(train_frame)
    active_model = fit_active_model(fit_frame, features, args)
    conditional_models = fit_conditional_models(fit_frame, features, args)

    cal_active = predict_active_probability(active_model, calibration_frame)
    cal_conditional = predict_conditional_quantiles(
        conditional_models, calibration_frame, counts_seen, args
    )
    cal_hazard_active = np.asarray(
        [hazard_active_probability(counts_seen, int(w), args) for w in calibration_frame["window"]],
        dtype=float,
    )
    cal_hazard_conditional = {
        policy: np.asarray(
            [
                recent_empirical_quantiles(counts_seen, int(w), args.recent_positive_window)[policy]
                for w in calibration_frame["window"]
            ],
            dtype=float,
        )
        for policy in POLICIES
    }
    cal_hybrid_active = np.maximum(cal_active, cal_hazard_active)
    cal_hybrid_conditional = {
        policy: np.maximum(cal_conditional[policy], cal_hazard_conditional[policy])
        for policy in POLICIES
    }
    calibration = calibration_frame[["window", "target_count", "target_active"]].copy()
    calibration["gbdt_p_active"] = cal_active
    calibration["hybrid_p_active"] = cal_hybrid_active
    for policy in POLICIES:
        calibration[f"gbdt_conditional_{policy}"] = cal_conditional[policy]
        calibration[f"hybrid_conditional_{policy}"] = cal_hybrid_conditional[policy]

    thresholds = {
        "gbdt": choose_gate_thresholds(
            calibration,
            p_active_col="gbdt_p_active",
            conditional_prefix="gbdt_conditional",
            grid_size=args.calibration_grid_size,
            activation_threshold=args.activation_threshold,
            calibration_mode=args.calibration_mode,
            under_cost=args.under_cost,
            over_cost=args.over_cost,
        ),
        "hybrid": choose_gate_thresholds(
            calibration,
            p_active_col="hybrid_p_active",
            conditional_prefix="hybrid_conditional",
            grid_size=args.calibration_grid_size,
            activation_threshold=args.activation_threshold,
            calibration_mode=args.calibration_mode,
            under_cost=args.under_cost,
            over_cost=args.over_cost,
        ),
    }

    calibration_forecast = add_forecast_columns(
        calibration_frame,
        p_active=cal_active,
        hazard_p_active=cal_hazard_active,
        conditional=cal_conditional,
        hazard_conditional=cal_hazard_conditional,
        thresholds=thresholds,
    )
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    count_shifts = compute_count_calibration_shifts(
        calibration_forecast=calibration_forecast,
        methods=methods,
        activation_threshold=args.activation_threshold,
        mode=args.count_calibration_mode,
        under_cost=args.under_cost,
        over_cost=args.over_cost,
        grid_size=args.count_calibration_grid_size,
    )

    eval_frame = build_prediction_frame(
        counts_seen,
        origin_window + 1,
        origin_window + horizon,
        first_window,
        window_ms,
    )
    p_active = predict_active_probability(active_model, eval_frame)
    conditional = predict_conditional_quantiles(conditional_models, eval_frame, counts_seen, args)
    hazard_active = np.asarray(
        [hazard_active_probability(counts_seen, int(w), args) for w in eval_frame["window"]],
        dtype=float,
    )
    hazard_conditional = {
        policy: np.asarray(
            [
                recent_empirical_quantiles(counts_seen, int(w), args.recent_positive_window)[policy]
                for w in eval_frame["window"]
            ],
            dtype=float,
        )
        for policy in POLICIES
    }
    forecast = add_forecast_columns(
        eval_frame,
        p_active=p_active,
        hazard_p_active=hazard_active,
        conditional=conditional,
        hazard_conditional=hazard_conditional,
        thresholds=thresholds,
    )
    forecast = apply_count_calibration_shifts(forecast, methods, count_shifts)
    forecast["origin_window"] = origin_window
    return forecast


def entry_forecast_for_method(
    forecast: pd.DataFrame,
    workflow_name: str,
    method: str,
    window_ms: int,
) -> pd.DataFrame:
    rows = []
    for _, row in forecast.iterrows():
        out = {
            "workflow_name": workflow_name,
            "method": f"dag-{method}",
            "window": int(row["window"]),
            "window_start_ms": int(row["window"]) * window_ms,
        }
        for policy in POLICIES:
            out[f"{policy}_count"] = float(row[f"{method}_{policy}"])
        # The shared propagation helper still expects p99; mirror p95 but do not
        # evaluate or report it in this proposal-facing script.
        out["p99_count"] = out["p95_count"]
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.horizon_windows <= 0:
        raise ValueError("--horizon-windows must be positive")
    step = args.origin_step_windows or args.horizon_windows
    if step <= 0:
        raise ValueError("--origin-step-windows must be positive")

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    supported = {
        "twostage-gbdt",
        "twostage-gbdt-calibrated",
        "twostage-hazard",
        "twostage-hybrid-calibrated",
    }
    unsupported = sorted(set(methods) - supported)
    if unsupported:
        raise ValueError(f"unsupported methods: {unsupported}")

    window_ms = resolve_window_ms(args)
    workflow = load_workflow(args.workflow_config)
    workflow_name = workflow.workflow_name
    trace = pd.read_csv(args.trace)
    _, test_ids, split_map = load_split(
        trace,
        workflow_name,
        args.split_map,
        args.train_ratio,
        args.split_strategy,
    )
    train_end_window, eval_start_window, eval_end_window = split_windows(split_map, window_ms)

    workflow_rows = trace[
        (trace["workflow_name"] == workflow_name)
        & (trace["status"] == "ok")
    ].copy()
    first_window = int(workflow_rows["entry_ts_ms"].min() // window_ms)
    full_counts = build_entry_counts(
        trace,
        workflow_name,
        window_ms,
        first_window,
        eval_end_window,
    )

    stage_rows = workflow_rows[workflow_rows["stage_name"] != "__entry__"].copy()
    stage_rows["dispatch_window"] = window_series(stage_rows, "dispatch_start_ms", window_ms)
    test_stage_rows = stage_rows[stage_rows["request_id"].isin(test_ids)].copy()
    if test_stage_rows.empty:
        raise ValueError("no test stage rows found")
    eval_end_window = min(eval_end_window, int(test_stage_rows["dispatch_window"].max()))
    actual_by_stage_window = actual_stage_by_window(test_stage_rows, window_ms)

    detail_rows = []
    forecast_frames = []
    last_origin = eval_end_window - 1
    for origin_window in range(train_end_window, last_origin + 1, step):
        horizon = min(args.horizon_windows, eval_end_window - origin_window)
        if horizon <= 0:
            continue

        counts_seen = build_observed_counts(full_counts, origin_window, eval_end_window)
        origin_forecast = fit_and_predict_origin(
            counts_seen=counts_seen,
            first_window=first_window,
            origin_window=origin_window,
            horizon=horizon,
            window_ms=window_ms,
            args=args,
        )
        history_stage_rows = stage_rows[stage_rows["dispatch_window"] <= origin_window].copy()

        for method in methods:
            entry_forecast = entry_forecast_for_method(
                origin_forecast,
                workflow_name=workflow_name,
                method=method,
                window_ms=window_ms,
            )
            stage_forecast = propagate_entry_forecast(
                workflow_name=workflow_name,
                workflow=workflow,
                entry_forecast=entry_forecast,
                train_stage_rows=history_stage_rows,
                window_ms=window_ms,
            )
            stage_forecast = add_alloc_columns(stage_forecast, args.activation_threshold)
            stage_forecast["origin_window"] = origin_window
            forecast_frames.append(stage_forecast)

            lookup = forecast_lookup(stage_forecast)
            method_name = f"dag-{method}"
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
    summary_path = out_dir / f"{workflow_name}_twostage_rolling_stage_compare_summary.csv"
    summary_by_stage_path = out_dir / f"{workflow_name}_twostage_rolling_stage_compare_by_stage.csv"
    detail_path = out_dir / f"{workflow_name}_twostage_rolling_stage_compare_detail.csv"
    forecast_path = out_dir / f"{workflow_name}_twostage_rolling_stage_forecasts.csv"
    metadata_path = out_dir / f"{workflow_name}_twostage_rolling_stage_compare_metadata.json"

    summary.to_csv(summary_path, index=False)
    summary_by_stage.to_csv(summary_by_stage_path, index=False)
    if args.write_detail:
        detail.to_csv(detail_path, index=False)
    if args.write_forecast_csvs and forecast_frames:
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
        "policies": list(POLICIES.keys()),
        "calibration_mode": args.calibration_mode,
        "count_calibration_mode": args.count_calibration_mode,
        "under_cost": args.under_cost,
        "over_cost": args.over_cost,
        "train_end_window": train_end_window,
        "eval_start_window": eval_start_window,
        "eval_end_window": eval_end_window,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote {summary_path}")
    print(f"wrote {summary_by_stage_path}")
    if args.write_detail:
        print(f"wrote {detail_path}")
    if args.write_forecast_csvs and forecast_frames:
        print(f"wrote {forecast_path}")
    print(f"wrote {metadata_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

