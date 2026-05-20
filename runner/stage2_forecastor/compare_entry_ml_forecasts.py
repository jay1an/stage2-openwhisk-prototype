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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .compare_stage_forecasts import load_split, resolve_window_ms
from ..workflow import load_workflow


POLICIES = ["p50", "p90", "p95"]
QUANTILES = {"p50": 0.50, "p90": 0.90, "p95": 0.95}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrained ML quantile baselines for workflow entry arrival forecasting."
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--split-map", required=True)
    parser.add_argument("--window-sec", type=int, default=60)
    parser.add_argument("--window-ms", type=int, default=None)
    parser.add_argument(
        "--methods",
        default="hgb-quantile,hgb-hurdle",
        help="comma-separated methods: hgb-quantile,hgb-hurdle,lightgbm-quantile",
    )
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-lag-windows",
        type=int,
        default=1440,
        help=(
            "number of initial windows reserved for lag/rolling features; "
            "use a smaller value for compressed short traces"
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--write-detail", action="store_true")
    parser.add_argument("--write-forecast-csv", action="store_true")
    return parser.parse_args()


def build_entry_counts(
    trace: pd.DataFrame,
    workflow_name: str,
    window_ms: int,
) -> pd.Series:
    entries = trace[
        (trace["workflow_name"] == workflow_name)
        & (trace["stage_name"] == "__entry__")
        & (trace["status"] == "ok")
    ].copy()
    entries["window"] = (entries["entry_ts_ms"] // window_ms).astype(int)
    first = int(entries["window"].min())
    last = int(entries["window"].max())
    return (
        entries.groupby("window")
        .size()
        .reindex(range(first, last + 1), fill_value=0)
        .astype(float)
    )


def consecutive_active_length(values: np.ndarray) -> int:
    length = 0
    for value in values[::-1]:
        if value <= 0:
            break
        length += 1
    return length


def time_since_active(values: np.ndarray, cap: int) -> int:
    active = np.flatnonzero(values > 0)
    if len(active) == 0:
        return cap
    return min(cap, len(values) - 1 - int(active[-1]))


def build_supervised_frame(counts: pd.Series, window_ms: int, max_lag: int) -> pd.DataFrame:
    values = counts.to_numpy(dtype=float)
    windows = counts.index.to_numpy(dtype=int)
    if max_lag <= 0:
        raise ValueError("--max-lag-windows must be positive")
    if len(values) <= max_lag + 1:
        max_lag = max(1, min(max_lag, len(values) // 3))
    rows = []
    for idx in range(max_lag, len(values)):
        history = values[:idx]
        target = float(values[idx])

        def lag(offset: int) -> float:
            if idx >= offset:
                return float(values[idx - offset])
            return float(values[0])

        row = {
            "window": int(windows[idx]),
            "target_count": target,
            "target_active": 1 if target > 0 else 0,
            "lag_1": lag(1),
            "lag_2": lag(2),
            "lag_3": lag(3),
            "lag_5": lag(5),
            "lag_10": lag(10),
            "lag_30": lag(30),
            "lag_60": lag(60),
            "active_run_length": consecutive_active_length(history[-120:]),
            "time_since_active": time_since_active(history, cap=max_lag),
        }
        for size in [3, 5, 10, 30, 60, 240, 1440]:
            recent = history[-size:]
            row[f"roll_sum_{size}"] = float(np.sum(recent))
            row[f"roll_mean_{size}"] = float(np.mean(recent))
            row[f"roll_max_{size}"] = float(np.max(recent))
            row[f"zero_ratio_{size}"] = float(np.mean(recent == 0))

        # The absolute date is arbitrary, but modulo features preserve repeated daily patterns.
        minutes = (int(windows[idx]) * window_ms // 60000)
        minute_of_day = minutes % 1440
        row["tod_sin"] = math.sin(2.0 * math.pi * minute_of_day / 1440.0)
        row["tod_cos"] = math.cos(2.0 * math.pi * minute_of_day / 1440.0)
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in {"window", "target_count", "target_active"}
    ]


def ceil_count(value: float) -> int:
    return int(math.ceil(max(0.0, value)))


def alloc_count(value: float, activation_threshold: float) -> int:
    if value < activation_threshold:
        return 0
    return ceil_count(value)


def summarize(detail: pd.DataFrame, window_ms: int) -> pd.DataFrame:
    rows = []
    for (workflow_name, method, policy), group in detail.groupby(["workflow_name", "method", "policy"]):
        actual = group["actual_count"].astype(float)
        forecast = group["forecast_count"].astype(float)
        allocated = group["allocated_count"].astype(float)
        active = group[group["actual_count"] > 0]
        actual_total = float(actual.sum())
        allocated_total = float(allocated.sum())
        under_total = float(group["under_count"].sum())
        over_total = float(group["over_count"].sum())
        rows.append(
            {
                "workflow_name": workflow_name,
                "method": method,
                "policy": policy,
                "forecast_rows": int(len(group)),
                "active_rows": int(len(active)),
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
                "mae": float(np.mean(np.abs(actual - forecast))),
                "rmse": float(np.sqrt(np.mean((actual - forecast) ** 2))),
                "max_actual": int(actual.max()) if len(actual) else 0,
                "max_allocated": int(allocated.max()) if len(allocated) else 0,
            }
        )
    return pd.DataFrame(rows)


def fit_hgb_quantile_models(
    train: pd.DataFrame,
    features: list[str],
    args: argparse.Namespace,
) -> dict[str, HistGradientBoostingRegressor]:
    models = {}
    x_train = train[features]
    y_train = train["target_count"].astype(float)
    sample_weight = np.where(train["target_count"].to_numpy(dtype=float) > 0, 8.0, 1.0)
    for policy, quantile in QUANTILES.items():
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            l2_regularization=args.l2_regularization,
            random_state=args.random_state,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        models[policy] = model
    return models


def fit_hgb_hurdle_models(
    train: pd.DataFrame,
    features: list[str],
    args: argparse.Namespace,
) -> tuple[HistGradientBoostingClassifier | None, dict[str, HistGradientBoostingRegressor]]:
    x_train = train[features]
    y_active = train["target_active"].astype(int)
    if y_active.nunique() == 1:
        classifier = None
    else:
        classifier = HistGradientBoostingClassifier(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            l2_regularization=args.l2_regularization,
            random_state=args.random_state,
        )
        active_weight = np.where(y_active.to_numpy() > 0, 12.0, 1.0)
        classifier.fit(x_train, y_active, sample_weight=active_weight)

    positive = train[train["target_count"] > 0].copy()
    if positive.empty:
        return classifier, {}
    regressors = {}
    for policy, quantile in QUANTILES.items():
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            l2_regularization=args.l2_regularization,
            random_state=args.random_state,
        )
        model.fit(positive[features], positive["target_count"].astype(float))
        regressors[policy] = model
    return classifier, regressors


def predict_hgb_quantile(
    models: dict[str, HistGradientBoostingRegressor],
    test: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    forecast = test[["window", "target_count"]].copy()
    for policy in POLICIES:
        forecast[f"{policy}_count"] = np.maximum(0.0, models[policy].predict(test[features]))
    forecast["p90_count"] = np.maximum(forecast["p90_count"], forecast["p50_count"])
    forecast["p95_count"] = np.maximum(forecast["p95_count"], forecast["p90_count"])
    return forecast


def fit_lightgbm_quantile_models(
    train: pd.DataFrame,
    features: list[str],
    args: argparse.Namespace,
):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError(
            "lightgbm-quantile requires LightGBM. Install it with `pip install lightgbm`."
        ) from exc

    models = {}
    x_train = train[features]
    y_train = train["target_count"].astype(float)
    sample_weight = np.where(train["target_count"].to_numpy(dtype=float) > 0, 8.0, 1.0)
    for policy, quantile in QUANTILES.items():
        model = LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=args.max_iter,
            learning_rate=args.learning_rate,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=args.l2_regularization,
            random_state=args.random_state,
            verbosity=-1,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        models[policy] = model
    return models


def predict_lightgbm_quantile(
    models,
    test: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    forecast = test[["window", "target_count"]].copy()
    for policy in POLICIES:
        forecast[f"{policy}_count"] = np.maximum(0.0, models[policy].predict(test[features]))
    forecast["p90_count"] = np.maximum(forecast["p90_count"], forecast["p50_count"])
    forecast["p95_count"] = np.maximum(forecast["p95_count"], forecast["p90_count"])
    return forecast


def predict_hgb_hurdle(
    classifier: HistGradientBoostingClassifier | None,
    regressors: dict[str, HistGradientBoostingRegressor],
    test: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    forecast = test[["window", "target_count"]].copy()
    if classifier is None:
        active_probability = np.ones(len(test), dtype=float)
    else:
        active_probability = classifier.predict_proba(test[features])[:, 1]
    forecast["active_probability"] = active_probability
    positive_predictions = {
        policy: np.maximum(0.0, regressors[policy].predict(test[features]))
        for policy in POLICIES
    } if regressors else {policy: np.zeros(len(test), dtype=float) for policy in POLICIES}
    for policy, quantile in QUANTILES.items():
        zero_mass = 1.0 - active_probability
        values = np.where(quantile <= zero_mass, 0.0, positive_predictions[policy])
        forecast[f"{policy}_count"] = values
    forecast["p90_count"] = np.maximum(forecast["p90_count"], forecast["p50_count"])
    forecast["p95_count"] = np.maximum(forecast["p95_count"], forecast["p90_count"])
    return forecast


def build_detail(
    forecast: pd.DataFrame,
    workflow_name: str,
    method: str,
    activation_threshold: float,
) -> pd.DataFrame:
    rows = []
    for _, row in forecast.iterrows():
        actual = int(row["target_count"])
        for policy in POLICIES:
            forecast_count = float(row[f"{policy}_count"])
            allocated = alloc_count(forecast_count, activation_threshold)
            rows.append(
                {
                    "workflow_name": workflow_name,
                    "method": method,
                    "policy": policy,
                    "window": int(row["window"]),
                    "actual_count": actual,
                    "forecast_count": forecast_count,
                    "allocated_count": allocated,
                    "under_count": max(0, actual - allocated),
                    "over_count": max(0, allocated - actual),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    window_ms = resolve_window_ms(args)
    workflow = load_workflow(args.workflow_config)
    workflow_name = workflow.workflow_name
    trace = pd.read_csv(args.trace)
    _, _, split_map = load_split(
        trace,
        workflow_name,
        args.split_map,
        train_ratio=0.7,
        split_strategy="time",
    )
    split_map["entry_window"] = (split_map["entry_ts_ms"] // window_ms).astype(int)
    train_end_window = int(split_map["split_cutoff_ms"].dropna().iloc[0] // window_ms)
    eval_start_window = train_end_window + 1
    eval_end_window = int(split_map[split_map["split"] == "test"]["entry_window"].max())

    counts = build_entry_counts(trace, workflow_name, window_ms)
    frame = build_supervised_frame(counts, window_ms, args.max_lag_windows)
    features = feature_columns(frame)
    train = frame[frame["window"] <= train_end_window].copy()
    test = frame[(frame["window"] >= eval_start_window) & (frame["window"] <= eval_end_window)].copy()
    if train.empty or test.empty:
        raise ValueError("train/test frame is empty")

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    details = []
    forecasts = []
    if "hgb-quantile" in methods:
        models = fit_hgb_quantile_models(train, features, args)
        forecast = predict_hgb_quantile(models, test, features)
        forecast["method"] = "hgb-quantile"
        forecasts.append(forecast)
        details.append(build_detail(forecast, workflow_name, "hgb-quantile", args.activation_threshold))
    if "lightgbm-quantile" in methods or "lgbm-quantile" in methods:
        models = fit_lightgbm_quantile_models(train, features, args)
        forecast = predict_lightgbm_quantile(models, test, features)
        forecast["method"] = "lightgbm-quantile"
        forecasts.append(forecast)
        details.append(build_detail(forecast, workflow_name, "lightgbm-quantile", args.activation_threshold))
    if "hgb-hurdle" in methods:
        classifier, regressors = fit_hgb_hurdle_models(train, features, args)
        forecast = predict_hgb_hurdle(classifier, regressors, test, features)
        forecast["method"] = "hgb-hurdle"
        forecasts.append(forecast)
        details.append(build_detail(forecast, workflow_name, "hgb-hurdle", args.activation_threshold))

    detail = pd.concat(details, ignore_index=True)
    summary = summarize(detail, window_ms)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{workflow_name}_entry_ml_compare_summary.csv"
    detail_path = out_dir / f"{workflow_name}_entry_ml_compare_detail.csv"
    forecast_path = out_dir / f"{workflow_name}_entry_ml_forecast.csv"
    metadata_path = out_dir / f"{workflow_name}_entry_ml_compare_metadata.json"

    summary.to_csv(summary_path, index=False)
    if args.write_detail:
        detail.to_csv(detail_path, index=False)
    if args.write_forecast_csv:
        pd.concat(forecasts, ignore_index=True).to_csv(forecast_path, index=False)
    metadata = {
        "workflow_name": workflow_name,
        "trace": args.trace,
        "workflow_config": args.workflow_config,
        "split_map": args.split_map,
        "window_ms": window_ms,
        "methods": methods,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_active_rows": int((train["target_count"] > 0).sum()),
        "test_active_rows": int((test["target_count"] > 0).sum()),
        "features": features,
        "model": "sklearn HistGradientBoosting and LightGBM quantile baselines",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    if args.write_detail:
        print(f"wrote {detail_path}")
    if args.write_forecast_csv:
        print(f"wrote {forecast_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

