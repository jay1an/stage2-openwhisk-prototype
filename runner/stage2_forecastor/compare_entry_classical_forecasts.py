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
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from .compare_entry_ml_forecasts import alloc_count, build_entry_counts, summarize
from .compare_stage_forecasts import load_split, resolve_window_ms
from ..workflow import load_workflow


POLICIES = {"p50": 0.50, "p90": 0.90, "p95": 0.95}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classical time-series baselines for workflow entry arrival forecasting. "
            "Outputs p50/p90/p95 with rolling residual calibration when needed."
        )
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--split-map", required=True)
    parser.add_argument("--window-sec", type=int, default=5)
    parser.add_argument("--window-ms", type=int, default=None)
    parser.add_argument(
        "--methods",
        default="naive,moving-average,seasonal-naive,arima,poisson-lag",
        help="comma-separated methods",
    )
    parser.add_argument("--history-windows", type=int, default=60)
    parser.add_argument("--season-windows", type=int, default=12)
    parser.add_argument("--max-lag-windows", type=int, default=60)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--arima-order", default="1,0,1")
    parser.add_argument(
        "--poisson-cap-multiplier",
        type=float,
        default=2.0,
        help="cap Poisson mean at this multiplier of the maximum training count",
    )
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--write-detail", action="store_true")
    parser.add_argument("--write-forecast-csv", action="store_true")
    return parser.parse_args()


def parse_order(value: str) -> tuple[int, int, int]:
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--arima-order must have the form p,d,q")
    if any(part < 0 for part in parts):
        raise ValueError("--arima-order values must be non-negative")
    return parts[0], parts[1], parts[2]


def split_series(
    counts: pd.Series,
    train_end_window: int,
    eval_start_window: int,
    eval_end_window: int,
) -> tuple[pd.Series, pd.Series]:
    train = counts[counts.index <= train_end_window].astype(float)
    test = counts[(counts.index >= eval_start_window) & (counts.index <= eval_end_window)].astype(float)
    if train.empty or test.empty:
        raise ValueError("train/test counts are empty")
    return train, test


def residual_quantiles(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    if len(residual) == 0:
        return {policy: 0.0 for policy in POLICIES}
    return {
        policy: float(np.quantile(residual, quantile))
        for policy, quantile in POLICIES.items()
    }


def enforce_quantile_order(frame: pd.DataFrame) -> pd.DataFrame:
    frame["p50_count"] = np.maximum(0.0, frame["p50_count"])
    frame["p90_count"] = np.maximum(frame["p90_count"], frame["p50_count"])
    frame["p95_count"] = np.maximum(frame["p95_count"], frame["p90_count"])
    return frame


def make_residual_calibrated_forecast(
    method: str,
    train: pd.Series,
    test: pd.Series,
    base_predictor,
    calibration_ratio: float,
) -> pd.DataFrame:
    cal_size = max(16, int(math.ceil(len(train) * calibration_ratio)))
    if cal_size >= len(train):
        cal_size = max(1, len(train) // 5)
    fit = train.iloc[:-cal_size]
    cal = train.iloc[-cal_size:]
    history = fit.copy()
    cal_pred = []
    for _, actual in cal.items():
        pred = float(base_predictor(history))
        cal_pred.append(max(0.0, pred))
        history = pd.concat([history, pd.Series([float(actual)], index=[cal.index[len(cal_pred) - 1]])])

    rq = residual_quantiles(cal.to_numpy(dtype=float), np.asarray(cal_pred, dtype=float))
    history = train.copy()
    rows = []
    for window, actual in test.items():
        base = max(0.0, float(base_predictor(history)))
        row = {"window": int(window), "actual_count": float(actual), "base_count": base}
        for policy in POLICIES:
            row[f"{policy}_count"] = max(0.0, base + rq[policy])
        rows.append(row)
        history = pd.concat([history, pd.Series([float(actual)], index=[window])])

    forecast = pd.DataFrame(rows)
    forecast["method"] = method
    return enforce_quantile_order(forecast)


def predict_naive(train: pd.Series, test: pd.Series, calibration_ratio: float) -> pd.DataFrame:
    return make_residual_calibrated_forecast(
        "naive",
        train,
        test,
        base_predictor=lambda history: float(history.iloc[-1]),
        calibration_ratio=calibration_ratio,
    )


def predict_moving_average(
    train: pd.Series,
    test: pd.Series,
    history_windows: int,
    calibration_ratio: float,
) -> pd.DataFrame:
    history_windows = max(1, int(history_windows))
    return make_residual_calibrated_forecast(
        "moving-average",
        train,
        test,
        base_predictor=lambda history: float(history.iloc[-history_windows:].mean()),
        calibration_ratio=calibration_ratio,
    )


def predict_seasonal_naive(
    train: pd.Series,
    test: pd.Series,
    season_windows: int,
    calibration_ratio: float,
) -> pd.DataFrame:
    season_windows = max(1, int(season_windows))

    def base_predictor(history: pd.Series) -> float:
        if len(history) <= season_windows:
            return float(history.iloc[-1])
        return float(history.iloc[-season_windows])

    return make_residual_calibrated_forecast(
        "seasonal-naive",
        train,
        test,
        base_predictor=base_predictor,
        calibration_ratio=calibration_ratio,
    )


def predict_arima(
    train: pd.Series,
    test: pd.Series,
    order: tuple[int, int, int],
    calibration_ratio: float,
) -> pd.DataFrame:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ModuleNotFoundError as exc:
        raise SystemExit("statsmodels is required for ARIMA baseline") from exc

    cal_size = max(16, int(math.ceil(len(train) * calibration_ratio)))
    if cal_size >= len(train):
        cal_size = max(1, len(train) // 5)
    fit_series = train.iloc[:-cal_size]
    cal_series = train.iloc[-cal_size:]
    fit_model = ARIMA(
        fit_series.to_numpy(dtype=float),
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    cal_pred = []
    result = fit_model
    for actual in cal_series.to_numpy(dtype=float):
        pred = result.get_forecast(steps=1).predicted_mean[0]
        cal_pred.append(max(0.0, float(pred)))
        result = result.append([float(actual)], refit=False)

    rq = residual_quantiles(cal_series.to_numpy(dtype=float), np.asarray(cal_pred, dtype=float))
    result = ARIMA(
        train.to_numpy(dtype=float),
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    rows = []
    for window, actual in test.items():
        pred = max(0.0, float(result.get_forecast(steps=1).predicted_mean[0]))
        row = {"window": int(window), "actual_count": float(actual), "base_count": pred}
        for policy in POLICIES:
            row[f"{policy}_count"] = max(0.0, pred + rq[policy])
        rows.append(row)
        result = result.append([float(actual)], refit=False)

    forecast = pd.DataFrame(rows)
    forecast["method"] = f"arima-{order[0]}{order[1]}{order[2]}"
    return enforce_quantile_order(forecast)


def build_lag_frame(series: pd.Series, max_lag: int) -> pd.DataFrame:
    max_lag = max(1, int(max_lag))
    rows = []
    values = series.to_numpy(dtype=float)
    windows = series.index.to_numpy(dtype=int)
    for idx in range(max_lag, len(values)):
        history = values[:idx]
        row = {
            "window": int(windows[idx]),
            "target_count": float(values[idx]),
            "lag_1": float(values[idx - 1]),
            "lag_2": float(values[idx - 2]) if idx >= 2 else float(values[idx - 1]),
            "lag_3": float(values[idx - 3]) if idx >= 3 else float(values[idx - 1]),
            "lag_5": float(values[idx - 5]) if idx >= 5 else float(values[idx - 1]),
            "lag_10": float(values[idx - 10]) if idx >= 10 else float(values[idx - 1]),
        }
        for size in [3, 5, 10, 30, max_lag]:
            recent = history[-min(size, len(history)) :]
            row[f"roll_mean_{size}"] = float(np.mean(recent))
            row[f"roll_max_{size}"] = float(np.max(recent))
            row[f"roll_std_{size}"] = float(np.std(recent))
        rows.append(row)
    return pd.DataFrame(rows)


def predict_poisson_lag(
    counts: pd.Series,
    train_end_window: int,
    eval_start_window: int,
    eval_end_window: int,
    max_lag: int,
    cap_multiplier: float,
) -> pd.DataFrame:
    frame = build_lag_frame(counts, max_lag=max_lag)
    train = frame[frame["window"] <= train_end_window].copy()
    test = frame[(frame["window"] >= eval_start_window) & (frame["window"] <= eval_end_window)].copy()
    if train.empty or test.empty:
        raise ValueError("poisson-lag train/test frame is empty")
    features = [col for col in frame.columns if col not in {"window", "target_count"}]
    model = PoissonRegressor(alpha=0.01, max_iter=1000)
    model.fit(train[features], train["target_count"].astype(float))
    cap = max(1.0, float(train["target_count"].max()) * max(1.0, cap_multiplier))
    mean = np.clip(np.maximum(0.0, model.predict(test[features])), 0.0, cap)
    forecast = test[["window", "target_count"]].rename(columns={"target_count": "actual_count"}).copy()
    forecast["base_count"] = mean
    for policy, quantile in POLICIES.items():
        forecast[f"{policy}_count"] = poisson.ppf(quantile, np.maximum(mean, 1e-9))
    forecast["method"] = "poisson-lag"
    return enforce_quantile_order(forecast)


def build_detail(
    forecast: pd.DataFrame,
    workflow_name: str,
    method: str,
    activation_threshold: float,
) -> pd.DataFrame:
    rows = []
    for _, row in forecast.iterrows():
        actual = int(row["actual_count"])
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
    train, test = split_series(counts, train_end_window, eval_start_window, eval_end_window)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    order = parse_order(args.arima_order)

    forecasts = []
    if "naive" in methods:
        forecasts.append(predict_naive(train, test, args.calibration_ratio))
    if "moving-average" in methods:
        forecasts.append(predict_moving_average(train, test, args.history_windows, args.calibration_ratio))
    if "seasonal-naive" in methods:
        forecasts.append(predict_seasonal_naive(train, test, args.season_windows, args.calibration_ratio))
    if "arima" in methods:
        forecasts.append(predict_arima(train, test, order, args.calibration_ratio))
    if "poisson-lag" in methods:
        forecasts.append(
            predict_poisson_lag(
                counts=counts,
                train_end_window=train_end_window,
                eval_start_window=eval_start_window,
                eval_end_window=eval_end_window,
                max_lag=args.max_lag_windows,
                cap_multiplier=args.poisson_cap_multiplier,
            )
        )

    if not forecasts:
        raise ValueError("no methods selected")
    details = [
        build_detail(forecast, workflow_name, str(forecast["method"].iloc[0]), args.activation_threshold)
        for forecast in forecasts
    ]
    detail = pd.concat(details, ignore_index=True)
    summary = summarize(detail, window_ms)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{workflow_name}_entry_classical_compare_summary.csv"
    detail_path = out_dir / f"{workflow_name}_entry_classical_compare_detail.csv"
    forecast_path = out_dir / f"{workflow_name}_entry_classical_forecast.csv"
    metadata_path = out_dir / f"{workflow_name}_entry_classical_compare_metadata.json"
    summary.to_csv(summary_path, index=False)
    if args.write_detail:
        detail.to_csv(detail_path, index=False)
    if args.write_forecast_csv:
        pd.concat(forecasts, ignore_index=True, sort=False).to_csv(forecast_path, index=False)
    metadata = {
        "workflow_name": workflow_name,
        "trace": args.trace,
        "workflow_config": args.workflow_config,
        "split_map": args.split_map,
        "window_ms": window_ms,
        "methods": methods,
        "history_windows": args.history_windows,
        "season_windows": args.season_windows,
        "max_lag_windows": args.max_lag_windows,
        "poisson_cap_multiplier": args.poisson_cap_multiplier,
        "calibration_ratio": args.calibration_ratio,
        "arima_order": order,
        "train_windows": int(len(train)),
        "test_windows": int(len(test)),
        "model": "classical one-step entry forecasting baselines",
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

