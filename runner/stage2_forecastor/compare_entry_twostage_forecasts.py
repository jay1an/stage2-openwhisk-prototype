import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .compare_stage_forecasts import load_split, resolve_window_ms
from ..workflow import load_workflow


POLICIES = {"p50": 0.50, "p90": 0.90, "p95": 0.95}
ROLL_WINDOWS = [3, 5, 10, 30, 60, 120, 240]
LAG_WINDOWS = [1, 2, 3, 5, 10, 30, 60]


@dataclass
class ActiveModel:
    classifier: HistGradientBoostingClassifier | None
    constant_probability: float
    features: list[str]


@dataclass
class ConditionalModels:
    regressors: dict[str, HistGradientBoostingRegressor]
    empirical_quantiles: dict[str, float]
    features: list[str]


@dataclass
class ForecastParts:
    p_active: float
    conditional: dict[str, float]
    unconditional: dict[str, float]
    calibrated: dict[str, float]
    gate_thresholds: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage probabilistic entry forecaster: estimate P(active), "
            "then estimate count quantiles conditional on active windows."
        )
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--split-map", default=None)
    parser.add_argument("--split-strategy", choices=["request-count", "time"], default="time")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--window-sec", type=int, default=5)
    parser.add_argument("--window-ms", type=int, default=None)
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument(
        "--methods",
        default="twostage-gbdt,twostage-gbdt-calibrated,twostage-hazard,twostage-hybrid-calibrated",
        help=(
            "comma-separated methods: twostage-gbdt,twostage-gbdt-calibrated,"
            "twostage-hazard,twostage-hybrid-calibrated"
        ),
    )
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.25,
        help="fraction of training windows reserved for gate-threshold calibration",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=["coverage-first", "cost-aware"],
        default="coverage-first",
        help=(
            "coverage-first matches nominal p50/p90/p95 on the calibration window; "
            "cost-aware minimizes under/over allocation trade-off instead"
        ),
    )
    parser.add_argument(
        "--count-calibration-mode",
        choices=["none", "conformal", "cost-aware"],
        default="none",
        help=(
            "post-hoc adjustment for p50/p90/p95 count magnitudes. "
            "conformal shifts each upper count quantile by calibration residuals; "
            "cost-aware chooses a scalar shift by under/over allocation cost."
        ),
    )
    parser.add_argument(
        "--under-cost",
        type=float,
        default=5.0,
        help="cost-aware penalty per missed request-window",
    )
    parser.add_argument(
        "--over-cost",
        type=float,
        default=1.0,
        help="cost-aware penalty per over-allocated replica-window",
    )
    parser.add_argument("--calibration-grid-size", type=int, default=80)
    parser.add_argument("--count-calibration-grid-size", type=int, default=81)
    parser.add_argument("--max-positive-weight", type=float, default=30.0)
    parser.add_argument("--min-positive-regression-rows", type=int, default=8)
    parser.add_argument("--recent-positive-window", type=int, default=240)
    parser.add_argument("--hazard-alpha", type=float, default=0.20)
    parser.add_argument("--hazard-bandwidth-windows", type=int, default=2)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--write-detail", action="store_true")
    parser.add_argument("--write-forecast-csv", action="store_true")
    return parser.parse_args()


def ceil_count(value: float) -> int:
    return int(math.ceil(max(0.0, float(value))))


def alloc_count(value: float, activation_threshold: float) -> int:
    value = max(0.0, float(value))
    if value < activation_threshold:
        return 0
    return ceil_count(value)


def build_entry_counts(
    trace: pd.DataFrame,
    workflow_name: str,
    window_ms: int,
    first_window: int,
    last_window: int,
) -> pd.Series:
    entries = trace[
        (trace["workflow_name"] == workflow_name)
        & (trace["stage_name"] == "__entry__")
        & (trace["status"] == "ok")
    ].copy()
    entries["window"] = (entries["entry_ts_ms"] // window_ms).astype(int)
    return (
        entries.groupby("window")
        .size()
        .reindex(range(first_window, last_window + 1), fill_value=0)
        .astype(float)
    )


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


def time_since_active(history: np.ndarray, cap: int) -> int:
    active = np.flatnonzero(history > 0)
    if len(active) == 0:
        return cap
    return min(cap, len(history) - 1 - int(active[-1]))


def active_run_length(history: np.ndarray) -> int:
    run = 0
    for value in history[::-1]:
        if value <= 0:
            break
        run += 1
    return run


def make_features_for_target(
    counts: pd.Series,
    target_window: int,
    first_window: int,
    window_ms: int,
) -> dict[str, float]:
    history = counts[counts.index < target_window].to_numpy(dtype=float)
    if len(history) == 0:
        history = np.array([0.0], dtype=float)

    row: dict[str, float] = {
        "target_window_offset": float(target_window - first_window),
        "active_run_length": float(active_run_length(history)),
        "time_since_active": float(time_since_active(history, cap=max(1, len(history)))),
    }
    for lag in LAG_WINDOWS:
        row[f"lag_{lag}"] = float(history[-lag]) if len(history) >= lag else 0.0
    for size in ROLL_WINDOWS:
        recent = history[-size:]
        row[f"roll_sum_{size}"] = float(np.sum(recent))
        row[f"roll_mean_{size}"] = float(np.mean(recent))
        row[f"roll_max_{size}"] = float(np.max(recent))
        row[f"zero_ratio_{size}"] = float(np.mean(recent == 0))

    # Time features are weak here, but keep them to let the model learn repeated phases if present.
    seconds = int(target_window) * window_ms / 1000.0
    row["phase_2h_sin"] = math.sin(2.0 * math.pi * seconds / 7200.0)
    row["phase_2h_cos"] = math.cos(2.0 * math.pi * seconds / 7200.0)
    row["phase_10m_sin"] = math.sin(2.0 * math.pi * seconds / 600.0)
    row["phase_10m_cos"] = math.cos(2.0 * math.pi * seconds / 600.0)
    return row


def build_supervised_frame(
    counts: pd.Series,
    start_window: int,
    end_window: int,
    first_window: int,
    window_ms: int,
) -> pd.DataFrame:
    rows = []
    for window in range(start_window, end_window + 1):
        row = make_features_for_target(counts, window, first_window, window_ms)
        target = float(counts.get(window, 0.0))
        row["window"] = int(window)
        row["target_count"] = target
        row["target_active"] = 1 if target > 0 else 0
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in {"window", "target_count", "target_active"}
    ]


def choose_fit_calibration_windows(train_windows: np.ndarray, calibration_fraction: float) -> tuple[int, int]:
    if len(train_windows) < 10:
        last = int(train_windows[-1])
        return last, last
    fraction = min(0.8, max(0.05, calibration_fraction))
    cut_pos = max(1, int(math.floor(len(train_windows) * (1.0 - fraction))))
    fit_end = int(train_windows[cut_pos - 1])
    cal_start = int(train_windows[cut_pos])
    return fit_end, cal_start


def fit_active_model(
    train: pd.DataFrame,
    features: list[str],
    args: argparse.Namespace,
) -> ActiveModel:
    y = train["target_active"].astype(int)
    constant = float(y.mean()) if len(y) else 0.0
    if y.nunique() < 2:
        return ActiveModel(classifier=None, constant_probability=constant, features=features)

    positives = int(y.sum())
    negatives = int(len(y) - positives)
    pos_weight = min(args.max_positive_weight, negatives / max(1, positives))
    sample_weight = np.where(y.to_numpy() > 0, pos_weight, 1.0)
    classifier = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        l2_regularization=args.l2_regularization,
        random_state=args.random_state,
    )
    classifier.fit(train[features], y, sample_weight=sample_weight)
    return ActiveModel(classifier=classifier, constant_probability=constant, features=features)


def predict_active_probability(model: ActiveModel, frame: pd.DataFrame) -> np.ndarray:
    if model.classifier is None:
        return np.full(len(frame), model.constant_probability, dtype=float)
    proba = model.classifier.predict_proba(frame[model.features])[:, 1]
    return np.clip(proba, 0.0, 1.0)


def empirical_positive_quantiles(positive: pd.DataFrame) -> dict[str, float]:
    if positive.empty:
        return {policy: 0.0 for policy in POLICIES}
    values = positive["target_count"].to_numpy(dtype=float)
    return {policy: float(np.quantile(values, q)) for policy, q in POLICIES.items()}


def fit_conditional_models(
    train: pd.DataFrame,
    features: list[str],
    args: argparse.Namespace,
) -> ConditionalModels:
    positive = train[train["target_count"] > 0].copy()
    empirical = empirical_positive_quantiles(positive)
    if len(positive) < args.min_positive_regression_rows:
        return ConditionalModels(regressors={}, empirical_quantiles=empirical, features=features)

    regressors: dict[str, HistGradientBoostingRegressor] = {}
    y = positive["target_count"].astype(float)
    for policy, quantile in POLICIES.items():
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            l2_regularization=args.l2_regularization,
            random_state=args.random_state,
        )
        model.fit(positive[features], y)
        regressors[policy] = model
    return ConditionalModels(regressors=regressors, empirical_quantiles=empirical, features=features)


def recent_empirical_quantiles(
    train_counts: pd.Series,
    target_window: int,
    recent_positive_window: int,
) -> dict[str, float]:
    history = train_counts[train_counts.index < target_window]
    recent = history.tail(max(1, recent_positive_window))
    positive = recent[recent > 0].to_numpy(dtype=float)
    if len(positive) == 0:
        positive = history[history > 0].to_numpy(dtype=float)
    if len(positive) == 0:
        return {policy: 0.0 for policy in POLICIES}
    return {policy: float(np.quantile(positive, q)) for policy, q in POLICIES.items()}


def predict_conditional_quantiles(
    models: ConditionalModels,
    frame: pd.DataFrame,
    counts: pd.Series,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    result = {}
    for policy in POLICIES:
        if policy in models.regressors:
            pred = np.maximum(0.0, models.regressors[policy].predict(frame[models.features]))
        else:
            pred = np.full(len(frame), models.empirical_quantiles.get(policy, 0.0), dtype=float)

        recent_values = []
        for window in frame["window"].astype(int):
            recent_values.append(
                recent_empirical_quantiles(counts, int(window), args.recent_positive_window)[policy]
            )
        recent = np.asarray(recent_values, dtype=float)
        # Blend model and recent empirical quantiles; recent empirical is more robust for rare bursts.
        if policy in models.regressors:
            result[policy] = np.maximum(0.0, 0.7 * pred + 0.3 * recent)
        else:
            result[policy] = recent

    result["p90"] = np.maximum(result["p90"], result["p50"])
    result["p95"] = np.maximum(result["p95"], result["p90"])
    return result


def ewma(values: np.ndarray, alpha: float) -> float:
    if len(values) == 0:
        return 0.0
    current = float(values[0])
    for value in values[1:]:
        current = alpha * float(value) + (1.0 - alpha) * current
    return current


def hazard_active_probability(
    counts: pd.Series,
    target_window: int,
    args: argparse.Namespace,
) -> float:
    history = counts[counts.index < target_window].to_numpy(dtype=float)
    if len(history) == 0:
        return 0.0
    active = (history > 0).astype(float)
    recent_rate = float(np.mean(active[-max(1, args.recent_positive_window):]))
    smooth_rate = ewma(active, args.hazard_alpha)

    active_pos = np.flatnonzero(active > 0)
    if len(active_pos) < 2:
        return min(1.0, max(0.0, 0.5 * recent_rate + 0.5 * smooth_rate))

    idle_age = len(history) - 1 - int(active_pos[-1])
    gaps = np.diff(active_pos).astype(int)
    at_risk = gaps > idle_age
    risk = int(np.sum(at_risk))
    if risk <= 0:
        hazard = 0.0
    else:
        low = max(idle_age + 1, idle_age + 1 - args.hazard_bandwidth_windows)
        high = idle_age + 1 + args.hazard_bandwidth_windows
        events = int(np.sum(at_risk & (gaps >= low) & (gaps <= high)))
        hazard = events / max(1, risk * (high - low + 1))
    probability = 0.55 * hazard + 0.25 * smooth_rate + 0.20 * recent_rate
    return min(1.0, max(0.0, float(probability)))


def zero_inflated_quantiles(p_active: np.ndarray, conditional: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for policy, quantile in POLICIES.items():
        values = np.zeros(len(p_active), dtype=float)
        active_mask = p_active > 0
        adjusted = np.zeros(len(p_active), dtype=float)
        adjusted[active_mask] = (quantile - (1.0 - p_active[active_mask])) / p_active[active_mask]
        use_positive = active_mask & (quantile > (1.0 - p_active))
        values[use_positive] = conditional[policy][use_positive]
        result[policy] = np.maximum(0.0, values)
    result["p90"] = np.maximum(result["p90"], result["p50"])
    result["p95"] = np.maximum(result["p95"], result["p90"])
    return result


def thresholded_quantiles(
    p_active: np.ndarray,
    conditional: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> dict[str, np.ndarray]:
    result = {}
    for policy in POLICIES:
        result[policy] = np.where(p_active >= thresholds.get(policy, 1.0), conditional[policy], 0.0)
    result["p90"] = np.maximum(result["p90"], result["p50"])
    result["p95"] = np.maximum(result["p95"], result["p90"])
    return result


def choose_gate_thresholds(
    calibration: pd.DataFrame,
    p_active_col: str,
    conditional_prefix: str,
    grid_size: int,
    activation_threshold: float,
    calibration_mode: str,
    under_cost: float,
    over_cost: float,
) -> dict[str, float]:
    if calibration.empty:
        return {policy: 1.0 - q for policy, q in POLICIES.items()}

    p_active = calibration[p_active_col].to_numpy(dtype=float)
    actual = calibration["target_count"].to_numpy(dtype=float)
    actual_total = float(np.sum(actual))
    max_p = max(float(np.max(p_active)), 1e-6)
    grid = np.linspace(0.0, max_p, max(2, grid_size))
    thresholds: dict[str, float] = {}

    for policy, target_coverage in POLICIES.items():
        conditional = calibration[f"{conditional_prefix}_{policy}"].to_numpy(dtype=float)
        candidates = []
        for threshold in grid:
            forecast = np.where(p_active >= threshold, conditional, 0.0)
            allocated = np.where(forecast >= activation_threshold, np.ceil(np.maximum(0.0, forecast)), 0.0)
            under = np.maximum(0.0, actual - allocated)
            over = np.maximum(0.0, allocated - actual)
            under_total = float(np.sum(under))
            allocated_total = float(np.sum(allocated))
            over_total = float(np.sum(over))
            coverage = 1.0 - under_total / actual_total if actual_total > 0 else 1.0
            objective = under_cost * under_total + over_cost * over_total
            candidates.append((coverage, allocated_total, over_total, objective, float(threshold)))

        if calibration_mode == "cost-aware":
            # Prefer the smallest expected cost, then avoid unnecessary allocation.
            chosen = min(candidates, key=lambda item: (item[3], item[1], -item[4]))
        else:
            feasible = [item for item in candidates if item[0] >= target_coverage]
            if feasible:
                chosen = min(feasible, key=lambda item: (item[1], -item[4]))
            else:
                chosen = min(candidates, key=lambda item: (-item[0], item[1], -item[4]))
        thresholds[policy] = chosen[4]
    return thresholds


def build_prediction_frame(
    counts: pd.Series,
    start_window: int,
    end_window: int,
    first_window: int,
    window_ms: int,
) -> pd.DataFrame:
    return build_supervised_frame(counts, start_window, end_window, first_window, window_ms)


def add_forecast_columns(
    frame: pd.DataFrame,
    p_active: np.ndarray,
    hazard_p_active: np.ndarray,
    conditional: dict[str, np.ndarray],
    hazard_conditional: dict[str, np.ndarray],
    thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    out = frame[["window", "target_count", "target_active"]].copy()
    out["gbdt_p_active"] = p_active
    out["hazard_p_active"] = hazard_p_active
    hybrid_p_active = np.maximum(p_active, hazard_p_active)
    out["hybrid_p_active"] = hybrid_p_active

    unconditional = zero_inflated_quantiles(p_active, conditional)
    gbdt_calibrated = thresholded_quantiles(p_active, conditional, thresholds["gbdt"])

    hazard_unconditional = zero_inflated_quantiles(hazard_p_active, hazard_conditional)
    hybrid_conditional = {
        policy: np.maximum(conditional[policy], hazard_conditional[policy]) for policy in POLICIES
    }
    hybrid_calibrated = thresholded_quantiles(
        hybrid_p_active, hybrid_conditional, thresholds["hybrid"]
    )

    for policy in POLICIES:
        out[f"gbdt_conditional_{policy}"] = conditional[policy]
        out[f"hazard_conditional_{policy}"] = hazard_conditional[policy]
        out[f"hybrid_conditional_{policy}"] = hybrid_conditional[policy]
        out[f"twostage-gbdt_{policy}"] = unconditional[policy]
        out[f"twostage-gbdt-calibrated_{policy}"] = gbdt_calibrated[policy]
        out[f"twostage-hazard_{policy}"] = hazard_unconditional[policy]
        out[f"twostage-hybrid-calibrated_{policy}"] = hybrid_calibrated[policy]
    return out


def compute_count_calibration_shifts(
    calibration_forecast: pd.DataFrame,
    methods: list[str],
    activation_threshold: float,
    mode: str,
    under_cost: float,
    over_cost: float,
    grid_size: int,
) -> dict[str, dict[str, float]]:
    shifts: dict[str, dict[str, float]] = {
        method: {policy: 0.0 for policy in POLICIES} for method in methods
    }
    if mode == "none" or calibration_forecast.empty:
        return shifts

    actual = calibration_forecast["target_count"].to_numpy(dtype=float)
    for method in methods:
        for policy, quantile in POLICIES.items():
            col = f"{method}_{policy}"
            if col not in calibration_forecast:
                continue
            pred = calibration_forecast[col].to_numpy(dtype=float)
            residual = actual - pred
            if len(residual) == 0:
                continue
            if mode == "conformal":
                shifts[method][policy] = float(np.quantile(residual, quantile))
                continue

            low = float(np.quantile(residual, 0.02))
            high = float(np.quantile(residual, 0.98))
            if math.isclose(low, high):
                grid = np.asarray([low], dtype=float)
            else:
                grid = np.linspace(low, high, max(2, grid_size))
            candidates = []
            for shift in grid:
                adjusted = np.maximum(0.0, pred + shift)
                allocated = np.where(
                    adjusted >= activation_threshold,
                    np.ceil(adjusted),
                    0.0,
                )
                under = np.maximum(0.0, actual - allocated)
                over = np.maximum(0.0, allocated - actual)
                objective = under_cost * float(np.sum(under)) + over_cost * float(np.sum(over))
                coverage = 1.0 - float(np.sum(under)) / float(np.sum(actual)) if np.sum(actual) > 0 else 1.0
                candidates.append((objective, -coverage, float(np.sum(allocated)), float(shift)))
            shifts[method][policy] = min(candidates)[3]
    return shifts


def apply_count_calibration_shifts(
    forecast: pd.DataFrame,
    methods: list[str],
    shifts: dict[str, dict[str, float]],
) -> pd.DataFrame:
    out = forecast.copy()
    for method in methods:
        for policy in POLICIES:
            col = f"{method}_{policy}"
            if col in out:
                out[col] = np.maximum(0.0, out[col].astype(float) + shifts.get(method, {}).get(policy, 0.0))
        p50 = f"{method}_p50"
        p90 = f"{method}_p90"
        p95 = f"{method}_p95"
        if p50 in out and p90 in out:
            out[p90] = np.maximum(out[p90].astype(float), out[p50].astype(float))
        if p90 in out and p95 in out:
            out[p95] = np.maximum(out[p95].astype(float), out[p90].astype(float))
    return out


def evaluate_detail(detail: pd.DataFrame, window_ms: int) -> pd.DataFrame:
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
                "mean_p_active": float(group["p_active"].mean()),
                "brier_active_probability": float(
                    np.mean(((group["actual_count"] > 0).astype(float) - group["p_active"].astype(float)) ** 2)
                ),
                "mae": float(np.mean(np.abs(actual - forecast))),
                "rmse": float(np.sqrt(np.mean((actual - forecast) ** 2))),
                "max_actual": int(actual.max()) if len(actual) else 0,
                "max_allocated": int(allocated.max()) if len(allocated) else 0,
            }
        )
    return pd.DataFrame(rows)


def build_detail(
    forecast: pd.DataFrame,
    workflow_name: str,
    methods: list[str],
    activation_threshold: float,
) -> pd.DataFrame:
    rows = []
    for _, row in forecast.iterrows():
        actual = int(row["target_count"])
        for method in methods:
            p_active_col = "hazard_p_active" if method == "twostage-hazard" else (
                "hybrid_p_active" if method == "twostage-hybrid-calibrated" else "gbdt_p_active"
            )
            for policy in POLICIES:
                forecast_count = float(row[f"{method}_{policy}"])
                conditional_col = (
                    f"hazard_conditional_{policy}"
                    if method == "twostage-hazard"
                    else f"hybrid_conditional_{policy}"
                    if method == "twostage-hybrid-calibrated"
                    else f"gbdt_conditional_{policy}"
                )
                allocated = alloc_count(forecast_count, activation_threshold)
                rows.append(
                    {
                        "workflow_name": workflow_name,
                        "method": method,
                        "policy": policy,
                        "target_window": int(row["window"]),
                        "actual_count": actual,
                        "actual_active": int(actual > 0),
                        "p_active": float(row[p_active_col]),
                        "conditional_forecast_count": float(row[conditional_col]),
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
    _, _, split_map = load_split(
        trace,
        workflow_name,
        args.split_map,
        args.train_ratio,
        args.split_strategy,
    )
    train_end_window, eval_start_window, eval_end_window = split_windows(split_map, window_ms)
    first_window = int(trace["entry_ts_ms"].min() // window_ms)
    counts = build_entry_counts(trace, workflow_name, window_ms, first_window, eval_end_window)

    train_windows = counts[counts.index <= train_end_window].index.to_numpy(dtype=int)
    fit_end_window, calibration_start_window = choose_fit_calibration_windows(
        train_windows, args.calibration_fraction
    )
    train_frame = build_supervised_frame(counts, first_window + 1, train_end_window, first_window, window_ms)
    fit_frame = train_frame[train_frame["window"] <= fit_end_window].copy()
    calibration_frame = train_frame[train_frame["window"] >= calibration_start_window].copy()
    if fit_frame.empty:
        fit_frame = train_frame.copy()
    if calibration_frame.empty:
        calibration_frame = train_frame.copy()

    features = feature_columns(train_frame)
    active_model = fit_active_model(fit_frame, features, args)
    conditional_models = fit_conditional_models(fit_frame, features, args)

    calibration_p_active = predict_active_probability(active_model, calibration_frame)
    calibration_conditional = predict_conditional_quantiles(
        conditional_models, calibration_frame, counts, args
    )
    calibration_hazard_p = np.asarray(
        [hazard_active_probability(counts, int(w), args) for w in calibration_frame["window"]],
        dtype=float,
    )
    calibration_hazard_conditional = {
        policy: np.asarray(
            [
                recent_empirical_quantiles(counts, int(w), args.recent_positive_window)[policy]
                for w in calibration_frame["window"]
            ],
            dtype=float,
        )
        for policy in POLICIES
    }
    calibration_hybrid_p = np.maximum(calibration_p_active, calibration_hazard_p)
    calibration_hybrid_conditional = {
        policy: np.maximum(calibration_conditional[policy], calibration_hazard_conditional[policy])
        for policy in POLICIES
    }
    cal = calibration_frame[["window", "target_count", "target_active"]].copy()
    cal["gbdt_p_active"] = calibration_p_active
    cal["hybrid_p_active"] = calibration_hybrid_p
    for policy in POLICIES:
        cal[f"gbdt_conditional_{policy}"] = calibration_conditional[policy]
        cal[f"hybrid_conditional_{policy}"] = calibration_hybrid_conditional[policy]

    thresholds = {
        "gbdt": choose_gate_thresholds(
            cal,
            p_active_col="gbdt_p_active",
            conditional_prefix="gbdt_conditional",
            grid_size=args.calibration_grid_size,
            activation_threshold=args.activation_threshold,
            calibration_mode=args.calibration_mode,
            under_cost=args.under_cost,
            over_cost=args.over_cost,
        ),
        "hybrid": choose_gate_thresholds(
            cal,
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
        p_active=calibration_p_active,
        hazard_p_active=calibration_hazard_p,
        conditional=calibration_conditional,
        hazard_conditional=calibration_hazard_conditional,
        thresholds=thresholds,
    )
    count_shifts = compute_count_calibration_shifts(
        calibration_forecast=calibration_forecast,
        methods=methods,
        activation_threshold=args.activation_threshold,
        mode=args.count_calibration_mode,
        under_cost=args.under_cost,
        over_cost=args.over_cost,
        grid_size=args.count_calibration_grid_size,
    )

    eval_frame = build_prediction_frame(counts, eval_start_window, eval_end_window, first_window, window_ms)
    p_active = predict_active_probability(active_model, eval_frame)
    conditional = predict_conditional_quantiles(conditional_models, eval_frame, counts, args)
    hazard_p_active = np.asarray(
        [hazard_active_probability(counts, int(w), args) for w in eval_frame["window"]],
        dtype=float,
    )
    hazard_conditional = {
        policy: np.asarray(
            [
                recent_empirical_quantiles(counts, int(w), args.recent_positive_window)[policy]
                for w in eval_frame["window"]
            ],
            dtype=float,
        )
        for policy in POLICIES
    }

    forecast = add_forecast_columns(
        eval_frame,
        p_active=p_active,
        hazard_p_active=hazard_p_active,
        conditional=conditional,
        hazard_conditional=hazard_conditional,
        thresholds=thresholds,
    )
    forecast = apply_count_calibration_shifts(forecast, methods, count_shifts)

    detail = build_detail(forecast, workflow_name, methods, args.activation_threshold)
    summary = evaluate_detail(detail, window_ms)
    if not summary.empty:
        summary.insert(0, "calibration_mode", args.calibration_mode)
        summary.insert(1, "under_cost", float(args.under_cost))
        summary.insert(2, "over_cost", float(args.over_cost))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{workflow_name}_entry_twostage_compare_summary.csv"
    detail_path = out_dir / f"{workflow_name}_entry_twostage_compare_detail.csv"
    forecast_path = out_dir / f"{workflow_name}_entry_twostage_forecast.csv"
    metadata_path = out_dir / f"{workflow_name}_entry_twostage_compare_metadata.json"
    summary.to_csv(summary_path, index=False)
    if args.write_detail:
        detail.to_csv(detail_path, index=False)
    if args.write_forecast_csv:
        forecast.to_csv(forecast_path, index=False)

    metadata = {
        "workflow_name": workflow_name,
        "trace": args.trace,
        "workflow_config": args.workflow_config,
        "split_map": args.split_map,
        "split_strategy": args.split_strategy if args.split_map is None else "provided-map",
        "window_ms": window_ms,
        "methods": methods,
        "policies": list(POLICIES.keys()),
        "calibration_mode": args.calibration_mode,
        "count_calibration_mode": args.count_calibration_mode,
        "under_cost": args.under_cost,
        "over_cost": args.over_cost,
        "first_window": first_window,
        "train_end_window": train_end_window,
        "eval_start_window": eval_start_window,
        "eval_end_window": eval_end_window,
        "fit_end_window": fit_end_window,
        "calibration_start_window": calibration_start_window,
        "train_rows": int(len(train_frame)),
        "fit_rows": int(len(fit_frame)),
        "calibration_rows": int(len(calibration_frame)),
        "eval_rows": int(len(eval_frame)),
        "train_active_rows": int((train_frame["target_count"] > 0).sum()),
        "fit_active_rows": int((fit_frame["target_count"] > 0).sum()),
        "calibration_active_rows": int((calibration_frame["target_count"] > 0).sum()),
        "eval_active_rows": int((eval_frame["target_count"] > 0).sum()),
        "features": features,
        "thresholds": thresholds,
        "count_calibration_shifts": count_shifts,
        "active_model": "HistGradientBoostingClassifier with weighted positives; constant fallback",
        "conditional_model": "HistGradientBoostingRegressor quantile on positive windows; empirical fallback",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote {summary_path}")
    if args.write_detail:
        print(f"wrote {detail_path}")
    if args.write_forecast_csv:
        print(f"wrote {forecast_path}")
    print(f"wrote {metadata_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

