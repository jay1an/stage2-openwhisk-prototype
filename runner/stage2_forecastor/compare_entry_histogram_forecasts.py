import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .compare_stage_forecasts import load_split
from ..workflow import load_workflow


POLICIES = {"p50": 0.50, "p90": 0.90, "p95": 0.95}


@dataclass
class ForecastComponents:
    p_active: float
    conditional_quantiles: dict[str, float]
    hurdle_quantiles: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rolling-origin entry forecasting for sparse serverless arrivals using "
            "HHP/LSTH-style histograms and calibrated/gated hurdle decisions."
        )
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--split-map", default=None)
    parser.add_argument("--split-strategy", choices=["request-count", "time"], default="time")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--window-sec", type=int, default=60)
    parser.add_argument("--window-ms", type=int, default=None)
    parser.add_argument("--horizon-windows", type=int, default=1)
    parser.add_argument("--origin-step-windows", type=int, default=1)
    parser.add_argument(
        "--methods",
        default="hhp,lsth,hhp-gated,lsth-gated,calibrated-hurdle",
        help="comma-separated methods: hhp,lsth,hhp-gated,lsth-gated,calibrated-hurdle",
    )
    parser.add_argument("--prob-alpha", type=float, default=0.08)
    parser.add_argument("--recent-prob-windows", type=int, default=120)
    parser.add_argument("--short-history-windows", type=int, default=None)
    parser.add_argument("--season-windows", type=int, default=None)
    parser.add_argument("--phase-bandwidth-windows", type=int, default=2)
    parser.add_argument("--gap-bandwidth-windows", type=int, default=1)
    parser.add_argument("--smoothing", type=float, default=5.0)
    parser.add_argument("--lsth-short-weight", type=float, default=0.35)
    parser.add_argument("--lsth-seasonal-weight", type=float, default=0.15)
    parser.add_argument("--calibration-windows", type=int, default=1440)
    parser.add_argument("--calibration-grid-size", type=int, default=80)
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--write-detail", action="store_true")
    return parser.parse_args()


def resolve_window_ms(args: argparse.Namespace) -> int:
    if args.window_ms is not None:
        if args.window_ms <= 0:
            raise ValueError("--window-ms must be positive")
        return args.window_ms
    if args.window_sec <= 0:
        raise ValueError("--window-sec must be positive")
    return args.window_sec * 1000


def default_short_history_windows(window_ms: int) -> int:
    # One day of windows is a useful short-term scale for Azure-like traces.
    return max(1, int(round(86_400_000 / window_ms)))


def default_season_windows(window_ms: int) -> int:
    return max(1, int(round(86_400_000 / window_ms)))


def ewma(values: np.ndarray, alpha: float) -> float:
    if len(values) == 0:
        return 0.0
    current = float(values[0])
    for value in values[1:]:
        current = alpha * float(value) + (1.0 - alpha) * current
    return current


def ceil_count(value: float) -> int:
    return int(math.ceil(max(0.0, float(value))))


def alloc_count(value: float, activation_threshold: float) -> int:
    value = max(0.0, float(value))
    if value < activation_threshold:
        return 0
    return ceil_count(value)


def weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0
    quantile = min(1.0, max(0.0, float(quantile)))
    if weights is None:
        return float(np.quantile(values, quantile))

    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(values):
        raise ValueError("weights and values must have the same length")
    if np.sum(weights) <= 0:
        return float(np.quantile(values, quantile))

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    return float(sorted_values[np.searchsorted(cdf, quantile, side="left")])


def circular_distance(a: np.ndarray, b: int, period: int) -> np.ndarray:
    raw = np.abs((a % period) - (b % period))
    return np.minimum(raw, period - raw)


def active_probability_fallback(counts: pd.Series, alpha: float, recent_windows: int) -> float:
    active = (counts.to_numpy(dtype=float) > 0).astype(float)
    if len(active) == 0:
        return 0.0
    smooth = ewma(active, alpha)
    recent = active[-max(1, recent_windows):]
    recent_rate = float(np.mean(recent)) if len(recent) else 0.0
    return min(1.0, max(0.0, 0.5 * smooth + 0.5 * recent_rate))


def hhp_active_probability(
    counts: pd.Series,
    horizon_step: int,
    alpha: float,
    recent_windows: int,
    gap_bandwidth: int,
    smoothing: float,
) -> float:
    fallback = active_probability_fallback(counts, alpha, recent_windows)
    active_positions = np.flatnonzero(counts.to_numpy(dtype=float) > 0)
    if len(active_positions) < 2:
        return fallback

    idle_age = int(len(counts) - 1 - active_positions[-1])
    target_gap = idle_age + max(1, int(horizon_step))
    gaps = np.diff(active_positions).astype(int)
    at_risk = gaps > idle_age
    risk_count = int(np.sum(at_risk))
    if risk_count <= 0:
        return fallback

    low = max(idle_age + 1, target_gap - max(0, gap_bandwidth))
    high = target_gap + max(0, gap_bandwidth)
    band_width = max(1, high - low + 1)
    events = int(np.sum(at_risk & (gaps >= low) & (gaps <= high)))

    # Convert a local-band event probability into a per-window probability.
    event_density = events / band_width
    probability = (event_density + smoothing * fallback) / (risk_count + smoothing)
    return min(1.0, max(0.0, float(probability)))


def seasonal_active_probability(
    counts: pd.Series,
    target_window: int,
    season_windows: int,
    phase_bandwidth: int,
    fallback: float,
    smoothing: float,
) -> float:
    if season_windows <= 1:
        return fallback
    windows = counts.index.to_numpy(dtype=int)
    if len(windows) == 0:
        return fallback
    distances = circular_distance(windows, target_window, season_windows)
    mask = distances <= max(0, phase_bandwidth)
    if not np.any(mask):
        return fallback
    active = (counts.to_numpy(dtype=float)[mask] > 0).astype(float)
    probability = (float(np.sum(active)) + smoothing * fallback) / (len(active) + smoothing)
    return min(1.0, max(0.0, float(probability)))


def hhp_components(
    counts: pd.Series,
    target_window: int,
    horizon_step: int,
    args: argparse.Namespace,
) -> ForecastComponents:
    p_active = hhp_active_probability(
        counts=counts,
        horizon_step=horizon_step,
        alpha=args.prob_alpha,
        recent_windows=args.recent_prob_windows,
        gap_bandwidth=args.gap_bandwidth_windows,
        smoothing=args.smoothing,
    )
    positive = counts[counts > 0].to_numpy(dtype=float)
    return build_hurdle_components(p_active, positive, None)


def lsth_positive_weights(
    counts: pd.Series,
    target_window: int,
    short_history_windows: int,
    season_windows: int,
    phase_bandwidth: int,
    short_weight: float,
    seasonal_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    positive_rows = counts[counts > 0]
    values = positive_rows.to_numpy(dtype=float)
    if len(values) == 0:
        return values, np.array([], dtype=float)

    windows = positive_rows.index.to_numpy(dtype=int)
    long_weight = max(0.0, 1.0 - short_weight - seasonal_weight)
    weights = np.full(len(values), max(0.05, long_weight), dtype=float)

    if short_history_windows > 0:
        weights += (windows >= target_window - short_history_windows).astype(float) * short_weight
    if season_windows > 1 and seasonal_weight > 0:
        same_phase = circular_distance(windows, target_window, season_windows) <= max(0, phase_bandwidth)
        weights += same_phase.astype(float) * seasonal_weight
    return values, weights


def lsth_components(
    counts: pd.Series,
    target_window: int,
    horizon_step: int,
    args: argparse.Namespace,
) -> ForecastComponents:
    short_history = int(args.short_history_windows)
    season = int(args.season_windows)
    short_weight = min(1.0, max(0.0, float(args.lsth_short_weight)))
    seasonal_weight = min(1.0 - short_weight, max(0.0, float(args.lsth_seasonal_weight)))
    long_weight = max(0.0, 1.0 - short_weight - seasonal_weight)

    long_prob = hhp_active_probability(
        counts=counts,
        horizon_step=horizon_step,
        alpha=args.prob_alpha,
        recent_windows=args.recent_prob_windows,
        gap_bandwidth=args.gap_bandwidth_windows,
        smoothing=args.smoothing,
    )

    short_counts = counts.iloc[-short_history:] if len(counts) > short_history else counts
    short_prob = hhp_active_probability(
        counts=short_counts,
        horizon_step=horizon_step,
        alpha=args.prob_alpha,
        recent_windows=min(args.recent_prob_windows, max(1, short_history)),
        gap_bandwidth=args.gap_bandwidth_windows,
        smoothing=args.smoothing,
    )

    fallback = active_probability_fallback(counts, args.prob_alpha, args.recent_prob_windows)
    seasonal_prob = seasonal_active_probability(
        counts=counts,
        target_window=target_window,
        season_windows=season,
        phase_bandwidth=args.phase_bandwidth_windows,
        fallback=fallback,
        smoothing=args.smoothing,
    )

    p_active = long_weight * long_prob + short_weight * short_prob + seasonal_weight * seasonal_prob
    p_active = min(1.0, max(0.0, float(p_active)))
    positive, weights = lsth_positive_weights(
        counts=counts,
        target_window=target_window,
        short_history_windows=short_history,
        season_windows=season,
        phase_bandwidth=args.phase_bandwidth_windows,
        short_weight=short_weight,
        seasonal_weight=seasonal_weight,
    )
    return build_hurdle_components(p_active, positive, weights)


def build_hurdle_components(
    p_active: float,
    positive_counts: np.ndarray,
    positive_weights: np.ndarray | None,
) -> ForecastComponents:
    conditional = {}
    hurdle = {}
    p_active = min(1.0, max(0.0, float(p_active)))
    for policy, q in POLICIES.items():
        conditional[policy] = weighted_quantile(positive_counts, q, positive_weights)
        if p_active <= 0.0 or len(positive_counts) == 0 or q <= (1.0 - p_active):
            hurdle[policy] = 0.0
        else:
            adjusted_q = (q - (1.0 - p_active)) / p_active
            hurdle[policy] = weighted_quantile(positive_counts, adjusted_q, positive_weights)
    return ForecastComponents(
        p_active=p_active,
        conditional_quantiles=conditional,
        hurdle_quantiles=hurdle,
    )


def forecast_components(
    method: str,
    counts: pd.Series,
    target_window: int,
    horizon_step: int,
    args: argparse.Namespace,
) -> ForecastComponents:
    if method in {"hhp", "hhp-gated"}:
        return hhp_components(counts, target_window, horizon_step, args)
    if method in {"lsth", "lsth-gated", "calibrated-hurdle"}:
        return lsth_components(counts, target_window, horizon_step, args)
    raise ValueError(f"unsupported method: {method}")


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


def forecast_count_for_policy(
    method: str,
    policy: str,
    components: ForecastComponents,
    thresholds: dict[str, float] | None,
) -> float:
    if method == "calibrated-hurdle":
        threshold = thresholds[policy] if thresholds else (1.0 - POLICIES[policy])
        if components.p_active >= threshold:
            return float(components.conditional_quantiles[policy])
        return 0.0
    if method in {"hhp-gated", "lsth-gated"}:
        threshold = 1.0 - POLICIES[policy]
        if components.p_active >= threshold:
            return float(components.conditional_quantiles[policy])
        return 0.0
    return float(components.hurdle_quantiles[policy])


def evaluate_policy_rows(rows: list[dict], window_ms: int) -> pd.DataFrame:
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    summaries = []
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
        summaries.append(
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
    return pd.DataFrame(summaries)


def collect_calibration_rows(
    train_counts: pd.Series,
    args: argparse.Namespace,
    calibration_windows: int,
) -> dict[str, pd.DataFrame]:
    rows_by_method = {"calibrated-hurdle": []}
    if len(train_counts) <= args.horizon_windows + 1:
        return {key: pd.DataFrame(value) for key, value in rows_by_method.items()}

    start_pos = max(0, len(train_counts) - calibration_windows - args.horizon_windows)
    end_pos = len(train_counts) - args.horizon_windows - 1
    for origin_pos in range(start_pos, end_pos + 1):
        history = train_counts.iloc[: origin_pos + 1]
        origin_window = int(history.index.max())
        for step in range(1, args.horizon_windows + 1):
            target_pos = origin_pos + step
            target_window = int(train_counts.index[target_pos])
            actual = int(train_counts.iloc[target_pos])
            components = forecast_components(
                "calibrated-hurdle",
                history,
                target_window,
                target_window - origin_window,
                args,
            )
            row = {
                "actual_count": actual,
                "p_active": components.p_active,
            }
            for policy in POLICIES:
                row[f"conditional_{policy}_count"] = components.conditional_quantiles[policy]
            rows_by_method["calibrated-hurdle"].append(row)
    return {key: pd.DataFrame(value) for key, value in rows_by_method.items()}


def choose_calibrated_thresholds(
    calibration: pd.DataFrame,
    grid_size: int,
    activation_threshold: float,
) -> dict[str, float]:
    if calibration.empty:
        return {policy: 1.0 - q for policy, q in POLICIES.items()}

    p_values = calibration["p_active"].to_numpy(dtype=float)
    max_p = max(float(np.max(p_values)), 1e-6)
    grid = np.linspace(0.0, max_p, max(2, grid_size))
    actual = calibration["actual_count"].to_numpy(dtype=float)
    actual_total = float(np.sum(actual))
    thresholds = {}

    for policy, q in POLICIES.items():
        cond = calibration[f"conditional_{policy}_count"].to_numpy(dtype=float)
        candidates = []
        for threshold in grid:
            forecast = np.where(p_values >= threshold, cond, 0.0)
            allocated = np.where(forecast >= activation_threshold, np.ceil(np.maximum(0.0, forecast)), 0.0)
            under = np.maximum(0.0, actual - allocated)
            allocated_total = float(np.sum(allocated))
            under_total = float(np.sum(under))
            demand_coverage = 1.0 - under_total / actual_total if actual_total > 0 else 1.0
            predicted_active = int(np.sum(allocated > 0))
            candidates.append((demand_coverage, allocated_total, predicted_active, float(threshold)))

        feasible = [item for item in candidates if item[0] >= q]
        if feasible:
            # Among coverage-satisfying thresholds, minimize allocation first.
            chosen = min(feasible, key=lambda item: (item[1], item[2], -item[3]))
        else:
            # If calibration cannot hit the nominal target, keep the highest coverage
            # and then minimize allocation among ties.
            chosen = min(candidates, key=lambda item: (-item[0], item[1], item[2], -item[3]))
        thresholds[policy] = chosen[3]
    return thresholds


def main() -> None:
    args = parse_args()
    if args.horizon_windows <= 0:
        raise ValueError("--horizon-windows must be positive")
    if args.origin_step_windows <= 0:
        raise ValueError("--origin-step-windows must be positive")

    window_ms = resolve_window_ms(args)
    if args.short_history_windows is None:
        args.short_history_windows = default_short_history_windows(window_ms)
    if args.season_windows is None:
        args.season_windows = default_season_windows(window_ms)

    workflow = load_workflow(args.workflow_config)
    workflow_name = workflow.workflow_name
    trace = pd.read_csv(args.trace)
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
    train_counts = counts[counts.index <= train_end_window]

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unsupported = sorted(set(methods) - {"hhp", "lsth", "hhp-gated", "lsth-gated", "calibrated-hurdle"})
    if unsupported:
        raise ValueError(f"unsupported methods: {unsupported}")

    calibration_rows = collect_calibration_rows(
        train_counts=train_counts,
        args=args,
        calibration_windows=max(1, args.calibration_windows),
    )
    calibrated_thresholds = choose_calibrated_thresholds(
        calibration_rows["calibrated-hurdle"],
        grid_size=args.calibration_grid_size,
        activation_threshold=args.activation_threshold,
    )

    rows = []
    last_origin = eval_end_window - 1
    for origin_window in range(train_end_window, last_origin + 1, args.origin_step_windows):
        history = counts[counts.index <= origin_window]
        if history.empty:
            continue
        horizon = min(args.horizon_windows, eval_end_window - origin_window)
        for step in range(1, horizon + 1):
            target_window = origin_window + step
            if target_window < eval_start_window or target_window > eval_end_window:
                continue
            actual = int(counts.get(target_window, 0.0))
            for method in methods:
                components = forecast_components(method, history, target_window, step, args)
                thresholds = calibrated_thresholds if method == "calibrated-hurdle" else None
                for policy in POLICIES:
                    forecast_count = forecast_count_for_policy(method, policy, components, thresholds)
                    allocated = alloc_count(forecast_count, args.activation_threshold)
                    rows.append(
                        {
                            "workflow_name": workflow_name,
                            "method": method,
                            "policy": policy,
                            "origin_window": int(origin_window),
                            "target_window": int(target_window),
                            "horizon_step": int(step),
                            "actual_count": actual,
                            "p_active": components.p_active,
                            "forecast_count": forecast_count,
                            "conditional_forecast_count": components.conditional_quantiles[policy],
                            "raw_hurdle_count": components.hurdle_quantiles[policy],
                            "allocated_count": allocated,
                            "under_count": max(0, actual - allocated),
                            "over_count": max(0, allocated - actual),
                        }
                    )

    detail = pd.DataFrame(rows)
    summary = evaluate_policy_rows(rows, window_ms)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{workflow_name}_histogram_entry_compare_summary.csv"
    detail_path = out_dir / f"{workflow_name}_histogram_entry_compare_detail.csv"
    metadata_path = out_dir / f"{workflow_name}_histogram_entry_compare_metadata.json"
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
        "origin_step_windows": args.origin_step_windows,
        "methods": methods,
        "policies": list(POLICIES.keys()),
        "train_end_window": train_end_window,
        "eval_start_window": eval_start_window,
        "eval_end_window": eval_end_window,
        "short_history_windows": args.short_history_windows,
        "season_windows": args.season_windows,
        "calibrated_thresholds": calibrated_thresholds,
        "calibration_rows": {
            key: int(len(value)) for key, value in calibration_rows.items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote {summary_path}")
    if args.write_detail:
        print(f"wrote {detail_path}")
    print(f"wrote {metadata_path}")
    if summary.empty:
        print("empty summary")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

