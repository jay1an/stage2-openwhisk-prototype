import argparse
import json
import math
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_forecast_calibration import (
    REQUIRED_COLUMNS,
    infer_level,
    refresh_counts,
    risk_bins,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline replay of an online adaptive forecast selector. The script "
            "takes precomputed expert forecasts and, at each target window, chooses "
            "an expert using only previous-window realized performance."
        )
    )
    parser.add_argument("--detail", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--detail-level",
        choices=["all", "entry", "stage"],
        default="stage",
        help="which rows to select over",
    )
    parser.add_argument("--policies", default="p90,p95")
    parser.add_argument("--include-method-family", default="")
    parser.add_argument("--exclude-method-family", default="")
    parser.add_argument("--include-calibration-method", default="")
    parser.add_argument("--exclude-calibration-method", default="")
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument("--recent-windows", type=int, default=48)
    parser.add_argument("--min-history", type=int, default=12)
    parser.add_argument("--under-cost", type=float, default=10.0)
    parser.add_argument("--over-cost", type=float, default=1.0)
    parser.add_argument("--pinball-weight", type=float, default=1.0)
    parser.add_argument(
        "--allocation-weight",
        type=float,
        default=0.0,
        help="optional penalty per allocated replica-window",
    )
    parser.add_argument(
        "--calibration-weight",
        type=float,
        default=2.0,
        help="penalty multiplier for recent empirical quantile calibration error",
    )
    parser.add_argument(
        "--coverage-tolerance",
        type=float,
        default=None,
        help=(
            "optional hard constraint: prefer experts whose recent empirical "
            "coverage is at least nominal_quantile - tolerance"
        ),
    )
    parser.add_argument(
        "--risk-budget-policies",
        default="p95",
        help="comma-separated policies protected by risk-budget fallback, e.g. p95 or p90,p95",
    )
    parser.add_argument(
        "--risk-coverage-tolerance",
        type=float,
        default=0.0,
        help="fallback triggers when recent selected coverage < nominal_quantile - tolerance",
    )
    parser.add_argument(
        "--fallback-windows",
        type=int,
        default=6,
        help="number of windows to force safer experts after risk-budget breach",
    )
    parser.add_argument(
        "--safe-calibration-methods",
        default="rolling_conformal_stage",
        help="comma-separated calibration_method names considered safer experts",
    )
    parser.add_argument(
        "--safe-method-keywords",
        default="conformal",
        help="comma-separated substrings; matching method/expert names are considered safer",
    )
    parser.add_argument(
        "--fallback-mode",
        choices=["safest", "best-score", "coverage-first"],
        default="coverage-first",
        help="how to choose among safer experts during risk-budget fallback",
    )
    parser.add_argument(
        "--hysteresis-ratio",
        type=float,
        default=0.05,
        help="relative improvement required before switching experts",
    )
    parser.add_argument("--hysteresis-abs", type=float, default=0.0)
    parser.add_argument("--cooldown-windows", type=int, default=3)
    parser.add_argument(
        "--default-mode",
        choices=["safest", "cheapest", "first"],
        default="safest",
        help="choice before enough online history is available",
    )
    parser.add_argument("--risk-bins", type=int, default=8)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def parse_csv_list(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def load_detail(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        frame = frame.copy()
        if "detail_level" not in frame.columns:
            frame["detail_level"] = infer_level(frame, path.name)
        if "calibration_method" not in frame.columns:
            frame["calibration_method"] = "raw"
        frame["source_file"] = str(path)
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out["fold_id"] = out["fold_id"].astype(int)
    out["target_window"] = out["target_window"].astype(int)
    out["actual_count"] = out["actual_count"].astype(float)
    out["forecast_count"] = out["forecast_count"].astype(float)
    out["nominal_quantile"] = out["nominal_quantile"].astype(float)
    out["stage_name"] = out["stage_name"].astype(str)
    out["method_family"] = out["method_family"].astype(str)
    out["method"] = out["method"].astype(str)
    out["policy"] = out["policy"].astype(str)
    out["calibration_method"] = out["calibration_method"].astype(str)
    out["detail_level"] = out["detail_level"].astype(str)
    out = refresh_counts(out, activation_threshold=0.1)
    out["expert_id"] = (
        out["calibration_method"]
        + "|"
        + out["method_family"]
        + "|"
        + out["method"]
    )
    dedup = [
        "detail_level",
        "workflow_name",
        "method_family",
        "method",
        "calibration_method",
        "policy",
        "nominal_quantile",
        "stage_name",
        "target_window",
    ]
    return out.drop_duplicates(dedup, keep="first").reset_index(drop=True)


def filter_detail(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = frame.copy()
    if args.detail_level != "all":
        out = out[out["detail_level"] == args.detail_level].copy()
    policies = parse_csv_list(args.policies)
    if policies:
        out = out[out["policy"].isin(policies)].copy()

    include_families = parse_csv_list(args.include_method_family)
    exclude_families = parse_csv_list(args.exclude_method_family)
    include_calibrations = parse_csv_list(args.include_calibration_method)
    exclude_calibrations = parse_csv_list(args.exclude_calibration_method)
    if include_families:
        out = out[out["method_family"].isin(include_families)].copy()
    if exclude_families:
        out = out[~out["method_family"].isin(exclude_families)].copy()
    if include_calibrations:
        out = out[out["calibration_method"].isin(include_calibrations)].copy()
    if exclude_calibrations:
        out = out[~out["calibration_method"].isin(exclude_calibrations)].copy()
    if out.empty:
        raise ValueError("no forecast rows remain after filtering")
    return refresh_counts(out, args.activation_threshold)


def expert_row_cost(row: pd.Series, args: argparse.Namespace) -> float:
    return float(
        args.under_cost * float(row["under_count"])
        + args.over_cost * float(row["over_count"])
        + args.pinball_weight * float(row["pinball_loss"])
        + args.allocation_weight * float(row["allocated_count"])
    )


def default_select(candidates: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "safest":
        ordered = candidates.sort_values(
            ["allocated_count", "forecast_count", "expert_id"],
            ascending=[False, False, True],
        )
    elif mode == "cheapest":
        ordered = candidates.sort_values(
            ["allocated_count", "forecast_count", "expert_id"],
            ascending=[True, True, True],
        )
    else:
        ordered = candidates.sort_values(["expert_id"])
    return ordered.iloc[0]


def safest_select(candidates: pd.DataFrame) -> pd.Series:
    return candidates.sort_values(
        ["allocated_count", "forecast_count", "expert_id"],
        ascending=[False, False, True],
    ).iloc[0]


def safe_candidate_subset(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    safe_calibrations = parse_csv_list(args.safe_calibration_methods)
    keywords = {value.lower() for value in parse_csv_list(args.safe_method_keywords)}
    mask = pd.Series(False, index=candidates.index)
    if safe_calibrations:
        mask = mask | candidates["calibration_method"].astype(str).isin(safe_calibrations)
    if keywords:
        searchable = (
            candidates["expert_id"].astype(str)
            + "|"
            + candidates["method_family"].astype(str)
            + "|"
            + candidates["method"].astype(str)
            + "|"
            + candidates["calibration_method"].astype(str)
        ).str.lower()
        for keyword in keywords:
            mask = mask | searchable.str.contains(keyword, regex=False)
    safe = candidates[mask].copy()
    return safe if not safe.empty else candidates.copy()


def fallback_select(
    candidates: pd.DataFrame,
    scores: dict[str, tuple[float, dict[str, float]]],
    args: argparse.Namespace,
) -> tuple[pd.Series, str]:
    safe = safe_candidate_subset(candidates, args)
    if args.fallback_mode == "safest":
        return safest_select(safe), "risk-budget-fallback-safest"

    finite_rows = []
    for _, candidate in safe.iterrows():
        expert_id = str(candidate["expert_id"])
        score, diagnostics = scores.get(expert_id, (math.inf, {}))
        if math.isfinite(score):
            finite_rows.append(
                {
                    "expert_id": expert_id,
                    "score": float(score),
                    "recent_empirical_coverage": float(
                        diagnostics.get("recent_empirical_coverage", -1.0)
                    ),
                }
            )
    if not finite_rows:
        return safest_select(safe), "risk-budget-fallback-safest-no-history"

    finite = pd.DataFrame(finite_rows)
    if args.fallback_mode == "coverage-first":
        chosen_expert = finite.sort_values(
            ["recent_empirical_coverage", "score", "expert_id"],
            ascending=[False, True, True],
        ).iloc[0]["expert_id"]
        return (
            safe[safe["expert_id"] == chosen_expert].iloc[0],
            "risk-budget-fallback-coverage-first",
        )

    chosen_expert = finite.sort_values(["score", "expert_id"], ascending=[True, True]).iloc[
        0
    ]["expert_id"]
    return safe[safe["expert_id"] == chosen_expert].iloc[0], "risk-budget-fallback-best-score"


def recent_expert_score(
    history: deque[dict],
    nominal: float,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    recent = list(history)[-max(1, args.recent_windows) :]
    if len(recent) < args.min_history:
        return math.inf, {
            "history_rows": float(len(recent)),
            "recent_cost": math.nan,
            "recent_empirical_coverage": math.nan,
            "recent_calibration_error": math.nan,
        }
    costs = np.asarray([item["cost"] for item in recent], dtype=float)
    hits = np.asarray([item["quantile_hit"] for item in recent], dtype=float)
    empirical = float(hits.mean())
    calibration_error = abs(empirical - float(nominal))
    score = float(costs.mean() + args.calibration_weight * calibration_error)
    return score, {
        "history_rows": float(len(recent)),
        "recent_cost": float(costs.mean()),
        "recent_empirical_coverage": empirical,
        "recent_calibration_error": calibration_error,
    }


def recent_selected_coverage(
    selected_history: deque[dict],
    recent_windows: int,
    min_history: int,
) -> tuple[float, int]:
    recent = list(selected_history)[-max(1, recent_windows) :]
    if len(recent) < min_history:
        return math.nan, len(recent)
    hits = np.asarray([item["quantile_hit"] for item in recent], dtype=float)
    return float(hits.mean()), len(recent)


def series_features(actual_history: list[float], recent_windows: int) -> dict[str, float | int | str]:
    if not actual_history:
        return {
            "recent_active_ratio": math.nan,
            "recent_mean": math.nan,
            "recent_std": math.nan,
            "recent_cv": math.nan,
            "time_since_last_active": -1,
            "online_regime_hint": "cold_start",
        }
    recent = np.asarray(actual_history[-max(1, recent_windows) :], dtype=float)
    active = recent > 0
    active_ratio = float(active.mean())
    mean = float(recent.mean())
    std = float(recent.std())
    cv = float(std / mean) if mean > 0 else math.inf
    last_active_positions = np.flatnonzero(np.asarray(actual_history, dtype=float) > 0)
    if len(last_active_positions) == 0:
        time_since_last_active = len(actual_history)
    else:
        time_since_last_active = len(actual_history) - 1 - int(last_active_positions[-1])

    if active_ratio < 0.05:
        regime = "mostly_idle"
    elif time_since_last_active >= max(6, recent_windows // 3):
        regime = "idle_gap"
    elif cv >= 2.0:
        regime = "bursty_or_drift"
    elif active_ratio >= 0.8 and cv < 1.0:
        regime = "continuous"
    else:
        regime = "mixed"

    return {
        "recent_active_ratio": active_ratio,
        "recent_mean": mean,
        "recent_std": std,
        "recent_cv": cv,
        "time_since_last_active": int(time_since_last_active),
        "online_regime_hint": regime,
    }


def selection_group_columns() -> list[str]:
    return ["detail_level", "workflow_name", "policy", "nominal_quantile", "stage_name"]


def choose_for_series(
    group: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    candidate_rows = []
    feature_rows = []
    history_by_expert: dict[str, deque[dict]] = defaultdict(
        lambda: deque(maxlen=max(args.recent_windows * 3, args.min_history * 3, 64))
    )
    selected_history: deque[dict] = deque(
        maxlen=max(args.recent_windows * 3, args.min_history * 3, 64)
    )
    actual_history: list[float] = []
    current_expert: str | None = None
    cooldown_remaining = 0
    fallback_remaining = 0
    risk_budget_policies = parse_csv_list(args.risk_budget_policies)

    group = group.sort_values(["target_window", "expert_id"]).copy()
    nominal = float(group["nominal_quantile"].iloc[0])
    series_key = {
        col: group[col].iloc[0] for col in selection_group_columns()
    }

    for target_window, candidates in group.groupby("target_window", sort=True):
        decision_start = time.perf_counter()
        candidates = candidates.copy()
        features = series_features(actual_history, args.recent_windows)
        selected_coverage, selected_coverage_rows = recent_selected_coverage(
            selected_history, args.recent_windows, args.min_history
        )
        risk_budget_triggered = False
        risk_threshold = nominal - max(0.0, float(args.risk_coverage_tolerance))
        if (
            args.fallback_windows > 0
            and str(group["policy"].iloc[0]) in risk_budget_policies
            and math.isfinite(selected_coverage)
            and selected_coverage < risk_threshold
        ):
            risk_budget_triggered = True
            fallback_remaining = max(fallback_remaining, int(args.fallback_windows))

        scores: dict[str, tuple[float, dict[str, float]]] = {}
        for _, candidate in candidates.iterrows():
            expert_id = str(candidate["expert_id"])
            score, diagnostics = recent_expert_score(
                history_by_expert[expert_id], nominal, args
            )
            scores[expert_id] = (score, diagnostics)

        finite_scores = {
            expert_id: value
            for expert_id, (value, _) in scores.items()
            if math.isfinite(value)
        }

        switched = False
        risk_fallback_active = fallback_remaining > 0
        if risk_fallback_active:
            chosen, reason = fallback_select(candidates, scores, args)
            best_score = scores.get(str(chosen["expert_id"]), (math.nan, {}))[0]
            current_score = (
                scores.get(str(current_expert), (math.nan, {}))[0]
                if current_expert is not None
                else math.nan
            )
            switched = current_expert is not None and str(chosen["expert_id"]) != current_expert
            if switched:
                cooldown_remaining = args.cooldown_windows
            fallback_remaining = max(0, fallback_remaining - 1)
        elif not finite_scores:
            chosen = default_select(candidates, args.default_mode)
            reason = f"cold-start-default-{args.default_mode}"
            best_score = math.nan
            current_score = math.nan
        else:
            feasible_scores = finite_scores
            if args.coverage_tolerance is not None:
                threshold = nominal - max(0.0, float(args.coverage_tolerance))
                constrained = {
                    expert_id: score
                    for expert_id, score in finite_scores.items()
                    if scores[expert_id][1]["recent_empirical_coverage"] >= threshold
                }
                if constrained:
                    feasible_scores = constrained
            best_expert = min(feasible_scores, key=feasible_scores.get)
            best_score = finite_scores[best_expert]
            if current_expert in set(candidates["expert_id"]) and cooldown_remaining > 0:
                chosen = candidates[candidates["expert_id"] == current_expert].iloc[0]
                current_score = scores[str(current_expert)][0]
                reason = "cooldown-keep-current"
                cooldown_remaining -= 1
            elif current_expert not in set(candidates["expert_id"]):
                chosen = candidates[candidates["expert_id"] == best_expert].iloc[0]
                current_score = math.nan
                reason = "best-recent-score-current-missing"
                switched = current_expert is not None
                cooldown_remaining = args.cooldown_windows if switched else 0
            else:
                current_score = scores[str(current_expert)][0]
                improvement = current_score - best_score
                threshold = max(
                    args.hysteresis_abs,
                    abs(current_score) * max(0.0, args.hysteresis_ratio),
                )
                if best_expert != current_expert and improvement > threshold:
                    chosen = candidates[candidates["expert_id"] == best_expert].iloc[0]
                    reason = "switch-best-score-after-hysteresis"
                    switched = True
                    cooldown_remaining = args.cooldown_windows
                else:
                    chosen = candidates[candidates["expert_id"] == current_expert].iloc[0]
                    reason = "hysteresis-keep-current"

        previous_expert = current_expert
        current_expert = str(chosen["expert_id"])
        chosen_score, chosen_diag = scores.get(
            current_expert,
            (
                math.nan,
                {
                    "history_rows": math.nan,
                    "recent_cost": math.nan,
                    "recent_empirical_coverage": math.nan,
                    "recent_calibration_error": math.nan,
                },
            ),
        )
        selected = chosen.to_dict()
        decision_ms = (time.perf_counter() - decision_start) * 1000.0
        selected.update(features)
        selected.update(
            {
                "selected_expert_id": current_expert,
                "previous_expert_id": previous_expert or "",
                "candidate_experts": int(candidates["expert_id"].nunique()),
                "selection_reason": reason,
                "selector_decision_ms": float(decision_ms),
                "selected_recent_score": chosen_score,
                "best_recent_score": best_score,
                "previous_current_score": current_score,
                "switched": bool(switched),
                "cooldown_remaining": int(cooldown_remaining),
                "risk_budget_triggered": bool(risk_budget_triggered),
                "risk_fallback_active": bool(risk_fallback_active),
                "risk_fallback_remaining": int(fallback_remaining),
                "selected_recent_quantile_coverage": selected_coverage,
                "selected_recent_coverage_rows": int(selected_coverage_rows),
                "risk_coverage_threshold": float(risk_threshold),
                **{f"selected_{k}": v for k, v in chosen_diag.items()},
            }
        )

        selected["source_method_family"] = selected["method_family"]
        selected["source_method"] = selected["method"]
        selected["source_calibration_method"] = selected["calibration_method"]
        selected["method_family"] = "online-adaptive-selector"
        selected["method"] = "online-adaptive-expert-bank"
        selected["calibration_method"] = "online_adaptive"
        selected_rows.append(selected)

        for _, candidate in candidates.iterrows():
            expert_id = str(candidate["expert_id"])
            score, diagnostics = scores[expert_id]
            candidate_out = candidate.to_dict()
            candidate_out.update(features)
            candidate_out.update(
                {
                    "candidate_recent_score": score,
                    "candidate_history_rows": diagnostics["history_rows"],
                    "candidate_recent_cost": diagnostics["recent_cost"],
                    "candidate_recent_empirical_coverage": diagnostics[
                        "recent_empirical_coverage"
                    ],
                    "candidate_recent_calibration_error": diagnostics[
                        "recent_calibration_error"
                    ],
                    "selected": expert_id == current_expert,
                    "selected_expert_id": current_expert,
                }
            )
            candidate_rows.append(candidate_out)

        feature_rows.append(
            {
                **series_key,
                "target_window": int(target_window),
                "actual_count": float(chosen["actual_count"]),
                "selected_expert_id": current_expert,
                "selection_reason": reason,
                "switched": bool(switched),
                **features,
            }
        )

        for _, candidate in candidates.iterrows():
            history_by_expert[str(candidate["expert_id"])].append(
                {
                    "target_window": int(candidate["target_window"]),
                    "cost": expert_row_cost(candidate, args),
                    "quantile_hit": float(bool(candidate["quantile_hit"])),
                    "under_count": float(candidate["under_count"]),
                    "over_count": float(candidate["over_count"]),
                    "allocated_count": float(candidate["allocated_count"]),
                    "pinball_loss": float(candidate["pinball_loss"]),
                }
            )
        selected_history.append(
            {
                "target_window": int(chosen["target_window"]),
                "quantile_hit": float(bool(chosen["quantile_hit"])),
                "under_count": float(chosen["under_count"]),
                "over_count": float(chosen["over_count"]),
                "allocated_count": float(chosen["allocated_count"]),
                "pinball_loss": float(chosen["pinball_loss"]),
            }
        )
        actual_history.append(float(chosen["actual_count"]))

    return (
        pd.DataFrame(selected_rows),
        pd.DataFrame(candidate_rows),
        pd.DataFrame(feature_rows),
    )


def online_select(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_parts = []
    candidate_parts = []
    feature_parts = []
    for _, group in frame.groupby(selection_group_columns(), dropna=False):
        selected, candidates, features = choose_for_series(group, args)
        selected_parts.append(selected)
        candidate_parts.append(candidates)
        feature_parts.append(features)
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    candidates = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    features = pd.concat(feature_parts, ignore_index=True) if feature_parts else pd.DataFrame()
    selected = refresh_counts(selected, args.activation_threshold)
    return selected, candidates, features


def switch_summary(selected: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["detail_level", "workflow_name", "policy", "stage_name"]
    rows = []
    for keys, group in selected.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "windows": int(len(group)),
                "switches": int(group["switched"].astype(bool).sum()),
                "switch_rate": float(group["switched"].astype(bool).mean())
                if len(group)
                else 0.0,
                "unique_experts_selected": int(group["selected_expert_id"].nunique()),
                "most_common_expert": str(group["selected_expert_id"].mode().iloc[0])
                if len(group)
                else "",
                "most_common_reason": str(group["selection_reason"].mode().iloc[0])
                if len(group)
                else "",
            }
        )
    return pd.DataFrame(rows)


def expert_usage(selected: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "detail_level",
        "workflow_name",
        "policy",
        "source_calibration_method",
        "source_method_family",
        "source_method",
        "selected_expert_id",
    ]
    rows = []
    for keys, group in selected.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "selected_rows": int(len(group)),
                "selected_fraction": float(len(group) / len(selected)) if len(selected) else 0.0,
                "actual_total": int(group["actual_count"].sum()),
                "allocated_replica_windows": int(group["allocated_count"].sum()),
                "under_total": int(group["under_count"].sum()),
                "over_total": int(group["over_count"].sum()),
                "mean_recent_active_ratio": float(group["recent_active_ratio"].mean()),
            }
        )
    return pd.DataFrame(rows)


def selector_latency_summary(selected: pd.DataFrame) -> pd.DataFrame:
    if "selector_decision_ms" not in selected.columns:
        return pd.DataFrame()
    rows = []
    group_cols = ["detail_level", "workflow_name", "policy"]
    for keys, group in selected.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["selector_decision_ms"].astype(float).to_numpy()
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "decision_rows": int(len(values)),
                "mean_selector_decision_ms": float(np.mean(values)),
                "p50_selector_decision_ms": float(np.quantile(values, 0.50)),
                "p90_selector_decision_ms": float(np.quantile(values, 0.90)),
                "p95_selector_decision_ms": float(np.quantile(values, 0.95)),
                "p99_selector_decision_ms": float(np.quantile(values, 0.99)),
                "max_selector_decision_ms": float(np.max(values)),
                "total_selector_decision_ms": float(np.sum(values)),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy()
    try:
        return display.to_markdown(index=False)
    except Exception:
        return display.to_csv(index=False)


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    warm_summary: pd.DataFrame,
    by_stage: pd.DataFrame,
    switches: pd.DataFrame,
    usage: pd.DataFrame,
    latency: pd.DataFrame,
) -> None:
    compact_cols = [
        "detail_level",
        "method_family",
        "method",
        "policy",
        "demand_coverage_rate",
        "allocated_replica_seconds",
        "over_allocation_ratio",
        "allocation_utilization",
        "empirical_quantile_coverage",
        "quantile_calibration_error",
        "pinball_loss_mean",
    ]
    compact = summary[[col for col in compact_cols if col in summary.columns]]
    warm_compact = warm_summary[[col for col in compact_cols if col in warm_summary.columns]]
    lines = [
        "# Online Adaptive Forecast Selector",
        "",
        "## Scope",
        "",
        "- Offline replay of an online selector over a precomputed expert forecast bank.",
        "- At each target window, the selector uses only previous-window expert errors.",
        "- Hysteresis and cooldown reduce control oscillation.",
        "- Training and model inference are outside this replay script; this isolates selector behavior.",
        "",
        "## Selected Forecast Summary",
        "",
        markdown_table(compact),
        "",
        "## Selected Forecast Summary After Warm-Up",
        "",
        "The first fold is reported separately because online selectors have little expert history at startup.",
        "",
        markdown_table(warm_compact),
        "",
        "## Expert Usage",
        "",
        markdown_table(usage.sort_values(["policy", "selected_rows"], ascending=[True, False])),
        "",
        "## Selector Decision Latency",
        "",
        markdown_table(latency),
        "",
        "## Switch Summary",
        "",
        markdown_table(switches.sort_values(["policy", "switch_rate"], ascending=[True, False])),
        "",
        "## Stage Summary",
        "",
        markdown_table(
            by_stage[
                [col for col in compact_cols + ["stage_name"] if col in by_stage.columns]
            ].sort_values(["policy", "stage_name"]),
            max_rows=30,
        ),
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = project_root()
    detail_paths = [resolve_path(root, value) for value in args.detail]
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = load_detail(detail_paths)
    bank = filter_detail(bank, args)
    selected, candidate_scores, features = online_select(bank, args)
    summary = summarize(selected)
    first_fold = int(selected["fold_id"].min())
    warm_selected = selected[selected["fold_id"] > first_fold].copy()
    warm_summary = summarize(warm_selected) if not warm_selected.empty else pd.DataFrame()
    by_stage = summarize(selected, extra_cols=["stage_name"])
    by_fold = summarize(selected, extra_cols=["fold_id"])
    switches = switch_summary(selected)
    usage = expert_usage(selected)
    latency = selector_latency_summary(selected)
    bins = risk_bins(selected, bins=max(1, args.risk_bins))

    bank.to_csv(out_dir / "expert_bank_detail.csv", index=False)
    selected.to_csv(out_dir / "online_selected_detail.csv", index=False)
    candidate_scores.to_csv(out_dir / "online_candidate_scores.csv", index=False)
    features.to_csv(out_dir / "online_regime_features.csv", index=False)
    summary.to_csv(out_dir / "online_selected_summary.csv", index=False)
    warm_summary.to_csv(out_dir / "online_selected_after_warmup_summary.csv", index=False)
    by_stage.to_csv(out_dir / "online_selected_by_stage.csv", index=False)
    by_fold.to_csv(out_dir / "online_selected_by_fold.csv", index=False)
    switches.to_csv(out_dir / "online_policy_switches.csv", index=False)
    usage.to_csv(out_dir / "online_expert_usage.csv", index=False)
    latency.to_csv(out_dir / "online_selector_latency.csv", index=False)
    bins.to_csv(out_dir / "online_risk_bin_table.csv", index=False)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "detail_files": [str(path) for path in detail_paths],
        "out_dir": str(out_dir),
        "detail_level": args.detail_level,
        "policies": args.policies,
        "activation_threshold": args.activation_threshold,
        "recent_windows": args.recent_windows,
        "min_history": args.min_history,
        "under_cost": args.under_cost,
        "over_cost": args.over_cost,
        "pinball_weight": args.pinball_weight,
        "allocation_weight": args.allocation_weight,
        "calibration_weight": args.calibration_weight,
        "coverage_tolerance": args.coverage_tolerance,
        "hysteresis_ratio": args.hysteresis_ratio,
        "hysteresis_abs": args.hysteresis_abs,
        "cooldown_windows": args.cooldown_windows,
        "default_mode": args.default_mode,
        "risk_budget_policies": args.risk_budget_policies,
        "risk_coverage_tolerance": args.risk_coverage_tolerance,
        "fallback_windows": args.fallback_windows,
        "safe_calibration_methods": args.safe_calibration_methods,
        "safe_method_keywords": args.safe_method_keywords,
        "fallback_mode": args.fallback_mode,
        "expert_rows": int(len(bank)),
        "selected_rows": int(len(selected)),
        "warmup_excluded_fold_id": first_fold,
        "unique_experts": int(bank["expert_id"].nunique()),
        "notes": [
            "This is an offline replay of online selection, not online model training.",
            "Every expert may produce a forecast each window; the selector observes all expert losses after the window completes.",
            "The selected output rewrites method_family/method to online-adaptive-selector while preserving source expert columns.",
        ],
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(out_dir, summary, warm_summary, by_stage, switches, usage, latency)
    print(f"wrote online adaptive selector report to {out_dir}")


if __name__ == "__main__":
    main()

