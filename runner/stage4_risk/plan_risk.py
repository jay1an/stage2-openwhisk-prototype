"""Unified analytical plan-risk API."""

from __future__ import annotations

from dataclasses import dataclass
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from runner.stage4_risk.dag_aggregation import (
    LogNormalParams,
    add_deterministic_shift,
    aggregate_civic_alert,
    fenton_wilkinson_sum,
)
from runner.stage4_risk.entry_cold import calibrated_entry_cold_probability
from runner.stage4_risk.scaling import scale_stage_for_memory_tier


BASE_MEMORY_MB = 1280
ENTRY_STAGE = "detect_object"
STAGES = [
    "detect_object",
    "estimate_pose",
    "match_face",
    "classify_scene",
    "translate_alert",
]
REPAIRED_V2_SYNC_SHIFT_WARM_MS = {
    "premium": 783.0,
    "free": 823.8,
}
REPAIRED_V2_SYNC_SHIFT_COLD_MS = {
    "premium": 51.1,
    "free": 43.0,
}
DEFAULT_REPAIRED_V2_COLD_OVERHEAD_TRACE = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "sweep_14tier_ceil_model"
    / "merged_trace.csv"
)


@dataclass
class PlanInput:
    memory_tier_per_stage: dict[str, int]
    entry_prewarm_count: float
    predicted_arrivals: float

    lognormal_params: dict[str, dict[str, LogNormalParams]]
    amdahl_params: pd.DataFrame
    cold_overhead_per_stage: dict[str, float]
    p_baseline: float


@dataclass
class PlanRiskResult:
    p_entry_cold: float
    e2e_warm_params: LogNormalParams
    e2e_cold_entry_params: LogNormalParams
    p_violation_warm: float
    p_violation_cold_entry: float
    p_violation_total: float
    expected_e2e_ms: float


def load_lognormal_params(params_csv_path: str | Path) -> dict[str, dict[str, LogNormalParams]]:
    df = pd.read_csv(params_csv_path)
    required = {"stage_name", "latency_class", "mu", "sigma"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"lognormal params missing required columns: {missing}")
    out: dict[str, dict[str, LogNormalParams]] = {}
    for row in df.itertuples(index=False):
        stage_name = str(getattr(row, "stage_name"))
        latency_class = str(getattr(row, "latency_class"))
        out.setdefault(stage_name, {})[latency_class] = LogNormalParams(
            mu=float(getattr(row, "mu")),
            sigma=float(getattr(row, "sigma")),
        )
    return out


def compute_cold_overhead_per_stage(
    lognormal_params: dict[str, dict[str, LogNormalParams]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for stage_name, by_class in lognormal_params.items():
        if "warm" not in by_class or "cold_like" not in by_class:
            continue
        out[stage_name] = max(0.0, by_class["cold_like"].mean - by_class["warm"].mean)
    return out


def _memory_for_stage(plan: PlanInput, stage_name: str) -> int:
    try:
        return int(plan.memory_tier_per_stage[stage_name])
    except KeyError as exc:
        raise ValueError(f"missing memory tier for stage={stage_name}") from exc


def _base_stage_params(plan: PlanInput, stage_name: str, latency_class: str) -> LogNormalParams:
    try:
        return plan.lognormal_params[stage_name][latency_class]
    except KeyError as exc:
        raise ValueError(f"missing lognormal params for stage={stage_name} class={latency_class}") from exc


def _scaled_stage(
    plan: PlanInput,
    stage_name: str,
    latency_class: str,
    base_memory_mb: int = BASE_MEMORY_MB,
    contention_factor: float = 1.0,
) -> LogNormalParams:
    return scale_stage_for_memory_tier(
        stage_name=stage_name,
        latency_class=latency_class,
        target_memory_mb=_memory_for_stage(plan, stage_name),
        base_memory_mb=base_memory_mb,
        base_params=_base_stage_params(plan, stage_name, latency_class),
        amdahl_params=plan.amdahl_params,
        cold_overhead_ms=plan.cold_overhead_per_stage.get(stage_name),
        contention_factor=contention_factor,
    )


def _scaled_scenario(
    plan: PlanInput, entry_cold: bool, contention_factor: float = 1.0
) -> dict[str, LogNormalParams]:
    stage_dists: dict[str, LogNormalParams] = {}
    for stage_name in STAGES:
        latency_class = "cold_like" if entry_cold and stage_name == ENTRY_STAGE else "warm"
        stage_dists[stage_name] = _scaled_stage(
            plan, stage_name, latency_class, contention_factor=contention_factor
        )
    return stage_dists


def _fit_lognormal_from_samples(samples: list[float]) -> LogNormalParams:
    values = [float(value) for value in samples if float(value) > 0.0 and math.isfinite(float(value))]
    if not values:
        raise ValueError("cannot fit lognormal from empty/invalid samples")
    if len(values) == 1:
        return LogNormalParams(mu=math.log(values[0]), sigma=0.0)
    logs = [math.log(value) for value in values]
    mean_log = sum(logs) / len(logs)
    variance = sum((value - mean_log) ** 2 for value in logs) / (len(logs) - 1)
    return LogNormalParams(mu=float(mean_log), sigma=math.sqrt(max(0.0, variance)))


def _fit_lognormal_from_mean_std(mean: float, std: float) -> LogNormalParams:
    if mean <= 0.0 or not math.isfinite(mean):
        raise ValueError(f"mean must be finite and positive, got {mean}")
    if std < 0.0 or not math.isfinite(std):
        raise ValueError(f"std must be finite and non-negative, got {std}")
    if std == 0.0:
        return LogNormalParams(mu=math.log(mean), sigma=0.0)
    sigma_sq = math.log1p((std * std) / (mean * mean))
    return LogNormalParams(mu=math.log(mean) - sigma_sq / 2.0, sigma=math.sqrt(sigma_sq))


@lru_cache(maxsize=None)
def _load_entry_cold_overhead_params(
    memory_mb: int,
    trace_path: str,
) -> LogNormalParams:
    """Fit detect cold-overhead distribution from sweep cold minus warm median."""

    path = Path(trace_path)
    if not path.exists():
        raise FileNotFoundError(f"repaired_v2 cold overhead trace not found: {path}")
    df = pd.read_csv(path)
    required = {"stage_name", "allocated_memory_mb", "cold_like", "dispatch_latency_ms"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    sub = df[(df["stage_name"] == ENTRY_STAGE) & (df["allocated_memory_mb"] == int(memory_mb))]
    if sub.empty:
        raise ValueError(f"no entry cold-overhead rows for {ENTRY_STAGE}@{memory_mb} in {path}")
    cold = (
        sub[sub["cold_like"].astype(str).str.lower().eq("true")]["dispatch_latency_ms"]
        .dropna()
        .astype(float)
        .tolist()
    )
    warm = (
        sub[sub["cold_like"].astype(str).str.lower().eq("false")]["dispatch_latency_ms"]
        .dropna()
        .astype(float)
        .tolist()
    )
    if not cold or not warm:
        raise ValueError(f"need both cold and warm samples for {ENTRY_STAGE}@{memory_mb} in {path}")
    warm_sorted = sorted(warm)
    mid = len(warm_sorted) // 2
    if len(warm_sorted) % 2:
        warm_median = warm_sorted[mid]
    else:
        warm_median = (warm_sorted[mid - 1] + warm_sorted[mid]) / 2.0
    overhead = [max(float(value) - warm_median, 1.0) for value in cold]
    mean = sum(overhead) / len(overhead)
    if len(overhead) == 1:
        std = 0.0
    else:
        std = math.sqrt(sum((value - mean) ** 2 for value in overhead) / (len(overhead) - 1))
    return _fit_lognormal_from_mean_std(mean, std)


def _validate_repaired_v2_class(slo_class: str | None) -> str:
    if slo_class not in REPAIRED_V2_SYNC_SHIFT_WARM_MS:
        allowed = sorted(REPAIRED_V2_SYNC_SHIFT_WARM_MS)
        raise ValueError(f"repaired_v2 requires slo_class in {allowed}, got {slo_class!r}")
    return str(slo_class)


def _validate_explicit_p_entry_cold(p_entry_cold: float | None) -> float:
    if p_entry_cold is None:
        raise ValueError("repaired_v2 requires explicit p_entry_cold")
    value = float(p_entry_cold)
    if not 0.0 <= value <= 1.0 or not math.isfinite(value):
        raise ValueError(f"p_entry_cold must be finite in [0, 1], got {p_entry_cold}")
    return value


def _compute_plan_risk_legacy(
    plan: PlanInput,
    slo_ms: float,
    rho: float,
    contention_factor: float,
) -> PlanRiskResult:
    p_entry_cold = calibrated_entry_cold_probability(
        predicted_arrivals=plan.predicted_arrivals,
        entry_prewarm_count=plan.entry_prewarm_count,
        zero_prewarm_cold_rate=plan.p_baseline,
        residual_floor=0.01,
    )

    warm_params = aggregate_civic_alert(
        _scaled_scenario(plan, entry_cold=False, contention_factor=contention_factor), rho=rho
    )
    cold_entry_params = aggregate_civic_alert(
        _scaled_scenario(plan, entry_cold=True, contention_factor=contention_factor), rho=rho
    )
    p_warm = warm_params.survival(float(slo_ms))
    p_cold_entry = cold_entry_params.survival(float(slo_ms))
    p_total = (1.0 - p_entry_cold) * p_warm + p_entry_cold * p_cold_entry
    expected_e2e_ms = (1.0 - p_entry_cold) * warm_params.mean + p_entry_cold * cold_entry_params.mean

    return PlanRiskResult(
        p_entry_cold=p_entry_cold,
        e2e_warm_params=warm_params,
        e2e_cold_entry_params=cold_entry_params,
        p_violation_warm=p_warm,
        p_violation_cold_entry=p_cold_entry,
        p_violation_total=p_total,
        expected_e2e_ms=expected_e2e_ms,
    )


def _compute_plan_risk_repaired_v2(
    plan: PlanInput,
    slo_ms: float,
    rho: float,
    contention_factor: float,
    *,
    slo_class: str | None,
    p_entry_cold: float | None,
    cold_overhead_trace_path: str | Path,
) -> PlanRiskResult:
    class_name = _validate_repaired_v2_class(slo_class)
    entry_cold_probability = _validate_explicit_p_entry_cold(p_entry_cold)
    warm_execution = aggregate_civic_alert(
        _scaled_scenario(plan, entry_cold=False, contention_factor=contention_factor), rho=rho
    )
    warm_params = add_deterministic_shift(
        warm_execution, REPAIRED_V2_SYNC_SHIFT_WARM_MS[class_name]
    )
    overhead_params = _load_entry_cold_overhead_params(
        _memory_for_stage(plan, ENTRY_STAGE), str(Path(cold_overhead_trace_path))
    )
    cold_entry_params = add_deterministic_shift(
        fenton_wilkinson_sum([warm_execution, overhead_params], rho=0.0),
        REPAIRED_V2_SYNC_SHIFT_COLD_MS[class_name],
    )
    p_warm = warm_params.survival(float(slo_ms))
    p_cold_entry = cold_entry_params.survival(float(slo_ms))
    p_total = (1.0 - entry_cold_probability) * p_warm + entry_cold_probability * p_cold_entry
    expected_e2e_ms = (
        (1.0 - entry_cold_probability) * warm_params.mean
        + entry_cold_probability * cold_entry_params.mean
    )

    return PlanRiskResult(
        p_entry_cold=entry_cold_probability,
        e2e_warm_params=warm_params,
        e2e_cold_entry_params=cold_entry_params,
        p_violation_warm=p_warm,
        p_violation_cold_entry=p_cold_entry,
        p_violation_total=p_total,
        expected_e2e_ms=expected_e2e_ms,
    )


def compute_plan_risk(
    plan: PlanInput,
    slo_ms: float,
    rho: float = 0.0,
    contention_factor: float = 1.0,
    *,
    risk_model: Literal["legacy", "repaired_v2"] = "legacy",
    slo_class: str | None = None,
    p_entry_cold: float | None = None,
    repaired_v2_cold_overhead_trace_path: str | Path = DEFAULT_REPAIRED_V2_COLD_OVERHEAD_TRACE,
) -> PlanRiskResult:
    """
    Compute P(E2E > SLO) for a plan using a two-scenario warm/cold-entry mixture.

    ``rho`` is the homogeneous inter-stage correlation passed to the
    Fenton-Wilkinson aggregation; ``rho=0`` keeps the legacy independent-sum
    behaviour. ``contention_factor`` inflates the per-stage warm mean to align
    the isolated spline with realized concurrent execution (~1.10); ``1.0``
    keeps the isolated baseline.
    """
    if slo_ms <= 0.0:
        raise ValueError(f"slo_ms must be positive, got {slo_ms}")
    if risk_model == "legacy":
        return _compute_plan_risk_legacy(plan, slo_ms, rho, contention_factor)
    if risk_model == "repaired_v2":
        return _compute_plan_risk_repaired_v2(
            plan,
            slo_ms,
            rho,
            contention_factor,
            slo_class=slo_class,
            p_entry_cold=p_entry_cold,
            cold_overhead_trace_path=repaired_v2_cold_overhead_trace_path,
        )
    raise ValueError(f"unknown risk_model={risk_model!r}")


def result_to_dict(result: PlanRiskResult) -> dict[str, Any]:
    return {
        "p_entry_cold": result.p_entry_cold,
        "warm_mu": result.e2e_warm_params.mu,
        "warm_sigma": result.e2e_warm_params.sigma,
        "warm_mean_ms": result.e2e_warm_params.mean,
        "cold_entry_mu": result.e2e_cold_entry_params.mu,
        "cold_entry_sigma": result.e2e_cold_entry_params.sigma,
        "cold_entry_mean_ms": result.e2e_cold_entry_params.mean,
        "p_violation_warm": result.p_violation_warm,
        "p_violation_cold_entry": result.p_violation_cold_entry,
        "p_violation_total": result.p_violation_total,
        "expected_e2e_ms": result.expected_e2e_ms,
    }
