"""Unified analytical plan-risk API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from runner.stage4_risk.dag_aggregation import LogNormalParams, aggregate_civic_alert
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


def compute_plan_risk(
    plan: PlanInput, slo_ms: float, rho: float = 0.0, contention_factor: float = 1.0
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
