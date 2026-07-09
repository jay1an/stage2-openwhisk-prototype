"""Orion-style right-sizing baseline.

This module intentionally uses Orion's own latency model instead of the
project's FW/Clark analytical model:

* root stage is cold-like; all other stages are warm;
* stages are independent, with no contention or cross-stage correlation;
* DAG latency is propagated numerically on a fixed time grid using PMFs.

The search follows ORION Algorithm 1's successor expansion shape: every stage
can be upgraded by one tier when expanding a state. ORION/SMIless-style
descriptions often write the priority as ``-latency * cost``. In this Python
implementation the queue is ``heapq`` (a min-heap), so we push
``+latency * cost`` to pop the smaller latency-cost product first. Using the
negative value directly with ``heapq`` reverses the intended ordering.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import ndtr

from runner.stage4_risk.dag_aggregation import LogNormalParams
from runner.stage4_risk.plan_risk import BASE_MEMORY_MB, PlanInput, compute_plan_risk
from runner.stage4_risk.scaling import scale_stage_for_memory_tier
from runner.stage5_control.multi_slo_planner import (
    PlannerConfig,
    ReferenceData,
    plan_cost_gbsec,
)
from runner.workflow import WorkflowSpec


DEFAULT_BIN_MS = 25.0
DEFAULT_MAX_MS = 60000.0
DEFAULT_EXPANSION_LIMIT = 2000
EPS = 1e-12


@dataclass(frozen=True)
class OrionStageDistribution:
    params: LogNormalParams
    pmf: np.ndarray


@dataclass(frozen=True)
class OrionEvaluation:
    state_key: tuple[int, ...]
    memory_tier_per_stage: dict[str, int]
    cost_gbsec: float
    p95_ms: float
    survival_at_slo: float
    feasible_by_orion: bool
    critical_path: tuple[str, ...]
    e2e_pmf: np.ndarray
    finish_pmfs: dict[str, np.ndarray]


@dataclass(frozen=True)
class OrionPlanResult:
    memory_tier_per_stage: dict[str, int]
    cost_gbsec: float
    orion_p95_ms: float
    orion_own_survival: float
    feasible_by_orion: bool
    feasible_by_ours: bool
    violation_rate: float
    expected_e2e_ms: float
    states_expanded: int
    states_evaluated: int
    search_exhausted: bool


def topological_stage_names(workflow: WorkflowSpec) -> list[str]:
    remaining = list(workflow.nodes)
    seen: set[str] = set()
    ordered: list[str] = []
    while remaining:
        progressed = False
        for stage_name in list(remaining):
            if all(parent in seen for parent in workflow.nodes[stage_name].parents):
                ordered.append(stage_name)
                seen.add(stage_name)
                remaining.remove(stage_name)
                progressed = True
        if not progressed:
            raise RuntimeError(f"workflow has a cycle or missing parent; remaining={remaining}")
    return ordered


def sink_stages(workflow: WorkflowSpec) -> list[str]:
    return [stage for stage in workflow.nodes if not workflow.children_of(stage)]


def _time_grid(bin_ms: float, max_ms: float) -> np.ndarray:
    if bin_ms <= 0.0 or max_ms <= 0.0:
        raise ValueError("bin_ms and max_ms must be positive")
    n_bins = int(math.ceil(max_ms / bin_ms))
    return np.arange(n_bins + 1, dtype=float) * float(bin_ms)


def lognormal_to_pmf(
    params: LogNormalParams,
    *,
    bin_ms: float = DEFAULT_BIN_MS,
    max_ms: float = DEFAULT_MAX_MS,
) -> np.ndarray:
    """Discretize a lognormal into interval masses over (edge_i, edge_{i+1}]."""

    edges = _time_grid(bin_ms, max_ms)
    if params.sigma == 0.0:
        cdf = (edges >= math.exp(params.mu)).astype(float)
    else:
        z = np.full_like(edges, -np.inf, dtype=float)
        positive = edges > 0.0
        z[positive] = (np.log(edges[positive]) - params.mu) / params.sigma
        cdf = ndtr(z)
    pmf = np.diff(cdf)
    overflow = max(0.0, 1.0 - float(cdf[-1]))
    if overflow:
        pmf[-1] += overflow
    total = float(pmf.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("invalid PMF mass while discretizing lognormal")
    pmf = np.maximum(pmf / total, 0.0)
    pmf /= pmf.sum()
    return pmf


def pmf_cdf(pmf: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(pmf)
    cdf[-1] = 1.0
    return cdf


def pmf_quantile(pmf: np.ndarray, q: float, *, bin_ms: float) -> float:
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0,1), got {q}")
    index = int(np.searchsorted(pmf_cdf(pmf), q, side="left"))
    return float((index + 1) * bin_ms)


def pmf_survival(pmf: np.ndarray, t_ms: float, *, bin_ms: float) -> float:
    if t_ms < 0.0:
        return 1.0
    index = int(math.floor(t_ms / bin_ms)) - 1
    if index < 0:
        return 1.0
    cdf = pmf_cdf(pmf)
    if index >= len(cdf):
        return 0.0
    return max(0.0, 1.0 - float(cdf[index]))


def pmf_mean(pmf: np.ndarray, *, bin_ms: float) -> float:
    centers = (np.arange(len(pmf), dtype=float) + 0.5) * float(bin_ms)
    return float(np.dot(pmf, centers))


def max_independent_pmfs(pmfs: list[np.ndarray]) -> np.ndarray:
    if not pmfs:
        raise ValueError("max_independent_pmfs requires at least one PMF")
    if len(pmfs) == 1:
        return pmfs[0]
    cdf = np.ones_like(pmfs[0], dtype=float)
    for pmf in pmfs:
        cdf *= pmf_cdf(pmf)
    pmf = np.diff(np.concatenate(([0.0], cdf)))
    pmf = np.maximum(pmf, 0.0)
    pmf /= pmf.sum()
    return pmf


def convolve_pmfs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    conv = np.convolve(a, b)
    n = len(a)
    if len(conv) > n:
        trimmed = conv[:n].copy()
        overflow = float(conv[n:].sum())
        trimmed[-1] += overflow
    else:
        trimmed = np.pad(conv, (0, n - len(conv)))
    trimmed = np.maximum(trimmed, 0.0)
    trimmed /= trimmed.sum()
    return trimmed


def orion_stage_distribution(
    *,
    stage_name: str,
    memory_mb: int,
    latency_class: str,
    ref_data: ReferenceData,
    bin_ms: float = DEFAULT_BIN_MS,
    max_ms: float = DEFAULT_MAX_MS,
) -> OrionStageDistribution:
    base_params = ref_data.lognormal_params[stage_name][latency_class]
    params = scale_stage_for_memory_tier(
        stage_name=stage_name,
        latency_class=latency_class,
        target_memory_mb=int(memory_mb),
        base_memory_mb=BASE_MEMORY_MB,
        base_params=base_params,
        amdahl_params=ref_data.amdahl_params,
        splines=ref_data.warm_splines,
        contention_factor=1.0,
    )
    return OrionStageDistribution(
        params=params,
        pmf=lognormal_to_pmf(params, bin_ms=bin_ms, max_ms=max_ms),
    )


def _state_to_memory(state_key: tuple[int, ...], config: PlannerConfig) -> dict[str, int]:
    return {
        stage_name: int(config.tiers[int(state_key[index])])
        for index, stage_name in enumerate(config.stages)
    }


def _critical_path(
    *,
    workflow: WorkflowSpec,
    finish_pmfs: dict[str, np.ndarray],
    sink: str,
    bin_ms: float,
) -> tuple[str, ...]:
    path = [sink]
    current = sink
    while workflow.nodes[current].parents:
        parent = max(
            workflow.nodes[current].parents,
            key=lambda stage: (pmf_quantile(finish_pmfs[stage], 0.95, bin_ms=bin_ms), stage),
        )
        path.append(parent)
        current = parent
    path.reverse()
    return tuple(path)


def evaluate_orion_state(
    *,
    workflow: WorkflowSpec,
    config: PlannerConfig,
    ref_data: ReferenceData,
    state_key: tuple[int, ...],
    bin_ms: float = DEFAULT_BIN_MS,
    max_ms: float = DEFAULT_MAX_MS,
) -> OrionEvaluation:
    memory = _state_to_memory(state_key, config)
    topo = topological_stage_names(workflow)
    stage_pmfs: dict[str, np.ndarray] = {}
    finish_pmfs: dict[str, np.ndarray] = {}
    for stage_name in topo:
        latency_class = "cold_like" if stage_name == workflow.entry else "warm"
        dist = orion_stage_distribution(
            stage_name=stage_name,
            memory_mb=int(memory[stage_name]),
            latency_class=latency_class,
            ref_data=ref_data,
            bin_ms=bin_ms,
            max_ms=max_ms,
        )
        stage_pmfs[stage_name] = dist.pmf
        parents = workflow.nodes[stage_name].parents
        if parents:
            start_pmf = max_independent_pmfs([finish_pmfs[parent] for parent in parents])
            finish_pmfs[stage_name] = convolve_pmfs(start_pmf, dist.pmf)
        else:
            finish_pmfs[stage_name] = dist.pmf

    sinks = sink_stages(workflow)
    if len(sinks) == 1:
        e2e_pmf = finish_pmfs[sinks[0]]
        critical_path = _critical_path(
            workflow=workflow,
            finish_pmfs=finish_pmfs,
            sink=sinks[0],
            bin_ms=bin_ms,
        )
    else:
        e2e_pmf = max_independent_pmfs([finish_pmfs[sink] for sink in sinks])
        sink = max(sinks, key=lambda stage: (pmf_quantile(finish_pmfs[stage], 0.95, bin_ms=bin_ms), stage))
        critical_path = _critical_path(
            workflow=workflow,
            finish_pmfs=finish_pmfs,
            sink=sink,
            bin_ms=bin_ms,
        )
    p95 = pmf_quantile(e2e_pmf, 0.95, bin_ms=bin_ms)
    survival = pmf_survival(e2e_pmf, config.slo_ms, bin_ms=bin_ms)
    cost = plan_cost_gbsec(
        memory_tier_per_stage=memory,
        entry_prewarm_count_value=0,
        warm_splines=ref_data.warm_splines,
        stages=config.stages,
    )
    return OrionEvaluation(
        state_key=state_key,
        memory_tier_per_stage=memory,
        cost_gbsec=float(cost),
        p95_ms=float(p95),
        survival_at_slo=float(survival),
        feasible_by_orion=bool(p95 <= float(config.slo_ms) + 1e-9),
        critical_path=critical_path,
        e2e_pmf=e2e_pmf,
        finish_pmfs=finish_pmfs,
    )


def _our_plan_violation(
    *,
    config: PlannerConfig,
    ref_data: ReferenceData,
    memory_tier_per_stage: dict[str, int],
    rho: float,
    contention_factor: float,
) -> tuple[float, float]:
    plan = PlanInput(
        memory_tier_per_stage=dict(memory_tier_per_stage),
        entry_prewarm_count=0.0,
        predicted_arrivals=float(config.predicted_arrivals),
        lognormal_params=ref_data.lognormal_params,
        amdahl_params=ref_data.amdahl_params,
        cold_overhead_per_stage=ref_data.cold_overhead_per_stage,
        p_baseline=ref_data.p_baseline,
    )
    risk = compute_plan_risk(
        plan,
        slo_ms=float(config.slo_ms),
        rho=float(rho),
        contention_factor=float(contention_factor),
    )
    return float(risk.p_violation_total), float(risk.expected_e2e_ms)


def orion_plan(
    *,
    workflow: WorkflowSpec,
    config: PlannerConfig,
    ref_data: ReferenceData,
    bin_ms: float = DEFAULT_BIN_MS,
    max_ms: float = DEFAULT_MAX_MS,
    expansion_limit: int = DEFAULT_EXPANSION_LIMIT,
    rho: float = 0.67,
    contention_factor: float = 1.10,
) -> OrionPlanResult:
    start_key = tuple([0] * len(config.stages))
    counter = 0
    frontier: list[tuple[float, int, tuple[int, ...]]] = []
    cache: dict[tuple[int, ...], OrionEvaluation] = {}

    def evaluate(state_key: tuple[int, ...]) -> OrionEvaluation:
        if state_key not in cache:
            cache[state_key] = evaluate_orion_state(
                workflow=workflow,
                config=config,
                ref_data=ref_data,
                state_key=state_key,
                bin_ms=bin_ms,
                max_ms=max_ms,
            )
        return cache[state_key]

    def priority(evaluation: OrionEvaluation) -> float:
        return float(evaluation.p95_ms) * float(evaluation.cost_gbsec)

    def result_from(
        best: OrionEvaluation,
        *,
        states_expanded: int,
        search_exhausted: bool,
    ) -> OrionPlanResult:
        violation, expected = _our_plan_violation(
            config=config,
            ref_data=ref_data,
            memory_tier_per_stage=best.memory_tier_per_stage,
            rho=rho,
            contention_factor=contention_factor,
        )
        return OrionPlanResult(
            memory_tier_per_stage=best.memory_tier_per_stage,
            cost_gbsec=best.cost_gbsec,
            orion_p95_ms=best.p95_ms,
            orion_own_survival=best.survival_at_slo,
            feasible_by_orion=best.feasible_by_orion,
            feasible_by_ours=bool(violation <= config.max_violation_rate + EPS),
            violation_rate=float(violation),
            expected_e2e_ms=float(expected),
            states_expanded=states_expanded,
            states_evaluated=len(cache),
            search_exhausted=search_exhausted,
        )

    start_eval = evaluate(start_key)
    if start_eval.feasible_by_orion:
        return result_from(start_eval, states_expanded=0, search_exhausted=False)

    heapq.heappush(frontier, (priority(start_eval), counter, start_key))
    queued: set[tuple[int, ...]] = {start_key}
    expanded: set[tuple[int, ...]] = set()

    while frontier and len(expanded) < int(expansion_limit):
        _, _, state_key = heapq.heappop(frontier)
        if state_key in expanded:
            continue
        expanded.add(state_key)
        evaluate(state_key)

        for stage_index, _stage_name in enumerate(config.stages):
            tier_index = int(state_key[stage_index])
            if tier_index >= len(config.tiers) - 1:
                continue
            next_key_list = list(state_key)
            next_key_list[stage_index] = tier_index + 1
            next_key = tuple(next_key_list)
            if next_key in queued or next_key in expanded:
                continue
            next_eval = evaluate(next_key)
            if next_eval.feasible_by_orion:
                return result_from(
                    next_eval,
                    states_expanded=len(expanded),
                    search_exhausted=False,
                )
            counter += 1
            queued.add(next_key)
            heapq.heappush(
                frontier,
                (priority(next_eval), counter, next_key),
            )

    best = min(cache.values(), key=lambda item: (item.p95_ms, item.cost_gbsec, item.state_key))
    return result_from(
        best,
        states_expanded=len(expanded),
        search_exhausted=bool(frontier and len(expanded) >= int(expansion_limit)),
    )


def run_orion_self_checks(
    *,
    workflow: WorkflowSpec,
    config: PlannerConfig,
    ref_data: ReferenceData,
) -> dict[str, float | bool]:
    stage_name = config.stages[0]
    tier = int(config.tiers[min(3, len(config.tiers) - 1)])
    dist = orion_stage_distribution(
        stage_name=stage_name,
        memory_mb=tier,
        latency_class="cold_like" if stage_name == workflow.entry else "warm",
        ref_data=ref_data,
        bin_ms=DEFAULT_BIN_MS,
    )
    numerical_p95 = pmf_quantile(dist.pmf, 0.95, bin_ms=DEFAULT_BIN_MS)
    analytic_p95 = dist.params.quantile(0.95)
    single_rel_error = abs(numerical_p95 - analytic_p95) / analytic_p95

    state_key = tuple([min(3, len(config.tiers) - 1)] * len(config.stages))
    p95_25 = evaluate_orion_state(
        workflow=workflow,
        config=config,
        ref_data=ref_data,
        state_key=state_key,
        bin_ms=25.0,
    ).p95_ms
    p95_12 = evaluate_orion_state(
        workflow=workflow,
        config=config,
        ref_data=ref_data,
        state_key=state_key,
        bin_ms=12.5,
    ).p95_ms
    convergence_rel_error = abs(p95_25 - p95_12) / p95_12
    return {
        "single_stage": stage_name,
        "single_tier": tier,
        "single_numerical_p95_ms": float(numerical_p95),
        "single_analytic_p95_ms": float(analytic_p95),
        "single_rel_error": float(single_rel_error),
        "single_pass": bool(single_rel_error < 0.005),
        "grid_p95_25ms": float(p95_25),
        "grid_p95_12p5ms": float(p95_12),
        "grid_rel_error": float(convergence_rel_error),
        "grid_pass": bool(convergence_rel_error < 0.003),
    }


def orion_result_row(
    *,
    slo_class: str,
    config: PlannerConfig,
    result: OrionPlanResult,
    brute_row: dict[str, Any] | None,
) -> dict[str, Any]:
    brute_cost = None
    brute_config = ""
    if brute_row is not None:
        brute_cost = float(brute_row["optimal_cost_gbsec"])
        brute_config = str(brute_row["optimal_memory_config"])
    config_string = ",".join(
        f"{stage}:{int(result.memory_tier_per_stage[stage])}" for stage in config.stages
    )
    return {
        "slo_class": slo_class,
        "slo_ms": config.slo_ms,
        "method": "orion",
        "cost_gbsec": result.cost_gbsec,
        "violation_rate": result.violation_rate,
        "expected_e2e_ms": result.expected_e2e_ms,
        "entry_prewarm_safety_factor": 0.0,
        "entry_prewarm_count": 0,
        "feasible": result.feasible_by_ours,
        "iterations": result.states_expanded,
        "states_evaluated": result.states_evaluated,
        "cost_gap_vs_brute_pct": (
            math.nan
            if brute_cost is None or not math.isfinite(brute_cost) or brute_cost <= 0.0
            else (result.cost_gbsec - brute_cost) / brute_cost * 100.0
        ),
        "configs_match_brute": bool(config_string == brute_config) if brute_config else False,
        "memory_config": config_string,
        "orion_p95_ms": result.orion_p95_ms,
        "orion_own_survival": result.orion_own_survival,
        "orion_feasible": result.feasible_by_orion,
        "orion_search_exhausted": result.search_exhausted,
    }
