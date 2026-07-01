#!/usr/bin/env python3
"""Runtime risk-price residual planner for dynamic tier upgrades.

The offline planner chooses a per-class baseline plan.  This module replans the
remaining, not-yet-started stages of one workflow after observing actual stage
completion times.  It is intentionally UP-only: a runtime decision may buy more
CPU/memory for pending stages, but it never lowers the plan or touches stages
that have already started.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from runner.stage4_risk.dag_aggregation import LogNormalParams, conditional_risk
from runner.stage4_risk.plan_risk import BASE_MEMORY_MB
from runner.stage4_risk.scaling import (
    cold_overhead_for_tier,
    load_cleansed_cold_overhead,
    memory_to_cpu_cores,
    scale_stage_for_memory_tier,
    spline_predict_warm_mean,
)
from runner.workflow import WorkflowSpec


DEFAULT_RHO = 0.67
DEFAULT_CONTENTION_FACTOR = 1.10
EPS = 1e-12


@dataclass(frozen=True)
class RuntimeStateEval:
    state_key: tuple[int, ...]
    memory_tier_per_stage: dict[str, int]
    conditional_risk: float
    cost_gbsec: float


@dataclass
class RuntimeRiskPriceContext:
    config: Any
    ref_data: Any
    workflow: WorkflowSpec
    completed_finish_ms: dict[str, float]
    original_key: tuple[int, ...]
    pending_stages: tuple[str, ...]
    tier_index_by_memory: dict[int, int]
    jit_lead_ms_by_stage: dict[str, float]
    jit_margin_ms: float
    min_jit_slack_ms: float
    max_changed_stages: int
    now_ms_since_workflow_start: float | None
    predicted_start_ms_by_stage: dict[str, float]
    predicted_completion_ms_by_stage: dict[str, float]
    rho: float
    contention_factor: float
    require_jit_coverage: bool
    eval_cache: dict[tuple[int, ...], RuntimeStateEval]

    def evaluate(self, state_key: tuple[int, ...]) -> RuntimeStateEval:
        cached = self.eval_cache.get(state_key)
        if cached is not None:
            return cached
        memory = _key_to_memory(state_key, self.config)
        risk = _runtime_conditional_risk(
            config=self.config,
            ref_data=self.ref_data,
            workflow=self.workflow,
            memory_tier_per_stage=memory,
            completed_finish_ms=self.completed_finish_ms,
            rho=self.rho,
            contention_factor=self.contention_factor,
        )
        cost = _runtime_plan_cost(
            config=self.config,
            ref_data=self.ref_data,
            memory_tier_per_stage=memory,
        )
        out = RuntimeStateEval(
            state_key=state_key,
            memory_tier_per_stage=memory,
            conditional_risk=float(risk),
            cost_gbsec=float(cost),
        )
        self.eval_cache[state_key] = out
        return out


def _key_to_memory(state_key: tuple[int, ...], config: Any) -> dict[str, int]:
    return {
        stage_name: int(config.tiers[state_key[index]])
        for index, stage_name in enumerate(config.stages)
    }


def _state_key_from_memory(
    *,
    config: Any,
    current_tiers: dict[str, int],
    tier_index_by_memory: dict[int, int],
) -> tuple[int, ...]:
    return tuple(
        int(tier_index_by_memory[int(current_tiers[stage_name])])
        for stage_name in config.stages
    )


def _stage_dist(
    *,
    stage_name: str,
    memory_mb: int,
    ref_data: Any,
    contention_factor: float,
) -> LogNormalParams:
    try:
        base_params = ref_data.lognormal_params[stage_name]["warm"]
    except KeyError as exc:
        raise ValueError(f"missing warm lognormal params for stage={stage_name}") from exc
    return scale_stage_for_memory_tier(
        stage_name=stage_name,
        latency_class="warm",
        target_memory_mb=int(memory_mb),
        base_memory_mb=BASE_MEMORY_MB,
        base_params=base_params,
        amdahl_params=ref_data.amdahl_params,
        splines=ref_data.warm_splines,
        contention_factor=float(contention_factor),
    )


def _runtime_stage_dists(
    *,
    memory_tier_per_stage: dict[str, int],
    ref_data: Any,
    contention_factor: float,
) -> dict[str, LogNormalParams]:
    return {
        stage_name: _stage_dist(
            stage_name=stage_name,
            memory_mb=int(memory_mb),
            ref_data=ref_data,
            contention_factor=contention_factor,
        )
        for stage_name, memory_mb in memory_tier_per_stage.items()
    }


def _runtime_conditional_risk(
    *,
    config: Any,
    ref_data: Any,
    workflow: WorkflowSpec,
    memory_tier_per_stage: dict[str, int],
    completed_finish_ms: dict[str, float],
    rho: float,
    contention_factor: float,
) -> float:
    return conditional_risk(
        workflow=workflow,
        stage_dists=_runtime_stage_dists(
            memory_tier_per_stage=memory_tier_per_stage,
            ref_data=ref_data,
            contention_factor=contention_factor,
        ),
        completed_finish_ms=completed_finish_ms,
        slo_ms=float(config.slo_ms),
        rho=float(rho),
    )


def _runtime_plan_cost(
    *,
    config: Any,
    ref_data: Any,
    memory_tier_per_stage: dict[str, int],
) -> float:
    total = 0.0
    for stage_name in config.stages:
        memory_mb = int(memory_tier_per_stage[stage_name])
        cpu_cores = memory_to_cpu_cores(memory_mb)
        warm_ms = spline_predict_warm_mean(stage_name, cpu_cores, ref_data.warm_splines)
        total += (memory_mb / 1024.0) * (warm_ms / 1000.0)
    return float(total)


def _stage_index(config: Any, stage_name: str) -> int:
    try:
        return list(config.stages).index(stage_name)
    except ValueError as exc:
        raise ValueError(f"unknown stage in pending set: {stage_name}") from exc


def _topological_stage_names(workflow: WorkflowSpec) -> list[str]:
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


def _state_with_value(
    state_key: tuple[int, ...],
    stage_index: int,
    tier_index: int,
) -> tuple[int, ...]:
    out = list(state_key)
    out[stage_index] = int(tier_index)
    return tuple(out)


def _predicted_warm_duration_ms(
    *,
    ctx: RuntimeRiskPriceContext,
    stage_name: str,
    memory_mb: int,
) -> float:
    cpu_cores = memory_to_cpu_cores(int(memory_mb))
    return float(spline_predict_warm_mean(stage_name, cpu_cores, ctx.ref_data.warm_splines))


def _candidate_predicted_start_ms(
    *,
    ctx: RuntimeRiskPriceContext,
    state_key: tuple[int, ...],
    target_stage: str,
) -> float | None:
    if ctx.now_ms_since_workflow_start is None:
        return None
    if not ctx.predicted_completion_ms_by_stage:
        return None

    pending_set = set(ctx.pending_stages)
    memory = _key_to_memory(state_key, ctx.config)
    starts: dict[str, float] = {}
    completions: dict[str, float] = {}
    now_ms = float(ctx.now_ms_since_workflow_start)

    for stage_name in _topological_stage_names(ctx.workflow):
        node = ctx.workflow.nodes[stage_name]
        if node.parents:
            start_ms = max(completions[parent] for parent in node.parents)
        else:
            start_ms = 0.0
        start_ms = max(start_ms, now_ms)
        starts[stage_name] = start_ms

        if stage_name in ctx.completed_finish_ms:
            completions[stage_name] = float(ctx.completed_finish_ms[stage_name])
        elif stage_name not in pending_set:
            fallback_completion = ctx.predicted_completion_ms_by_stage.get(stage_name)
            if fallback_completion is None:
                return None
            completions[stage_name] = max(float(fallback_completion), now_ms)
        else:
            completions[stage_name] = start_ms + _predicted_warm_duration_ms(
                ctx=ctx,
                stage_name=stage_name,
                memory_mb=int(memory[stage_name]),
            )

    return starts.get(target_stage)


def _candidate_jit_slack_ms(
    *,
    ctx: RuntimeRiskPriceContext,
    state_key: tuple[int, ...],
    stage_name: str,
    memory_mb: int,
) -> float:
    if not ctx.require_jit_coverage:
        return math.inf
    candidate_start_ms = _candidate_predicted_start_ms(
        ctx=ctx,
        state_key=state_key,
        target_stage=stage_name,
    )
    if candidate_start_ms is not None and ctx.now_ms_since_workflow_start is not None:
        lead_ms = candidate_start_ms - float(ctx.now_ms_since_workflow_start)
    elif stage_name in ctx.jit_lead_ms_by_stage:
        lead_ms = float(ctx.jit_lead_ms_by_stage[stage_name])
    else:
        return -math.inf
    cold_table = load_cleansed_cold_overhead()
    cold_ms = cold_overhead_for_tier(stage_name, int(memory_mb), cold_table)
    required_ms = (
        cold_ms
        + max(0.0, float(ctx.jit_margin_ms))
        + max(0.0, float(ctx.min_jit_slack_ms))
    )
    return lead_ms - required_ms


def _state_jit_safe(ctx: RuntimeRiskPriceContext, state_key: tuple[int, ...]) -> bool:
    if not ctx.require_jit_coverage:
        return True
    for stage_name in ctx.pending_stages:
        stage_index = _stage_index(ctx.config, stage_name)
        if int(state_key[stage_index]) <= int(ctx.original_key[stage_index]):
            continue
        memory_mb = int(ctx.config.tiers[state_key[stage_index]])
        if (
            _candidate_jit_slack_ms(
                ctx=ctx,
                state_key=state_key,
                stage_name=stage_name,
                memory_mb=memory_mb,
            )
            < -EPS
        ):
            return False
    return True


def _changed_stage_count(ctx: RuntimeRiskPriceContext, state_key: tuple[int, ...]) -> int:
    return sum(
        1
        for stage_name in ctx.pending_stages
        if int(state_key[_stage_index(ctx.config, stage_name)])
        != int(ctx.original_key[_stage_index(ctx.config, stage_name)])
    )


def _single_change_candidates(
    *,
    ctx: RuntimeRiskPriceContext,
    state_key: tuple[int, ...],
) -> list[dict[str, Any]]:
    current = ctx.evaluate(state_key)
    candidates: list[dict[str, Any]] = []
    for stage_name in ctx.pending_stages:
        stage_index = _stage_index(ctx.config, stage_name)
        current_index = int(state_key[stage_index])
        for tier_index in range(current_index + 1, len(ctx.config.tiers)):
            memory_mb = int(ctx.config.tiers[tier_index])
            candidate_key = _state_with_value(state_key, stage_index, tier_index)
            jit_slack_ms = _candidate_jit_slack_ms(
                ctx=ctx,
                state_key=candidate_key,
                stage_name=stage_name,
                memory_mb=memory_mb,
            )
            if jit_slack_ms < -EPS:
                continue
            evaluation = ctx.evaluate(candidate_key)
            candidates.append(
                {
                    "stage_name": stage_name,
                    "stage_index": stage_index,
                    "tier_index": tier_index,
                    "memory_mb": memory_mb,
                    "state_key": candidate_key,
                    "evaluation": evaluation,
                    "risk_delta": current.conditional_risk - evaluation.conditional_risk,
                    "cost_delta": evaluation.cost_gbsec - current.cost_gbsec,
                    "jit_slack_ms": jit_slack_ms,
                }
            )
    return candidates


def _candidate_efficiency_key(candidate: dict[str, Any]) -> tuple[float, float, float, str, int]:
    risk_delta = float(candidate["risk_delta"])
    cost_delta = float(candidate["cost_delta"])
    if risk_delta <= EPS:
        efficiency = -math.inf
    elif cost_delta <= 0.0:
        efficiency = math.inf
    else:
        efficiency = risk_delta / cost_delta
    return (
        -efficiency,
        float(candidate["evaluation"].conditional_risk),
        -float(candidate["jit_slack_ms"]),
        str(candidate["stage_name"]),
        int(candidate["tier_index"]),
    )


def _build_lambda_grid(candidates: list[dict[str, Any]]) -> list[float]:
    ratios: set[float] = {0.0}
    for candidate in candidates:
        risk_delta = float(candidate["risk_delta"])
        cost_delta = float(candidate["cost_delta"])
        if risk_delta <= EPS or cost_delta <= 0.0:
            continue
        ratio = cost_delta / risk_delta
        ratios.add(ratio * 0.5)
        ratios.add(ratio)
        ratios.add(ratio * 2.0)
    return sorted(value for value in ratios if math.isfinite(value))


def _choose_by_lambda(
    *,
    ctx: RuntimeRiskPriceContext,
    candidates: list[dict[str, Any]],
    lambda_value: float,
) -> tuple[int, ...]:
    chosen = list(ctx.original_key)
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_stage.setdefault(str(candidate["stage_name"]), []).append(candidate)

    for stage_name, items in by_stage.items():
        stage_index = _stage_index(ctx.config, stage_name)
        best_score = 0.0
        best_tier_index = int(ctx.original_key[stage_index])
        for candidate in items:
            score = float(candidate["cost_delta"]) - lambda_value * float(candidate["risk_delta"])
            if score < best_score - EPS:
                best_score = score
                best_tier_index = int(candidate["tier_index"])
        chosen[stage_index] = best_tier_index
    return tuple(chosen)


def _repair_until_feasible(
    *,
    ctx: RuntimeRiskPriceContext,
    state_key: tuple[int, ...],
) -> tuple[int, ...]:
    current = ctx.evaluate(state_key)
    max_steps = len(ctx.pending_stages) * max(0, len(ctx.config.tiers) - 1)
    steps = 0
    while current.conditional_risk > float(ctx.config.max_violation_rate) + EPS and steps < max_steps:
        candidates = [
            candidate
            for candidate in _single_change_candidates(ctx=ctx, state_key=state_key)
            if float(candidate["risk_delta"]) > EPS
        ]
        if not candidates:
            break
        chosen = sorted(candidates, key=_candidate_efficiency_key)[0]
        state_key = chosen["state_key"]
        current = chosen["evaluation"]
        steps += 1
    return state_key


def _local_cost_prune(
    *,
    ctx: RuntimeRiskPriceContext,
    state_key: tuple[int, ...],
) -> tuple[int, ...]:
    current = ctx.evaluate(state_key)
    if current.conditional_risk > float(ctx.config.max_violation_rate) + EPS:
        return state_key

    improved = True
    while improved:
        improved = False
        best_key = state_key
        best_eval = current
        for stage_name in ctx.pending_stages:
            stage_index = _stage_index(ctx.config, stage_name)
            original_index = int(ctx.original_key[stage_index])
            current_index = int(state_key[stage_index])
            for tier_index in range(original_index, current_index):
                candidate_key = _state_with_value(state_key, stage_index, tier_index)
                candidate_eval = ctx.evaluate(candidate_key)
                if candidate_eval.conditional_risk > float(ctx.config.max_violation_rate) + EPS:
                    continue
                if candidate_eval.cost_gbsec < best_eval.cost_gbsec - EPS:
                    best_key = candidate_key
                    best_eval = candidate_eval
        if best_key != state_key:
            state_key = best_key
            current = best_eval
            improved = True
    return state_key


def risk_price_dynamic_upgrade(
    *,
    config: Any,
    ref_data: Any,
    workflow: WorkflowSpec,
    current_tiers: dict[str, int],
    completed_finish_ms: dict[str, float],
    pending_stages: list[str],
    jit_lead_ms_by_stage: dict[str, float] | None = None,
    jit_margin_ms: float = 0.0,
    min_jit_slack_ms: float = 1000.0,
    max_changed_stages: int = 3,
    allow_partial: bool = False,
    now_ms_since_workflow_start: float | None = None,
    predicted_start_ms_by_stage: dict[str, float] | None = None,
    predicted_completion_ms_by_stage: dict[str, float] | None = None,
    rho: float = DEFAULT_RHO,
    contention_factor: float = DEFAULT_CONTENTION_FACTOR,
    require_jit_coverage: bool = True,
) -> dict[str, int] | None:
    """Return JIT-safe risk-price upgrades for pending stages.

    ``jit_lead_ms_by_stage`` is the time from now until the predicted stage
    start.  A candidate tier is considered only when that lead can cover the
    tier-specific cold overhead, ``jit_margin_ms``, and an extra slack budget.
    This prevents a runtime upgrade from creating a new unhidden cold start or
    a long sync wait.
    """

    if not pending_stages:
        return None
    tier_index_by_memory = {
        int(memory_mb): index for index, memory_mb in enumerate(config.tiers)
    }
    original_key = _state_key_from_memory(
        config=config,
        current_tiers=current_tiers,
        tier_index_by_memory=tier_index_by_memory,
    )
    ctx = RuntimeRiskPriceContext(
        config=config,
        ref_data=ref_data,
        workflow=workflow,
        completed_finish_ms=dict(completed_finish_ms),
        original_key=original_key,
        pending_stages=tuple(pending_stages),
        tier_index_by_memory=tier_index_by_memory,
        jit_lead_ms_by_stage=dict(jit_lead_ms_by_stage or {}),
        jit_margin_ms=float(jit_margin_ms),
        min_jit_slack_ms=float(min_jit_slack_ms),
        max_changed_stages=max(0, int(max_changed_stages)),
        now_ms_since_workflow_start=(
            None if now_ms_since_workflow_start is None else float(now_ms_since_workflow_start)
        ),
        predicted_start_ms_by_stage=dict(predicted_start_ms_by_stage or {}),
        predicted_completion_ms_by_stage=dict(predicted_completion_ms_by_stage or {}),
        rho=float(rho),
        contention_factor=float(contention_factor),
        require_jit_coverage=bool(require_jit_coverage),
        eval_cache={},
    )

    start_eval = ctx.evaluate(original_key)
    if start_eval.conditional_risk <= float(config.max_violation_rate) + EPS:
        return None

    initial_candidates = [
        candidate
        for candidate in _single_change_candidates(ctx=ctx, state_key=original_key)
        if float(candidate["risk_delta"]) > EPS
    ]
    if not initial_candidates:
        return None

    candidate_keys: set[tuple[int, ...]] = set()
    for lambda_value in _build_lambda_grid(initial_candidates):
        proposed = _choose_by_lambda(
            ctx=ctx,
            candidates=initial_candidates,
            lambda_value=lambda_value,
        )
        if not _state_jit_safe(ctx, proposed):
            continue
        repaired = _repair_until_feasible(ctx=ctx, state_key=proposed)
        if _state_jit_safe(ctx, repaired):
            candidate_keys.add(_local_cost_prune(ctx=ctx, state_key=repaired))

    greedy_repaired = _repair_until_feasible(ctx=ctx, state_key=original_key)
    if _state_jit_safe(ctx, greedy_repaired):
        candidate_keys.add(_local_cost_prune(ctx=ctx, state_key=greedy_repaired))

    improved_keys = [
        key
        for key in candidate_keys
        if key != original_key
        and ctx.evaluate(key).conditional_risk < start_eval.conditional_risk - EPS
        and _changed_stage_count(ctx, key) <= ctx.max_changed_stages
    ]
    if not improved_keys:
        return None

    feasible_keys = [
        key
        for key in improved_keys
        if ctx.evaluate(key).conditional_risk <= float(config.max_violation_rate) + EPS
    ]
    if not feasible_keys and not allow_partial:
        return None
    if feasible_keys:
        best_key = min(
            feasible_keys,
            key=lambda key: (
                ctx.evaluate(key).cost_gbsec,
                ctx.evaluate(key).conditional_risk,
                key,
            ),
        )
    else:
        best_key = min(
            improved_keys,
            key=lambda key: (
                ctx.evaluate(key).conditional_risk,
                ctx.evaluate(key).cost_gbsec,
                key,
            ),
        )

    best_memory = ctx.evaluate(best_key).memory_tier_per_stage
    return {
        stage_name: int(best_memory[stage_name])
        for stage_name in pending_stages
        if int(best_memory[stage_name]) != int(current_tiers[stage_name])
    } or None
