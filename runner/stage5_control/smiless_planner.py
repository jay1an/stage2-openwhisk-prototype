#!/usr/bin/env python3
"""SMIless-style A* resource search adapted to the memory-tier planner.

SMIless' source implementation (blinkbear/smiless-ad,
``optimizer-engine/optimizer/path_search.py``) performs a prefix search over
topologically ordered DAG nodes. Each node chooses one of two devices
(``cpu``/``cuda``), children are tried in per-node cost order, infeasible
prefixes are pruned with a remaining-minimum-execution-time SLA bound, and the
priority queue is ordered by accumulated prefix cost plus the SMIless
``calc_heuristic_cost`` value.

This module keeps that search shape but adapts the decision space to this
project: every stage chooses one memory tier. The planner's own feasibility
test is a deterministic p95 DAG bound (root cold-like, downstream warm). The
returned plan is then re-evaluated with this project's risk model so it can be
compared against greedy/risk-price/brute-force under the same final metric.
"""

from __future__ import annotations

import argparse
import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from runner.stage4_risk.plan_risk import BASE_MEMORY_MB, PlanInput, compute_plan_risk
from runner.stage4_risk.scaling import (
    memory_to_cpu_cores,
    scale_stage_for_memory_tier,
    spline_predict_warm_mean,
)
from runner.stage5_control.brute_force_planner import format_memory_config
from runner.stage5_control.multi_slo_planner import (
    DEFAULT_BASELINE_TRACE,
    DEFAULT_LOGNORMAL_PARAMS,
    DEFAULT_SAFETY_FACTORS,
    DEFAULT_TIERS,
    STAGES,
    PlannerConfig,
    ReferenceData,
    load_reference_data,
    plan_cost_gbsec,
)
from runner.workflow import WorkflowSpec, load_workflow


DEFAULT_WORKFLOW = Path(__file__).resolve().parents[2] / "configs" / "civic_alert_flow.yaml"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "smiless_planner"
DEFAULT_EXPANSION_LIMIT = 200_000
EPS = 1e-12


@dataclass(frozen=True)
class SMIlessPlanResult:
    memory_tier_per_stage: dict[str, int]
    cost_gbsec: float
    smiless_p95_ms: float
    feasible_by_smiless: bool
    feasible_by_ours: bool
    violation_rate: float
    expected_e2e_ms: float
    states_expanded: int
    states_evaluated: int
    search_exhausted: bool
    trace: tuple[dict[str, Any], ...]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


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
            raise ValueError(f"workflow has a cycle or missing parent; remaining={remaining}")
    return ordered


def _deterministic_e2e_ms(
    *,
    workflow: WorkflowSpec,
    durations_ms: dict[str, float],
) -> float:
    finish: dict[str, float] = {}
    for stage_name in _topological_stage_names(workflow):
        parents = workflow.nodes[stage_name].parents
        start = max((finish[parent] for parent in parents), default=0.0)
        finish[stage_name] = start + float(durations_ms[stage_name])
    sinks = [stage for stage in workflow.nodes if not workflow.children_of(stage)]
    return max(finish[sink] for sink in sinks)


def _stage_params(
    *,
    stage_name: str,
    memory_mb: int,
    latency_class: str,
    ref_data: ReferenceData,
):
    return scale_stage_for_memory_tier(
        stage_name=stage_name,
        latency_class=latency_class,
        target_memory_mb=int(memory_mb),
        base_memory_mb=BASE_MEMORY_MB,
        base_params=ref_data.lognormal_params[stage_name][latency_class],
        amdahl_params=ref_data.amdahl_params,
        splines=ref_data.warm_splines,
        contention_factor=1.0,
    )


def _precompute_stage_tables(
    *,
    workflow: WorkflowSpec,
    config: PlannerConfig,
    ref_data: ReferenceData,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[int]]]:
    """Return per-stage p95s, costs, and tier order sorted like SMIless devices."""

    p95_by_stage: dict[str, list[float]] = {}
    cost_by_stage: dict[str, list[float]] = {}
    tier_order_by_stage: dict[str, list[int]] = {}
    for stage_name in config.stages:
        p95s: list[float] = []
        costs: list[float] = []
        for tier in config.tiers:
            latency_class = "cold_like" if stage_name == workflow.entry else "warm"
            params = _stage_params(
                stage_name=stage_name,
                memory_mb=int(tier),
                latency_class=latency_class,
                ref_data=ref_data,
            )
            p95s.append(float(params.quantile(0.95)))
            warm_ms = spline_predict_warm_mean(
                stage_name,
                memory_to_cpu_cores(int(tier)),
                ref_data.warm_splines,
            )
            costs.append((int(tier) / 1024.0) * (warm_ms / 1000.0))
        p95_by_stage[stage_name] = p95s
        cost_by_stage[stage_name] = costs
        tier_order_by_stage[stage_name] = sorted(
            range(len(config.tiers)),
            key=lambda idx: (costs[idx], p95s[idx], int(config.tiers[idx])),
        )
    return p95_by_stage, cost_by_stage, tier_order_by_stage


def _memory_from_prefix(prefix: tuple[int, ...], config: PlannerConfig) -> dict[str, int]:
    return {
        stage_name: int(config.tiers[prefix[index]])
        for index, stage_name in enumerate(config.stages)
    }


def _prefix_cost(
    prefix: tuple[int, ...],
    *,
    config: PlannerConfig,
    cost_by_stage: dict[str, list[float]],
) -> float:
    return sum(
        cost_by_stage[stage_name][prefix[index]]
        for index, stage_name in enumerate(config.stages[: len(prefix)])
    )


def _bound_durations(
    prefix: tuple[int, ...],
    *,
    config: PlannerConfig,
    p95_by_stage: dict[str, list[float]],
) -> dict[str, float]:
    durations: dict[str, float] = {}
    for index, stage_name in enumerate(config.stages):
        if index < len(prefix):
            durations[stage_name] = p95_by_stage[stage_name][prefix[index]]
        else:
            durations[stage_name] = min(p95_by_stage[stage_name])
    return durations


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


def smiless_plan(
    *,
    workflow: WorkflowSpec,
    config: PlannerConfig,
    ref_data: ReferenceData,
    expansion_limit: int = DEFAULT_EXPANSION_LIMIT,
    rho: float = 0.67,
    contention_factor: float = 1.10,
    trace_limit: int = 64,
) -> SMIlessPlanResult:
    """Run the SMIless-style prefix A* search on memory tiers."""

    p95_by_stage, cost_by_stage, tier_order_by_stage = _precompute_stage_tables(
        workflow=workflow,
        config=config,
        ref_data=ref_data,
    )
    max_remaining_cost_suffix = [0.0] * (len(config.stages) + 1)
    for index in range(len(config.stages) - 1, -1, -1):
        stage_name = config.stages[index]
        max_remaining_cost_suffix[index] = (
            max_remaining_cost_suffix[index + 1] + max(cost_by_stage[stage_name])
        )

    def priority(prefix: tuple[int, ...]) -> float:
        # Adapt SMIless calc_current_cost + calc_heuristic_cost.
        # The heuristic part is the remaining max-cost suffix, matching source
        # path_search.py's max_cost - prefix_max_cost behavior.
        return _prefix_cost(prefix, config=config, cost_by_stage=cost_by_stage) + max_remaining_cost_suffix[
            len(prefix)
        ]

    def lower_bound_p95(prefix: tuple[int, ...]) -> float:
        return _deterministic_e2e_ms(
            workflow=workflow,
            durations_ms=_bound_durations(prefix, config=config, p95_by_stage=p95_by_stage),
        )

    def result_from(
        prefix: tuple[int, ...],
        *,
        states_expanded: int,
        states_evaluated: int,
        search_exhausted: bool,
        trace_rows: list[dict[str, Any]],
    ) -> SMIlessPlanResult:
        memory = _memory_from_prefix(prefix, config)
        p95 = _deterministic_e2e_ms(
            workflow=workflow,
            durations_ms={
                stage_name: p95_by_stage[stage_name][prefix[index]]
                for index, stage_name in enumerate(config.stages)
            },
        )
        cost = plan_cost_gbsec(
            memory_tier_per_stage=memory,
            entry_prewarm_count_value=0,
            warm_splines=ref_data.warm_splines,
            stages=config.stages,
        )
        violation, expected = _our_plan_violation(
            config=config,
            ref_data=ref_data,
            memory_tier_per_stage=memory,
            rho=rho,
            contention_factor=contention_factor,
        )
        return SMIlessPlanResult(
            memory_tier_per_stage=memory,
            cost_gbsec=float(cost),
            smiless_p95_ms=float(p95),
            feasible_by_smiless=bool(p95 <= float(config.slo_ms) + EPS),
            feasible_by_ours=bool(violation <= config.max_violation_rate + EPS),
            violation_rate=float(violation),
            expected_e2e_ms=float(expected),
            states_expanded=states_expanded,
            states_evaluated=states_evaluated,
            search_exhausted=search_exhausted,
            trace=tuple(trace_rows),
        )

    start: tuple[int, ...] = tuple()
    frontier: list[tuple[float, int, tuple[int, ...]]] = [(priority(start), 0, start)]
    queued: set[tuple[int, ...]] = {start}
    expanded: set[tuple[int, ...]] = set()
    counter = 0
    states_evaluated = 0
    best_prefix = start
    best_bound = math.inf
    trace_rows: list[dict[str, Any]] = []

    while frontier and len(expanded) < int(expansion_limit):
        item_priority, _, prefix = heapq.heappop(frontier)
        if prefix in expanded:
            continue
        expanded.add(prefix)
        bound = lower_bound_p95(prefix)
        states_evaluated += 1
        if bound < best_bound:
            best_bound = bound
            best_prefix = prefix
        if len(trace_rows) < trace_limit:
            trace_rows.append(
                {
                    "event": "expand",
                    "prefix_len": len(prefix),
                    "priority": item_priority,
                    "bound_p95_ms": bound,
                    "prefix": ",".join(str(config.tiers[idx]) for idx in prefix),
                }
            )
        if bound > float(config.slo_ms) + EPS:
            continue
        if len(prefix) == len(config.stages):
            return result_from(
                prefix,
                states_expanded=len(expanded),
                states_evaluated=states_evaluated,
                search_exhausted=False,
                trace_rows=trace_rows,
            )

        stage_index = len(prefix)
        stage_name = config.stages[stage_index]
        for tier_index in tier_order_by_stage[stage_name]:
            child = prefix + (tier_index,)
            if child in queued or child in expanded:
                continue
            child_bound = lower_bound_p95(child)
            states_evaluated += 1
            if child_bound > float(config.slo_ms) + EPS:
                if len(trace_rows) < trace_limit:
                    trace_rows.append(
                        {
                            "event": "prune",
                            "prefix_len": len(child),
                            "priority": priority(child),
                            "bound_p95_ms": child_bound,
                            "prefix": ",".join(str(config.tiers[idx]) for idx in child),
                        }
                    )
                continue
            if child_bound < best_bound:
                best_bound = child_bound
                best_prefix = child
            queued.add(child)
            counter += 1
            heapq.heappush(frontier, (priority(child), counter, child))

    fallback = best_prefix
    if len(fallback) < len(config.stages):
        # Fill undecided suffix with fastest tiers so the fallback can be
        # re-evaluated and reported transparently.
        suffix = []
        for stage_name in config.stages[len(fallback) :]:
            suffix.append(min(range(len(config.tiers)), key=lambda idx: p95_by_stage[stage_name][idx]))
        fallback = fallback + tuple(suffix)
    return result_from(
        fallback,
        states_expanded=len(expanded),
        states_evaluated=states_evaluated,
        search_exhausted=bool(frontier and len(expanded) >= int(expansion_limit)),
        trace_rows=trace_rows,
    )


def smiless_result_row(
    *,
    slo_class: str,
    config: PlannerConfig,
    result: SMIlessPlanResult,
) -> dict[str, Any]:
    return {
        "slo_class": slo_class,
        "slo_ms": float(config.slo_ms),
        "method": "smiless",
        "cost_gbsec": result.cost_gbsec,
        "violation_rate": result.violation_rate,
        "expected_e2e_ms": result.expected_e2e_ms,
        "feasible": result.feasible_by_ours,
        "iterations": result.states_expanded,
        "states_evaluated": result.states_evaluated,
        "memory_config": format_memory_config(result.memory_tier_per_stage, list(config.stages)),
        "smiless_p95_ms": result.smiless_p95_ms,
        "smiless_feasible": result.feasible_by_smiless,
        "smiless_search_exhausted": result.search_exhausted,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    parser.add_argument("--lognormal-params", default=str(DEFAULT_LOGNORMAL_PARAMS))
    parser.add_argument("--baseline-trace", default=str(DEFAULT_BASELINE_TRACE))
    parser.add_argument("--predicted-arrivals", type=float, default=5.0)
    parser.add_argument("--slo-premium-ms", type=float, default=18000.0)
    parser.add_argument("--slo-free-ms", type=float, default=22000.0)
    parser.add_argument("--rho", type=float, default=0.67)
    parser.add_argument("--contention-factor", type=float, default=1.10)
    parser.add_argument("--expansion-limit", type=int, default=DEFAULT_EXPANSION_LIMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = _resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref_data = load_reference_data(
        lognormal_params_path=args.lognormal_params,
        baseline_trace_path=args.baseline_trace,
    )
    workflow = load_workflow(str(_resolve(args.workflow)))
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for slo_class, slo_ms in [("premium", args.slo_premium_ms), ("free", args.slo_free_ms)]:
        config = PlannerConfig(
            slo_ms=float(slo_ms),
            max_violation_rate=0.05,
            predicted_arrivals=float(args.predicted_arrivals),
            tiers=list(DEFAULT_TIERS),
            safety_factors=list(DEFAULT_SAFETY_FACTORS),
            stages=list(STAGES),
        )
        result = smiless_plan(
            workflow=workflow,
            config=config,
            ref_data=ref_data,
            expansion_limit=int(args.expansion_limit),
            rho=float(args.rho),
            contention_factor=float(args.contention_factor),
        )
        rows.append(smiless_result_row(slo_class=slo_class, config=config, result=result))
        for item in result.trace:
            trace_rows.append({"slo_class": slo_class, **item})
    method_df = pd.DataFrame(rows)
    trace_df = pd.DataFrame(trace_rows)
    method_df.to_csv(out / "method_comparison.csv", index=False)
    trace_df.to_csv(out / "search_trace.csv", index=False)
    print("method_comparison:")
    print(method_df.round(8).to_string(index=False), flush=True)
    print(f"wrote {out / 'method_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
