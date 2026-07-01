#!/usr/bin/env python3
"""Risk-price planner suite for multi-SLO resource planning.

The solver treats SLO violation probability as a priced resource.  It is
intended to be reused for offline planning and, later, dynamic residual
replanning.  This file focuses on the offline case and compares against
greedy, beam search, and an existing brute-force oracle.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from runner.stage4_risk.plan_risk import PlanInput, PlanRiskResult, compute_plan_risk
from runner.stage5_control.brute_force_planner import format_memory_config
from runner.stage5_control.multi_slo_planner import (
    DEFAULT_BASELINE_TRACE,
    DEFAULT_LOGNORMAL_PARAMS,
    DEFAULT_SAFETY_FACTORS,
    DEFAULT_TIERS,
    STAGES,
    PlannerConfig,
    ReferenceData,
    entry_prewarm_count,
    load_reference_data,
    plan_cost_gbsec,
)


DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parents[3] / "reports" / "risk_price_planner"
)
DEFAULT_BRUTE_FORCE = (
    Path(__file__).resolve().parents[3]
    / "reports"
    / "replan_sigma_rho_mean"
    / "brute_force_optimal.csv"
)
DEFAULT_RHO = 0.67
DEFAULT_CONTENTION_FACTOR = 1.10
EPS = 1e-12


@dataclass(frozen=True)
class StateEval:
    state_key: tuple[int, ...]
    memory_tier_per_stage: dict[str, int]
    safety_factor: float
    entry_prewarm_count: int
    risk_result: PlanRiskResult
    cost_gbsec: float

    @property
    def violation_rate(self) -> float:
        return float(self.risk_result.p_violation_total)

    @property
    def expected_e2e_ms(self) -> float:
        return float(self.risk_result.expected_e2e_ms)


@dataclass
class EvalContext:
    config: PlannerConfig
    ref_data: ReferenceData
    rho: float
    contention_factor: float
    eval_cache: dict[tuple[int, ...], StateEval]

    def evaluate(self, state_key: tuple[int, ...]) -> StateEval:
        if state_key in self.eval_cache:
            return self.eval_cache[state_key]
        memory = key_to_memory(state_key, self.config)
        safety_factor = key_to_safety_factor(state_key, self.config)
        prewarm_count = entry_prewarm_count(
            safety_factor=safety_factor,
            predicted_arrivals=self.config.predicted_arrivals,
        )
        plan = PlanInput(
            memory_tier_per_stage=memory,
            entry_prewarm_count=float(prewarm_count),
            predicted_arrivals=float(self.config.predicted_arrivals),
            lognormal_params=self.ref_data.lognormal_params,
            amdahl_params=self.ref_data.amdahl_params,
            cold_overhead_per_stage=self.ref_data.cold_overhead_per_stage,
            p_baseline=self.ref_data.p_baseline,
        )
        risk = compute_plan_risk(
            plan,
            slo_ms=self.config.slo_ms,
            rho=self.rho,
            contention_factor=self.contention_factor,
        )
        cost = plan_cost_gbsec(
            memory_tier_per_stage=memory,
            entry_prewarm_count_value=prewarm_count,
            warm_splines=self.ref_data.warm_splines,
            stages=self.config.stages,
        )
        out = StateEval(
            state_key=state_key,
            memory_tier_per_stage=memory,
            safety_factor=float(safety_factor),
            entry_prewarm_count=int(prewarm_count),
            risk_result=risk,
            cost_gbsec=float(cost),
        )
        self.eval_cache[state_key] = out
        return out


@dataclass(frozen=True)
class PlanResult:
    method: str
    state_key: tuple[int, ...]
    evaluation: StateEval
    feasible: bool
    iterations: int
    states_evaluated: int
    trace: tuple[dict[str, Any], ...]


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def initial_state_key(config: PlannerConfig) -> tuple[int, ...]:
    return tuple([0] * len(config.stages) + [0])


def key_to_memory(state_key: tuple[int, ...], config: PlannerConfig) -> dict[str, int]:
    return {
        stage_name: int(config.tiers[state_key[index]])
        for index, stage_name in enumerate(config.stages)
    }


def key_to_safety_factor(state_key: tuple[int, ...], config: PlannerConfig) -> float:
    return float(config.safety_factors[state_key[-1]])


def memory_config(eval_result: StateEval, config: PlannerConfig) -> str:
    return format_memory_config(eval_result.memory_tier_per_stage, config.stages)


def state_with_group_value(
    state_key: tuple[int, ...],
    group_index: int,
    value_index: int,
) -> tuple[int, ...]:
    out = list(state_key)
    out[group_index] = int(value_index)
    return tuple(out)


def group_count(config: PlannerConfig) -> int:
    return len(config.stages) + 1


def group_name(config: PlannerConfig, group_index: int) -> str:
    if group_index < len(config.stages):
        return config.stages[group_index]
    return "__entry_safety__"


def group_max_index(config: PlannerConfig, group_index: int) -> int:
    if group_index < len(config.stages):
        return len(config.tiers) - 1
    return len(config.safety_factors) - 1


def group_value_label(config: PlannerConfig, group_index: int, value_index: int) -> str:
    if group_index < len(config.stages):
        return str(config.tiers[value_index])
    return str(config.safety_factors[value_index])


def trace_row(
    *,
    step: int,
    action: str,
    evaluation: StateEval,
) -> dict[str, Any]:
    return {
        "step": int(step),
        "action": action,
        "violation_rate": evaluation.violation_rate,
        "expected_e2e_ms": evaluation.expected_e2e_ms,
        "cost_gbsec": evaluation.cost_gbsec,
    }


def is_feasible(eval_result: StateEval, config: PlannerConfig) -> bool:
    return eval_result.violation_rate <= config.max_violation_rate + EPS


def cost_gap(cost: float, brute_cost: float | None) -> float:
    if brute_cost is None or not math.isfinite(brute_cost) or brute_cost <= 0.0:
        return math.nan
    return (float(cost) - float(brute_cost)) / float(brute_cost) * 100.0


def single_change_candidates(
    *,
    ctx: EvalContext,
    state_key: tuple[int, ...],
    all_higher: bool,
) -> list[dict[str, Any]]:
    current = ctx.evaluate(state_key)
    candidates: list[dict[str, Any]] = []
    for group_index in range(group_count(ctx.config)):
        current_index = state_key[group_index]
        max_index = group_max_index(ctx.config, group_index)
        if current_index >= max_index:
            continue
        value_range: Iterable[int]
        if all_higher:
            value_range = range(current_index + 1, max_index + 1)
        else:
            value_range = [current_index + 1]
        for value_index in value_range:
            next_key = state_with_group_value(state_key, group_index, value_index)
            evaluation = ctx.evaluate(next_key)
            candidates.append(
                {
                    "group_index": group_index,
                    "group_name": group_name(ctx.config, group_index),
                    "value_index": value_index,
                    "value_label": group_value_label(ctx.config, group_index, value_index),
                    "state_key": next_key,
                    "evaluation": evaluation,
                    "risk_delta": current.violation_rate - evaluation.violation_rate,
                    "expected_delta": current.expected_e2e_ms - evaluation.expected_e2e_ms,
                    "cost_delta": evaluation.cost_gbsec - current.cost_gbsec,
                }
            )
    return candidates


def efficiency_key(candidate: dict[str, Any]) -> tuple[float, float, int, int]:
    risk_delta = float(candidate["risk_delta"])
    expected_delta = float(candidate["expected_delta"])
    cost_delta = float(candidate["cost_delta"])
    if risk_delta > EPS:
        if cost_delta <= 0.0:
            efficiency = math.inf
        else:
            efficiency = risk_delta / cost_delta
        primary = 0
        score = -efficiency
    elif expected_delta > 1e-9:
        if cost_delta <= 0.0:
            efficiency = math.inf
        else:
            efficiency = expected_delta / cost_delta
        primary = 1
        score = -efficiency
    else:
        primary = 2
        score = 0.0
    return (
        primary,
        score,
        abs(cost_delta),
        int(candidate["group_index"]),
    )


def greedy_plan(ctx: EvalContext) -> PlanResult:
    state_key = initial_state_key(ctx.config)
    current = ctx.evaluate(state_key)
    trace = [trace_row(step=0, action="init", evaluation=current)]
    iterations = 0
    max_steps = (len(ctx.config.tiers) - 1) * len(ctx.config.stages) + (
        len(ctx.config.safety_factors) - 1
    )

    while not is_feasible(current, ctx.config) and iterations < max_steps:
        candidates = single_change_candidates(ctx=ctx, state_key=state_key, all_higher=False)
        improving = [
            item
            for item in candidates
            if item["risk_delta"] > EPS
            or (
                item["expected_delta"] > 1e-9
                and item["evaluation"].violation_rate <= current.violation_rate + EPS
            )
        ]
        if not improving:
            break
        chosen = sorted(improving, key=efficiency_key)[0]
        state_key = chosen["state_key"]
        current = chosen["evaluation"]
        iterations += 1
        action = (
            f"set {chosen['group_name']} -> {chosen['value_label']}"
            f" risk_delta={chosen['risk_delta']:.6g}"
            f" cost_delta={chosen['cost_delta']:.6g}"
        )
        trace.append(trace_row(step=iterations, action=action, evaluation=current))

    return PlanResult(
        method="greedy",
        state_key=state_key,
        evaluation=current,
        feasible=is_feasible(current, ctx.config),
        iterations=iterations,
        states_evaluated=len(ctx.eval_cache),
        trace=tuple(trace),
    )


def beam_rank_key(eval_result: StateEval, config: PlannerConfig) -> tuple[float, float, float, tuple[int, ...]]:
    if is_feasible(eval_result, config):
        return (0.0, eval_result.cost_gbsec, eval_result.violation_rate, eval_result.state_key)
    return (1.0, eval_result.violation_rate, eval_result.cost_gbsec, eval_result.state_key)


def beam_plan(ctx: EvalContext, beam_width: int) -> PlanResult:
    if beam_width <= 0:
        raise ValueError(f"beam_width must be positive, got {beam_width}")
    start_key = initial_state_key(ctx.config)
    start_eval = ctx.evaluate(start_key)
    beam: list[tuple[tuple[int, ...], tuple[dict[str, Any], ...]]] = [
        (start_key, (trace_row(step=0, action="init", evaluation=start_eval),))
    ]
    expanded: set[tuple[int, ...]] = set()
    best_feasible: tuple[tuple[int, ...], tuple[dict[str, Any], ...]] | None = None
    iterations = 0
    max_iterations = (len(ctx.config.tiers) - 1) * len(ctx.config.stages) + (
        len(ctx.config.safety_factors) - 1
    )

    for iteration in range(1, max_iterations + 1):
        pooled: dict[tuple[int, ...], tuple[tuple[int, ...], tuple[dict[str, Any], ...]]] = {}
        for state_key, trace in beam:
            expanded.add(state_key)
            for cand in single_change_candidates(ctx=ctx, state_key=state_key, all_higher=False):
                next_key = cand["state_key"]
                if next_key in expanded:
                    continue
                action = f"set {cand['group_name']} -> {cand['value_label']}"
                next_trace = trace + (
                    trace_row(step=len(trace), action=action, evaluation=cand["evaluation"]),
                )
                existing = pooled.get(next_key)
                if existing is None:
                    pooled[next_key] = (next_key, next_trace)
                else:
                    if beam_rank_key(cand["evaluation"], ctx.config) < beam_rank_key(
                        ctx.evaluate(existing[0]), ctx.config
                    ):
                        pooled[next_key] = (next_key, next_trace)

        if not pooled:
            break
        candidates = list(pooled.values())
        for item in candidates:
            evaluation = ctx.evaluate(item[0])
            if is_feasible(evaluation, ctx.config):
                if best_feasible is None or beam_rank_key(evaluation, ctx.config) < beam_rank_key(
                    ctx.evaluate(best_feasible[0]), ctx.config
                ):
                    best_feasible = item
        candidates.sort(key=lambda item: beam_rank_key(ctx.evaluate(item[0]), ctx.config))
        beam = candidates[:beam_width]
        iterations = iteration

    if best_feasible is not None:
        state_key, trace = best_feasible
    else:
        state_key, trace = sorted(
            beam, key=lambda item: beam_rank_key(ctx.evaluate(item[0]), ctx.config)
        )[0]
    evaluation = ctx.evaluate(state_key)
    return PlanResult(
        method=f"beam_k{beam_width}",
        state_key=state_key,
        evaluation=evaluation,
        feasible=is_feasible(evaluation, ctx.config),
        iterations=iterations,
        states_evaluated=len(ctx.eval_cache),
        trace=trace,
    )


def build_lambda_grid(effects: list[dict[str, Any]]) -> list[float]:
    ratios: list[float] = [0.0]
    for item in effects:
        risk_delta = float(item["risk_delta"])
        cost_delta = float(item["cost_delta"])
        if risk_delta <= EPS:
            continue
        if cost_delta <= 0.0:
            ratios.append(0.0)
            continue
        base = cost_delta / risk_delta
        for scale in [0.25, 0.5, 1.0, 2.0, 4.0]:
            ratios.append(base * scale)
    finite = sorted({round(value, 12) for value in ratios if math.isfinite(value) and value >= 0.0})
    if finite:
        finite.append(finite[-1] * 10.0 + 1.0)
    else:
        finite = [0.0, 1.0]
    return finite


def choose_by_lambda(
    *,
    ctx: EvalContext,
    state_key: tuple[int, ...],
    effects: list[dict[str, Any]],
    lambda_value: float,
) -> tuple[int, ...]:
    chosen = list(state_key)
    by_group: dict[int, list[dict[str, Any]]] = {}
    for item in effects:
        by_group.setdefault(int(item["group_index"]), []).append(item)
    for group_index, items in by_group.items():
        best_score = 0.0
        best_value = state_key[group_index]
        for item in items:
            score = float(item["cost_delta"]) - lambda_value * float(item["risk_delta"])
            if score < best_score - EPS:
                best_score = score
                best_value = int(item["value_index"])
        chosen[group_index] = best_value
    return tuple(chosen)


def repair_until_feasible(ctx: EvalContext, state_key: tuple[int, ...]) -> tuple[int, ...]:
    current = ctx.evaluate(state_key)
    max_steps = (len(ctx.config.tiers) - 1) * len(ctx.config.stages) + (
        len(ctx.config.safety_factors) - 1
    )
    steps = 0
    while not is_feasible(current, ctx.config) and steps < max_steps:
        candidates = single_change_candidates(ctx=ctx, state_key=state_key, all_higher=True)
        improving = [
            item
            for item in candidates
            if item["risk_delta"] > EPS
            or (
                item["expected_delta"] > 1e-9
                and item["evaluation"].violation_rate <= current.violation_rate + EPS
            )
        ]
        if not improving:
            break
        chosen = sorted(improving, key=efficiency_key)[0]
        state_key = chosen["state_key"]
        current = chosen["evaluation"]
        steps += 1
    return state_key


def local_cost_improve(
    ctx: EvalContext,
    state_key: tuple[int, ...],
    *,
    pairwise: bool,
) -> tuple[int, ...]:
    current = ctx.evaluate(state_key)
    if not is_feasible(current, ctx.config):
        return state_key
    improved = True
    while improved:
        improved = False
        best_key = state_key
        best_eval = current
        for group_index in range(group_count(ctx.config)):
            max_index = group_max_index(ctx.config, group_index)
            for value_index in range(0, max_index + 1):
                if value_index == state_key[group_index]:
                    continue
                candidate_key = state_with_group_value(state_key, group_index, value_index)
                candidate_eval = ctx.evaluate(candidate_key)
                if not is_feasible(candidate_eval, ctx.config):
                    continue
                if candidate_eval.cost_gbsec < best_eval.cost_gbsec - EPS:
                    best_key = candidate_key
                    best_eval = candidate_eval
        if best_key != state_key:
            state_key = best_key
            current = best_eval
            improved = True
            continue

        if not pairwise:
            break

        for left_group in range(group_count(ctx.config)):
            for right_group in range(left_group + 1, group_count(ctx.config)):
                left_max = group_max_index(ctx.config, left_group)
                right_max = group_max_index(ctx.config, right_group)
                for left_value in range(0, left_max + 1):
                    if left_value == state_key[left_group]:
                        continue
                    for right_value in range(0, right_max + 1):
                        if right_value == state_key[right_group]:
                            continue
                        candidate = list(state_key)
                        candidate[left_group] = left_value
                        candidate[right_group] = right_value
                        candidate_key = tuple(candidate)
                        candidate_eval = ctx.evaluate(candidate_key)
                        if not is_feasible(candidate_eval, ctx.config):
                            continue
                        if candidate_eval.cost_gbsec < best_eval.cost_gbsec - EPS:
                            best_key = candidate_key
                            best_eval = candidate_eval

        if best_key != state_key:
            state_key = best_key
            current = best_eval
            improved = True
    return state_key


def risk_price_plan(ctx: EvalContext, *, pairwise: bool) -> PlanResult:
    start_key = initial_state_key(ctx.config)
    start_eval = ctx.evaluate(start_key)
    if is_feasible(start_eval, ctx.config):
        return PlanResult(
            method="risk_price",
            state_key=start_key,
            evaluation=start_eval,
            feasible=True,
            iterations=0,
            states_evaluated=len(ctx.eval_cache),
            trace=(trace_row(step=0, action="init feasible", evaluation=start_eval),),
        )

    effects = single_change_candidates(ctx=ctx, state_key=start_key, all_higher=True)
    lambdas = build_lambda_grid(effects)
    candidate_keys: set[tuple[int, ...]] = set()
    trace: list[dict[str, Any]] = [trace_row(step=0, action="init", evaluation=start_eval)]

    for lambda_value in lambdas:
        proposed = choose_by_lambda(
            ctx=ctx,
            state_key=start_key,
            effects=effects,
            lambda_value=lambda_value,
        )
        repaired = repair_until_feasible(ctx, proposed)
        improved = local_cost_improve(ctx, repaired, pairwise=pairwise)
        candidate_keys.add(improved)

    # Add a pure greedy-repair seed so the method always has a robust fallback.
    candidate_keys.add(
        local_cost_improve(ctx, repair_until_feasible(ctx, start_key), pairwise=pairwise)
    )

    feasible_keys = [
        key for key in candidate_keys if is_feasible(ctx.evaluate(key), ctx.config)
    ]
    if feasible_keys:
        best_key = min(
            feasible_keys,
            key=lambda key: (
                ctx.evaluate(key).cost_gbsec,
                ctx.evaluate(key).violation_rate,
                key,
            ),
        )
    else:
        best_key = min(
            candidate_keys,
            key=lambda key: (
                ctx.evaluate(key).violation_rate,
                ctx.evaluate(key).cost_gbsec,
                key,
            ),
        )
    best_eval = ctx.evaluate(best_key)
    trace.append(
        trace_row(
            step=1,
            action=f"risk-price sweep over {len(lambdas)} lambda values",
            evaluation=best_eval,
        )
    )
    return PlanResult(
        method="risk_price_pairwise" if pairwise else "risk_price_fast",
        state_key=best_key,
        evaluation=best_eval,
        feasible=is_feasible(best_eval, ctx.config),
        iterations=len(lambdas),
        states_evaluated=len(ctx.eval_cache),
        trace=tuple(trace),
    )


def load_brute_force_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    rows: dict[str, dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        rows[str(row["slo_class"])] = row
    return rows


def result_row(
    *,
    slo_class: str,
    config: PlannerConfig,
    result: PlanResult,
    brute_row: dict[str, Any] | None,
) -> dict[str, Any]:
    brute_cost = None
    brute_config = ""
    if brute_row is not None:
        brute_cost = float(brute_row["optimal_cost_gbsec"])
        brute_config = str(brute_row["optimal_memory_config"])
    config_string = memory_config(result.evaluation, config)
    return {
        "slo_class": slo_class,
        "slo_ms": config.slo_ms,
        "method": result.method,
        "cost_gbsec": result.evaluation.cost_gbsec,
        "violation_rate": result.evaluation.violation_rate,
        "expected_e2e_ms": result.evaluation.expected_e2e_ms,
        "entry_prewarm_safety_factor": result.evaluation.safety_factor,
        "entry_prewarm_count": result.evaluation.entry_prewarm_count,
        "feasible": result.feasible,
        "iterations": result.iterations,
        "states_evaluated": result.states_evaluated,
        "cost_gap_vs_brute_pct": cost_gap(result.evaluation.cost_gbsec, brute_cost),
        "configs_match_brute": bool(config_string == brute_config) if brute_config else False,
        "memory_config": config_string,
    }


def brute_result_row(
    *,
    slo_class: str,
    config: PlannerConfig,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "slo_class": slo_class,
        "slo_ms": config.slo_ms,
        "method": "brute_force",
        "cost_gbsec": float(row["optimal_cost_gbsec"]),
        "violation_rate": float(row["optimal_violation_rate"]),
        "expected_e2e_ms": math.nan,
        "entry_prewarm_safety_factor": float(row["optimal_safety_factor"]),
        "entry_prewarm_count": entry_prewarm_count(
            float(row["optimal_safety_factor"]), config.predicted_arrivals
        ),
        "feasible": float(row["optimal_violation_rate"]) <= config.max_violation_rate + EPS,
        "iterations": 0,
        "states_evaluated": int(row["n_total_evaluated"]),
        "cost_gap_vs_brute_pct": 0.0,
        "configs_match_brute": True,
        "memory_config": str(row["optimal_memory_config"]),
    }


def complexity_rows(
    *,
    n_stages: int,
    n_tiers: int,
    n_safety: int,
    beam_widths: list[int],
    h_lambdas: int,
) -> list[dict[str, Any]]:
    groups = n_stages + 1
    upgrade_depth = n_stages * (n_tiers - 1) + (n_safety - 1)
    full_space = (n_tiers**n_stages) * n_safety
    rows = [
        {
            "method": "brute_force",
            "asymptotic": "O(|S| * |T|^n)",
            "model_eval_bound": full_space,
            "note": "Exact oracle; exponential in stage count.",
        },
        {
            "method": "greedy",
            "asymptotic": "O(U * (n+1))",
            "model_eval_bound": upgrade_depth * groups,
            "note": "U = total one-step upgrades; next-tier only.",
        },
        {
            "method": "risk_price_fast",
            "asymptotic": "O(n|T| + h * repair + coordinate prune)",
            "model_eval_bound": n_stages * n_tiers + n_safety + h_lambdas * upgrade_depth * groups,
            "note": "Dynamic-friendly variant: lambda proposals, exact repair, single-coordinate prune.",
        },
        {
            "method": "risk_price_pairwise",
            "asymptotic": "O(n|T| + h * repair + pairwise local search)",
            "model_eval_bound": (
                n_stages * n_tiers
                + n_safety
                + h_lambdas
                * upgrade_depth
                * (n_stages * n_tiers + n_safety)
            ),
            "note": "Offline stronger variant: pairwise local improvement and exact-risk verification.",
        },
    ]
    for width in beam_widths:
        rows.append(
            {
                "method": f"beam_k{width}",
                "asymptotic": "O(k * U * (n+1))",
                "model_eval_bound": width * upgrade_depth * groups,
                "note": "Heuristic width-k frontier.",
            }
        )
    return rows


def write_report(
    *,
    path: Path,
    method_df: pd.DataFrame,
    complexity_df: pd.DataFrame,
    rho: float,
    contention_factor: float,
    brute_force_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Risk-Price Planner Suite")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Risk model: `compute_plan_risk(rho={rho}, contention_factor={contention_factor})`.")
    lines.append(f"- Brute-force oracle: `{brute_force_path}`.")
    lines.append("- Decision space: per-stage memory tier plus entry prewarm safety factor.")
    lines.append("")
    lines.append("## Method Comparison")
    lines.append("```text")
    lines.append(method_df.round(8).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Complexity")
    lines.append("```text")
    lines.append(complexity_df.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append("- `risk_price` uses risk-price lambda proposals, exact repair, and exact feasibility verification.")
    lines.append("- `brute_force` is retained as a small-workflow oracle, not as the scalable planner.")
    lines.append("- Real-machine replay should use the selected risk-price plans only after checking cluster capacity and warm-pool settings.")
    path.write_text("\n".join(lines) + "\n")


def run_suite(
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    brute_force_path: str | Path = DEFAULT_BRUTE_FORCE,
    lognormal_params_path: str | Path = DEFAULT_LOGNORMAL_PARAMS,
    baseline_trace_path: str | Path = DEFAULT_BASELINE_TRACE,
    predicted_arrivals: float = 5.0,
    rho: float = DEFAULT_RHO,
    contention_factor: float = DEFAULT_CONTENTION_FACTOR,
    beam_widths: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    beam_widths = beam_widths or [3, 5]
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref_data = load_reference_data(
        lognormal_params_path=lognormal_params_path,
        baseline_trace_path=baseline_trace_path,
    )
    brute_rows = load_brute_force_rows(resolve(brute_force_path))
    method_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    h_lambdas_max = 0
    for slo_class, slo_ms in [("premium", 15000.0), ("free", 20000.0)]:
        config = PlannerConfig(
            slo_ms=slo_ms,
            max_violation_rate=0.05,
            predicted_arrivals=predicted_arrivals,
            tiers=list(DEFAULT_TIERS),
            safety_factors=list(DEFAULT_SAFETY_FACTORS),
            stages=list(STAGES),
        )
        brute_row = brute_rows.get(slo_class)

        for runner in ["greedy", "risk_price_fast", "risk_price_pairwise"]:
            ctx = EvalContext(
                config=config,
                ref_data=ref_data,
                rho=float(rho),
                contention_factor=float(contention_factor),
                eval_cache={},
            )
            if runner == "greedy":
                result = greedy_plan(ctx)
            else:
                start_key = initial_state_key(config)
                effects = single_change_candidates(ctx=ctx, state_key=start_key, all_higher=True)
                h_lambdas_max = max(h_lambdas_max, len(build_lambda_grid(effects)))
                result = risk_price_plan(ctx, pairwise=(runner == "risk_price_pairwise"))
            method_rows.append(
                result_row(
                    slo_class=slo_class,
                    config=config,
                    result=result,
                    brute_row=brute_row,
                )
            )
            for item in result.trace:
                trace_rows.append({"slo_class": slo_class, "method": result.method, **item})

        for width in beam_widths:
            ctx = EvalContext(
                config=config,
                ref_data=ref_data,
                rho=float(rho),
                contention_factor=float(contention_factor),
                eval_cache={},
            )
            result = beam_plan(ctx, beam_width=width)
            method_rows.append(
                result_row(
                    slo_class=slo_class,
                    config=config,
                    result=result,
                    brute_row=brute_row,
                )
            )
            for item in result.trace:
                trace_rows.append({"slo_class": slo_class, "method": result.method, **item})

        if brute_row is not None:
            method_rows.append(brute_result_row(slo_class=slo_class, config=config, row=brute_row))

    method_df = pd.DataFrame(method_rows)
    trace_df = pd.DataFrame(trace_rows)
    complexity_df = pd.DataFrame(
        complexity_rows(
            n_stages=len(STAGES),
            n_tiers=len(DEFAULT_TIERS),
            n_safety=len(DEFAULT_SAFETY_FACTORS),
            beam_widths=beam_widths,
            h_lambdas=h_lambdas_max,
        )
    )

    method_df.to_csv(out / "method_comparison.csv", index=False)
    trace_df.to_csv(out / "risk_price_trace.csv", index=False)
    complexity_df.to_csv(out / "complexity_summary.csv", index=False)
    write_report(
        path=out / "report.md",
        method_df=method_df,
        complexity_df=complexity_df,
        rho=float(rho),
        contention_factor=float(contention_factor),
        brute_force_path=resolve(brute_force_path),
    )
    return {
        "method_comparison": method_df,
        "risk_price_trace": trace_df,
        "complexity_summary": complexity_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--brute-force-path", default=str(DEFAULT_BRUTE_FORCE))
    parser.add_argument("--lognormal-params", default=str(DEFAULT_LOGNORMAL_PARAMS))
    parser.add_argument("--baseline-trace", default=str(DEFAULT_BASELINE_TRACE))
    parser.add_argument("--predicted-arrivals", type=float, default=5.0)
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument("--contention-factor", type=float, default=DEFAULT_CONTENTION_FACTOR)
    parser.add_argument("--beam-width", type=int, action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_suite(
        out_dir=args.out_dir,
        brute_force_path=args.brute_force_path,
        lognormal_params_path=args.lognormal_params,
        baseline_trace_path=args.baseline_trace,
        predicted_arrivals=args.predicted_arrivals,
        rho=args.rho,
        contention_factor=args.contention_factor,
        beam_widths=args.beam_width,
    )
    print("method_comparison:")
    print(outputs["method_comparison"].round(8).to_string(index=False))
    print()
    print("complexity_summary:")
    print(outputs["complexity_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
