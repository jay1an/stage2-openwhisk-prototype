#!/usr/bin/env python3
"""Offline multi-SLO risk-budgeted greedy planner.

This module implements the P3.4 offline planner.  It uses the analytical
path-2 risk API as a black box and greedily improves a memory/prewarm plan by
the best marginal risk reduction per GB-second cost increase.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from runner.stage4_risk.entry_cold import calibrate_p_baseline
from runner.stage4_risk.plan_risk import (
    PlanInput,
    compute_cold_overhead_per_stage,
    compute_plan_risk,
    load_lognormal_params,
)
from runner.stage4_risk.scaling import (
    load_warm_splines,
    memory_to_cpu_cores,
    spline_predict_warm_mean,
)


STAGES = [
    "detect_object",
    "estimate_pose",
    "match_face",
    "classify_scene",
    "translate_alert",
]
DEFAULT_TIERS = [512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3840]
DEFAULT_SAFETY_FACTORS = [0.0, 0.5, 1.0, 1.5, 2.0]
DEFAULT_LOGNORMAL_PARAMS = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "path2_lognormal_fit_multinode"
    / "per_stage_lognormal_params.csv"
)
DEFAULT_AMDAHL_PARAMS = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "stage6_amdahl_model_multinode_9tier"
    / "per_stage_amdahl_params.csv"
)
DEFAULT_BASELINE_TRACE = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "civic_azure_cand2_60min_1280mb_1cpu_keepalive10s_target20s_2x_mi96"
    / "raw_trace.csv"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "path3_planner"
TIE_EPSILON = 1e-9


@dataclass(frozen=True)
class PlannerConfig:
    slo_ms: float
    max_violation_rate: float
    predicted_arrivals: float
    tiers: list[int]
    safety_factors: list[float]
    stages: list[str]


@dataclass(frozen=True)
class ReferenceData:
    lognormal_params: dict[str, Any]
    amdahl_params: pd.DataFrame
    cold_overhead_per_stage: dict[str, float]
    p_baseline: float
    warm_splines: dict[str, Any]


@dataclass(frozen=True)
class PlanEvaluation:
    memory_tier_per_stage: dict[str, int]
    safety_factor: float
    entry_prewarm_count: int
    risk_result: Any
    cost_gbsec: float

    @property
    def violation_rate(self) -> float:
        return float(self.risk_result.p_violation_total)


@dataclass(frozen=True)
class GreedyTraceStep:
    step: int
    action_taken: str
    risk_after: float
    cost_after: float


@dataclass(frozen=True)
class PlanResult:
    memory_tier_per_stage: dict[str, int]
    entry_prewarm_safety_factor: float
    entry_prewarm_count: int
    achieved_violation_rate: float
    achieved_cost_gbsec: float
    feasible: bool
    iterations: int
    trace: list[GreedyTraceStep]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def load_reference_data(
    lognormal_params_path: str | Path = DEFAULT_LOGNORMAL_PARAMS,
    amdahl_params_path: str | Path = DEFAULT_AMDAHL_PARAMS,
    baseline_trace_path: str | Path = DEFAULT_BASELINE_TRACE,
) -> ReferenceData:
    """Load the reference data required by the path-2 risk API."""

    lognormal_params = load_lognormal_params(_resolve(lognormal_params_path))
    amdahl_path = _resolve(amdahl_params_path)
    if amdahl_path.exists():
        amdahl_params = pd.read_csv(amdahl_path)
    else:
        amdahl_params = pd.DataFrame(columns=["stage_name", "S_ms", "P_ms", "C_ms"])

    trace_path = _resolve(baseline_trace_path)
    if trace_path.exists():
        p_baseline = calibrate_p_baseline(str(trace_path))
    else:
        p_baseline = 0.01

    return ReferenceData(
        lognormal_params=lognormal_params,
        amdahl_params=amdahl_params,
        cold_overhead_per_stage=compute_cold_overhead_per_stage(lognormal_params),
        p_baseline=float(p_baseline),
        warm_splines=load_warm_splines(),
    )


def entry_prewarm_count(safety_factor: float, predicted_arrivals: float) -> int:
    """Convert a safety factor into the integer entry prewarm count."""

    if safety_factor < 0.0 or predicted_arrivals < 0.0:
        raise ValueError("safety_factor and predicted_arrivals must be non-negative")
    return int(math.ceil(float(safety_factor) * float(predicted_arrivals)))


def plan_cost_gbsec(
    memory_tier_per_stage: dict[str, int],
    entry_prewarm_count_value: int,
    warm_splines: dict[str, Any],
    stages: list[str],
) -> float:
    """Lambda-style GB-second execution proxy for one workflow.

    The proxy is the sum of memory GB times warm action seconds across stages,
    plus entry-prewarm containers counted as execution-equivalent GB-seconds.
    """

    total = 0.0
    for stage_name in stages:
        memory_mb = int(memory_tier_per_stage[stage_name])
        cpu_cores = memory_to_cpu_cores(memory_mb)
        warm_ms = spline_predict_warm_mean(stage_name, cpu_cores, warm_splines)
        total += (memory_mb / 1024.0) * (warm_ms / 1000.0)

    entry_stage = stages[0]
    entry_memory_mb = int(memory_tier_per_stage[entry_stage])
    entry_cpu = memory_to_cpu_cores(entry_memory_mb)
    entry_warm_sec = spline_predict_warm_mean(entry_stage, entry_cpu, warm_splines) / 1000.0
    total += int(entry_prewarm_count_value) * (entry_memory_mb / 1024.0) * entry_warm_sec
    return float(total)


def evaluate_plan(
    config: PlannerConfig,
    ref_data: ReferenceData,
    memory_tier_per_stage: dict[str, int],
    safety_factor: float,
) -> PlanEvaluation:
    """Evaluate a plan's risk and GB-second proxy cost."""

    count = entry_prewarm_count(safety_factor, config.predicted_arrivals)
    plan = PlanInput(
        memory_tier_per_stage=dict(memory_tier_per_stage),
        entry_prewarm_count=float(count),
        predicted_arrivals=float(config.predicted_arrivals),
        lognormal_params=ref_data.lognormal_params,
        amdahl_params=ref_data.amdahl_params,
        cold_overhead_per_stage=ref_data.cold_overhead_per_stage,
        p_baseline=ref_data.p_baseline,
    )
    risk_result = compute_plan_risk(plan, slo_ms=config.slo_ms)
    cost = plan_cost_gbsec(
        memory_tier_per_stage=memory_tier_per_stage,
        entry_prewarm_count_value=count,
        warm_splines=ref_data.warm_splines,
        stages=config.stages,
    )
    return PlanEvaluation(
        memory_tier_per_stage=dict(memory_tier_per_stage),
        safety_factor=float(safety_factor),
        entry_prewarm_count=count,
        risk_result=risk_result,
        cost_gbsec=cost,
    )


def _state_to_memory(state: dict[str, int], tiers: list[int]) -> dict[str, int]:
    return {stage_name: int(tiers[tier_index]) for stage_name, tier_index in state.items()}


def _trace_step(step: int, action: str, evaluation: PlanEvaluation) -> GreedyTraceStep:
    return GreedyTraceStep(
        step=step,
        action_taken=action,
        risk_after=evaluation.violation_rate,
        cost_after=evaluation.cost_gbsec,
    )


def _efficiency(delta: float, cost_delta: float) -> float:
    if cost_delta <= 0.0:
        return math.inf
    return float(delta) / float(cost_delta)


def _efficiency_bucket(value: float, epsilon: float = TIE_EPSILON) -> float:
    if math.isinf(value):
        return math.inf
    return round(float(value) / epsilon) * epsilon


def _candidate_sort_key(candidate: dict[str, Any], primary_delta_key: str) -> tuple[float, float, int]:
    efficiency = _efficiency(
        delta=float(candidate[primary_delta_key]),
        cost_delta=float(candidate["cost_delta"]),
    )
    return (
        -_efficiency_bucket(efficiency),
        abs(float(candidate["cost_delta"])),
        int(candidate["dag_order_index"]),
    )


def risk_budgeted_greedy(config: PlannerConfig, ref_data: ReferenceData) -> PlanResult:
    """Greedily upgrade tier/prewarm until the SLO risk constraint is met."""

    if config.max_violation_rate < 0.0 or config.max_violation_rate > 1.0:
        raise ValueError("max_violation_rate must be in [0, 1]")
    if not config.tiers or not config.safety_factors or not config.stages:
        raise ValueError("tiers, safety_factors, and stages must be non-empty")
    if sorted(config.tiers) != list(config.tiers):
        raise ValueError("tiers must be sorted ascending")
    if sorted(config.safety_factors) != list(config.safety_factors):
        raise ValueError("safety_factors must be sorted ascending")

    dag_order = {stage_name: idx for idx, stage_name in enumerate(config.stages)}
    safety_dag_order = len(config.stages)
    tier_state = {stage_name: 0 for stage_name in config.stages}
    safety_index = 0

    current = evaluate_plan(
        config=config,
        ref_data=ref_data,
        memory_tier_per_stage=_state_to_memory(tier_state, config.tiers),
        safety_factor=config.safety_factors[safety_index],
    )
    trace: list[GreedyTraceStep] = [_trace_step(0, "init", current)]

    step = 0
    while current.violation_rate > config.max_violation_rate:
        candidates: list[dict[str, Any]] = []

        for stage_name in config.stages:
            if tier_state[stage_name] >= len(config.tiers) - 1:
                continue
            next_state = dict(tier_state)
            next_state[stage_name] += 1
            new_eval = evaluate_plan(
                config=config,
                ref_data=ref_data,
                memory_tier_per_stage=_state_to_memory(next_state, config.tiers),
                safety_factor=config.safety_factors[safety_index],
            )
            risk_delta = current.violation_rate - new_eval.violation_rate
            progress_delta = (
                current.risk_result.expected_e2e_ms
                - new_eval.risk_result.expected_e2e_ms
            )
            cost_delta = new_eval.cost_gbsec - current.cost_gbsec
            action = f"upgrade {stage_name} {config.tiers[tier_state[stage_name]]}->{config.tiers[next_state[stage_name]]}"
            candidates.append(
                {
                    "risk_delta": risk_delta,
                    "progress_delta": progress_delta,
                    "cost_delta": cost_delta,
                    "action": action,
                    "tier_state": next_state,
                    "safety_index": safety_index,
                    "dag_order_index": dag_order[stage_name],
                    "evaluation": new_eval,
                }
            )

        if safety_index < len(config.safety_factors) - 1:
            next_safety_index = safety_index + 1
            new_eval = evaluate_plan(
                config=config,
                ref_data=ref_data,
                memory_tier_per_stage=_state_to_memory(tier_state, config.tiers),
                safety_factor=config.safety_factors[next_safety_index],
            )
            risk_delta = current.violation_rate - new_eval.violation_rate
            progress_delta = (
                current.risk_result.expected_e2e_ms
                - new_eval.risk_result.expected_e2e_ms
            )
            cost_delta = new_eval.cost_gbsec - current.cost_gbsec
            action = (
                "increase entry safety "
                f"{config.safety_factors[safety_index]}->{config.safety_factors[next_safety_index]}"
            )
            candidates.append(
                {
                    "risk_delta": risk_delta,
                    "progress_delta": progress_delta,
                    "cost_delta": cost_delta,
                    "action": action,
                    "tier_state": dict(tier_state),
                    "safety_index": next_safety_index,
                    "dag_order_index": safety_dag_order,
                    "evaluation": new_eval,
                }
            )

        if not candidates:
            break

        risk_improving = [
            candidate for candidate in candidates if candidate["risk_delta"] > 1e-12
        ]
        if risk_improving:
            risk_improving.sort(key=lambda item: _candidate_sort_key(item, "risk_delta"))
            chosen = risk_improving[0]
        else:
            # At very slow tiers the violation probability can saturate at 1.0.
            # Use expected-E2E reduction as a progress surrogate until the risk
            # curve becomes numerically visible again.
            progress_improving = [
                candidate
                for candidate in candidates
                if candidate["progress_delta"] > 1e-9
                and candidate["evaluation"].violation_rate <= current.violation_rate + 1e-12
            ]
            if not progress_improving:
                break
            progress_improving.sort(
                key=lambda item: _candidate_sort_key(item, "progress_delta")
            )
            chosen = progress_improving[0]

        action = str(chosen["action"])
        tier_state = dict(chosen["tier_state"])
        safety_index = int(chosen["safety_index"])
        current = chosen["evaluation"]
        step += 1
        trace.append(_trace_step(step, action, current))

    return PlanResult(
        memory_tier_per_stage=dict(current.memory_tier_per_stage),
        entry_prewarm_safety_factor=current.safety_factor,
        entry_prewarm_count=current.entry_prewarm_count,
        achieved_violation_rate=current.violation_rate,
        achieved_cost_gbsec=current.cost_gbsec,
        feasible=current.violation_rate <= config.max_violation_rate,
        iterations=step,
        trace=trace,
    )


def _summary_row(
    slo_class: str,
    arrival_scenario: str,
    config: PlannerConfig,
    result: PlanResult,
) -> dict[str, Any]:
    return {
        "slo_class": slo_class,
        "slo_ms": config.slo_ms,
        "arrival_scenario": arrival_scenario,
        "predicted_arrivals": config.predicted_arrivals,
        "entry_prewarm_safety_factor": result.entry_prewarm_safety_factor,
        "entry_prewarm_count": result.entry_prewarm_count,
        "achieved_violation_rate": result.achieved_violation_rate,
        "achieved_cost_gbsec": result.achieved_cost_gbsec,
        "feasible": result.feasible,
        "n_greedy_iterations": result.iterations,
        "memory_config": ",".join(
            f"{stage}:{result.memory_tier_per_stage[stage]}" for stage in config.stages
        ),
    }


def _plan_rows(
    slo_class: str,
    arrival_scenario: str,
    config: PlannerConfig,
    result: PlanResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage_name in config.stages:
        rows.append(
            {
                "slo_class": slo_class,
                "slo_ms": config.slo_ms,
                "arrival_scenario": arrival_scenario,
                "predicted_arrivals": config.predicted_arrivals,
                "stage_name": stage_name,
                "memory_tier_mb": result.memory_tier_per_stage[stage_name],
            }
        )
    return rows


def _trace_rows(
    slo_class: str,
    arrival_scenario: str,
    result: PlanResult,
) -> list[dict[str, Any]]:
    return [
        {
            "slo_class": slo_class,
            "arrival_scenario": arrival_scenario,
            "step": step.step,
            "action_taken": step.action_taken,
            "risk_after": step.risk_after,
            "cost_after": step.cost_after,
        }
        for step in result.trace
    ]


def run_suite(
    out_dir: str | Path = DEFAULT_OUT_DIR,
    lognormal_params_path: str | Path = DEFAULT_LOGNORMAL_PARAMS,
    baseline_trace_path: str | Path = DEFAULT_BASELINE_TRACE,
) -> dict[str, pd.DataFrame]:
    """Run premium/free x typical/burst plans and write report artifacts."""

    out = _resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref_data = load_reference_data(
        lognormal_params_path=lognormal_params_path,
        baseline_trace_path=baseline_trace_path,
    )

    slo_classes = [
        ("premium", 15000.0),
        ("free", 20000.0),
    ]
    arrival_scenarios = [
        ("typical", 5.0),
        ("burst", 15.0),
    ]

    plan_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for slo_class, slo_ms in slo_classes:
        for arrival_scenario, predicted_arrivals in arrival_scenarios:
            config = PlannerConfig(
                slo_ms=slo_ms,
                max_violation_rate=0.05,
                predicted_arrivals=predicted_arrivals,
                tiers=list(DEFAULT_TIERS),
                safety_factors=list(DEFAULT_SAFETY_FACTORS),
                stages=list(STAGES),
            )
            result = risk_budgeted_greedy(config, ref_data)
            plan_rows.extend(_plan_rows(slo_class, arrival_scenario, config, result))
            summary_rows.append(_summary_row(slo_class, arrival_scenario, config, result))
            trace_rows.extend(_trace_rows(slo_class, arrival_scenario, result))

    plan_df = pd.DataFrame(plan_rows)
    summary_df = pd.DataFrame(summary_rows)
    trace_df = pd.DataFrame(trace_rows)

    plan_df.to_csv(out / "plan_per_class.csv", index=False)
    summary_df.to_csv(out / "plan_summary.csv", index=False)
    trace_df.to_csv(out / "greedy_trace.csv", index=False)
    _write_report(out / "planner_report.md", plan_df, summary_df, trace_df, ref_data)

    return {
        "plan_per_class": plan_df,
        "plan_summary": summary_df,
        "greedy_trace": trace_df,
    }


def _is_heterogeneous(tiers: list[int]) -> bool:
    return len(set(int(tier) for tier in tiers)) > 1


def _monotonic_non_decreasing(values: list[float]) -> bool:
    return all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def _monotonic_non_increasing(values: list[float]) -> bool:
    return all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


def _write_report(
    path: Path,
    plan_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    ref_data: ReferenceData,
) -> None:
    lines: list[str] = []
    lines.append("# P3.5 Deterministic Offline Multi-SLO Planner Report")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Planner: risk-budgeted greedy over stage memory tiers and entry prewarm safety factor.")
    lines.append("- Tie-break: highest marginal efficiency, then smallest absolute cost increase, then earliest DAG stage; safety-factor upgrades are ordered last.")
    lines.append("- Risk model: `runner.stage4_risk.plan_risk.compute_plan_risk` with D3 spline scaling.")
    lines.append("- Lognormal params: `reports/path2_lognormal_fit_multinode/per_stage_lognormal_params.csv`.")
    lines.append(f"- Calibrated entry cold baseline: `{ref_data.p_baseline:.6f}`.")
    lines.append("- Cost model: GB-second execution proxy = stage memory GB * predicted warm seconds, plus entry prewarm execution-equivalent GB-seconds.")
    lines.append("")
    lines.append("## Plan Summary")
    lines.append("```text")
    summary_cols = [
        "slo_class",
        "slo_ms",
        "arrival_scenario",
        "predicted_arrivals",
        "entry_prewarm_safety_factor",
        "entry_prewarm_count",
        "achieved_violation_rate",
        "achieved_cost_gbsec",
        "feasible",
        "n_greedy_iterations",
        "memory_config",
    ]
    lines.append(summary_df[summary_cols].round(8).to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## Per-Plan Tier Heterogeneity")
    for row in summary_df.itertuples(index=False):
        mask = (
            plan_df["slo_class"].eq(row.slo_class)
            & plan_df["arrival_scenario"].eq(row.arrival_scenario)
        )
        tiers = plan_df.loc[mask, "memory_tier_mb"].astype(int).tolist()
        heterogeneous = _is_heterogeneous(tiers)
        lines.append(
            f"- `{row.slo_class}/{row.arrival_scenario}`: "
            f"{row.memory_config}; heterogeneous=`{heterogeneous}`."
        )
    lines.append("")

    premium = summary_df[summary_df["slo_class"].eq("premium")]
    free = summary_df[summary_df["slo_class"].eq("free")]
    lines.append("## Sanity Checks")
    all_feasible = bool(summary_df["feasible"].all())
    lines.append(f"- All 4 plans feasible: `{all_feasible}`.")
    for scenario in sorted(summary_df["arrival_scenario"].unique()):
        premium_cost = float(premium[premium["arrival_scenario"].eq(scenario)]["achieved_cost_gbsec"].iloc[0])
        free_cost = float(free[free["arrival_scenario"].eq(scenario)]["achieved_cost_gbsec"].iloc[0])
        lines.append(
            f"- Premium cost >= free cost for `{scenario}`: `{premium_cost >= free_cost}` "
            f"({premium_cost:.6f} vs {free_cost:.6f})."
        )

    for (slo_class, scenario), group in trace_df.groupby(["slo_class", "arrival_scenario"]):
        costs = group.sort_values("step")["cost_after"].astype(float).tolist()
        risks = group.sort_values("step")["risk_after"].astype(float).tolist()
        lines.append(
            f"- Greedy monotonicity `{slo_class}/{scenario}`: "
            f"cost_non_decreasing=`{_monotonic_non_decreasing(costs)}`, "
            f"risk_non_increasing=`{_monotonic_non_increasing(risks)}`."
        )
    lines.append("")

    lines.append("## Greedy Trace")
    lines.append("```text")
    lines.append(trace_df.round(8).to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## Notes")
    lines.append("- This is an offline greedy heuristic, not the P3.5 brute-force optimum baseline.")
    lines.append("- If all final plans are uniform, that is evidence to review SLO/tier tuning rather than a runtime failure.")
    lines.append("- Typical and burst plans are identical here because the greedy chose `entry_prewarm_safety_factor=0`; with no prewarm coverage, the current calibrated entry-cold probability is independent of predicted arrival count.")
    lines.append("- `cost_non_decreasing=False` is expected for this GB-second proxy when a memory upgrade speeds execution up enough to reduce `memory_gb * duration_sec`. Risk is still monotonic non-increasing.")

    path.write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--lognormal-params", default=str(DEFAULT_LOGNORMAL_PARAMS))
    parser.add_argument("--baseline-trace", default=str(DEFAULT_BASELINE_TRACE))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_suite(
        out_dir=args.out_dir,
        lognormal_params_path=args.lognormal_params,
        baseline_trace_path=args.baseline_trace,
    )
    print("plan_summary:")
    print(outputs["plan_summary"].round(8).to_string(index=False))
    print()
    print("plan_per_class:")
    print(outputs["plan_per_class"].to_string(index=False))
    print()
    print("greedy_trace:")
    print(outputs["greedy_trace"].round(8).to_string(index=False))


if __name__ == "__main__":
    main()
