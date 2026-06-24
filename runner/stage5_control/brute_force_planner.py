#!/usr/bin/env python3
"""Brute-force optimality baseline for the P3.4/P3.5 greedy planner."""

from __future__ import annotations

import argparse
import itertools
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from runner.stage4_risk.plan_risk import PlanInput, compute_plan_risk
from runner.stage5_control.multi_slo_planner import (
    DEFAULT_BASELINE_TRACE,
    DEFAULT_LOGNORMAL_PARAMS,
    DEFAULT_SAFETY_FACTORS,
    DEFAULT_TIERS,
    STAGES,
    ReferenceData,
    entry_prewarm_count,
    load_reference_data,
    plan_cost_gbsec,
)


DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "path3_brute_force"
DEFAULT_GREEDY_SUMMARY = Path(__file__).resolve().parents[2] / "reports" / "path3_planner" / "plan_summary.csv"
MAX_FULL_SECONDS_PER_CLASS = 30.0 * 60.0
TIMING_SUBSET_SIZE = 1000
RANDOM_SEED = 20260529


@dataclass(frozen=True)
class SearchDecision:
    subset_size: int
    wall_time_sec: float
    estimated_full_time_sec: float
    decision: str
    full_space_size: int
    evaluated_space_size: int


@dataclass(frozen=True)
class PlanRecord:
    memory_tier_per_stage: dict[str, int]
    safety_factor: float
    violation_rate: float
    cost_gbsec: float


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def format_memory_config(memory_tier_per_stage: dict[str, int], stages: list[str] = STAGES) -> str:
    return ",".join(f"{stage}:{int(memory_tier_per_stage[stage])}" for stage in stages)


def parse_memory_config(config: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in str(config).split(","):
        stage, value = part.split(":", 1)
        out[stage] = int(value)
    return out


def evaluate_candidate(
    *,
    memory_tier_per_stage: dict[str, int],
    safety_factor: float,
    slo_ms: float,
    predicted_arrivals: float,
    ref_data: ReferenceData,
    rho: float = 0.0,
    contention_factor: float = 1.0,
) -> PlanRecord:
    """Evaluate one brute-force candidate using the same cost helper as greedy."""

    prewarm_count = entry_prewarm_count(safety_factor, predicted_arrivals)
    plan = PlanInput(
        memory_tier_per_stage=dict(memory_tier_per_stage),
        entry_prewarm_count=float(prewarm_count),
        predicted_arrivals=float(predicted_arrivals),
        lognormal_params=ref_data.lognormal_params,
        amdahl_params=ref_data.amdahl_params,
        cold_overhead_per_stage=ref_data.cold_overhead_per_stage,
        p_baseline=ref_data.p_baseline,
    )
    risk = compute_plan_risk(plan, slo_ms=slo_ms, rho=rho, contention_factor=contention_factor)
    cost = plan_cost_gbsec(
        memory_tier_per_stage=memory_tier_per_stage,
        entry_prewarm_count_value=prewarm_count,
        warm_splines=ref_data.warm_splines,
        stages=STAGES,
    )
    return PlanRecord(
        memory_tier_per_stage=dict(memory_tier_per_stage),
        safety_factor=float(safety_factor),
        violation_rate=float(risk.p_violation_total),
        cost_gbsec=float(cost),
    )


def _random_memory_config(rng: random.Random, tiers: list[int]) -> dict[str, int]:
    return {stage: int(rng.choice(tiers)) for stage in STAGES}


def timing_trial(
    *,
    ref_data: ReferenceData,
    subset_size: int = TIMING_SUBSET_SIZE,
    tiers: list[int] = DEFAULT_TIERS,
    safety_factors: list[float] = DEFAULT_SAFETY_FACTORS,
) -> SearchDecision:
    """Evaluate random plans and decide whether full enumeration is affordable."""

    rng = random.Random(RANDOM_SEED)
    start = time.perf_counter()
    for _ in range(subset_size):
        evaluate_candidate(
            memory_tier_per_stage=_random_memory_config(rng, tiers),
            safety_factor=float(rng.choice(safety_factors)),
            slo_ms=15000.0,
            predicted_arrivals=5.0,
            ref_data=ref_data,
        )
    wall = time.perf_counter() - start
    full_space = (len(tiers) ** len(STAGES)) * len(safety_factors)
    estimated_full = wall * (full_space / float(subset_size))
    if estimated_full <= MAX_FULL_SECONDS_PER_CLASS:
        decision = "full"
        evaluated_space = full_space
    else:
        decision = "safety_zero_grid"
        evaluated_space = len(tiers) ** len(STAGES)
    return SearchDecision(
        subset_size=subset_size,
        wall_time_sec=float(wall),
        estimated_full_time_sec=float(estimated_full),
        decision=decision,
        full_space_size=full_space,
        evaluated_space_size=evaluated_space,
    )


def _candidate_iter(
    *,
    tiers: list[int],
    safety_factors: list[float],
    decision: str,
) -> Iterable[tuple[dict[str, int], float]]:
    active_safety_factors = safety_factors if decision == "full" else [0.0]
    for tier_tuple in itertools.product(tiers, repeat=len(STAGES)):
        memory = {stage: int(tier) for stage, tier in zip(STAGES, tier_tuple)}
        for safety_factor in active_safety_factors:
            yield memory, float(safety_factor)


def brute_force_one_slo(
    *,
    slo_class: str,
    slo_ms: float,
    predicted_arrivals: float,
    ref_data: ReferenceData,
    decision: SearchDecision,
    tiers: list[int] = DEFAULT_TIERS,
    safety_factors: list[float] = DEFAULT_SAFETY_FACTORS,
    rho: float = 0.0,
    contention_factor: float = 1.0,
) -> dict[str, Any]:
    """Enumerate the chosen search space and return the best feasible plan."""

    best: PlanRecord | None = None
    n_total = 0
    n_feasible = 0
    start = time.perf_counter()
    for memory, safety_factor in _candidate_iter(
        tiers=tiers,
        safety_factors=safety_factors,
        decision=decision.decision,
    ):
        n_total += 1
        record = evaluate_candidate(
            memory_tier_per_stage=memory,
            safety_factor=safety_factor,
            slo_ms=slo_ms,
            predicted_arrivals=predicted_arrivals,
            ref_data=ref_data,
            rho=rho,
            contention_factor=contention_factor,
        )
        if record.violation_rate > 0.05:
            continue
        n_feasible += 1
        if best is None:
            best = record
            continue
        if record.cost_gbsec < best.cost_gbsec - 1e-12:
            best = record
        elif abs(record.cost_gbsec - best.cost_gbsec) <= 1e-12:
            current_key = (record.safety_factor, format_memory_config(record.memory_tier_per_stage))
            best_key = (best.safety_factor, format_memory_config(best.memory_tier_per_stage))
            if current_key < best_key:
                best = record

    wall = time.perf_counter() - start
    if best is None:
        return {
            "slo_class": slo_class,
            "slo_ms": slo_ms,
            "optimal_cost_gbsec": math.nan,
            "optimal_violation_rate": math.nan,
            "n_feasible_plans": n_feasible,
            "n_total_evaluated": n_total,
            "optimal_memory_config": "",
            "optimal_safety_factor": math.nan,
            "search_wall_time_sec": wall,
            "search_decision": decision.decision,
        }
    return {
        "slo_class": slo_class,
        "slo_ms": slo_ms,
        "optimal_cost_gbsec": best.cost_gbsec,
        "optimal_violation_rate": best.violation_rate,
        "n_feasible_plans": n_feasible,
        "n_total_evaluated": n_total,
        "optimal_memory_config": format_memory_config(best.memory_tier_per_stage),
        "optimal_safety_factor": best.safety_factor,
        "search_wall_time_sec": wall,
        "search_decision": decision.decision,
    }


def _load_greedy_typical(greedy_summary_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(_resolve(greedy_summary_path))
    required = {
        "slo_class",
        "arrival_scenario",
        "achieved_cost_gbsec",
        "achieved_violation_rate",
        "feasible",
        "memory_config",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"greedy summary is missing columns: {sorted(missing)}")
    return df[df["arrival_scenario"].eq("typical")].copy()


def compare_greedy_vs_optimal(
    *,
    optimal_df: pd.DataFrame,
    greedy_summary_path: str | Path = DEFAULT_GREEDY_SUMMARY,
) -> pd.DataFrame:
    greedy = _load_greedy_typical(greedy_summary_path)
    rows: list[dict[str, Any]] = []
    for optimal in optimal_df.itertuples(index=False):
        greedy_row = greedy[greedy["slo_class"].eq(optimal.slo_class)]
        if greedy_row.empty:
            raise ValueError(f"missing greedy result for slo_class={optimal.slo_class}")
        g = greedy_row.iloc[0]
        greedy_cost = float(g["achieved_cost_gbsec"])
        optimal_cost = float(optimal.optimal_cost_gbsec)
        if not math.isfinite(optimal_cost) or optimal_cost <= 0.0:
            gap = math.nan
        else:
            gap = (greedy_cost - optimal_cost) / optimal_cost * 100.0
        greedy_config = str(g["memory_config"])
        optimal_config = str(optimal.optimal_memory_config)
        rows.append(
            {
                "slo_class": optimal.slo_class,
                "greedy_cost": greedy_cost,
                "optimal_cost": optimal_cost,
                "optimality_gap_pct": gap,
                "greedy_config": greedy_config,
                "optimal_config": optimal_config,
                "configs_match": greedy_config == optimal_config,
                "greedy_violation_rate": float(g["achieved_violation_rate"]),
                "optimal_violation_rate": float(optimal.optimal_violation_rate),
                "greedy_feasible": bool(g["feasible"]),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    *,
    path: Path,
    timing_df: pd.DataFrame,
    optimal_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# P3.5 Brute Force Optimality Validation")
    lines.append("")
    lines.append("## Timing Trial")
    lines.append("```text")
    lines.append(timing_df.round(6).to_string(index=False))
    lines.append("```")
    decision = str(timing_df["decision"].iloc[0])
    if decision == "full":
        lines.append("- Decision: full enumeration across all 295,245 plans per SLO class.")
    else:
        lines.append("- Decision: reduced grid with `safety_factor=0` only because estimated full enumeration exceeded the 30-minute gate.")
        lines.append("- This matches P3.4's finding that greedy selected zero prewarm for all scenarios.")
    lines.append("")

    lines.append("## Brute Force Optimum")
    lines.append("```text")
    lines.append(optimal_df.round(8).to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## Greedy vs Optimal")
    lines.append("```text")
    lines.append(comparison_df.round(8).to_string(index=False))
    lines.append("```")
    lines.append("")

    acceptable = bool((comparison_df["optimality_gap_pct"] <= 5.0 + 1e-9).all())
    lines.append("## Verdict")
    lines.append(f"- Greedy optimality gap < 5% for every SLO class: `{acceptable}`.")
    for row in comparison_df.itertuples(index=False):
        gap = float(row.optimality_gap_pct)
        if gap > 10.0:
            lines.append(
                f"- `{row.slo_class}` gap is `{gap:.2f}%`; lookahead or beam search should be considered."
            )
        elif gap > 5.0:
            lines.append(
                f"- `{row.slo_class}` gap is `{gap:.2f}%`; greedy is close but misses the <5% target."
            )
        else:
            lines.append(
                f"- `{row.slo_class}` gap is `{gap:.2f}%`; greedy is acceptable by the <5% criterion."
            )
        if not bool(row.configs_match):
            lines.append(
                f"- `{row.slo_class}` optimal config differs from greedy, so the greedy did choose at least one suboptimal upgrade sequence."
            )
    lines.append("")

    lines.append("## Sanity Checks")
    for row in comparison_df.itertuples(index=False):
        opt_le_greedy = float(row.optimal_cost) <= float(row.greedy_cost) + 1e-9
        both_feasible = (
            bool(row.greedy_feasible)
            and float(row.greedy_violation_rate) <= 0.05 + 1e-12
            and float(row.optimal_violation_rate) <= 0.05 + 1e-12
        )
        lines.append(
            f"- `{row.slo_class}` optimal_cost <= greedy_cost: `{opt_le_greedy}`; both feasible: `{both_feasible}`."
        )
    lines.append("- Burst was not enumerated because P3.4 deterministic greedy produced identical typical/burst plans; typical arrivals=5 is representative for this validation.")

    path.write_text("\n".join(lines) + "\n")


def run_brute_force_suite(
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    lognormal_params_path: str | Path = DEFAULT_LOGNORMAL_PARAMS,
    baseline_trace_path: str | Path = DEFAULT_BASELINE_TRACE,
    greedy_summary_path: str | Path = DEFAULT_GREEDY_SUMMARY,
    rho: float = 0.0,
    contention_factor: float = 1.0,
) -> dict[str, pd.DataFrame]:
    out = _resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref_data = load_reference_data(
        lognormal_params_path=lognormal_params_path,
        baseline_trace_path=baseline_trace_path,
    )

    decision = timing_trial(ref_data=ref_data)
    timing_df = pd.DataFrame(
        [
            {
                "subset_size": decision.subset_size,
                "wall_time_sec": decision.wall_time_sec,
                "estimated_full_time_sec": decision.estimated_full_time_sec,
                "decision": decision.decision,
                "full_space_size": decision.full_space_size,
                "evaluated_space_size": decision.evaluated_space_size,
            }
        ]
    )
    timing_df.to_csv(out / "timing_trial.csv", index=False)
    print("timing_trial:")
    print(timing_df.round(6).to_string(index=False), flush=True)

    optimal_rows = []
    for slo_class, slo_ms in [("premium", 15000.0), ("free", 20000.0)]:
        print(f"enumerating {slo_class} slo={slo_ms} decision={decision.decision}", flush=True)
        optimal_rows.append(
            brute_force_one_slo(
                slo_class=slo_class,
                slo_ms=slo_ms,
                predicted_arrivals=5.0,
                ref_data=ref_data,
                decision=decision,
                rho=rho,
                contention_factor=contention_factor,
            )
        )
    optimal_df = pd.DataFrame(optimal_rows)
    optimal_df.to_csv(out / "brute_force_optimal.csv", index=False)

    comparison_df = compare_greedy_vs_optimal(
        optimal_df=optimal_df,
        greedy_summary_path=greedy_summary_path,
    )
    comparison_df.to_csv(out / "greedy_vs_optimal.csv", index=False)
    _write_report(
        path=out / "comparison_report.md",
        timing_df=timing_df,
        optimal_df=optimal_df,
        comparison_df=comparison_df,
    )

    return {
        "timing_trial": timing_df,
        "brute_force_optimal": optimal_df,
        "greedy_vs_optimal": comparison_df,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--lognormal-params", default=str(DEFAULT_LOGNORMAL_PARAMS))
    parser.add_argument("--baseline-trace", default=str(DEFAULT_BASELINE_TRACE))
    parser.add_argument("--greedy-summary", default=str(DEFAULT_GREEDY_SUMMARY))
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--contention-factor", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_brute_force_suite(
        out_dir=args.out_dir,
        lognormal_params_path=args.lognormal_params,
        baseline_trace_path=args.baseline_trace,
        greedy_summary_path=args.greedy_summary,
        rho=args.rho,
        contention_factor=args.contention_factor,
    )
    print()
    print("brute_force_optimal:")
    print(outputs["brute_force_optimal"].round(8).to_string(index=False))
    print()
    print("greedy_vs_optimal:")
    print(outputs["greedy_vs_optimal"].round(8).to_string(index=False))


if __name__ == "__main__":
    main()
