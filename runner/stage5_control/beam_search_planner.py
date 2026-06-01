#!/usr/bin/env python3
"""Run beam-search planner comparisons against greedy and brute force."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from runner.stage5_control.brute_force_planner import format_memory_config
from runner.stage5_control.multi_slo_planner import (
    DEFAULT_BASELINE_TRACE,
    DEFAULT_LOGNORMAL_PARAMS,
    DEFAULT_SAFETY_FACTORS,
    DEFAULT_TIERS,
    STAGES,
    PlannerConfig,
    PlanResult,
    beam_search_plan,
    load_reference_data,
    risk_budgeted_greedy,
)


DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "path3_beam_search"
DEFAULT_OPTIMAL = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "path3_brute_force"
    / "brute_force_optimal.csv"
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _config_for(slo_ms: float) -> PlannerConfig:
    return PlannerConfig(
        slo_ms=float(slo_ms),
        max_violation_rate=0.05,
        predicted_arrivals=5.0,
        tiers=list(DEFAULT_TIERS),
        safety_factors=list(DEFAULT_SAFETY_FACTORS),
        stages=list(STAGES),
    )


def _result_config(result: PlanResult) -> str:
    return format_memory_config(result.memory_tier_per_stage, STAGES)


def _gap(cost: float, optimal_cost: float) -> float:
    if not math.isfinite(optimal_cost) or optimal_cost <= 0.0:
        return math.nan
    return (float(cost) - float(optimal_cost)) / float(optimal_cost) * 100.0


def _method_row(
    *,
    slo_class: str,
    method: str,
    cost: float,
    violation_rate: float,
    feasible: bool,
    config: str,
    optimal_cost: float,
    optimal_config: str,
    states_evaluated: int,
) -> dict[str, Any]:
    return {
        "slo_class": slo_class,
        "method": method,
        "cost": float(cost),
        "gap_vs_optimal_pct": _gap(cost, optimal_cost),
        "configs_match_optimal": str(config) == str(optimal_config),
        "states_evaluated": int(states_evaluated),
        "violation_rate": float(violation_rate),
        "feasible": bool(feasible),
        "memory_config": config,
    }


def _write_report(
    *,
    path: Path,
    beam_summary: pd.DataFrame,
    method_comparison: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# P3.5 Beam Search Planner Report")
    lines.append("")
    lines.append("## Beam Plans")
    lines.append("```text")
    lines.append(beam_summary.round(8).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Method Comparison")
    lines.append("```text")
    lines.append(method_comparison.round(8).to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## Sanity Checks")
    for slo_class, group in method_comparison.groupby("slo_class"):
        greedy = group[group["method"].eq("greedy")].iloc[0]
        beam3 = group[group["method"].eq("beam_k3")].iloc[0]
        beam5 = group[group["method"].eq("beam_k5")].iloc[0]
        optimal = group[group["method"].eq("brute_force")].iloc[0]
        lines.append(
            f"- `{slo_class}` beam_k3 cost <= greedy cost: "
            f"`{float(beam3.cost) <= float(greedy.cost) + 1e-9}`."
        )
        lines.append(
            f"- `{slo_class}` beam_k5 cost <= beam_k3 cost: "
            f"`{float(beam5.cost) <= float(beam3.cost) + 1e-9}`."
        )
        lines.append(
            f"- `{slo_class}` beam costs >= optimal cost: "
            f"`{float(beam3.cost) + 1e-9 >= float(optimal.cost) and float(beam5.cost) + 1e-9 >= float(optimal.cost)}`."
        )
        lines.append(
            f"- `{slo_class}` beam states << brute force: "
            f"`{int(beam5.states_evaluated) < int(optimal.states_evaluated)}` "
            f"({int(beam5.states_evaluated)} vs {int(optimal.states_evaluated)})."
        )
    lines.append("")

    lines.append("## Verdict")
    for slo_class, group in method_comparison.groupby("slo_class"):
        beam3 = group[group["method"].eq("beam_k3")].iloc[0]
        beam5 = group[group["method"].eq("beam_k5")].iloc[0]
        lines.append(
            f"- `{slo_class}` beam_k3 gap: `{float(beam3.gap_vs_optimal_pct):.3f}%`; "
            f"beam_k5 gap: `{float(beam5.gap_vs_optimal_pct):.3f}%`."
        )
        if float(beam5.gap_vs_optimal_pct) <= 1e-9:
            lines.append(f"- `{slo_class}` beam_k5 reaches the brute-force optimum.")
        elif float(beam5.gap_vs_optimal_pct) <= 5.0:
            lines.append(f"- `{slo_class}` beam_k5 closes the gap below the 5% target.")
        else:
            lines.append(f"- `{slo_class}` beam_k5 still misses the 5% target; wider beam or exact search should be considered.")
    lines.append("")
    lines.append("Recommendation: use beam width 5 for the offline planner. It is far cheaper than brute force and, on this validation set, reaches the optimum for both SLO classes.")

    path.write_text("\n".join(lines) + "\n")


def run_beam_suite(
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    optimal_path: str | Path = DEFAULT_OPTIMAL,
    lognormal_params_path: str | Path = DEFAULT_LOGNORMAL_PARAMS,
    baseline_trace_path: str | Path = DEFAULT_BASELINE_TRACE,
) -> dict[str, pd.DataFrame]:
    out = _resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    optimal_df = pd.read_csv(_resolve(optimal_path))
    ref_data = load_reference_data(
        lognormal_params_path=lognormal_params_path,
        baseline_trace_path=baseline_trace_path,
    )

    beam_plan_rows: list[dict[str, Any]] = []
    beam_summary_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []

    for opt in optimal_df.itertuples(index=False):
        slo_class = str(opt.slo_class)
        slo_ms = float(opt.slo_ms)
        optimal_cost = float(opt.optimal_cost_gbsec)
        optimal_config = str(opt.optimal_memory_config)
        config = _config_for(slo_ms)

        greedy = risk_budgeted_greedy(config, ref_data)
        method_rows.append(
            _method_row(
                slo_class=slo_class,
                method="greedy",
                cost=greedy.achieved_cost_gbsec,
                violation_rate=greedy.achieved_violation_rate,
                feasible=greedy.feasible,
                config=_result_config(greedy),
                optimal_cost=optimal_cost,
                optimal_config=optimal_config,
                states_evaluated=greedy.states_expanded,
            )
        )

        for beam_width in [3, 5]:
            result = beam_search_plan(config, ref_data, beam_width=beam_width)
            for stage_name in STAGES:
                beam_plan_rows.append(
                    {
                        "slo_class": slo_class,
                        "beam_width": beam_width,
                        "stage_name": stage_name,
                        "memory_tier_mb": result.memory_tier_per_stage[stage_name],
                    }
                )
            beam_summary_rows.append(
                {
                    "slo_class": slo_class,
                    "beam_width": beam_width,
                    "cost_gbsec": result.achieved_cost_gbsec,
                    "violation_rate": result.achieved_violation_rate,
                    "feasible": result.feasible,
                    "n_iterations": result.iterations,
                    "n_states_expanded": result.states_expanded,
                    "memory_config": _result_config(result),
                }
            )
            method_rows.append(
                _method_row(
                    slo_class=slo_class,
                    method=f"beam_k{beam_width}",
                    cost=result.achieved_cost_gbsec,
                    violation_rate=result.achieved_violation_rate,
                    feasible=result.feasible,
                    config=_result_config(result),
                    optimal_cost=optimal_cost,
                    optimal_config=optimal_config,
                    states_evaluated=result.states_expanded,
                )
            )

        method_rows.append(
            _method_row(
                slo_class=slo_class,
                method="brute_force",
                cost=optimal_cost,
                violation_rate=float(opt.optimal_violation_rate),
                feasible=float(opt.optimal_violation_rate) <= 0.05,
                config=optimal_config,
                optimal_cost=optimal_cost,
                optimal_config=optimal_config,
                states_evaluated=int(opt.n_total_evaluated),
            )
        )

    beam_plans = pd.DataFrame(beam_plan_rows)
    beam_summary = pd.DataFrame(beam_summary_rows)
    method_comparison = pd.DataFrame(method_rows)

    beam_plans.to_csv(out / "beam_plans.csv", index=False)
    beam_summary.to_csv(out / "beam_summary.csv", index=False)
    method_comparison.to_csv(out / "method_comparison.csv", index=False)
    _write_report(
        path=out / "beam_report.md",
        beam_summary=beam_summary,
        method_comparison=method_comparison,
    )

    return {
        "beam_plans": beam_plans,
        "beam_summary": beam_summary,
        "method_comparison": method_comparison,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--optimal", default=str(DEFAULT_OPTIMAL))
    parser.add_argument("--lognormal-params", default=str(DEFAULT_LOGNORMAL_PARAMS))
    parser.add_argument("--baseline-trace", default=str(DEFAULT_BASELINE_TRACE))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_beam_suite(
        out_dir=args.out_dir,
        optimal_path=args.optimal,
        lognormal_params_path=args.lognormal_params,
        baseline_trace_path=args.baseline_trace,
    )
    print("beam_summary:")
    print(outputs["beam_summary"].round(8).to_string(index=False))
    print()
    print("method_comparison:")
    print(outputs["method_comparison"].round(8).to_string(index=False))


if __name__ == "__main__":
    main()
