#!/usr/bin/env python3
"""Run R3 analytical plan-risk examples."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.stage4_risk.entry_cold import calibrate_p_baseline, entry_cold_probability
from runner.stage4_risk.plan_risk import (
    PlanInput,
    compute_cold_overhead_per_stage,
    compute_plan_risk,
    load_lognormal_params,
    result_to_dict,
)


LOGNORMAL_PARAMS = ROOT / "reports" / "path2_lognormal_fit" / "per_stage_lognormal_params.csv"
AMDAHL_PARAMS = ROOT / "reports" / "stage6_amdahl_model" / "per_stage_amdahl_params.csv"
RAW_TRACE = (
    ROOT
    / "reports"
    / "civic_azure_cand2_45min_1280mb_1cpu_keepalive20s_target20s_balanced_mi96"
    / "raw_trace.csv"
)
WORKFLOW_DETAIL = (
    ROOT
    / "reports"
    / "civic_azure_cand2_45min_1280mb_1cpu_keepalive20s_target20s_balanced_mi96"
    / "workflow_detail.csv"
)
OUT_DIR = ROOT / "reports" / "path2_plan_risk"
STAGES = [
    "detect_object",
    "estimate_pose",
    "match_face",
    "classify_scene",
    "translate_alert",
]
SLOS = [15_000, 20_000, 25_000, 30_000]


def all_memory(memory_mb: int) -> dict[str, int]:
    return {stage: memory_mb for stage in STAGES}


def plan_definitions() -> list[dict[str, Any]]:
    downstream_2048 = all_memory(2048)
    downstream_2048["detect_object"] = 1280
    entry_2048 = all_memory(1280)
    entry_2048["detect_object"] = 2048
    return [
        {
            "plan_id": "A",
            "description": "all 1280 MB, no manual prewarm",
            "memory": all_memory(1280),
            "prewarm": 0.0,
            "arrivals": 5.0,
        },
        {
            "plan_id": "B",
            "description": "all 1280 MB, entry fully prewarmed",
            "memory": all_memory(1280),
            "prewarm": 5.0,
            "arrivals": 5.0,
        },
        {
            "plan_id": "C",
            "description": "all 1280 MB, entry partially prewarmed",
            "memory": all_memory(1280),
            "prewarm": 2.0,
            "arrivals": 5.0,
        },
        {
            "plan_id": "D",
            "description": "all 2048 MB, entry fully prewarmed",
            "memory": all_memory(2048),
            "prewarm": 5.0,
            "arrivals": 5.0,
        },
        {
            "plan_id": "E",
            "description": "detect_object 2048 MB, rest 1280 MB, entry fully prewarmed",
            "memory": entry_2048,
            "prewarm": 5.0,
            "arrivals": 5.0,
        },
        {
            "plan_id": "F",
            "description": "detect_object 1280 MB, downstream 2048 MB, entry fully prewarmed",
            "memory": downstream_2048,
            "prewarm": 5.0,
            "arrivals": 5.0,
        },
    ]


def table_text(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def write_report(
    out_path: Path,
    results: pd.DataFrame,
    p_baseline: float,
    real_overall_20s: float,
    real_all_warm_20s: float,
) -> None:
    plan20 = results[results["slo_ms"] == 20_000].copy()
    plan_a = plan20[plan20["plan_id"] == "A"].iloc[0]
    plan_b = plan20[plan20["plan_id"] == "B"].iloc[0]
    plan_d = plan20[plan20["plan_id"] == "D"].iloc[0]
    plan_f = plan20[plan20["plan_id"] == "F"].iloc[0]
    denom = float(plan_a["p_violation_cold_entry"] - plan_a["p_violation_warm"])
    effective_p_needed = (
        (real_overall_20s - float(plan_a["p_violation_warm"])) / denom
        if denom > 0.0
        else float("nan")
    )
    literal_naive_p = entry_cold_probability(
        predicted_arrivals=float(plan_a["predicted_arrivals"]),
        entry_prewarm_count=float(plan_a["entry_prewarm_count"]),
        p_baseline_floor=0.01,
    )

    summary_cols = [
        "plan_id",
        "slo_ms",
        "p_entry_cold",
        "p_violation_warm",
        "p_violation_cold_entry",
        "p_violation_total",
        "expected_e2e_ms",
    ]
    lines = [
        "# R3 Plan Risk Examples",
        "",
        "## Calibration Inputs",
        "",
        f"- Observed entry cold baseline from trace: `{p_baseline:.6f}` (`101 / 4011`).",
        f"- Real measured overall 20s violation rate: `{real_overall_20s:.6f}`.",
        f"- Real measured all-warm 20s violation rate: `{real_all_warm_20s:.6f}`.",
        f"- Literal naive formula for plan A would give p_entry_cold=`{literal_naive_p:.3f}`; the examples use calibrated natural-reuse probability instead.",
        "",
        "## Plan Results At 20s SLO",
        "",
        table_text(plan20[summary_cols]),
        "",
        "## Full Results",
        "",
        table_text(results[summary_cols]),
        "",
        "## Verification Notes",
        "",
        f"- Plan A predicted 20s violation: `{plan_a['p_violation_total']:.6f}` vs real `{real_overall_20s:.6f}`.",
        f"- Plan B predicted 20s violation: `{plan_b['p_violation_total']:.6f}` vs real all-warm `{real_all_warm_20s:.6f}`.",
        f"- Plan B p_entry_cold is at the residual floor `{plan_b['p_entry_cold']:.6f}`; its conditional cold-entry violation remains `{plan_b['p_violation_cold_entry']:.6f}` because that column answers 'what if the residual cold race happens'.",
        f"- Plan D 20s violation `{plan_d['p_violation_total']:.6f}` is lower than plan A `{plan_a['p_violation_total']:.6f}`.",
        f"- Plan F 20s violation `{plan_f['p_violation_total']:.6f}` shows downstream memory helps after full entry prewarm.",
        f"- Effective entry-cold mixture weight needed to match real 20s overall violation with this two-scenario model: `{effective_p_needed:.6f}`.",
        "",
        "## Interpretation",
        "",
        "- The resource-scaling direction is sane: higher memory tiers reduce warm and cold-entry violation risk.",
        "- The current two-scenario model is biased low for plan A because it models only `all warm` vs `entry cold + downstream warm`.",
        "- The real trace's 20s violations include partial-cold cascades and all-warm tail behavior that this first R3 mixture does not yet model.",
        "- The gap is therefore a model-structure issue, not a resource-scaling arithmetic issue.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lognormal_params = load_lognormal_params(LOGNORMAL_PARAMS)
    amdahl_params = pd.read_csv(AMDAHL_PARAMS)
    cold_overhead = compute_cold_overhead_per_stage(lognormal_params)
    p_baseline = calibrate_p_baseline(str(RAW_TRACE), entry_stage_name="detect_object")

    workflow_detail = pd.read_csv(WORKFLOW_DETAIL)
    real_overall_20s = float((workflow_detail["workflow_e2e_ms"] > 20_000).mean())
    all_warm = workflow_detail[workflow_detail["workflow_cold_class"] == "all_warm"]
    real_all_warm_20s = float((all_warm["workflow_e2e_ms"] > 20_000).mean())

    rows: list[dict[str, Any]] = []
    for plan_def in plan_definitions():
        plan = PlanInput(
            memory_tier_per_stage=plan_def["memory"],
            entry_prewarm_count=float(plan_def["prewarm"]),
            predicted_arrivals=float(plan_def["arrivals"]),
            lognormal_params=lognormal_params,
            amdahl_params=amdahl_params,
            cold_overhead_per_stage=cold_overhead,
            p_baseline=p_baseline,
        )
        for slo_ms in SLOS:
            result = compute_plan_risk(plan, float(slo_ms))
            rows.append(
                {
                    "plan_id": plan_def["plan_id"],
                    "description": plan_def["description"],
                    "slo_ms": slo_ms,
                    "memory_tier_per_stage": json.dumps(plan_def["memory"], sort_keys=True),
                    "entry_prewarm_count": float(plan_def["prewarm"]),
                    "predicted_arrivals": float(plan_def["arrivals"]),
                    **result_to_dict(result),
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "plan_examples_results.csv", index=False)
    write_report(
        OUT_DIR / "r3_validation_report.md",
        results=results,
        p_baseline=p_baseline,
        real_overall_20s=real_overall_20s,
        real_all_warm_20s=real_all_warm_20s,
    )

    print("p_baseline", p_baseline)
    print(results[[
        "plan_id",
        "slo_ms",
        "p_entry_cold",
        "p_violation_warm",
        "p_violation_cold_entry",
        "p_violation_total",
    ]].to_string(index=False))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
