#!/usr/bin/env python3
"""Offline deterministic demo for runtime UP-only dynamic upgrades.

The demo perturbs measured completion times inside the civic_alert DAG and asks
the runtime risk-price upgrader how it reacts for the remaining pending stages.
It does not contact OpenWhisk or Kubernetes.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

from runner.stage4_risk.scaling import memory_to_cpu_cores, spline_predict_warm_mean
from runner.stage5_control.multi_slo_planner import (
    DEFAULT_SAFETY_FACTORS,
    DEFAULT_TIERS,
    STAGES,
    PlannerConfig,
    _dynamic_conditional_risk,
    dynamic_upgrade,
    load_reference_data,
)
from runner.stage5_control.risk_price_planner.suite import (
    EvalContext,
    key_to_memory,
    risk_price_plan,
)
from runner.workflow import WorkflowSpec, load_workflow


RHO = 0.67
CONTENTION_FACTOR = 1.10


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


def ancestors_of(workflow: WorkflowSpec, stage_name: str) -> set[str]:
    out: set[str] = set()
    stack = list(workflow.nodes[stage_name].parents)
    while stack:
        parent = stack.pop()
        if parent in out:
            continue
        out.add(parent)
        stack.extend(workflow.nodes[parent].parents)
    return out


def sink_stages(workflow: WorkflowSpec) -> set[str]:
    return {stage for stage in workflow.nodes if not workflow.children_of(stage)}


def apply_changes(plan: dict[str, int], changes: dict[str, int] | None) -> dict[str, int]:
    out = dict(plan)
    if changes:
        out.update({stage: int(memory) for stage, memory in changes.items()})
    return out


def format_plan(plan: dict[str, int], stages: Iterable[str] = STAGES) -> str:
    return ",".join(f"{stage}:{int(plan[stage])}" for stage in stages)


def format_changes(changes: dict[str, int] | None) -> str:
    if not changes:
        return "none"
    return ",".join(f"{stage}->{int(memory)}" for stage, memory in changes.items())


def predicted_timing_table(
    *,
    workflow: WorkflowSpec,
    topo_order: list[str],
    plan: dict[str, int],
    ref_data,
    contention_factor: float,
) -> list[dict[str, float | str | int]]:
    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}
    rows: list[dict[str, float | str | int]] = []
    for stage_name in topo_order:
        node = workflow.nodes[stage_name]
        start = max((finishes[parent] for parent in node.parents), default=0.0)
        tier = int(plan[stage_name])
        cpu = memory_to_cpu_cores(tier)
        warm_ms = (
            spline_predict_warm_mean(stage_name, cpu, ref_data.warm_splines)
            * contention_factor
        )
        finish = start + warm_ms
        starts[stage_name] = float(start)
        finishes[stage_name] = float(finish)
        rows.append(
            {
                "stage_name": stage_name,
                "parents": ",".join(node.parents),
                "tier_mb": tier,
                "cpu_cores": float(cpu),
                "pred_warm_ms": float(warm_ms),
                "pred_start_ms": float(start),
                "pred_finish_ms": float(finish),
            }
        )
    return rows


def completed_for_perturbation(
    *,
    workflow: WorkflowSpec,
    stage_name: str,
    delay_ms: float,
    predicted_finish: dict[str, float],
) -> dict[str, float]:
    completed_stages = ancestors_of(workflow, stage_name) | {stage_name}
    completed: dict[str, float] = {}
    for completed_stage in completed_stages:
        value = float(predicted_finish[completed_stage])
        if completed_stage == stage_name:
            value += float(delay_ms)
        completed[completed_stage] = value
    return completed


def evaluate_scenario(
    *,
    workflow: WorkflowSpec,
    topo_order: list[str],
    config: PlannerConfig,
    ref_data,
    baseline_plan: dict[str, int],
    predicted_start: dict[str, float],
    predicted_finish: dict[str, float],
    stage_name: str,
    delay_ms: float,
) -> dict[str, object]:
    completed = completed_for_perturbation(
        workflow=workflow,
        stage_name=stage_name,
        delay_ms=delay_ms,
        predicted_finish=predicted_finish,
    )
    pending = [stage for stage in topo_order if stage not in completed]
    now_ms = float(completed[stage_name])

    r0 = _dynamic_conditional_risk(
        config=config,
        ref_data=ref_data,
        workflow=workflow,
        memory_tier_per_stage=baseline_plan,
        completed_finish_ms=completed,
        rho=RHO,
        contention_factor=CONTENTION_FACTOR,
    )
    changes = dynamic_upgrade(
        config=config,
        ref_data=ref_data,
        workflow=workflow,
        current_tiers=baseline_plan,
        completed_finish_ms=completed,
        pending_stages=pending,
        now_ms_since_workflow_start=now_ms,
        predicted_start_ms_by_stage=predicted_start,
        predicted_completion_ms_by_stage=predicted_finish,
        rho=RHO,
        contention_factor=CONTENTION_FACTOR,
    )
    upgraded_plan = apply_changes(baseline_plan, changes)
    r1 = _dynamic_conditional_risk(
        config=config,
        ref_data=ref_data,
        workflow=workflow,
        memory_tier_per_stage=upgraded_plan,
        completed_finish_ms=completed,
        cold_upgrade_stages=set(changes or {}),
        rho=RHO,
        contention_factor=CONTENTION_FACTOR,
    )
    return {
        "perturbed_stage": stage_name,
        "delay_ms": float(delay_ms),
        "now_ms": now_ms,
        "completed": ",".join(stage for stage in topo_order if stage in completed),
        "pending": ",".join(pending),
        "r0": float(r0),
        "changes": format_changes(changes),
        "r1_cold_upgrade": float(r1),
        "recovered": bool(r1 <= config.max_violation_rate + 1e-12),
    }


def write_result_markdown(
    *,
    path: Path,
    dag_rows: list[dict[str, object]],
    baseline_plan: dict[str, int],
    prediction_rows: list[dict[str, object]],
    scenario_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Dynamic Perturbation Demo")
    lines.append("")
    lines.append("## DAG")
    lines.append("```text")
    lines.append(pd.DataFrame(dag_rows).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Baseline Premium Plan")
    lines.append("```text")
    lines.append(format_plan(baseline_plan))
    lines.append("```")
    lines.append("")
    lines.append("## Predicted Schedule")
    lines.append("```text")
    lines.append(pd.DataFrame(prediction_rows).round(3).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## One Perturbation Per Non-Sink Stage")
    lines.append("Delay is 3000ms for this table.")
    lines.append("```text")
    lines.append(scenario_df.round(6).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Estimate Pose Delay Sweep")
    lines.append("```text")
    lines.append(sweep_df.round(6).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "- `r0` is the conditional SLO-violation risk after substituting the measured completion time."
    )
    lines.append(
        "- `changes` is the UP-only runtime tier change chosen by `risk_price_dynamic` for pending stages."
    )
    lines.append(
        "- `r1_cold_upgrade` re-evaluates the changed plan with upgraded stages charged as cold-like, showing the conservative decision-a cost."
    )
    lines.append(
        "- Immediate children often cannot be upgraded if there is not enough predicted JIT lead time; later descendants are the stages dynamic can still rescue."
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="reports/dynamic_perturbation_demo")
    parser.add_argument("--workflow", default="configs/civic_alert_flow.yaml")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    workflow = load_workflow(args.workflow)
    ref_data = load_reference_data()
    topo_order = topological_stage_names(workflow)
    dag_rows = [
        {
            "stage_name": stage,
            "parents": ",".join(workflow.nodes[stage].parents) or "-",
            "children": ",".join(workflow.children_of(stage)) or "-",
        }
        for stage in topo_order
    ]
    print("DAG:")
    print(pd.DataFrame(dag_rows).to_string(index=False), flush=True)
    print(f"topological_order={topo_order}", flush=True)

    config = PlannerConfig(
        slo_ms=18000.0,
        max_violation_rate=0.05,
        predicted_arrivals=5.0,
        tiers=DEFAULT_TIERS,
        safety_factors=DEFAULT_SAFETY_FACTORS,
        stages=STAGES,
    )
    ctx = EvalContext(config, ref_data, rho=RHO, contention_factor=CONTENTION_FACTOR, eval_cache={})
    baseline_result = risk_price_plan(ctx, pairwise=True)
    baseline_plan = key_to_memory(baseline_result.state_key, config)
    print("\nbaseline_plan:")
    print(format_plan(baseline_plan), flush=True)
    print(
        f"baseline_offline_violation={baseline_result.evaluation.violation_rate:.6f} "
        f"cost_gbsec={baseline_result.evaluation.cost_gbsec:.6f}",
        flush=True,
    )

    prediction_rows = predicted_timing_table(
        workflow=workflow,
        topo_order=topo_order,
        plan=baseline_plan,
        ref_data=ref_data,
        contention_factor=CONTENTION_FACTOR,
    )
    prediction_df = pd.DataFrame(prediction_rows)
    predicted_start = dict(zip(prediction_df["stage_name"], prediction_df["pred_start_ms"]))
    predicted_finish = dict(zip(prediction_df["stage_name"], prediction_df["pred_finish_ms"]))
    print("\npredicted_schedule:")
    print(prediction_df.round(3).to_string(index=False), flush=True)

    non_sink = [stage for stage in topo_order if stage not in sink_stages(workflow)]
    scenario_rows = [
        evaluate_scenario(
            workflow=workflow,
            topo_order=topo_order,
            config=config,
            ref_data=ref_data,
            baseline_plan=baseline_plan,
            predicted_start=predicted_start,
            predicted_finish=predicted_finish,
            stage_name=stage,
            delay_ms=3000.0,
        )
        for stage in non_sink
    ]
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(out_dir / "one_perturbation_per_stage.csv", index=False)
    print("\none_perturbation_per_stage:")
    print(scenario_df.round(6).to_string(index=False), flush=True)

    sweep_rows = [
        evaluate_scenario(
            workflow=workflow,
            topo_order=topo_order,
            config=config,
            ref_data=ref_data,
            baseline_plan=baseline_plan,
            predicted_start=predicted_start,
            predicted_finish=predicted_finish,
            stage_name="estimate_pose",
            delay_ms=delay_ms,
        )
        for delay_ms in [0, 500, 1000, 2000, 3000, 5000]
    ]
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(out_dir / "perturbation_sweep.csv", index=False)
    print("\nperturbation_sweep:")
    print(sweep_df.round(6).to_string(index=False), flush=True)

    write_result_markdown(
        path=out_dir / "result.md",
        dag_rows=dag_rows,
        baseline_plan=baseline_plan,
        prediction_rows=prediction_rows,
        scenario_df=scenario_df,
        sweep_df=sweep_df,
    )


if __name__ == "__main__":
    main()
