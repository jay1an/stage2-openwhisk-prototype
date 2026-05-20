import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..workflow import WorkflowSpec, load_workflow


POLICY_TO_QUANTILE = {
    "p50": 0.50,
    "p90": 0.90,
    "p95": 0.95,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline joint-control planner for prewarming, keep-alive, memory sizing, "
            "and DAG slack-aware stage priority."
        )
    )
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--forecast-detail", required=True)
    parser.add_argument("--latency-samples", required=True)
    parser.add_argument("--method", default="online-adaptive-expert-bank")
    parser.add_argument("--policy", choices=["p50", "p90", "p95"], default="p95")
    parser.add_argument("--slo-ms", type=float, required=True)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--memory-tiers-mb", default="128,256,512,1024,2048")
    parser.add_argument("--base-memory-mb", type=int, default=256)
    parser.add_argument("--cpu-alpha", type=float, default=1.0)
    parser.add_argument("--overhead-alpha", type=float, default=0.08)
    parser.add_argument("--prewarm-safety", type=float, default=1.0)
    parser.add_argument("--min-keepalive-sec", type=float, default=5.0)
    parser.add_argument("--max-keepalive-sec", type=float, default=60.0)
    parser.add_argument("--max-windows", type=int, default=0, help="0 means all windows")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_memory_tiers(value: str) -> list[int]:
    tiers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not tiers:
        raise ValueError("--memory-tiers-mb must contain at least one tier")
    return tiers


def topological_nodes(workflow: WorkflowSpec) -> list[str]:
    remaining = set(workflow.nodes)
    ordered: list[str] = []
    while remaining:
        ready = sorted(
            node
            for node in remaining
            if all(parent in ordered for parent in workflow.nodes[node].parents)
        )
        if not ready:
            raise ValueError("workflow DAG contains a cycle or an unknown parent")
        ordered.extend(ready)
        remaining.difference_update(ready)
    return ordered


def latency_quantiles(samples: pd.DataFrame, workflow_name: str, quantile: float) -> pd.DataFrame:
    rows = samples[(samples["workflow_name"] == workflow_name) & (samples["stage_name"] != "__entry__")].copy()
    for col in ["dispatch_latency_ms", "platform_overhead_ms", "action_duration_ms"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows.dropna(subset=["dispatch_latency_ms", "platform_overhead_ms", "action_duration_ms"])
    if rows.empty:
        raise ValueError(f"no latency samples found for workflow {workflow_name}")

    records = []
    for stage_name, group in rows.groupby("stage_name"):
        warm = group[group["latency_class"].astype(str) == "warm"]
        cold = group[group["latency_class"].astype(str).str.startswith("cold_like")]
        if warm.empty:
            warm = group
        if cold.empty:
            cold = group
        records.append(
            {
                "stage_name": str(stage_name),
                "warm_dispatch_q_ms": float(warm["dispatch_latency_ms"].quantile(quantile)),
                "warm_overhead_q_ms": float(warm["platform_overhead_ms"].quantile(quantile)),
                "warm_action_q_ms": float(warm["action_duration_ms"].quantile(quantile)),
                "cold_dispatch_q_ms": float(cold["dispatch_latency_ms"].quantile(quantile)),
                "cold_overhead_q_ms": float(cold["platform_overhead_ms"].quantile(quantile)),
                "cold_action_q_ms": float(cold["action_duration_ms"].quantile(quantile)),
                "warm_sample_count": int(len(warm)),
                "cold_sample_count": int(len(cold)),
            }
        )
    return pd.DataFrame(records)


def adjusted_latency_ms(
    overhead_ms: float,
    action_ms: float,
    memory_mb: int,
    base_memory_mb: int,
    cpu_alpha: float,
    overhead_alpha: float,
) -> float:
    ratio = max(memory_mb, 1) / max(base_memory_mb, 1)
    adjusted_action = action_ms / (ratio ** cpu_alpha)
    adjusted_overhead = overhead_ms / (ratio ** overhead_alpha)
    return max(1.0, adjusted_overhead + adjusted_action)


def dag_slack(workflow: WorkflowSpec, ordered: list[str], stage_duration: dict[str, float], slo_ms: float) -> pd.DataFrame:
    earliest_start: dict[str, float] = {}
    earliest_finish: dict[str, float] = {}
    for stage in ordered:
        parents = workflow.nodes[stage].parents
        earliest_start[stage] = max((earliest_finish[parent] for parent in parents), default=0.0)
        earliest_finish[stage] = earliest_start[stage] + stage_duration[stage]

    children = {stage: workflow.children_of(stage) for stage in ordered}
    latest_finish: dict[str, float] = {}
    latest_start: dict[str, float] = {}
    for stage in reversed(ordered):
        if not children[stage]:
            latest_finish[stage] = slo_ms
        else:
            latest_finish[stage] = min(latest_start[child] for child in children[stage])
        latest_start[stage] = latest_finish[stage] - stage_duration[stage]

    rows = []
    for stage in ordered:
        slack = latest_start[stage] - earliest_start[stage]
        rows.append(
            {
                "stage_name": stage,
                "earliest_start_ms": earliest_start[stage],
                "earliest_finish_ms": earliest_finish[stage],
                "latest_start_ms": latest_start[stage],
                "latest_finish_ms": latest_finish[stage],
                "stage_budget_ms": latest_finish[stage] - earliest_start[stage],
                "slack_ms": slack,
                "critical_like": slack <= 0,
            }
        )
    return pd.DataFrame(rows)


def choose_memory_plan(
    workflow: WorkflowSpec,
    latency_profile: pd.DataFrame,
    slack_table: pd.DataFrame,
    memory_tiers: list[int],
    base_memory_mb: int,
    cpu_alpha: float,
    overhead_alpha: float,
    policy: str,
) -> pd.DataFrame:
    slack_lookup = slack_table.set_index("stage_name").to_dict("index")
    rows = []
    for _, profile in latency_profile.iterrows():
        stage = str(profile["stage_name"])
        slack = float(slack_lookup[stage]["slack_ms"])
        budget = float(slack_lookup[stage]["stage_budget_ms"])
        candidates = []
        for memory_mb in memory_tiers:
            warm_q = adjusted_latency_ms(
                profile["warm_overhead_q_ms"],
                profile["warm_action_q_ms"],
                memory_mb,
                base_memory_mb,
                cpu_alpha,
                overhead_alpha,
            )
            cold_q = adjusted_latency_ms(
                profile["cold_overhead_q_ms"],
                profile["cold_action_q_ms"],
                memory_mb,
                base_memory_mb,
                cpu_alpha,
                overhead_alpha,
            )
            memory_seconds_per_invocation = (memory_mb / 1024.0) * (warm_q / 1000.0)
            candidates.append(
                {
                    "memory_mb": memory_mb,
                    "estimated_warm_q_ms": warm_q,
                    "estimated_cold_q_ms": cold_q,
                    "memory_seconds_per_warm_invocation": memory_seconds_per_invocation,
                }
            )
        candidate_df = pd.DataFrame(candidates)
        feasible = candidate_df[candidate_df["estimated_warm_q_ms"] <= max(1.0, budget)]
        if feasible.empty:
            # If SLO budget is already impossible, choose the fastest tier.
            chosen = candidate_df.sort_values("estimated_warm_q_ms").iloc[0]
            reason = "fastest-tier-budget-infeasible"
        elif slack <= profile["cold_dispatch_q_ms"] * 0.25:
            # Tight slack stages should avoid cold-path amplification.
            chosen = feasible.sort_values(["estimated_cold_q_ms", "memory_seconds_per_warm_invocation"]).iloc[0]
            reason = "cold-risk-tight-slack"
        else:
            chosen = feasible.sort_values("memory_seconds_per_warm_invocation").iloc[0]
            reason = "lowest-memory-seconds-within-budget"

        rows.append(
            {
                "workflow_name": workflow.workflow_name,
                "stage_name": stage,
                "action": workflow.nodes[stage].action,
                "policy": policy,
                "selected_memory_mb": int(chosen["memory_mb"]),
                "selection_reason": reason,
                "slack_ms": slack,
                "stage_budget_ms": budget,
                "estimated_warm_q_ms": float(chosen["estimated_warm_q_ms"]),
                "estimated_cold_q_ms": float(chosen["estimated_cold_q_ms"]),
                "memory_seconds_per_warm_invocation": float(chosen["memory_seconds_per_warm_invocation"]),
                "warm_sample_count": int(profile["warm_sample_count"]),
                "cold_sample_count": int(profile["cold_sample_count"]),
            }
        )
    return pd.DataFrame(rows)


def normalize_feature(value: float, cap: float) -> float:
    if pd.isna(value) or cap <= 0:
        return 0.0
    return float(min(1.0, max(0.0, value / cap)))


def build_control_plan(
    detail: pd.DataFrame,
    workflow: WorkflowSpec,
    memory_plan: pd.DataFrame,
    policy: str,
    method: str,
    prewarm_safety: float,
    min_keepalive_sec: float,
    max_keepalive_sec: float,
    max_windows: int,
) -> pd.DataFrame:
    selected = detail[
        (detail["workflow_name"] == workflow.workflow_name)
        & (detail["policy"] == policy)
    ].copy()
    if "method" in selected.columns:
        selected = selected[selected["method"] == method].copy()
    if selected.empty:
        raise ValueError(f"no forecast rows found for workflow={workflow.workflow_name}, method={method}, policy={policy}")

    selected["target_window"] = pd.to_numeric(selected["target_window"], errors="coerce").astype("Int64")
    selected = selected.dropna(subset=["target_window"]).copy()
    windows = sorted(int(value) for value in selected["target_window"].unique())
    if max_windows and max_windows > 0:
        windows = windows[:max_windows]
        selected = selected[selected["target_window"].astype(int).isin(windows)].copy()

    memory_lookup = memory_plan.set_index("stage_name").to_dict("index")
    max_alloc = max(1.0, float(pd.to_numeric(selected["allocated_count"], errors="coerce").max()))
    max_slack = max(1.0, float(memory_plan["slack_ms"].clip(lower=0).max()))
    rows = []
    for _, row in selected.iterrows():
        stage = str(row["stage_name"])
        if stage not in memory_lookup:
            continue
        stage_plan = memory_lookup[stage]
        allocated = max(0, int(math.ceil(float(row.get("allocated_count", 0) or 0))))
        forecast = max(0.0, float(row.get("forecast_count", 0) or 0))
        target_prewarm = int(math.ceil(allocated * prewarm_safety)) if allocated > 0 else 0
        slack_ms = float(stage_plan["slack_ms"])
        urgency = 1.0 - normalize_feature(max(0.0, slack_ms), max_slack)
        active_ratio = float(row.get("recent_active_ratio", 0.0) or 0.0)
        alloc_pressure = normalize_feature(allocated, max_alloc)
        if target_prewarm <= 0:
            keepalive_sec = 0.0
            strategy = "allow-scale-to-zero"
        else:
            keepalive_ratio = min(1.0, 0.50 * urgency + 0.30 * active_ratio + 0.20 * alloc_pressure)
            keepalive_sec = min_keepalive_sec + keepalive_ratio * (max_keepalive_sec - min_keepalive_sec)
            strategy = "prewarm-and-keepalive"
        rows.append(
            {
                "workflow_name": workflow.workflow_name,
                "policy": policy,
                "method": method,
                "target_window": int(row["target_window"]),
                "stage_name": stage,
                "action": workflow.nodes[stage].action,
                "forecast_count": forecast,
                "allocated_count": allocated,
                "prewarm_target": target_prewarm,
                "keepalive_sec": round(float(keepalive_sec), 3),
                "selected_memory_mb": int(stage_plan["selected_memory_mb"]),
                "slack_ms": slack_ms,
                "urgency_score": urgency,
                "control_strategy": strategy,
                "source_expert_id": row.get("selected_expert_id", row.get("expert_id", "")),
                "source_method_family": row.get("source_method_family", row.get("method_family", "")),
                "source_calibration_method": row.get("source_calibration_method", row.get("calibration_method", "")),
            }
        )
    plan = pd.DataFrame(rows)
    if plan.empty:
        return plan
    plan["slack_priority_rank"] = (
        plan.sort_values(["target_window", "slack_ms", "allocated_count"], ascending=[True, True, False])
        .groupby("target_window")
        .cumcount()
        + 1
    )
    return plan.sort_values(["target_window", "slack_priority_rank", "stage_name"]).reset_index(drop=True)


def write_command_templates(out_dir: Path, workflow: WorkflowSpec, memory_plan: pd.DataFrame) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Purpose: apply selected OpenWhisk memory tiers.",
        "# Review before running. This file is generated for deployment-time control, not executed by the planner.",
        "",
    ]
    for _, row in memory_plan.iterrows():
        action = row["action"]
        memory = int(row["selected_memory_mb"])
        lines.append(f"# stage={row['stage_name']} reason={row['selection_reason']}")
        lines.append(f"wsk action update {action} actions/sebs_mock.py --kind python:3 --memory {memory} -i")
        lines.append("")
    (out_dir / "apply_memory_plan_template.sh").write_text("\n".join(lines), encoding="utf-8")

    warmup_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Purpose: example warmup invocation for each action. The online controller should repeat this",
        "# until observed warm replicas reach the per-window `prewarm_target` in control_plan.csv.",
        "",
    ]
    for stage, node in workflow.nodes.items():
        warmup_lines.append(f"# stage={stage}")
        warmup_lines.append(
            "wsk action invoke "
            f"{node.action} "
            "-p __warmup true "
            f"-p workflow_name {workflow.workflow_name} "
            f"-p stage_name {stage} "
            "-p request_id warmup-${RANDOM} "
            "--blocking --result -i"
        )
        warmup_lines.append("")
    (out_dir / "warmup_invoke_template.sh").write_text("\n".join(warmup_lines), encoding="utf-8")


def write_readme(out_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Joint Control Plan",
        "",
        "## Scope",
        "",
        "This directory contains an offline control plan for three proposal components:",
        "",
        "- fast cold-start mitigation: `prewarm_target` and `keepalive_sec`",
        "- slow resource sizing: `selected_memory_mb`",
        "- DAG slack-aware scheduling: `slack_ms`, `urgency_score`, and `slack_priority_rank`",
        "",
        "The planner does not invoke OpenWhisk and does not modify action memory. It emits CSVs and command templates.",
        "",
        "## Main Files",
        "",
        "- `control_plan.csv`: per-window, per-stage control decision.",
        "- `stage_resource_plan.csv`: selected memory tier and static slack profile by stage.",
        "- `dag_slack_profile.csv`: earliest/latest times and slack under the chosen SLO.",
        "- `apply_memory_plan_template.sh`: deployment-time memory update template.",
        "- `warmup_invoke_template.sh`: warmup invocation template for a future online warm manager.",
        "",
        "## Important Guardrail",
        "",
        "Current memory-tier latency is estimated by a scaling model unless real per-memory OpenWhisk traces are supplied. "
        "Treat it as a control-plane prototype, then replace the scaling model with real `memory_mb` profiles later.",
        "",
        "## Inputs",
        "",
        f"- Workflow config: `{args.workflow_config}`",
        f"- Forecast detail: `{args.forecast_detail}`",
        f"- Latency samples: `{args.latency_samples}`",
        f"- Method: `{args.method}`",
        f"- Policy: `{args.policy}`",
        f"- SLO: `{args.slo_ms}` ms",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = project_root()
    workflow = load_workflow(str(resolve_path(root, args.workflow_config)))
    ordered = topological_nodes(workflow)
    memory_tiers = parse_memory_tiers(args.memory_tiers_mb)
    quantile = POLICY_TO_QUANTILE[args.policy]

    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latency_samples = pd.read_csv(resolve_path(root, args.latency_samples))
    profile = latency_quantiles(latency_samples, workflow.workflow_name, quantile)
    profile.to_csv(out_dir / "latency_quantile_profile.csv", index=False)

    duration_lookup = dict(zip(profile["stage_name"], profile["warm_dispatch_q_ms"]))
    missing = [stage for stage in ordered if stage not in duration_lookup]
    if missing:
        raise ValueError(f"missing latency profile rows for stages: {missing}")
    slack = dag_slack(workflow, ordered, duration_lookup, args.slo_ms)
    slack.to_csv(out_dir / "dag_slack_profile.csv", index=False)

    memory_plan = choose_memory_plan(
        workflow=workflow,
        latency_profile=profile,
        slack_table=slack,
        memory_tiers=memory_tiers,
        base_memory_mb=args.base_memory_mb,
        cpu_alpha=args.cpu_alpha,
        overhead_alpha=args.overhead_alpha,
        policy=args.policy,
    )
    memory_plan.to_csv(out_dir / "stage_resource_plan.csv", index=False)

    forecast_detail = pd.read_csv(resolve_path(root, args.forecast_detail))
    control_plan = build_control_plan(
        forecast_detail,
        workflow=workflow,
        memory_plan=memory_plan,
        policy=args.policy,
        method=args.method,
        prewarm_safety=args.prewarm_safety,
        min_keepalive_sec=args.min_keepalive_sec,
        max_keepalive_sec=args.max_keepalive_sec,
        max_windows=args.max_windows,
    )
    control_plan.to_csv(out_dir / "control_plan.csv", index=False)

    by_stage = (
        control_plan.groupby("stage_name", as_index=False)
        .agg(
            windows=("target_window", "nunique"),
            mean_prewarm_target=("prewarm_target", "mean"),
            max_prewarm_target=("prewarm_target", "max"),
            mean_keepalive_sec=("keepalive_sec", "mean"),
            selected_memory_mb=("selected_memory_mb", "max"),
            mean_urgency_score=("urgency_score", "mean"),
        )
        if not control_plan.empty
        else pd.DataFrame()
    )
    by_stage.to_csv(out_dir / "control_summary_by_stage.csv", index=False)

    write_command_templates(out_dir, workflow, memory_plan)
    write_readme(out_dir, args)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_config": str(resolve_path(root, args.workflow_config)),
        "forecast_detail": str(resolve_path(root, args.forecast_detail)),
        "latency_samples": str(resolve_path(root, args.latency_samples)),
        "method": args.method,
        "policy": args.policy,
        "slo_ms": args.slo_ms,
        "window_sec": args.window_sec,
        "memory_tiers_mb": memory_tiers,
        "base_memory_mb": args.base_memory_mb,
        "cpu_alpha": args.cpu_alpha,
        "overhead_alpha": args.overhead_alpha,
        "prewarm_safety": args.prewarm_safety,
        "min_keepalive_sec": args.min_keepalive_sec,
        "max_keepalive_sec": args.max_keepalive_sec,
        "notes": [
            "Resource sizing uses a latency scaling model until real per-memory profiles are available.",
            "Prewarm targets come from allocated_count in the selected forecast detail.",
            "Slack priority is computed from DAG earliest/latest timing under the SLO.",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")
    if not by_stage.empty:
        print(by_stage.to_string(index=False))


if __name__ == "__main__":
    main()

