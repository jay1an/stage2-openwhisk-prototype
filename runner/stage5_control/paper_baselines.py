from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..workflow import WorkflowSpec, load_workflow
from .control_plan import ControlPlan, PlanRow, plan_to_frame, save_control_plan
from .cost_model import estimate_control_plan_cost
from .plan_joint_control import (
    adjusted_latency_ms,
    dag_slack,
    latency_quantiles,
    parse_memory_tiers,
    topological_nodes,
)
from .propagator import propagate_entry_to_stage


POLICY_TO_QUANTILE = {
    "p50": 0.50,
    "p90": 0.90,
    "p95": 0.95,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper baseline control plans: scale-to-zero, always-warm, "
            "ORION-style right-prewarming/right-sizing, and StepConf-style "
            "critical-path memory configuration."
        )
    )
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--forecast-detail", default=None)
    parser.add_argument("--entry-forecast", default=None)
    parser.add_argument("--delay-kernel", default=None)
    parser.add_argument("--latency-samples", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--policy", choices=["p50", "p90", "p95"], default="p95")
    parser.add_argument("--fold-id", type=int, default=None)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--slo-ms", type=float, required=True)
    parser.add_argument("--memory-tiers-mb", default="128,256,512,1024")
    parser.add_argument("--base-memory-mb", type=int, default=256)
    parser.add_argument("--cpu-alpha", type=float, default=1.0)
    parser.add_argument("--overhead-alpha", type=float, default=0.08)
    parser.add_argument(
        "--platform-keepalive-sec",
        type=float,
        default=20.0,
        help="scaled platform idle retention used by default-style baselines",
    )
    parser.add_argument(
        "--orion-downstream-warm-discount",
        type=float,
        default=0.7,
        help=(
            "discount applied to ORION-style downstream prewarming to approximate "
            "reactive prewarm delay"
        ),
    )
    parser.add_argument(
        "--warmup-mode",
        choices=["window", "dag_jit"],
        default="window",
        help="timing model used for costing planned warm capacity",
    )
    parser.add_argument("--always-warm-mode", choices=["peak", "one"], default="peak")
    parser.add_argument("--warm-source", choices=["allocated_count", "forecast_count"], default="allocated_count")
    parser.add_argument("--max-plan-windows", type=int, default=0, help="0 means all selected windows")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def selected_forecast_detail(
    detail: pd.DataFrame,
    *,
    workflow_name: str,
    method: str,
    policy: str,
    fold_id: int | None,
    max_plan_windows: int,
) -> pd.DataFrame:
    if "target_window" not in detail.columns and "window" in detail.columns:
        detail = detail.copy()
        detail["target_window"] = detail["window"]
    selected = detail[(detail["workflow_name"] == workflow_name) & (detail["policy"] == policy)].copy()
    if "method" in selected.columns:
        selected = selected[selected["method"] == method].copy()
    if fold_id is not None and "fold_id" in selected.columns:
        selected = selected[selected["fold_id"] == fold_id].copy()
    if selected.empty:
        raise ValueError(
            f"no forecast rows for workflow={workflow_name}, method={method}, policy={policy}, fold_id={fold_id}"
        )
    for col in ["target_window", "forecast_count", "allocated_count"]:
        if col in selected.columns:
            selected[col] = pd.to_numeric(selected[col], errors="coerce")
    selected = selected.dropna(subset=["target_window"]).copy()
    selected["target_window"] = selected["target_window"].astype(int)
    if max_plan_windows and max_plan_windows > 0:
        windows = sorted(selected["target_window"].unique())[:max_plan_windows]
        selected = selected[selected["target_window"].isin(windows)].copy()
    return selected


def selected_entry_forecast(
    entry: pd.DataFrame,
    *,
    workflow_name: str,
    method: str,
    policy: str,
    max_plan_windows: int,
) -> pd.DataFrame:
    selected = entry[
        (entry["workflow_name"].astype(str) == workflow_name)
        & (entry["policy"].astype(str) == policy)
    ].copy()
    if "method" in selected.columns:
        selected = selected[selected["method"].astype(str) == method].copy()
    if selected.empty:
        raise ValueError(
            f"no entry forecast rows for workflow={workflow_name}, method={method}, policy={policy}"
        )
    selected["target_window"] = pd.to_numeric(selected["target_window"], errors="coerce")
    selected["forecast_count"] = pd.to_numeric(selected["forecast_count"], errors="coerce")
    selected = selected.dropna(subset=["target_window", "forecast_count"]).copy()
    selected["target_window"] = selected["target_window"].astype(int)
    if max_plan_windows and max_plan_windows > 0:
        windows = sorted(selected["target_window"].unique())[:max_plan_windows]
        selected = selected[selected["target_window"].isin(windows)].copy()
    return selected


def stage_window_table(selected: pd.DataFrame) -> pd.DataFrame:
    return (
        selected.groupby(["stage_name", "target_window"], as_index=False)[
            ["forecast_count", "allocated_count"]
        ]
        .max()
        .reset_index(drop=True)
    )


def baseline_prev_warm(
    baseline_name: str,
    *,
    warm_count: int,
    keepalive_ttl_sec: float,
) -> bool:
    if baseline_name == "cold_every_time":
        return False
    if baseline_name == "platform_default":
        return keepalive_ttl_sec > 0.0
    if baseline_name == "always_warm":
        return True
    if baseline_name == "orion_style":
        return warm_count > 0
    if baseline_name == "stepconf_style":
        return keepalive_ttl_sec > 0.0
    return bool(warm_count > 0 or keepalive_ttl_sec > 0.0)


def stage_duration(
    profile_lookup: dict[str, dict],
    stage_name: str,
    *,
    memory_mb: int,
    base_memory_mb: int,
    cpu_alpha: float,
    overhead_alpha: float,
    latency_class: str,
) -> float:
    profile = profile_lookup[stage_name]
    prefix = "cold" if latency_class == "cold_like" else "warm"
    return adjusted_latency_ms(
        profile[f"{prefix}_overhead_q_ms"],
        profile[f"{prefix}_action_q_ms"],
        memory_mb,
        base_memory_mb,
        cpu_alpha,
        overhead_alpha,
    )


def workflow_latency_estimate(
    workflow: WorkflowSpec,
    ordered: list[str],
    duration_lookup: dict[str, float],
) -> float:
    completions: dict[str, float] = {}
    for stage in ordered:
        ready = max((completions[parent] for parent in workflow.nodes[stage].parents), default=0.0)
        completions[stage] = ready + duration_lookup[stage]
    return max(completions.values())


def orion_style_memory_plan(
    workflow: WorkflowSpec,
    ordered: list[str],
    latency_profile: pd.DataFrame,
    memory_tiers: list[int],
    *,
    base_memory_mb: int,
    cpu_alpha: float,
    overhead_alpha: float,
    slo_ms: float,
) -> pd.DataFrame:
    profile_lookup = latency_profile.set_index("stage_name").to_dict("index")
    roots = {stage for stage in workflow.nodes if not workflow.nodes[stage].parents}
    state = {stage: min(memory_tiers, key=lambda tier: abs(tier - base_memory_mb)) for stage in ordered}

    def duration_for_state() -> dict[str, float]:
        return {
            stage: stage_duration(
                profile_lookup,
                stage,
                memory_mb=int(state[stage]),
                base_memory_mb=base_memory_mb,
                cpu_alpha=cpu_alpha,
                overhead_alpha=overhead_alpha,
                latency_class="cold_like" if stage in roots else "warm",
            )
            for stage in ordered
        }

    current_latency = workflow_latency_estimate(workflow, ordered, duration_for_state())
    changes = 0
    while current_latency > slo_ms:
        best = None
        for stage in ordered:
            current_tier = int(state[stage])
            larger = [tier for tier in memory_tiers if tier > current_tier]
            if not larger:
                continue
            next_tier = min(larger)
            old_duration = stage_duration(
                profile_lookup,
                stage,
                memory_mb=current_tier,
                base_memory_mb=base_memory_mb,
                cpu_alpha=cpu_alpha,
                overhead_alpha=overhead_alpha,
                latency_class="cold_like" if stage in roots else "warm",
            )
            new_duration = stage_duration(
                profile_lookup,
                stage,
                memory_mb=next_tier,
                base_memory_mb=base_memory_mb,
                cpu_alpha=cpu_alpha,
                overhead_alpha=overhead_alpha,
                latency_class="cold_like" if stage in roots else "warm",
            )
            trial = dict(state)
            trial[stage] = next_tier
            trial_duration = {
                node: stage_duration(
                    profile_lookup,
                    node,
                    memory_mb=int(trial[node]),
                    base_memory_mb=base_memory_mb,
                    cpu_alpha=cpu_alpha,
                    overhead_alpha=overhead_alpha,
                    latency_class="cold_like" if node in roots else "warm",
                )
                for node in ordered
            }
            trial_latency = workflow_latency_estimate(workflow, ordered, trial_duration)
            latency_gain = current_latency - trial_latency
            extra_gb_seconds = max(
                1e-9,
                (next_tier / 1024.0) * (new_duration / 1000.0)
                - (current_tier / 1024.0) * (old_duration / 1000.0),
            )
            score = latency_gain / extra_gb_seconds
            if best is None or score > best["score"]:
                best = {
                    "stage": stage,
                    "next_tier": next_tier,
                    "trial_latency": trial_latency,
                    "score": score,
                }
        if best is None or best["trial_latency"] >= current_latency:
            break
        state[best["stage"]] = best["next_tier"]
        current_latency = float(best["trial_latency"])
        changes += 1

    rows = []
    for stage in ordered:
        rows.append(
            {
                "stage_name": stage,
                "selected_memory_mb": int(state[stage]),
                "selection_reason": "orion-style-bfs-right-sizing",
                "estimated_workflow_latency_ms": current_latency,
                "memory_changes": changes,
            }
        )
    return pd.DataFrame(rows)


def stepconf_style_memory_plan(
    workflow: WorkflowSpec,
    ordered: list[str],
    latency_profile: pd.DataFrame,
    memory_tiers: list[int],
    *,
    base_memory_mb: int,
    cpu_alpha: float,
    overhead_alpha: float,
    slo_ms: float,
) -> pd.DataFrame:
    profile_lookup = latency_profile.set_index("stage_name").to_dict("index")
    cost_effective_duration: dict[str, float] = {}
    cost_effective_tier: dict[str, int] = {}
    for stage in ordered:
        candidates = []
        for memory_mb in memory_tiers:
            warm_ms = stage_duration(
                profile_lookup,
                stage,
                memory_mb=memory_mb,
                base_memory_mb=base_memory_mb,
                cpu_alpha=cpu_alpha,
                overhead_alpha=overhead_alpha,
                latency_class="warm",
            )
            gb_seconds = (memory_mb / 1024.0) * (warm_ms / 1000.0)
            candidates.append((gb_seconds, warm_ms, memory_mb))
        gb_seconds, warm_ms, memory_mb = sorted(candidates)[0]
        cost_effective_duration[stage] = warm_ms
        cost_effective_tier[stage] = memory_mb

    slack = dag_slack(workflow, ordered, cost_effective_duration, slo_ms)
    budget_lookup = slack.set_index("stage_name")["stage_budget_ms"].to_dict()
    rows = []
    for stage in ordered:
        candidates = []
        for memory_mb in memory_tiers:
            warm_ms = stage_duration(
                profile_lookup,
                stage,
                memory_mb=memory_mb,
                base_memory_mb=base_memory_mb,
                cpu_alpha=cpu_alpha,
                overhead_alpha=overhead_alpha,
                latency_class="warm",
            )
            gb_seconds = (memory_mb / 1024.0) * (warm_ms / 1000.0)
            candidates.append(
                {
                    "memory_mb": memory_mb,
                    "warm_q_ms": warm_ms,
                    "gb_seconds_per_invocation": gb_seconds,
                }
            )
        candidate_df = pd.DataFrame(candidates)
        budget = max(1.0, float(budget_lookup[stage]))
        feasible = candidate_df[candidate_df["warm_q_ms"] <= budget]
        if feasible.empty:
            chosen = candidate_df.sort_values("warm_q_ms").iloc[0]
            reason = "stepconf-fastest-budget-infeasible"
        else:
            chosen = feasible.sort_values("gb_seconds_per_invocation").iloc[0]
            reason = "stepconf-cheapest-within-sub-slo"
        rows.append(
            {
                "stage_name": stage,
                "selected_memory_mb": int(chosen["memory_mb"]),
                "selection_reason": reason,
                "stage_budget_ms": budget,
                "cost_effective_memory_mb": cost_effective_tier[stage],
            }
        )
    return pd.DataFrame(rows)


def build_plan(
    *,
    baseline_name: str,
    selected: pd.DataFrame,
    workflow: WorkflowSpec,
    window_sec: float,
    memory_lookup: dict[str, int],
    warm_source: str,
    always_warm_mode: str,
    platform_keepalive_sec: float,
    orion_downstream_warm_discount: float,
    warmup_mode: str,
) -> ControlPlan:
    table = stage_window_table(selected)
    peak_by_stage = (
        table.groupby("stage_name")[warm_source].max().clip(lower=1).apply(math.ceil).to_dict()
    )


def build_plan_from_entry(
    *,
    baseline_name: str,
    entry_forecast: pd.DataFrame,
    delay_kernel: pd.DataFrame,
    workflow: WorkflowSpec,
    ordered: list[str],
    policy: str,
    window_sec: float,
    memory_lookup: dict[str, int],
    warm_source: str,
    always_warm_mode: str,
    platform_keepalive_sec: float,
    orion_downstream_warm_discount: float,
    warmup_mode: str,
) -> tuple[ControlPlan, pd.DataFrame]:
    windows = sorted(entry_forecast["target_window"].astype(int).unique())
    if always_warm_mode == "one":
        peak_by_stage = {stage: 1 for stage in ordered}
    else:
        peak_by_stage = {}
        for stage in ordered:
            values = [
                propagate_entry_to_stage(
                    entry_forecast,
                    delay_kernel,
                    workflow_name=workflow.workflow_name,
                    stage_name=stage,
                    target_window=window,
                    policy=policy,
                    prev_warm=True,
                )
                for window in windows
            ]
            peak_by_stage[stage] = max(1, int(math.ceil(max(values) if values else 0.0)))

    prev_warm = {stage: False for stage in ordered}
    rows: list[PlanRow] = []
    forecast_rows: list[dict] = []
    for window in windows:
        next_prev = dict(prev_warm)
        for stage in ordered:
            forecast_count = propagate_entry_to_stage(
                entry_forecast,
                delay_kernel,
                workflow_name=workflow.workflow_name,
                stage_name=stage,
                target_window=window,
                policy=policy,
                prev_warm=prev_warm.get(stage, False),
            )
            allocated_count = int(math.ceil(max(0.0, forecast_count)))
            source_count = forecast_count if warm_source == "forecast_count" else float(allocated_count)
            memory_mb = int(memory_lookup.get(stage, 256))
            if baseline_name == "cold_every_time":
                warm_count = 0
                keepalive_ttl_sec = 0.0
            elif baseline_name == "platform_default":
                warm_count = 0
                keepalive_ttl_sec = platform_keepalive_sec
            elif baseline_name == "always_warm":
                warm_count = int(peak_by_stage[stage])
                keepalive_ttl_sec = platform_keepalive_sec
            elif baseline_name == "orion_style":
                warm_count = 0 if not workflow.nodes[stage].parents else int(math.ceil(source_count * orion_downstream_warm_discount))
                keepalive_ttl_sec = platform_keepalive_sec
            elif baseline_name == "stepconf_style":
                warm_count = 0
                keepalive_ttl_sec = platform_keepalive_sec
            else:
                raise ValueError(f"unknown baseline {baseline_name}")
            rows.append(
                PlanRow(
                    workflow_name=workflow.workflow_name,
                    stage_name=stage,
                    window=int(window),
                    warm_count=float(warm_count),
                    keepalive_ttl_sec=keepalive_ttl_sec,
                    memory_mb=memory_mb,
                    source="paper_baseline",
                    note=baseline_name,
                )
            )
            forecast_rows.append(
                {
                    "workflow_name": workflow.workflow_name,
                    "method": "entry-kernel",
                    "stage_name": stage,
                    "target_window": int(window),
                    "policy": policy,
                    "forecast_count": float(forecast_count),
                    "allocated_count": allocated_count,
                }
            )
            next_prev[stage] = baseline_prev_warm(
                baseline_name,
                warm_count=warm_count,
                keepalive_ttl_sec=keepalive_ttl_sec,
            )
        prev_warm = next_prev
    return (
        ControlPlan(
            rows=rows,
            window_sec=window_sec,
            metadata={
                "workflow_name": workflow.workflow_name,
                "baseline": baseline_name,
                "warmup_mode": warmup_mode,
                "input_mode": "entry_forecast_delay_kernel",
            },
        ),
        pd.DataFrame(forecast_rows),
    )
    rows: list[PlanRow] = []
    for record in table.to_dict(orient="records"):
        stage = str(record["stage_name"])
        source_count = max(0.0, float(record.get(warm_source, 0.0) or 0.0))
        memory_mb = int(memory_lookup.get(stage, 256))
        if baseline_name == "cold_every_time":
            warm_count = 0
            keepalive_ttl_sec = 0.0
        elif baseline_name == "platform_default":
            warm_count = 0
            keepalive_ttl_sec = platform_keepalive_sec
        elif baseline_name == "always_warm":
            warm_count = 1 if always_warm_mode == "one" else int(peak_by_stage[stage])
            keepalive_ttl_sec = platform_keepalive_sec
        elif baseline_name == "orion_style":
            if not workflow.nodes[stage].parents:
                warm_count = 0
            else:
                warm_count = int(math.ceil(source_count * orion_downstream_warm_discount))
            keepalive_ttl_sec = platform_keepalive_sec
        elif baseline_name == "stepconf_style":
            warm_count = 0
            keepalive_ttl_sec = platform_keepalive_sec
        else:
            raise ValueError(f"unknown baseline {baseline_name}")

        rows.append(
            PlanRow(
                workflow_name=workflow.workflow_name,
                stage_name=stage,
                window=int(record["target_window"]),
                warm_count=float(warm_count),
                keepalive_ttl_sec=keepalive_ttl_sec,
                memory_mb=memory_mb,
                source="paper_baseline",
                note=baseline_name,
            )
        )
    return ControlPlan(
        rows=rows,
        window_sec=window_sec,
        metadata={
            "workflow_name": workflow.workflow_name,
            "baseline": baseline_name,
            "warmup_mode": warmup_mode,
        },
    )


def write_method_notes(out_dir: Path) -> None:
    lines = [
        "# Paper Baseline Adaptation Notes",
        "",
        "## cold_every_time",
        "No prewarming, no platform idle retention. This is an extreme lower-control diagnostic, not a normal managed FaaS default.",
        "",
        "## platform_default",
        "No proactive prewarming, but containers created by traffic are retained for the scaled platform idle timeout.",
        "",
        "## always_warm",
        "Keeps a fixed warm pool for every stage/window and uses the scaled platform idle timeout. The default `peak` mode uses the maximum selected forecast allocation per stage.",
        "",
        "## orion_style",
        "Adapted from ORION's right-prewarming/right-sizing idea for DAG serverless workflows. This implementation does not reproduce function bundling. Root stages are not prewarmed because ORION's prewarming is primarily triggered after a workflow starts; downstream stages are prewarmed using a discounted selected forecast demand as a conservative approximation of reactive prewarming delay. Memory is chosen by a BFS-style right-sizing loop over the workflow latency estimate.",
        "",
        "## stepconf_style",
        "Adapted from StepConf's SLO-aware workflow memory configuration. It chooses per-stage memory tiers using a sub-SLO budget derived from DAG slack, does not proactively prewarm containers, and relies on the scaled platform idle timeout.",
        "",
        "Use Stage4 with `--control-plan` and `--enable-memory-scaling` when evaluating these baselines.",
        "",
    ]
    (out_dir / "method_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    use_entry_kernel = args.entry_forecast is not None or args.delay_kernel is not None
    if use_entry_kernel:
        if args.forecast_detail is not None:
            raise SystemExit("--entry-forecast/--delay-kernel and --forecast-detail are mutually exclusive")
        if args.entry_forecast is None or args.delay_kernel is None:
            raise SystemExit("--entry-forecast and --delay-kernel must be provided together")
    elif args.forecast_detail is None:
        raise SystemExit("one of --forecast-detail or --entry-forecast/--delay-kernel is required")

    root = project_root()
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    workflow = load_workflow(str(resolve_path(root, args.workflow_config)))
    ordered = topological_nodes(workflow)
    memory_tiers = parse_memory_tiers(args.memory_tiers_mb)
    if use_entry_kernel:
        selected_entry = selected_entry_forecast(
            pd.read_csv(resolve_path(root, args.entry_forecast)),
            workflow_name=workflow.workflow_name,
            method=args.method,
            policy=args.policy,
            max_plan_windows=args.max_plan_windows,
        )
        delay_kernel = pd.read_csv(resolve_path(root, args.delay_kernel))
        selected = pd.DataFrame()
    else:
        forecast_detail = pd.read_csv(resolve_path(root, args.forecast_detail))
        selected = selected_forecast_detail(
            forecast_detail,
            workflow_name=workflow.workflow_name,
            method=args.method,
            policy=args.policy,
            fold_id=args.fold_id,
            max_plan_windows=args.max_plan_windows,
        )
    latency_samples = pd.read_csv(resolve_path(root, args.latency_samples))
    latency_profile = latency_quantiles(
        latency_samples,
        workflow.workflow_name,
        POLICY_TO_QUANTILE[args.policy],
    )
    latency_profile.to_csv(out_dir / "baseline_latency_quantile_profile.csv", index=False)

    base_memory = {stage: args.base_memory_mb for stage in workflow.nodes}
    orion_memory = orion_style_memory_plan(
        workflow,
        ordered,
        latency_profile,
        memory_tiers,
        base_memory_mb=args.base_memory_mb,
        cpu_alpha=args.cpu_alpha,
        overhead_alpha=args.overhead_alpha,
        slo_ms=args.slo_ms,
    )
    stepconf_memory = stepconf_style_memory_plan(
        workflow,
        ordered,
        latency_profile,
        memory_tiers,
        base_memory_mb=args.base_memory_mb,
        cpu_alpha=args.cpu_alpha,
        overhead_alpha=args.overhead_alpha,
        slo_ms=args.slo_ms,
    )
    orion_memory.to_csv(out_dir / "orion_style_memory_plan.csv", index=False)
    stepconf_memory.to_csv(out_dir / "stepconf_style_memory_plan.csv", index=False)

    memory_plans = {
        "cold_every_time": base_memory,
        "platform_default": base_memory,
        "always_warm": base_memory,
        "orion_style": orion_memory.set_index("stage_name")["selected_memory_mb"].to_dict(),
        "stepconf_style": stepconf_memory.set_index("stage_name")["selected_memory_mb"].to_dict(),
    }

    summary_rows = []
    for baseline_name, memory_lookup in memory_plans.items():
        plan_warmup_mode = "window" if baseline_name == "always_warm" else args.warmup_mode
        if use_entry_kernel:
            plan, forecast_for_cost = build_plan_from_entry(
                baseline_name=baseline_name,
                entry_forecast=selected_entry,
                delay_kernel=delay_kernel,
                workflow=workflow,
                ordered=ordered,
                policy=args.policy,
                window_sec=args.window_sec,
                memory_lookup=memory_lookup,
                warm_source=args.warm_source,
                always_warm_mode=args.always_warm_mode,
                platform_keepalive_sec=args.platform_keepalive_sec,
                orion_downstream_warm_discount=args.orion_downstream_warm_discount,
                warmup_mode=plan_warmup_mode,
            )
            forecast_for_cost.to_csv(out_dir / f"{baseline_name}_propagated_stage_forecast.csv", index=False)
        else:
            plan = build_plan(
                baseline_name=baseline_name,
                selected=selected,
                workflow=workflow,
                window_sec=args.window_sec,
                memory_lookup=memory_lookup,
                warm_source=args.warm_source,
                always_warm_mode=args.always_warm_mode,
                platform_keepalive_sec=args.platform_keepalive_sec,
                orion_downstream_warm_discount=args.orion_downstream_warm_discount,
                warmup_mode=plan_warmup_mode,
            )
            forecast_for_cost = selected
        save_control_plan(plan, out_dir / f"{baseline_name}_control_plan.json")
        frame = plan_to_frame(plan)
        frame.to_csv(out_dir / f"{baseline_name}_control_plan.csv", index=False)
        cost = estimate_control_plan_cost(
            plan,
            forecast_detail=forecast_for_cost,
            latency_samples=latency_samples,
            workflow_name=workflow.workflow_name,
            window_sec=args.window_sec,
            demand_column="forecast_count",
            base_memory_mb=args.base_memory_mb,
            cpu_alpha=args.cpu_alpha,
            warmup_mode=plan_warmup_mode,
            workflow=workflow,
        )
        summary_rows.append(
            {
                "baseline": baseline_name,
                "control_plan_json": str(out_dir / f"{baseline_name}_control_plan.json"),
                "control_plan_csv": str(out_dir / f"{baseline_name}_control_plan.csv"),
                "rows": len(plan.rows),
                "mean_warm_count": float(frame["warm_count"].mean()) if not frame.empty else 0.0,
                "max_warm_count": float(frame["warm_count"].max()) if not frame.empty else 0.0,
                "mean_keepalive_ttl_sec": float(frame["keepalive_ttl_sec"].mean()) if not frame.empty else 0.0,
                "mean_memory_mb": float(frame["memory_mb"].mean()) if not frame.empty else 0.0,
                "total_gb_seconds": cost.total_gb_seconds,
                "execution_gb_seconds": cost.execution_gb_seconds,
                "warm_gb_seconds": cost.warm_gb_seconds,
                "warmup_mode": plan_warmup_mode,
            }
        )

    pd.DataFrame(summary_rows).to_csv(out_dir / "baseline_plan_summary.csv", index=False)
    write_method_notes(out_dir)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_config": str(resolve_path(root, args.workflow_config)),
        "forecast_detail": str(resolve_path(root, args.forecast_detail)) if args.forecast_detail is not None else None,
        "entry_forecast": str(resolve_path(root, args.entry_forecast)) if args.entry_forecast is not None else None,
        "delay_kernel": str(resolve_path(root, args.delay_kernel)) if args.delay_kernel is not None else None,
        "latency_samples": str(resolve_path(root, args.latency_samples)),
        "method": args.method,
        "policy": args.policy,
        "fold_id": args.fold_id,
        "window_sec": args.window_sec,
        "slo_ms": args.slo_ms,
        "memory_tiers_mb": memory_tiers,
        "base_memory_mb": args.base_memory_mb,
        "cpu_alpha": args.cpu_alpha,
        "overhead_alpha": args.overhead_alpha,
        "platform_keepalive_sec": args.platform_keepalive_sec,
        "orion_downstream_warm_discount": args.orion_downstream_warm_discount,
        "warmup_mode": args.warmup_mode,
        "warm_source": args.warm_source,
        "always_warm_mode": args.always_warm_mode,
        "notes": [
            "ORION-style baseline omits bundling and adapts right-prewarming to window-level warm_count.",
            "ORION-style downstream warm_count uses a discount to approximate reactive prewarm delay.",
            "Default-style baselines use scaled platform keepalive except cold_every_time.",
            "StepConf-style baseline uses per-stage memory tiers and no prewarming.",
            "Evaluate these plans with Stage4 --control-plan --enable-memory-scaling.",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
