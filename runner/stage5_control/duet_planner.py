"""DUET planner: Demand-Uncertainty Elastic Time-aware joint controller.

Compared to the existing baselines + smiless_pareto, DUET makes three novel
choices:

1. Per-window confidence-blended warm allocation.
   smiless_pareto and always_warm pick a single warm multiplier (e.g. 0.5x
   forecast) and apply it to every window. DUET blends p50 + beta * (p95-p50)
   per window, so windows where the forecaster is confident provision only
   ceil(p50) while windows with large forecast spread expand toward p95.
   This converts forecast uncertainty into a localized, controllable risk
   budget, instead of a static safety margin paid every window.

2. Spread-elastic keepalive applied only when persistence is plausible.
   smiless_pareto keepalive is fixed per stage. DUET defaults to keepalive=0
   (scale-to-zero between windows) and only opens an adaptive TTL when:
     (a) the previous window already provisioned warm capacity, AND
     (b) the next window's forecast spread or absolute demand justifies the
         retention cost.
   This means warm pool retention is paid only across genuinely periodic
   busy stretches, never on isolated bursts or steady-state idle.

3. Critical-path-aware memory upgrade + DAG-JIT warmup costing.
   stepconf_style picks the cheapest tier inside the sub-SLO budget and
   does not look at cold-start tails. DUET starts from the same cost-effective
   tier but only upgrades stages whose adjusted cold dispatch dominates a
   configurable share of the DAG slack budget. For stages downstream of a
   root we use warmup_mode=dag_jit so planned warm capacity is charged only
   for the interval the DAG actually needs it, not for the whole control
   window. This is consistent with how a real DAG-aware warm manager would
   issue prewarm calls.

DUET is deterministic and single-pass -- no beam search -- so it is also
cheaper to compute than smiless_pareto. The objective is not to dominate
smiless_pareto on a single metric but to consistently improve the
cost-vs-SLA Pareto frontier on the spoken_dialog DAG workload.
"""

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


POLICY_TO_QUANTILE = {"p50": 0.50, "p90": 0.90, "p95": 0.95}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "DUET planner: forecast-quantile-spread-driven per-window keepalive, "
            "critical-path-aware per-stage memory, and dag_jit warm provisioning."
        )
    )
    p.add_argument("--workflow-config", required=True)
    p.add_argument("--forecast-detail", required=True)
    p.add_argument("--latency-samples", required=True)
    p.add_argument("--method", required=True)
    p.add_argument(
        "--policy",
        choices=["p50", "p90", "p95"],
        default="p95",
        help="quantile policy whose forecast_count rows define the warm budget",
    )
    p.add_argument(
        "--reference-policy",
        choices=["p50", "p90", "p95"],
        default="p50",
        help="lower-quantile policy used to derive forecast uncertainty spread",
    )
    p.add_argument("--fold-id", type=int, default=None)
    p.add_argument("--window-sec", type=float, default=5.0)
    p.add_argument("--slo-ms", type=float, required=True)
    p.add_argument("--memory-tiers-mb", default="128,256,512,1024")
    p.add_argument("--base-memory-mb", type=int, default=256)
    p.add_argument("--cpu-alpha", type=float, default=1.0)
    p.add_argument("--overhead-alpha", type=float, default=0.08)
    p.add_argument(
        "--platform-keepalive-sec",
        type=float,
        default=15.0,
        help="upper cap on adaptive keepalive ttl",
    )
    p.add_argument(
        "--min-keepalive-sec",
        type=float,
        default=0.0,
        help=(
            "baseline keepalive ttl when a stage is provisioned; DUET defaults to 0 "
            "(scale-to-zero between windows) and only adds positive ttl when "
            "persistence + uncertainty bonuses are justified"
        ),
    )
    p.add_argument(
        "--warm-blend-beta",
        type=float,
        default=0.35,
        help=(
            "fraction of the p_q - p_ref forecast spread added to the p_ref base "
            "when choosing warm_count = ceil(p_ref + beta * spread); 0.0 means "
            "warm_count = ceil(p_ref), 1.0 means warm_count = ceil(p_q)"
        ),
    )
    p.add_argument(
        "--warm-scale-multiplier",
        type=float,
        default=0.55,
        help=(
            "multiplicative shrink applied to the blended warm_count to account for "
            "queue smoothing within a window (similar to smiless_pareto's "
            "global warm multiplier, but layered on top of the per-window blend)"
        ),
    )
    p.add_argument(
        "--uncertainty-gain-sec",
        type=float,
        default=8.0,
        help=(
            "how many extra keepalive seconds full forecast-spread uncertainty buys; "
            "actual contribution scales linearly with normalized spread"
        ),
    )
    p.add_argument(
        "--persistence-gain-sec",
        type=float,
        default=3.0,
        help=(
            "bonus added when the preceding control window already needed "
            "non-zero warm capacity (helps periodic patterns)"
        ),
    )
    p.add_argument(
        "--critical-bonus-sec",
        type=float,
        default=2.0,
        help="extra keepalive seconds for stages flagged cold-tail-critical",
    )
    p.add_argument(
        "--critical-slack-ratio",
        type=float,
        default=0.5,
        help=(
            "slack budget threshold; stages where cold_dispatch_q_ms exceeds "
            "ratio * dag_path_budget classify as cold-tail-critical and may "
            "upgrade one memory tier"
        ),
    )
    p.add_argument(
        "--memory-upgrade-tier",
        type=int,
        default=0,
        help=(
            "explicit memory tier (MB) for cold-tail-critical stages; 0 means "
            "auto-upgrade to one tier above cost-effective"
        ),
    )
    p.add_argument(
        "--memory-mode",
        choices=["auto", "base", "fixed"],
        default="base",
        help=(
            "base (default) = start from --base-memory-mb and only upgrade cold-tail-"
            "critical stages; auto = smallest uniform tier whose warm critical "
            "path fits SLO; fixed = always use --memory-upgrade-tier"
        ),
    )
    p.add_argument(
        "--zero-demand-threshold",
        type=float,
        default=0.5,
        help=(
            "forecast point estimates below this threshold are treated as zero "
            "demand so no warm capacity is provisioned"
        ),
    )
    p.add_argument(
        "--keepalive-demand-floor",
        type=float,
        default=1.0,
        help=(
            "minimum forecast spread (p_q - p_ref) required to open keepalive; "
            "if the spread is below this floor only persistence bonus can fire"
        ),
    )
    p.add_argument(
        "--warmup-mode",
        choices=["window", "dag_jit"],
        default="dag_jit",
        help="dag_jit recommended; only root stages are charged from window start",
    )
    p.add_argument("--max-plan-windows", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def filtered_forecast(
    detail: pd.DataFrame,
    *,
    workflow_name: str,
    method: str,
    fold_id: int | None,
    max_plan_windows: int,
) -> pd.DataFrame:
    if "target_window" not in detail.columns and "window" in detail.columns:
        detail = detail.copy()
        detail["target_window"] = detail["window"]
    rows = detail[detail["workflow_name"].astype(str) == workflow_name].copy()
    if "method" in rows.columns:
        rows = rows[rows["method"].astype(str) == method].copy()
    if fold_id is not None and "fold_id" in rows.columns:
        rows = rows[pd.to_numeric(rows["fold_id"], errors="coerce") == fold_id].copy()
    if rows.empty:
        raise ValueError(
            f"forecast_detail has no rows for workflow={workflow_name}, method={method}, fold_id={fold_id}"
        )
    for col in ["target_window", "forecast_count", "allocated_count"]:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows.dropna(subset=["target_window"]).copy()
    rows["target_window"] = rows["target_window"].astype(int)
    if max_plan_windows and max_plan_windows > 0:
        windows = sorted(rows["target_window"].unique())[:max_plan_windows]
        rows = rows[rows["target_window"].isin(windows)].copy()
    return rows


def stage_quantile_pivot(
    rows: pd.DataFrame, policy_quantile: str, policy_reference: str
) -> pd.DataFrame:
    """Return wide-format DataFrame: stage_name, target_window, p_q, p_ref."""
    sel = rows[rows["policy"].astype(str).isin({policy_quantile, policy_reference})].copy()
    if sel.empty:
        raise ValueError(
            f"forecast_detail missing rows for policies {policy_quantile} or {policy_reference}"
        )
    pivot = (
        sel.pivot_table(
            index=["stage_name", "target_window"],
            columns="policy",
            values="forecast_count",
            aggfunc="max",
        )
        .reset_index()
    )
    pivot = pivot.rename(
        columns={policy_quantile: "demand_q", policy_reference: "demand_ref"}
    )
    if "demand_q" not in pivot.columns or "demand_ref" not in pivot.columns:
        raise ValueError(
            f"could not pivot forecast detail for {policy_quantile}/{policy_reference}"
        )
    pivot["demand_q"] = pivot["demand_q"].fillna(0.0)
    pivot["demand_ref"] = pivot["demand_ref"].fillna(0.0)
    return pivot


def critical_path_memory_plan(
    workflow: WorkflowSpec,
    ordered: list[str],
    latency_profile: pd.DataFrame,
    memory_tiers: list[int],
    *,
    base_memory_mb: int,
    cpu_alpha: float,
    overhead_alpha: float,
    slo_ms: float,
    critical_slack_ratio: float,
    memory_upgrade_tier: int,
    memory_mode: str,
) -> pd.DataFrame:
    """Choose the smallest tier whose uniform-warm workflow latency fits the SLO.

    Stages whose cold-path latency consumes more than `critical_slack_ratio` of
    the workflow's warm critical path are upgraded one tier above the base, but
    only when an upgrade actually reduces the critical-path warm latency.
    """
    profile_lookup = latency_profile.set_index("stage_name").to_dict("index")

    def warm_duration(stage: str, memory_mb: int) -> float:
        return adjusted_latency_ms(
            profile_lookup[stage]["warm_overhead_q_ms"],
            profile_lookup[stage]["warm_action_q_ms"],
            memory_mb,
            base_memory_mb,
            cpu_alpha,
            overhead_alpha,
        )

    def cold_duration(stage: str, memory_mb: int) -> float:
        return adjusted_latency_ms(
            profile_lookup[stage]["cold_overhead_q_ms"],
            profile_lookup[stage]["cold_action_q_ms"],
            memory_mb,
            base_memory_mb,
            cpu_alpha,
            overhead_alpha,
        )

    def workflow_warm_latency(memory_mb: int) -> float:
        durations = {stage: warm_duration(stage, memory_mb) for stage in ordered}
        return workflow_warm_critical_path(workflow, ordered, durations)

    sorted_tiers = sorted(memory_tiers)
    if memory_mode == "base":
        cost_tier = base_memory_mb if base_memory_mb in memory_tiers else sorted_tiers[0]
    elif memory_mode == "fixed":
        cost_tier = (
            memory_upgrade_tier if memory_upgrade_tier in memory_tiers else sorted_tiers[-1]
        )
    else:
        feasible_tiers = [t for t in sorted_tiers if workflow_warm_latency(t) <= slo_ms]
        cost_tier = feasible_tiers[0] if feasible_tiers else sorted_tiers[-1]
    cost_tier_latency = workflow_warm_latency(cost_tier)

    # 2. critical-path slack using uniform cost-tier durations
    base_durations = {stage: warm_duration(stage, cost_tier) for stage in ordered}
    slack_df = dag_slack(workflow, ordered, base_durations, slo_ms)
    slack_lookup = slack_df.set_index("stage_name").to_dict("index")

    rows = []
    for stage in ordered:
        warm_at_cost = warm_duration(stage, cost_tier)
        cold_at_cost = cold_duration(stage, cost_tier)
        slack_ms = float(slack_lookup[stage]["slack_ms"])
        budget_ms = float(slack_lookup[stage]["stage_budget_ms"])
        # Cold-tail-critical: cold path at cost tier is large relative to the
        # workflow's warm critical-path budget. Only upgrade when the stage
        # contributes meaningfully to the critical path.
        cold_share = cold_at_cost / max(cost_tier_latency, 1.0)
        is_critical = cold_share >= critical_slack_ratio

        chosen_tier = cost_tier
        reason = "cost-effective"
        if is_critical:
            larger = [t for t in sorted_tiers if t > cost_tier]
            if memory_upgrade_tier > 0 and memory_upgrade_tier in memory_tiers:
                upgrade_target = memory_upgrade_tier
            elif larger:
                upgrade_target = larger[0]
            else:
                upgrade_target = cost_tier
            if upgrade_target > cost_tier and cold_duration(stage, upgrade_target) < cold_at_cost:
                chosen_tier = upgrade_target
                reason = "critical-cold-tail-upgrade"
            else:
                reason = "cost-effective-no-upgrade-available"
        chosen_warm = warm_duration(stage, chosen_tier)
        chosen_cold = cold_duration(stage, chosen_tier)
        rows.append(
            {
                "stage_name": stage,
                "selected_memory_mb": int(chosen_tier),
                "warm_q_ms": float(chosen_warm),
                "cold_q_ms": float(chosen_cold),
                "slack_ms": slack_ms,
                "is_critical": bool(is_critical),
                "stage_budget_ms": budget_ms,
                "reason": reason,
                "cost_effective_tier_mb": cost_tier,
                "warm_q_at_cost_tier_ms": warm_at_cost,
                "cold_q_at_cost_tier_ms": cold_at_cost,
                "cost_tier_workflow_latency_ms": cost_tier_latency,
                "cold_share_of_critical_path": cold_share,
            }
        )
    return pd.DataFrame(rows)


def workflow_warm_critical_path(
    workflow: WorkflowSpec, ordered: list[str], stage_duration: dict[str, float]
) -> float:
    completions: dict[str, float] = {}
    for stage in ordered:
        ready = max(
            (completions[parent] for parent in workflow.nodes[stage].parents), default=0.0
        )
        completions[stage] = ready + stage_duration[stage]
    return max(completions.values()) if completions else 0.0


def build_duet_plan(
    *,
    workflow: WorkflowSpec,
    forecast_rows: pd.DataFrame,
    policy_q: str,
    policy_ref: str,
    memory_plan: pd.DataFrame,
    window_sec: float,
    min_keepalive_sec: float,
    max_keepalive_sec: float,
    warm_blend_beta: float,
    warm_scale_multiplier: float,
    uncertainty_gain_sec: float,
    persistence_gain_sec: float,
    critical_bonus_sec: float,
    zero_demand_threshold: float,
    keepalive_demand_floor: float,
) -> tuple[ControlPlan, pd.DataFrame]:
    pivot = stage_quantile_pivot(forecast_rows, policy_q, policy_ref)
    memory_lookup = memory_plan.set_index("stage_name").to_dict("index")
    pivot = pivot.sort_values(["stage_name", "target_window"]).reset_index(drop=True)

    rows: list[PlanRow] = []
    diag_rows: list[dict] = []
    for stage_name, stage_group in pivot.groupby("stage_name"):
        stage_group = stage_group.sort_values("target_window").reset_index(drop=True)
        mem_info = memory_lookup.get(str(stage_name), {})
        memory_mb = int(mem_info.get("selected_memory_mb", 256))
        is_critical = bool(mem_info.get("is_critical", False))
        prev_warm_active = False
        for _, row in stage_group.iterrows():
            window = int(row["target_window"])
            demand_q = max(0.0, float(row["demand_q"]))
            demand_ref = max(0.0, float(row["demand_ref"]))
            spread = max(0.0, demand_q - demand_ref)

            # 1. confidence-blended warm allocation
            blended = demand_ref + warm_blend_beta * spread
            scaled = warm_scale_multiplier * blended
            if demand_ref < zero_demand_threshold and demand_q < zero_demand_threshold:
                warm_count = 0
            else:
                # round up but never drop below 1 when there is any meaningful demand
                warm_count = max(1, int(math.ceil(scaled))) if scaled > 0 else 0

            # 2. spread-elastic keepalive (default off)
            base_ref = max(demand_ref, 1.0)
            rel_spread = min(1.0, spread / base_ref)
            spread_qualifies = spread >= keepalive_demand_floor
            if warm_count <= 0:
                keepalive_sec = 0.0
                strategy = "scale-to-zero"
            else:
                bonus_unc = uncertainty_gain_sec * rel_spread if spread_qualifies else 0.0
                bonus_pers = persistence_gain_sec if prev_warm_active else 0.0
                bonus_crit = critical_bonus_sec if is_critical else 0.0
                ttl = min_keepalive_sec + bonus_unc + bonus_pers + bonus_crit
                ttl = float(max(0.0, min(max_keepalive_sec, ttl)))
                keepalive_sec = ttl
                if ttl <= 0.0:
                    strategy = "warm-no-keepalive"
                elif bonus_unc > 0 and bonus_pers > 0:
                    strategy = "persistent-uncertain-warm"
                elif bonus_unc > 0:
                    strategy = "uncertain-spike-warm"
                elif bonus_pers > 0:
                    strategy = "persistent-warm"
                else:
                    strategy = "critical-warm"

            rows.append(
                PlanRow(
                    workflow_name=workflow.workflow_name,
                    stage_name=str(stage_name),
                    window=window,
                    warm_count=float(warm_count),
                    keepalive_ttl_sec=keepalive_sec,
                    memory_mb=memory_mb,
                    source="duet_planner",
                    note=strategy,
                )
            )
            diag_rows.append(
                {
                    "stage_name": stage_name,
                    "target_window": window,
                    "demand_ref": demand_ref,
                    "demand_q": demand_q,
                    "spread": spread,
                    "warm_count": warm_count,
                    "keepalive_ttl_sec": keepalive_sec,
                    "memory_mb": memory_mb,
                    "is_critical": is_critical,
                    "strategy": strategy,
                    "prev_warm_active": prev_warm_active,
                }
            )
            prev_warm_active = warm_count > 0

    diag = pd.DataFrame(diag_rows)
    plan = ControlPlan(
        rows=rows,
        window_sec=window_sec,
        metadata={
            "workflow_name": workflow.workflow_name,
            "planner": "duet_planner",
            "policy_quantile": policy_q,
            "policy_reference": policy_ref,
            "warm_blend_beta": warm_blend_beta,
            "warm_scale_multiplier": warm_scale_multiplier,
        },
    )
    return plan, diag


def main() -> None:
    args = parse_args()
    root = project_root()
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    workflow = load_workflow(str(resolve_path(root, args.workflow_config)))
    ordered = topological_nodes(workflow)
    memory_tiers = parse_memory_tiers(args.memory_tiers_mb)
    quantile = POLICY_TO_QUANTILE[args.policy]

    forecast_detail = pd.read_csv(resolve_path(root, args.forecast_detail))
    forecast_rows = filtered_forecast(
        forecast_detail,
        workflow_name=workflow.workflow_name,
        method=args.method,
        fold_id=args.fold_id,
        max_plan_windows=args.max_plan_windows,
    )

    latency_samples = pd.read_csv(resolve_path(root, args.latency_samples))
    latency_profile = latency_quantiles(latency_samples, workflow.workflow_name, quantile)
    latency_profile.to_csv(out_dir / "latency_quantile_profile.csv", index=False)

    memory_plan = critical_path_memory_plan(
        workflow,
        ordered,
        latency_profile,
        memory_tiers,
        base_memory_mb=args.base_memory_mb,
        cpu_alpha=args.cpu_alpha,
        overhead_alpha=args.overhead_alpha,
        slo_ms=args.slo_ms,
        critical_slack_ratio=args.critical_slack_ratio,
        memory_upgrade_tier=args.memory_upgrade_tier,
        memory_mode=args.memory_mode,
    )
    memory_plan.to_csv(out_dir / "duet_memory_plan.csv", index=False)

    plan, diag = build_duet_plan(
        workflow=workflow,
        forecast_rows=forecast_rows,
        policy_q=args.policy,
        policy_ref=args.reference_policy,
        memory_plan=memory_plan,
        window_sec=args.window_sec,
        min_keepalive_sec=args.min_keepalive_sec,
        max_keepalive_sec=args.platform_keepalive_sec,
        warm_blend_beta=args.warm_blend_beta,
        warm_scale_multiplier=args.warm_scale_multiplier,
        uncertainty_gain_sec=args.uncertainty_gain_sec,
        persistence_gain_sec=args.persistence_gain_sec,
        critical_bonus_sec=args.critical_bonus_sec,
        zero_demand_threshold=args.zero_demand_threshold,
        keepalive_demand_floor=args.keepalive_demand_floor,
    )

    save_control_plan(plan, out_dir / "duet_control_plan.json")
    frame = plan_to_frame(plan)
    frame.to_csv(out_dir / "duet_control_plan.csv", index=False)
    diag.to_csv(out_dir / "duet_per_window_decisions.csv", index=False)

    # forecast frame for cost estimation (uses the chosen policy's forecast_count)
    forecast_for_cost = forecast_rows[
        forecast_rows["policy"].astype(str) == args.policy
    ].copy()
    cost = estimate_control_plan_cost(
        plan,
        forecast_detail=forecast_for_cost,
        latency_samples=latency_samples,
        workflow_name=workflow.workflow_name,
        window_sec=args.window_sec,
        demand_column="forecast_count",
        base_memory_mb=args.base_memory_mb,
        cpu_alpha=args.cpu_alpha,
        warmup_mode=args.warmup_mode,
        workflow=workflow,
    )

    summary = {
        "planner": "duet_planner",
        "workflow_name": workflow.workflow_name,
        "policy_quantile": args.policy,
        "policy_reference": args.reference_policy,
        "warmup_mode": args.warmup_mode,
        "min_keepalive_sec": args.min_keepalive_sec,
        "max_keepalive_sec": args.platform_keepalive_sec,
        "warm_blend_beta": args.warm_blend_beta,
        "warm_scale_multiplier": args.warm_scale_multiplier,
        "uncertainty_gain_sec": args.uncertainty_gain_sec,
        "persistence_gain_sec": args.persistence_gain_sec,
        "critical_bonus_sec": args.critical_bonus_sec,
        "zero_demand_threshold": args.zero_demand_threshold,
        "keepalive_demand_floor": args.keepalive_demand_floor,
        "critical_slack_ratio": args.critical_slack_ratio,
        "memory_tiers_mb": memory_tiers,
        "rows": len(plan.rows),
        "mean_warm_count": float(frame["warm_count"].mean()) if not frame.empty else 0.0,
        "max_warm_count": float(frame["warm_count"].max()) if not frame.empty else 0.0,
        "mean_keepalive_ttl_sec": float(frame["keepalive_ttl_sec"].mean()) if not frame.empty else 0.0,
        "max_keepalive_ttl_sec": float(frame["keepalive_ttl_sec"].max()) if not frame.empty else 0.0,
        "mean_memory_mb": float(frame["memory_mb"].mean()) if not frame.empty else 0.0,
        "total_gb_seconds": cost.total_gb_seconds,
        "execution_gb_seconds": cost.execution_gb_seconds,
        "warm_gb_seconds": cost.warm_gb_seconds,
    }
    (out_dir / "duet_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_config": str(resolve_path(root, args.workflow_config)),
        "forecast_detail": str(resolve_path(root, args.forecast_detail)),
        "latency_samples": str(resolve_path(root, args.latency_samples)),
        "method": args.method,
        "policy": args.policy,
        "fold_id": args.fold_id,
        "slo_ms": args.slo_ms,
        "window_sec": args.window_sec,
        "memory_tiers_mb": memory_tiers,
        "warmup_mode": args.warmup_mode,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("DUET planner summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
