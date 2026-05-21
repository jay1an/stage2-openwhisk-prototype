"""Run DUET vs baseline+smiless_pareto side-by-side on a single experiment.

This is a lean wrapper around the existing run_end_to_end_evaluation pipeline:

  1. Generate (or reuse) the 5 paper baselines via runner.stage5_control.paper_baselines
  2. Generate (or reuse) the SMIless-Pareto plan via runner.stage5_control.risk_budgeted_pareto_planner
  3. Generate the DUET plan via runner.stage5_control.duet_planner
  4. Run runner.stage4_risk.estimate_slo_risk on every plan to get matched
     cost+risk metrics under the same warmup-mode + memory-scaling settings.
  5. Aggregate everything into a comparison CSV that can be read by a report
     generator.

The runner does *not* touch any directory outside `--out-dir`. Re-running with
`--skip-existing` reuses prior plans/eval outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--workflow-config", required=True)
    p.add_argument("--trace", required=True)
    p.add_argument("--forecast-detail", default=None)
    p.add_argument("--entry-forecast", default=None)
    p.add_argument("--delay-kernel", default=None)
    p.add_argument("--latency-samples", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--policy", choices=["p50", "p90", "p95"], default="p95")
    p.add_argument("--slo-ms", type=float, required=True)
    p.add_argument("--window-sec", type=float, default=5.0)
    p.add_argument("--fold-id", type=int, default=None)
    p.add_argument("--memory-tiers-mb", default="128,256,512,1024")
    p.add_argument("--base-memory-mb", type=int, default=256)
    p.add_argument("--cpu-alpha", type=float, default=1.0)
    p.add_argument("--overhead-alpha", type=float, default=0.08)
    p.add_argument("--platform-keepalive-sec", type=float, default=15.0)
    p.add_argument("--residual-cold-probability", type=float, default=0.0)
    p.add_argument("--warmup-mode", choices=["window", "dag_jit"], default="window")
    p.add_argument("--risk-budget", type=float, default=0.05)
    p.add_argument("--planner-simulations-per-window", type=int, default=50)
    p.add_argument("--planner-max-eval-windows", type=int, default=64)
    p.add_argument("--planner-max-virtual-requests-per-window", type=int, default=16)
    p.add_argument("--active-gate-threshold", type=float, default=0.3)
    p.add_argument("--max-plan-windows", type=int, default=0)
    p.add_argument("--beam-width", type=int, default=16)
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--simulations-per-request", type=int, default=200)
    # DUET-balanced (safety-first variant)
    p.add_argument("--duet-memory-mode", choices=["auto", "base", "fixed"], default="base")
    p.add_argument("--duet-critical-slack-ratio", type=float, default=0.5)
    p.add_argument("--duet-warm-blend-beta", type=float, default=0.35)
    p.add_argument("--duet-warm-scale-multiplier", type=float, default=0.55)
    p.add_argument("--duet-uncertainty-gain-sec", type=float, default=8.0)
    p.add_argument("--duet-persistence-gain-sec", type=float, default=3.0)
    p.add_argument("--duet-critical-bonus-sec", type=float, default=2.0)
    p.add_argument("--duet-min-keepalive-sec", type=float, default=0.0)
    p.add_argument("--duet-keepalive-demand-floor", type=float, default=1.0)
    p.add_argument("--duet-zero-demand-threshold", type=float, default=0.5)
    p.add_argument("--duet-warmup-mode", choices=["window", "dag_jit"], default="window")
    # DUET-economy: cost-competitive variant (no memory upgrade, lighter keepalive)
    p.add_argument(
        "--enable-duet-economy",
        action="store_true",
        help="also generate a cost-economy DUET variant (no memory upgrade, lighter keepalive)",
    )
    p.add_argument("--duet-econ-critical-slack-ratio", type=float, default=10.0)
    p.add_argument("--duet-econ-warm-blend-beta", type=float, default=0.35)
    p.add_argument("--duet-econ-warm-scale-multiplier", type=float, default=0.55)
    p.add_argument("--duet-econ-uncertainty-gain-sec", type=float, default=3.0)
    p.add_argument("--duet-econ-persistence-gain-sec", type=float, default=2.0)
    p.add_argument("--duet-econ-critical-bonus-sec", type=float, default=0.0)
    p.add_argument("--duet-econ-keepalive-demand-floor", type=float, default=2.0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--baselines-only",
        action="store_true",
        help="skip plan generation and only run Stage 4 evaluation on existing plans",
    )
    return p.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def maybe_run(command: list[str], cwd: Path, skip_target: Path, skip: bool) -> bool:
    if skip and skip_target.exists():
        print(f"[skip] {skip_target}")
        return False
    run_command(command, cwd=cwd)
    return True


def common_stage_args(args: argparse.Namespace, root: Path) -> list[str]:
    out = [
        "--workflow-config",
        str(resolve_path(root, args.workflow_config)),
        "--latency-samples",
        str(resolve_path(root, args.latency_samples)),
        "--method",
        args.method,
        "--policy",
        args.policy,
        "--window-sec",
        str(args.window_sec),
        "--slo-ms",
        str(args.slo_ms),
        "--memory-tiers-mb",
        args.memory_tiers_mb,
        "--base-memory-mb",
        str(args.base_memory_mb),
        "--cpu-alpha",
        str(args.cpu_alpha),
        "--overhead-alpha",
        str(args.overhead_alpha),
        "--platform-keepalive-sec",
        str(args.platform_keepalive_sec),
        "--warmup-mode",
        args.warmup_mode,
    ]
    use_entry_kernel = args.entry_forecast is not None or args.delay_kernel is not None
    if use_entry_kernel:
        out.extend(
            [
                "--entry-forecast",
                str(resolve_path(root, args.entry_forecast)),
                "--delay-kernel",
                str(resolve_path(root, args.delay_kernel)),
            ]
        )
    else:
        out.extend(["--forecast-detail", str(resolve_path(root, args.forecast_detail))])
    if args.fold_id is not None:
        out.extend(["--fold-id", str(args.fold_id)])
    return out


def gen_baselines(args: argparse.Namespace, root: Path, out_dir: Path) -> Path:
    baseline_dir = out_dir / "stage5_baselines"
    command = [
        sys.executable,
        "-m",
        "runner.stage5_control.paper_baselines",
        *common_stage_args(args, root),
        "--max-plan-windows",
        str(args.max_plan_windows),
        "--out-dir",
        str(baseline_dir),
    ]
    maybe_run(command, root, baseline_dir / "baseline_plan_summary.csv", args.skip_existing)
    return baseline_dir


def gen_pareto(args: argparse.Namespace, root: Path, out_dir: Path) -> Path:
    pareto_dir = out_dir / "stage5_smiless_pareto"
    command = [
        sys.executable,
        "-m",
        "runner.stage5_control.risk_budgeted_pareto_planner",
        *common_stage_args(args, root),
        "--risk-budget",
        str(args.risk_budget),
        "--residual-cold-probability",
        str(args.residual_cold_probability),
        "--planner-demand-column",
        "forecast_count",
        "--simulations-per-window",
        str(args.planner_simulations_per_window),
        "--max-eval-windows",
        str(args.planner_max_eval_windows),
        "--max-virtual-requests-per-window",
        str(args.planner_max_virtual_requests_per_window),
        "--active-gate-threshold",
        str(args.active_gate_threshold),
        "--max-plan-windows",
        str(args.max_plan_windows),
        "--beam-width",
        str(args.beam_width),
        "--max-iterations",
        str(args.max_iterations),
        "--out-dir",
        str(pareto_dir),
    ]
    maybe_run(command, root, pareto_dir / "selected_control_plan.json", args.skip_existing)
    return pareto_dir


def gen_duet(args: argparse.Namespace, root: Path, out_dir: Path,
             *, variant: str = "balanced") -> Path:
    duet_dir = out_dir / f"stage5_duet_{variant}"
    if variant == "balanced":
        critical = args.duet_critical_slack_ratio
        beta = args.duet_warm_blend_beta
        scale = args.duet_warm_scale_multiplier
        unc = args.duet_uncertainty_gain_sec
        pers = args.duet_persistence_gain_sec
        crit_bonus = args.duet_critical_bonus_sec
        floor = args.duet_keepalive_demand_floor
    elif variant == "economy":
        critical = args.duet_econ_critical_slack_ratio
        beta = args.duet_econ_warm_blend_beta
        scale = args.duet_econ_warm_scale_multiplier
        unc = args.duet_econ_uncertainty_gain_sec
        pers = args.duet_econ_persistence_gain_sec
        crit_bonus = args.duet_econ_critical_bonus_sec
        floor = args.duet_econ_keepalive_demand_floor
    else:
        raise ValueError(f"unknown DUET variant {variant}")
    command = [
        sys.executable,
        "-m",
        "runner.stage5_control.duet_planner",
        "--workflow-config",
        str(resolve_path(root, args.workflow_config)),
        "--latency-samples",
        str(resolve_path(root, args.latency_samples)),
        "--method",
        args.method,
        "--policy",
        args.policy,
        "--slo-ms",
        str(args.slo_ms),
        "--window-sec",
        str(args.window_sec),
        "--memory-tiers-mb",
        args.memory_tiers_mb,
        "--base-memory-mb",
        str(args.base_memory_mb),
        "--cpu-alpha",
        str(args.cpu_alpha),
        "--overhead-alpha",
        str(args.overhead_alpha),
        "--platform-keepalive-sec",
        str(args.platform_keepalive_sec),
        "--memory-mode",
        args.duet_memory_mode,
        "--critical-slack-ratio",
        str(critical),
        "--warm-blend-beta",
        str(beta),
        "--warm-scale-multiplier",
        str(scale),
        "--min-keepalive-sec",
        str(args.duet_min_keepalive_sec),
        "--uncertainty-gain-sec",
        str(unc),
        "--persistence-gain-sec",
        str(pers),
        "--critical-bonus-sec",
        str(crit_bonus),
        "--keepalive-demand-floor",
        str(floor),
        "--zero-demand-threshold",
        str(args.duet_zero_demand_threshold),
        "--warmup-mode",
        args.duet_warmup_mode,
        "--max-plan-windows",
        str(args.max_plan_windows),
        "--out-dir",
        str(duet_dir),
    ]
    use_entry_kernel = args.entry_forecast is not None or args.delay_kernel is not None
    if use_entry_kernel:
        command.extend(
            [
                "--entry-forecast",
                str(resolve_path(root, args.entry_forecast)),
                "--delay-kernel",
                str(resolve_path(root, args.delay_kernel)),
            ]
        )
    else:
        command.extend(["--forecast-detail", str(resolve_path(root, args.forecast_detail))])
    if args.fold_id is not None:
        command.extend(["--fold-id", str(args.fold_id)])
    maybe_run(command, root, duet_dir / "duet_control_plan.json", args.skip_existing)
    return duet_dir


def evaluate(args: argparse.Namespace, root: Path, out_dir: Path, *, plan_path: Path,
             label: str, warmup_mode_override: str | None = None,
             forecast_detail_override: Path | None = None,
             method_override: str | None = None) -> Path:
    eval_dir = out_dir / "stage4_eval" / label
    warmup_mode = warmup_mode_override or args.warmup_mode
    forecast_detail = forecast_detail_override
    if forecast_detail is None and args.forecast_detail is not None:
        forecast_detail = resolve_path(root, args.forecast_detail)
    if forecast_detail is None:
        raise ValueError(f"no forecast detail available for Stage4 evaluation of {label}")
    command = [
        sys.executable,
        "-m",
        "runner.stage4_risk.estimate_slo_risk",
        "--workflow-config",
        str(resolve_path(root, args.workflow_config)),
        "--trace",
        str(resolve_path(root, args.trace)),
        "--forecast-detail",
        str(forecast_detail),
        "--latency-samples",
        str(resolve_path(root, args.latency_samples)),
        "--control-plan",
        str(plan_path),
        "--method",
        method_override or args.method,
        "--policy",
        args.policy,
        "--window-sec",
        str(args.window_sec),
        "--slo-ms",
        str(args.slo_ms),
        "--simulations-per-request",
        str(args.simulations_per_request),
        "--residual-cold-probability",
        str(args.residual_cold_probability),
        "--enable-memory-scaling",
        "--base-memory-mb",
        str(args.base_memory_mb),
        "--cpu-alpha",
        str(args.cpu_alpha),
        "--overhead-alpha",
        str(args.overhead_alpha),
        "--warmup-mode",
        warmup_mode,
        "--out-dir",
        str(eval_dir),
    ]
    if args.fold_id is not None:
        command.extend(["--fold-id", str(args.fold_id)])
    maybe_run(command, root, eval_dir / "risk_summary.csv", args.skip_existing)
    return eval_dir


def read_risk_summary(eval_dir: Path) -> dict[str, Any]:
    summary_path = eval_dir / "risk_summary.csv"
    if not summary_path.exists():
        return {}
    frame = pd.read_csv(summary_path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def read_baseline_costs(baseline_dir: Path) -> dict[str, dict[str, Any]]:
    summary = baseline_dir / "baseline_plan_summary.csv"
    if not summary.exists():
        return {}
    frame = pd.read_csv(summary)
    return {str(row["baseline"]): dict(row) for row in frame.to_dict(orient="records")}


def read_pareto_cost(pareto_dir: Path) -> dict[str, Any]:
    selected = pareto_dir / "selected_candidate.json"
    if not selected.exists():
        return {}
    payload = json.loads(selected.read_text(encoding="utf-8"))
    return {
        "total_gb_seconds": payload.get("cost_gb_seconds"),
        "execution_gb_seconds": payload.get("execution_gb_seconds"),
        "warm_gb_seconds": payload.get("warm_gb_seconds"),
        "mean_warm_count": payload.get("mean_warm_count"),
        "max_warm_count": payload.get("max_warm_count"),
        "mean_keepalive_ttl_sec": payload.get("mean_keepalive_ttl_sec"),
        "mean_memory_mb": payload.get("mean_memory_mb"),
    }


def read_duet_cost(duet_dir: Path) -> dict[str, Any]:
    summary = duet_dir / "duet_summary.json"
    if not summary.exists():
        return {}
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return {
        "total_gb_seconds": payload.get("total_gb_seconds"),
        "execution_gb_seconds": payload.get("execution_gb_seconds"),
        "warm_gb_seconds": payload.get("warm_gb_seconds"),
        "mean_warm_count": payload.get("mean_warm_count"),
        "max_warm_count": payload.get("max_warm_count"),
        "mean_keepalive_ttl_sec": payload.get("mean_keepalive_ttl_sec"),
        "mean_memory_mb": payload.get("mean_memory_mb"),
    }


METHOD_LABELS = {
    "cold_every_time": "Cold-Every-Time",
    "platform_default": "Platform-Default",
    "always_warm": "Always-Warm",
    "orion_style": "ORION-Style",
    "stepconf_style": "StepConf-Style",
    "scale_to_zero": "Scale-To-Zero",
    "smiless_pareto": "SMIless-Pareto",
    "duet_balanced": "DUET-Balanced (ours)",
    "duet_economy": "DUET-Economy (ours)",
}


def aggregate(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    eval_dir = out_dir / "stage4_eval"
    baseline_costs = read_baseline_costs(out_dir / "stage5_baselines")
    pareto_cost = read_pareto_cost(out_dir / "stage5_smiless_pareto")
    duet_balanced_cost = read_duet_cost(out_dir / "stage5_duet_balanced")
    duet_economy_cost = read_duet_cost(out_dir / "stage5_duet_economy")

    rows: list[dict[str, Any]] = []
    for baseline_name, cost in baseline_costs.items():
        risk = read_risk_summary(eval_dir / baseline_name)
        rows.append(_build_row(args, baseline_name, METHOD_LABELS.get(baseline_name, baseline_name), cost, risk))
    if pareto_cost:
        risk = read_risk_summary(eval_dir / "smiless_pareto")
        rows.append(_build_row(args, "smiless_pareto", METHOD_LABELS["smiless_pareto"], pareto_cost, risk))
    if duet_balanced_cost:
        risk = read_risk_summary(eval_dir / "duet_balanced")
        rows.append(_build_row(args, "duet_balanced", METHOD_LABELS["duet_balanced"], duet_balanced_cost, risk))
    if duet_economy_cost:
        risk = read_risk_summary(eval_dir / "duet_economy")
        rows.append(_build_row(args, "duet_economy", METHOD_LABELS["duet_economy"], duet_economy_cost, risk))
    frame = pd.DataFrame(rows)
    return frame


def _build_row(args: argparse.Namespace, method: str, label: str, cost: dict[str, Any],
               risk: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": method,
        "method_label": label,
        "policy": args.policy,
        "slo_ms": args.slo_ms,
        "warmup_mode": args.warmup_mode,
        "cost_gb_seconds": cost.get("total_gb_seconds"),
        "execution_gb_seconds": cost.get("execution_gb_seconds"),
        "warm_gb_seconds": cost.get("warm_gb_seconds"),
        "mean_warm_count": cost.get("mean_warm_count"),
        "max_warm_count": cost.get("max_warm_count"),
        "mean_keepalive_ttl_sec": cost.get("mean_keepalive_ttl_sec"),
        "mean_memory_mb": cost.get("mean_memory_mb"),
        "predicted_violation_probability": risk.get("predicted_violation_probability"),
        "predicted_latency_p50_ms": risk.get("predicted_latency_p50_ms"),
        "predicted_latency_p90_ms": risk.get("predicted_latency_p90_ms"),
        "predicted_latency_p95_ms": risk.get("predicted_latency_p95_ms"),
        "observed_violation_rate": risk.get("observed_violation_rate"),
        "mean_cold_like_stages_per_sim": risk.get("mean_cold_like_stages_per_sim"),
        "requests": risk.get("requests"),
        "simulation_rows": risk.get("simulation_rows"),
    }


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

    duet_balanced_dir = out_dir / "stage5_duet_balanced"
    duet_economy_dir = out_dir / "stage5_duet_economy"
    if not args.baselines_only:
        baseline_dir = gen_baselines(args, root, out_dir)
        pareto_dir = gen_pareto(args, root, out_dir)
        duet_balanced_dir = gen_duet(args, root, out_dir, variant="balanced")
        if args.enable_duet_economy:
            duet_economy_dir = gen_duet(args, root, out_dir, variant="economy")
    else:
        baseline_dir = out_dir / "stage5_baselines"
        pareto_dir = out_dir / "stage5_smiless_pareto"

    baseline_costs = read_baseline_costs(baseline_dir)
    for baseline_name, cost in baseline_costs.items():
        plan_path = Path(cost["control_plan_json"])
        if not plan_path.is_absolute():
            plan_path = root / plan_path
        if not plan_path.exists():
            plan_path = baseline_dir / f"{baseline_name}_control_plan.json"
        warmup_mode = "window" if baseline_name == "always_warm" else args.warmup_mode
        forecast_override = (
            baseline_dir / f"{baseline_name}_propagated_stage_forecast.csv"
            if use_entry_kernel
            else None
        )
        evaluate(args, root, out_dir, plan_path=plan_path, label=baseline_name,
                 warmup_mode_override=warmup_mode,
                 forecast_detail_override=forecast_override if forecast_override and forecast_override.exists() else None,
                 method_override="entry-kernel" if use_entry_kernel else None)

    pareto_plan = pareto_dir / "selected_control_plan.json"
    if pareto_plan.exists():
        forecast_override = pareto_dir / "selected_propagated_stage_forecast.csv"
        evaluate(
            args,
            root,
            out_dir,
            plan_path=pareto_plan,
            label="smiless_pareto",
            forecast_detail_override=forecast_override if use_entry_kernel and forecast_override.exists() else None,
            method_override="entry-kernel" if use_entry_kernel else None,
        )

    duet_balanced_plan = duet_balanced_dir / "duet_control_plan.json"
    if duet_balanced_plan.exists():
        forecast_override = duet_balanced_dir / "duet_propagated_stage_forecast.csv"
        evaluate(args, root, out_dir, plan_path=duet_balanced_plan,
                 label="duet_balanced",
                 warmup_mode_override=args.duet_warmup_mode,
                 forecast_detail_override=forecast_override if use_entry_kernel and forecast_override.exists() else None,
                 method_override="entry-kernel" if use_entry_kernel else None)

    duet_economy_plan = duet_economy_dir / "duet_control_plan.json"
    if duet_economy_plan.exists():
        forecast_override = duet_economy_dir / "duet_propagated_stage_forecast.csv"
        evaluate(args, root, out_dir, plan_path=duet_economy_plan,
                 label="duet_economy",
                 warmup_mode_override=args.duet_warmup_mode,
                 forecast_detail_override=forecast_override if use_entry_kernel and forecast_override.exists() else None,
                 method_override="entry-kernel" if use_entry_kernel else None)

    frame = aggregate(args, out_dir)
    frame = frame.sort_values("cost_gb_seconds")
    out_csv = out_dir / "comparison_summary.csv"
    frame.to_csv(out_csv, index=False)

    print(f"\nwrote {out_csv}\n")
    show_cols = [
        "method_label",
        "cost_gb_seconds",
        "warm_gb_seconds",
        "mean_warm_count",
        "mean_keepalive_ttl_sec",
        "mean_memory_mb",
        "predicted_violation_probability",
        "predicted_latency_p95_ms",
    ]
    present_cols = [c for c in show_cols if c in frame.columns]
    print(frame[present_cols].to_string(index=False))


if __name__ == "__main__":
    main()
