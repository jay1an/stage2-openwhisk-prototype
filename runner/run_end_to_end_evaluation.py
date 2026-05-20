from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .workflow import load_workflow


BASELINE_LABELS = {
    "cold_every_time": "cold-every-time",
    "platform_default": "platform-default",
    "always_warm": "always-warm",
    "orion_style": "ORION-style",
    "stepconf_style": "StepConf-style",
}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    workflow: str
    regime: str
    workflow_config: str
    trace: str
    forecast_detail: str
    latency_samples: str
    method: str
    policy: str
    slo_ms: float
    fold_id: int | None = None
    window_sec: float = 5.0
    risk_budget: float = 0.05
    residual_cold_probability: float = 0.0
    memory_tiers_mb: str = "128,256,512,1024"
    base_memory_mb: int = 256
    cpu_alpha: float = 1.0
    overhead_alpha: float = 0.08
    platform_keepalive_sec: float = 20.0
    orion_downstream_warm_discount: float = 0.7
    warmup_mode: str = "window"
    planner_demand_column: str = "forecast_count"
    schedule: str | None = None
    notes: str = ""


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def default_experiments() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            name="sebs_video_periodic_drift_p95",
            workflow="sebs_video",
            regime="azure_periodic_drift_scaled30",
            workflow_config="configs/sebs_video.yaml",
            trace="data/stage_synthetic/sebs_video_azure_periodic_drift_challenge_scaled30_stage_trace.csv",
            schedule="data/azure_schedules_challenge/schedule_azure_periodic_drift_challenge_scaled30_sebs_video.csv",
            forecast_detail="reports/online_adaptive_selector_azure_periodic_drift_scaled30_riskbudget/online_selected_detail.csv",
            latency_samples="reports/stage3_latency_augmented_cold_sebs_video_periodic_drift/latency_samples_for_monte_carlo_augmented.csv",
            method="online-adaptive-expert-bank",
            policy="p95",
            fold_id=3,
            slo_ms=2500.0,
            residual_cold_probability=0.01,
            notes=(
                "Uses the current Stage-2 risk-budget selector and augmented "
                "sebs_video latency pool."
            ),
        ),
        ExperimentSpec(
            name="civic_alert_periodic_drift_p95",
            workflow="civic_alert_flow",
            regime="profiled_periodic_drift",
            workflow_config="configs/civic_alert_flow.yaml",
            trace="data/stage_synthetic/civic_alert_flow_profiled_periodic_drift_stage_trace.csv",
            forecast_detail="reports/civic_alert_stage2_hazard/civic_alert_flow_hazard-hurdle_stage_compare_detail.csv",
            latency_samples="reports/stage3_civic_alert_profiled/latency_samples_for_monte_carlo.csv",
            method="dag-hazard-hurdle",
            policy="p95",
            slo_ms=4000.0,
            residual_cold_probability=0.0,
            notes=(
                "Uses the civic-alert DAG prototype. This is broader than SeBS "
                "but still an offline synthetic-stage evaluation."
            ),
        ),
        ExperimentSpec(
            name="spoken_dialog_periodic_drift_p95",
            workflow="spoken_dialog_flow",
            regime="profiled_periodic_drift",
            workflow_config="configs/spoken_dialog_flow.yaml",
            trace="data/stage_synthetic/spoken_dialog_flow_profiled_periodic_drift_stage_trace.csv",
            schedule="data/azure_schedules_challenge/schedule_azure_periodic_drift_challenge_scaled30_spoken_dialog_flow.csv",
            forecast_detail="reports/spoken_dialog_stage2_hazard/spoken_dialog_flow_hazard-hurdle_stage_compare_detail.csv",
            latency_samples="reports/stage3_spoken_dialog_profiled/latency_samples_for_monte_carlo.csv",
            method="dag-hazard-hurdle",
            policy="p95",
            slo_ms=3000.0,
            residual_cold_probability=0.0,
            notes=(
                "Profiled synthetic spoken-dialog workflow driven by the Azure "
                "periodic-drift entry schedule."
            ),
        ),
        ExperimentSpec(
            name="visual_qa_periodic_drift_p95",
            workflow="visual_qa_flow",
            regime="profiled_periodic_drift",
            workflow_config="configs/visual_qa_flow.yaml",
            trace="data/stage_synthetic/visual_qa_flow_profiled_periodic_drift_stage_trace.csv",
            schedule="data/azure_schedules_challenge/schedule_azure_periodic_drift_challenge_scaled30_visual_qa_flow.csv",
            forecast_detail="reports/visual_qa_stage2_hazard/visual_qa_flow_hazard-hurdle_stage_compare_detail.csv",
            latency_samples="reports/stage3_visual_qa_profiled/latency_samples_for_monte_carlo.csv",
            method="dag-hazard-hurdle",
            policy="p95",
            slo_ms=2300.0,
            residual_cold_probability=0.0,
            notes=(
                "Profiled synthetic visual-QA workflow driven by the Azure "
                "periodic-drift entry schedule."
            ),
        ),
    ]


def load_manifest(path: Path) -> list[ExperimentSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    experiments = raw.get("experiments", raw if isinstance(raw, list) else [])
    specs: list[ExperimentSpec] = []
    for item in experiments:
        specs.append(
            ExperimentSpec(
                name=str(item["name"]),
                workflow=str(item["workflow"]),
                regime=str(item["regime"]),
                workflow_config=str(item["workflow_config"]),
                trace=str(item["trace"]),
                forecast_detail=str(item["forecast_detail"]),
                latency_samples=str(item["latency_samples"]),
                method=str(item["method"]),
                policy=str(item.get("policy", "p95")),
                fold_id=(
                    None
                    if item.get("fold_id") in (None, "", "all")
                    else int(item["fold_id"])
                ),
                slo_ms=float(item["slo_ms"]),
                window_sec=float(item.get("window_sec", 5.0)),
                risk_budget=float(item.get("risk_budget", 0.05)),
                residual_cold_probability=float(
                    item.get("residual_cold_probability", 0.0)
                ),
                memory_tiers_mb=str(item.get("memory_tiers_mb", "128,256,512,1024")),
                base_memory_mb=int(item.get("base_memory_mb", 256)),
                cpu_alpha=float(item.get("cpu_alpha", 1.0)),
                overhead_alpha=float(item.get("overhead_alpha", 0.08)),
                platform_keepalive_sec=float(item.get("platform_keepalive_sec", 20.0)),
                orion_downstream_warm_discount=float(
                    item.get("orion_downstream_warm_discount", 0.7)
                ),
                warmup_mode=str(item.get("warmup_mode", "window")),
                planner_demand_column=str(
                    item.get("planner_demand_column", "forecast_count")
                ),
                schedule=item.get("schedule"),
                notes=str(item.get("notes", "")),
            )
        )
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an offline end-to-end evaluation by composing Stage 2 forecast "
            "details, Stage 3 latency samples, Stage 4 risk estimation, and "
            "Stage 5 control planning."
        )
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="JSON file with an experiments array. Omit to use the built-in available prototypes.",
    )
    parser.add_argument("--out-dir", default="reports/end_to_end_evaluation")
    parser.add_argument("--only-workflow", default=None)
    parser.add_argument("--only-regime", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Fail instead of recording skipped experiments when required inputs are missing.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--risk-simulations-per-request", type=int, default=20)
    parser.add_argument("--planner-simulations-per-window", type=int, default=20)
    parser.add_argument(
        "--planner-risk-budget-scale",
        type=float,
        default=1.0,
        help=(
            "Safety factor applied only inside the Pareto planner. For example, "
            "0.4 plans against 40% of the manifest risk budget, while Stage 4 "
            "still reports the full plan-conditioned violation probability."
        ),
    )
    parser.add_argument(
        "--planner-max-eval-windows",
        type=int,
        default=64,
        help=(
            "Maximum active windows sampled inside the Pareto planner risk model. "
            "Stage 4 still evaluates the selected plan over the requested trace."
        ),
    )
    parser.add_argument(
        "--planner-max-virtual-requests-per-window",
        type=int,
        default=16,
        help=(
            "Maximum synthetic requests per active window inside the Pareto planner "
            "risk model."
        ),
    )
    parser.add_argument(
        "--active-gate-threshold",
        type=float,
        default=0.3,
        help="Planner P(active) threshold below which a window receives no prewarm capacity.",
    )
    parser.add_argument("--max-plan-windows", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument(
        "--warmup-mode",
        choices=["manifest", "window", "dag_jit"],
        default="manifest",
        help="Override manifest warmup_mode for Stage 4/5 timing; manifest preserves per-experiment settings.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use a lightweight configuration for wiring checks. The resulting "
            "numbers are not paper evidence."
        ),
    )
    return parser.parse_args()


def run_command(
    command: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    skip_existing_path: Path | None = None,
    skip_existing: bool = False,
) -> bool:
    if skip_existing and skip_existing_path is not None and skip_existing_path.exists():
        print(f"[skip] {skip_existing_path}")
        return False
    print("[run] " + " ".join(command))
    if dry_run:
        return False
    subprocess.run(command, cwd=str(cwd), check=True)
    return True


def common_stage_args(spec: ExperimentSpec, root: Path) -> list[str]:
    args = [
        "--workflow-config",
        str(resolve_path(root, spec.workflow_config)),
        "--forecast-detail",
        str(resolve_path(root, spec.forecast_detail)),
        "--latency-samples",
        str(resolve_path(root, spec.latency_samples)),
        "--method",
        spec.method,
        "--policy",
        spec.policy,
        "--window-sec",
        str(spec.window_sec),
        "--slo-ms",
        str(spec.slo_ms),
        "--memory-tiers-mb",
        spec.memory_tiers_mb,
        "--base-memory-mb",
        str(spec.base_memory_mb),
        "--cpu-alpha",
        str(spec.cpu_alpha),
        "--overhead-alpha",
        str(spec.overhead_alpha),
        "--platform-keepalive-sec",
        str(spec.platform_keepalive_sec),
    ]
    warmup_mode = spec.warmup_mode
    args.extend(["--warmup-mode", warmup_mode])
    if spec.fold_id is not None:
        args.extend(["--fold-id", str(spec.fold_id)])
    return args


def ensure_inputs(spec: ExperimentSpec, root: Path) -> list[str]:
    required = {
        "workflow_config": spec.workflow_config,
        "trace": spec.trace,
        "forecast_detail": spec.forecast_detail,
        "latency_samples": spec.latency_samples,
    }
    missing = []
    for label, value in required.items():
        resolved = resolve_path(root, value)
        if resolved is None or not resolved.exists():
            missing.append(f"{label}={value}")
    if spec.schedule is not None:
        schedule = resolve_path(root, spec.schedule)
        if schedule is not None and not schedule.exists():
            missing.append(f"schedule={spec.schedule}")
    return missing


def scoped_eval_inputs(
    spec: ExperimentSpec,
    *,
    root: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    forecast_path = resolve_path(root, spec.forecast_detail)
    trace_path = resolve_path(root, spec.trace)
    assert forecast_path is not None
    assert trace_path is not None
    if args.max_plan_windows <= 0 or args.dry_run:
        return forecast_path, trace_path

    scope_dir = out_dir / spec.name / "scoped_inputs"
    scope_dir.mkdir(parents=True, exist_ok=True)
    scoped_forecast = scope_dir / f"forecast_first_{args.max_plan_windows}_windows.csv"
    scoped_trace = scope_dir / f"trace_first_{args.max_plan_windows}_windows.csv"
    if args.skip_existing and scoped_forecast.exists() and scoped_trace.exists():
        return scoped_forecast, scoped_trace

    detail = pd.read_csv(forecast_path)
    if "target_window" not in detail.columns and "window" in detail.columns:
        detail = detail.copy()
        detail["target_window"] = detail["window"]

    selected = detail[
        (detail["workflow_name"].astype(str) == spec.workflow)
        & (detail["policy"].astype(str) == spec.policy)
    ].copy()
    if "method" in selected.columns:
        selected = selected[selected["method"].astype(str) == spec.method].copy()
    if spec.fold_id is not None and "fold_id" in selected.columns:
        selected = selected[pd.to_numeric(selected["fold_id"], errors="coerce") == spec.fold_id]
    trace = pd.read_csv(trace_path)
    window_ms = int(round(spec.window_sec * 1000.0))
    rows = trace[trace["stage_name"].astype(str) != "__entry__"].copy()
    rows["stage_window"] = (
        pd.to_numeric(rows["dispatch_start_ms"], errors="coerce") // window_ms
    ).astype("Int64")
    selected["target_window"] = pd.to_numeric(selected["target_window"], errors="coerce")
    selected = selected.dropna(subset=["target_window"]).copy()
    all_windows = sorted(selected["target_window"].astype(int).unique())
    windows = all_windows[: args.max_plan_windows]
    if len(all_windows) > args.max_plan_windows:
        request_windows = [
            set(int(value) for value in group["stage_window"].dropna().astype(int).unique())
            for _, group in rows.groupby("request_id")
        ]
        best_windows = windows
        best_score = -1
        for start_idx in range(0, len(all_windows) - args.max_plan_windows + 1):
            candidate = all_windows[start_idx : start_idx + args.max_plan_windows]
            candidate_set = set(candidate)
            score = sum(1 for stage_windows in request_windows if stage_windows and stage_windows <= candidate_set)
            if score > best_score:
                best_score = score
                best_windows = candidate
        windows = best_windows

    selected = selected[selected["target_window"].astype(int).isin(windows)].copy()
    selected.to_csv(scoped_forecast, index=False)

    window_set = set(windows)
    request_ids = set(
        str(request_id)
        for request_id, group in rows.groupby("request_id")
        if set(int(value) for value in group["stage_window"].dropna().astype(int).unique()) <= window_set
    )
    scoped = trace[trace["request_id"].astype(str).isin(request_ids)].copy()
    scoped.to_csv(scoped_trace, index=False)
    return scoped_forecast, scoped_trace


def generate_baseline_plans(
    spec: ExperimentSpec,
    *,
    root: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Path:
    baseline_dir = out_dir / spec.name / "stage5_baselines"
    command = [
        sys.executable,
        "-m",
        "runner.stage5_control.paper_baselines",
        *common_stage_args(spec, root),
        "--orion-downstream-warm-discount",
        str(spec.orion_downstream_warm_discount),
        "--max-plan-windows",
        str(args.max_plan_windows),
        "--out-dir",
        str(baseline_dir),
    ]
    run_command(
        command,
        cwd=root,
        dry_run=args.dry_run,
        skip_existing_path=baseline_dir / "baseline_plan_summary.csv",
        skip_existing=args.skip_existing,
    )
    return baseline_dir


def generate_pareto_plan(
    spec: ExperimentSpec,
    *,
    root: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Path:
    planner_dir = out_dir / spec.name / "stage5_smiless_pareto"
    command = [
        sys.executable,
        "-m",
        "runner.stage5_control.risk_budgeted_pareto_planner",
        *common_stage_args(spec, root),
        "--risk-budget",
        str(spec.risk_budget * args.planner_risk_budget_scale),
        "--residual-cold-probability",
        str(spec.residual_cold_probability),
        "--planner-demand-column",
        spec.planner_demand_column,
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
        str(planner_dir),
    ]
    run_command(
        command,
        cwd=root,
        dry_run=args.dry_run,
        skip_existing_path=planner_dir / "selected_control_plan.json",
        skip_existing=args.skip_existing,
    )
    return planner_dir


def evaluate_plan(
    spec: ExperimentSpec,
    *,
    root: Path,
    out_dir: Path,
    plan_path: Path,
    label: str,
    forecast_detail_path: Path,
    trace_path: Path,
    args: argparse.Namespace,
) -> Path:
    eval_dir = out_dir / spec.name / "stage4_eval" / label
    command = [
        sys.executable,
        "-m",
        "runner.stage4_risk.estimate_slo_risk",
        "--workflow-config",
        str(resolve_path(root, spec.workflow_config)),
        "--trace",
        str(trace_path),
        "--forecast-detail",
        str(forecast_detail_path),
        "--latency-samples",
        str(resolve_path(root, spec.latency_samples)),
        "--control-plan",
        str(plan_path),
        "--method",
        spec.method,
        "--policy",
        spec.policy,
        "--window-sec",
        str(spec.window_sec),
        "--slo-ms",
        str(spec.slo_ms),
        "--simulations-per-request",
        str(args.risk_simulations_per_request),
        "--residual-cold-probability",
        str(spec.residual_cold_probability),
        "--enable-memory-scaling",
        "--base-memory-mb",
        str(spec.base_memory_mb),
        "--cpu-alpha",
        str(spec.cpu_alpha),
        "--overhead-alpha",
        str(spec.overhead_alpha),
        "--warmup-mode",
        spec.warmup_mode,
        "--out-dir",
        str(eval_dir),
    ]
    if spec.fold_id is not None:
        command.extend(["--fold-id", str(spec.fold_id)])
    run_command(
        command,
        cwd=root,
        dry_run=args.dry_run,
        skip_existing_path=eval_dir / "risk_summary.csv",
        skip_existing=args.skip_existing,
    )
    return eval_dir


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cost_rows_from_baselines(path: Path) -> dict[str, dict[str, Any]]:
    summary_path = path / "baseline_plan_summary.csv"
    if not summary_path.exists():
        return {}
    frame = pd.read_csv(summary_path)
    return {str(row["baseline"]): row for row in frame.to_dict(orient="records")}


def cost_row_from_pareto(path: Path) -> dict[str, Any]:
    payload = read_json(path / "selected_candidate.json")
    if not payload:
        return {}
    return {
        "total_gb_seconds": payload.get("cost_gb_seconds"),
        "execution_gb_seconds": payload.get("execution_gb_seconds"),
        "warm_gb_seconds": payload.get("warm_gb_seconds"),
        "mean_warm_count": payload.get("mean_warm_count"),
        "max_warm_count": payload.get("max_warm_count"),
        "mean_keepalive_ttl_sec": payload.get("mean_keepalive_ttl_sec"),
        "mean_memory_mb": payload.get("mean_memory_mb"),
        "planner_internal_risk": payload.get("risk"),
        "planner_feasible": payload.get("feasible"),
        "planner_candidate_id": payload.get("candidate_id"),
    }


def read_risk_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "risk_summary.csv"
    if not summary_path.exists():
        return {}
    frame = pd.read_csv(summary_path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def metric_record(
    *,
    spec: ExperimentSpec,
    method: str,
    label: str,
    risk_dir: Path,
    cost: dict[str, Any],
    stage_count: int,
    root: Path,
) -> dict[str, Any]:
    risk = read_risk_summary(risk_dir)
    mean_cold = risk.get("mean_cold_like_stages_per_sim")
    cold_ratio = None
    if mean_cold is not None and pd.notna(mean_cold) and stage_count > 0:
        cold_ratio = float(mean_cold) / float(stage_count)

    return {
        "experiment": spec.name,
        "workflow": spec.workflow,
        "regime": spec.regime,
        "baseline_or_system": method,
        "method_label": label,
        "forecast_method": spec.method,
        "policy": spec.policy,
        "fold_id": "all" if spec.fold_id is None else spec.fold_id,
        "slo_ms": spec.slo_ms,
        "risk_budget": spec.risk_budget,
        "cost_gb_seconds": cost.get("total_gb_seconds"),
        "execution_gb_seconds": cost.get("execution_gb_seconds"),
        "warm_gb_seconds": cost.get("warm_gb_seconds"),
        "mean_warm_count": cost.get("mean_warm_count"),
        "max_warm_count": cost.get("max_warm_count"),
        "mean_keepalive_ttl_sec": cost.get("mean_keepalive_ttl_sec"),
        "mean_memory_mb": cost.get("mean_memory_mb"),
        "planner_internal_risk": cost.get("planner_internal_risk"),
        "planner_feasible": cost.get("planner_feasible"),
        "predicted_slo_violation_probability": risk.get(
            "predicted_violation_probability"
        ),
        "historical_observed_slo_violation_rate": risk.get("observed_violation_rate"),
        "observed_slo_violation_rate": risk.get("observed_violation_rate"),
        "predicted_latency_p50_ms": risk.get("predicted_latency_p50_ms"),
        "predicted_latency_p90_ms": risk.get("predicted_latency_p90_ms"),
        "predicted_latency_p95_ms": risk.get("predicted_latency_p95_ms"),
        "observed_latency_p50_ms": risk.get("observed_latency_p50_ms"),
        "observed_latency_p90_ms": risk.get("observed_latency_p90_ms"),
        "observed_latency_p95_ms": risk.get("observed_latency_p95_ms"),
        "mean_cold_like_stages_per_sim": mean_cold,
        "cold_start_ratio_proxy": cold_ratio,
        "requests": risk.get("requests"),
        "simulation_rows": risk.get("simulation_rows"),
        "workflow_config": str(resolve_path(root, spec.workflow_config)),
        "trace": str(resolve_path(root, spec.trace)),
        "schedule": str(resolve_path(root, spec.schedule)) if spec.schedule else "",
        "forecast_detail": str(resolve_path(root, spec.forecast_detail)),
        "latency_samples": str(resolve_path(root, spec.latency_samples)),
        "risk_report_dir": str(risk_dir),
        "what_if_metric_note": (
            "predicted_* metrics are plan-conditioned Monte Carlo what-if estimates; "
            "historical_observed_* metrics are raw trace references and are not "
            "plan-conditioned."
        ),
        "notes": spec.notes,
    }


def collect_experiment_metrics(
    spec: ExperimentSpec,
    *,
    root: Path,
    out_dir: Path,
    baseline_dir: Path,
    planner_dir: Path,
) -> list[dict[str, Any]]:
    workflow = load_workflow(str(resolve_path(root, spec.workflow_config)))
    stage_count = len(workflow.nodes)
    records: list[dict[str, Any]] = []

    baseline_costs = cost_rows_from_baselines(baseline_dir)
    for baseline, cost in baseline_costs.items():
        label = BASELINE_LABELS.get(baseline, baseline)
        risk_dir = out_dir / spec.name / "stage4_eval" / baseline
        records.append(
            metric_record(
                spec=spec,
                method=baseline,
                label=label,
                risk_dir=risk_dir,
                cost=cost,
                stage_count=stage_count,
                root=root,
            )
        )

    pareto_cost = cost_row_from_pareto(planner_dir)
    pareto_dir = out_dir / spec.name / "stage4_eval" / "smiless_pareto"
    records.append(
        metric_record(
            spec=spec,
            method="smiless_pareto",
            label="SMIless-Pareto",
            risk_dir=pareto_dir,
            cost=pareto_cost,
            stage_count=stage_count,
            root=root,
        )
    )
    return records


def run_experiment(
    spec: ExperimentSpec,
    *,
    root: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    missing = ensure_inputs(spec, root)
    if missing:
        message = {
            "experiment": spec.name,
            "workflow": spec.workflow,
            "regime": spec.regime,
            "status": "skipped_missing_input",
            "missing": "; ".join(missing),
        }
        if args.fail_on_missing:
            raise FileNotFoundError(message["missing"])
        return [message]

    eval_forecast_detail, eval_trace = scoped_eval_inputs(
        spec,
        root=root,
        out_dir=out_dir,
        args=args,
    )
    plan_spec = (
        replace(spec, forecast_detail=str(eval_forecast_detail))
        if args.max_plan_windows > 0
        else spec
    )

    baseline_dir = generate_baseline_plans(plan_spec, root=root, out_dir=out_dir, args=args)
    planner_dir = generate_pareto_plan(plan_spec, root=root, out_dir=out_dir, args=args)

    baseline_summary = baseline_dir / "baseline_plan_summary.csv"
    if baseline_summary.exists():
        baselines = pd.read_csv(baseline_summary)
        for row in baselines.to_dict(orient="records"):
            evaluate_plan(
                spec,
                root=root,
                out_dir=out_dir,
                plan_path=Path(row["control_plan_json"]),
                label=str(row["baseline"]),
                forecast_detail_path=eval_forecast_detail,
                trace_path=eval_trace,
                args=args,
            )

    pareto_plan = planner_dir / "selected_control_plan.json"
    if pareto_plan.exists():
        evaluate_plan(
            spec,
            root=root,
            out_dir=out_dir,
            plan_path=pareto_plan,
            label="smiless_pareto",
            forecast_detail_path=eval_forecast_detail,
            trace_path=eval_trace,
            args=args,
        )

    return collect_experiment_metrics(
        spec,
        root=root,
        out_dir=out_dir,
        baseline_dir=baseline_dir,
        planner_dir=planner_dir,
    )


def latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "--"
    try:
        if pd.isna(value):
            return "--"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_latex_table(metrics: pd.DataFrame, path: Path) -> None:
    usable = metrics[
        metrics.get("status", pd.Series(index=metrics.index, dtype=object)).isna()
    ].copy()
    columns = [
        "Workflow",
        "Regime",
        "Method",
        "Cost",
        "Pred. viol.",
        "Hist. viol.",
        "Pred. p95",
    ]
    lines = [
        "% Generated by runner.run_end_to_end_evaluation.",
        "% Offline prototype results; do not present as real OpenWhisk E2E evidence.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Offline end-to-end prototype comparison.}",
        "\\label{tab:offline-e2e-prototype}",
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for row in usable.to_dict(orient="records"):
        lines.append(
            " & ".join(
                [
                    latex_escape(row.get("workflow")),
                    latex_escape(row.get("regime")),
                    latex_escape(row.get("method_label")),
                    fmt(row.get("cost_gb_seconds"), 1),
                    fmt(row.get("predicted_slo_violation_probability"), 3),
                    fmt(row.get("historical_observed_slo_violation_rate"), 3),
                    fmt(row.get("predicted_latency_p95_ms"), 1),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(out_dir: Path, metrics: pd.DataFrame, args: argparse.Namespace) -> None:
    usable = metrics[
        metrics.get("status", pd.Series(index=metrics.index, dtype=object)).isna()
    ]
    skipped = metrics[
        metrics.get("status", pd.Series(index=metrics.index, dtype=object)).notna()
    ]
    lines = [
        "# Offline End-to-End Evaluation",
        "",
        "This report is generated by `python -m runner.run_end_to_end_evaluation`.",
        "",
        "It composes existing stage artifacts instead of claiming a live controller:",
        "",
        "1. Stage 2 forecast detail CSVs provide stage-level demand forecasts.",
        "2. Stage 3 latency sample pools provide warm/cold-like latency samples.",
        "3. Stage 5 generates baseline and SMIless-Pareto control plans.",
        "4. Stage 4 re-evaluates each plan with workflow-level Monte Carlo risk.",
        "",
        "The output is suitable for checking the pipeline shape and comparing "
        "offline prototype plans. It is not yet real OpenWhisk closed-loop evidence.",
        "",
        "`predicted_*` metrics are plan-conditioned what-if estimates from Stage 4. "
        "`historical_observed_*` metrics are raw trace references and should not be "
        "read as observed outcomes after applying each plan.",
        "",
        "## Outputs",
        "",
        "- `end_to_end_metrics.csv`: unified metric table.",
        "- `end_to_end_metrics_compact.csv`: compact columns for paper drafting.",
        "- `end_to_end_table.tex`: generated LaTeX table snippet.",
        "- `<experiment>/stage5_*`: generated control plans.",
        "- `<experiment>/stage4_eval/*`: Stage 4 risk reports per method.",
        "",
        "## Run Configuration",
        "",
        f"- risk simulations per request: `{args.risk_simulations_per_request}`",
        f"- planner simulations per window: `{args.planner_simulations_per_window}`",
        f"- planner risk budget scale: `{args.planner_risk_budget_scale}`",
        f"- planner max eval windows: `{args.planner_max_eval_windows}`",
        f"- planner max virtual requests per window: `{args.planner_max_virtual_requests_per_window}`",
        f"- active gate threshold: `{args.active_gate_threshold}`",
        f"- max plan windows: `{args.max_plan_windows}`",
        f"- warmup mode override: `{args.warmup_mode}`",
        f"- smoke mode: `{bool(args.smoke)}`",
        "",
        "## Current Coverage",
        "",
        f"- completed metric rows: `{len(usable)}`",
        f"- skipped rows: `{len(skipped)}`",
    ]
    if not skipped.empty:
        lines.extend(["", "## Skipped Inputs", ""])
        for row in skipped.to_dict(orient="records"):
            lines.append(
                f"- `{row.get('experiment')}`: {row.get('missing', row.get('status'))}"
            )
    lines.append("")
    out_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.risk_simulations_per_request = min(args.risk_simulations_per_request, 2)
        args.planner_simulations_per_window = min(args.planner_simulations_per_window, 4)
        if args.max_plan_windows == 0:
            args.max_plan_windows = 12
        args.beam_width = min(args.beam_width, 6)
        args.max_iterations = min(args.max_iterations, 1)

    root = project_root()
    out_dir = resolve_path(root, args.out_dir)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = (
        load_manifest(resolve_path(root, args.manifest))
        if args.manifest is not None
        else default_experiments()
    )
    if args.warmup_mode != "manifest":
        specs = [replace(spec, warmup_mode=args.warmup_mode) for spec in specs]
    if args.only_workflow:
        specs = [spec for spec in specs if spec.workflow == args.only_workflow]
    if args.only_regime:
        specs = [spec for spec in specs if spec.regime == args.only_regime]

    all_records: list[dict[str, Any]] = []
    for spec in specs:
        print(f"\n=== {spec.name} ({spec.workflow}, {spec.regime}) ===")
        all_records.extend(run_experiment(spec, root=root, out_dir=out_dir, args=args))

    metrics = pd.DataFrame(all_records)
    metrics.insert(0, "generated_at_utc", datetime.now(timezone.utc).isoformat())
    metrics.to_csv(out_dir / "end_to_end_metrics.csv", index=False)

    compact_columns = [
        col
        for col in [
            "workflow",
            "regime",
            "method_label",
            "cost_gb_seconds",
            "predicted_slo_violation_probability",
            "historical_observed_slo_violation_rate",
            "predicted_latency_p95_ms",
            "observed_latency_p95_ms",
            "cold_start_ratio_proxy",
            "mean_warm_count",
            "mean_memory_mb",
            "status",
            "missing",
        ]
        if col in metrics.columns
    ]
    metrics[compact_columns].to_csv(
        out_dir / "end_to_end_metrics_compact.csv", index=False
    )
    write_latex_table(metrics, out_dir / "end_to_end_table.tex")
    write_readme(out_dir, metrics, args)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "runner.run_end_to_end_evaluation",
        "manifest": args.manifest,
        "experiments": [spec.__dict__ for spec in specs],
        "smoke": bool(args.smoke),
        "risk_simulations_per_request": args.risk_simulations_per_request,
        "planner_simulations_per_window": args.planner_simulations_per_window,
        "planner_risk_budget_scale": args.planner_risk_budget_scale,
        "planner_max_eval_windows": args.planner_max_eval_windows,
        "planner_max_virtual_requests_per_window": args.planner_max_virtual_requests_per_window,
        "active_gate_threshold": args.active_gate_threshold,
        "max_plan_windows": args.max_plan_windows,
        "warmup_mode_override": args.warmup_mode,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nWrote {out_dir / 'end_to_end_metrics.csv'}")
    print(metrics[compact_columns].to_string(index=False))


if __name__ == "__main__":
    main()
