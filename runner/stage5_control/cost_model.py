from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .control_plan import ControlPlan, expand_control_plan, load_control_plan
from .dag_warmup_scheduler import cold_overhead_lead_times_ms, warm_interval_start_sec
from ..workflow import WorkflowSpec, load_workflow


@dataclass
class CostBreakdown:
    total_gb_seconds: float
    execution_gb_seconds: float
    warm_gb_seconds: float
    reconfiguration_penalty: float
    expanded_plan: pd.DataFrame
    execution_detail: pd.DataFrame


def _mean_action_duration_sec(latency_samples: pd.DataFrame | None) -> dict[str, float]:
    if latency_samples is None or latency_samples.empty:
        return {}
    if "action_duration_ms" not in latency_samples.columns:
        return {}

    by_stage = (
        latency_samples.groupby("stage_name")["action_duration_ms"]
        .mean()
        .div(1000.0)
        .to_dict()
    )
    global_mean = float(latency_samples["action_duration_ms"].mean()) / 1000.0
    by_stage["*"] = global_mean
    return by_stage


def _forecast_demand(
    forecast_detail: pd.DataFrame | None,
    *,
    demand_column: str,
) -> pd.DataFrame:
    if forecast_detail is None or forecast_detail.empty:
        return pd.DataFrame(columns=["stage_name", "window", "demand"])
    if "stage_name" not in forecast_detail.columns or "target_window" not in forecast_detail.columns:
        raise ValueError("forecast_detail must contain stage_name and target_window")

    column = demand_column
    if column not in forecast_detail.columns:
        if "forecast_count" in forecast_detail.columns:
            column = "forecast_count"
        elif "actual_count" in forecast_detail.columns:
            column = "actual_count"
        else:
            raise ValueError(
                f"forecast_detail does not contain {demand_column}, forecast_count, or actual_count"
            )

    return (
        forecast_detail.groupby(["stage_name", "target_window"], as_index=False)[column]
        .max()
        .rename(columns={"target_window": "window", column: "demand"})
    )


def _warm_cost_from_intervals(
    expanded_plan: pd.DataFrame,
    *,
    window_sec: float,
    demand: pd.DataFrame | None = None,
    warmup_mode: str = "window",
    workflow: WorkflowSpec | None = None,
    prewarm_lead_sec_by_stage: dict[str, float] | None = None,
) -> float:
    if expanded_plan.empty:
        return 0.0

    rows = expanded_plan.copy()
    if demand is not None and not demand.empty:
        rows = rows.merge(demand, on=["stage_name", "window"], how="left")
        rows["demand"] = rows["demand"].fillna(0.0)
    else:
        rows["demand"] = 0.0

    total = 0.0
    for stage_name, stage_rows in rows.groupby("stage_name"):
        events: list[tuple[float, float]] = []
        for row in stage_rows.to_dict(orient="records"):
            warm_count = max(0.0, float(row["warm_count"]))
            memory_gb = max(0.0, float(row["memory_mb"]) / 1024.0)
            window_start = float(row["window"]) * window_sec
            window_end = (float(row["window"]) + 1.0) * window_sec
            lead_sec = 0.0
            if prewarm_lead_sec_by_stage:
                lead_sec = float(
                    prewarm_lead_sec_by_stage.get(
                        str(stage_name),
                        prewarm_lead_sec_by_stage.get("*", 0.0),
                    )
                )
            start = warm_interval_start_sec(
                warmup_mode=warmup_mode,
                workflow=workflow,
                stage_name=str(stage_name),
                window_start_sec=window_start,
                window_end_sec=window_end,
                lead_time_sec=lead_sec,
            )
            keepalive_ttl = max(0.0, float(row["keepalive_ttl_sec"]))
            end = window_end + keepalive_ttl
            if warm_count > 0:
                weighted_capacity = warm_count * memory_gb
                events.append((start, weighted_capacity))
                events.append((end, -weighted_capacity))

            # Invocations that were not explicitly prewarmed can still leave warm
            # containers behind. If keep-alive retains them, charge their idle tail.
            demand_count = max(0.0, float(row.get("demand", 0.0) or 0.0))
            extra_invocation_born = max(0.0, demand_count - warm_count)
            if keepalive_ttl > 0 and extra_invocation_born > 0:
                weighted_capacity = extra_invocation_born * memory_gb
                events.append((window_end, weighted_capacity))
                events.append((end, -weighted_capacity))

        if not events:
            continue
        events.sort(key=lambda item: (item[0], -item[1]))
        active_capacity = 0.0
        previous_time = events[0][0]
        for event_time, delta in events:
            if event_time > previous_time and active_capacity > 0:
                total += active_capacity * (event_time - previous_time)
            active_capacity = max(0.0, active_capacity + delta)
            previous_time = event_time
    return total


def _execution_cost(
    expanded_plan: pd.DataFrame,
    demand: pd.DataFrame,
    action_duration_sec: dict[str, float],
    *,
    base_memory_mb: int,
    cpu_alpha: float,
) -> tuple[float, pd.DataFrame]:
    if expanded_plan.empty or demand.empty:
        return 0.0, pd.DataFrame()

    merged = demand.merge(expanded_plan, on=["stage_name", "window"], how="left")
    if merged["memory_mb"].isna().any():
        missing = merged.loc[merged["memory_mb"].isna(), ["stage_name", "window"]]
        raise ValueError(
            "control plan is missing stage/window entries for execution demand: "
            f"{missing.head(10).to_dict(orient='records')}"
        )

    durations = []
    for row in merged.to_dict(orient="records"):
        base_duration = action_duration_sec.get(row["stage_name"], action_duration_sec.get("*", 0.0))
        ratio = max(1.0, float(row["memory_mb"])) / max(1.0, float(base_memory_mb))
        durations.append(base_duration / (ratio ** float(cpu_alpha)))
    merged["mean_action_duration_sec"] = durations
    merged["execution_gb_seconds"] = (
        merged["demand"].astype(float)
        * (merged["memory_mb"].astype(float) / 1024.0)
        * merged["mean_action_duration_sec"].astype(float)
    )
    return float(merged["execution_gb_seconds"].sum()), merged


def estimate_control_plan_cost(
    plan: ControlPlan,
    *,
    forecast_detail: pd.DataFrame | None = None,
    latency_samples: pd.DataFrame | None = None,
    workflow_name: str | None = None,
    workflow: WorkflowSpec | None = None,
    window_sec: float | None = None,
    demand_column: str = "forecast_count",
    reconfiguration_penalty: float = 0.0,
    base_memory_mb: int = 256,
    cpu_alpha: float = 1.0,
    warmup_mode: str = "window",
) -> CostBreakdown:
    effective_window_sec = float(window_sec or plan.window_sec)
    demand = _forecast_demand(forecast_detail, demand_column=demand_column)

    if not demand.empty:
        stages = sorted(demand["stage_name"].astype(str).unique())
        windows = sorted(demand["window"].astype(int).unique())
    else:
        concrete_rows = [row for row in plan.rows if row.window >= 0]
        stages = sorted({row.stage_name for row in plan.rows if row.stage_name != "*"})
        windows = sorted({row.window for row in concrete_rows}) or [0]

    expanded = expand_control_plan(plan, stages=stages, windows=windows)
    if workflow_name is not None and not expanded.empty:
        expanded["workflow_name"] = expanded["workflow_name"].fillna(workflow_name)

    action_duration_sec = _mean_action_duration_sec(latency_samples)
    execution_gb_seconds, execution_detail = _execution_cost(
        expanded,
        demand,
        action_duration_sec,
        base_memory_mb=base_memory_mb,
        cpu_alpha=cpu_alpha,
    )
    lead_ms_by_stage = cold_overhead_lead_times_ms(latency_samples, workflow_name)
    lead_sec_by_stage = {
        stage: float(value) / 1000.0 for stage, value in lead_ms_by_stage.items()
    }
    warm_gb_seconds = _warm_cost_from_intervals(
        expanded,
        window_sec=effective_window_sec,
        demand=demand,
        warmup_mode=warmup_mode,
        workflow=workflow,
        prewarm_lead_sec_by_stage=lead_sec_by_stage,
    )
    total = execution_gb_seconds + warm_gb_seconds + float(reconfiguration_penalty)
    return CostBreakdown(
        total_gb_seconds=total,
        execution_gb_seconds=execution_gb_seconds,
        warm_gb_seconds=warm_gb_seconds,
        reconfiguration_penalty=float(reconfiguration_penalty),
        expanded_plan=expanded,
        execution_detail=execution_detail,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate GB-second proxy cost for a control plan.")
    parser.add_argument("--control-plan", required=True)
    parser.add_argument("--workflow-config", default=None)
    parser.add_argument("--forecast-detail", default=None)
    parser.add_argument("--latency-samples", default=None)
    parser.add_argument("--workflow-name", default=None)
    parser.add_argument("--window-sec", type=float, default=None)
    parser.add_argument("--demand-column", default="forecast_count")
    parser.add_argument("--reconfiguration-penalty", type=float, default=0.0)
    parser.add_argument("--base-memory-mb", type=int, default=256)
    parser.add_argument("--cpu-alpha", type=float, default=1.0)
    parser.add_argument("--warmup-mode", choices=["window", "dag_jit"], default="window")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_control_plan(args.control_plan, default_window_sec=args.window_sec or 5.0)
    forecast_detail = (
        pd.read_csv(args.forecast_detail) if args.forecast_detail is not None else None
    )
    latency_samples = (
        pd.read_csv(args.latency_samples) if args.latency_samples is not None else None
    )
    workflow = load_workflow(args.workflow_config) if args.workflow_config is not None else None
    breakdown = estimate_control_plan_cost(
        plan,
        forecast_detail=forecast_detail,
        latency_samples=latency_samples,
        workflow_name=args.workflow_name,
        workflow=workflow,
        window_sec=args.window_sec,
        demand_column=args.demand_column,
        reconfiguration_penalty=args.reconfiguration_penalty,
        base_memory_mb=args.base_memory_mb,
        cpu_alpha=args.cpu_alpha,
        warmup_mode=args.warmup_mode,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "total_gb_seconds": breakdown.total_gb_seconds,
                "execution_gb_seconds": breakdown.execution_gb_seconds,
                "warm_gb_seconds": breakdown.warm_gb_seconds,
                "reconfiguration_penalty": breakdown.reconfiguration_penalty,
            }
        ]
    ).to_csv(out_dir / "plan_cost_summary.csv", index=False)
    breakdown.expanded_plan.to_csv(out_dir / "expanded_control_plan.csv", index=False)
    breakdown.execution_detail.to_csv(out_dir / "plan_execution_cost_detail.csv", index=False)
    print(f"wrote plan cost report to {out_dir}")


if __name__ == "__main__":
    main()
