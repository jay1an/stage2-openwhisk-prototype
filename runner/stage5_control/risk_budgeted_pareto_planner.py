from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..stage4_risk.container_pool_cold_model import ContainerPoolColdModel
from ..workflow import WorkflowSpec, load_workflow
from .control_plan import ControlPlan, PlanRow, plan_to_frame, save_control_plan
from .cost_model import estimate_control_plan_cost
from .dag_warmup_scheduler import (
    cold_overhead_lead_times_ms,
    stage_lead_time_ms,
    warmup_timing_for_stage,
)
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


@dataclass
class CandidateEvaluation:
    candidate_id: str
    parent_id: str
    action: str
    risk: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    cost_gb_seconds: float
    execution_gb_seconds: float
    warm_gb_seconds: float
    mean_warm_count: float
    max_warm_count: float
    mean_keepalive_ttl_sec: float
    mean_memory_mb: float
    feasible: bool
    state: dict[str, dict[str, float]]
    plan: ControlPlan


class PlannerLatencyPools:
    def __init__(self, samples: pd.DataFrame, workflow_name: str, rng: np.random.Generator):
        self.workflow_name = workflow_name
        self.rng = rng
        rows = samples[(samples["workflow_name"] == workflow_name) & (samples["stage_name"] != "__entry__")].copy()
        for col in ["platform_overhead_ms", "action_duration_ms"]:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
        rows = rows.dropna(subset=["platform_overhead_ms", "action_duration_ms"])
        if rows.empty:
            raise ValueError(f"no latency samples found for workflow {workflow_name}")
        self.rows = rows
        self.by_stage_class: dict[tuple[str, str], pd.DataFrame] = {}
        self.by_stage: dict[str, pd.DataFrame] = {}
        for (stage, klass), group in rows.groupby(["stage_name", "latency_class"]):
            self.by_stage_class[(str(stage), str(klass))] = group.reset_index(drop=True)
        for stage, group in rows.groupby("stage_name"):
            self.by_stage[str(stage)] = group.reset_index(drop=True)
        self.global_rows = rows.reset_index(drop=True)
        self.global_cold_rows = rows[
            rows["latency_class"].astype(str).str.startswith("cold_like")
        ].reset_index(drop=True)
        if self.global_cold_rows.empty:
            self.global_cold_rows = self.global_rows
        self.prewarm_lead_ms_by_stage = cold_overhead_lead_times_ms(rows, workflow_name)
        self._mean_duration_cache: dict[tuple[str, int, int, float, float], float] = {}

    def _pool(self, stage_name: str, cold_like: bool) -> pd.DataFrame:
        if cold_like:
            for klass in ("cold_like", "cold_like_composed"):
                pool = self.by_stage_class.get((stage_name, klass))
                if pool is not None and not pool.empty:
                    return pool
            return self.global_cold_rows

        pool = self.by_stage_class.get((stage_name, "warm"))
        if pool is not None and not pool.empty:
            return pool
        stage_pool = self.by_stage.get(stage_name)
        if stage_pool is not None and not stage_pool.empty:
            return stage_pool
        warm_global = self.global_rows[self.global_rows["latency_class"].astype(str) == "warm"]
        return warm_global.reset_index(drop=True) if not warm_global.empty else self.global_rows

    def sample(
        self,
        stage_name: str,
        *,
        cold_like: bool,
        memory_mb: int,
        base_memory_mb: int,
        cpu_alpha: float,
        overhead_alpha: float,
    ) -> float:
        pool = self._pool(stage_name, cold_like)
        row = pool.iloc[int(self.rng.integers(0, len(pool)))]
        return adjusted_latency_ms(
            overhead_ms=float(row["platform_overhead_ms"]),
            action_ms=float(row["action_duration_ms"]),
            memory_mb=int(memory_mb),
            base_memory_mb=base_memory_mb,
            cpu_alpha=cpu_alpha,
            overhead_alpha=overhead_alpha,
        )

    def mean_duration_ms(
        self,
        stage_name: str,
        *,
        memory_mb: int,
        base_memory_mb: int,
        cpu_alpha: float,
        overhead_alpha: float,
    ) -> float:
        key = (
            stage_name,
            int(memory_mb),
            int(base_memory_mb),
            round(float(cpu_alpha), 6),
            round(float(overhead_alpha), 6),
        )
        cached = self._mean_duration_cache.get(key)
        if cached is not None:
            return cached
        pool = self._pool(stage_name, cold_like=False)
        values = [
            adjusted_latency_ms(
                overhead_ms=float(row["platform_overhead_ms"]),
                action_ms=float(row["action_duration_ms"]),
                memory_mb=int(memory_mb),
                base_memory_mb=base_memory_mb,
                cpu_alpha=cpu_alpha,
                overhead_alpha=overhead_alpha,
            )
            for _, row in pool.iterrows()
        ]
        value = float(np.mean(values)) if values else 1.0
        self._mean_duration_cache[key] = value
        return value

    def prewarm_lead_time_ms(self, stage_name: str) -> float:
        return stage_lead_time_ms(stage_name, self.prewarm_lead_ms_by_stage)


def parse_float_list(value: str) -> list[float]:
    items = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not items:
        raise ValueError("list must contain at least one value")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-5A risk-budgeted Pareto planner. It searches warm-count, "
            "keep-alive, and memory candidates, evaluates SLO risk, prunes dominated "
            "plans, and writes the lowest-cost feasible control plan."
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
    parser.add_argument("--risk-budget", type=float, default=0.05)
    parser.add_argument("--memory-tiers-mb", default="128,256,512,1024")
    parser.add_argument("--base-memory-mb", type=int, default=256)
    parser.add_argument("--warm-multipliers", default="0,0.5,1.0,1.25,1.5,2.0")
    parser.add_argument("--keepalive-options-sec", default="0,5,15,20,30,60")
    parser.add_argument("--active-gate-threshold", type=float, default=0.3)
    parser.add_argument("--max-virtual-requests-per-window", type=int, default=16)
    parser.add_argument(
        "--platform-keepalive-sec",
        type=float,
        default=20.0,
        help="scaled platform idle retention used for default-behavior planner seeds",
    )
    parser.add_argument(
        "--warmup-mode",
        choices=["window", "dag_jit"],
        default="window",
        help=(
            "window makes planned warm_count available at control-window start; "
            "dag_jit delays downstream planned warm_count until the cold-overhead lead time has elapsed"
        ),
    )
    parser.add_argument("--warm-source", choices=["allocated_count", "forecast_count"], default="allocated_count")
    parser.add_argument(
        "--planner-demand-column",
        choices=["forecast_count", "allocated_count", "actual_count"],
        default="forecast_count",
        help=(
            "Demand column used inside planner Monte Carlo. Use forecast_count for "
            "online simulation; use actual_count only for offline validation and "
            "calibration studies."
        ),
    )
    parser.add_argument("--cpu-alpha", type=float, default=1.0)
    parser.add_argument("--overhead-alpha", type=float, default=0.08)
    parser.add_argument("--residual-cold-probability", type=float, default=0.0)
    parser.add_argument("--simulations-per-window", type=int, default=40)
    parser.add_argument("--max-eval-windows", type=int, default=64)
    parser.add_argument("--max-plan-windows", type=int, default=0, help="0 means all selected forecast windows")
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--expand-top-stages", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
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

    for col in ["target_window", "forecast_count", "allocated_count", "actual_count", "p_active"]:
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
    for col in ["target_window", "forecast_count", "allocated_count"]:
        if col in selected.columns:
            selected[col] = pd.to_numeric(selected[col], errors="coerce")
    selected = selected.dropna(subset=["target_window", "forecast_count"]).copy()
    selected["target_window"] = selected["target_window"].astype(int)
    if max_plan_windows and max_plan_windows > 0:
        windows = sorted(selected["target_window"].unique())[:max_plan_windows]
        selected = selected[selected["target_window"].isin(windows)].copy()
    return selected


def stage_window_table(selected: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        col
        for col in ["forecast_count", "allocated_count", "actual_count", "p_active"]
        if col in selected.columns
    ]
    return (
        selected.groupby(["stage_name", "target_window"], as_index=False)[value_cols]
        .max()
        .reset_index(drop=True)
    )


def probability_value(value: Any, default: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if not math.isfinite(parsed):
        parsed = float(default)
    return min(1.0, max(0.0, parsed))


def initial_state(
    stages: list[str],
    *,
    warm_multiplier: float,
    keepalive_ttl_sec: float,
    memory_mb: int,
    min_warm_count: float = 0.0,
) -> dict[str, dict[str, float]]:
    return {
        "warm_multiplier": {stage: float(warm_multiplier) for stage in stages},
        "keepalive_ttl_sec": {stage: float(keepalive_ttl_sec) for stage in stages},
        "memory_mb": {stage: float(memory_mb) for stage in stages},
        "min_warm_count": {stage: float(min_warm_count) for stage in stages},
    }


def state_key(state: dict[str, dict[str, float]], stages: list[str]) -> tuple[tuple[str, float, float, float, float], ...]:
    return tuple(
        (
            stage,
            round(float(state["warm_multiplier"][stage]), 6),
            round(float(state["keepalive_ttl_sec"][stage]), 6),
            round(float(state["memory_mb"][stage]), 6),
            round(float(state.get("min_warm_count", {}).get(stage, 0.0)), 6),
        )
        for stage in stages
    )


def clone_state(state: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        "warm_multiplier": dict(state["warm_multiplier"]),
        "keepalive_ttl_sec": dict(state["keepalive_ttl_sec"]),
        "memory_mb": dict(state["memory_mb"]),
        "min_warm_count": dict(state.get("min_warm_count", {})),
    }


def next_larger(value: float, options: list[float]) -> float | None:
    for option in options:
        if option > value + 1e-9:
            return option
    return None


def warm_count_for_record(
    state: dict[str, dict[str, float]],
    *,
    stage: str,
    record: dict[str, Any],
    warm_source: str,
    active_gate_threshold: float,
) -> int:
    source_count = max(0.0, float(record.get(warm_source, 0.0) or 0.0))
    p_active = probability_value(record.get("p_active", 1.0), default=1.0)
    multiplier = float(state["warm_multiplier"][stage])
    if p_active < active_gate_threshold:
        warm_count = 0
    else:
        warm_count = int(math.ceil(source_count * multiplier)) if source_count > 0 else 0
    min_warm_count = float(state.get("min_warm_count", {}).get(stage, 0.0))
    if p_active >= active_gate_threshold and source_count <= 0 and min_warm_count > 0:
        warm_count = max(warm_count, int(math.ceil(min_warm_count)))
    return int(warm_count)


def candidate_selected_from_entry(
    state: dict[str, dict[str, float]],
    *,
    entry_forecast: pd.DataFrame,
    delay_kernel: pd.DataFrame,
    workflow: WorkflowSpec,
    ordered_nodes: list[str],
    policy: str,
    warm_source: str,
    active_gate_threshold: float,
) -> pd.DataFrame:
    windows = sorted(entry_forecast["target_window"].astype(int).unique())
    prev_warm = {stage: False for stage in ordered_nodes}
    rows: list[dict[str, Any]] = []
    for window in windows:
        next_prev = dict(prev_warm)
        for stage in ordered_nodes:
            forecast_count = propagate_entry_to_stage(
                entry_forecast,
                delay_kernel,
                workflow_name=workflow.workflow_name,
                stage_name=stage,
                target_window=window,
                policy=policy,
                prev_warm=prev_warm.get(stage, False),
            )
            record = {
                "workflow_name": workflow.workflow_name,
                "method": "entry-kernel",
                "stage_name": stage,
                "target_window": int(window),
                "policy": policy,
                "forecast_count": float(forecast_count),
                "allocated_count": int(math.ceil(max(0.0, float(forecast_count)))),
                "p_active": 1.0,
            }
            warm_count = warm_count_for_record(
                state,
                stage=stage,
                record=record,
                warm_source=warm_source,
                active_gate_threshold=active_gate_threshold,
            )
            keepalive = float(state["keepalive_ttl_sec"][stage])
            next_prev[stage] = bool(warm_count > 0 or keepalive > 0.0)
            rows.append(record)
        prev_warm = next_prev
    return pd.DataFrame(rows)


def build_plan_from_state(
    state: dict[str, dict[str, float]],
    *,
    selected: pd.DataFrame,
    workflow_name: str,
    window_sec: float,
    warm_source: str,
    candidate_id: str,
    active_gate_threshold: float,
) -> ControlPlan:
    rows: list[PlanRow] = []
    table = stage_window_table(selected)
    for record in table.to_dict(orient="records"):
        stage = str(record["stage_name"])
        warm_count = warm_count_for_record(
            state,
            stage=stage,
            record=record,
            warm_source=warm_source,
            active_gate_threshold=active_gate_threshold,
        )
        keepalive = float(state["keepalive_ttl_sec"][stage])
        rows.append(
            PlanRow(
                workflow_name=workflow_name,
                stage_name=stage,
                window=int(record["target_window"]),
                warm_count=float(warm_count),
                keepalive_ttl_sec=keepalive,
                memory_mb=int(state["memory_mb"][stage]),
                source="risk_budgeted_pareto_planner",
                note=candidate_id,
            )
        )
    return ControlPlan(
        rows=rows,
        window_sec=window_sec,
        metadata={
            "workflow_name": workflow_name,
            "candidate_id": candidate_id,
            "warm_source": warm_source,
            "planner": "risk_budgeted_pareto",
        },
    )


def cold_probability_from_plan(
    *,
    forecast_count: float,
    warm_count: float,
    window_sec: float,
    mean_duration_ms: float,
    residual_cold_probability: float,
) -> float:
    demand = max(0.0, float(forecast_count))
    if demand <= 0.0:
        return residual_cold_probability
    reuse_factor = max(1.0, float(window_sec) * 1000.0 / max(1.0, float(mean_duration_ms)))
    effective_capacity = max(0.0, float(warm_count)) * reuse_factor
    deficit = max(0.0, demand - effective_capacity) / demand
    return min(1.0, max(float(residual_cold_probability), deficit))


def active_eval_windows(selected: pd.DataFrame, max_eval_windows: int) -> list[int]:
    value_cols = [col for col in ["forecast_count", "allocated_count", "actual_count"] if col in selected.columns]
    grouped = selected.groupby("target_window", as_index=False)[value_cols].max()
    grouped["activity"] = grouped[value_cols].max(axis=1)
    windows = sorted(int(value) for value in grouped[grouped["activity"] > 0]["target_window"].unique())
    if not windows:
        windows = sorted(int(value) for value in grouped["target_window"].unique())
    if max_eval_windows and max_eval_windows > 0 and len(windows) > max_eval_windows:
        indexes = np.linspace(0, len(windows) - 1, max_eval_windows).round().astype(int)
        windows = [windows[int(idx)] for idx in indexes]
    return windows


def evaluate_workflow_risk(
    *,
    plan: ControlPlan,
    selected: pd.DataFrame,
    workflow: WorkflowSpec,
    ordered_nodes: list[str],
    latency_pools: PlannerLatencyPools,
    rng: np.random.Generator,
    slo_ms: float,
    simulations_per_window: int,
    max_eval_windows: int,
    residual_cold_probability: float,
    base_memory_mb: int,
    cpu_alpha: float,
    overhead_alpha: float,
    demand_column: str,
    warmup_mode: str,
    max_virtual_requests_per_window: int,
) -> tuple[float, float, float, float]:
    if demand_column not in selected.columns:
        raise ValueError(f"planner demand column {demand_column} is not present in forecast detail")
    table_rows = stage_window_table(selected).to_dict(orient="records")
    demand_lookup = {
        (str(row["stage_name"]), int(row["target_window"])): max(0.0, float(row[demand_column]))
        for row in table_rows
    }
    p_active_lookup = {
        (str(row["stage_name"]), int(row["target_window"])): probability_value(
            row.get("p_active", 1.0),
            default=1.0,
        )
        for row in table_rows
    }
    windows = active_eval_windows(selected, max_eval_windows)
    latencies: list[float] = []
    violation_count = 0.0
    total = 0.0
    window_ms = float(plan.window_sec) * 1000.0
    max_virtual = max(1, int(max_virtual_requests_per_window))

    for _ in range(simulations_per_window):
        pool_state = {
            stage: ContainerPoolColdModel()
            for stage in ordered_nodes
        }
        for window in windows:
            max_demand = max(demand_lookup.get((stage, window), 0.0) for stage in ordered_nodes)
            max_p_active = max(p_active_lookup.get((stage, window), 1.0) for stage in ordered_nodes)
            if max_demand <= 0.0 or max_p_active <= 0.0:
                continue
            if max_p_active < 1.0 and rng.random() >= max_p_active:
                continue
            request_count = max(1, int(math.ceil(min(float(max_virtual), max_demand))))
            request_weight = max_demand / float(request_count)
            window_start_ms = float(window) * window_ms
            offsets = np.sort(rng.uniform(0.0, window_ms, size=request_count))
            for offset_ms in offsets:
                entry_abs_ms = window_start_ms + float(offset_ms)
                completions_abs: dict[str, float] = {}
                for stage in ordered_nodes:
                    parents = workflow.nodes[stage].parents
                    ready_abs_ms = max((completions_abs[parent] for parent in parents), default=entry_abs_ms)
                    effective_window = int(math.floor(ready_abs_ms / max(1.0, window_ms)))
                    plan_row = plan.lookup(stage, effective_window) or plan.lookup(stage, window)
                    if plan_row is None:
                        warm_count = 0.0
                        memory_mb = base_memory_mb
                        keepalive_ttl_sec = 0.0
                    else:
                        warm_count = float(plan_row.warm_count)
                        memory_mb = int(plan_row.memory_mb)
                        keepalive_ttl_sec = float(plan_row.keepalive_ttl_sec)
                    keepalive_ms = max(0.0, keepalive_ttl_sec * 1000.0)
                    stage_window_start_ms = float(effective_window) * window_ms
                    stage_window_end_ms = stage_window_start_ms + window_ms
                    timing = warmup_timing_for_stage(
                        warmup_mode=warmup_mode,
                        is_root=not parents,
                        ready_abs_ms=ready_abs_ms,
                        window_start_ms=stage_window_start_ms,
                        lead_time_ms=latency_pools.prewarm_lead_time_ms(stage),
                    )
                    pool = pool_state.setdefault(stage, ContainerPoolColdModel())
                    pool.ensure_warm_capacity(
                        warm_count=warm_count,
                        window_start_ms=timing.ready_ms,
                        window_end_ms=stage_window_end_ms,
                        keepalive_ms=keepalive_ms,
                        now_ms=ready_abs_ms,
                    )
                    pool_index, pool_cold = pool.reserve(ready_abs_ms)
                    residual_cold = (
                        (not pool_cold)
                        and residual_cold_probability > 0.0
                        and bool(rng.random() < residual_cold_probability)
                    )
                    is_cold = bool(pool_cold or residual_cold)
                    duration = latency_pools.sample(
                        stage,
                        cold_like=is_cold,
                        memory_mb=memory_mb,
                        base_memory_mb=base_memory_mb,
                        cpu_alpha=cpu_alpha,
                        overhead_alpha=overhead_alpha,
                    )
                    pool.complete(
                        index=pool_index,
                        ready_time_ms=ready_abs_ms,
                        duration_ms=duration,
                        keepalive_ms=keepalive_ms,
                    )
                    completions_abs[stage] = ready_abs_ms + duration
                workflow_latency = max(completions_abs.values()) - entry_abs_ms
                latencies.append(workflow_latency)
                violation_count += float(workflow_latency > slo_ms) * request_weight
                total += request_weight

    values = np.asarray(latencies, dtype=float)
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(violation_count / total),
        float(np.quantile(values, 0.50)),
        float(np.quantile(values, 0.90)),
        float(np.quantile(values, 0.95)),
    )


def summarize_plan_frame(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "mean_warm_count": 0.0,
            "max_warm_count": 0.0,
            "mean_keepalive_ttl_sec": 0.0,
            "mean_memory_mb": 0.0,
        }
    return {
        "mean_warm_count": float(frame["warm_count"].mean()),
        "max_warm_count": float(frame["warm_count"].max()),
        "mean_keepalive_ttl_sec": float(frame["keepalive_ttl_sec"].mean()),
        "mean_memory_mb": float(frame["memory_mb"].mean()),
    }


def evaluate_candidate(
    *,
    candidate_id: str,
    parent_id: str,
    action: str,
    state: dict[str, dict[str, float]],
    selected: pd.DataFrame,
    entry_forecast: pd.DataFrame | None,
    delay_kernel: pd.DataFrame | None,
    workflow: WorkflowSpec,
    ordered_nodes: list[str],
    latency_pools: PlannerLatencyPools,
    latency_samples: pd.DataFrame,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> CandidateEvaluation:
    candidate_selected = selected
    if entry_forecast is not None and delay_kernel is not None:
        candidate_selected = candidate_selected_from_entry(
            state,
            entry_forecast=entry_forecast,
            delay_kernel=delay_kernel,
            workflow=workflow,
            ordered_nodes=ordered_nodes,
            policy=args.policy,
            warm_source=args.warm_source,
            active_gate_threshold=args.active_gate_threshold,
        )
    plan = build_plan_from_state(
        state,
        selected=candidate_selected,
        workflow_name=workflow.workflow_name,
        window_sec=args.window_sec,
        warm_source=args.warm_source,
        candidate_id=candidate_id,
        active_gate_threshold=args.active_gate_threshold,
    )
    plan.metadata["warmup_mode"] = args.warmup_mode
    risk, p50_latency, p90_latency, p95_latency = evaluate_workflow_risk(
        plan=plan,
        selected=candidate_selected,
        workflow=workflow,
        ordered_nodes=ordered_nodes,
        latency_pools=latency_pools,
        rng=rng,
        slo_ms=args.slo_ms,
        simulations_per_window=args.simulations_per_window,
        max_eval_windows=args.max_eval_windows,
        residual_cold_probability=args.residual_cold_probability,
        base_memory_mb=args.base_memory_mb,
        cpu_alpha=args.cpu_alpha,
        overhead_alpha=args.overhead_alpha,
        demand_column=args.planner_demand_column,
        warmup_mode=args.warmup_mode,
        max_virtual_requests_per_window=args.max_virtual_requests_per_window,
    )
    cost = estimate_control_plan_cost(
        plan,
        forecast_detail=candidate_selected,
        latency_samples=latency_samples,
        workflow_name=workflow.workflow_name,
        window_sec=args.window_sec,
        demand_column="forecast_count",
        base_memory_mb=args.base_memory_mb,
        cpu_alpha=args.cpu_alpha,
        warmup_mode=args.warmup_mode,
        workflow=workflow,
    )
    plan_stats = summarize_plan_frame(plan_to_frame(plan))
    return CandidateEvaluation(
        candidate_id=candidate_id,
        parent_id=parent_id,
        action=action,
        risk=risk,
        p50_latency_ms=p50_latency,
        p90_latency_ms=p90_latency,
        p95_latency_ms=p95_latency,
        cost_gb_seconds=cost.total_gb_seconds,
        execution_gb_seconds=cost.execution_gb_seconds,
        warm_gb_seconds=cost.warm_gb_seconds,
        feasible=bool(risk <= args.risk_budget),
        state=state,
        plan=plan,
        **plan_stats,
    )


def is_dominated(row: CandidateEvaluation, other: CandidateEvaluation) -> bool:
    no_worse = other.cost_gb_seconds <= row.cost_gb_seconds and other.risk <= row.risk
    strictly_better = other.cost_gb_seconds < row.cost_gb_seconds or other.risk < row.risk
    return bool(no_worse and strictly_better)


def pareto_front(candidates: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    front = []
    for candidate in candidates:
        if any(is_dominated(candidate, other) for other in candidates if other is not candidate):
            continue
        front.append(candidate)
    return sorted(front, key=lambda item: (item.feasible is False, item.risk, item.cost_gb_seconds))


def select_beam(candidates: list[CandidateEvaluation], *, beam_width: int, risk_budget: float) -> list[CandidateEvaluation]:
    front = pareto_front(candidates)
    front.sort(key=lambda item: (max(0.0, item.risk - risk_budget), item.cost_gb_seconds, item.risk))
    return front[: max(1, beam_width)]


def candidate_summary(candidates: list[CandidateEvaluation]) -> pd.DataFrame:
    rows = []
    for item in candidates:
        rows.append(
            {
                "candidate_id": item.candidate_id,
                "parent_id": item.parent_id,
                "action": item.action,
                "risk": item.risk,
                "feasible": item.feasible,
                "cost_gb_seconds": item.cost_gb_seconds,
                "execution_gb_seconds": item.execution_gb_seconds,
                "warm_gb_seconds": item.warm_gb_seconds,
                "p50_latency_ms": item.p50_latency_ms,
                "p90_latency_ms": item.p90_latency_ms,
                "p95_latency_ms": item.p95_latency_ms,
                "mean_warm_count": item.mean_warm_count,
                "max_warm_count": item.max_warm_count,
                "mean_keepalive_ttl_sec": item.mean_keepalive_ttl_sec,
                "mean_memory_mb": item.mean_memory_mb,
            }
        )
    return pd.DataFrame(rows)


def build_stage_priority(
    *,
    selected: pd.DataFrame,
    workflow: WorkflowSpec,
    ordered_nodes: list[str],
    latency_samples: pd.DataFrame,
    policy: str,
    slo_ms: float,
) -> pd.DataFrame:
    profile = latency_quantiles(latency_samples, workflow.workflow_name, POLICY_TO_QUANTILE[policy])
    duration_lookup = dict(zip(profile["stage_name"], profile["warm_dispatch_q_ms"]))
    slack = dag_slack(workflow, ordered_nodes, duration_lookup, slo_ms)
    demand = selected.groupby("stage_name", as_index=False)["forecast_count"].sum()
    table = profile.merge(slack[["stage_name", "slack_ms"]], on="stage_name", how="left")
    table = table.merge(demand, on="stage_name", how="left")
    table["forecast_count"] = table["forecast_count"].fillna(0.0)
    table["cold_gap_ms"] = table["cold_dispatch_q_ms"] - table["warm_dispatch_q_ms"]
    max_demand = max(1.0, float(table["forecast_count"].max()))
    max_gap = max(1.0, float(table["cold_gap_ms"].clip(lower=0).max()))
    max_slack = max(1.0, float(table["slack_ms"].clip(lower=0).max()))
    table["demand_score"] = table["forecast_count"] / max_demand
    table["cold_gap_score"] = table["cold_gap_ms"].clip(lower=0) / max_gap
    table["slack_pressure_score"] = 1.0 - (table["slack_ms"].clip(lower=0) / max_slack)
    table["priority_score"] = (
        0.40 * table["demand_score"]
        + 0.30 * table["cold_gap_score"]
        + 0.30 * table["slack_pressure_score"]
    )
    return table.sort_values("priority_score", ascending=False).reset_index(drop=True)


def expand_candidate(
    item: CandidateEvaluation,
    *,
    stage_order: list[str],
    warm_options: list[float],
    keepalive_options: list[float],
    memory_tiers: list[int],
    top_stages: int,
    zero_forecast_stages: set[str],
) -> list[tuple[str, dict[str, dict[str, float]]]]:
    expansions: list[tuple[str, dict[str, dict[str, float]]]] = []
    for stage in stage_order[:top_stages]:
        warm_next = next_larger(item.state["warm_multiplier"][stage], warm_options)
        if warm_next is not None:
            state = clone_state(item.state)
            state["warm_multiplier"][stage] = warm_next
            expansions.append((f"raise_warm_multiplier:{stage}:{warm_next}", state))

        ttl_next = next_larger(item.state["keepalive_ttl_sec"][stage], keepalive_options)
        if ttl_next is not None:
            state = clone_state(item.state)
            state["keepalive_ttl_sec"][stage] = ttl_next
            expansions.append((f"raise_keepalive:{stage}:{ttl_next}", state))

        memory_next = next_larger(item.state["memory_mb"][stage], [float(value) for value in memory_tiers])
        if memory_next is not None:
            state = clone_state(item.state)
            state["memory_mb"][stage] = memory_next
            expansions.append((f"raise_memory:{stage}:{int(memory_next)}", state))

        if stage in zero_forecast_stages and item.state.get("min_warm_count", {}).get(stage, 0.0) < 1.0:
            state = clone_state(item.state)
            state.setdefault("min_warm_count", {})[stage] = 1.0
            expansions.append((f"raise_min_warm_when_zero:{stage}:1", state))
    return expansions


def choose_selected_candidate(candidates: list[CandidateEvaluation], risk_budget: float) -> CandidateEvaluation:
    feasible = [item for item in candidates if item.risk <= risk_budget]
    if feasible:
        return sorted(feasible, key=lambda item: (item.cost_gb_seconds, item.risk))[0]
    return sorted(candidates, key=lambda item: (item.risk, item.cost_gb_seconds))[0]


def write_readme(out_dir: Path, args: argparse.Namespace, selected: CandidateEvaluation) -> None:
    lines = [
        "# Risk-Budgeted Pareto Planner",
        "",
        "This is the first Stage-5A planner implementation. It searches candidate plans for:",
        "",
        "- `warm_count(stage, window)`",
        "- `keepalive_ttl_sec(stage)`",
        "- `memory_mb(stage)`",
        "",
        "The selected plan minimizes the current GB-second proxy cost among candidates that satisfy the configured SLO risk budget. If no candidate satisfies the budget, the planner writes the lowest-risk candidate and marks it infeasible in `selected_candidate.json`.",
        "",
        "## Main Outputs",
        "",
        "- `selected_control_plan.json` / `selected_control_plan.csv`: plan consumable by Stage4 `--control-plan`.",
        "- `candidate_summary.csv`: all evaluated candidates.",
        "- `pareto_front.csv`: nondominated candidates by `(cost, risk)`.",
        "- `stage_priority.csv`: expansion priority based on demand, cold gap, and DAG slack.",
        "- `selected_candidate.json`: selected cost/risk summary.",
        "",
        "## Selected Candidate",
        "",
        f"- candidate: `{selected.candidate_id}`",
        f"- risk: `{selected.risk:.6f}` under budget `{args.risk_budget:.6f}`",
        f"- cost_gb_seconds: `{selected.cost_gb_seconds:.6f}`",
        f"- feasible: `{selected.feasible}`",
        "",
        "## Guardrail",
        "",
        "This planner still uses an offline Monte Carlo proxy and memory-scaling model. The next step is to validate the chosen plan through Stage4 and then replace the scaling proxy with real memory-tier profiles.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


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
    workflow = load_workflow(str(resolve_path(root, args.workflow_config)))
    ordered_nodes = topological_nodes(workflow)
    memory_tiers = parse_memory_tiers(args.memory_tiers_mb)
    warm_options = parse_float_list(args.warm_multipliers)
    keepalive_options = parse_float_list(args.keepalive_options_sec)
    rng = np.random.default_rng(args.seed)

    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = sorted(workflow.nodes)
    entry_selected: pd.DataFrame | None = None
    delay_kernel: pd.DataFrame | None = None
    if use_entry_kernel:
        entry_selected = selected_entry_forecast(
            pd.read_csv(resolve_path(root, args.entry_forecast)),
            workflow_name=workflow.workflow_name,
            method=args.method,
            policy=args.policy,
            max_plan_windows=args.max_plan_windows,
        )
        delay_kernel = pd.read_csv(resolve_path(root, args.delay_kernel))
        selected = candidate_selected_from_entry(
            initial_state(
                stages,
                warm_multiplier=0.0,
                keepalive_ttl_sec=0.0,
                memory_mb=args.base_memory_mb,
            ),
            entry_forecast=entry_selected,
            delay_kernel=delay_kernel,
            workflow=workflow,
            ordered_nodes=ordered_nodes,
            policy=args.policy,
            warm_source=args.warm_source,
            active_gate_threshold=args.active_gate_threshold,
        )
        selected.to_csv(out_dir / "initial_propagated_stage_forecast.csv", index=False)
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
    latency_pools = PlannerLatencyPools(latency_samples, workflow.workflow_name, rng)

    stage_priority = build_stage_priority(
        selected=selected,
        workflow=workflow,
        ordered_nodes=ordered_nodes,
        latency_samples=latency_samples,
        policy=args.policy,
        slo_ms=args.slo_ms,
    )
    stage_priority.to_csv(out_dir / "stage_priority.csv", index=False)
    stage_order = [str(stage) for stage in stage_priority["stage_name"].tolist()]

    evaluated: list[CandidateEvaluation] = []
    seen: set[tuple[tuple[str, float, float, float, float], ...]] = set()
    stages = sorted(selected["stage_name"].astype(str).unique())
    source_table = stage_window_table(selected)
    zero_forecast_stages = {
        str(row["stage_name"])
        for row in source_table.to_dict(orient="records")
        if max(0.0, float(row.get(args.warm_source, 0.0) or 0.0)) <= 0.0
    }

    seed_memories = sorted(set(memory_tiers))
    seed_keepalive = min(keepalive_options)
    seed_states: list[tuple[str, dict[str, dict[str, float]]]] = []
    for memory_mb in seed_memories:
        seed_states.extend(
            [
                (
                    f"seed:scale_to_zero:mem{memory_mb}",
                    initial_state(stages, warm_multiplier=0.0, keepalive_ttl_sec=0.0, memory_mb=memory_mb),
                ),
                (
                    f"seed:platform_default:mem{memory_mb}",
                    initial_state(
                        stages,
                        warm_multiplier=0.0,
                        keepalive_ttl_sec=args.platform_keepalive_sec,
                        memory_mb=memory_mb,
                    ),
                ),
                (
                    f"seed:forecast_warm:mem{memory_mb}",
                    initial_state(stages, warm_multiplier=1.0, keepalive_ttl_sec=seed_keepalive, memory_mb=memory_mb),
                ),
                (
                    f"seed:forecast_warm_platform_keepalive:mem{memory_mb}",
                    initial_state(
                        stages,
                        warm_multiplier=1.0,
                        keepalive_ttl_sec=args.platform_keepalive_sec,
                        memory_mb=memory_mb,
                    ),
                ),
                (
                    f"seed:keepalive_dominant:mem{memory_mb}",
                    initial_state(stages, warm_multiplier=0.0, keepalive_ttl_sec=60.0, memory_mb=memory_mb),
                ),
            ]
        )
        for multiplier in warm_options:
            if multiplier in {0.0, 1.0}:
                continue
            seed_states.append(
                (
                    f"seed:global_warm:{multiplier}:mem{memory_mb}",
                    initial_state(
                        stages,
                        warm_multiplier=multiplier,
                        keepalive_ttl_sec=seed_keepalive,
                        memory_mb=memory_mb,
                    ),
                )
            )
    higher_memory = next_larger(float(args.base_memory_mb), [float(value) for value in memory_tiers])
    if higher_memory is not None:
        seed_states.append(
            (
                f"seed:memory_dominant:mem{int(higher_memory)}",
                initial_state(
                    stages,
                    warm_multiplier=1.0,
                    keepalive_ttl_sec=0.0,
                    memory_mb=int(higher_memory),
                ),
            )
        )

    candidate_counter = 0
    beam: list[CandidateEvaluation] = []
    for action, state in seed_states:
        key = state_key(state, stages)
        if key in seen:
            continue
        seen.add(key)
        candidate_counter += 1
        candidate = evaluate_candidate(
            candidate_id=f"c{candidate_counter:04d}",
            parent_id="",
            action=action,
            state=state,
            selected=selected,
            entry_forecast=entry_selected,
            delay_kernel=delay_kernel,
            workflow=workflow,
            ordered_nodes=ordered_nodes,
            latency_pools=latency_pools,
            latency_samples=latency_samples,
            rng=rng,
            args=args,
        )
        evaluated.append(candidate)
        beam.append(candidate)
    beam = select_beam(beam, beam_width=args.beam_width, risk_budget=args.risk_budget)

    for _ in range(args.max_iterations):
        next_round: list[CandidateEvaluation] = list(beam)
        for item in beam:
            for action, state in expand_candidate(
                item,
                stage_order=stage_order,
                warm_options=warm_options,
                keepalive_options=keepalive_options,
                memory_tiers=memory_tiers,
                top_stages=args.expand_top_stages,
                zero_forecast_stages=zero_forecast_stages,
            ):
                key = state_key(state, stages)
                if key in seen:
                    continue
                seen.add(key)
                candidate_counter += 1
                candidate = evaluate_candidate(
                    candidate_id=f"c{candidate_counter:04d}",
                    parent_id=item.candidate_id,
                    action=action,
                    state=state,
                    selected=selected,
                    entry_forecast=entry_selected,
                    delay_kernel=delay_kernel,
                    workflow=workflow,
                    ordered_nodes=ordered_nodes,
                    latency_pools=latency_pools,
                    latency_samples=latency_samples,
                    rng=rng,
                    args=args,
                )
                evaluated.append(candidate)
                next_round.append(candidate)
        beam = select_beam(next_round, beam_width=args.beam_width, risk_budget=args.risk_budget)

    selected_candidate = choose_selected_candidate(evaluated, args.risk_budget)
    save_control_plan(selected_candidate.plan, out_dir / "selected_control_plan.json")
    plan_to_frame(selected_candidate.plan).to_csv(out_dir / "selected_control_plan.csv", index=False)
    if entry_selected is not None and delay_kernel is not None:
        selected_stage_forecast = candidate_selected_from_entry(
            selected_candidate.state,
            entry_forecast=entry_selected,
            delay_kernel=delay_kernel,
            workflow=workflow,
            ordered_nodes=ordered_nodes,
            policy=args.policy,
            warm_source=args.warm_source,
            active_gate_threshold=args.active_gate_threshold,
        )
        selected_stage_forecast.to_csv(out_dir / "selected_propagated_stage_forecast.csv", index=False)

    summary = candidate_summary(evaluated)
    summary.to_csv(out_dir / "candidate_summary.csv", index=False)
    front = candidate_summary(pareto_front(evaluated))
    front.to_csv(out_dir / "pareto_front.csv", index=False)

    selected_payload: dict[str, Any] = {
        "candidate_id": selected_candidate.candidate_id,
        "parent_id": selected_candidate.parent_id,
        "action": selected_candidate.action,
        "risk": selected_candidate.risk,
        "risk_budget": args.risk_budget,
        "feasible": selected_candidate.feasible,
        "warmup_mode": args.warmup_mode,
        "cost_gb_seconds": selected_candidate.cost_gb_seconds,
        "execution_gb_seconds": selected_candidate.execution_gb_seconds,
        "warm_gb_seconds": selected_candidate.warm_gb_seconds,
        "p50_latency_ms": selected_candidate.p50_latency_ms,
        "p90_latency_ms": selected_candidate.p90_latency_ms,
        "p95_latency_ms": selected_candidate.p95_latency_ms,
        "mean_warm_count": selected_candidate.mean_warm_count,
        "max_warm_count": selected_candidate.max_warm_count,
        "mean_keepalive_ttl_sec": selected_candidate.mean_keepalive_ttl_sec,
        "mean_memory_mb": selected_candidate.mean_memory_mb,
        "state": selected_candidate.state,
    }
    (out_dir / "selected_candidate.json").write_text(
        json.dumps(selected_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

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
        "risk_budget": args.risk_budget,
        "memory_tiers_mb": memory_tiers,
        "warm_multipliers": warm_options,
        "keepalive_options_sec": keepalive_options,
        "platform_keepalive_sec": args.platform_keepalive_sec,
        "warmup_mode": args.warmup_mode,
        "active_gate_threshold": args.active_gate_threshold,
        "max_virtual_requests_per_window": args.max_virtual_requests_per_window,
        "prewarm_lead_ms_by_stage": latency_pools.prewarm_lead_ms_by_stage,
        "zero_forecast_stages_with_min_warm_candidate": sorted(zero_forecast_stages),
        "planner_demand_column": args.planner_demand_column,
        "simulations_per_window": args.simulations_per_window,
        "max_eval_windows": args.max_eval_windows,
        "max_plan_windows": args.max_plan_windows,
        "beam_width": args.beam_width,
        "max_iterations": args.max_iterations,
        "expand_top_stages": args.expand_top_stages,
        "candidate_count": len(evaluated),
        "notes": [
            "Planner risk is an offline proxy based on forecast demand and sampled Stage-3 latency pools.",
            "The selected plan should be re-evaluated with runner.stage4_risk.estimate_slo_risk --control-plan.",
            "Memory effects are estimated by the current memory-scaling model until real memory-tier profiles are collected.",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_readme(out_dir, args, selected_candidate)

    print(f"wrote {out_dir}")
    print(
        summary.sort_values(["feasible", "cost_gb_seconds"], ascending=[False, True])
        .head(10)
        .to_string(index=False)
    )
    print(
        "selected "
        f"{selected_candidate.candidate_id}: risk={selected_candidate.risk:.4f}, "
        f"cost={selected_candidate.cost_gb_seconds:.3f}, feasible={selected_candidate.feasible}"
    )


if __name__ == "__main__":
    main()
