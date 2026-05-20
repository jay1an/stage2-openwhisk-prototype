import argparse
import json
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..workflow import WorkflowSpec, load_workflow
from ..stage5_control.control_plan import ControlPlan, load_control_plan, plan_to_frame
from ..stage5_control.dag_warmup_scheduler import (
    cold_overhead_lead_times_ms,
    stage_lead_time_ms,
    warmup_timing_for_stage,
)
from .container_pool_cold_model import ContainerPoolColdModel

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Stage-4A workflow SLO-risk estimator. It combines a selected "
            "stage allocation forecast with empirical warm/cold-like latency pools "
            "and simulates workflow latency through the DAG."
        )
    )
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--forecast-detail", required=True)
    parser.add_argument("--latency-samples", required=True)
    parser.add_argument(
        "--control-plan",
        default=None,
        help=(
            "Optional Stage-5 control plan JSON/CSV. When provided, plan warm_count "
            "overrides selected forecast allocated_count for matching stage/window rows."
        ),
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--policy", required=True, choices=["p50", "p90", "p95"])
    parser.add_argument("--fold-id", type=int, default=None)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--slo-ms", type=float, required=True)
    parser.add_argument("--simulations-per-request", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--risk-bins", type=int, default=10)
    parser.add_argument(
        "--write-stage-samples",
        action="store_true",
        help="write per-simulation stage samples; useful for debugging but can be large",
    )
    parser.add_argument(
        "--residual-cold-probability",
        type=float,
        default=0.0,
        help="minimum cold-like probability even when allocated_count covers actual_count",
    )
    parser.add_argument(
        "--cold-model",
        choices=["pool", "deficit"],
        default="pool",
        help=(
            "pool tracks per-stage warm/busy/expired containers over request time; "
            "deficit uses the older per-window allocation deficit probability"
        ),
    )
    parser.add_argument(
        "--warmup-mode",
        choices=["window", "dag_jit"],
        default="window",
        help=(
            "window makes planned warm_count available at control-window start; "
            "dag_jit makes root warm_count available at window start and downstream "
            "warm_count available only after a cold-overhead lead-time prewarm"
        ),
    )
    parser.add_argument(
        "--enable-memory-scaling",
        action="store_true",
        help="scale sampled platform/action latency by plan_memory_mb when a control plan provides memory decisions",
    )
    parser.add_argument("--base-memory-mb", type=int, default=256)
    parser.add_argument("--cpu-alpha", type=float, default=1.0)
    parser.add_argument("--overhead-alpha", type=float, default=0.08)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def clean_stage_trace(trace: pd.DataFrame, workflow_name: str, window_ms: int) -> pd.DataFrame:
    rows = trace[(trace["workflow_name"] == workflow_name) & (trace["stage_name"] != "__entry__")].copy()
    if "status" in rows.columns:
        rows = rows[rows["status"].fillna("ok") == "ok"].copy()
    for col in ["entry_ts_ms", "dispatch_start_ms", "dispatch_end_ms", "dispatch_latency_ms"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows["stage_window"] = (rows["dispatch_start_ms"] // window_ms).astype(int)
    return rows


def selected_forecast_detail(
    detail: pd.DataFrame,
    workflow_name: str,
    method: str,
    policy: str,
    fold_id: int | None,
) -> pd.DataFrame:
    if "target_window" not in detail.columns and "window" in detail.columns:
        detail = detail.copy()
        detail["target_window"] = detail["window"]
    selected = detail[
        (detail["workflow_name"] == workflow_name)
        & (detail["method"] == method)
        & (detail["policy"] == policy)
    ].copy()
    if fold_id is not None and "fold_id" in selected.columns:
        selected = selected[selected["fold_id"] == fold_id].copy()
    if selected.empty:
        raise ValueError(
            f"No forecast rows found for workflow={workflow_name}, method={method}, "
            f"policy={policy}, fold_id={fold_id}"
        )
    return selected


def apply_control_plan_to_detail(
    detail: pd.DataFrame,
    plan: ControlPlan,
) -> tuple[pd.DataFrame, dict[str, int]]:
    selected = detail.copy()
    matched = 0
    unmatched = 0
    selected["plan_applied"] = False
    selected["plan_warm_count"] = np.nan
    selected["plan_keepalive_ttl_sec"] = np.nan
    selected["plan_memory_mb"] = np.nan

    for idx, row in selected.iterrows():
        plan_row = plan.lookup(str(row["stage_name"]), int(row["target_window"]))
        if plan_row is None:
            unmatched += 1
            continue
        matched += 1
        selected.at[idx, "allocated_count"] = float(plan_row.warm_count)
        selected.at[idx, "plan_applied"] = True
        selected.at[idx, "plan_warm_count"] = float(plan_row.warm_count)
        selected.at[idx, "plan_keepalive_ttl_sec"] = float(plan_row.keepalive_ttl_sec)
        selected.at[idx, "plan_memory_mb"] = int(plan_row.memory_mb)

    return selected, {
        "control_plan_rows": len(plan.rows),
        "matched_forecast_rows": matched,
        "unmatched_forecast_rows": unmatched,
    }


def apply_keepalive_carryover_to_detail(
    detail: pd.DataFrame,
    *,
    window_sec: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if "plan_keepalive_ttl_sec" not in detail.columns:
        return detail, {"keepalive_carryover_rows": 0}

    selected = detail.copy()
    selected["allocated_count_without_keepalive"] = selected["allocated_count"]
    selected["keepalive_carry_count"] = 0.0
    carryover_rows = 0

    for _, group in selected.sort_values("target_window").groupby("stage_name"):
        carry_count = 0.0
        carry_until_sec = -1.0
        for idx, row in group.iterrows():
            window = int(row["target_window"])
            window_start_sec = float(window) * window_sec
            window_end_sec = (float(window) + 1.0) * window_sec
            base_allocated = max(0.0, float(row["allocated_count"]))
            effective_allocated = base_allocated
            if window_start_sec <= carry_until_sec and carry_count > effective_allocated:
                effective_allocated = carry_count
                carryover_rows += 1

            selected.at[idx, "allocated_count"] = effective_allocated
            selected.at[idx, "keepalive_carry_count"] = max(0.0, effective_allocated - base_allocated)

            keepalive_ttl_sec = row.get("plan_keepalive_ttl_sec", np.nan)
            keepalive_ttl_sec = 0.0 if pd.isna(keepalive_ttl_sec) else float(keepalive_ttl_sec)
            actual_count = max(0.0, float(row.get("actual_count", 0.0) or 0.0))
            post_warm_count = max(effective_allocated, actual_count)
            if keepalive_ttl_sec > 0 and post_warm_count > 0:
                carry_count = post_warm_count
                carry_until_sec = window_end_sec + keepalive_ttl_sec
            else:
                carry_count = 0.0
                carry_until_sec = -1.0

    return selected, {"keepalive_carryover_rows": carryover_rows}


def allocation_lookup(detail: pd.DataFrame) -> dict[tuple[str, int], dict[str, float]]:
    value_columns = ["actual_count", "forecast_count", "allocated_count"]
    optional_columns = [
        col
        for col in [
            "plan_applied",
            "plan_warm_count",
            "plan_keepalive_ttl_sec",
            "plan_memory_mb",
            "allocated_count_without_keepalive",
            "keepalive_carry_count",
        ]
        if col in detail.columns
    ]
    grouped = (
        detail.groupby(["stage_name", "target_window"], as_index=False)[value_columns + optional_columns]
        .max()
        .reset_index(drop=True)
    )
    lookup = {}
    for _, row in grouped.iterrows():
        record = {
            "actual_count": float(row["actual_count"]),
            "forecast_count": float(row["forecast_count"]),
            "allocated_count": float(row["allocated_count"]),
        }
        for col in optional_columns:
            record[col] = float(row[col]) if col != "plan_applied" else bool(row[col])
        lookup[(str(row["stage_name"]), int(row["target_window"]))] = record
    return lookup


def topological_nodes(workflow: WorkflowSpec) -> list[str]:
    remaining = set(workflow.nodes)
    ordered: list[str] = []
    while remaining:
        ready = sorted(
            name
            for name in remaining
            if all(parent in ordered for parent in workflow.nodes[name].parents)
        )
        if not ready:
            raise ValueError("workflow DAG contains a cycle or missing parent")
        ordered.extend(ready)
        remaining.difference_update(ready)
    return ordered


class LatencySampler:
    def __init__(self, samples: pd.DataFrame, workflow_name: str, rng: np.random.Generator):
        self.samples = samples.copy()
        self.workflow_name = workflow_name
        self.rng = rng
        for col in ["dispatch_latency_ms", "platform_overhead_ms", "action_duration_ms"]:
            self.samples[col] = pd.to_numeric(self.samples[col], errors="coerce")
        self.samples = self.samples.dropna(
            subset=["dispatch_latency_ms", "platform_overhead_ms", "action_duration_ms"]
        )
        self.dispatch_pools: dict[tuple[str, str, str], np.ndarray] = {}
        self.stage_class_dispatch_pools: dict[tuple[str, str], np.ndarray] = {}
        self.component_pools: dict[tuple[str, str, str], np.ndarray] = {}
        self.stage_class_component_pools: dict[tuple[str, str], np.ndarray] = {}
        for (workflow, stage, klass), group in self.samples.groupby(
            ["workflow_name", "stage_name", "latency_class"]
        ):
            self.dispatch_pools[(str(workflow), str(stage), str(klass))] = group[
                "dispatch_latency_ms"
            ].to_numpy(dtype=float)
            self.component_pools[(str(workflow), str(stage), str(klass))] = group[
                ["dispatch_latency_ms", "platform_overhead_ms", "action_duration_ms"]
            ].to_numpy(dtype=float)
        for (stage, klass), group in self.samples.groupby(["stage_name", "latency_class"]):
            self.stage_class_dispatch_pools[(str(stage), str(klass))] = group[
                "dispatch_latency_ms"
            ].to_numpy(dtype=float)
            self.stage_class_component_pools[(str(stage), str(klass))] = group[
                ["dispatch_latency_ms", "platform_overhead_ms", "action_duration_ms"]
            ].to_numpy(dtype=float)

        self.stage_action_pools: dict[tuple[str, str], np.ndarray] = {}
        for (workflow, stage), group in self.samples.groupby(["workflow_name", "stage_name"]):
            self.stage_action_pools[(str(workflow), str(stage))] = group[
                "action_duration_ms"
            ].to_numpy(dtype=float)

        self.global_cold_overheads = self.samples[self.samples["latency_class"] == "cold_like"][
            "platform_overhead_ms"
        ].to_numpy(dtype=float)
        if len(self.global_cold_overheads) == 0:
            self.global_cold_overheads = self.samples["platform_overhead_ms"].to_numpy(dtype=float)
        self.global_action_durations = self.samples["action_duration_ms"].to_numpy(dtype=float)
        self.global_warm_dispatch = self.samples[self.samples["latency_class"] == "warm"][
            "dispatch_latency_ms"
        ].to_numpy(dtype=float)
        if len(self.global_warm_dispatch) == 0:
            self.global_warm_dispatch = self.samples["dispatch_latency_ms"].to_numpy(dtype=float)

    def _dispatch_pool(self, stage_name: str, latency_class: str) -> np.ndarray:
        pool = self.dispatch_pools.get((self.workflow_name, stage_name, latency_class))
        if pool is not None and len(pool) > 0:
            return pool
        pool = self.stage_class_dispatch_pools.get((stage_name, latency_class))
        if pool is not None and len(pool) > 0:
            return pool
        return np.asarray([], dtype=float)

    def _component_pool(self, stage_name: str, latency_class: str) -> np.ndarray:
        pool = self.component_pools.get((self.workflow_name, stage_name, latency_class))
        if pool is not None and len(pool) > 0:
            return pool
        pool = self.stage_class_component_pools.get((stage_name, latency_class))
        if pool is not None and len(pool) > 0:
            return pool
        return np.asarray([], dtype=float)

    def _choice(self, values: np.ndarray) -> float:
        return float(values[int(self.rng.integers(0, len(values)))])

    def _scaled_dispatch(
        self,
        component: np.ndarray,
        memory_mb: int | None,
        base_memory_mb: int,
        cpu_alpha: float,
        overhead_alpha: float,
    ) -> float:
        dispatch_ms = float(component[0])
        if memory_mb is None:
            return dispatch_ms
        ratio = max(1.0, float(memory_mb)) / max(1.0, float(base_memory_mb))
        overhead_ms = float(component[1]) / (ratio ** overhead_alpha)
        action_ms = float(component[2]) / (ratio ** cpu_alpha)
        return max(1.0, overhead_ms + action_ms)

    def sample(
        self,
        stage_name: str,
        cold_like: bool,
        memory_mb: int | None = None,
        base_memory_mb: int = 256,
        cpu_alpha: float = 1.0,
        overhead_alpha: float = 0.08,
    ) -> tuple[float, str]:
        latency_class = "cold_like" if cold_like else "warm"
        component_pool = self._component_pool(stage_name, latency_class)
        if len(component_pool) > 0:
            component = component_pool[int(self.rng.integers(0, len(component_pool)))]
            return (
                self._scaled_dispatch(
                    component,
                    memory_mb=memory_mb,
                    base_memory_mb=base_memory_mb,
                    cpu_alpha=cpu_alpha,
                    overhead_alpha=overhead_alpha,
                ),
                latency_class,
            )

        pool = self._dispatch_pool(stage_name, latency_class)
        if len(pool) > 0:
            return self._choice(pool), latency_class

        if cold_like:
            # If the target workflow has no cold-like sample for this stage, compose a
            # cold path from global cold-like platform overhead and this stage's action duration.
            stage_action = self.stage_action_pools.get((self.workflow_name, stage_name))
            if stage_action is None or len(stage_action) == 0:
                stage_action = self.global_action_durations
            overhead = self._choice(self.global_cold_overheads)
            action = self._choice(stage_action)
            if memory_mb is not None:
                ratio = max(1.0, float(memory_mb)) / max(1.0, float(base_memory_mb))
                overhead = overhead / (ratio ** overhead_alpha)
                action = action / (ratio ** cpu_alpha)
            return max(1.0, overhead + action), "cold_like_composed"

        return self._choice(self.global_warm_dispatch), "warm_fallback"


def cold_probability(
    allocation: dict[str, float] | None,
    residual_cold_probability: float,
) -> float:
    if allocation is None:
        return residual_cold_probability
    actual = max(0.0, float(allocation["actual_count"]))
    allocated = max(0.0, float(allocation["allocated_count"]))
    if actual <= 0:
        return residual_cold_probability
    deficit_fraction = max(0.0, actual - allocated) / actual
    return min(1.0, max(residual_cold_probability, deficit_fraction))


def allocation_for_stage(
    *,
    stage_name: str,
    stage_window: int | None,
    ready_abs_ms: float,
    allocations: dict[tuple[str, int], dict[str, float]],
    window_ms: int,
) -> tuple[int | None, dict[str, float] | None]:
    ready_window = int(float(ready_abs_ms) // float(window_ms))
    allocation = allocations.get((stage_name, ready_window))
    if allocation is not None:
        return ready_window, allocation

    if stage_window is not None:
        window = int(stage_window)
        allocation = allocations.get((stage_name, window))
        if allocation is not None:
            return window, allocation

    return stage_window, None


def build_eval_requests(
    stage_rows: pd.DataFrame,
    forecast_detail: pd.DataFrame,
    workflow: WorkflowSpec,
) -> list[dict]:
    target_windows = set(int(value) for value in forecast_detail["target_window"].unique())
    needed_stages = set(workflow.nodes)
    requests = []
    for request_id, group in stage_rows.groupby("request_id"):
        stages = set(group["stage_name"].astype(str))
        if not needed_stages.issubset(stages):
            continue
        stage_windows = {
            str(row["stage_name"]): int(row["stage_window"])
            for _, row in group.drop_duplicates("stage_name").iterrows()
        }
        if not all(stage in stage_windows for stage in needed_stages):
            continue
        if not set(stage_windows.values()).issubset(target_windows):
            continue
        entry_ts_ms = float(group["entry_ts_ms"].iloc[0])
        observed_latency = float(group["dispatch_end_ms"].max() - entry_ts_ms)
        requests.append(
            {
                "request_id": request_id,
                "entry_ts_ms": entry_ts_ms,
                "observed_workflow_latency_ms": observed_latency,
                "stage_windows": stage_windows,
            }
        )
    return sorted(requests, key=lambda item: (float(item["entry_ts_ms"]), str(item["request_id"])))


def simulate_one_request(
    workflow: WorkflowSpec,
    ordered_nodes: list[str],
    stage_windows: dict[str, int],
    allocations: dict[tuple[str, int], dict[str, float]],
    sampler: LatencySampler,
    rng: np.random.Generator,
    residual_cold_probability: float,
    enable_memory_scaling: bool = False,
    base_memory_mb: int = 256,
    cpu_alpha: float = 1.0,
    overhead_alpha: float = 0.08,
) -> tuple[float, int, int]:
    predicted_latency, cold_count, composed_cold_count, _ = simulate_one_request_detailed(
        workflow=workflow,
        ordered_nodes=ordered_nodes,
        stage_windows=stage_windows,
        allocations=allocations,
        sampler=sampler,
        rng=rng,
        residual_cold_probability=residual_cold_probability,
        enable_memory_scaling=enable_memory_scaling,
        base_memory_mb=base_memory_mb,
        cpu_alpha=cpu_alpha,
        overhead_alpha=overhead_alpha,
        cold_model="deficit",
    )
    return predicted_latency, cold_count, composed_cold_count


def simulate_one_request_detailed(
    workflow: WorkflowSpec,
    ordered_nodes: list[str],
    stage_windows: dict[str, int],
    allocations: dict[tuple[str, int], dict[str, float]],
    sampler: LatencySampler,
    rng: np.random.Generator,
    residual_cold_probability: float,
    enable_memory_scaling: bool = False,
    base_memory_mb: int = 256,
    cpu_alpha: float = 1.0,
    overhead_alpha: float = 0.08,
    cold_model: str = "deficit",
    pool_state: dict[str, ContainerPoolColdModel] | None = None,
    entry_ts_ms: float | None = None,
    window_ms: int | None = None,
    warmup_mode: str = "window",
    prewarm_lead_ms_by_stage: dict[str, float] | None = None,
) -> tuple[float, int, int, list[dict]]:
    completions: dict[str, float] = {}
    stage_records: dict[str, dict] = {}
    cold_count = 0
    composed_cold_count = 0
    if cold_model == "pool":
        if pool_state is None:
            raise ValueError("pool_state is required when cold_model='pool'")
        if entry_ts_ms is None:
            raise ValueError("entry_ts_ms is required when cold_model='pool'")
        if window_ms is None:
            raise ValueError("window_ms is required when cold_model='pool'")
    for stage_name in ordered_nodes:
        parents = workflow.nodes[stage_name].parents
        ready_time = max((completions[parent] for parent in parents), default=0.0)
        ready_abs_ms = float(entry_ts_ms or 0.0) + ready_time
        window, allocation = allocation_for_stage(
            stage_name=stage_name,
            stage_window=stage_windows.get(stage_name),
            ready_abs_ms=ready_abs_ms,
            allocations=allocations,
            window_ms=window_ms or 1,
        )
        memory_mb = None
        if enable_memory_scaling and allocation is not None and not pd.isna(allocation.get("plan_memory_mb", np.nan)):
            memory_mb = int(allocation["plan_memory_mb"])

        pool_cold = False
        pool_added_warm = 0
        pool_size_after_dispatch = np.nan
        p_cold = cold_probability(allocation, residual_cold_probability)
        if cold_model == "pool":
            effective_window = (
                int(window)
                if window is not None
                else int(float(ready_abs_ms) // float(window_ms or 1))
            )
            window_start_ms = float(effective_window) * float(window_ms or 1)
            window_end_ms = window_start_ms + float(window_ms or 1)
            warm_count = max(0.0, float(allocation.get("allocated_count", 0.0))) if allocation else 0.0
            keepalive_ttl_sec = (
                float(allocation.get("plan_keepalive_ttl_sec", 0.0))
                if allocation and not pd.isna(allocation.get("plan_keepalive_ttl_sec", np.nan))
                else 0.0
            )
            keepalive_ms = max(0.0, keepalive_ttl_sec * 1000.0)
            pool = pool_state.setdefault(stage_name, ContainerPoolColdModel())
            lead_time_ms = stage_lead_time_ms(stage_name, prewarm_lead_ms_by_stage)
            warmup_timing = warmup_timing_for_stage(
                warmup_mode=warmup_mode,
                is_root=not parents,
                ready_abs_ms=ready_abs_ms,
                window_start_ms=window_start_ms,
                lead_time_ms=lead_time_ms,
            )
            pool_added_warm = pool.ensure_warm_capacity(
                warm_count=warm_count,
                window_start_ms=warmup_timing.ready_ms,
                window_end_ms=window_end_ms,
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
            p_cold = 1.0 if pool_cold else float(residual_cold_probability)
        else:
            is_cold = bool(rng.random() < p_cold)

        duration, sampled_class = sampler.sample(
            stage_name,
            is_cold,
            memory_mb=memory_mb,
            base_memory_mb=base_memory_mb,
            cpu_alpha=cpu_alpha,
            overhead_alpha=overhead_alpha,
        )
        if cold_model == "pool":
            keepalive_ttl_sec = (
                float(allocation.get("plan_keepalive_ttl_sec", 0.0))
                if allocation and not pd.isna(allocation.get("plan_keepalive_ttl_sec", np.nan))
                else 0.0
            )
            pool.complete(
                index=pool_index,
                ready_time_ms=ready_abs_ms,
                duration_ms=duration,
                keepalive_ms=max(0.0, keepalive_ttl_sec * 1000.0),
            )
            pool_size_after_dispatch = float(len(pool.containers))
        if sampled_class.startswith("cold_like"):
            cold_count += 1
        if sampled_class == "cold_like_composed":
            composed_cold_count += 1
        completions[stage_name] = ready_time + duration
        stage_records[stage_name] = {
            "stage_name": stage_name,
            "stage_window": window,
            "ready_time_ms": ready_time,
            "ready_abs_ms": ready_abs_ms,
            "sampled_duration_ms": duration,
            "completion_time_ms": completions[stage_name],
            "sampled_class": sampled_class,
            "cold_draw": is_cold,
            "p_cold": p_cold,
            "cold_model": cold_model,
            "pool_cold": pool_cold,
            "pool_added_warm": float(pool_added_warm),
            "pool_size_after_dispatch": pool_size_after_dispatch,
            "warmup_mode": warmup_mode,
            "warmup_lead_time_ms": (
                float(lead_time_ms) if cold_model == "pool" else np.nan
            ),
            "planned_warm_ready_abs_ms": (
                float(warmup_timing.ready_ms) if cold_model == "pool" else np.nan
            ),
            "actual_count": float(allocation["actual_count"]) if allocation else np.nan,
            "forecast_count": float(allocation["forecast_count"]) if allocation else np.nan,
            "allocated_count": float(allocation["allocated_count"]) if allocation else np.nan,
            "plan_applied": bool(allocation.get("plan_applied", False)) if allocation else False,
            "plan_warm_count": float(allocation.get("plan_warm_count", np.nan)) if allocation else np.nan,
            "plan_keepalive_ttl_sec": float(allocation.get("plan_keepalive_ttl_sec", np.nan)) if allocation else np.nan,
            "plan_memory_mb": float(allocation.get("plan_memory_mb", np.nan)) if allocation else np.nan,
            "memory_scaling_applied": bool(memory_mb is not None),
            "allocated_count_without_keepalive": float(allocation.get("allocated_count_without_keepalive", np.nan)) if allocation else np.nan,
            "keepalive_carry_count": float(allocation.get("keepalive_carry_count", np.nan)) if allocation else np.nan,
        }

    workflow_latency = max(completions.values())
    sink = max(completions, key=completions.get)
    critical_path: set[str] = set()
    current = sink
    while current:
        critical_path.add(current)
        parents = workflow.nodes[current].parents
        if not parents:
            break
        current = max(parents, key=lambda parent: completions[parent])
    for stage_name, record in stage_records.items():
        record["critical_path"] = stage_name in critical_path
        record["workflow_predicted_latency_ms"] = workflow_latency
    return workflow_latency, cold_count, composed_cold_count, list(stage_records.values())


def update_stage_accumulator(acc: dict, stage_records: list[dict]) -> None:
    for record in stage_records:
        stage_name = record["stage_name"]
        bucket = acc[stage_name]
        bucket["duration"].append(float(record["sampled_duration_ms"]))
        bucket["p_cold"].append(float(record["p_cold"]))
        bucket["cold_draw"].append(float(bool(record["cold_draw"])))
        bucket["cold_sample"].append(float(str(record["sampled_class"]).startswith("cold_like")))
        bucket["composed_cold"].append(float(record["sampled_class"] == "cold_like_composed"))
        bucket["critical_path"].append(float(bool(record["critical_path"])))
        for col in [
            "actual_count",
            "forecast_count",
            "allocated_count",
            "plan_applied",
            "plan_warm_count",
            "plan_keepalive_ttl_sec",
            "plan_memory_mb",
            "allocated_count_without_keepalive",
            "keepalive_carry_count",
            "memory_scaling_applied",
            "pool_cold",
            "pool_added_warm",
            "pool_size_after_dispatch",
        ]:
            value = record.get(col, np.nan)
            if not pd.isna(value):
                bucket[col].append(float(value))


def summarize_stage_contribution(stage_acc: dict, workflow_name: str, method: str, policy: str, fold_id) -> pd.DataFrame:
    rows = []
    for stage_name, bucket in sorted(stage_acc.items()):
        durations = np.asarray(bucket["duration"], dtype=float)
        if len(durations) == 0:
            continue
        critical = np.asarray(bucket["critical_path"], dtype=float)
        critical_durations = durations[critical > 0.0]
        row = {
            "workflow_name": workflow_name,
            "method": method,
            "policy": policy,
            "fold_id": fold_id if fold_id is not None else "all",
            "stage_name": stage_name,
            "simulation_rows": int(len(durations)),
            "mean_duration_ms": float(np.mean(durations)),
            "p50_duration_ms": float(np.quantile(durations, 0.50)),
            "p90_duration_ms": float(np.quantile(durations, 0.90)),
            "p95_duration_ms": float(np.quantile(durations, 0.95)),
            "mean_p_cold": float(np.mean(bucket["p_cold"])),
            "cold_draw_rate": float(np.mean(bucket["cold_draw"])),
            "cold_sample_rate": float(np.mean(bucket["cold_sample"])),
            "composed_cold_rate": float(np.mean(bucket["composed_cold"])),
            "critical_path_rate": float(np.mean(critical)),
            "mean_duration_when_critical_ms": float(np.mean(critical_durations)) if len(critical_durations) else 0.0,
            "mean_actual_count": float(np.mean(bucket["actual_count"])) if bucket["actual_count"] else np.nan,
            "mean_forecast_count": float(np.mean(bucket["forecast_count"])) if bucket["forecast_count"] else np.nan,
            "mean_allocated_count": float(np.mean(bucket["allocated_count"])) if bucket["allocated_count"] else np.nan,
            "plan_applied_rate": float(np.mean(bucket["plan_applied"])) if bucket["plan_applied"] else 0.0,
            "mean_plan_warm_count": float(np.mean(bucket["plan_warm_count"])) if bucket["plan_warm_count"] else np.nan,
            "mean_plan_keepalive_ttl_sec": float(np.mean(bucket["plan_keepalive_ttl_sec"])) if bucket["plan_keepalive_ttl_sec"] else np.nan,
            "mean_plan_memory_mb": float(np.mean(bucket["plan_memory_mb"])) if bucket["plan_memory_mb"] else np.nan,
            "mean_allocated_count_without_keepalive": float(np.mean(bucket["allocated_count_without_keepalive"])) if bucket["allocated_count_without_keepalive"] else np.nan,
            "mean_keepalive_carry_count": float(np.mean(bucket["keepalive_carry_count"])) if bucket["keepalive_carry_count"] else np.nan,
            "pool_cold_rate": float(np.mean(bucket["pool_cold"])) if bucket["pool_cold"] else np.nan,
            "mean_pool_added_warm": float(np.mean(bucket["pool_added_warm"])) if bucket["pool_added_warm"] else np.nan,
            "mean_pool_size_after_dispatch": float(np.mean(bucket["pool_size_after_dispatch"])) if bucket["pool_size_after_dispatch"] else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_instances(instances: pd.DataFrame, slo_ms: float) -> pd.DataFrame:
    predicted = instances["predicted_latency_ms"]
    observed = instances.drop_duplicates("request_id")["observed_workflow_latency_ms"]
    row = {
        "workflow_name": instances["workflow_name"].iloc[0],
        "method": instances["method"].iloc[0],
        "policy": instances["policy"].iloc[0],
        "fold_id": instances["fold_id"].iloc[0],
        "slo_ms": slo_ms,
        "requests": int(instances["request_id"].nunique()),
        "simulation_rows": int(len(instances)),
        "predicted_violation_probability": float((predicted > slo_ms).mean()),
        "observed_violation_rate": float((observed > slo_ms).mean()),
        "predicted_latency_p50_ms": float(predicted.quantile(0.50)),
        "predicted_latency_p90_ms": float(predicted.quantile(0.90)),
        "predicted_latency_p95_ms": float(predicted.quantile(0.95)),
        "observed_latency_p50_ms": float(observed.quantile(0.50)),
        "observed_latency_p90_ms": float(observed.quantile(0.90)),
        "observed_latency_p95_ms": float(observed.quantile(0.95)),
        "mean_cold_like_stages_per_sim": float(instances["cold_like_stage_count"].mean()),
        "mean_composed_cold_stages_per_sim": float(instances["composed_cold_stage_count"].mean()),
    }
    return pd.DataFrame([row])


def build_per_request_risk(instances: pd.DataFrame, slo_ms: float) -> pd.DataFrame:
    return (
        instances.groupby("request_id")
        .agg(
            observed_workflow_latency_ms=("observed_workflow_latency_ms", "first"),
            predicted_p50_ms=("predicted_latency_ms", lambda x: float(x.quantile(0.50))),
            predicted_p90_ms=("predicted_latency_ms", lambda x: float(x.quantile(0.90))),
            predicted_p95_ms=("predicted_latency_ms", lambda x: float(x.quantile(0.95))),
            violation_probability=("predicted_latency_ms", lambda x: float((x > slo_ms).mean())),
        )
        .reset_index()
        .assign(observed_violation=lambda df: (df["observed_workflow_latency_ms"] > slo_ms).astype(int))
    )


def risk_calibration_tables(per_request: pd.DataFrame, bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    scored = per_request.copy()
    scored["risk_bin"] = np.minimum(
        (scored["violation_probability"] * bins).astype(int),
        bins - 1,
    )
    for bin_id, group in scored.groupby("risk_bin"):
        start = bin_id / bins
        end = (bin_id + 1) / bins
        mean_risk = float(group["violation_probability"].mean())
        observed_rate = float(group["observed_violation"].mean())
        rows.append(
            {
                "risk_bin": int(bin_id),
                "risk_bin_start": start,
                "risk_bin_end": end,
                "requests": int(len(group)),
                "mean_predicted_violation_probability": mean_risk,
                "observed_violation_rate": observed_rate,
                "absolute_calibration_error": abs(mean_risk - observed_rate),
                "bin_brier_score": float(
                    np.mean((group["violation_probability"] - group["observed_violation"]) ** 2)
                ),
            }
        )
    bin_table = pd.DataFrame(rows)
    if bin_table.empty:
        summary = pd.DataFrame(
            [
                {
                    "requests": 0,
                    "brier_score": np.nan,
                    "expected_calibration_error": np.nan,
                    "max_calibration_error": np.nan,
                }
            ]
        )
    else:
        weights = bin_table["requests"] / bin_table["requests"].sum()
        summary = pd.DataFrame(
            [
                {
                    "requests": int(len(per_request)),
                    "brier_score": float(
                        np.mean(
                            (per_request["violation_probability"] - per_request["observed_violation"]) ** 2
                        )
                    ),
                    "expected_calibration_error": float(
                        np.sum(weights * bin_table["absolute_calibration_error"])
                    ),
                    "max_calibration_error": float(bin_table["absolute_calibration_error"].max()),
                }
            ]
        )
    return bin_table, summary


def write_plots(out_dir: Path, instances: pd.DataFrame, slo_ms: float) -> pd.DataFrame:
    per_request = build_per_request_risk(instances, slo_ms).sort_values("observed_workflow_latency_ms")
    per_request.to_csv(out_dir / "risk_by_request.csv", index=False)
    if not HAS_MATPLOTLIB:
        (out_dir / "plot_warning.txt").write_text(
            "matplotlib is not installed; CSV risk outputs were generated but PNG plots were skipped.\n",
            encoding="utf-8",
        )
        return per_request

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.hist(instances["predicted_latency_ms"], bins=60, alpha=0.75, color="#3182ce")
    ax.axvline(slo_ms, color="#c53030", linestyle="--", linewidth=1.5, label=f"SLO={slo_ms:g} ms")
    ax.set_title("Monte Carlo workflow latency distribution")
    ax.set_xlabel("workflow latency (ms)")
    ax.set_ylabel("simulation count")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "monte_carlo_workflow_latency_hist.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    x = np.arange(len(per_request))
    ax.plot(x, per_request["observed_workflow_latency_ms"], label="observed", color="#2d3748")
    ax.plot(x, per_request["predicted_p50_ms"], label="predicted p50", color="#2f855a")
    ax.plot(x, per_request["predicted_p95_ms"], label="predicted p95", color="#dd6b20")
    ax.axhline(slo_ms, color="#c53030", linestyle="--", linewidth=1.2, label="SLO")
    ax.set_title("Per-request observed latency vs predicted Monte Carlo bands")
    ax.set_xlabel("request sorted by observed latency")
    ax.set_ylabel("workflow latency (ms)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "per_request_predicted_vs_observed.png")
    plt.close(fig)
    return per_request


def write_readme(out_dir: Path, summary: pd.DataFrame) -> None:
    row = summary.iloc[0]
    lines = [
        "# Stage 4-A Minimum SLO Risk Estimator",
        "",
        "## Scope",
        "",
        "- Offline diagnostic Monte Carlo estimator.",
        "- Connects Stage-2 stage allocation forecasts with Stage-3 empirical latency samples.",
        "- Supports a container-pool cold model and the older allocation-deficit proxy.",
        "- Does not yet model queueing, controller feedback, keep-alive state transitions, or resource-size changes.",
        "",
        "## Result",
        "",
        f"- Method: `{row['method']}`.",
        f"- Policy: `{row['policy']}`.",
        f"- Fold: `{row['fold_id']}`.",
        f"- SLO: `{row['slo_ms']:.2f} ms`.",
        f"- Requests: `{int(row['requests'])}`.",
        f"- Predicted violation probability: `{row['predicted_violation_probability']:.6f}`.",
        f"- Observed violation rate: `{row['observed_violation_rate']:.6f}`.",
        f"- Predicted p95 workflow latency: `{row['predicted_latency_p95_ms']:.2f} ms`.",
        f"- Observed p95 workflow latency: `{row['observed_latency_p95_ms']:.2f} ms`.",
        "",
        "## Outputs",
        "",
        "- `risk_summary.csv`: one-row summary.",
        "- `risk_simulation_instances.csv`: Monte Carlo samples.",
        "- `risk_by_request.csv`: per-request predicted bands and violation probability.",
        "- `risk_bin_table.csv`: calibration bins for predicted SLO violation probability.",
        "- `risk_calibration_summary.csv`: Brier score and calibration error.",
        "- `stage_risk_contribution.csv`: per-stage cold/critical-path contribution summary.",
        "- `monte_carlo_workflow_latency_hist.png`: predicted latency histogram.",
        "- `per_request_predicted_vs_observed.png`: observed vs predicted bands.",
        "",
        "## Guardrail",
        "",
        "This is an offline diagnostic model. Pool mode is closer to the profiled trace simulator, but it is still not a live controller measurement.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_estimation(args: argparse.Namespace) -> pd.DataFrame:
    root = project_root()
    workflow = load_workflow(str(resolve_path(root, args.workflow_config)))
    window_ms = int(args.window_sec * 1000)
    rng = np.random.default_rng(args.seed)

    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = pd.read_csv(resolve_path(root, args.trace))
    stage_rows = clean_stage_trace(trace, workflow.workflow_name, window_ms)
    detail = pd.read_csv(resolve_path(root, args.forecast_detail))
    selected_detail = selected_forecast_detail(
        detail,
        workflow_name=workflow.workflow_name,
        method=args.method,
        policy=args.policy,
        fold_id=args.fold_id,
    )
    control_plan = None
    effective_warmup_mode = args.warmup_mode
    control_plan_metadata: dict[str, int | str | float | None] = {
        "control_plan_path": None,
        "control_plan_rows": 0,
        "matched_forecast_rows": 0,
        "unmatched_forecast_rows": 0,
    }
    if args.control_plan:
        control_plan_path = resolve_path(root, args.control_plan)
        control_plan = load_control_plan(control_plan_path, default_window_sec=args.window_sec)
        effective_warmup_mode = str(control_plan.metadata.get("warmup_mode", args.warmup_mode))
        selected_detail, match_metadata = apply_control_plan_to_detail(selected_detail, control_plan)
        selected_detail, carry_metadata = apply_keepalive_carryover_to_detail(
            selected_detail,
            window_sec=args.window_sec,
        )
        control_plan_metadata.update(match_metadata)
        control_plan_metadata.update(carry_metadata)
        control_plan_metadata["control_plan_path"] = str(control_plan_path)
        control_plan_metadata["control_plan_window_sec"] = control_plan.window_sec
        control_plan_metadata["effective_warmup_mode"] = effective_warmup_mode

    allocations = allocation_lookup(selected_detail)
    eval_requests = build_eval_requests(stage_rows, selected_detail, workflow)
    if not eval_requests:
        raise ValueError("No evaluation requests overlap the selected forecast detail")

    latency_samples = pd.read_csv(resolve_path(root, args.latency_samples))
    sampler = LatencySampler(latency_samples, workflow.workflow_name, rng)
    prewarm_lead_ms_by_stage = cold_overhead_lead_times_ms(
        latency_samples,
        workflow.workflow_name,
    )
    ordered_nodes = topological_nodes(workflow)

    rows = []
    stage_acc = defaultdict(lambda: defaultdict(list))
    stage_sample_rows = []

    if args.cold_model == "pool":
        for sim_id in range(args.simulations_per_request):
            pool_state = {
                stage_name: ContainerPoolColdModel()
                for stage_name in ordered_nodes
            }
            for request in eval_requests:
                predicted_latency, cold_count, composed_cold_count, stage_records = simulate_one_request_detailed(
                    workflow=workflow,
                    ordered_nodes=ordered_nodes,
                    stage_windows=request["stage_windows"],
                    allocations=allocations,
                    sampler=sampler,
                    rng=rng,
                    residual_cold_probability=args.residual_cold_probability,
                    enable_memory_scaling=args.enable_memory_scaling,
                    base_memory_mb=args.base_memory_mb,
                    cpu_alpha=args.cpu_alpha,
                    overhead_alpha=args.overhead_alpha,
                    cold_model=args.cold_model,
                    pool_state=pool_state,
                    entry_ts_ms=float(request["entry_ts_ms"]),
                    window_ms=window_ms,
                    warmup_mode=effective_warmup_mode,
                    prewarm_lead_ms_by_stage=prewarm_lead_ms_by_stage,
                )
                update_stage_accumulator(stage_acc, stage_records)
                if args.write_stage_samples:
                    for stage_record in stage_records:
                        stage_sample_rows.append(
                            {
                                "workflow_name": workflow.workflow_name,
                                "request_id": request["request_id"],
                                "simulation_id": sim_id,
                                "method": args.method,
                                "policy": args.policy,
                                "fold_id": args.fold_id if args.fold_id is not None else "all",
                                **stage_record,
                            }
                        )
                rows.append(
                    {
                        "workflow_name": workflow.workflow_name,
                        "request_id": request["request_id"],
                        "simulation_id": sim_id,
                        "method": args.method,
                        "policy": args.policy,
                        "fold_id": args.fold_id if args.fold_id is not None else "all",
                        "slo_ms": args.slo_ms,
                        "observed_workflow_latency_ms": request["observed_workflow_latency_ms"],
                        "predicted_latency_ms": predicted_latency,
                        "cold_like_stage_count": cold_count,
                        "composed_cold_stage_count": composed_cold_count,
                    }
                )
    else:
        for request in eval_requests:
            for sim_id in range(args.simulations_per_request):
                predicted_latency, cold_count, composed_cold_count, stage_records = simulate_one_request_detailed(
                    workflow=workflow,
                    ordered_nodes=ordered_nodes,
                    stage_windows=request["stage_windows"],
                    allocations=allocations,
                    sampler=sampler,
                    rng=rng,
                    residual_cold_probability=args.residual_cold_probability,
                    enable_memory_scaling=args.enable_memory_scaling,
                    base_memory_mb=args.base_memory_mb,
                    cpu_alpha=args.cpu_alpha,
                    overhead_alpha=args.overhead_alpha,
                    cold_model=args.cold_model,
                    warmup_mode=effective_warmup_mode,
                    prewarm_lead_ms_by_stage=prewarm_lead_ms_by_stage,
                )
                update_stage_accumulator(stage_acc, stage_records)
                if args.write_stage_samples:
                    for stage_record in stage_records:
                        stage_sample_rows.append(
                            {
                                "workflow_name": workflow.workflow_name,
                                "request_id": request["request_id"],
                                "simulation_id": sim_id,
                                "method": args.method,
                                "policy": args.policy,
                                "fold_id": args.fold_id if args.fold_id is not None else "all",
                                **stage_record,
                            }
                        )
                rows.append(
                    {
                        "workflow_name": workflow.workflow_name,
                        "request_id": request["request_id"],
                        "simulation_id": sim_id,
                        "method": args.method,
                        "policy": args.policy,
                        "fold_id": args.fold_id if args.fold_id is not None else "all",
                        "slo_ms": args.slo_ms,
                        "observed_workflow_latency_ms": request["observed_workflow_latency_ms"],
                        "predicted_latency_ms": predicted_latency,
                        "cold_like_stage_count": cold_count,
                        "composed_cold_stage_count": composed_cold_count,
                    }
                )
    instances = pd.DataFrame(rows)
    summary = summarize_instances(instances, args.slo_ms)
    stage_summary = summarize_stage_contribution(
        stage_acc,
        workflow_name=workflow.workflow_name,
        method=args.method,
        policy=args.policy,
        fold_id=args.fold_id,
    )

    instances.to_csv(out_dir / "risk_simulation_instances.csv", index=False)
    summary.to_csv(out_dir / "risk_summary.csv", index=False)
    stage_summary.to_csv(out_dir / "stage_risk_contribution.csv", index=False)
    if args.write_stage_samples:
        pd.DataFrame(stage_sample_rows).to_csv(out_dir / "risk_stage_simulation_samples.csv", index=False)
    selected_detail.to_csv(out_dir / "selected_forecast_detail.csv", index=False)
    if control_plan is not None:
        plan_to_frame(control_plan).to_csv(out_dir / "input_control_plan.csv", index=False)
    per_request = write_plots(out_dir, instances, args.slo_ms)
    bin_table, calibration_summary = risk_calibration_tables(per_request, args.risk_bins)
    bin_table.to_csv(out_dir / "risk_bin_table.csv", index=False)
    calibration_summary.to_csv(out_dir / "risk_calibration_summary.csv", index=False)
    write_readme(out_dir, summary)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_config": str(resolve_path(root, args.workflow_config)),
        "trace": str(resolve_path(root, args.trace)),
        "forecast_detail": str(resolve_path(root, args.forecast_detail)),
        "latency_samples": str(resolve_path(root, args.latency_samples)),
        "control_plan": control_plan_metadata,
        "method": args.method,
        "policy": args.policy,
        "fold_id": args.fold_id,
        "window_ms": window_ms,
        "slo_ms": args.slo_ms,
        "simulations_per_request": args.simulations_per_request,
        "cold_model": args.cold_model,
        "warmup_mode": args.warmup_mode,
        "effective_warmup_mode": effective_warmup_mode,
        "prewarm_lead_ms_by_stage": prewarm_lead_ms_by_stage,
        "residual_cold_probability": args.residual_cold_probability,
        "enable_memory_scaling": args.enable_memory_scaling,
        "base_memory_mb": args.base_memory_mb,
        "cpu_alpha": args.cpu_alpha,
        "overhead_alpha": args.overhead_alpha,
        "risk_bins": args.risk_bins,
        "write_stage_samples": args.write_stage_samples,
        "notes": [
            "cold_model=pool tracks per-stage warm/busy/expired containers across requests inside each Monte Carlo replication.",
            "warmup_mode=dag_jit makes downstream planned warm capacity available only after a cold-overhead lead-time prewarm attempt.",
            "cold_model=deficit keeps the older allocation-deficit cold-like probability proxy.",
            "Missing stage-specific cold-like samples are composed from global cold-like overhead plus stage action duration.",
            "This is an offline diagnostic model, not a final controller.",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote {out_dir}")
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    args = parse_args()
    run_estimation(args)


if __name__ == "__main__":
    main()

