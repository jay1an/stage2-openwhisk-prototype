import argparse
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, List

from .openwhisk_client import OpenWhiskClient
from .resource_profiles import memory_to_cpu_cores as profile_memory_to_cpu_cores
from .trace_store import CsvTraceStore
from .workflow import NodeSpec, WorkflowSpec, load_workflow, suffix_action_name


DEPLOYED_MEMORY_TIERS = (512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3840)
DEPLOYED_MEMORY_TIER_SET = set(DEPLOYED_MEMORY_TIERS)


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def to_float_or_none(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def latency_fields(
    dispatch_start_ms: int,
    dispatch_end_ms: int,
    action_duration_ms: object = "",
    ow_wait_ms: object = "",
    ow_init_ms: object = "",
    ow_duration_ms: object = "",
) -> dict:
    dispatch_latency_ms = dispatch_end_ms - dispatch_start_ms
    action_duration = to_float_or_none(action_duration_ms)
    wait = to_float_or_none(ow_wait_ms)
    init = to_float_or_none(ow_init_ms)
    ow_duration = to_float_or_none(ow_duration_ms)
    platform_overhead_ms = (
        dispatch_latency_ms - action_duration
        if action_duration is not None
        else ""
    )
    ow_runtime_overhead_ms = (
        ow_duration - (init or 0.0) - action_duration
        if ow_duration is not None and action_duration is not None
        else ""
    )
    client_gateway_overhead_ms = (
        dispatch_latency_ms - wait - ow_duration
        if wait is not None and ow_duration is not None
        else ""
    )
    return {
        "dispatch_latency_ms": dispatch_latency_ms,
        "platform_overhead_ms": platform_overhead_ms,
        "ow_runtime_overhead_ms": ow_runtime_overhead_ms,
        "client_gateway_overhead_ms": client_gateway_overhead_ms,
    }


def activation_annotations(activation: dict) -> dict:
    return {
        annotation.get("key"): annotation.get("value")
        for annotation in activation.get("annotations", [])
        if isinstance(annotation, dict) and annotation.get("key")
    }


def openwhisk_memory_to_cpu_cores(memory_mb: int) -> float:
    from .stage4_risk.scaling import memory_to_cpu_cores

    return memory_to_cpu_cores(memory_mb)


def validate_stage_plan(workflow: WorkflowSpec, plan: dict[str, int]) -> dict[str, int]:
    missing = [stage_name for stage_name in workflow.nodes if stage_name not in plan]
    if missing:
        raise ValueError(f"plan is missing tiers for stage(s): {', '.join(missing)}")

    normalized: dict[str, int] = {}
    for stage_name in workflow.nodes:
        raw_tier = plan[stage_name]
        try:
            tier = int(raw_tier)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"plan tier for stage {stage_name!r} must be an integer, got {raw_tier!r}"
            ) from exc
        if tier not in DEPLOYED_MEMORY_TIER_SET:
            raise ValueError(
                f"plan tier for stage {stage_name!r} must be one of "
                f"{list(DEPLOYED_MEMORY_TIERS)}, got {tier}"
            )
        normalized[stage_name] = tier
    return normalized


def invoke_node(
    client: OpenWhiskClient,
    workflow: WorkflowSpec,
    node: NodeSpec,
    request_id: str,
    entry_ts_ms: int,
    parent_results: Dict[str, dict],
    allocated_memory_mb: int | None = None,
    allocated_cpu_cores: float | None = None,
    action_name: str | None = None,
    slo_class: str | None = None,
) -> dict:
    resolved_action_name = action_name or node.action
    dispatch_start_ms = now_ms()
    real_invoke_monotonic = time.monotonic()
    try:
        activation = client.invoke_activation(
            resolved_action_name,
            {
                "workflow_name": workflow.workflow_name,
                "request_id": request_id,
                "entry_ts_ms": entry_ts_ms,
                "stage_name": node.name,
                "parent_stages": node.parents,
                "sleep_ms": node.sleep_ms,
                "cpu_iters": node.cpu_iters,
                "serial_cpu_iters": node.serial_cpu_iters,
                "parallel_cpu_iters": node.parallel_cpu_iters,
                "serial_fraction": node.serial_fraction,
                "io_wait_ms": node.io_wait_ms,
                "parallel_workers": node.parallel_workers,
                "max_parallel_workers": node.max_parallel_workers,
                "memory_kb": node.memory_kb,
                "memory_passes": node.memory_passes,
                "memory_stride": node.memory_stride,
                "output_items": node.output_items,
                "allocated_memory_mb": allocated_memory_mb,
                "allocated_cpu_cores": allocated_cpu_cores,
                "payload": {
                    "parents": {
                        parent: parent_results.get(parent, {})
                        for parent in node.parents
                    }
                },
            },
        )
        dispatch_end_ms = now_ms()
        response = activation.get("response", {})
        result = response.get("result", {}) if isinstance(response, dict) else {}
        if not isinstance(result, dict):
            result = {"error": result}
        annotations = activation_annotations(activation)
        action_duration_ms = result.get("action_duration_ms", "")
        ow_wait_ms = annotations.get("waitTime", "")
        ow_init_ms = annotations.get("initTime", 0)
        ow_duration_ms = activation.get("duration", "")
        limits = annotations.get("limits", {})
        ow_memory_mb = limits.get("memory", "") if isinstance(limits, dict) else ""
        activation_status = response.get("status", "") if isinstance(response, dict) else ""
        activation_error = result.get("error", "") if isinstance(result, dict) else ""
        row_status = "ok"
        row_error = ""
        if not response:
            row_status = "error"
            row_error = "activation did not return a completed response"
        elif activation_status and activation_status != "success":
            row_status = "error"
            row_error = f"activation status={activation_status}: {activation_error}"
        elif action_duration_ms in ("", None):
            row_status = "error"
            row_error = "activation result is missing action_duration_ms"
        return {
            "workflow_name": workflow.workflow_name,
            "request_id": request_id,
            "stage_name": node.name,
            "parent_stages": ",".join(node.parents),
            "slo_class": slo_class or "",
            "entry_ts_ms": entry_ts_ms,
            "dispatch_start_ms": dispatch_start_ms,
            "dispatch_end_ms": dispatch_end_ms,
            "real_invoke_monotonic": real_invoke_monotonic,
            "resolved_action_name": resolved_action_name,
            **latency_fields(
                dispatch_start_ms,
                dispatch_end_ms,
                action_duration_ms,
                ow_wait_ms,
                ow_init_ms,
                ow_duration_ms,
            ),
            "action_start_ns": result.get("action_start_ns", ""),
            "action_end_ns": result.get("action_end_ns", ""),
            "action_duration_ms": action_duration_ms,
            "container_id": result.get("container_id", ""),
            "container_invocation_index": result.get("container_invocation_index", ""),
            "container_uptime_ms": result.get("container_uptime_ms", ""),
            "previous_action_end_ns": result.get("previous_action_end_ns", ""),
            "idle_since_prev_ms": result.get("idle_since_prev_ms", ""),
            "cold_like": result.get("cold_like", ""),
            "pod_name": result.get("pod_name", ""),
            "activation_id": activation.get("activationId", ""),
            "action_version": activation.get("version", ""),
            "ow_cold_start": "initTime" in annotations,
            "ow_memory_mb": ow_memory_mb,
            "allocated_memory_mb": result.get("allocated_memory_mb", allocated_memory_mb or ""),
            "allocated_cpu_cores": result.get("allocated_cpu_cores", allocated_cpu_cores or ""),
            "detected_cpu_cores": result.get("detected_cpu_cores", ""),
            "ow_wait_ms": ow_wait_ms,
            "ow_init_ms": ow_init_ms,
            "ow_duration_ms": ow_duration_ms,
            "cpu_user_ms": result.get("cpu_user_ms", ""),
            "cpu_system_ms": result.get("cpu_system_ms", ""),
            "cpu_self_ms": result.get("cpu_self_ms", ""),
            "cpu_self_process_ms": result.get("cpu_self_process_ms", ""),
            "cpu_children_user_ms": result.get("cpu_children_user_ms", ""),
            "cpu_children_system_ms": result.get("cpu_children_system_ms", ""),
            "cpu_children_ms": result.get("cpu_children_ms", ""),
            "cpu_process_ms": result.get("cpu_process_ms", ""),
            "cpu_total_ms": result.get("cpu_total_ms", ""),
            "parallel_cpu_ms": result.get("parallel_cpu_ms", ""),
            "observed_effective_cores": result.get("observed_effective_cores", ""),
            "observed_parallel_cores": result.get("observed_parallel_cores", ""),
            "workload_mode": result.get("workload_mode", ""),
            "serial_cpu_iters": result.get("serial_cpu_iters", ""),
            "parallel_cpu_iters": result.get("parallel_cpu_iters", ""),
            "io_wait_ms": result.get("io_wait_ms", ""),
            "parallel_workers": result.get("parallel_workers", ""),
            "parallel_workers_used": result.get("parallel_workers_used", ""),
            "serial_wall_ms": result.get("serial_wall_ms", ""),
            "io_wall_ms": result.get("io_wall_ms", ""),
            "parallel_wall_ms": result.get("parallel_wall_ms", ""),
            "memory_wall_ms": result.get("memory_wall_ms", ""),
            "mem_rss_kb": result.get("mem_rss_kb", ""),
            "mem_peak_kb": result.get("mem_peak_kb", ""),
            "status": row_status,
            "error": row_error,
            "_result": result,
        }
    except Exception as exc:
        dispatch_end_ms = now_ms()
        return {
            "workflow_name": workflow.workflow_name,
            "request_id": request_id,
            "stage_name": node.name,
            "parent_stages": ",".join(node.parents),
            "slo_class": slo_class or "",
            "entry_ts_ms": entry_ts_ms,
            "dispatch_start_ms": dispatch_start_ms,
            "dispatch_end_ms": dispatch_end_ms,
            "real_invoke_monotonic": real_invoke_monotonic,
            "resolved_action_name": resolved_action_name,
            **latency_fields(dispatch_start_ms, dispatch_end_ms),
            "allocated_memory_mb": allocated_memory_mb or "",
            "allocated_cpu_cores": allocated_cpu_cores or "",
            "status": "error",
            "error": str(exc),
            "_result": {},
        }


def status_value(status: object, key: str, default: object = "") -> object:
    if isinstance(status, dict):
        return status.get(key, default)
    return getattr(status, key, default)


def run_one_workflow(
    workflow: WorkflowSpec,
    client: OpenWhiskClient,
    max_workers: int,
    allocated_memory_mb: int | None = None,
    allocated_cpu_cores: float | None = None,
    raise_on_error: bool = True,
    plan: dict[str, int] | None = None,
    slo_class: str | None = None,
    jit_scheduler: object | None = None,
    enable_jit: bool = False,
    jit_margin_ms: float = 600.0,
    jit_fire_settle_ms: float = 0.0,
    jit_warmup_tracker: object | None = None,
    enable_jit_sync: bool = False,
    jit_sync_pause_grace_ms: float = 3000.0,
) -> List[dict]:
    normalized_plan = validate_stage_plan(workflow, plan) if plan is not None else None
    jit_active = bool(enable_jit and jit_scheduler is not None and normalized_plan is not None)
    jit_sync_active = bool(jit_active and enable_jit_sync and jit_warmup_tracker is not None)
    warm_splines = None
    cold_overhead_table = None
    if jit_active:
        from .stage4_risk.scaling import (
            cold_overhead_for_tier,
            load_cleansed_cold_overhead,
            load_warm_splines,
            spline_predict_warm_mean,
        )
        from .stage5_control.jit_scheduler import WarmupTask

        warm_splines = load_warm_splines()
        cold_overhead_table = load_cleansed_cold_overhead()

    request_id = str(uuid.uuid4())
    entry_ts_ms = now_ms()
    workflow_start_monotonic = time.monotonic()
    completed: Dict[str, dict] = {}
    running = {}
    started_at: Dict[str, float] = {}
    measured_completion_at: Dict[str, float] = {}
    jit_current_fire_times: Dict[str, float] = {}
    jit_scheduled_count = 0
    jit_upsert_count = 0
    jit_late_count = 0
    jit_initial_late_count = 0
    jit_upsert_late_count = 0
    if normalized_plan is not None:
        entry_memory_mb = normalized_plan.get(workflow.entry, "")
    else:
        entry_memory_mb = allocated_memory_mb or ""
    entry_cpu_cores = (
        openwhisk_memory_to_cpu_cores(entry_memory_mb)
        if isinstance(entry_memory_mb, int)
        else allocated_cpu_cores or ""
    )
    trace_rows: List[dict] = [
        {
            "workflow_name": workflow.workflow_name,
            "request_id": request_id,
            "stage_name": "__entry__",
            "parent_stages": "",
            "slo_class": slo_class or "",
            "entry_ts_ms": entry_ts_ms,
            "workflow_start_ms": entry_ts_ms,
            "workflow_end_ms": "",
            "workflow_e2e_ms": "",
            "dispatch_start_ms": entry_ts_ms,
            "dispatch_end_ms": entry_ts_ms,
            "dispatch_latency_ms": 0,
            "platform_overhead_ms": "",
            "allocated_memory_mb": entry_memory_mb,
            "allocated_cpu_cores": entry_cpu_cores,
            "jit_enabled": jit_active,
            "jit_scheduled_count": 0,
            "jit_upsert_count": 0,
            "jit_late_count": 0,
            "jit_initial_late_count": 0,
            "jit_upsert_late_count": 0,
            "status": "ok",
            "error": "",
        }
    ]

    def topological_stage_names() -> list[str]:
        remaining = list(workflow.nodes)
        seen = set()
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
                raise RuntimeError(
                    f"workflow has a cycle or missing parent; remaining={remaining}"
                )
        return ordered

    topo_order = topological_stage_names()

    def predicted_warm_duration_ms(stage_name: str) -> float:
        if normalized_plan is None or warm_splines is None:
            raise RuntimeError("JIT warm duration requested without an active plan")
        tier = normalized_plan[stage_name]
        cpu = openwhisk_memory_to_cpu_cores(tier)
        return float(spline_predict_warm_mean(stage_name, cpu, warm_splines))

    def stage_cold_overhead_ms(stage_name: str) -> float:
        if normalized_plan is None or cold_overhead_table is None:
            raise RuntimeError("JIT cold overhead requested without an active plan")
        return float(
            cold_overhead_for_tier(
                stage_name,
                normalized_plan[stage_name],
                cold_overhead_table,
            )
        )

    def predicted_duration_seconds(stage_name: str) -> float:
        duration_ms = predicted_warm_duration_ms(stage_name)
        if stage_name == workflow.entry:
            duration_ms += stage_cold_overhead_ms(stage_name)
        return duration_ms / 1000.0

    def compute_predicted_times() -> tuple[dict[str, float], dict[str, float]]:
        predicted_start: dict[str, float] = {}
        predicted_completion: dict[str, float] = {}
        for stage_name in topo_order:
            node = workflow.nodes[stage_name]
            if node.parents:
                start_at = max(predicted_completion[parent] for parent in node.parents)
            else:
                start_at = workflow_start_monotonic
            if stage_name in started_at:
                start_at = started_at[stage_name]
            predicted_start[stage_name] = start_at

            if stage_name in measured_completion_at:
                predicted_completion[stage_name] = measured_completion_at[stage_name]
            else:
                predicted_completion[stage_name] = (
                    start_at + predicted_duration_seconds(stage_name)
                )
        return predicted_start, predicted_completion

    def schedule_stage_warmup(
        stage_name: str,
        needed_at: float,
        schedule_phase: str,
    ) -> None:
        nonlocal jit_scheduled_count
        nonlocal jit_upsert_count
        nonlocal jit_late_count
        nonlocal jit_initial_late_count
        nonlocal jit_upsert_late_count

        if not jit_active or normalized_plan is None or cold_overhead_table is None:
            return

        now = time.monotonic()
        existing_fire_time = jit_current_fire_times.get(stage_name)
        if schedule_phase == "upsert" and existing_fire_time is not None:
            if existing_fire_time <= now:
                return

        node = workflow.nodes[stage_name]
        stage_tier = normalized_plan[stage_name]
        cold_overhead_ms = stage_cold_overhead_ms(stage_name)
        raw_fire_time = (
            needed_at
            - (cold_overhead_ms + jit_fire_settle_ms) / 1000.0
            - jit_margin_ms / 1000.0
        )
        late_jit = raw_fire_time <= now
        fire_time = now if late_jit else raw_fire_time
        if late_jit:
            jit_late_count += 1
            if schedule_phase == "initial":
                jit_initial_late_count += 1
            else:
                jit_upsert_late_count += 1

        if schedule_phase == "initial":
            jit_scheduled_count += 1
        else:
            jit_upsert_count += 1

        task = WarmupTask(
            task_key=f"{request_id}:{stage_name}",
            fire_time=fire_time,
            action_name=suffix_action_name(
                node.action,
                f"_{stage_tier}",
            ),
            metadata={
                "request_id": request_id,
                "workflow_name": workflow.workflow_name,
                "stage_name": stage_name,
                "tier_mb": stage_tier,
                "slo_class": slo_class or "",
                "needed_at": needed_at,
                "raw_fire_time": raw_fire_time,
                "fire_time": fire_time,
                "late_jit": late_jit,
                "jit_margin_ms": jit_margin_ms,
                "jit_fire_settle_ms": jit_fire_settle_ms,
                "cold_overhead_ms": cold_overhead_ms,
                "parents": list(node.parents),
                "schedule_phase": schedule_phase,
            },
        )
        jit_current_fire_times[stage_name] = fire_time
        jit_scheduler.schedule(task)

    def enqueue_initial_jit_warmups() -> None:
        if not jit_active:
            return
        predicted_start, _ = compute_predicted_times()
        for stage_name in topo_order:
            if stage_name == workflow.entry:
                continue
            schedule_stage_warmup(
                stage_name,
                predicted_start[stage_name],
                schedule_phase="initial",
            )

    def upsert_pending_jit_warmups() -> None:
        if not jit_active:
            return
        predicted_start, _ = compute_predicted_times()
        for stage_name in topo_order:
            if stage_name == workflow.entry:
                continue
            if stage_name in completed or stage_name in running or stage_name in started_at:
                continue
            schedule_stage_warmup(
                stage_name,
                predicted_start[stage_name],
                schedule_phase="upsert",
            )

    def wait_for_stage_warmup(stage_name: str) -> dict:
        sync_info = {
            "jit_sync_enabled": jit_sync_active,
            "jit_sync_waited_ms": 0.0,
            "jit_sync_dispatch_after_warmup": False,
            "jit_sync_status": "not_applicable",
            "jit_sync_warmup_issued_monotonic": "",
            "jit_sync_warmup_completed_monotonic": "",
        }
        if not jit_sync_active or stage_name == workflow.entry:
            return sync_info

        start_wait = time.monotonic()
        max_wait_s = max(0.0, stage_cold_overhead_ms(stage_name) / 1000.0)
        pause_grace_s = max(0.0, jit_sync_pause_grace_ms / 1000.0)
        completion_deadline = start_wait + max_wait_s
        deadline = completion_deadline + pause_grace_s

        status = {}
        if hasattr(jit_warmup_tracker, "get_status"):
            status = jit_warmup_tracker.get_status(request_id, stage_name)
        issued = status_value(status, "issued_monotonic", "")
        completed = status_value(status, "completed_monotonic", "")
        if issued not in ("", None):
            sync_info["jit_sync_warmup_issued_monotonic"] = issued
            sync_info["jit_sync_status"] = "in_flight"
        else:
            sync_info["jit_sync_status"] = "not_issued"

        if completed in ("", None) and hasattr(jit_warmup_tracker, "wait_until_completed"):
            remaining = max(0.0, completion_deadline - time.monotonic())
            if remaining > 0.0:
                status = jit_warmup_tracker.wait_until_completed(
                    request_id,
                    stage_name,
                    remaining,
                )
                issued = status_value(status, "issued_monotonic", issued)
                completed = status_value(status, "completed_monotonic", completed)

        if issued not in ("", None):
            sync_info["jit_sync_warmup_issued_monotonic"] = issued
        if completed not in ("", None):
            sync_info["jit_sync_warmup_completed_monotonic"] = completed
            sync_info["jit_sync_dispatch_after_warmup"] = True
            sync_info["jit_sync_status"] = "completed"
            ready_at = float(completed) + pause_grace_s
            remaining_grace = ready_at - time.monotonic()
            remaining_budget = deadline - time.monotonic()
            if remaining_grace > 0.0 and remaining_budget > 0.0:
                time.sleep(min(remaining_grace, remaining_budget))
        elif issued not in ("", None):
            sync_info["jit_sync_status"] = "timed_out_in_flight"
        else:
            sync_info["jit_sync_status"] = "timed_out_not_issued"

        sync_info["jit_sync_waited_ms"] = (time.monotonic() - start_wait) * 1000.0
        return sync_info

    enqueue_initial_jit_warmups()
    trace_rows[0]["jit_scheduled_count"] = jit_scheduled_count
    trace_rows[0]["jit_upsert_count"] = jit_upsert_count
    trace_rows[0]["jit_late_count"] = jit_late_count
    trace_rows[0]["jit_initial_late_count"] = jit_initial_late_count
    trace_rows[0]["jit_upsert_late_count"] = jit_upsert_late_count

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while len(completed) < len(workflow.nodes):
            ready = workflow.ready_nodes(completed.keys(), running.keys())
            for node in ready:
                node_memory_mb = allocated_memory_mb
                node_cpu_cores = allocated_cpu_cores
                node_action_name = node.action
                if normalized_plan is not None:
                    node_memory_mb = normalized_plan[node.name]
                    node_cpu_cores = openwhisk_memory_to_cpu_cores(node_memory_mb)
                    node_action_name = suffix_action_name(node.action, f"_{node_memory_mb}")

                started_at[node.name] = time.monotonic()

                def invoke_with_optional_sync(
                    ready_node: NodeSpec = node,
                    ready_memory_mb: int | None = node_memory_mb,
                    ready_cpu_cores: float | None = node_cpu_cores,
                    ready_action_name: str = node_action_name,
                ) -> dict:
                    sync_info = wait_for_stage_warmup(ready_node.name)
                    row = invoke_node(
                        client,
                        workflow,
                        ready_node,
                        request_id,
                        entry_ts_ms,
                        completed,
                        ready_memory_mb,
                        ready_cpu_cores,
                        ready_action_name,
                        slo_class,
                    )
                    row.update(sync_info)
                    return row

                future = pool.submit(
                    invoke_with_optional_sync,
                )
                running[node.name] = future

            trace_rows[0]["jit_scheduled_count"] = jit_scheduled_count
            trace_rows[0]["jit_upsert_count"] = jit_upsert_count
            trace_rows[0]["jit_late_count"] = jit_late_count
            trace_rows[0]["jit_initial_late_count"] = jit_initial_late_count
            trace_rows[0]["jit_upsert_late_count"] = jit_upsert_late_count

            if not running:
                missing = sorted(set(workflow.nodes) - set(completed))
                error = f"workflow is stuck; remaining nodes: {missing}"
                if raise_on_error:
                    raise RuntimeError(error)
                workflow_end_ms = now_ms()
                trace_rows[0]["workflow_end_ms"] = workflow_end_ms
                trace_rows[0]["workflow_e2e_ms"] = workflow_end_ms - entry_ts_ms
                trace_rows[0]["dispatch_end_ms"] = workflow_end_ms
                trace_rows[0]["dispatch_latency_ms"] = workflow_end_ms - entry_ts_ms
                trace_rows[0]["status"] = "error"
                trace_rows[0]["error"] = error
                return trace_rows

            done, _ = wait(running.values(), return_when=FIRST_COMPLETED)
            for future in done:
                node_name = next(name for name, item in running.items() if item is future)
                row = future.result()
                row["stage_start_monotonic"] = started_at.get(node_name, "")
                trace_rows.append({k: v for k, v in row.items() if not k.startswith("_")})
                if row["status"] != "ok":
                    error = f"node {node_name} failed: {row['error']}"
                    if raise_on_error:
                        raise RuntimeError(error)
                    del running[node_name]
                    for pending in running.values():
                        pending.cancel()
                    workflow_end_ms = now_ms()
                    trace_rows[0]["workflow_end_ms"] = workflow_end_ms
                    trace_rows[0]["workflow_e2e_ms"] = workflow_end_ms - entry_ts_ms
                    trace_rows[0]["dispatch_end_ms"] = workflow_end_ms
                    trace_rows[0]["dispatch_latency_ms"] = workflow_end_ms - entry_ts_ms
                    trace_rows[0]["status"] = "error"
                    trace_rows[0]["error"] = error
                    return trace_rows
                measured_completion_at[node_name] = time.monotonic()
                completed[node_name] = row["_result"]
                del running[node_name]
                upsert_pending_jit_warmups()

    workflow_end_ms = now_ms()
    trace_rows[0]["workflow_end_ms"] = workflow_end_ms
    trace_rows[0]["workflow_e2e_ms"] = workflow_end_ms - entry_ts_ms
    trace_rows[0]["dispatch_end_ms"] = workflow_end_ms
    trace_rows[0]["dispatch_latency_ms"] = workflow_end_ms - entry_ts_ms
    trace_rows[0]["jit_scheduled_count"] = jit_scheduled_count
    trace_rows[0]["jit_upsert_count"] = jit_upsert_count
    trace_rows[0]["jit_late_count"] = jit_late_count
    trace_rows[0]["jit_initial_late_count"] = jit_initial_late_count
    trace_rows[0]["jit_upsert_late_count"] = jit_upsert_late_count
    return trace_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--apihost", required=True)
    parser.add_argument("--auth", required=True)
    parser.add_argument("--trace", default="data/traces.csv")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--interval-ms", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--allocated-memory-mb", type=int, default=None)
    parser.add_argument("--allocated-cpu-cores", type=float, default=None)
    parser.add_argument(
        "--cpu-profile",
        default="huawei_functiongraph",
        choices=["huawei_functiongraph", "huawei", "functiongraph", "legacy_256mb_250m", "openwhisk_256mb_250m", "custom"],
    )
    parser.add_argument(
        "--cpu-per-memory-mb",
        type=float,
        default=None,
        help="used only with --cpu-profile custom",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workflow = load_workflow(args.workflow)
    client = OpenWhiskClient(
        apihost=args.apihost,
        auth=args.auth,
        namespace=workflow.namespace,
    )
    store = CsvTraceStore(args.trace)
    allocated_cpu_cores = args.allocated_cpu_cores
    if allocated_cpu_cores is None and args.allocated_memory_mb is not None:
        allocated_cpu_cores = profile_memory_to_cpu_cores(
            args.allocated_memory_mb,
            profile=args.cpu_profile,
            cpu_per_memory_mb=args.cpu_per_memory_mb,
        )

    for idx in range(args.count):
        rows = run_one_workflow(
            workflow,
            client,
            args.max_workers,
            allocated_memory_mb=args.allocated_memory_mb,
            allocated_cpu_cores=allocated_cpu_cores,
        )
        store.append_many(rows)
        print(
            f"[{idx + 1}/{args.count}] workflow={workflow.workflow_name} "
            f"request_id={rows[0]['request_id']} rows={len(rows)}"
        )
        if args.interval_ms > 0:
            time.sleep(args.interval_ms / 1000.0)


if __name__ == "__main__":
    main()
