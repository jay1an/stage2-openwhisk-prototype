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
            **latency_fields(dispatch_start_ms, dispatch_end_ms),
            "allocated_memory_mb": allocated_memory_mb or "",
            "allocated_cpu_cores": allocated_cpu_cores or "",
            "status": "error",
            "error": str(exc),
            "_result": {},
        }


def run_one_workflow(
    workflow: WorkflowSpec,
    client: OpenWhiskClient,
    max_workers: int,
    allocated_memory_mb: int | None = None,
    allocated_cpu_cores: float | None = None,
    raise_on_error: bool = True,
    plan: dict[str, int] | None = None,
    slo_class: str | None = None,
) -> List[dict]:
    normalized_plan = validate_stage_plan(workflow, plan) if plan is not None else None
    request_id = str(uuid.uuid4())
    entry_ts_ms = now_ms()
    completed: Dict[str, dict] = {}
    running = {}
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
            "status": "ok",
            "error": "",
        }
    ]

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

                future = pool.submit(
                    invoke_node,
                    client,
                    workflow,
                    node,
                    request_id,
                    entry_ts_ms,
                    completed,
                    node_memory_mb,
                    node_cpu_cores,
                    node_action_name,
                    slo_class,
                )
                running[node.name] = future

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
                completed[node_name] = row["_result"]
                del running[node_name]

    workflow_end_ms = now_ms()
    trace_rows[0]["workflow_end_ms"] = workflow_end_ms
    trace_rows[0]["workflow_e2e_ms"] = workflow_end_ms - entry_ts_ms
    trace_rows[0]["dispatch_end_ms"] = workflow_end_ms
    trace_rows[0]["dispatch_latency_ms"] = workflow_end_ms - entry_ts_ms
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
