import argparse
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, List

from .openwhisk_client import OpenWhiskClient
from .trace_store import CsvTraceStore
from .workflow import NodeSpec, WorkflowSpec, load_workflow


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
) -> dict:
    dispatch_latency_ms = dispatch_end_ms - dispatch_start_ms
    action_duration = to_float_or_none(action_duration_ms)
    platform_overhead_ms = (
        dispatch_latency_ms - action_duration
        if action_duration is not None
        else ""
    )
    return {
        "dispatch_latency_ms": dispatch_latency_ms,
        "platform_overhead_ms": platform_overhead_ms,
    }


def invoke_node(
    client: OpenWhiskClient,
    workflow: WorkflowSpec,
    node: NodeSpec,
    request_id: str,
    entry_ts_ms: int,
    parent_results: Dict[str, dict],
) -> dict:
    dispatch_start_ms = now_ms()
    try:
        result = client.invoke_action(
            node.action,
            {
                "workflow_name": workflow.workflow_name,
                "request_id": request_id,
                "entry_ts_ms": entry_ts_ms,
                "stage_name": node.name,
                "parent_stages": node.parents,
                "sleep_ms": node.sleep_ms,
                "cpu_iters": node.cpu_iters,
                "memory_kb": node.memory_kb,
                "memory_passes": node.memory_passes,
                "memory_stride": node.memory_stride,
                "output_items": node.output_items,
                "payload": {
                    "parents": {
                        parent: parent_results.get(parent, {})
                        for parent in node.parents
                    }
                },
            },
        )
        dispatch_end_ms = now_ms()
        action_duration_ms = result.get("action_duration_ms", "")
        return {
            "workflow_name": workflow.workflow_name,
            "request_id": request_id,
            "stage_name": node.name,
            "parent_stages": ",".join(node.parents),
            "entry_ts_ms": entry_ts_ms,
            "dispatch_start_ms": dispatch_start_ms,
            "dispatch_end_ms": dispatch_end_ms,
            **latency_fields(
                dispatch_start_ms,
                dispatch_end_ms,
                action_duration_ms,
            ),
            "action_start_ns": result.get("action_start_ns", ""),
            "action_end_ns": result.get("action_end_ns", ""),
            "action_duration_ms": action_duration_ms,
            "container_id": result.get("container_id", ""),
            "cold_like": result.get("cold_like", ""),
            "status": "ok",
            "error": "",
            "_result": result,
        }
    except Exception as exc:
        dispatch_end_ms = now_ms()
        return {
            "workflow_name": workflow.workflow_name,
            "request_id": request_id,
            "stage_name": node.name,
            "parent_stages": ",".join(node.parents),
            "entry_ts_ms": entry_ts_ms,
            "dispatch_start_ms": dispatch_start_ms,
            "dispatch_end_ms": dispatch_end_ms,
            **latency_fields(dispatch_start_ms, dispatch_end_ms),
            "status": "error",
            "error": str(exc),
            "_result": {},
        }


def run_one_workflow(
    workflow: WorkflowSpec,
    client: OpenWhiskClient,
    max_workers: int,
) -> List[dict]:
    request_id = str(uuid.uuid4())
    entry_ts_ms = now_ms()
    completed: Dict[str, dict] = {}
    running = {}
    trace_rows: List[dict] = [
        {
            "workflow_name": workflow.workflow_name,
            "request_id": request_id,
            "stage_name": "__entry__",
            "parent_stages": "",
            "entry_ts_ms": entry_ts_ms,
            "dispatch_start_ms": entry_ts_ms,
            "dispatch_end_ms": entry_ts_ms,
            "dispatch_latency_ms": 0,
            "platform_overhead_ms": "",
            "status": "ok",
            "error": "",
        }
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while len(completed) < len(workflow.nodes):
            ready = workflow.ready_nodes(completed.keys(), running.keys())
            for node in ready:
                future = pool.submit(
                    invoke_node,
                    client,
                    workflow,
                    node,
                    request_id,
                    entry_ts_ms,
                    completed,
                )
                running[node.name] = future

            if not running:
                missing = sorted(set(workflow.nodes) - set(completed))
                raise RuntimeError(f"workflow is stuck; remaining nodes: {missing}")

            done, _ = wait(running.values(), return_when=FIRST_COMPLETED)
            for future in done:
                node_name = next(name for name, item in running.items() if item is future)
                row = future.result()
                trace_rows.append({k: v for k, v in row.items() if not k.startswith("_")})
                if row["status"] != "ok":
                    raise RuntimeError(f"node {node_name} failed: {row['error']}")
                completed[node_name] = row["_result"]
                del running[node_name]

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

    for idx in range(args.count):
        rows = run_one_workflow(workflow, client, args.max_workers)
        store.append_many(rows)
        print(
            f"[{idx + 1}/{args.count}] workflow={workflow.workflow_name} "
            f"request_id={rows[0]['request_id']} rows={len(rows)}"
        )
        if args.interval_ms > 0:
            time.sleep(args.interval_ms / 1000.0)


if __name__ == "__main__":
    main()
