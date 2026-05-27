#!/usr/bin/env python3
"""Prewarm resource-suffixed workflow actions by issuing concurrent warmup invokes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.openwhisk_client import OpenWhiskClient
from runner.workflow import load_workflow, with_action_suffix
from scripts.replay_civic_azure_schedule import auth_from_args


DEFAULT_WORKFLOW = ROOT / "configs" / "civic_alert_flow.yaml"


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def activation_annotations(activation: dict[str, Any]) -> dict[str, Any]:
    return {
        annotation.get("key"): annotation.get("value")
        for annotation in activation.get("annotations", [])
        if isinstance(annotation, dict) and annotation.get("key")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create warm containers for workflow actions. Use concurrent warmup "
            "invokes to create more than one container per action."
        )
    )
    parser.add_argument("--apihost", required=True)
    parser.add_argument(
        "--auth",
        default="",
        help="OpenWhisk AUTH; when omitted, read owdev-whisk.auth guest auth via kubectl.",
    )
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    parser.add_argument("--memory-mb", type=int, default=1280)
    parser.add_argument("--cpu-cores", type=float, default=1.0)
    parser.add_argument(
        "--containers-per-action",
        type=int,
        default=1,
        help="number of warm containers to create for each action",
    )
    parser.add_argument(
        "--warmup-hold-ms",
        type=int,
        default=3000,
        help="how long each warmup invocation holds the container busy",
    )
    parser.add_argument("--invoke-timeout-sec", type=int, default=120)
    parser.add_argument("--kube-namespace", default="openwhisk")
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument(
        "--skip-pod-query",
        action="store_true",
        help="do not query Kubernetes pod creation timestamps after warmup",
    )
    parser.add_argument(
        "--resource-action-suffix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="append _<memory-mb> to every action name",
    )
    parser.add_argument(
        "--action-suffix",
        default="",
        help="custom suffix appended to every action name; overrides --resource-action-suffix",
    )
    return parser.parse_args()


def resolve_action_suffix(args: argparse.Namespace) -> str:
    if args.action_suffix:
        return args.action_suffix
    if args.resource_action_suffix:
        return f"_{args.memory_mb}"
    return ""


def invoke_warmup(
    client: OpenWhiskClient,
    workflow_name: str,
    stage_name: str,
    action_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    invoke_start_ms = now_ms()
    activation = client.invoke_activation(
        action_name,
        {
            "__warmup": True,
            "workflow_name": workflow_name,
            "request_id": f"prewarm-{uuid.uuid4()}",
            "stage_name": stage_name,
            "warmup_hold_ms": args.warmup_hold_ms,
            "allocated_memory_mb": args.memory_mb,
            "allocated_cpu_cores": args.cpu_cores,
        },
    )
    invoke_end_ms = now_ms()
    response = activation.get("response", {})
    result = response.get("result", {}) if isinstance(response, dict) else {}
    if not isinstance(result, dict):
        result = {"raw_result": result}
    annotations = activation_annotations(activation)
    client_latency_ms = invoke_end_ms - invoke_start_ms
    action_duration_ms = as_float(result.get("action_duration_ms"))
    ow_wait_ms = as_float(annotations.get("waitTime"))
    ow_init_ms = as_float(annotations.get("initTime"))
    ow_duration_ms = as_float(activation.get("duration"))
    platform_overhead_ms = (
        client_latency_ms - action_duration_ms
        if action_duration_ms is not None
        else ""
    )
    client_gateway_overhead_ms = (
        client_latency_ms - ow_wait_ms - ow_duration_ms
        if ow_wait_ms is not None and ow_duration_ms is not None
        else ""
    )
    return {
        "action": action_name,
        "stage": stage_name,
        "status": response.get("status", "") if isinstance(response, dict) else "",
        "activation_id": activation.get("activationId", ""),
        "container_id": result.get("container_id", ""),
        "container_uptime_ms": result.get("container_uptime_ms", ""),
        "pod_name": result.get("pod_name", ""),
        "cold_like": result.get("cold_like", ""),
        "invoke_start_ms": invoke_start_ms,
        "invoke_end_ms": invoke_end_ms,
        "client_latency_ms": client_latency_ms,
        "action_duration_ms": result.get("action_duration_ms", ""),
        "ow_wait_ms": annotations.get("waitTime", ""),
        "ow_init_ms": annotations.get("initTime", ""),
        "ow_duration_ms": activation.get("duration", ""),
        "platform_overhead_ms": platform_overhead_ms,
        "client_gateway_overhead_ms": client_gateway_overhead_ms,
        "warmup_hold_ms": result.get("warmup_hold_ms", args.warmup_hold_ms),
        "error": result.get("error", ""),
    }


def pod_timestamps(args: argparse.Namespace, pod_names: list[str]) -> dict[str, dict[str, str]]:
    if args.skip_pod_query or not pod_names:
        return {}
    timestamps: dict[str, dict[str, str]] = {}
    for pod_name in sorted(set(pod_names)):
        result = subprocess.run(
            [
                args.kubectl,
                "get",
                "pod",
                pod_name,
                "-n",
                args.kube_namespace,
                "-o",
                "json",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            timestamps[pod_name] = {
                "creation_timestamp": "",
                "start_time": "",
                "query_error": result.stderr.strip(),
            }
            continue
        pod = json.loads(result.stdout)
        status = pod.get("status", {}) if isinstance(pod, dict) else {}
        metadata = pod.get("metadata", {}) if isinstance(pod, dict) else {}
        timestamps[pod_name] = {
            "creation_timestamp": str(metadata.get("creationTimestamp", "")),
            "start_time": str(status.get("startTime", "")),
            "query_error": "",
        }
    return timestamps


def mean_value(rows: list[dict[str, Any]], key: str) -> str:
    values = [as_float(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return ""
    return f"{sum(clean) / len(clean):.1f}"


def main() -> None:
    args = parse_args()
    if args.containers_per_action < 1:
        raise ValueError("--containers-per-action must be >= 1")

    auth = auth_from_args(args.auth)
    suffix = resolve_action_suffix(args)
    workflow = with_action_suffix(load_workflow(args.workflow), suffix)
    client = OpenWhiskClient(
        apihost=args.apihost,
        auth=auth,
        namespace=workflow.namespace,
        timeout_sec=args.invoke_timeout_sec,
    )

    jobs = []
    for node in workflow.nodes.values():
        for _ in range(args.containers_per_action):
            jobs.append((node.name, node.action))

    print(
        f"prewarming workflow={workflow.workflow_name} suffix={suffix or '<none>'} "
        f"containers_per_action={args.containers_per_action} "
        f"warmup_hold_ms={args.warmup_hold_ms}",
        flush=True,
    )

    rows = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [
            pool.submit(invoke_warmup, client, workflow.workflow_name, stage, action, args)
            for stage, action in jobs
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)

    pod_info = pod_timestamps(
        args,
        [str(row["pod_name"]) for row in rows if row.get("pod_name")],
    )
    for row in rows:
        info = pod_info.get(str(row.get("pod_name", "")), {})
        row["pod_creation_timestamp"] = info.get("creation_timestamp", "")
        row["pod_start_time"] = info.get("start_time", "")
        row["pod_query_error"] = info.get("query_error", "")
        print(
            f"{row['action']} status={row['status']} "
            f"cold_like={row['cold_like']} pod={row['pod_name']} "
            f"pod_created={row['pod_creation_timestamp']} "
            f"client_ms={row['client_latency_ms']} "
            f"ow_wait_ms={row['ow_wait_ms']} "
            f"ow_init_ms={row['ow_init_ms']} "
            f"ow_duration_ms={row['ow_duration_ms']} "
            f"action_ms={row['action_duration_ms']} "
            f"platform_overhead_ms={row['platform_overhead_ms']} "
            f"container_uptime_ms={row['container_uptime_ms']} "
            f"container={row['container_id']}",
            flush=True,
        )

    print("\nsummary:")
    for node in workflow.nodes.values():
        subset = [row for row in rows if row["action"] == node.action]
        pods = sorted({str(row["pod_name"]) for row in subset if row.get("pod_name")})
        containers = sorted(
            {str(row["container_id"]) for row in subset if row.get("container_id")}
        )
        print(
            f"{node.action}: invokes={len(subset)} "
            f"unique_pods={len(pods)} unique_containers={len(containers)} "
            f"client_mean_ms={mean_value(subset, 'client_latency_ms')} "
            f"ow_wait_mean_ms={mean_value(subset, 'ow_wait_ms')} "
            f"ow_init_mean_ms={mean_value(subset, 'ow_init_ms')} "
            f"platform_overhead_mean_ms={mean_value(subset, 'platform_overhead_ms')}"
        )
        for pod in pods:
            info = pod_info.get(pod, {})
            created = info.get("creation_timestamp", "")
            started = info.get("start_time", "")
            error = info.get("query_error", "")
            suffix = f" created={created} started={started}" if created or started else ""
            if error:
                suffix = f" query_error={error}"
            print(f"  pod={pod}{suffix}")


if __name__ == "__main__":
    main()
