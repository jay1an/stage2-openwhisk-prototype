#!/usr/bin/env python3
"""Replay a civic_alert Azure-derived schedule and summarize latency/cost."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.openwhisk_client import OpenWhiskClient
from runner.run_workflow import activation_annotations, run_one_workflow
from runner.trace_store import CsvTraceStore
from runner.workflow import WorkflowSpec, load_workflow, suffix_action_name, with_action_suffix


DEFAULT_SCHEDULE = (
    ROOT
    / "data"
    / "Azure schedule trace"
    / "schedule_cand2_30real_min_civic_alert_flow.csv"
)
DEFAULT_WORKFLOW = ROOT / "configs" / "civic_alert_flow.yaml"
DEFAULT_ACTION_FILE = ROOT / "actions" / "workflow_action.py"
DEFAULT_PLAN_CSV = ROOT / "reports" / "path3_planner" / "plan_summary.csv"
STAGE_ORDER = [
    "detect_object",
    "estimate_pose",
    "match_face",
    "classify_scene",
    "translate_alert",
]
WORKFLOW_COLUMNS = [
    "workflow_name",
    "index",
    "request_id",
    "source_label",
    "source_app",
    "source_func",
    "source_start_s",
    "source_end_s",
    "source_duration_ms",
    "source_target_offset_ms",
    "target_offset_ms",
    "target_ms",
    "submit_ms",
    "start_ms",
    "end_ms",
    "target_lag_ms",
    "status",
    "error",
    "stage_count",
    "cold_stage_count",
    "ow_cold_stage_count",
    "workflow_cold_class",
    "workflow_e2e_ms",
    "sum_stage_dispatch_ms",
    "sum_stage_action_ms",
    "sum_stage_ow_duration_ms",
    "sum_stage_wait_ms",
    "sum_stage_init_ms",
    "execution_gb_seconds",
    "execution_vcpu_seconds",
    "request_count",
    "memory_cost",
    "cpu_cost",
    "request_cost",
    "total_cost",
]
STAGE_DETAIL_COLUMNS = [
    "workflow_name",
    "index",
    "request_id",
    "workflow_cold_class",
    "stage_name",
    "stage_latency_class",
    "status",
    "error",
    "action_start_ns",
    "action_end_ns",
    "dispatch_latency_ms",
    "action_duration_ms",
    "platform_overhead_ms",
    "ow_wait_ms",
    "ow_init_ms",
    "ow_duration_ms",
    "ow_runtime_overhead_ms",
    "client_gateway_overhead_ms",
    "jit_sync_enabled",
    "jit_sync_waited_ms",
    "jit_sync_dispatch_after_warmup",
    "jit_sync_status",
    "jit_sync_warmup_issued_monotonic",
    "jit_sync_warmup_completed_monotonic",
    "serial_wall_ms",
    "io_wall_ms",
    "parallel_wall_ms",
    "memory_wall_ms",
    "cpu_process_ms",
    "observed_effective_cores",
    "container_id",
    "container_invocation_index",
    "container_uptime_ms",
    "previous_action_end_ns",
    "idle_since_prev_ms",
    "pod_name",
    "cold_like",
    "ow_cold_start",
    "execution_gb_seconds",
    "execution_vcpu_seconds",
    "total_cost",
]
CONTAINER_IDLE_COLUMNS = [
    "container_id",
    "stage_name",
    "invocation_count",
    "warm_reuse_count",
    "cold_like_count",
    "first_action_start_ns",
    "last_action_end_ns",
    "active_ms",
    "reported_between_idle_ms",
    "computed_between_idle_ms",
    "assumed_tail_idle_ms",
    "total_idle_ms",
    "idle_gb_seconds",
    "idle_vcpu_seconds",
]
PER_CLASS_SUMMARY_COLUMNS = [
    "slo_class",
    "slo_ms",
    "count",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "e2e_p99_ms",
    "slo_satisfaction_rate",
    "entry_cold_rate",
    "downstream_cold_rate",
    "cost_gbsec",
    "entry_prewarm_count",
    "entry_prewarm_gbsec",
    "jit_sync_waited_ms_mean",
    "jit_sync_status_counts",
    "dynamic_trigger_count",
    "total_upgrades",
]


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the already-compressed civic_alert Azure schedule without any "
            "additional time scaling. The default resource setting is 1280MiB/1vCPU."
        )
    )
    parser.add_argument("--apihost", required=True)
    parser.add_argument(
        "--auth",
        default="",
        help="OpenWhisk AUTH; when omitted, read owdev-whisk.auth guest auth via kubectl.",
    )
    parser.add_argument("--schedule", default=str(DEFAULT_SCHEDULE))
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    parser.add_argument("--action-file", default=str(DEFAULT_ACTION_FILE))
    parser.add_argument("--kind", default="python:3")
    parser.add_argument("--memory-mb", type=int, default=1280)
    parser.add_argument("--cpu-cores", type=float, default=1.0)
    parser.add_argument("--keepalive-sec", type=float, default=20.0)
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--invoke-timeout-sec", type=int, default=120)
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--stage-max-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 means replay all rows")
    parser.add_argument("--wsk-cli", default="wsk")
    parser.add_argument(
        "--resource-action-suffix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "append _<memory-mb> to every action name so each resource tier has "
            "its own deployed action variant"
        ),
    )
    parser.add_argument(
        "--action-suffix",
        default="",
        help="custom suffix appended to every action name; overrides --resource-action-suffix",
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="do not update/create civic actions before replay",
    )
    parser.add_argument(
        "--force-deploy",
        action="store_true",
        help="update civic actions even when memory/timeout/kind already match",
    )
    parser.add_argument("--price-per-vcpu-second", type=float, default=0.0)
    parser.add_argument("--price-per-gb-second", type=float, default=0.0)
    parser.add_argument("--price-per-request", type=float, default=0.0)
    parser.add_argument(
        "--out-dir",
        default="",
        help="defaults to reports/civic_azure_replay_<timestamp>",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="print progress every N submissions/completions",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="remove existing output files in --out-dir before running",
    )
    parser.add_argument("--plan-csv", default=str(DEFAULT_PLAN_CSV))
    parser.add_argument("--enable-jit", action="store_true", default=False)
    parser.add_argument("--enable-jit-sync", action="store_true", default=False)
    parser.add_argument("--jit-sync-pause-grace-ms", type=float, default=0.0)
    parser.add_argument("--enable-dynamic", action="store_true", default=False)
    parser.add_argument("--enable-entry-prewarm", action="store_true", default=False)
    parser.add_argument(
        "--entry-prewarm-lead-sec",
        type=float,
        default=2.5,
        help="oracle entry warmup lead time before each scheduled arrival",
    )
    parser.add_argument("--premium-ratio", type=float, default=0.5)
    parser.add_argument("--slo-class-seed", type=int, default=20260609)
    parser.add_argument("--slo-premium-ms", type=float, default=15000.0)
    parser.add_argument("--slo-free-ms", type=float, default=20000.0)
    return parser.parse_args()


def read_cluster_auth() -> str:
    encoded = subprocess.check_output(
        [
            "kubectl",
            "get",
            "secret",
            "owdev-whisk.auth",
            "-n",
            "openwhisk",
            "-o",
            "jsonpath={.data.guest}",
        ],
        text=True,
    ).strip()
    return subprocess.check_output(["base64", "-d"], input=encoded, text=True).strip()


def auth_from_args(value: str) -> str:
    return value or read_cluster_auth()


def load_schedule(path: str, limit: int) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"schedule is empty: {path}")
    if "target_offset_ms" not in rows[0]:
        raise ValueError("schedule CSV must contain target_offset_ms")
    rows.sort(
        key=lambda row: (
            int(float(row.get("index", 0))),
            int(float(row["target_offset_ms"])),
        )
    )
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


def parse_memory_config(value: str) -> dict[str, int]:
    plan: dict[str, int] = {}
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid memory_config item: {item!r}")
        stage_name, tier = item.split(":", 1)
        stage_name = stage_name.strip()
        try:
            plan[stage_name] = int(tier)
        except ValueError as exc:
            raise ValueError(f"invalid tier in memory_config item: {item!r}") from exc
    missing = sorted(set(STAGE_ORDER) - set(plan))
    unknown = sorted(set(plan) - set(STAGE_ORDER))
    if missing or unknown:
        raise ValueError(
            f"memory_config stage mismatch: missing={missing} unknown={unknown}"
        )
    return {stage: int(plan[stage]) for stage in STAGE_ORDER}


def load_plan_by_class(path: str | Path) -> dict[str, dict[str, int]]:
    plan_path = Path(path)
    with plan_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"plan CSV is empty: {plan_path}")
    required = {"slo_class", "memory_config"}
    missing_columns = sorted(required - set(rows[0]))
    if missing_columns:
        raise ValueError(f"plan CSV missing columns: {missing_columns}")

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        slo_class = str(row.get("slo_class", "")).strip().lower()
        if slo_class:
            by_class[slo_class].append(row)

    out: dict[str, dict[str, int]] = {}
    for slo_class in ["premium", "free"]:
        candidates = by_class.get(slo_class, [])
        if not candidates:
            raise ValueError(f"plan CSV has no row for slo_class={slo_class!r}")
        selected = next(
            (
                row
                for row in candidates
                if str(row.get("arrival_scenario", "")).strip().lower() == "typical"
            ),
            candidates[0],
        )
        out[slo_class] = parse_memory_config(str(selected.get("memory_config", "")))
    return out


def assign_slo_class(rng: random.Random, premium_ratio: float) -> str:
    if premium_ratio < 0.0 or premium_ratio > 1.0:
        raise ValueError(f"premium_ratio must be in [0, 1], got {premium_ratio}")
    return "premium" if rng.random() < float(premium_ratio) else "free"


def preassign_slo_classes(
    schedule_rows: list[dict[str, Any]],
    rng: random.Random,
    premium_ratio: float,
    slo_ms_by_class: dict[str, float],
) -> None:
    for row in schedule_rows:
        slo_class = assign_slo_class(rng, premium_ratio)
        row["slo_class"] = slo_class
        row["slo_ms"] = float(slo_ms_by_class[slo_class])


def cpu_cores_for_memory(memory_mb: int) -> float:
    from runner.stage4_risk.scaling import memory_to_cpu_cores

    return float(memory_to_cpu_cores(int(memory_mb)))


def resolve_action_suffix(args: argparse.Namespace) -> str:
    if args.action_suffix:
        return args.action_suffix
    if args.resource_action_suffix:
        return f"_{args.memory_mb}"
    return ""


def update_actions(
    args: argparse.Namespace,
    auth: str,
    action_names: list[str],
) -> None:
    source_sha256 = action_file_sha256(args.action_file)
    common = [
        args.wsk_cli,
        "-i",
        "--apihost",
        args.apihost,
        "--auth",
        auth,
        "action",
    ]
    for action in dict.fromkeys(action_names):
        get_result = subprocess.run(
            [*common, "get", action],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        action_spec = parse_wsk_action_get(get_result.stdout) if get_result.returncode == 0 else None
        operation = "update" if action_spec else "create"
        reasons = (
            action_update_reasons(action_spec, args, source_sha256)
            if action_spec
            else ["missing"]
        )
        if action_spec and not args.force_deploy and not reasons:
            print(
                f"  skip {action}: already memory={args.memory_mb}MiB "
                f"timeout={args.timeout_ms}ms kind={args.kind} source={source_sha256[:12]}",
                flush=True,
            )
            continue
        reason_text = "forced" if args.force_deploy and action_spec else ", ".join(reasons)
        print(f"  {operation} {action}: {reason_text}", flush=True)
        subprocess.run(
            [
                *common,
                operation,
                action,
                args.action_file,
                "--kind",
                args.kind,
                "--memory",
                str(args.memory_mb),
                "--timeout",
                str(args.timeout_ms),
                "--annotation",
                "source_sha256",
                source_sha256,
            ],
            stdout=subprocess.DEVNULL,
            check=True,
        )


def action_file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_wsk_action_get(stdout: str) -> dict[str, Any] | None:
    start = stdout.find("{")
    if start < 0:
        return None
    try:
        parsed = json.loads(stdout[start:])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def action_kind(action_spec: dict[str, Any]) -> str:
    exec_spec = action_spec.get("exec", {})
    if isinstance(exec_spec, dict) and exec_spec.get("kind"):
        return str(exec_spec["kind"])
    annotations = action_spec.get("annotations", [])
    if isinstance(annotations, list):
        for item in annotations:
            if isinstance(item, dict) and item.get("key") == "exec":
                return str(item.get("value", ""))
    return ""


def action_annotation(action_spec: dict[str, Any], key: str) -> str:
    annotations = action_spec.get("annotations", [])
    if not isinstance(annotations, list):
        return ""
    for item in annotations:
        if isinstance(item, dict) and item.get("key") == key:
            return str(item.get("value", ""))
    return ""


def action_update_reasons(
    action_spec: dict[str, Any] | None,
    args: argparse.Namespace,
    source_sha256: str,
) -> list[str]:
    if not action_spec:
        return ["missing"]
    reasons = []
    limits = action_spec.get("limits", {})
    memory = limits.get("memory") if isinstance(limits, dict) else None
    timeout = limits.get("timeout") if isinstance(limits, dict) else None
    if int(memory or -1) != int(args.memory_mb):
        reasons.append(f"memory {memory} != {args.memory_mb}")
    if int(timeout or -1) != int(args.timeout_ms):
        reasons.append(f"timeout {timeout} != {args.timeout_ms}")
    kind = action_kind(action_spec)
    if kind != args.kind:
        reasons.append(f"kind {kind or '<unknown>'} != {args.kind}")
    deployed_sha256 = action_annotation(action_spec, "source_sha256")
    if deployed_sha256 != source_sha256:
        reasons.append(
            f"source_sha256 {deployed_sha256[:12] or '<missing>'} != {source_sha256[:12]}"
        )
    return reasons


def as_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def stage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("stage_name") != "__entry__"]


def workflow_e2e_ms(rows: list[dict[str, Any]], fallback_ms: float) -> float:
    if rows:
        reported = rows[0].get("workflow_e2e_ms")
        if reported not in ("", None):
            return as_float(reported)
    return fallback_ms


def workflow_cold_class(stages: list[dict[str, Any]]) -> str:
    if not stages:
        return "error"
    if any(row.get("status") != "ok" for row in stages):
        return "error"
    cold_count = sum(
        1 for row in stages if as_bool(row.get("cold_like")) or as_bool(row.get("ow_cold_start"))
    )
    if cold_count == 0:
        return "all_warm"
    if cold_count == len(stages):
        return "all_cold"
    return "partial_cold"


def stage_latency_class(row: dict[str, Any]) -> str:
    if row.get("status") != "ok" or row.get("action_duration_ms") in ("", None):
        return "error"
    if as_bool(row.get("cold_like")) or as_bool(row.get("ow_cold_start")):
        return "cold"
    return "warm"


def stage_cost(
    row: dict[str, Any],
    *,
    memory_mb: int,
    cpu_cores: float,
    price_per_gb_second: float,
    price_per_vcpu_second: float,
    price_per_request: float,
) -> dict[str, float]:
    billed_ms = as_float(row.get("ow_duration_ms"))
    if billed_ms <= 0:
        billed_ms = as_float(row.get("action_duration_ms"))
    duration_sec = max(0.0, billed_ms) / 1000.0
    gb_seconds = duration_sec * (float(memory_mb) / 1024.0)
    vcpu_seconds = duration_sec * float(cpu_cores)
    memory_cost = gb_seconds * float(price_per_gb_second)
    cpu_cost = vcpu_seconds * float(price_per_vcpu_second)
    total_cost = memory_cost + cpu_cost + float(price_per_request)
    return {
        "execution_gb_seconds": gb_seconds,
        "execution_vcpu_seconds": vcpu_seconds,
        "memory_cost": memory_cost,
        "cpu_cost": cpu_cost,
        "request_cost": float(price_per_request),
        "total_cost": total_cost,
    }


def entry_prewarm_cost(
    record: dict[str, Any],
    *,
    price_per_gb_second: float,
    price_per_vcpu_second: float,
    price_per_request: float,
) -> dict[str, float]:
    memory_mb = int(as_float(record.get("tier_mb"), 0.0))
    if memory_mb <= 0:
        return {
            "execution_gb_seconds": 0.0,
            "execution_vcpu_seconds": 0.0,
            "memory_cost": 0.0,
            "cpu_cost": 0.0,
            "request_cost": float(price_per_request),
            "total_cost": float(price_per_request),
        }
    duration_ms = as_float(record.get("ow_duration_ms"))
    if duration_ms <= 0:
        duration_ms = as_float(record.get("action_duration_ms"))
    return stage_cost(
        {
            "ow_duration_ms": duration_ms,
            "action_duration_ms": duration_ms,
        },
        memory_mb=memory_mb,
        cpu_cores=cpu_cores_for_memory(memory_mb),
        price_per_gb_second=price_per_gb_second,
        price_per_vcpu_second=price_per_vcpu_second,
        price_per_request=price_per_request,
    )


class WarmupStatusTracker:
    def __init__(self):
        self.condition = threading.Condition()
        self.statuses: dict[tuple[str, str], dict[str, Any]] = {}

    def _status_locked(
        self, request_id: str, stage_name: str, tier: object
    ) -> dict[str, Any]:
        key = (str(request_id), str(stage_name), str(tier))
        if key not in self.statuses:
            self.statuses[key] = {
                "issued_monotonic": "",
                "completed_monotonic": "",
                "activation_id": "",
                "container_id": "",
                "error": "",
                "event": threading.Event(),
            }
        return self.statuses[key]

    def mark_issued(
        self, request_id: str, stage_name: str, tier: object, issued_monotonic: float
    ) -> None:
        with self.condition:
            status = self._status_locked(request_id, stage_name, tier)
            status["issued_monotonic"] = issued_monotonic
            status["completed_monotonic"] = ""
            status["activation_id"] = ""
            status["container_id"] = ""
            status["error"] = ""
            status["event"] = threading.Event()
            self.condition.notify_all()

    def mark_completed(
        self,
        request_id: str,
        stage_name: str,
        tier: object,
        completed_monotonic: float,
        activation_id: str = "",
        container_id: str = "",
        error: str = "",
    ) -> None:
        with self.condition:
            status = self._status_locked(request_id, stage_name, tier)
            status["completed_monotonic"] = completed_monotonic
            status["activation_id"] = activation_id
            status["container_id"] = container_id
            status["error"] = error
            status["event"].set()
            self.condition.notify_all()

    def get_status(self, request_id: str, stage_name: str, tier: object) -> dict[str, Any]:
        with self.condition:
            status = self._status_locked(request_id, stage_name, tier)
            return {key: value for key, value in status.items() if key != "event"}

    def wait_until_completed(
        self,
        request_id: str,
        stage_name: str,
        tier: object,
        timeout: float,
    ) -> dict[str, Any]:
        with self.condition:
            status = self._status_locked(request_id, stage_name, tier)
            event = status["event"]
        event.wait(timeout=max(0.0, timeout))
        return self.get_status(request_id, stage_name, tier)


def build_detail_records(
    *,
    source_row: dict[str, Any],
    schedule_row: dict[str, Any],
    rows: list[dict[str, Any]],
    memory_mb: int,
    cpu_cores: float,
    price_per_gb_second: float,
    price_per_vcpu_second: float,
    price_per_request: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stages = stage_rows(rows)
    e2e = workflow_e2e_ms(rows, schedule_row["end_ms"] - schedule_row["start_ms"])
    cold_class = "error" if schedule_row.get("status") != "ok" else workflow_cold_class(stages)
    stage_details = []

    totals = {
        "dispatch": 0.0,
        "action": 0.0,
        "ow_duration": 0.0,
        "wait": 0.0,
        "init": 0.0,
        "gb_seconds": 0.0,
        "vcpu_seconds": 0.0,
        "memory_cost": 0.0,
        "cpu_cost": 0.0,
        "request_cost": 0.0,
        "total_cost": 0.0,
    }

    for row in stages:
        row_memory_mb = int(as_float(row.get("allocated_memory_mb"), memory_mb))
        row_cpu_cores = as_float(row.get("allocated_cpu_cores"), cpu_cores)
        costs = stage_cost(
            row,
            memory_mb=row_memory_mb,
            cpu_cores=row_cpu_cores,
            price_per_gb_second=price_per_gb_second,
            price_per_vcpu_second=price_per_vcpu_second,
            price_per_request=price_per_request,
        )
        totals["dispatch"] += as_float(row.get("dispatch_latency_ms"))
        if row.get("status") == "ok":
            totals["action"] += as_float(row.get("action_duration_ms"))
            totals["ow_duration"] += as_float(row.get("ow_duration_ms"))
            totals["wait"] += as_float(row.get("ow_wait_ms"))
            totals["init"] += as_float(row.get("ow_init_ms"))
        totals["gb_seconds"] += costs["execution_gb_seconds"]
        totals["vcpu_seconds"] += costs["execution_vcpu_seconds"]
        totals["memory_cost"] += costs["memory_cost"]
        totals["cpu_cost"] += costs["cpu_cost"]
        totals["request_cost"] += costs["request_cost"]
        totals["total_cost"] += costs["total_cost"]

        latency_class = stage_latency_class(row)
        stage_details.append(
            {
                "workflow_name": schedule_row["workflow_name"],
                "index": schedule_row["index"],
                "request_id": schedule_row["request_id"],
                "workflow_cold_class": cold_class,
                "stage_name": row.get("stage_name", ""),
                "stage_latency_class": latency_class,
                "status": row.get("status", ""),
                "error": row.get("error", ""),
                "action_start_ns": row.get("action_start_ns", ""),
                "action_end_ns": row.get("action_end_ns", ""),
                "dispatch_latency_ms": row.get("dispatch_latency_ms", ""),
                "action_duration_ms": row.get("action_duration_ms", ""),
                "platform_overhead_ms": row.get("platform_overhead_ms", ""),
                "ow_wait_ms": row.get("ow_wait_ms", ""),
                "ow_init_ms": row.get("ow_init_ms", ""),
                "ow_duration_ms": row.get("ow_duration_ms", ""),
                "ow_runtime_overhead_ms": row.get("ow_runtime_overhead_ms", ""),
                "client_gateway_overhead_ms": row.get("client_gateway_overhead_ms", ""),
                "jit_sync_enabled": row.get("jit_sync_enabled", ""),
                "jit_sync_waited_ms": row.get("jit_sync_waited_ms", ""),
                "jit_sync_dispatch_after_warmup": row.get(
                    "jit_sync_dispatch_after_warmup",
                    "",
                ),
                "jit_sync_status": row.get("jit_sync_status", ""),
                "jit_sync_warmup_issued_monotonic": row.get(
                    "jit_sync_warmup_issued_monotonic",
                    "",
                ),
                "jit_sync_warmup_completed_monotonic": row.get(
                    "jit_sync_warmup_completed_monotonic",
                    "",
                ),
                "serial_wall_ms": row.get("serial_wall_ms", ""),
                "io_wall_ms": row.get("io_wall_ms", ""),
                "parallel_wall_ms": row.get("parallel_wall_ms", ""),
                "memory_wall_ms": row.get("memory_wall_ms", ""),
                "cpu_process_ms": row.get("cpu_process_ms", ""),
                "observed_effective_cores": row.get("observed_effective_cores", ""),
                "container_id": row.get("container_id", ""),
                "container_invocation_index": row.get("container_invocation_index", ""),
                "container_uptime_ms": row.get("container_uptime_ms", ""),
                "previous_action_end_ns": row.get("previous_action_end_ns", ""),
                "idle_since_prev_ms": row.get("idle_since_prev_ms", ""),
                "pod_name": row.get("pod_name", ""),
                "cold_like": row.get("cold_like", ""),
                "ow_cold_start": row.get("ow_cold_start", ""),
                "execution_gb_seconds": costs["execution_gb_seconds"],
                "execution_vcpu_seconds": costs["execution_vcpu_seconds"],
                "total_cost": costs["total_cost"],
            }
        )

    entry_prewarm_count = int(as_float(schedule_row.get("entry_prewarm_count"), 0.0))
    entry_prewarm_gb_seconds = as_float(schedule_row.get("entry_prewarm_gb_seconds"))
    entry_prewarm_vcpu_seconds = as_float(schedule_row.get("entry_prewarm_vcpu_seconds"))
    entry_prewarm_memory_cost = as_float(schedule_row.get("entry_prewarm_memory_cost"))
    entry_prewarm_cpu_cost = as_float(schedule_row.get("entry_prewarm_cpu_cost"))
    entry_prewarm_request_cost = as_float(schedule_row.get("entry_prewarm_request_cost"))
    entry_prewarm_total_cost = as_float(schedule_row.get("entry_prewarm_total_cost"))
    totals["gb_seconds"] += entry_prewarm_gb_seconds
    totals["vcpu_seconds"] += entry_prewarm_vcpu_seconds
    totals["memory_cost"] += entry_prewarm_memory_cost
    totals["cpu_cost"] += entry_prewarm_cpu_cost
    totals["request_cost"] += entry_prewarm_request_cost
    totals["total_cost"] += entry_prewarm_total_cost

    workflow_record = {
        "workflow_name": schedule_row["workflow_name"],
        "index": schedule_row["index"],
        "request_id": schedule_row["request_id"],
        "source_label": source_row.get("source_label", ""),
        "source_app": source_row.get("source_app", ""),
        "source_func": source_row.get("source_func", ""),
        "slo_class": schedule_row.get("slo_class", ""),
        "slo_ms": schedule_row.get("slo_ms", ""),
        "source_start_s": source_row.get("source_start_s", ""),
        "source_end_s": source_row.get("source_end_s", ""),
        "source_duration_ms": source_row.get("source_duration_ms", ""),
        "source_target_offset_ms": source_row.get("target_offset_ms", ""),
        "target_offset_ms": schedule_row["target_offset_ms"],
        "target_ms": schedule_row["target_ms"],
        "submit_ms": schedule_row["submit_ms"],
        "start_ms": schedule_row["start_ms"],
        "end_ms": schedule_row["end_ms"],
        "target_lag_ms": schedule_row["target_lag_ms"],
        "status": schedule_row["status"],
        "error": schedule_row["error"],
        "stage_count": len(stages),
        "cold_stage_count": sum(1 for row in stages if as_bool(row.get("cold_like"))),
        "ow_cold_stage_count": sum(1 for row in stages if as_bool(row.get("ow_cold_start"))),
        "workflow_cold_class": cold_class,
        "workflow_e2e_ms": e2e,
        "sum_stage_dispatch_ms": totals["dispatch"],
        "sum_stage_action_ms": totals["action"],
        "sum_stage_ow_duration_ms": totals["ow_duration"],
        "sum_stage_wait_ms": totals["wait"],
        "sum_stage_init_ms": totals["init"],
        "execution_gb_seconds": totals["gb_seconds"],
        "execution_vcpu_seconds": totals["vcpu_seconds"],
        "request_count": len(stages) + entry_prewarm_count,
        "memory_cost": totals["memory_cost"],
        "cpu_cost": totals["cpu_cost"],
        "request_cost": totals["request_cost"],
        "total_cost": totals["total_cost"],
        "dynamic_upgraded": schedule_row.get("dynamic_upgraded", False),
        "dynamic_upgrade_count": schedule_row.get("dynamic_upgrade_count", 0),
        "entry_prewarm_count": entry_prewarm_count,
        "entry_prewarm_status": schedule_row.get("entry_prewarm_status", ""),
        "entry_prewarm_action": schedule_row.get("entry_prewarm_action", ""),
        "entry_prewarm_tier_mb": schedule_row.get("entry_prewarm_tier_mb", ""),
        "entry_prewarm_ow_duration_ms": schedule_row.get("entry_prewarm_ow_duration_ms", ""),
        "entry_prewarm_gb_seconds": entry_prewarm_gb_seconds,
        "entry_prewarm_vcpu_seconds": entry_prewarm_vcpu_seconds,
    }
    return workflow_record, stage_details


def append_csv_row(path: Path, columns: list[str], row: dict[str, Any]) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def workflow_columns(include_slo: bool) -> list[str]:
    if not include_slo:
        return list(WORKFLOW_COLUMNS)
    columns = list(WORKFLOW_COLUMNS)
    insert_at = columns.index("source_start_s")
    for column in ["slo_class", "slo_ms"]:
        if column not in columns:
            columns.insert(insert_at, column)
            insert_at += 1
    for column in ["dynamic_upgraded", "dynamic_upgrade_count"]:
        if column not in columns:
            columns.append(column)
    for column in [
        "entry_prewarm_count",
        "entry_prewarm_status",
        "entry_prewarm_action",
        "entry_prewarm_tier_mb",
        "entry_prewarm_ow_duration_ms",
        "entry_prewarm_gb_seconds",
        "entry_prewarm_vcpu_seconds",
    ]:
        if column not in columns:
            columns.append(column)
    return columns


def stats(values: list[float]) -> dict[str, float]:
    clean = [float(value) for value in values]
    if not clean:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(clean),
        "mean": statistics.mean(clean),
        "min": min(clean),
        "max": max(clean),
    }


def percentile(values: list[float], fraction: float) -> float:
    clean = sorted(float(value) for value in values)
    if not clean:
        return 0.0
    if fraction <= 0.0:
        return clean[0]
    if fraction >= 1.0:
        return clean[-1]
    position = (len(clean) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def value_counts_json(rows: list[dict[str, Any]], field: str) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        value = str(row.get(field, "") or "")
        if value:
            counts[value] += 1
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def as_int_or_none(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def summarize_workflows(workflows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    group_names = ["all", "ok", "all_warm", "partial_cold", "all_cold", "error"]
    for group_name in group_names:
        if group_name == "all":
            subset = workflows
        elif group_name == "ok":
            subset = [row for row in workflows if row.get("status") == "ok"]
        elif group_name == "error":
            subset = [row for row in workflows if row.get("status") != "ok"]
        else:
            subset = [
                row for row in workflows
                if row.get("status") == "ok"
                and row.get("workflow_cold_class") == group_name
            ]
        e2e = stats([as_float(row.get("workflow_e2e_ms")) for row in subset])
        cost = stats([as_float(row.get("total_cost")) for row in subset])
        rows.append(
            {
                "workflow_cold_class": group_name,
                "count": len(subset),
                "e2e_mean_ms": e2e["mean"],
                "e2e_min_ms": e2e["min"],
                "e2e_max_ms": e2e["max"],
                "cost_mean": cost["mean"],
                "cost_min": cost["min"],
                "cost_max": cost["max"],
                "execution_gb_seconds_total": sum(
                    as_float(row.get("execution_gb_seconds")) for row in subset
                ),
                "execution_vcpu_seconds_total": sum(
                    as_float(row.get("execution_vcpu_seconds")) for row in subset
                ),
                "total_cost": sum(as_float(row.get("total_cost")) for row in subset),
            }
        )
    return rows


def summarize_stages(stage_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    keys = []
    for stage in STAGE_ORDER:
        keys.append((stage, "all"))
        keys.append((stage, "warm"))
        keys.append((stage, "cold"))
        keys.append((stage, "error"))
    for stage, latency_class in keys:
        subset = [
            row for row in stage_details
            if row.get("stage_name") == stage
            and (latency_class == "all" or row.get("stage_latency_class") == latency_class)
        ]
        dispatch = stats(numeric_values(subset, "dispatch_latency_ms"))
        action = stats(numeric_values(subset, "action_duration_ms"))
        wait_stats = stats(numeric_values(subset, "ow_wait_ms"))
        init_stats = stats(numeric_values(subset, "ow_init_ms"))
        idle_stats = stats(numeric_values(subset, "idle_since_prev_ms"))
        rows.append(
            {
                "stage_name": stage,
                "stage_latency_class": latency_class,
                "count": len(subset),
                "cold_count": sum(1 for row in subset if row.get("stage_latency_class") == "cold"),
                "warm_count": sum(1 for row in subset if row.get("stage_latency_class") == "warm"),
                "error_count": sum(1 for row in subset if row.get("stage_latency_class") == "error"),
                "dispatch_total_ms": sum(numeric_values(subset, "dispatch_latency_ms")),
                "dispatch_mean_ms": dispatch["mean"],
                "dispatch_min_ms": dispatch["min"],
                "dispatch_max_ms": dispatch["max"],
                "action_total_ms": sum(numeric_values(subset, "action_duration_ms")),
                "action_mean_ms": action["mean"],
                "action_min_ms": action["min"],
                "action_max_ms": action["max"],
                "ow_wait_total_ms": sum(numeric_values(subset, "ow_wait_ms")),
                "ow_wait_mean_ms": wait_stats["mean"],
                "ow_wait_min_ms": wait_stats["min"],
                "ow_wait_max_ms": wait_stats["max"],
                "ow_init_total_ms": sum(numeric_values(subset, "ow_init_ms")),
                "ow_init_mean_ms": init_stats["mean"],
                "ow_init_min_ms": init_stats["min"],
                "ow_init_max_ms": init_stats["max"],
                "reported_between_idle_total_ms": sum(numeric_values(subset, "idle_since_prev_ms")),
                "reported_between_idle_mean_ms": idle_stats["mean"],
                "reported_between_idle_min_ms": idle_stats["min"],
                "reported_between_idle_max_ms": idle_stats["max"],
                "execution_gb_seconds_total": sum(
                    as_float(row.get("execution_gb_seconds")) for row in subset
                ),
                "execution_vcpu_seconds_total": sum(
                    as_float(row.get("execution_vcpu_seconds")) for row in subset
                ),
                "total_cost": sum(as_float(row.get("total_cost")) for row in subset),
            }
        )
    return rows


def summarize_per_class(
    workflows: list[dict[str, Any]],
    stage_details: list[dict[str, Any]],
    slo_ms_by_class: dict[str, float],
) -> list[dict[str, Any]]:
    stages_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage_details:
        stages_by_request[str(row.get("request_id", ""))].append(row)

    rows = []
    for slo_class in ["premium", "free"]:
        subset = [
            row for row in workflows if str(row.get("slo_class", "")).lower() == slo_class
        ]
        e2e_values = numeric_values(subset, "workflow_e2e_ms")
        slo_ms = float(slo_ms_by_class[slo_class])
        satisfied = [
            as_float(row.get("workflow_e2e_ms")) <= slo_ms
            for row in subset
            if row.get("workflow_e2e_ms") not in ("", None)
        ]
        class_stage_rows = [
            stage_row
            for workflow_row in subset
            for stage_row in stages_by_request.get(str(workflow_row.get("request_id", "")), [])
        ]
        entry_rows = [
            row for row in class_stage_rows if row.get("stage_name") == STAGE_ORDER[0]
        ]
        downstream_rows = [
            row for row in class_stage_rows if row.get("stage_name") != STAGE_ORDER[0]
        ]
        sync_wait = numeric_values(downstream_rows, "jit_sync_waited_ms")
        rows.append(
            {
                "slo_class": slo_class,
                "slo_ms": slo_ms,
                "count": len(subset),
                "e2e_p50_ms": percentile(e2e_values, 0.50),
                "e2e_p95_ms": percentile(e2e_values, 0.95),
                "e2e_p99_ms": percentile(e2e_values, 0.99),
                "slo_satisfaction_rate": (
                    sum(1 for item in satisfied if item) / len(satisfied)
                    if satisfied
                    else 0.0
                ),
                "entry_cold_rate": (
                    sum(1 for row in entry_rows if row.get("stage_latency_class") == "cold")
                    / len(entry_rows)
                    if entry_rows
                    else 0.0
                ),
                "downstream_cold_rate": (
                    sum(
                        1
                        for row in downstream_rows
                        if row.get("stage_latency_class") == "cold"
                    )
                    / len(downstream_rows)
                    if downstream_rows
                    else 0.0
                ),
                "cost_gbsec": sum(as_float(row.get("execution_gb_seconds")) for row in subset),
                "entry_prewarm_count": sum(
                    int(as_float(row.get("entry_prewarm_count"))) for row in subset
                ),
                "entry_prewarm_gbsec": sum(
                    as_float(row.get("entry_prewarm_gb_seconds")) for row in subset
                ),
                "jit_sync_waited_ms_mean": statistics.mean(sync_wait) if sync_wait else 0.0,
                "jit_sync_status_counts": value_counts_json(
                    downstream_rows,
                    "jit_sync_status",
                ),
                "dynamic_trigger_count": sum(
                    1 for row in subset if as_bool(row.get("dynamic_upgraded"))
                ),
                "total_upgrades": sum(
                    int(as_float(row.get("dynamic_upgrade_count"))) for row in subset
                ),
            }
        )
    return rows


def build_container_idle_records(
    stage_details: list[dict[str, Any]],
    *,
    keepalive_sec: float,
    memory_mb: int,
    cpu_cores: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage_details:
        container_id = str(row.get("container_id") or "")
        if not container_id or row.get("status") != "ok":
            continue
        if as_int_or_none(row.get("action_start_ns")) is None:
            continue
        if as_int_or_none(row.get("action_end_ns")) is None:
            continue
        grouped[container_id].append(row)

    records = []
    memory_gb = float(memory_mb) / 1024.0
    for container_id, rows in grouped.items():
        rows.sort(key=lambda row: as_int_or_none(row.get("action_start_ns")) or 0)
        stages = sorted({str(row.get("stage_name") or "") for row in rows})
        active_ms = sum(numeric_values(rows, "action_duration_ms"))
        reported_values = numeric_values(rows, "idle_since_prev_ms")
        reported_between_idle_ms = sum(reported_values)

        computed_between_idle_ms = 0.0
        for previous, current in zip(rows, rows[1:]):
            previous_end = as_int_or_none(previous.get("action_end_ns"))
            current_start = as_int_or_none(current.get("action_start_ns"))
            if previous_end is not None and current_start is not None:
                computed_between_idle_ms += max(0.0, (current_start - previous_end) / 1_000_000.0)

        between_idle_for_cost_ms = (
            reported_between_idle_ms if reported_values else computed_between_idle_ms
        )
        assumed_tail_idle_ms = max(0.0, float(keepalive_sec) * 1000.0)
        total_idle_ms = between_idle_for_cost_ms + assumed_tail_idle_ms
        idle_sec = total_idle_ms / 1000.0
        first_start = as_int_or_none(rows[0].get("action_start_ns")) or ""
        last_end = as_int_or_none(rows[-1].get("action_end_ns")) or ""

        records.append(
            {
                "container_id": container_id,
                "stage_name": ",".join(stages),
                "invocation_count": len(rows),
                "warm_reuse_count": len(reported_values),
                "cold_like_count": sum(1 for row in rows if as_bool(row.get("cold_like"))),
                "first_action_start_ns": first_start,
                "last_action_end_ns": last_end,
                "active_ms": active_ms,
                "reported_between_idle_ms": reported_between_idle_ms,
                "computed_between_idle_ms": computed_between_idle_ms,
                "assumed_tail_idle_ms": assumed_tail_idle_ms,
                "total_idle_ms": total_idle_ms,
                "idle_gb_seconds": idle_sec * memory_gb,
                "idle_vcpu_seconds": 0.0,  # paused containers have CPU=0 (cgroup freezer)
            }
        )

    records.sort(key=lambda row: (str(row["stage_name"]), str(row["container_id"])))
    return records


def summarize_cost(
    workflows: list[dict[str, Any]],
    container_idle_records: list[dict[str, Any]],
    *,
    memory_mb: int,
    cpu_cores: float,
    keepalive_sec: float,
    price_per_gb_second: float,
    price_per_vcpu_second: float,
    price_per_request: float,
) -> list[dict[str, Any]]:
    execution_gb_seconds = sum(
        as_float(row.get("execution_gb_seconds")) for row in workflows
    )
    execution_vcpu_seconds = sum(
        as_float(row.get("execution_vcpu_seconds")) for row in workflows
    )
    idle_gb_seconds = sum(
        as_float(row.get("idle_gb_seconds")) for row in container_idle_records
    )
    idle_vcpu_seconds = sum(
        as_float(row.get("idle_vcpu_seconds")) for row in container_idle_records
    )
    execution_memory_cost = execution_gb_seconds * float(price_per_gb_second)
    execution_cpu_cost = execution_vcpu_seconds * float(price_per_vcpu_second)
    idle_memory_cost = idle_gb_seconds * float(price_per_gb_second)
    idle_cpu_cost = idle_vcpu_seconds * float(price_per_vcpu_second)
    request_cost = sum(as_float(row.get("request_count")) for row in workflows) * float(
        price_per_request
    )
    lambda_style_cost = execution_gb_seconds * float(price_per_gb_second) + request_cost
    provider_style_total_cost = (
        (execution_gb_seconds + idle_gb_seconds) * float(price_per_gb_second)
        + execution_vcpu_seconds * float(price_per_vcpu_second)
        + request_cost
    )
    return [
        {
            "memory_mb": memory_mb,
            "memory_gb": memory_mb / 1024.0,
            "cpu_cores": cpu_cores,
            "keepalive_sec": keepalive_sec,
            "workflow_count": len(workflows),
            "request_count": sum(as_float(row.get("request_count")) for row in workflows),
            "container_count": len(container_idle_records),
            "execution_gb_seconds_total": execution_gb_seconds,
            "execution_vcpu_seconds_total": execution_vcpu_seconds,
            "idle_gb_seconds_total": idle_gb_seconds,
            "idle_vcpu_seconds_total": idle_vcpu_seconds,
            "total_gb_seconds_including_idle": execution_gb_seconds + idle_gb_seconds,
            "total_vcpu_seconds_including_idle": execution_vcpu_seconds + idle_vcpu_seconds,
            "reported_between_idle_ms_total": sum(
                as_float(row.get("reported_between_idle_ms")) for row in container_idle_records
            ),
            "computed_between_idle_ms_total": sum(
                as_float(row.get("computed_between_idle_ms")) for row in container_idle_records
            ),
            "assumed_tail_idle_ms_total": sum(
                as_float(row.get("assumed_tail_idle_ms")) for row in container_idle_records
            ),
            "total_idle_ms": sum(
                as_float(row.get("total_idle_ms")) for row in container_idle_records
            ),
            "price_per_gb_second": price_per_gb_second,
            "price_per_vcpu_second": price_per_vcpu_second,
            "price_per_request": price_per_request,
            "lambda_style_gb_seconds": execution_gb_seconds,
            "lambda_style_cost": lambda_style_cost,
            "provider_style_total_cost": provider_style_total_cost,
            "execution_memory_cost_total": execution_memory_cost,
            "execution_cpu_cost_total": execution_cpu_cost,
            "idle_memory_cost_total": idle_memory_cost,
            "idle_cpu_cost_total": idle_cpu_cost,
            "memory_cost_total": execution_memory_cost + idle_memory_cost,
            "cpu_cost_total": execution_cpu_cost + idle_cpu_cost,
            "request_cost_total": request_cost,
            "execution_total_cost": execution_memory_cost + execution_cpu_cost + request_cost,
            "idle_total_cost": idle_memory_cost + idle_cpu_cost,
            "total_cost": (
                execution_memory_cost
                + execution_cpu_cost
                + idle_memory_cost
                + idle_cpu_cost
                + request_cost
            ),
        }
    ]


def run_scheduled_workflow(
    *,
    workflow: WorkflowSpec,
    apihost: str,
    auth: str,
    source_row: dict[str, Any],
    target_ms: int,
    target_offset_ms: int,
    submit_ms: int,
    stage_max_workers: int,
    invoke_timeout_sec: int,
    memory_mb: int,
    cpu_cores: float,
    plan: dict[str, int] | None = None,
    slo_class: str = "",
    slo_ms: float | None = None,
    jit_scheduler: object | None = None,
    enable_jit: bool = False,
    jit_warmup_tracker: object | None = None,
    enable_jit_sync: bool = False,
    jit_sync_pause_grace_ms: float = 0.0,
    enable_dynamic: bool = False,
    dynamic_config: object | None = None,
    dynamic_ref_data: object | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    client = OpenWhiskClient(
        apihost=apihost,
        auth=auth,
        namespace=workflow.namespace,
        timeout_sec=invoke_timeout_sec,
    )
    start_ms = now_ms()
    request_id = ""
    status = "ok"
    error = ""
    rows: list[dict[str, Any]] = []

    try:
        if plan is not None:
            rows = run_one_workflow(
                workflow,
                client,
                stage_max_workers,
                raise_on_error=False,
                plan=plan,
                slo_class=slo_class,
                jit_scheduler=jit_scheduler,
                enable_jit=enable_jit,
                jit_warmup_tracker=jit_warmup_tracker,
                enable_jit_sync=enable_jit_sync,
                jit_sync_pause_grace_ms=jit_sync_pause_grace_ms,
                enable_dynamic=enable_dynamic,
                dynamic_config=dynamic_config,
                dynamic_ref_data=dynamic_ref_data,
            )
        else:
            rows = run_one_workflow(
                workflow,
                client,
                stage_max_workers,
                allocated_memory_mb=memory_mb,
                allocated_cpu_cores=cpu_cores,
                raise_on_error=False,
            )
        request_id = rows[0]["request_id"]
        error_rows = [
            row for row in rows
            if row.get("stage_name") != "__entry__" and row.get("status") != "ok"
        ]
        if rows[0].get("status") != "ok":
            status = "error"
            error = str(rows[0].get("error", ""))
        elif error_rows:
            status = "error"
            error = str(error_rows[0].get("error", ""))
    except Exception as exc:
        status = "error"
        error = str(exc)

    end_ms = now_ms()
    dynamic_upgrade_count = 0
    for row in rows:
        raw = row.get("dynamic_upgrades", "")
        if raw in ("", None):
            continue
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            dynamic_upgrade_count += len(parsed)
    schedule_row = {
        "workflow_name": workflow.workflow_name,
        "index": int(float(source_row.get("index", 0))),
        "request_id": request_id,
        "slo_class": slo_class,
        "slo_ms": float(slo_ms) if slo_ms is not None else "",
        "target_ms": target_ms,
        "target_offset_ms": target_offset_ms,
        "submit_ms": submit_ms,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "target_lag_ms": start_ms - target_ms,
        "status": status,
        "error": error,
        "dynamic_upgraded": dynamic_upgrade_count > 0,
        "dynamic_upgrade_count": dynamic_upgrade_count,
        "entry_prewarm_count": int(as_float(source_row.get("entry_prewarm_count"), 0.0)),
        "entry_prewarm_action": source_row.get("entry_prewarm_action", ""),
        "entry_prewarm_tier_mb": source_row.get("entry_prewarm_tier_mb", ""),
    }
    return schedule_row, rows


def invoke_replay_warmup(
    client: OpenWhiskClient,
    task: Any,
    tracker: WarmupStatusTracker | None = None,
    warmup_records: dict[int, dict[str, Any]] | None = None,
    warmup_records_lock: threading.Lock | None = None,
) -> None:
    params = {
        "__warmup": True,
        "request_id": task.metadata.get("request_id", ""),
        "workflow_name": task.metadata.get("workflow_name", ""),
        "stage_name": task.metadata.get("stage_name", ""),
        "allocated_memory_mb": task.metadata.get("tier_mb", ""),
        "allocated_cpu_cores": task.metadata.get("cpu_cores", ""),
    }
    activation: dict[str, Any] = {}
    result: dict[str, Any] = {}
    annotations: dict[str, Any] = {}
    status = "ok"
    error = ""
    sent_ms = now_ms()
    warmup_sent_monotonic = time.monotonic()
    request_id = str(task.metadata.get("request_id", ""))
    stage_name = str(task.metadata.get("stage_name", ""))
    warmup_tier = str(task.metadata.get("tier_mb", ""))
    if tracker is not None and request_id and stage_name:
        tracker.mark_issued(request_id, stage_name, warmup_tier, warmup_sent_monotonic)
    try:
        activation = client.invoke_activation(task.action_name, params)
        response = activation.get("response", {})
        result = response.get("result", {}) if isinstance(response, dict) else {}
        if not isinstance(result, dict):
            result = {"error": result}
        annotations = activation_annotations(activation)
        activation_status = response.get("status", "") if isinstance(response, dict) else ""
        if not response:
            status = "error"
            error = "activation did not return a completed response"
        elif activation_status and activation_status != "success":
            status = "error"
            error = str(result.get("error", ""))
    except Exception as exc:
        status = "error"
        error = str(exc)
        print(
            f"warmup failed action={task.action_name} task={task.task_key}: {exc}",
            file=sys.stderr,
            flush=True,
        )
    ended_ms = now_ms()
    completed_monotonic = time.monotonic()
    if tracker is not None and request_id and stage_name:
        tracker.mark_completed(
            request_id,
            stage_name,
            warmup_tier,
            completed_monotonic,
            activation_id=activation.get("activationId", ""),
            container_id=result.get("container_id", ""),
            error=error,
        )
    if (
        warmup_records is not None
        and warmup_records_lock is not None
        and task.metadata.get("prewarm_kind") == "entry_oracle"
    ):
        try:
            schedule_index = int(float(task.metadata.get("schedule_index", "")))
        except (TypeError, ValueError):
            schedule_index = -1
        if schedule_index >= 0:
            record = {
                "task_key": task.task_key,
                "schedule_index": schedule_index,
                "slo_class": task.metadata.get("slo_class", ""),
                "action_name": task.action_name,
                "stage_name": task.metadata.get("stage_name", ""),
                "tier_mb": task.metadata.get("tier_mb", ""),
                "cpu_cores": task.metadata.get("cpu_cores", ""),
                "status": status,
                "error": error,
                "sent_ms": sent_ms,
                "ended_ms": ended_ms,
                "activation_id": activation.get("activationId", ""),
                "ow_duration_ms": activation.get("duration", ""),
                "action_duration_ms": result.get("action_duration_ms", ""),
                "ow_wait_ms": annotations.get("waitTime", ""),
                "ow_init_ms": annotations.get("initTime", ""),
                "container_id": result.get("container_id", ""),
                "cold_like": result.get("cold_like", ""),
            }
            with warmup_records_lock:
                warmup_records[schedule_index] = record


def make_replay_warmup_callback(
    client: OpenWhiskClient,
    tracker: WarmupStatusTracker | None = None,
    warmup_records: dict[int, dict[str, Any]] | None = None,
    warmup_records_lock: threading.Lock | None = None,
):
    def callback(task: Any) -> None:
        worker = threading.Thread(
            target=invoke_replay_warmup,
            args=(client, task, tracker, warmup_records, warmup_records_lock),
            daemon=True,
        )
        worker.start()

    return callback


def schedule_entry_prewarm_tasks(
    *,
    scheduler: Any,
    workflow: WorkflowSpec,
    schedule_rows: list[dict[str, Any]],
    plan_by_class: dict[str, dict[str, int]],
    base_monotonic: float,
    lead_sec: float,
) -> int:
    from runner.stage5_control.jit_scheduler import WarmupTask

    entry_node = workflow.nodes[workflow.entry]
    scheduled = 0
    for row in schedule_rows:
        slo_class = str(row.get("slo_class", "")).strip().lower()
        if not slo_class:
            raise ValueError("entry prewarm requires preassigned slo_class")
        tier_mb = int(plan_by_class[slo_class][entry_node.name])
        target_offset_sec = int(float(row["target_offset_ms"])) / 1000.0
        fire_time = base_monotonic + max(0.0, target_offset_sec - float(lead_sec))
        schedule_index = int(float(row.get("index", scheduled)))
        action_name = suffix_action_name(entry_node.action, f"_{tier_mb}")
        cpu_cores = cpu_cores_for_memory(tier_mb)
        task = WarmupTask(
            task_key=f"entry:{schedule_index}:{slo_class}:{tier_mb}",
            fire_time=fire_time,
            action_name=action_name,
            metadata={
                "prewarm_kind": "entry_oracle",
                "request_id": f"entry-prewarm-{schedule_index}",
                "workflow_name": workflow.workflow_name,
                "stage_name": entry_node.name,
                "tier_mb": tier_mb,
                "cpu_cores": cpu_cores,
                "slo_class": slo_class,
                "schedule_index": schedule_index,
                "target_offset_ms": int(float(row["target_offset_ms"])),
                "lead_sec": float(lead_sec),
            },
        )
        row["entry_prewarm_count"] = 1
        row["entry_prewarm_action"] = action_name
        row["entry_prewarm_tier_mb"] = tier_mb
        scheduler.schedule(task)
        scheduled += 1
    return scheduled


def prepare_outputs(
    out_dir: Path,
    overwrite: bool,
    include_per_class: bool = False,
) -> dict[str, Path]:
    paths = {
        "trace": out_dir / "raw_trace.csv",
        "workflow_detail": out_dir / "workflow_detail.csv",
        "stage_detail": out_dir / "stage_detail.csv",
        "container_idle_detail": out_dir / "container_idle_detail.csv",
        "workflow_summary": out_dir / "workflow_summary.csv",
        "stage_summary": out_dir / "stage_summary.csv",
        "cost_summary": out_dir / "cost_summary.csv",
        "metadata": out_dir / "run_metadata.json",
    }
    if include_per_class:
        paths["per_class_summary"] = out_dir / "per_class_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output files already exist; pass --overwrite to replace: {names}")
    if overwrite:
        for path in existing:
            path.unlink()
    return paths


def drain_completed(
    active: set[Future],
    *,
    block: bool,
    total: int,
    completed_count: int,
    store: CsvTraceStore,
    paths: dict[str, Path],
    workflow_records: list[dict[str, Any]],
    stage_records: list[dict[str, Any]],
    source_rows_by_index: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    include_slo: bool = False,
    entry_prewarm_records: dict[int, dict[str, Any]] | None = None,
    entry_prewarm_records_lock: threading.Lock | None = None,
) -> tuple[int, int]:
    if not active:
        return completed_count, 0
    if block:
        done, _ = wait(active, return_when=FIRST_COMPLETED)
    else:
        done = {future for future in active if future.done()}
    if not done:
        return completed_count, 0
    active.difference_update(done)

    processed = 0
    for future in done:
        schedule_row, rows = future.result()
        index = int(schedule_row["index"])
        source_row = source_rows_by_index[index]
        if (
            entry_prewarm_records is not None
            and entry_prewarm_records_lock is not None
            and int(as_float(schedule_row.get("entry_prewarm_count"), 0.0)) > 0
        ):
            with entry_prewarm_records_lock:
                warmup_record = dict(entry_prewarm_records.get(index, {}))
            if warmup_record:
                costs = entry_prewarm_cost(
                    warmup_record,
                    price_per_gb_second=args.price_per_gb_second,
                    price_per_vcpu_second=args.price_per_vcpu_second,
                    price_per_request=args.price_per_request,
                )
                schedule_row["entry_prewarm_status"] = warmup_record.get("status", "")
                schedule_row["entry_prewarm_action"] = warmup_record.get("action_name", "")
                schedule_row["entry_prewarm_tier_mb"] = warmup_record.get("tier_mb", "")
                schedule_row["entry_prewarm_ow_duration_ms"] = warmup_record.get(
                    "ow_duration_ms",
                    "",
                )
                schedule_row["entry_prewarm_gb_seconds"] = costs["execution_gb_seconds"]
                schedule_row["entry_prewarm_vcpu_seconds"] = costs["execution_vcpu_seconds"]
                schedule_row["entry_prewarm_memory_cost"] = costs["memory_cost"]
                schedule_row["entry_prewarm_cpu_cost"] = costs["cpu_cost"]
                schedule_row["entry_prewarm_request_cost"] = costs["request_cost"]
                schedule_row["entry_prewarm_total_cost"] = costs["total_cost"]
            else:
                schedule_row["entry_prewarm_status"] = "missing_record"
                schedule_row["entry_prewarm_request_cost"] = args.price_per_request
                schedule_row["entry_prewarm_total_cost"] = args.price_per_request
        if rows:
            store.append_many(rows)
        workflow_record, stage_detail = build_detail_records(
            source_row=source_row,
            schedule_row=schedule_row,
            rows=rows,
            memory_mb=args.memory_mb,
            cpu_cores=args.cpu_cores,
            price_per_gb_second=args.price_per_gb_second,
            price_per_vcpu_second=args.price_per_vcpu_second,
            price_per_request=args.price_per_request,
        )
        append_csv_row(paths["workflow_detail"], workflow_columns(include_slo), workflow_record)
        for row in stage_detail:
            append_csv_row(paths["stage_detail"], STAGE_DETAIL_COLUMNS, row)
        workflow_records.append(workflow_record)
        stage_records.extend(stage_detail)
        completed_count += 1
        processed += 1
        if completed_count % max(1, args.progress_every) == 0 or completed_count == total:
            print(
                f"completed [{completed_count}/{total}] "
                f"index={index} status={workflow_record['status']} "
                f"class={workflow_record['workflow_cold_class']} "
                f"e2e_ms={as_float(workflow_record['workflow_e2e_ms']):.1f} "
                f"target_lag_ms={as_float(workflow_record['target_lag_ms']):.1f}",
                flush=True,
            )
    return completed_count, processed


def main() -> None:
    args = parse_args()
    if args.max_inflight < 1:
        raise ValueError("--max-inflight must be >= 1")
    if args.stage_max_workers < 1:
        raise ValueError("--stage-max-workers must be >= 1")
    if args.premium_ratio < 0.0 or args.premium_ratio > 1.0:
        raise ValueError("--premium-ratio must be in [0, 1]")
    if args.enable_jit_sync and not args.enable_jit:
        raise ValueError("--enable-jit-sync requires --enable-jit")
    if args.jit_sync_pause_grace_ms < 0.0:
        raise ValueError("--jit-sync-pause-grace-ms must be >= 0")
    if args.entry_prewarm_lead_sec < 0.0:
        raise ValueError("--entry-prewarm-lead-sec must be >= 0")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "reports" / f"civic_azure_replay_{run_id}"
    multi_slo_active = bool(args.enable_jit or args.enable_dynamic or args.enable_entry_prewarm)
    paths = prepare_outputs(out_dir, args.overwrite, include_per_class=multi_slo_active)
    auth = auth_from_args(args.auth)
    base_workflow = load_workflow(args.workflow)
    action_suffix = "" if multi_slo_active else resolve_action_suffix(args)
    workflow = with_action_suffix(base_workflow, action_suffix)
    schedule_rows = load_schedule(args.schedule, args.limit)
    source_rows_by_index = {
        int(float(row.get("index", idx))): row for idx, row in enumerate(schedule_rows)
    }
    plan_by_class: dict[str, dict[str, int]] = {}
    slo_ms_by_class = {
        "premium": float(args.slo_premium_ms),
        "free": float(args.slo_free_ms),
    }
    slo_rng = random.Random(args.slo_class_seed)
    dynamic_ref_data = None
    dynamic_config_by_class: dict[str, Any] = {}
    jit_scheduler = None
    jit_warmup_tracker = WarmupStatusTracker() if args.enable_jit_sync else None
    entry_prewarm_records: dict[int, dict[str, Any]] = {}
    entry_prewarm_records_lock = threading.Lock()
    entry_prewarm_scheduled = 0

    if multi_slo_active:
        plan_by_class = load_plan_by_class(args.plan_csv)
        preassign_slo_classes(schedule_rows, slo_rng, args.premium_ratio, slo_ms_by_class)

    if args.enable_dynamic:
        from runner.stage5_control.multi_slo_planner import (
            DEFAULT_SAFETY_FACTORS,
            DEFAULT_TIERS,
            STAGES,
            PlannerConfig,
            load_reference_data,
        )

        dynamic_ref_data = load_reference_data()
        dynamic_config_by_class = {
            slo_class: PlannerConfig(
                slo_ms=slo_ms,
                max_violation_rate=0.05,
                predicted_arrivals=5.0,
                tiers=list(DEFAULT_TIERS),
                safety_factors=list(DEFAULT_SAFETY_FACTORS),
                stages=list(STAGES),
            )
            for slo_class, slo_ms in slo_ms_by_class.items()
        }

    if args.enable_jit or args.enable_entry_prewarm:
        from runner.stage5_control.jit_scheduler import JitScheduler

        warmup_client = OpenWhiskClient(
            apihost=args.apihost,
            auth=auth,
            namespace=workflow.namespace,
            timeout_sec=args.invoke_timeout_sec,
        )
        jit_scheduler = JitScheduler(
            make_replay_warmup_callback(
                warmup_client,
                jit_warmup_tracker,
                entry_prewarm_records,
                entry_prewarm_records_lock,
            )
        )
        jit_scheduler.start()

    if not args.skip_deploy and not multi_slo_active:
        action_names = [node.action for node in workflow.nodes.values()]
        print(
            f"checking {len(dict.fromkeys(action_names))} actions for memory={args.memory_mb}MiB "
            f"timeout={args.timeout_ms}ms kind={args.kind}",
            flush=True,
        )
        update_actions(args, auth, action_names)
    elif not args.skip_deploy and multi_slo_active:
        print(
            "multi-SLO plan mode uses predeployed per-tier variants; "
            "skipping uniform action deployment",
            flush=True,
        )

    metadata = {
        "run_id": run_id,
        "workflow": str(args.workflow),
        "action_suffix": action_suffix,
        "resource_action_suffix": args.resource_action_suffix,
        "schedule": str(args.schedule),
        "out_dir": str(out_dir),
        "memory_mb": args.memory_mb,
        "cpu_cores": args.cpu_cores,
        "multi_slo_active": multi_slo_active,
        "enable_jit": args.enable_jit,
        "enable_jit_sync": args.enable_jit_sync,
        "jit_sync_pause_grace_ms": args.jit_sync_pause_grace_ms,
        "enable_dynamic": args.enable_dynamic,
        "enable_entry_prewarm": args.enable_entry_prewarm,
        "entry_prewarm_lead_sec": args.entry_prewarm_lead_sec,
        "plan_csv": str(args.plan_csv),
        "premium_ratio": args.premium_ratio,
        "slo_class_seed": args.slo_class_seed,
        "slo_premium_ms": args.slo_premium_ms,
        "slo_free_ms": args.slo_free_ms,
        "plans_by_class": plan_by_class,
        "keepalive_sec": args.keepalive_sec,
        "max_inflight": args.max_inflight,
        "stage_max_workers": args.stage_max_workers,
        "workflow_count": len(schedule_rows),
        "target_offset_max_ms": int(float(schedule_rows[-1]["target_offset_ms"])),
        "note": "target_offset_ms is replayed as-is; no extra time scaling is applied.",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"workflow={workflow.workflow_name}")
    print(f"action_suffix={action_suffix or '<none>'}")
    print(f"schedule={args.schedule}")
    print(f"out_dir={out_dir}")
    print(f"workflows={len(schedule_rows)}")
    print(f"replay_span_s={metadata['target_offset_max_ms'] / 1000.0:.3f}")
    print(f"resource={args.memory_mb}MiB/{args.cpu_cores}vCPU keepalive={args.keepalive_sec}s")
    print(f"max_inflight={args.max_inflight}; stage_max_workers={args.stage_max_workers}")
    if multi_slo_active:
        print(
            f"multi_slo=on enable_jit={args.enable_jit} enable_dynamic={args.enable_dynamic} "
            f"enable_jit_sync={args.enable_jit_sync} "
            f"enable_entry_prewarm={args.enable_entry_prewarm} "
            f"premium_ratio={args.premium_ratio} seed={args.slo_class_seed}"
        )
    print("No additional schedule compression will be applied.")

    store = CsvTraceStore(str(paths["trace"]))
    workflow_records: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    active: set[Future] = set()
    submitted = 0
    completed = 0
    base_ms = now_ms()
    base_monotonic = time.monotonic()
    if args.enable_entry_prewarm:
        if jit_scheduler is None:
            raise RuntimeError("entry prewarm requires an active JIT scheduler")
        entry_prewarm_scheduled = schedule_entry_prewarm_tasks(
            scheduler=jit_scheduler,
            workflow=base_workflow,
            schedule_rows=schedule_rows,
            plan_by_class=plan_by_class,
            base_monotonic=base_monotonic,
            lead_sec=args.entry_prewarm_lead_sec,
        )
        metadata["entry_prewarm_scheduled"] = entry_prewarm_scheduled
        paths["metadata"].write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            f"entry_prewarm=on scheduled={entry_prewarm_scheduled} "
            f"lead_sec={args.entry_prewarm_lead_sec}",
            flush=True,
        )

    try:
        with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
            for source_row in schedule_rows:
                target_offset_ms = int(float(source_row["target_offset_ms"]))
                target_ms = base_ms + target_offset_ms
                while True:
                    completed, _ = drain_completed(
                        active,
                        block=False,
                        total=len(schedule_rows),
                        completed_count=completed,
                        store=store,
                        paths=paths,
                        workflow_records=workflow_records,
                        stage_records=stage_records,
                        source_rows_by_index=source_rows_by_index,
                        args=args,
                        include_slo=multi_slo_active,
                        entry_prewarm_records=entry_prewarm_records,
                        entry_prewarm_records_lock=entry_prewarm_records_lock,
                    )
                    delay_ms = target_ms - now_ms()
                    if delay_ms <= 0:
                        break
                    time.sleep(min(delay_ms / 1000.0, 0.25))

                while len(active) >= args.max_inflight:
                    completed, _ = drain_completed(
                        active,
                        block=True,
                        total=len(schedule_rows),
                        completed_count=completed,
                        store=store,
                        paths=paths,
                        workflow_records=workflow_records,
                        stage_records=stage_records,
                        source_rows_by_index=source_rows_by_index,
                        args=args,
                        include_slo=multi_slo_active,
                        entry_prewarm_records=entry_prewarm_records,
                        entry_prewarm_records_lock=entry_prewarm_records_lock,
                    )

                plan = None
                slo_class = ""
                slo_ms = None
                dynamic_config = None
                if multi_slo_active:
                    slo_class = str(source_row.get("slo_class", "")).strip().lower()
                    slo_ms = float(source_row.get("slo_ms", slo_ms_by_class[slo_class]))
                    plan = plan_by_class[slo_class]
                    dynamic_config = dynamic_config_by_class.get(slo_class)

                submit_ms = now_ms()
                future = pool.submit(
                    run_scheduled_workflow,
                    workflow=workflow,
                    apihost=args.apihost,
                    auth=auth,
                    source_row=source_row,
                    target_ms=target_ms,
                    target_offset_ms=target_offset_ms,
                    submit_ms=submit_ms,
                    stage_max_workers=args.stage_max_workers,
                    invoke_timeout_sec=args.invoke_timeout_sec,
                    memory_mb=args.memory_mb,
                    cpu_cores=args.cpu_cores,
                    plan=plan,
                    slo_class=slo_class,
                    slo_ms=slo_ms,
                    jit_scheduler=jit_scheduler,
                    enable_jit=args.enable_jit,
                    jit_warmup_tracker=jit_warmup_tracker,
                    enable_jit_sync=args.enable_jit_sync,
                    jit_sync_pause_grace_ms=args.jit_sync_pause_grace_ms,
                    enable_dynamic=args.enable_dynamic,
                    dynamic_config=dynamic_config,
                    dynamic_ref_data=dynamic_ref_data,
                )
                active.add(future)
                submitted += 1
                if submitted % max(1, args.progress_every) == 0 or submitted == len(schedule_rows):
                    print(
                        f"submitted [{submitted}/{len(schedule_rows)}] "
                        f"target_offset_ms={target_offset_ms} "
                        f"active={len(active)}",
                        flush=True,
                    )

            while active:
                completed, _ = drain_completed(
                    active,
                    block=True,
                    total=len(schedule_rows),
                    completed_count=completed,
                    store=store,
                    paths=paths,
                    workflow_records=workflow_records,
                    stage_records=stage_records,
                    source_rows_by_index=source_rows_by_index,
                    args=args,
                    include_slo=multi_slo_active,
                    entry_prewarm_records=entry_prewarm_records,
                    entry_prewarm_records_lock=entry_prewarm_records_lock,
                )
    finally:
        if jit_scheduler is not None:
            jit_scheduler.stop()

    workflow_summary = summarize_workflows(workflow_records)
    stage_summary = summarize_stages(stage_records)
    container_idle_records = build_container_idle_records(
        stage_records,
        keepalive_sec=args.keepalive_sec,
        memory_mb=args.memory_mb,
        cpu_cores=args.cpu_cores,
    )
    cost_summary = summarize_cost(
        workflow_records,
        container_idle_records,
        memory_mb=args.memory_mb,
        cpu_cores=args.cpu_cores,
        keepalive_sec=args.keepalive_sec,
        price_per_gb_second=args.price_per_gb_second,
        price_per_vcpu_second=args.price_per_vcpu_second,
        price_per_request=args.price_per_request,
    )
    write_csv(paths["workflow_summary"], list(workflow_summary[0]), workflow_summary)
    write_csv(paths["stage_summary"], list(stage_summary[0]), stage_summary)
    write_csv(paths["container_idle_detail"], CONTAINER_IDLE_COLUMNS, container_idle_records)
    write_csv(paths["cost_summary"], list(cost_summary[0]), cost_summary)
    if multi_slo_active:
        per_class_summary = summarize_per_class(
            workflow_records,
            stage_records,
            slo_ms_by_class,
        )
        write_csv(
            paths["per_class_summary"],
            PER_CLASS_SUMMARY_COLUMNS,
            per_class_summary,
        )

    overall = next(row for row in workflow_summary if row["workflow_cold_class"] == "all")
    successful = next(row for row in workflow_summary if row["workflow_cold_class"] == "ok")
    failed = next(row for row in workflow_summary if row["workflow_cold_class"] == "error")
    cost_overall = cost_summary[0]
    print("\nReplay complete.")
    print(f"raw_trace={paths['trace']}")
    print(f"workflow_detail={paths['workflow_detail']}")
    print(f"stage_detail={paths['stage_detail']}")
    print(f"container_idle_detail={paths['container_idle_detail']}")
    print(f"workflow_summary={paths['workflow_summary']}")
    print(f"stage_summary={paths['stage_summary']}")
    print(f"cost_summary={paths['cost_summary']}")
    if multi_slo_active:
        print(f"per_class_summary={paths['per_class_summary']}")
    print(
        "overall: "
        f"count={overall['count']} "
        f"e2e_mean_ms={overall['e2e_mean_ms']:.1f} "
        f"e2e_min_ms={overall['e2e_min_ms']:.1f} "
        f"e2e_max_ms={overall['e2e_max_ms']:.1f} "
        f"total_cost_including_idle={cost_overall['total_cost']:.9f}"
    )
    print(
        "successful: "
        f"count={successful['count']} "
        f"e2e_mean_ms={successful['e2e_mean_ms']:.1f} "
        f"e2e_min_ms={successful['e2e_min_ms']:.1f} "
        f"e2e_max_ms={successful['e2e_max_ms']:.1f}"
    )
    print(f"errors: count={failed['count']}")
    print(
        "idle: "
        f"containers={cost_overall['container_count']} "
        f"between_idle_ms={cost_overall['reported_between_idle_ms_total']:.1f} "
        f"tail_idle_ms={cost_overall['assumed_tail_idle_ms_total']:.1f}"
    )


if __name__ == "__main__":
    main()
