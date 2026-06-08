#!/usr/bin/env python3
"""Standalone oracle entry prewarm validation for path 3."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.openwhisk_client import OpenWhiskClient
from runner.stage4_risk.scaling import memory_to_cpu_cores
from runner.stage5_control.jit_scheduler import JitScheduler, WarmupTask
from runner.workflow import NodeSpec, load_workflow


DEFAULT_SCHEDULE = (
    ROOT
    / "data"
    / "Azure schedule trace"
    / "schedule_cand2_60real_min_civic_alert_flow_target20s_2x.csv"
)
DEFAULT_WORKFLOW = ROOT / "configs" / "civic_alert_flow.yaml"
DEFAULT_ACTION_FILE = ROOT / "actions" / "workflow_action.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    parser.add_argument("--schedule", default=str(DEFAULT_SCHEDULE))
    parser.add_argument("--action-file", default=str(DEFAULT_ACTION_FILE))
    parser.add_argument("--entry-action", default="wf_civic_detect_object_1280")
    parser.add_argument("--entry-stage", default="detect_object")
    parser.add_argument("--entry-memory-mb", type=int, default=1280)
    parser.add_argument("--apihost", default="https://192.168.123.17:31001")
    parser.add_argument(
        "--auth",
        default="",
        help="OpenWhisk AUTH. If omitted, read owdev-whisk.auth guest auth via kubectl.",
    )
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--prewarm-lead-sec", type=float, default=2.0)
    parser.add_argument("--max-arrivals", type=int, default=200)
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="divide schedule offsets by this value; 1.0 preserves real timing",
    )
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--warmup-workers", type=int, default=32)
    parser.add_argument("--kind", default="python:3")
    parser.add_argument("--redeploy-before", action="store_true")
    parser.add_argument("--restore-after", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--no-prewarm", action="store_true")
    mode.add_argument("--oracle-prewarm", action="store_true")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_guest_auth() -> str:
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
    )
    return subprocess.check_output(["base64", "-d"], input=encoded, text=True).strip()


def activation_annotations(activation: dict) -> dict:
    return {
        item.get("key"): item.get("value")
        for item in activation.get("annotations", [])
        if isinstance(item, dict) and item.get("key")
    }


def safe_float(value: object) -> float | str:
    if value in ("", None):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def load_arrivals(path: Path, max_arrivals: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "target_offset_ms" not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing target_offset_ms")
        for idx, row in enumerate(reader):
            if max_arrivals > 0 and idx >= max_arrivals:
                break
            rows.append(
                {
                    "arrival_index": idx,
                    "target_offset_ms": float(row["target_offset_ms"]),
                    "workflow_name": row.get("workflow_name", ""),
                }
            )
    if not rows:
        raise ValueError(f"No arrivals loaded from {path}")
    base = rows[0]["target_offset_ms"]
    for row in rows:
        row["relative_offset_ms"] = row["target_offset_ms"] - base
    return rows


def build_window_rows(
    arrivals: list[dict[str, Any]],
    *,
    window_sec: float,
    time_scale: float,
) -> list[dict[str, Any]]:
    windows: dict[int, dict[str, Any]] = {}
    for arrival in arrivals:
        scaled_offset_sec = arrival["relative_offset_ms"] / 1000.0 / time_scale
        window_index = int(math.floor(scaled_offset_sec / window_sec))
        window = windows.setdefault(
            window_index,
            {
                "window_index": window_index,
                "window_start_sec": window_index * window_sec,
                "window_end_sec": (window_index + 1) * window_sec,
                "predicted_arrivals": 0,
                "warmups_fired": 0,
                "arrivals": 0,
                "cold_count": 0,
            },
        )
        window["predicted_arrivals"] += 1
        window["arrivals"] += 1
    return [windows[key] for key in sorted(windows)]


def variant_timeout_ms(memory_mb: int) -> int:
    return 120000 if memory_mb == 1280 else 240000


def redeploy_entry_action(args: argparse.Namespace, auth: str) -> None:
    subprocess.run(
        [
            "wsk",
            "-i",
            "--apihost",
            args.apihost,
            "--auth",
            auth,
            "action",
            "update",
            args.entry_action,
            args.action_file,
            "--kind",
            args.kind,
            "--memory",
            str(args.entry_memory_mb),
            "--timeout",
            str(variant_timeout_ms(args.entry_memory_mb)),
        ],
        stdout=subprocess.DEVNULL,
        check=True,
    )


def real_params(
    *,
    workflow_name: str,
    node: NodeSpec,
    request_id: str,
    memory_mb: int,
) -> dict:
    return {
        "workflow_name": workflow_name,
        "request_id": request_id,
        "entry_ts_ms": int(time.time() * 1000),
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
        "allocated_memory_mb": memory_mb,
        "allocated_cpu_cores": memory_to_cpu_cores(memory_mb),
        "payload": {"parents": {}},
    }


class WarmupRecorder:
    def __init__(
        self,
        *,
        client: OpenWhiskClient,
        action_name: str,
        workflow_name: str,
        memory_mb: int,
        max_workers: int,
    ):
        self.client = client
        self.action_name = action_name
        self.workflow_name = workflow_name
        self.memory_mb = memory_mb
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        self.records: list[dict[str, Any]] = []
        self.futures = []

    def callback(self, task: WarmupTask) -> None:
        future = self.executor.submit(self._invoke_warmup, task, time.monotonic())
        with self.lock:
            self.futures.append(future)

    def _invoke_warmup(self, task: WarmupTask, callback_start: float) -> dict[str, Any]:
        params = {
            "__warmup": True,
            "workflow_name": self.workflow_name,
            "request_id": task.metadata.get("request_id", ""),
            "stage_name": task.metadata.get("stage_name", ""),
            "allocated_memory_mb": self.memory_mb,
            "allocated_cpu_cores": memory_to_cpu_cores(self.memory_mb),
        }
        sent = time.monotonic()
        activation = {}
        result = {}
        annotations = {}
        error = ""
        try:
            activation = self.client.invoke_activation(self.action_name, params)
            response = activation.get("response", {})
            result = response.get("result", {}) if isinstance(response, dict) else {}
            if not isinstance(result, dict):
                result = {"error": result}
            annotations = activation_annotations(activation)
        except Exception as exc:
            error = str(exc)
        completed = time.monotonic()
        row = {
            "task_key": task.task_key,
            "window_index": task.metadata.get("window_index", ""),
            "warmup_index": task.metadata.get("warmup_index", ""),
            "scheduled_fire_sec": task.metadata.get("scheduled_fire_sec", ""),
            "callback_start_monotonic": callback_start,
            "sent_monotonic": sent,
            "completed_monotonic": completed,
            "activation_id": activation.get("activationId", ""),
            "container_id": result.get("container_id", ""),
            "cold_like": result.get("cold_like", ""),
            "ow_wait_ms": annotations.get("waitTime", ""),
            "ow_init_ms": annotations.get("initTime", ""),
            "ow_duration_ms": activation.get("duration", ""),
            "error": error,
        }
        with self.lock:
            self.records.append(row)
        return row

    def wait(self) -> None:
        with self.lock:
            futures = list(self.futures)
        for future in as_completed(futures):
            future.result()
        self.executor.shutdown(wait=True)

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.records)


def invoke_entry(
    *,
    client: OpenWhiskClient,
    action_name: str,
    workflow_name: str,
    node: NodeSpec,
    memory_mb: int,
    arrival: dict[str, Any],
    scaled_offset_sec: float,
    run_start: float,
) -> dict[str, Any]:
    request_id = str(uuid.uuid4())
    params = real_params(
        workflow_name=workflow_name,
        node=node,
        request_id=request_id,
        memory_mb=memory_mb,
    )
    invoke_sent = time.monotonic()
    activation = {}
    result = {}
    annotations = {}
    error = ""
    try:
        activation = client.invoke_activation(action_name, params)
        response = activation.get("response", {})
        result = response.get("result", {}) if isinstance(response, dict) else {}
        if not isinstance(result, dict):
            result = {"error": result}
        annotations = activation_annotations(activation)
    except Exception as exc:
        error = str(exc)
    invoke_completed = time.monotonic()
    cold_like = str(result.get("cold_like", "")).lower() == "true"
    return {
        "arrival_index": arrival["arrival_index"],
        "request_id": request_id,
        "target_offset_ms": arrival["target_offset_ms"],
        "relative_offset_ms": arrival["relative_offset_ms"],
        "scaled_arrival_sec": scaled_offset_sec,
        "scheduled_arrival_monotonic": run_start + scaled_offset_sec,
        "invoke_sent_monotonic": invoke_sent,
        "invoke_completed_monotonic": invoke_completed,
        "arrival_lag_ms": (invoke_sent - (run_start + scaled_offset_sec)) * 1000.0,
        "activation_id": activation.get("activationId", ""),
        "container_id": result.get("container_id", ""),
        "cold_like": cold_like,
        "hit_warm": not cold_like,
        "ow_wait_ms": annotations.get("waitTime", ""),
        "ow_init_ms": annotations.get("initTime", ""),
        "ow_duration_ms": activation.get("duration", ""),
        "action_duration_ms": result.get("action_duration_ms", ""),
        "container_invocation_index": result.get("container_invocation_index", ""),
        "container_uptime_ms": result.get("container_uptime_ms", ""),
        "error": error,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_arrivals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    waits = [
        float(row["ow_wait_ms"])
        for row in rows
        if row.get("ow_wait_ms") not in ("", None)
    ]
    cold_count = sum(bool(row.get("cold_like")) for row in rows)
    return {
        "arrivals": len(rows),
        "cold_count": cold_count,
        "cold_rate": cold_count / len(rows) if rows else 0.0,
        "warm_count": len(rows) - cold_count,
        "mean_ow_wait_ms": sum(waits) / len(waits) if waits else 0.0,
        "max_ow_wait_ms": max(waits) if waits else 0.0,
    }


def write_report(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    arrival_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
    warmup_rows: list[dict[str, Any]],
    total_duration_sec: float,
) -> None:
    summary = summarize_arrivals(arrival_rows)
    mode = "oracle-prewarm" if args.oracle_prewarm else "no-prewarm"
    warmup_errors = [row for row in warmup_rows if row.get("error")]
    lines = [
        f"# Entry Prewarm Run: {mode}",
        "",
        "## Configuration",
        "",
        f"- schedule: `{args.schedule}`",
        f"- entry action: `{args.entry_action}`",
        f"- arrivals tested: {len(arrival_rows)}",
        f"- window_sec: {args.window_sec}",
        f"- prewarm_lead_sec: {args.prewarm_lead_sec}",
        f"- time_scale: {args.time_scale}",
        f"- replay duration: {total_duration_sec:.3f}s",
        f"- redeploy_before: {args.redeploy_before}",
        "",
        "TODO: this simple oracle fires one warmup per predicted arrival. It does",
        "not yet optimize for the ~10s Ready/keepalive window, which can reduce",
        "warmup cost by reusing already-ready entry containers across windows.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| arrivals | {summary['arrivals']} |",
        f"| cold_count | {summary['cold_count']} |",
        f"| cold_rate | {summary['cold_rate']:.2%} |",
        f"| mean_ow_wait_ms | {summary['mean_ow_wait_ms']:.1f} |",
        f"| max_ow_wait_ms | {summary['max_ow_wait_ms']:.1f} |",
        f"| warmups_fired | {len(warmup_rows)} |",
        f"| warmup_errors | {len(warmup_errors)} |",
        "",
        "## Per-Window",
        "",
        "| window | predicted_N | warmups_fired | arrivals | cold_count |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in window_rows:
        lines.append(
            f"| {row['window_index']} | {row['predicted_arrivals']} | "
            f"{row['warmups_fired']} | {row['arrivals']} | {row['cold_count']} |"
        )
    lines.append("")
    (out_dir / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.time_scale <= 0:
        raise ValueError("--time-scale must be positive")
    if args.window_sec <= 0 or args.prewarm_lead_sec < 0:
        raise ValueError("--window-sec must be positive and --prewarm-lead-sec non-negative")

    auth = args.auth or read_guest_auth()
    workflow = load_workflow(args.workflow)
    if args.entry_stage not in workflow.nodes:
        raise ValueError(f"entry stage {args.entry_stage!r} is not in workflow")
    node = workflow.nodes[args.entry_stage]
    arrivals = load_arrivals(Path(args.schedule), args.max_arrivals)
    window_rows = build_window_rows(
        arrivals,
        window_sec=args.window_sec,
        time_scale=args.time_scale,
    )
    window_by_index = {int(row["window_index"]): row for row in window_rows}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.redeploy_before:
        redeploy_entry_action(args, auth)

    client = OpenWhiskClient(
        apihost=args.apihost,
        auth=auth,
        namespace=workflow.namespace,
        timeout_sec=args.timeout_sec,
    )
    warmup_client = OpenWhiskClient(
        apihost=args.apihost,
        auth=auth,
        namespace=workflow.namespace,
        timeout_sec=args.timeout_sec,
    )

    workflow_name = workflow.workflow_name
    # In oracle mode, delay the replay start enough that window 0 can receive
    # its full prewarm lead. Baseline mode starts after a short fixed grace.
    run_start = time.monotonic() + 1.0 + (args.prewarm_lead_sec if args.oracle_prewarm else 0.0)
    scheduler = None
    recorder = None
    if args.oracle_prewarm:
        recorder = WarmupRecorder(
            client=warmup_client,
            action_name=args.entry_action,
            workflow_name=workflow_name,
            memory_mb=args.entry_memory_mb,
            max_workers=args.warmup_workers,
        )
        scheduler = JitScheduler(recorder.callback)
        scheduler.start()
        # Simple version: fire N_t warmups per window. Future optimization can
        # use the ~10s Ready/keepalive window to reduce redundant warmups.
        for window in window_rows:
            fire_sec = float(window["window_start_sec"]) - args.prewarm_lead_sec
            fire_time = max(time.monotonic(), run_start + fire_sec)
            for idx in range(int(window["predicted_arrivals"])):
                task = WarmupTask(
                    task_key=f"entry:{window['window_index']}:{idx}",
                    fire_time=fire_time,
                    action_name=args.entry_action,
                    metadata={
                        "request_id": f"entry-prewarm-window-{window['window_index']}-{idx}",
                        "stage_name": args.entry_stage,
                        "window_index": window["window_index"],
                        "warmup_index": idx,
                        "scheduled_fire_sec": fire_sec,
                    },
                )
                scheduler.schedule(task)
                window["warmups_fired"] += 1

    arrival_rows: list[dict[str, Any]] = []
    futures = []
    replay_started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for arrival in arrivals:
                scaled_offset_sec = arrival["relative_offset_ms"] / 1000.0 / args.time_scale
                target = run_start + scaled_offset_sec
                sleep_for = target - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                future = executor.submit(
                    invoke_entry,
                    client=client,
                    action_name=args.entry_action,
                    workflow_name=workflow_name,
                    node=node,
                    memory_mb=args.entry_memory_mb,
                    arrival=arrival,
                    scaled_offset_sec=scaled_offset_sec,
                    run_start=run_start,
                )
                futures.append(future)

            for future in as_completed(futures):
                row = future.result()
                arrival_rows.append(row)
                window_index = int(math.floor(row["scaled_arrival_sec"] / args.window_sec))
                if row["cold_like"]:
                    window_by_index[window_index]["cold_count"] += 1
    finally:
        if scheduler is not None:
            while scheduler.pending_count() > 0:
                time.sleep(0.05)
            scheduler.stop()
        if recorder is not None:
            recorder.wait()

    total_duration_sec = time.monotonic() - replay_started
    arrival_rows.sort(key=lambda row: int(row["arrival_index"]))
    warmup_rows = recorder.snapshot() if recorder is not None else []
    warmup_rows.sort(
        key=lambda row: (
            int(row["window_index"]) if row.get("window_index") not in ("", None) else -1,
            int(row["warmup_index"]) if row.get("warmup_index") not in ("", None) else -1,
        )
    )

    arrival_fields = [
        "arrival_index",
        "request_id",
        "target_offset_ms",
        "relative_offset_ms",
        "scaled_arrival_sec",
        "scheduled_arrival_monotonic",
        "invoke_sent_monotonic",
        "invoke_completed_monotonic",
        "arrival_lag_ms",
        "activation_id",
        "container_id",
        "cold_like",
        "hit_warm",
        "ow_wait_ms",
        "ow_init_ms",
        "ow_duration_ms",
        "action_duration_ms",
        "container_invocation_index",
        "container_uptime_ms",
        "error",
    ]
    window_fields = [
        "window_index",
        "window_start_sec",
        "window_end_sec",
        "predicted_arrivals",
        "warmups_fired",
        "arrivals",
        "cold_count",
    ]
    write_csv(out_dir / "entry_arrivals.csv", arrival_rows, arrival_fields)
    write_csv(out_dir / "entry_windows.csv", window_rows, window_fields)
    if warmup_rows:
        write_csv(out_dir / "entry_warmups.csv", warmup_rows)
    else:
        write_csv(
            out_dir / "entry_warmups.csv",
            [],
            [
                "task_key",
                "window_index",
                "warmup_index",
                "scheduled_fire_sec",
                "callback_start_monotonic",
                "sent_monotonic",
                "completed_monotonic",
                "activation_id",
                "container_id",
                "cold_like",
                "ow_wait_ms",
                "ow_init_ms",
                "ow_duration_ms",
                "error",
            ],
        )
    write_report(
        args=args,
        out_dir=out_dir,
        arrival_rows=arrival_rows,
        window_rows=window_rows,
        warmup_rows=warmup_rows,
        total_duration_sec=total_duration_sec,
    )

    summary = summarize_arrivals(arrival_rows)
    print(
        f"mode={'oracle' if args.oracle_prewarm else 'baseline'} "
        f"arrivals={summary['arrivals']} cold_rate={summary['cold_rate']:.2%} "
        f"mean_ow_wait_ms={summary['mean_ow_wait_ms']:.1f} "
        f"warmups={len(warmup_rows)} duration_s={total_duration_sec:.1f}"
    )

    if args.restore_after:
        redeploy_entry_action(args, auth)


if __name__ == "__main__":
    main()
