#!/usr/bin/env python3
"""Exercise P3.C JIT warmup scheduling against real OpenWhisk actions."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.openwhisk_client import OpenWhiskClient
from runner.run_workflow import run_one_workflow
from runner.stage5_control.jit_scheduler import JitScheduler
from runner.workflow import load_workflow, suffix_action_name


STAGE_ORDER = [
    "detect_object",
    "estimate_pose",
    "match_face",
    "classify_scene",
    "translate_alert",
]

PREMIUM_PLAN = {
    "detect_object": 1536,
    "estimate_pose": 1280,
    "match_face": 2048,
    "classify_scene": 3072,
    "translate_alert": 1024,
}

P3C_COLD_RATES = {
    "detect_object": 1.00,
    "estimate_pose": 0.30,
    "match_face": 0.70,
    "classify_scene": 0.60,
    "translate_alert": 0.50,
}

P3C_LATE_JIT_COUNT = 30
P3C_WARMUP_COUNT = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", default=str(ROOT / "configs" / "civic_alert_flow.yaml"))
    parser.add_argument("--action-file", default=str(ROOT / "actions" / "workflow_action.py"))
    parser.add_argument("--apihost", default="https://192.168.123.17:31001")
    parser.add_argument(
        "--auth",
        default="",
        help="OpenWhisk AUTH. If omitted, read owdev-whisk.auth guest auth via kubectl.",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--jit-margin-ms", type=float, default=600.0)
    parser.add_argument("--jit-sync-pause-grace-ms", type=float, default=3000.0)
    parser.add_argument("--skip-reset", action="store_true")
    parser.add_argument("--jit-only", action="store_true")
    parser.add_argument("--enable-jit-sync", action="store_true")
    parser.add_argument("--reset-once-before", action="store_true")
    parser.add_argument("--experiment-label", default="jit_diag")
    parser.add_argument(
        "--diag-out-dir",
        default=str(ROOT / "reports" / "path3_jit_sync"),
    )
    parser.add_argument("--diag-csv", default="")
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
        if isinstance(item, dict)
    }


def variant_timeout_ms(tier: int) -> int:
    return 120000 if tier == 1280 else 240000


def reset_plan_variants(args: argparse.Namespace, auth: str, workflow) -> None:
    for stage_name in STAGE_ORDER:
        node = workflow.nodes[stage_name]
        tier = PREMIUM_PLAN[stage_name]
        action = suffix_action_name(node.action, f"_{tier}")
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
                action,
                args.action_file,
                "--kind",
                "python:3",
                "--memory",
                str(tier),
                "--timeout",
                str(variant_timeout_ms(tier)),
            ],
            stdout=subprocess.DEVNULL,
            check=True,
        )


class WarmupStatusTracker:
    def __init__(self):
        self.condition = threading.Condition()
        self.statuses: dict[tuple[str, str], dict] = {}

    def _status_locked(self, request_id: str, stage_name: str) -> dict:
        key = (request_id, stage_name)
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

    def mark_issued(self, request_id: str, stage_name: str, issued_monotonic: float) -> None:
        with self.condition:
            status = self._status_locked(request_id, stage_name)
            status["issued_monotonic"] = issued_monotonic
            self.condition.notify_all()

    def mark_completed(
        self,
        request_id: str,
        stage_name: str,
        completed_monotonic: float,
        activation_id: str = "",
        container_id: str = "",
        error: str = "",
    ) -> None:
        with self.condition:
            status = self._status_locked(request_id, stage_name)
            status["completed_monotonic"] = completed_monotonic
            status["activation_id"] = activation_id
            status["container_id"] = container_id
            status["error"] = error
            status["event"].set()
            self.condition.notify_all()

    def get_status(self, request_id: str, stage_name: str) -> dict:
        with self.condition:
            status = self._status_locked(request_id, stage_name)
            return {key: value for key, value in status.items() if key != "event"}

    def wait_until_completed(
        self,
        request_id: str,
        stage_name: str,
        timeout: float,
    ) -> dict:
        with self.condition:
            status = self._status_locked(request_id, stage_name)
            event = status["event"]
        event.wait(timeout=max(0.0, timeout))
        return self.get_status(request_id, stage_name)


class WarmupRecorder:
    def __init__(self, client: OpenWhiskClient, tracker: WarmupStatusTracker | None = None):
        self.client = client
        self.tracker = tracker
        self.condition = threading.Condition()
        self.records: list[dict] = []

    def callback(self, task) -> None:
        callback_start = time.monotonic()
        worker = threading.Thread(
            target=self._invoke_warmup,
            args=(task, callback_start),
            daemon=True,
        )
        worker.start()

    def _invoke_warmup(self, task, callback_start: float) -> None:
        params = {
            "__warmup": True,
            "request_id": task.metadata.get("request_id", ""),
            "workflow_name": task.metadata.get("workflow_name", ""),
            "stage_name": task.metadata.get("stage_name", ""),
            "allocated_memory_mb": task.metadata.get("tier_mb", ""),
        }
        warmup_sent_monotonic = time.monotonic()
        request_id = task.metadata.get("request_id", "")
        stage_name = task.metadata.get("stage_name", "")
        if self.tracker is not None:
            self.tracker.mark_issued(request_id, stage_name, warmup_sent_monotonic)
        activation = {}
        result = {}
        annotations = {}
        error = ""
        try:
            activation = self.client.invoke_activation(task.action_name, params)
            result = activation.get("response", {}).get("result", {})
            annotations = activation_annotations(activation)
        except Exception as exc:
            error = str(exc)
        callback_end = time.monotonic()
        if self.tracker is not None:
            self.tracker.mark_completed(
                request_id,
                stage_name,
                callback_end,
                activation_id=activation.get("activationId", ""),
                container_id=result.get("container_id", ""),
                error=error,
            )
        with self.condition:
            self.records.append(
                {
                    "task_key": task.task_key,
                    "request_id": task.metadata.get("request_id", ""),
                    "stage_name": task.metadata.get("stage_name", ""),
                    "action_name": task.action_name,
                    "tier_mb": task.metadata.get("tier_mb", ""),
                    "scheduled_fire_time": task.fire_time,
                    "raw_fire_time": task.metadata.get("raw_fire_time", ""),
                    "needed_at": task.metadata.get("needed_at", ""),
                    "late_jit": task.metadata.get("late_jit", ""),
                    "schedule_phase": task.metadata.get("schedule_phase", ""),
                    "callback_start": callback_start,
                    "warmup_sent_monotonic": warmup_sent_monotonic,
                    "callback_end": callback_end,
                    "activation_id": activation.get("activationId", ""),
                    "activation_duration_ms": activation.get("duration", ""),
                    "action_duration_ms": result.get("action_duration_ms", ""),
                    "cold_like": result.get("cold_like", ""),
                    "container_id": result.get("container_id", ""),
                    "initTime": annotations.get("initTime", ""),
                    "waitTime": annotations.get("waitTime", ""),
                    "error": error,
                }
            )
            self.condition.notify_all()

    def snapshot(self) -> list[dict]:
        with self.condition:
            return list(self.records)

    def wait_for_request_count(
        self,
        request_id: str,
        expected_count: int,
        timeout: float = 60.0,
    ) -> None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                count = sum(1 for record in self.records if record["request_id"] == request_id)
                if count >= expected_count:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"only saw {count}/{expected_count} warmups for request_id={request_id}"
                    )
                self.condition.wait(timeout=remaining)


def workflow_e2e(rows: list[dict]) -> float:
    return float(rows[0].get("workflow_e2e_ms", 0.0))


def wait_for_scheduler_empty(scheduler: JitScheduler, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while scheduler.pending_count() > 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    if scheduler.pending_count() > 0:
        raise RuntimeError(f"scheduler still has {scheduler.pending_count()} pending tasks")


def run_batch(
    *,
    label: str,
    enable_jit: bool,
    args: argparse.Namespace,
    auth: str,
    workflow,
    client: OpenWhiskClient,
    scheduler: JitScheduler | None,
    recorder: WarmupRecorder | None = None,
    tracker: WarmupStatusTracker | None = None,
    enable_jit_sync: bool = False,
) -> list[list[dict]]:
    results = []
    print(f"\n== {label} ==", flush=True)
    for idx in range(1, args.runs + 1):
        if not args.skip_reset:
            reset_plan_variants(args, auth, workflow)
        rows = run_one_workflow(
            workflow,
            client,
            max_workers=args.max_workers,
            plan=PREMIUM_PLAN,
            slo_class="premium",
            enable_jit=enable_jit,
            jit_scheduler=scheduler,
            jit_margin_ms=args.jit_margin_ms,
            jit_warmup_tracker=tracker,
            enable_jit_sync=enable_jit_sync,
            jit_sync_pause_grace_ms=args.jit_sync_pause_grace_ms,
        )
        if scheduler is not None:
            wait_for_scheduler_empty(scheduler)
            if recorder is not None:
                recorder.wait_for_request_count(
                    rows[0]["request_id"],
                    int(rows[0].get("jit_scheduled_count", 0) or 0),
                )
        results.append(rows)
        print(
            f"[{idx}/{args.runs}] request_id={rows[0]['request_id']} "
            f"e2e_ms={workflow_e2e(rows):.1f} "
            f"jit_scheduled={rows[0].get('jit_scheduled_count', 0)} "
            f"jit_upserts={rows[0].get('jit_upsert_count', 0)} "
            f"jit_late={rows[0].get('jit_late_count', 0)}",
            flush=True,
        )
    return results


def build_timing_rows(
    *,
    experiment_label: str,
    jit_rows: list[list[dict]],
    warmups: list[dict],
) -> list[dict]:
    warmup_by_stage = {
        (record.get("request_id"), record.get("stage_name")): record
        for record in warmups
    }
    timing_rows: list[dict] = []
    for run_index, workflow_rows in enumerate(jit_rows, start=1):
        request_id = workflow_rows[0]["request_id"]
        stage_rows = [
            row
            for row in workflow_rows
            if row.get("stage_name") in STAGE_ORDER
        ]
        for row in stage_rows:
            stage_name = row["stage_name"]
            warmup = warmup_by_stage.get((request_id, stage_name), {})
            warmup_ready = warmup.get("callback_end", "")
            real_invoke = row.get("real_invoke_monotonic", "")
            gap_ms = ""
            if warmup_ready not in ("", None) and real_invoke not in ("", None):
                gap_ms = (float(real_invoke) - float(warmup_ready)) * 1000.0
            warmup_container = warmup.get("container_id", "")
            real_container = row.get("container_id", "")
            same_container = (
                bool(warmup_container)
                and bool(real_container)
                and warmup_container == real_container
            )
            timing_rows.append(
                {
                    "experiment": experiment_label,
                    "run_index": run_index,
                    "request_id": request_id,
                    "stage_name": stage_name,
                    "tier": PREMIUM_PLAN[stage_name],
                    "warmup_scheduled_fire_monotonic": warmup.get("scheduled_fire_time", ""),
                    "warmup_fire_monotonic": warmup.get("callback_start", ""),
                    "warmup_sent_monotonic": warmup.get("warmup_sent_monotonic", ""),
                    "warmup_ready_monotonic": warmup_ready,
                    "warmup_activation_id": warmup.get("activation_id", ""),
                    "warmup_container_id": warmup_container,
                    "warmup_cold_like": warmup.get("cold_like", ""),
                    "warmup_wait_ms": warmup.get("waitTime", ""),
                    "warmup_init_ms": warmup.get("initTime", ""),
                    "warmup_duration_ms": warmup.get("activation_duration_ms", ""),
                    "warmup_error": warmup.get("error", ""),
                    "real_invoke_monotonic": real_invoke,
                    "real_activation_id": row.get("activation_id", ""),
                    "real_container_id": real_container,
                    "real_ow_wait_ms": row.get("ow_wait_ms", ""),
                    "real_ow_init_ms": row.get("ow_init_ms", ""),
                    "real_ow_duration_ms": row.get("ow_duration_ms", ""),
                    "same_container": same_container,
                    "cold_like": row.get("cold_like", ""),
                    "gap_warmup_ready_to_real_ms": gap_ms,
                    "stage_start_monotonic": row.get("stage_start_monotonic", ""),
                    "resolved_action_name": row.get("resolved_action_name", ""),
                    "jit_sync_enabled": row.get("jit_sync_enabled", ""),
                    "jit_sync_waited_ms": row.get("jit_sync_waited_ms", ""),
                    "jit_sync_dispatch_after_warmup": row.get("jit_sync_dispatch_after_warmup", ""),
                    "jit_sync_status": row.get("jit_sync_status", ""),
                    "jit_sync_warmup_issued_monotonic": row.get("jit_sync_warmup_issued_monotonic", ""),
                    "jit_sync_warmup_completed_monotonic": row.get("jit_sync_warmup_completed_monotonic", ""),
                }
            )
    return timing_rows


def write_timing_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "run_index",
        "request_id",
        "stage_name",
        "tier",
        "warmup_scheduled_fire_monotonic",
        "warmup_fire_monotonic",
        "warmup_sent_monotonic",
        "warmup_ready_monotonic",
        "warmup_activation_id",
        "warmup_container_id",
        "warmup_cold_like",
        "warmup_wait_ms",
        "warmup_init_ms",
        "warmup_duration_ms",
        "warmup_error",
        "real_invoke_monotonic",
        "real_activation_id",
        "real_container_id",
        "real_ow_wait_ms",
        "real_ow_init_ms",
        "real_ow_duration_ms",
        "same_container",
        "cold_like",
        "gap_warmup_ready_to_real_ms",
        "stage_start_monotonic",
        "resolved_action_name",
        "jit_sync_enabled",
        "jit_sync_waited_ms",
        "jit_sync_dispatch_after_warmup",
        "jit_sync_status",
        "jit_sync_warmup_issued_monotonic",
        "jit_sync_warmup_completed_monotonic",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_estimate_progression(timing_rows: list[dict]) -> None:
    print("\nestimate_pose per-run timing")
    print(
        "run | cold_like | same_container | gap_ready_to_real_ms | "
        "waited_ms | sync_status | warm_wait | warm_init | real_wait | real_init"
    )
    print("-" * 125)
    for row in timing_rows:
        if row["stage_name"] != "estimate_pose":
            continue
        gap = row["gap_warmup_ready_to_real_ms"]
        gap_text = f"{float(gap):.1f}" if gap not in ("", None) else "n/a"
        waited = row.get("jit_sync_waited_ms", "")
        waited_text = f"{float(waited):.1f}" if waited not in ("", None) else "n/a"
        print(
            f"{row['run_index']:<3} | {row['cold_like']!s:<9} | "
            f"{row['same_container']!s:<14} | {gap_text:<20} | "
            f"{waited_text:<9} | {row.get('jit_sync_status', '')!s:<11} | "
            f"{row['warmup_wait_ms']!s:<9} | {row['warmup_init_ms']!s:<9} | "
            f"{row['real_ow_wait_ms']!s:<9} | {row['real_ow_init_ms']!s:<9}"
        )


def print_jit_only_summary(on_rows: list[list[dict]], warmups: list[dict]) -> None:
    on_rates = summarize_stage_cold_rates(on_rows)
    on_e2e = [workflow_e2e(rows) for rows in on_rows]
    deltas = summarize_warmup_timing(on_rows, warmups)
    print("\nJIT-only cold rates")
    print("stage           | cold_rate | warmup_count | invoke_after_warmup_ms")
    print("-" * 78)
    for stage in STAGE_ORDER:
        stage_warmups = [record for record in warmups if record["stage_name"] == stage]
        stage_deltas = deltas.get(stage, [])
        delta_text = (
            f"mean={sum(stage_deltas) / len(stage_deltas):.1f}, min={min(stage_deltas):.1f}"
            if stage_deltas
            else "n/a"
        )
        print(
            f"{stage:<15} | {on_rates[stage]:<9.2%} | "
            f"{len(stage_warmups):<12} | {delta_text}"
        )
    print(
        f"\nJIT on: mean={sum(on_e2e) / len(on_e2e):.1f} ms, "
        f"min={min(on_e2e):.1f}, max={max(on_e2e):.1f}, samples={on_e2e}"
    )
    late_count = sum(1 for record in warmups if record.get("late_jit"))
    print(f"Warmups fired: {len(warmups)}; late_jit={late_count}/{len(warmups)}")


def summarize_stage_cold_rates(rows_by_workflow: list[list[dict]]) -> dict[str, float]:
    rates = {}
    for stage in STAGE_ORDER:
        rows = [
            row
            for workflow_rows in rows_by_workflow
            for row in workflow_rows
            if row.get("stage_name") == stage
        ]
        if not rows:
            rates[stage] = 0.0
            continue
        cold_count = sum(1 for row in rows if str(row.get("cold_like")).lower() == "true")
        rates[stage] = cold_count / len(rows)
    return rates


def mean_stage_field(rows_by_workflow: list[list[dict]], field: str) -> dict[str, float]:
    values_by_stage: dict[str, list[float]] = {stage: [] for stage in STAGE_ORDER}
    for workflow_rows in rows_by_workflow:
        for row in workflow_rows:
            stage = row.get("stage_name")
            if stage not in values_by_stage:
                continue
            value = row.get(field, "")
            if value in ("", None):
                continue
            try:
                values_by_stage[stage].append(float(value))
            except (TypeError, ValueError):
                continue
    return {
        stage: (sum(values) / len(values) if values else 0.0)
        for stage, values in values_by_stage.items()
    }


def same_container_rates(timing_rows: list[dict]) -> dict[str, float]:
    rates = {}
    for stage in STAGE_ORDER:
        rows = [row for row in timing_rows if row.get("stage_name") == stage]
        if not rows:
            rates[stage] = 0.0
            continue
        rates[stage] = sum(str(row.get("same_container")).lower() == "true" for row in rows) / len(rows)
    return rates


def print_three_variant_summary(
    *,
    off_rows: list[list[dict]],
    no_sync_rows: list[list[dict]],
    sync_rows: list[list[dict]],
    no_sync_timing: list[dict],
    sync_timing: list[dict],
) -> None:
    off_rates = summarize_stage_cold_rates(off_rows)
    no_sync_rates = summarize_stage_cold_rates(no_sync_rows)
    sync_rates = summarize_stage_cold_rates(sync_rows)
    no_sync_same = same_container_rates(no_sync_timing)
    sync_same = same_container_rates(sync_timing)
    waited_ms = mean_stage_field(sync_rows, "jit_sync_waited_ms")

    print("\nThree-variant cold-rate comparison")
    print(
        "stage           | JIT off | JIT no-sync | JIT sync | "
        "no-sync same | sync same | sync waited_ms"
    )
    print("-" * 108)
    for stage in STAGE_ORDER:
        print(
            f"{stage:<15} | {off_rates[stage]:<7.2%} | "
            f"{no_sync_rates[stage]:<11.2%} | {sync_rates[stage]:<8.2%} | "
            f"{no_sync_same[stage]:<12.2%} | {sync_same[stage]:<9.2%} | "
            f"{waited_ms[stage]:.1f}"
        )

    off_e2e = [workflow_e2e(rows) for rows in off_rows]
    no_sync_e2e = [workflow_e2e(rows) for rows in no_sync_rows]
    sync_e2e = [workflow_e2e(rows) for rows in sync_rows]
    print("\nWorkflow E2E")
    print(
        f"JIT off    : mean={sum(off_e2e) / len(off_e2e):.1f} ms, "
        f"min={min(off_e2e):.1f}, max={max(off_e2e):.1f}, samples={off_e2e}"
    )
    print(
        f"JIT no-sync: mean={sum(no_sync_e2e) / len(no_sync_e2e):.1f} ms, "
        f"min={min(no_sync_e2e):.1f}, max={max(no_sync_e2e):.1f}, samples={no_sync_e2e}"
    )
    print(
        f"JIT sync   : mean={sum(sync_e2e) / len(sync_e2e):.1f} ms, "
        f"min={min(sync_e2e):.1f}, max={max(sync_e2e):.1f}, samples={sync_e2e}"
    )


def summarize_warmup_timing(jit_rows: list[list[dict]], warmups: list[dict]) -> dict[str, list[float]]:
    stage_starts = {
        (workflow_rows[0]["request_id"], row["stage_name"]): float(row["stage_start_monotonic"])
        for workflow_rows in jit_rows
        for row in workflow_rows
        if row.get("stage_name") in STAGE_ORDER and row.get("stage_start_monotonic") not in ("", None)
    }
    deltas: dict[str, list[float]] = {stage: [] for stage in STAGE_ORDER}
    for record in warmups:
        key = (record["request_id"], record["stage_name"])
        if key in stage_starts:
            deltas[record["stage_name"]].append(
                (stage_starts[key] - float(record["callback_end"])) * 1000.0
            )
    return deltas


def print_summary(off_rows: list[list[dict]], on_rows: list[list[dict]], warmups: list[dict]) -> None:
    off_rates = summarize_stage_cold_rates(off_rows)
    on_rates = summarize_stage_cold_rates(on_rows)
    off_e2e = [workflow_e2e(rows) for rows in off_rows]
    on_e2e = [workflow_e2e(rows) for rows in on_rows]
    deltas = summarize_warmup_timing(on_rows, warmups)

    print("\nPer-stage cold rate comparison")
    print(
        "stage           | JIT off cold_rate | P3.C cold_rate | "
        "P3.C-fix cold_rate | warmup_count | invoke_after_warmup_ms"
    )
    print("-" * 118)
    for stage in STAGE_ORDER:
        stage_warmups = [record for record in warmups if record["stage_name"] == stage]
        stage_deltas = deltas.get(stage, [])
        delta_text = (
            f"mean={sum(stage_deltas) / len(stage_deltas):.1f}, min={min(stage_deltas):.1f}"
            if stage_deltas
            else "n/a"
        )
        note = "entry" if stage == "detect_object" else ""
        print(
            f"{stage:<15} | {off_rates[stage]:<17.2%} | "
            f"{P3C_COLD_RATES[stage]:<14.2%} | {on_rates[stage]:<19.2%} | "
            f"{len(stage_warmups):<12} | {delta_text} {note}"
        )

    print("\nWorkflow E2E")
    print(
        f"JIT off: mean={sum(off_e2e) / len(off_e2e):.1f} ms, "
        f"min={min(off_e2e):.1f}, max={max(off_e2e):.1f}, samples={off_e2e}"
    )
    print(
        f"JIT on : mean={sum(on_e2e) / len(on_e2e):.1f} ms, "
        f"min={min(on_e2e):.1f}, max={max(on_e2e):.1f}, samples={on_e2e}"
    )

    late_count = sum(1 for record in warmups if record.get("late_jit"))
    initial_late_count = sum(
        1
        for record in warmups
        if record.get("late_jit") and record.get("schedule_phase") == "initial"
    )
    upsert_late_count = sum(
        1
        for record in warmups
        if record.get("late_jit") and record.get("schedule_phase") == "upsert"
    )
    print(
        f"\nWarmups fired: {len(warmups)}; "
        f"P3.C late_jit={P3C_LATE_JIT_COUNT}/{P3C_WARMUP_COUNT}; "
        f"P3.C-fix late_jit={late_count}/{len(warmups)} "
        f"(initial={initial_late_count}, upsert={upsert_late_count})"
    )


def run_jit_variant(
    *,
    label: str,
    experiment_label: str,
    enable_jit_sync: bool,
    args: argparse.Namespace,
    auth: str,
    workflow,
    client: OpenWhiskClient,
    warmup_client: OpenWhiskClient,
) -> tuple[list[list[dict]], list[dict], list[dict]]:
    tracker = WarmupStatusTracker()
    recorder = WarmupRecorder(warmup_client, tracker=tracker)
    scheduler = JitScheduler(recorder.callback)
    scheduler.start()
    try:
        rows = run_batch(
            label=label,
            enable_jit=True,
            args=args,
            auth=auth,
            workflow=workflow,
            client=client,
            scheduler=scheduler,
            recorder=recorder,
            tracker=tracker,
            enable_jit_sync=enable_jit_sync,
        )
        warmups = recorder.snapshot()
        timing_rows = build_timing_rows(
            experiment_label=experiment_label,
            jit_rows=rows,
            warmups=warmups,
        )
        return rows, warmups, timing_rows
    finally:
        scheduler.stop()


def main() -> None:
    args = parse_args()
    auth = args.auth or read_guest_auth()
    workflow = load_workflow(args.workflow)
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

    if args.reset_once_before:
        reset_plan_variants(args, auth, workflow)

    if args.jit_only:
        on_rows, warmups, timing_rows = run_jit_variant(
            label=(
                "JIT on sync treatment"
                if args.enable_jit_sync
                else "JIT on speculative treatment"
            ),
            experiment_label=args.experiment_label,
            enable_jit_sync=args.enable_jit_sync,
            args=args,
            auth=auth,
            workflow=workflow,
            client=client,
            warmup_client=warmup_client,
        )
        diag_csv = (
            Path(args.diag_csv)
            if args.diag_csv
            else Path(args.diag_out_dir) / f"{args.experiment_label}_per_request_timing.csv"
        )
        write_timing_csv(diag_csv, timing_rows)
        print(f"\nDiagnostic timing CSV: {diag_csv}")
        print_estimate_progression(timing_rows)
        print_jit_only_summary(on_rows, warmups)
        return

    off_rows = run_batch(
        label="JIT off baseline",
        enable_jit=False,
        args=args,
        auth=auth,
        workflow=workflow,
        client=client,
        scheduler=None,
        recorder=None,
    )
    no_sync_rows, no_sync_warmups, no_sync_timing = run_jit_variant(
        label="JIT on no-sync treatment",
        experiment_label=f"{args.experiment_label}_no_sync",
        enable_jit_sync=False,
        args=args,
        auth=auth,
        workflow=workflow,
        client=client,
        warmup_client=warmup_client,
    )
    sync_rows, sync_warmups, sync_timing = run_jit_variant(
        label="JIT on sync treatment",
        experiment_label=f"{args.experiment_label}_sync",
        enable_jit_sync=True,
        args=args,
        auth=auth,
        workflow=workflow,
        client=client,
        warmup_client=warmup_client,
    )
    timing_rows = no_sync_timing + sync_timing
    diag_csv = (
        Path(args.diag_csv)
        if args.diag_csv
        else Path(args.diag_out_dir) / f"{args.experiment_label}_per_request_timing.csv"
    )
    write_timing_csv(diag_csv, timing_rows)
    print(f"\nDiagnostic timing CSV: {diag_csv}")
    print("\nNo-sync estimate_pose")
    print_estimate_progression(no_sync_timing)
    print("\nSync estimate_pose")
    print_estimate_progression(sync_timing)
    print_three_variant_summary(
        off_rows=off_rows,
        no_sync_rows=no_sync_rows,
        sync_rows=sync_rows,
        no_sync_timing=no_sync_timing,
        sync_timing=sync_timing,
    )

    # Preserve the older two-way summary for continuity with P3.C-fix output.
    print("\nLegacy no-sync summary")
    print_summary(off_rows, no_sync_rows, no_sync_warmups)

    sync_report_csv = Path(args.diag_out_dir) / f"{args.experiment_label}_sync_warmups.csv"
    write_timing_csv(
        Path(args.diag_out_dir) / f"{args.experiment_label}_sync_per_request_timing.csv",
        sync_timing,
    )
    if sync_report_csv:
        # Keep warmup records accessible without changing their schema.
        sync_report_csv.parent.mkdir(parents=True, exist_ok=True)
        if sync_warmups:
            with sync_report_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(sync_warmups[0].keys()))
                writer.writeheader()
                writer.writerows(sync_warmups)


if __name__ == "__main__":
    main()
