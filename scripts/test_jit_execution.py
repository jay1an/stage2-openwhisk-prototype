#!/usr/bin/env python3
"""Exercise P3.C JIT warmup scheduling against real OpenWhisk actions."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--skip-reset", action="store_true")
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


class WarmupRecorder:
    def __init__(self, client: OpenWhiskClient):
        self.client = client
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
        activation = self.client.invoke_activation(task.action_name, params)
        callback_end = time.monotonic()
        result = activation.get("response", {}).get("result", {})
        annotations = activation_annotations(activation)
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
                    "callback_start": callback_start,
                    "callback_end": callback_end,
                    "activation_duration_ms": activation.get("duration", ""),
                    "action_duration_ms": result.get("action_duration_ms", ""),
                    "cold_like": result.get("cold_like", ""),
                    "container_id": result.get("container_id", ""),
                    "initTime": annotations.get("initTime", ""),
                    "waitTime": annotations.get("waitTime", ""),
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
            f"jit_late={rows[0].get('jit_late_count', 0)}",
            flush=True,
        )
    return results


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
    print("stage           | JIT off cold_rate | JIT on cold_rate | warmup_count | invoke_after_warmup_ms")
    print("-" * 95)
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
            f"{stage:<15} | {off_rates[stage]:<17.2%} | {on_rates[stage]:<16.2%} | "
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
    print(f"\nWarmups fired: {len(warmups)}; late_jit_count={late_count}")


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
    recorder = WarmupRecorder(warmup_client)
    scheduler = JitScheduler(recorder.callback)
    scheduler.start()
    try:
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
        on_rows = run_batch(
            label="JIT on treatment",
            enable_jit=True,
            args=args,
            auth=auth,
            workflow=workflow,
            client=client,
            scheduler=scheduler,
            recorder=recorder,
        )
        print_summary(off_rows, on_rows, recorder.snapshot())
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()
