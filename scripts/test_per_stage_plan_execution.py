#!/usr/bin/env python3
"""Smoke-test per-stage tier routing through run_one_workflow(plan=...)."""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.openwhisk_client import OpenWhiskClient
from runner.run_workflow import run_one_workflow
from runner.workflow import load_workflow


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

UNIFORM_1280_PLAN = {stage: 1280 for stage in STAGE_ORDER}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workflow",
        default=str(ROOT / "configs" / "civic_alert_flow.yaml"),
    )
    parser.add_argument("--apihost", default="https://192.168.123.17:31001")
    parser.add_argument(
        "--auth",
        default="",
        help="OpenWhisk AUTH. If omitted, read owdev-whisk.auth guest auth via kubectl.",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=300)
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


def normalize_int(value: object) -> int | str:
    if value in ("", None):
        return ""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return str(value)


def unique_values(values: list[object]) -> list[object]:
    normalized = [normalize_int(value) for value in values]
    return sorted(set(normalized), key=lambda item: (str(type(item)), str(item)))


def e2e_ms(rows: list[dict]) -> float:
    return float(rows[0].get("workflow_e2e_ms", 0.0))


def run_plan(
    *,
    label: str,
    plan: dict[str, int],
    slo_class: str,
    workflow,
    client: OpenWhiskClient,
    runs: int,
    max_workers: int,
) -> list[list[dict]]:
    all_rows = []
    print(f"\n== {label} ==", flush=True)
    for idx in range(1, runs + 1):
        rows = run_one_workflow(
            workflow,
            client,
            max_workers=max_workers,
            plan=plan,
            slo_class=slo_class,
        )
        all_rows.append(rows)
        print(
            f"[{idx}/{runs}] request_id={rows[0]['request_id']} "
            f"e2e_ms={e2e_ms(rows):.1f}",
            flush=True,
        )
    return all_rows


def print_routing_table(label: str, plan: dict[str, int], all_rows: list[list[dict]]) -> None:
    print(f"\nRouting verification: {label}")
    print(
        "stage           | planned_tier | observed_allocated_mb | ow_memory_mb | match"
    )
    print("-" * 79)
    stage_rows = [
        row
        for rows in all_rows
        for row in rows
        if row.get("stage_name") != "__entry__"
    ]
    by_stage = {
        stage: [row for row in stage_rows if row.get("stage_name") == stage]
        for stage in STAGE_ORDER
    }
    for stage in STAGE_ORDER:
        planned = plan[stage]
        observed_allocated = unique_values(
            [row.get("allocated_memory_mb", "") for row in by_stage[stage]]
        )
        observed_ow = unique_values([row.get("ow_memory_mb", "") for row in by_stage[stage]])
        ok = observed_allocated == [planned] and observed_ow == [planned]
        print(
            f"{stage:<15} | {planned:<12} | {str(observed_allocated):<21} | "
            f"{str(observed_ow):<12} | {'OK' if ok else 'FAIL'}"
        )


def print_e2e_summary(label: str, all_rows: list[list[dict]]) -> None:
    values = [e2e_ms(rows) for rows in all_rows]
    print(
        f"{label}: mean_e2e_ms={statistics.mean(values):.1f}, "
        f"min={min(values):.1f}, max={max(values):.1f}, samples={values}"
    )


def main() -> None:
    args = parse_args()
    workflow = load_workflow(args.workflow)
    auth = args.auth or read_guest_auth()
    client = OpenWhiskClient(
        apihost=args.apihost,
        auth=auth,
        namespace=workflow.namespace,
        timeout_sec=args.timeout_sec,
    )

    premium_rows = run_plan(
        label="premium_heterogeneous",
        plan=PREMIUM_PLAN,
        slo_class="premium",
        workflow=workflow,
        client=client,
        runs=args.runs,
        max_workers=args.max_workers,
    )
    uniform_rows = run_plan(
        label="uniform_1280_plan_path",
        plan=UNIFORM_1280_PLAN,
        slo_class="premium",
        workflow=workflow,
        client=client,
        runs=args.runs,
        max_workers=args.max_workers,
    )

    print_routing_table("premium_heterogeneous", PREMIUM_PLAN, premium_rows)
    print_routing_table("uniform_1280_plan_path", UNIFORM_1280_PLAN, uniform_rows)
    print()
    print_e2e_summary("premium_heterogeneous", premium_rows)
    print_e2e_summary("uniform_1280_plan_path", uniform_rows)


if __name__ == "__main__":
    main()
