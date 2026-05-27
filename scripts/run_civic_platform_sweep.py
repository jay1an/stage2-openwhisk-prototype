#!/usr/bin/env python3
"""Run paired all-cold/all-warm civic workflow samples by memory tier."""

import argparse
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.openwhisk_client import OpenWhiskClient
from runner.resource_profiles import memory_to_cpu_cores
from runner.run_workflow import run_one_workflow
from runner.trace_store import CsvTraceStore
from runner.workflow import load_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure civic_alert_flow with request-level OpenWhisk platform "
            "breakdown at one or more memory tiers."
        )
    )
    parser.add_argument("--apihost", required=True)
    parser.add_argument(
        "--auth",
        default="",
        help="OpenWhisk AUTH; when omitted, read owdev-whisk.auth guest auth via kubectl.",
    )
    parser.add_argument(
        "--workflow",
        default=str(ROOT / "configs" / "civic_alert_flow.yaml"),
    )
    parser.add_argument(
        "--action-file",
        default=str(ROOT / "actions" / "workflow_action.py"),
        help="action source redeployed before every cold sample",
    )
    parser.add_argument(
        "--memory-tiers",
        nargs="+",
        type=int,
        default=[256, 512, 1280, 2560],
        metavar="MB",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="number of paired all-cold then all-warm workflow runs per tier",
    )
    parser.add_argument("--kind", default="python:3")
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--wsk-cli", default="wsk")
    parser.add_argument(
        "--trace",
        default="",
        help="CSV output path; defaults to a timestamped data/civic_platform_sweep path",
    )
    parser.add_argument(
        "--cpu-base-memory-mb",
        type=int,
        default=256,
        help="deployed OpenWhisk CPU-scaling base memory, used only for table labels",
    )
    parser.add_argument(
        "--cpu-base-millicpu",
        type=int,
        default=200,
        help="deployed OpenWhisk CPU-scaling millicpu at base memory",
    )
    parser.add_argument(
        "--cpu-max-millicpu",
        type=int,
        default=3200,
        help="deployed OpenWhisk CPU-scaling cap, used only for table labels",
    )
    parser.add_argument(
        "--cpu-profile",
        default="huawei_functiongraph",
        choices=[
            "huawei_functiongraph",
            "huawei",
            "functiongraph",
            "legacy_256mb_250m",
            "openwhisk_256mb_250m",
            "custom",
        ],
        help="memory-to-CPU profile passed to workflow actions",
    )
    parser.add_argument(
        "--cpu-per-memory-mb",
        type=float,
        default=None,
        help="used only with --cpu-profile custom",
    )
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


def estimated_cpu_m(memory_mb: int, args: argparse.Namespace) -> int:
    if args.cpu_profile == "custom":
        scaled = memory_mb * args.cpu_base_millicpu / args.cpu_base_memory_mb
        return min(args.cpu_max_millicpu, int(round(scaled)))
    cores = memory_to_cpu_cores(
        memory_mb,
        profile=args.cpu_profile,
        cpu_per_memory_mb=args.cpu_per_memory_mb,
    )
    return min(args.cpu_max_millicpu, int(round(float(cores or 0.0) * 1000.0)))


def estimated_cpu_cores(memory_mb: int, args: argparse.Namespace) -> float:
    return estimated_cpu_m(memory_mb, args) / 1000.0


def update_actions(
    args: argparse.Namespace,
    auth: str,
    action_names: list[str],
    memory_mb: int,
) -> None:
    common = [
        args.wsk_cli,
        "-i",
        "--apihost",
        args.apihost,
        "--auth",
        auth,
        "action",
    ]
    for action in action_names:
        get_result = subprocess.run(
            [*common, "get", action],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        operation = "update" if get_result.returncode == 0 else "create"
        subprocess.run(
            [
                *common,
                operation,
                action,
                args.action_file,
                "--kind",
                args.kind,
                "--memory",
                str(memory_mb),
                "--timeout",
                str(args.timeout_ms),
            ],
            stdout=subprocess.DEVNULL,
            check=True,
        )


def stage_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["stage_name"] != "__entry__"]


def workflow_e2e_ms(rows: list[dict]) -> float:
    reported = rows[0].get("workflow_e2e_ms")
    if reported not in ("", None):
        return float(reported)
    entry_ts_ms = float(rows[0]["entry_ts_ms"])
    return max(float(row["dispatch_end_ms"]) for row in stage_rows(rows)) - entry_ts_ms


def as_bool(value: object) -> bool:
    return str(value).lower() == "true"


def assert_phase(rows: list[dict], phase: str) -> None:
    stages = stage_rows(rows)
    expect_cold = phase == "cold"
    app_cold = sum(as_bool(row.get("cold_like")) for row in stages)
    ow_cold = sum(as_bool(row.get("ow_cold_start")) for row in stages)
    expected = len(stages) if expect_cold else 0
    if app_cold != expected or ow_cold != expected:
        raise RuntimeError(
            f"{phase} validation failed: action cold={app_cold}/{len(stages)}, "
            f"OpenWhisk cold={ow_cold}/{len(stages)}"
        )


def number(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def fmt(value: object, digits: int = 1) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    text_rows = [[fmt(value) for value in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in text_rows))
        for index in range(len(headers))
    ]
    separator = "-+-".join("-" * width for width in widths)
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print(separator)
    for row in text_rows:
        print(" | ".join(value.rjust(widths[index]) for index, value in enumerate(row)))


def print_summary(samples: list[dict], args: argparse.Namespace) -> None:
    print("\nE2E summary (mean over paired cycles)")
    e2e_rows = []
    for memory_mb in args.memory_tiers:
        for phase in ("cold", "warm"):
            group = [
                sample["e2e_ms"]
                for sample in samples
                if sample["memory_mb"] == memory_mb and sample["phase"] == phase
            ]
            e2e_rows.append(
                [
                    memory_mb,
                    estimated_cpu_m(memory_mb, args),
                    phase,
                    len(group),
                    statistics.mean(group),
                    min(group),
                    max(group),
                ]
            )
    print_table(
        ["mem_MiB", "cpu_m", "phase", "n", "e2e_mean_ms", "min_ms", "max_ms"],
        e2e_rows,
    )

    groups: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for sample in samples:
        for row in stage_rows(sample["rows"]):
            groups[(sample["memory_mb"], sample["phase"], row["stage_name"])].append(row)

    print("\nPer-stage breakdown (mean ms; non_action = dispatch - action)")
    detail_rows = []
    for memory_mb in args.memory_tiers:
        for phase in ("cold", "warm"):
            for stage in (
                "detect_object",
                "estimate_pose",
                "match_face",
                "classify_scene",
                "translate_alert",
            ):
                rows = groups[(memory_mb, phase, stage)]
                mean = lambda key: statistics.mean(number(row, key) for row in rows)
                detail_rows.append(
                    [
                        memory_mb,
                        estimated_cpu_cores(memory_mb, args),
                        phase,
                        stage,
                        mean("dispatch_latency_ms"),
                        mean("action_duration_ms"),
                        mean("cpu_process_ms"),
                        mean("serial_wall_ms"),
                        mean("io_wall_ms"),
                        mean("parallel_wall_ms"),
                        mean("memory_wall_ms"),
                        mean("parallel_workers_used"),
                        mean("observed_effective_cores"),
                        mean("ow_wait_ms"),
                        mean("ow_init_ms"),
                        mean("ow_runtime_overhead_ms"),
                        mean("client_gateway_overhead_ms"),
                        mean("platform_overhead_ms"),
                    ]
                )
    print_table(
        [
            "mem",
            "cpu",
            "phase",
            "stage",
            "dispatch",
            "action",
            "cpu",
            "serial",
            "io",
            "parallel",
            "memory",
            "workers",
            "eff_cores",
            "wait",
            "init",
            "ow_wrap",
            "edge",
            "non_action",
        ],
        detail_rows,
    )


def main() -> None:
    args = parse_args()
    if args.cycles < 1:
        raise ValueError("--cycles must be >= 1")
    if not args.memory_tiers or any(memory < 1 for memory in args.memory_tiers):
        raise ValueError("--memory-tiers must contain positive MB values")

    auth = auth_from_args(args.auth)
    workflow = load_workflow(args.workflow)
    action_names = [node.action for node in workflow.nodes.values()]
    trace_path = Path(args.trace) if args.trace else (
        ROOT
        / "data"
        / f"civic_platform_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        / "trace.csv"
    )
    store = CsvTraceStore(str(trace_path))
    client = OpenWhiskClient(
        apihost=args.apihost,
        auth=auth,
        namespace=workflow.namespace,
        timeout_sec=max(60, args.timeout_ms // 1000 + 10),
    )

    print(f"workflow={workflow.workflow_name}")
    print(f"memory_tiers={args.memory_tiers}; paired_cycles={args.cycles}")
    print(f"cpu_profile={args.cpu_profile}")
    print(f"trace={trace_path}")
    print("Cold samples are forced by redeploying each action revision before the call.")

    samples = []
    for memory_mb in args.memory_tiers:
        print(
            f"\n=== memory={memory_mb} MiB "
            f"(configured CPU estimate={estimated_cpu_m(memory_mb, args)}m) ==="
        )
        for cycle in range(1, args.cycles + 1):
            update_actions(args, auth, action_names, memory_mb)

            cold_rows = run_one_workflow(
                workflow,
                client,
                args.max_workers,
                allocated_memory_mb=memory_mb,
                allocated_cpu_cores=estimated_cpu_cores(memory_mb, args),
            )
            assert_phase(cold_rows, "cold")
            store.append_many(cold_rows)
            cold_sample = {
                "memory_mb": memory_mb,
                "phase": "cold",
                "cycle": cycle,
                "e2e_ms": workflow_e2e_ms(cold_rows),
                "rows": cold_rows,
            }
            samples.append(cold_sample)

            warm_rows = run_one_workflow(
                workflow,
                client,
                args.max_workers,
                allocated_memory_mb=memory_mb,
                allocated_cpu_cores=estimated_cpu_cores(memory_mb, args),
            )
            assert_phase(warm_rows, "warm")
            store.append_many(warm_rows)
            warm_sample = {
                "memory_mb": memory_mb,
                "phase": "warm",
                "cycle": cycle,
                "e2e_ms": workflow_e2e_ms(warm_rows),
                "rows": warm_rows,
            }
            samples.append(warm_sample)

            print(
                f"cycle={cycle} cold_e2e_ms={cold_sample['e2e_ms']:.1f} "
                f"warm_e2e_ms={warm_sample['e2e_ms']:.1f}"
            )

    print_summary(samples, args)
    print(f"\nRaw trace written to {trace_path}")
    print(f"Actions remain configured at the final tier: {args.memory_tiers[-1]} MiB.")


if __name__ == "__main__":
    main()
