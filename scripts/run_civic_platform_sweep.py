#!/usr/bin/env python3
"""Run paired all-cold/all-warm civic workflow samples by memory tier."""

import argparse
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
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
    parser.add_argument(
        "--tier-cycles",
        default="",
        help=(
            "optional per-tier cycle overrides, e.g. "
            "'512:15,768:15,1280:5'; unspecified tiers use --cycles"
        ),
    )
    parser.add_argument(
        "--worker-rule",
        choices=["auto", "round", "ceil"],
        default="auto",
        help=(
            "force an explicit per-tier parallel_workers count. "
            "'auto' preserves workflow/action defaults; 'round' and 'ceil' "
            "derive workers from the configured CPU estimate."
        ),
    )
    parser.add_argument(
        "--parallel-workers-override",
        default="",
        help=(
            "optional explicit tier:workers map, e.g. '1536:2,3072:3'. "
            "Entries override --worker-rule for those tiers."
        ),
    )
    parser.add_argument(
        "--between-tier-sleep-sec",
        type=float,
        default=0.0,
        help="optional sleep between memory tiers to let old containers drain",
    )
    parser.add_argument(
        "--abort-on-pending",
        action="store_true",
        help="abort if any OpenWhisk pod is Pending during the sweep",
    )
    parser.add_argument(
        "--pending-namespace",
        default="openwhisk",
        help="namespace checked by --abort-on-pending",
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


def resolve_tier_cycles(args: argparse.Namespace) -> dict[int, int]:
    cycles_by_tier = {memory_mb: args.cycles for memory_mb in args.memory_tiers}
    if not args.tier_cycles:
        return cycles_by_tier

    for item in args.tier_cycles.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid --tier-cycles entry: {item!r}")
        tier_text, cycles_text = item.split(":", 1)
        try:
            tier = int(tier_text.strip())
            cycles = int(cycles_text.strip())
        except ValueError as exc:
            raise ValueError(f"invalid --tier-cycles entry: {item!r}") from exc
        if tier < 1 or cycles < 1:
            raise ValueError(f"--tier-cycles entries must be positive: {item!r}")
        if tier in cycles_by_tier:
            cycles_by_tier[tier] = cycles
    return cycles_by_tier


def parse_parallel_worker_overrides(value: str) -> dict[int, int]:
    overrides: dict[int, int] = {}
    if not value:
        return overrides

    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid --parallel-workers-override entry: {item!r}")
        tier_text, workers_text = item.split(":", 1)
        try:
            tier = int(tier_text.strip())
            workers = int(workers_text.strip())
        except ValueError as exc:
            raise ValueError(
                f"invalid --parallel-workers-override entry: {item!r}"
            ) from exc
        if tier < 1 or workers < 1:
            raise ValueError(
                "--parallel-workers-override entries must be positive: "
                f"{item!r}"
            )
        overrides[tier] = workers
    return overrides


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


def workers_for_rule(memory_mb: int, args: argparse.Namespace, rule: str) -> int:
    cpu = estimated_cpu_cores(memory_mb, args)
    if rule == "round":
        return max(1, round(cpu))
    if rule == "ceil":
        return max(1, math.ceil(cpu))
    raise ValueError(f"unsupported worker rule: {rule}")


def resolve_worker_counts(
    args: argparse.Namespace,
    overrides: dict[int, int],
) -> dict[int, int | None]:
    counts: dict[int, int | None] = {}
    for memory_mb in args.memory_tiers:
        if memory_mb in overrides:
            counts[memory_mb] = overrides[memory_mb]
        elif args.worker_rule == "auto":
            counts[memory_mb] = None
        else:
            counts[memory_mb] = workers_for_rule(memory_mb, args, args.worker_rule)
    return counts


def with_parallel_workers(workflow, workers: int | None):
    if workers is None:
        return workflow
    return replace(
        workflow,
        nodes={
            name: replace(node, parallel_workers=str(workers))
            for name, node in workflow.nodes.items()
        },
    )


def pending_pod_count(namespace: str) -> int:
    result = subprocess.run(
        ["kubectl", "get", "pods", "-n", namespace, "--no-headers"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return sum(1 for line in result.stdout.splitlines() if "Pending" in line)


def assert_no_pending(namespace: str, context: str) -> None:
    count = pending_pod_count(namespace)
    print(f"pending_pods[{context}]={count}", flush=True)
    if count:
        raise RuntimeError(f"aborting: {count} Pending pods detected during {context}")


def sleep_between_tiers(seconds: float, namespace: str, abort_on_pending: bool) -> None:
    if seconds <= 0:
        return
    print(f"sleeping_between_tiers_sec={seconds:.1f}", flush=True)
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(5.0, remaining))
        if abort_on_pending:
            assert_no_pending(namespace, "between_tiers")


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


def print_worker_resolution(
    args: argparse.Namespace,
    worker_counts: dict[int, int | None],
) -> None:
    rows = []
    for memory_mb in args.memory_tiers:
        forced = worker_counts[memory_mb]
        rows.append(
            [
                memory_mb,
                estimated_cpu_cores(memory_mb, args),
                workers_for_rule(memory_mb, args, "round"),
                workers_for_rule(memory_mb, args, "ceil"),
                "auto" if forced is None else forced,
            ]
        )
    print("\nWorker resolution")
    print_table(["mem_MiB", "cpu_cores", "round", "ceil", "used"], rows)


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
    if args.between_tier_sleep_sec < 0:
        raise ValueError("--between-tier-sleep-sec must be >= 0")
    tier_cycles = resolve_tier_cycles(args)
    worker_overrides = parse_parallel_worker_overrides(args.parallel_workers_override)
    worker_counts = resolve_worker_counts(args, worker_overrides)

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
    if args.tier_cycles:
        print(f"tier_cycles={tier_cycles}")
    print(f"cpu_profile={args.cpu_profile}")
    print(f"worker_rule={args.worker_rule}; worker_overrides={worker_overrides}")
    print(f"trace={trace_path}")
    print("Cold samples are forced by redeploying each action revision before the call.")
    print_worker_resolution(args, worker_counts)

    samples = []
    for tier_index, memory_mb in enumerate(args.memory_tiers):
        tier_workflow = with_parallel_workers(workflow, worker_counts[memory_mb])
        worker_label = "auto" if worker_counts[memory_mb] is None else worker_counts[memory_mb]
        print(
            f"\n=== memory={memory_mb} MiB "
            f"(configured CPU estimate={estimated_cpu_m(memory_mb, args)}m; "
            f"parallel_workers={worker_label}) ==="
        )
        if args.abort_on_pending:
            assert_no_pending(args.pending_namespace, f"before_tier_{memory_mb}")
        for cycle in range(1, tier_cycles[memory_mb] + 1):
            if args.abort_on_pending:
                assert_no_pending(args.pending_namespace, f"before_deploy_{memory_mb}_{cycle}")
            update_actions(args, auth, action_names, memory_mb)
            if args.abort_on_pending:
                assert_no_pending(args.pending_namespace, f"after_deploy_{memory_mb}_{cycle}")

            cold_rows = run_one_workflow(
                tier_workflow,
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
            if args.abort_on_pending:
                assert_no_pending(args.pending_namespace, f"after_cold_{memory_mb}_{cycle}")

            warm_rows = run_one_workflow(
                tier_workflow,
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
            if args.abort_on_pending:
                assert_no_pending(args.pending_namespace, f"after_warm_{memory_mb}_{cycle}")

        if tier_index < len(args.memory_tiers) - 1:
            sleep_between_tiers(
                args.between_tier_sleep_sec,
                args.pending_namespace,
                args.abort_on_pending,
            )

    print_summary(samples, args)
    print(f"\nRaw trace written to {trace_path}")
    print(f"Actions remain configured at the final tier: {args.memory_tiers[-1]} MiB.")


if __name__ == "__main__":
    main()
