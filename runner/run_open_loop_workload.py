import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

from .openwhisk_client import OpenWhiskClient
from .run_workflow import run_one_workflow
from .trace_store import CsvTraceStore
from .workflow import WorkflowSpec, load_workflow
from .workload import WorkloadEvent, generate_workload_events


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--apihost", required=True)
    parser.add_argument("--auth", required=True)
    parser.add_argument("--trace", default="data/traces_open_loop.csv")
    parser.add_argument("--schedule-out", default="data/workload_schedule_open_loop.csv")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--pattern",
        choices=["constant", "burst", "periodic", "sparse", "poisson"],
        default="constant",
    )
    parser.add_argument("--base-interval-ms", type=int, default=500)
    parser.add_argument("--burst-every", type=int, default=20)
    parser.add_argument("--burst-size", type=int, default=5)
    parser.add_argument("--burst-interval-ms", type=int, default=50)
    parser.add_argument("--idle-interval-ms", type=int, default=3000)
    parser.add_argument("--period-steps", type=int, default=30)
    parser.add_argument("--amplitude", type=float, default=0.6)
    parser.add_argument("--sparse-probability", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=4,
        help="maximum concurrently running workflow invocations",
    )
    parser.add_argument(
        "--stage-max-workers",
        type=int,
        default=8,
        help="maximum concurrent ready stages inside one workflow invocation",
    )
    return parser.parse_args()


def build_target_schedule(events: Iterable[WorkloadEvent]) -> List[Tuple[WorkloadEvent, int]]:
    offset_ms = 0
    schedule = []
    for event in events:
        offset_ms += event.sleep_before_ms
        schedule.append((event, offset_ms))
    return schedule


def append_schedule_row(path: str, row: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    columns = [
        "workflow_name",
        "pattern",
        "index",
        "request_id",
        "target_ms",
        "target_offset_ms",
        "sleep_before_ms",
        "start_ms",
        "end_ms",
        "target_lag_ms",
        "status",
        "error",
    ]
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in columns})


def run_scheduled_workflow(
    workflow: WorkflowSpec,
    apihost: str,
    auth: str,
    event: WorkloadEvent,
    target_ms: int,
    target_offset_ms: int,
    stage_max_workers: int,
) -> tuple[dict, list[dict]]:
    client = OpenWhiskClient(
        apihost=apihost,
        auth=auth,
        namespace=workflow.namespace,
    )
    start_ms = now_ms()
    request_id = ""
    status = "ok"
    error = ""
    rows: list[dict] = []

    try:
        rows = run_one_workflow(workflow, client, stage_max_workers)
        request_id = rows[0]["request_id"]
    except Exception as exc:
        status = "error"
        error = str(exc)

    end_ms = now_ms()
    schedule_row = {
        "workflow_name": workflow.workflow_name,
        "pattern": event.pattern,
        "index": event.index,
        "request_id": request_id,
        "target_ms": target_ms,
        "target_offset_ms": target_offset_ms,
        "sleep_before_ms": event.sleep_before_ms,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "target_lag_ms": start_ms - target_ms,
        "status": status,
        "error": error,
    }
    return schedule_row, rows


def main() -> None:
    args = parse_args()
    workflow = load_workflow(args.workflow)
    store = CsvTraceStore(args.trace)
    events = generate_workload_events(
        pattern=args.pattern,
        count=args.count,
        base_interval_ms=args.base_interval_ms,
        seed=args.seed,
        burst_every=args.burst_every,
        burst_size=args.burst_size,
        burst_interval_ms=args.burst_interval_ms,
        idle_interval_ms=args.idle_interval_ms,
        period_steps=args.period_steps,
        amplitude=args.amplitude,
        sparse_probability=args.sparse_probability,
    )
    schedule = build_target_schedule(events)
    base_ms = now_ms()

    futures = []
    with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
        for event, target_offset_ms in schedule:
            target_ms = base_ms + target_offset_ms
            delay_ms = target_ms - now_ms()
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            futures.append(
                pool.submit(
                    run_scheduled_workflow,
                    workflow,
                    args.apihost,
                    args.auth,
                    event,
                    target_ms,
                    target_offset_ms,
                    args.stage_max_workers,
                )
            )
            print(
                f"submitted [{event.index + 1}/{len(schedule)}] "
                f"workflow={workflow.workflow_name} pattern={event.pattern} "
                f"target_offset_ms={target_offset_ms}"
            )

        for future in as_completed(futures):
            schedule_row, rows = future.result()
            if rows:
                store.append_many(rows)
            append_schedule_row(args.schedule_out, schedule_row)
            print(
                f"completed [{schedule_row['index'] + 1}/{len(schedule)}] "
                f"request_id={schedule_row['request_id'] or '-'} "
                f"status={schedule_row['status']} "
                f"target_lag_ms={schedule_row['target_lag_ms']}"
            )
            if schedule_row["status"] != "ok":
                print(f"  error={schedule_row['error']}")


if __name__ == "__main__":
    main()
