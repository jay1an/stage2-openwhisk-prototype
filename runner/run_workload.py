import argparse
import csv
import time
from pathlib import Path

from .openwhisk_client import OpenWhiskClient
from .run_workflow import run_one_workflow
from .trace_store import CsvTraceStore
from .workflow import load_workflow
from .workload import generate_workload_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--apihost", required=True)
    parser.add_argument("--auth", required=True)
    parser.add_argument("--trace", default="data/traces.csv")
    parser.add_argument("--schedule-out", default="data/workload_schedule.csv")
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
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def append_schedule_row(path: str, row: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    columns = [
        "workflow_name",
        "pattern",
        "index",
        "request_id",
        "sleep_before_ms",
        "start_ms",
        "end_ms",
        "status",
        "error",
    ]
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in columns})


def main() -> None:
    args = parse_args()
    workflow = load_workflow(args.workflow)
    client = OpenWhiskClient(
        apihost=args.apihost,
        auth=args.auth,
        namespace=workflow.namespace,
    )
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

    for event in events:
        if event.sleep_before_ms > 0:
            time.sleep(event.sleep_before_ms / 1000.0)

        start_ms = time.time_ns() // 1_000_000
        request_id = ""
        status = "ok"
        error = ""
        try:
            rows = run_one_workflow(workflow, client, args.max_workers)
            request_id = rows[0]["request_id"]
            store.append_many(rows)
        except Exception as exc:
            status = "error"
            error = str(exc)
        end_ms = time.time_ns() // 1_000_000

        append_schedule_row(
            args.schedule_out,
            {
                "workflow_name": workflow.workflow_name,
                "pattern": event.pattern,
                "index": event.index,
                "request_id": request_id,
                "sleep_before_ms": event.sleep_before_ms,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "status": status,
                "error": error,
            },
        )

        print(
            f"[{event.index + 1}/{len(events)}] workflow={workflow.workflow_name} "
            f"pattern={event.pattern} sleep_before_ms={event.sleep_before_ms} "
            f"request_id={request_id or '-'} status={status}"
        )
        if status != "ok":
            print(f"  error={error}")


if __name__ == "__main__":
    main()

