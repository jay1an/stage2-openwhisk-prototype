import argparse
import csv
import math
import statistics
import time
import uuid
from pathlib import Path

from ..openwhisk_client import OpenWhiskClient


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark whether OpenWhisk CPU scaling improves a CPU-bound action. "
            "Deploy the same actions/cpu_probe.py at several memory tiers first."
        )
    )
    parser.add_argument("--apihost", required=True)
    parser.add_argument("--auth", required=True)
    parser.add_argument("--namespace", default="guest")
    parser.add_argument(
        "--actions",
        default="cpu_probe_256:256,cpu_probe_512:512,cpu_probe_1024:1024",
        help="comma-separated action_name:memory_mb pairs",
    )
    parser.add_argument("--iterations", type=int, default=8_000_000)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--out", default="data/cpu_scaling_benchmark.csv")
    return parser.parse_args()


def parse_actions(value: str) -> list[tuple[str, int]]:
    pairs = []
    for item in value.split(","):
        if not item.strip():
            continue
        name, memory = item.split(":", 1)
        pairs.append((name.strip(), int(memory)))
    if not pairs:
        raise ValueError("--actions must contain at least one action_name:memory_mb pair")
    return pairs


def quantile(values: list[float], q: float) -> float:
    clean = sorted(float(value) for value in values if not math.isnan(float(value)))
    if not clean:
        return float("nan")
    index = min(len(clean) - 1, max(0, int(math.ceil(q * len(clean)) - 1)))
    return clean[index]


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for action in sorted({row["action"] for row in rows}):
        group = [row for row in rows if row["action"] == action and row["phase"] == "sample"]
        if not group:
            continue
        dispatch = [float(row["dispatch_latency_ms"]) for row in group]
        action_duration = [float(row["action_duration_ms"]) for row in group]
        process_cpu = [float(row["process_cpu_ms"]) for row in group]
        summary.append(
            {
                "action": action,
                "memory_mb": group[0]["memory_mb"],
                "samples": len(group),
                "dispatch_mean_ms": statistics.mean(dispatch),
                "dispatch_p50_ms": statistics.median(dispatch),
                "dispatch_p95_ms": quantile(dispatch, 0.95),
                "action_mean_ms": statistics.mean(action_duration),
                "action_p50_ms": statistics.median(action_duration),
                "action_p95_ms": quantile(action_duration, 0.95),
                "process_cpu_mean_ms": statistics.mean(process_cpu),
                "cold_like_count": sum(1 for row in group if str(row["cold_like"]).lower() == "true"),
            }
        )
    return summary


def write_csv(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "phase",
        "action",
        "memory_mb",
        "sample_index",
        "request_id",
        "dispatch_latency_ms",
        "action_duration_ms",
        "process_cpu_ms",
        "cold_like",
        "container_id",
        "checksum",
        "iterations",
        "repeat",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def main() -> None:
    args = parse_args()
    actions = parse_actions(args.actions)
    client = OpenWhiskClient(
        apihost=args.apihost,
        auth=args.auth,
        namespace=args.namespace,
        timeout_sec=args.timeout_sec,
    )
    rows = []
    for action, memory_mb in actions:
        total = args.warmup + args.samples
        for index in range(total):
            phase = "warmup" if index < args.warmup else "sample"
            request_id = f"cpu-scale-{memory_mb}-{uuid.uuid4()}"
            start_ms = now_ms()
            result = client.invoke_action(
                action,
                {
                    "workflow_name": "cpu_scaling",
                    "stage_name": action,
                    "request_id": request_id,
                    "iterations": args.iterations,
                    "repeat": args.repeat,
                },
            )
            end_ms = now_ms()
            row = {
                "phase": phase,
                "action": action,
                "memory_mb": memory_mb,
                "sample_index": index,
                "request_id": request_id,
                "dispatch_latency_ms": end_ms - start_ms,
                "action_duration_ms": result.get("action_duration_ms", ""),
                "process_cpu_ms": result.get("process_cpu_ms", ""),
                "cold_like": result.get("cold_like", ""),
                "container_id": result.get("container_id", ""),
                "checksum": result.get("checksum", ""),
                "iterations": result.get("iterations", args.iterations),
                "repeat": result.get("repeat", args.repeat),
            }
            rows.append(row)
            print(
                f"{phase} action={action} memory={memory_mb} "
                f"dispatch_ms={row['dispatch_latency_ms']} "
                f"action_ms={row['action_duration_ms']} "
                f"cold_like={row['cold_like']}"
            )

    write_csv(args.out, rows)
    print(f"\nwrote detail: {args.out}")
    print("\nsummary:")
    for item in summarize(rows):
        print(
            "action={action} memory={memory_mb} samples={samples} "
            "action_p50_ms={action_p50_ms:.2f} action_p95_ms={action_p95_ms:.2f} "
            "dispatch_p50_ms={dispatch_p50_ms:.2f} cold_like_count={cold_like_count}".format(
                **item
            )
        )


if __name__ == "__main__":
    main()

