import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .openwhisk_client import OpenWhiskClient
from .run_workflow import run_one_workflow
from .trace_store import CsvTraceStore
from .workflow import WorkflowSpec, load_workflow


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a real workflow-entry schedule using open-loop timing."
    )
    parser.add_argument("--workflow", required=True, help="workflow YAML config")
    parser.add_argument("--schedule", required=True, help="CSV with target_offset_ms")
    parser.add_argument("--apihost", required=True)
    parser.add_argument("--auth", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--schedule-out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means replay all rows")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="divide inter-arrival gaps by this value; use >1 to speed up replay",
    )
    parser.add_argument(
        "--max-gap-ms",
        type=int,
        default=0,
        help="cap each inter-arrival gap after time scaling; 0 disables capping",
    )
    parser.add_argument(
        "--min-gap-ms",
        type=int,
        default=0,
        help="floor each positive inter-arrival gap after time scaling; 0 disables flooring",
    )
    parser.add_argument(
        "--invoke-timeout-sec",
        type=int,
        default=60,
        help="HTTP timeout for each blocking OpenWhisk action invoke",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=8,
        help="maximum concurrently running workflow invocations",
    )
    parser.add_argument(
        "--stage-max-workers",
        type=int,
        default=8,
        help="maximum concurrent ready stages inside one workflow invocation",
    )
    return parser.parse_args()


def load_schedule(path: str, limit: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "target_offset_ms" not in df.columns:
        raise ValueError("schedule CSV must contain target_offset_ms")
    if "index" in df.columns:
        df = df.sort_values("index")
    else:
        df = df.sort_values("target_offset_ms").reset_index(drop=True)
        df["index"] = range(len(df))
    if limit and limit > 0:
        df = df.head(limit)
    return df.reset_index(drop=True)


def build_replay_offsets(
    source_offsets: Iterable[int],
    time_scale: float,
    max_gap_ms: int,
    min_gap_ms: int,
) -> List[int]:
    if time_scale <= 0:
        raise ValueError("--time-scale must be > 0")
    if min_gap_ms < 0:
        raise ValueError("--min-gap-ms must be >= 0")

    replay_offsets: List[int] = []
    previous_source = None
    current_replay = 0

    for source in source_offsets:
        source = int(source)
        if previous_source is None:
            current_replay = 0
        else:
            gap = max(0, source - previous_source)
            scaled_gap = int(round(gap / time_scale))
            if gap > 0 and min_gap_ms and min_gap_ms > 0:
                scaled_gap = max(scaled_gap, min_gap_ms)
            if max_gap_ms and max_gap_ms > 0:
                scaled_gap = min(scaled_gap, max_gap_ms)
            current_replay += max(0, scaled_gap)
        replay_offsets.append(current_replay)
        previous_source = source

    return replay_offsets


def append_schedule_row(path: str, row: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    columns = [
        "workflow_name",
        "source_label",
        "source_app",
        "source_func",
        "source_start_s",
        "source_end_s",
        "source_duration_ms",
        "index",
        "request_id",
        "source_target_offset_ms",
        "target_ms",
        "target_offset_ms",
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
    source_row: dict,
    target_ms: int,
    replay_offset_ms: int,
    stage_max_workers: int,
    invoke_timeout_sec: int,
) -> tuple[dict, list[dict]]:
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
        "source_label": source_row.get("source_label", ""),
        "source_app": source_row.get("source_app", ""),
        "source_func": source_row.get("source_func", ""),
        "source_start_s": source_row.get("source_start_s", ""),
        "source_end_s": source_row.get("source_end_s", ""),
        "source_duration_ms": source_row.get("source_duration_ms", ""),
        "index": int(source_row.get("index", 0)),
        "request_id": request_id,
        "source_target_offset_ms": int(source_row.get("target_offset_ms", 0)),
        "target_ms": target_ms,
        "target_offset_ms": replay_offset_ms,
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
    schedule_df = load_schedule(args.schedule, args.limit)
    replay_offsets = build_replay_offsets(
        schedule_df["target_offset_ms"].astype(int).tolist(),
        time_scale=args.time_scale,
        max_gap_ms=args.max_gap_ms,
        min_gap_ms=args.min_gap_ms,
    )

    store = CsvTraceStore(args.trace)
    base_ms = now_ms()
    total = len(schedule_df)
    futures = []

    with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
        for row_index, source_row in enumerate(schedule_df.to_dict("records")):
            replay_offset_ms = replay_offsets[row_index]
            target_ms = base_ms + replay_offset_ms
            delay_ms = target_ms - now_ms()
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            futures.append(
                pool.submit(
                    run_scheduled_workflow,
                    workflow,
                    args.apihost,
                    args.auth,
                    source_row,
                    target_ms,
                    replay_offset_ms,
                    args.stage_max_workers,
                    args.invoke_timeout_sec,
                )
            )
            print(
                f"submitted [{row_index + 1}/{total}] "
                f"workflow={workflow.workflow_name} "
                f"source_label={source_row.get('source_label', '')} "
                f"target_offset_ms={replay_offset_ms}"
            )

        for future in as_completed(futures):
            schedule_row, rows = future.result()
            if rows:
                store.append_many(rows)
            append_schedule_row(args.schedule_out, schedule_row)
            print(
                f"completed [{schedule_row['index'] + 1}/{total}] "
                f"request_id={schedule_row['request_id'] or '-'} "
                f"status={schedule_row['status']} "
                f"target_lag_ms={schedule_row['target_lag_ms']}"
            )
            if schedule_row["status"] != "ok":
                print(f"  error={schedule_row['error']}")


if __name__ == "__main__":
    main()
