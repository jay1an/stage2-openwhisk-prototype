import argparse
import json
import uuid
from pathlib import Path

import pandas as pd

from ..workflow import load_workflow


TRACE_COLUMNS = [
    "workflow_name",
    "request_id",
    "stage_name",
    "parent_stages",
    "entry_ts_ms",
    "dispatch_start_ms",
    "dispatch_end_ms",
    "dispatch_latency_ms",
    "action_start_ns",
    "action_end_ns",
    "action_duration_ms",
    "platform_overhead_ms",
    "container_id",
    "cold_like",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract an entry-only workflow trace from AzureFunctionsInvocationTrace2021 "
            "for a selected Azure (app, func) pair."
        )
    )
    parser.add_argument("--azure-trace", required=True, help="Azure invocation trace CSV/TXT")
    parser.add_argument("--workflow-config", required=True, help="workflow YAML config")
    parser.add_argument(
        "--seed-schedule",
        default=None,
        help="optional existing schedule CSV used to infer source_app/source_func/source_label",
    )
    parser.add_argument("--source-app", default=None)
    parser.add_argument("--source-func", default=None)
    parser.add_argument("--source-label", default=None)
    parser.add_argument("--base-entry-ts-ms", type=int, default=1_900_000_000_000)
    parser.add_argument(
        "--time-compression",
        type=float,
        default=1.0,
        help=(
            "compress Azure inter-arrival offsets by this factor; "
            "use 30 to map one minute of source time to two seconds"
        ),
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--out-schedule", required=True)
    parser.add_argument("--out-trace", required=True)
    parser.add_argument("--metadata-out", required=True)
    return parser.parse_args()


def infer_source(args: argparse.Namespace) -> tuple[str, str, str]:
    source_app = args.source_app
    source_func = args.source_func
    source_label = args.source_label
    if args.seed_schedule:
        seed = pd.read_csv(args.seed_schedule)
        if seed.empty:
            raise ValueError("--seed-schedule is empty")
        first = seed.iloc[0]
        source_app = source_app or str(first["source_app"])
        source_func = source_func or str(first["source_func"])
        source_label = source_label or str(first.get("source_label", "azure"))

    if not source_app or not source_func:
        raise ValueError("source app/function must be provided or inferable from --seed-schedule")
    return source_app, source_func, source_label or "azure"


def extract_source_rows(
    azure_trace: Path,
    source_app: str,
    source_func: str,
    chunksize: int,
) -> pd.DataFrame:
    frames = []
    for chunk in pd.read_csv(
        azure_trace,
        chunksize=chunksize,
        usecols=["app", "func", "end_timestamp", "duration"],
    ):
        matched = chunk[(chunk["app"] == source_app) & (chunk["func"] == source_func)].copy()
        if matched.empty:
            continue
        matched["source_start_s"] = (
            matched["end_timestamp"].astype(float) - matched["duration"].astype(float)
        )
        matched["source_end_s"] = matched["end_timestamp"].astype(float)
        matched["source_duration_ms"] = (matched["duration"].astype(float) * 1000.0).round().astype(int)
        frames.append(
            matched[
                [
                    "source_start_s",
                    "source_end_s",
                    "source_duration_ms",
                ]
            ]
        )

    if not frames:
        raise ValueError("no rows found for the selected Azure app/function")

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["source_start_s", "source_end_s", "source_duration_ms"])
        .reset_index(drop=True)
    )


def build_schedule(
    rows: pd.DataFrame,
    workflow_name: str,
    source_label: str,
    source_app: str,
    source_func: str,
    time_compression: float,
) -> pd.DataFrame:
    if time_compression <= 0:
        raise ValueError("--time-compression must be positive")
    first_start_s = float(rows["source_start_s"].iloc[0])
    schedule = rows.copy()
    schedule.insert(0, "workflow_name", workflow_name)
    schedule.insert(1, "index", range(len(schedule)))
    schedule.insert(
        2,
        "target_offset_ms",
        (((schedule["source_start_s"] - first_start_s) * 1000.0) / time_compression)
        .round()
        .astype(int),
    )
    schedule.insert(3, "source_label", source_label)
    schedule.insert(4, "source_app", source_app)
    schedule.insert(5, "source_func", source_func)
    return schedule[
        [
            "workflow_name",
            "index",
            "target_offset_ms",
            "source_label",
            "source_app",
            "source_func",
            "source_start_s",
            "source_end_s",
            "source_duration_ms",
        ]
    ]


def build_entry_trace(
    schedule: pd.DataFrame,
    workflow_name: str,
    base_entry_ts_ms: int,
) -> pd.DataFrame:
    rows = []
    for _, row in schedule.iterrows():
        entry_ts_ms = int(base_entry_ts_ms + int(row["target_offset_ms"]))
        request_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{workflow_name}:{int(row['index'])}:{entry_ts_ms}"))
        rows.append(
            {
                "workflow_name": workflow_name,
                "request_id": request_id,
                "stage_name": "__entry__",
                "parent_stages": "",
                "entry_ts_ms": entry_ts_ms,
                "dispatch_start_ms": entry_ts_ms,
                "dispatch_end_ms": entry_ts_ms,
                "dispatch_latency_ms": 0,
                "action_start_ns": "",
                "action_end_ns": "",
                "action_duration_ms": "",
                "platform_overhead_ms": "",
                "container_id": "",
                "cold_like": "",
                "status": "ok",
                "error": "",
            }
        )
    return pd.DataFrame(rows, columns=TRACE_COLUMNS)


def main() -> None:
    args = parse_args()
    workflow = load_workflow(args.workflow_config)
    source_app, source_func, source_label = infer_source(args)
    azure_trace = Path(args.azure_trace)

    rows = extract_source_rows(
        azure_trace=azure_trace,
        source_app=source_app,
        source_func=source_func,
        chunksize=args.chunksize,
    )
    schedule = build_schedule(
        rows=rows,
        workflow_name=workflow.workflow_name,
        source_label=source_label,
        source_app=source_app,
        source_func=source_func,
        time_compression=args.time_compression,
    )
    entry_trace = build_entry_trace(
        schedule=schedule,
        workflow_name=workflow.workflow_name,
        base_entry_ts_ms=args.base_entry_ts_ms,
    )

    out_schedule = Path(args.out_schedule)
    out_trace = Path(args.out_trace)
    metadata_out = Path(args.metadata_out)
    out_schedule.parent.mkdir(parents=True, exist_ok=True)
    out_trace.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(out_schedule, index=False)
    entry_trace.to_csv(out_trace, index=False)

    intervals_ms = schedule["target_offset_ms"].diff().dropna()
    metadata = {
        "workflow_name": workflow.workflow_name,
        "workflow_config": args.workflow_config,
        "azure_trace": str(azure_trace),
        "seed_schedule": args.seed_schedule,
        "source_label": source_label,
        "source_app": source_app,
        "source_func": source_func,
        "rows": int(len(schedule)),
        "base_entry_ts_ms": int(args.base_entry_ts_ms),
        "time_compression": float(args.time_compression),
        "first_source_start_s": float(schedule["source_start_s"].min()),
        "last_source_start_s": float(schedule["source_start_s"].max()),
        "span_s": float(schedule["source_start_s"].max() - schedule["source_start_s"].min()),
        "scaled_span_s": float(
            (schedule["target_offset_ms"].max() - schedule["target_offset_ms"].min()) / 1000.0
        ),
        "interval_ms": {
            "count": int(len(intervals_ms)),
            "mean": float(intervals_ms.mean()) if len(intervals_ms) else 0.0,
            "median": float(intervals_ms.median()) if len(intervals_ms) else 0.0,
            "p90": float(intervals_ms.quantile(0.90)) if len(intervals_ms) else 0.0,
            "p95": float(intervals_ms.quantile(0.95)) if len(intervals_ms) else 0.0,
            "max": float(intervals_ms.max()) if len(intervals_ms) else 0.0,
        },
        "notes": [
            "This is an entry-only trace for forecasting experiments.",
            "No DAG stage execution rows are generated in this file.",
            "Entry timestamps preserve the selected Azure app/function start-time offsets.",
        ],
    }
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote {out_schedule}")
    print(f"wrote {out_trace}")
    print(f"wrote {metadata_out}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

