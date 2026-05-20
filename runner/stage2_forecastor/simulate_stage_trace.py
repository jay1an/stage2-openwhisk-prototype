import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .generate_synthetic_traces import (
    build_calibration,
    simulate_workflow_trace,
    summarize_trace,
    write_split_traces,
)
from ..workflow import load_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one calibrated synthetic stage-level workflow trace from "
            "an existing Azure-derived entry schedule."
        )
    )
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--out-trace", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-split-dir", required=True)
    parser.add_argument("--out-metadata", required=True)
    parser.add_argument("--base-entry-ts-ms", type=int, default=1_900_000_000_000)
    parser.add_argument("--keepalive-ms", type=int, default=60_000)
    parser.add_argument("--burnin-copies", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent.parent
    workflow_path = root / args.workflow_config
    schedule_path = root / args.schedule
    workflow = load_workflow(str(workflow_path))
    schedule = pd.read_csv(schedule_path).sort_values("index").reset_index(drop=True)

    calibration = build_calibration(root, keepalive_ms=args.keepalive_ms)
    rng = np.random.default_rng(args.seed)
    trace = simulate_workflow_trace(
        workflow=workflow,
        schedule=schedule,
        calibration=calibration,
        rng=rng,
        base_entry_ts_ms=args.base_entry_ts_ms,
        burnin_copies=args.burnin_copies,
    )

    out_trace = root / args.out_trace
    out_summary = root / args.out_summary
    out_split_dir = root / args.out_split_dir
    out_metadata = root / args.out_metadata
    out_trace.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_split_dir.mkdir(parents=True, exist_ok=True)
    out_metadata.parent.mkdir(parents=True, exist_ok=True)

    trace.to_csv(out_trace, index=False)
    summarize_trace(trace).to_csv(out_summary, index=False)
    split_info = write_split_traces(
        trace=trace,
        out_dir=out_split_dir,
        file_prefix=out_trace.stem,
        train_ratio=args.train_ratio,
    )

    metadata = {
        "workflow_name": workflow.workflow_name,
        "workflow_config": str(workflow_path),
        "schedule": str(schedule_path),
        "out_trace": str(out_trace),
        "out_summary": str(out_summary),
        "split_info": split_info,
        "base_entry_ts_ms": args.base_entry_ts_ms,
        "keepalive_ms": args.keepalive_ms,
        "burnin_copies": args.burnin_copies,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "rows": int(len(trace)),
        "entry_requests": int((trace["stage_name"] == "__entry__").sum()),
        "stage_rows": int((trace["stage_name"] != "__entry__").sum()),
        "notes": [
            "Entry arrival offsets are copied from the Azure-derived schedule.",
            "Stage execution is simulated with SeBS DAG dependencies and OpenWhisk pilot calibration.",
            "This is offline method-development data, not a replacement for real replay traces.",
        ],
    }
    out_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote {out_trace}")
    print(f"wrote {out_summary}")
    print(f"wrote {out_metadata}")
    print(pd.DataFrame([metadata]).drop(columns=["notes", "split_info"]).to_string(index=False))
    print("split_info:")
    print(json.dumps(split_info, indent=2))


if __name__ == "__main__":
    main()

