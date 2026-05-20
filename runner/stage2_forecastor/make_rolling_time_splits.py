import argparse
import json
import math
from pathlib import Path

import pandas as pd

from ..workflow import load_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create rolling-origin time splits for workflow entry traces. "
            "Each fold uses all arrivals up to a cutoff for training and the "
            "next fixed time interval for testing."
        )
    )
    parser.add_argument("--trace", required=True, help="workflow trace CSV")
    parser.add_argument("--workflow-config", required=True, help="workflow YAML config")
    parser.add_argument("--train-minutes", type=float, default=45.0)
    parser.add_argument("--test-minutes", type=float, default=15.0)
    parser.add_argument("--step-minutes", type=float, default=15.0)
    parser.add_argument("--min-train-entries", type=int, default=20)
    parser.add_argument("--min-test-entries", type=int, default=1)
    parser.add_argument("--max-folds", type=int, default=12)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def ms_from_minutes(value: float) -> int:
    if value <= 0:
        raise ValueError("duration must be positive")
    return int(round(value * 60_000))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent.parent
    trace_path = root / args.trace
    workflow_path = root / args.workflow_config
    workflow = load_workflow(str(workflow_path))
    trace = pd.read_csv(trace_path)

    entry_rows = (
        trace[
            (trace["workflow_name"] == workflow.workflow_name)
            & (trace["stage_name"] == "__entry__")
            & (trace["status"] == "ok")
        ][["request_id", "entry_ts_ms"]]
        .drop_duplicates()
        .sort_values(["entry_ts_ms", "request_id"])
        .reset_index(drop=True)
    )
    if entry_rows.empty:
        raise SystemExit(f"no entry rows found for workflow={workflow.workflow_name}")

    start_ms = int(entry_rows["entry_ts_ms"].min())
    end_ms = int(entry_rows["entry_ts_ms"].max())
    train_ms = ms_from_minutes(args.train_minutes)
    test_ms = ms_from_minutes(args.test_minutes)
    step_ms = ms_from_minutes(args.step_minutes)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = trace_path.stem

    fold_rows = []
    fold_id = 0
    cutoff = start_ms + train_ms
    while cutoff < end_ms and fold_id < args.max_folds:
        test_start = cutoff + 1
        test_end = min(end_ms, cutoff + test_ms)
        if test_end < test_start:
            break

        train_mask = entry_rows["entry_ts_ms"] <= cutoff
        test_mask = (entry_rows["entry_ts_ms"] >= test_start) & (entry_rows["entry_ts_ms"] <= test_end)
        train_count = int(train_mask.sum())
        test_count = int(test_mask.sum())
        if train_count >= args.min_train_entries and test_count >= args.min_test_entries:
            split_map = entry_rows.copy()
            split_map["split"] = "ignore"
            split_map.loc[train_mask, "split"] = "train"
            split_map.loc[test_mask, "split"] = "test"
            split_map["split_strategy"] = "rolling-time"
            split_map["fold_id"] = fold_id
            split_map["split_cutoff_ms"] = cutoff
            split_map["eval_start_ms"] = test_start
            split_map["eval_end_ms"] = test_end

            split_path = out_dir / f"{prefix}_rolling_fold{fold_id:02d}_split.csv"
            split_map.to_csv(split_path, index=False)
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "split_path": str(split_path),
                    "trace": str(trace_path),
                    "workflow_config": str(workflow_path),
                    "workflow_name": workflow.workflow_name,
                    "train_start_ms": start_ms,
                    "train_end_ms": cutoff,
                    "test_start_ms": test_start,
                    "test_end_ms": test_end,
                    "train_entries": train_count,
                    "test_entries": test_count,
                    "train_span_min": (cutoff - start_ms) / 60_000.0,
                    "test_span_min": (test_end - test_start + 1) / 60_000.0,
                }
            )
            fold_id += 1

        cutoff += step_ms

    if not fold_rows:
        raise SystemExit(
            "no rolling folds produced; reduce --min-train-entries/--min-test-entries "
            "or change train/test durations"
        )

    index = pd.DataFrame(fold_rows)
    index_path = out_dir / f"{prefix}_rolling_index.csv"
    metadata_path = out_dir / f"{prefix}_rolling_metadata.json"
    index.to_csv(index_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "trace": str(trace_path),
                "workflow_config": str(workflow_path),
                "workflow_name": workflow.workflow_name,
                "train_minutes": args.train_minutes,
                "test_minutes": args.test_minutes,
                "step_minutes": args.step_minutes,
                "min_train_entries": args.min_train_entries,
                "min_test_entries": args.min_test_entries,
                "max_folds": args.max_folds,
                "folds": len(index),
                "trace_start_ms": start_ms,
                "trace_end_ms": end_ms,
                "trace_span_min": (end_ms - start_ms) / 60_000.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {index_path}")
    print(f"wrote {metadata_path}")
    print(index[["fold_id", "train_span_min", "test_span_min", "train_entries", "test_entries"]].to_string(index=False))


if __name__ == "__main__":
    main()

