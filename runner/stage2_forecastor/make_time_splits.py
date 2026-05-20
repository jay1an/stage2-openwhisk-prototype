import argparse
import math
from pathlib import Path

import pandas as pd

from ..workflow import load_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a time-based train/test split for a workflow trace."
    )
    parser.add_argument("--trace", required=True, help="workflow trace CSV")
    parser.add_argument("--workflow-config", required=True, help="workflow YAML config")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--out-dir", default="data/synthetic/time_splits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent.parent
    workflow = load_workflow(str(root / args.workflow_config))
    trace_path = root / args.trace
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

    start = int(entry_rows["entry_ts_ms"].min())
    end = int(entry_rows["entry_ts_ms"].max())
    cutoff = start + int(math.floor((end - start) * args.train_ratio))

    split_map = entry_rows.copy()
    split_map["split"] = "test"
    split_map.loc[split_map["entry_ts_ms"] <= cutoff, "split"] = "train"
    split_map["split_strategy"] = "time"
    split_map["split_cutoff_ms"] = cutoff
    train_ids = set(split_map[split_map["split"] == "train"]["request_id"])
    test_ids = set(split_map[split_map["split"] == "test"]["request_id"])
    if not train_ids or not test_ids:
        raise SystemExit(
            "time split produced an empty train or test set; adjust --train-ratio"
        )

    prefix = trace_path.stem
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / f"{prefix}_time_split.csv"
    train_path = out_dir / f"{prefix}_time_train.csv"
    test_path = out_dir / f"{prefix}_time_test.csv"

    split_map.to_csv(split_path, index=False)
    trace[trace["request_id"].isin(train_ids)].to_csv(train_path, index=False)
    trace[trace["request_id"].isin(test_ids)].to_csv(test_path, index=False)

    print(f"wrote {split_path}")
    print(f"wrote {train_path}")
    print(f"wrote {test_path}")
    print(
        split_map.assign(entry_sec=(split_map["entry_ts_ms"] // 1000).astype(int))
        .groupby("split")["entry_sec"]
        .agg(["min", "max", "count"])
        .to_string()
    )


if __name__ == "__main__":
    main()

