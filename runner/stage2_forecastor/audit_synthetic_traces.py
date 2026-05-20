import argparse
from pathlib import Path

import pandas as pd

from ..workflow import load_workflow


DEFAULT_CONFIG_MAP = {
    "sebs_trip_booking": "configs/sebs_trip_booking.yaml",
    "sebs_video": "configs/sebs_video.yaml",
    "sebs_map_reduce": "configs/sebs_map_reduce.yaml",
    "sebs_ml": "configs/sebs_ml.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit calibrated synthetic traces for structural and timing consistency."
    )
    parser.add_argument(
        "--trace-dir",
        default="data/synthetic/traces",
        help="directory containing *_synthetic.csv files",
    )
    parser.add_argument(
        "--split-dir",
        default="data/synthetic/splits",
        help="directory containing *_train.csv, *_test.csv, and *_split.csv files",
    )
    return parser.parse_args()


def count_parent_violations(stage_rows: pd.DataFrame) -> int:
    violations = 0
    for _, group in stage_rows.groupby("request_id"):
        end_by_stage = dict(zip(group["stage_name"], group["dispatch_end_ms"]))
        for _, row in group.iterrows():
            parents = [item for item in str(row["parent_stages"]).split(",") if item and item != "nan"]
            if parents and row["dispatch_start_ms"] < max(end_by_stage[parent] for parent in parents):
                violations += 1
    return violations


def count_container_overlaps(stage_rows: pd.DataFrame) -> int:
    overlaps = 0
    for _, group in stage_rows.groupby("container_id"):
        group = group.sort_values(["dispatch_start_ms", "dispatch_end_ms"])
        prev_end = None
        for _, row in group.iterrows():
            if prev_end is not None and row["dispatch_start_ms"] < prev_end:
                overlaps += 1
            prev_end = row["dispatch_end_ms"]
    return overlaps


def audit_trace(trace_path: Path, root: Path) -> dict:
    trace = pd.read_csv(trace_path)
    workflow_name = str(trace["workflow_name"].iloc[0])
    workflow = load_workflow(str(root / DEFAULT_CONFIG_MAP[workflow_name]))
    expected_nodes = set(workflow.nodes.keys())

    entry_rows = trace[trace["stage_name"] == "__entry__"].copy()
    stage_rows = trace[trace["stage_name"] != "__entry__"].copy()

    stage_count_mismatch = 0
    stage_set_mismatch = 0
    for _, group in stage_rows.groupby("request_id"):
        if len(group) != len(expected_nodes):
            stage_count_mismatch += 1
        if set(group["stage_name"]) != expected_nodes:
            stage_set_mismatch += 1

    dispatch_identity_mismatch = int(
        (stage_rows["dispatch_end_ms"] - stage_rows["dispatch_start_ms"] != stage_rows["dispatch_latency_ms"]).sum()
    )
    overhead_identity_mismatch = int(
        (
            (
                stage_rows["dispatch_latency_ms"]
                - stage_rows["action_duration_ms"]
                - stage_rows["platform_overhead_ms"]
            ).abs()
            > 1.1
        ).sum()
    )
    action_ns_identity_mismatch = int(
        (
            (
                (stage_rows["action_end_ns"] - stage_rows["action_start_ns"]) / 1_000_000
                - stage_rows["action_duration_ms"]
            ).abs()
            > 0.01
        ).sum()
    )
    parent_violations = count_parent_violations(stage_rows)
    container_overlaps = count_container_overlaps(stage_rows)
    multiple_cold_marks = int((stage_rows.groupby("container_id")["cold_like"].sum() > 1).sum())
    empty_stage_container_id = int(
        stage_rows["container_id"].isna().sum()
        + (stage_rows["container_id"].astype(str).str.len() == 0).sum()
    )
    reused_container_count = int((stage_rows.groupby("container_id").size() > 1).sum())
    total_container_count = int(stage_rows["container_id"].nunique())
    cold_rate = float(stage_rows["cold_like"].astype(bool).mean())

    return {
        "trace_file": trace_path.name,
        "workflow_name": workflow_name,
        "saved_requests": int(entry_rows["request_id"].nunique()),
        "saved_stage_rows": int(len(stage_rows)),
        "cold_rate": cold_rate,
        "reused_container_count": reused_container_count,
        "total_container_count": total_container_count,
        "stage_count_mismatch_requests": stage_count_mismatch,
        "stage_set_mismatch_requests": stage_set_mismatch,
        "dispatch_identity_mismatch_rows": dispatch_identity_mismatch,
        "overhead_identity_mismatch_rows": overhead_identity_mismatch,
        "action_ns_identity_mismatch_rows": action_ns_identity_mismatch,
        "parent_order_violation_rows": parent_violations,
        "container_overlap_rows": container_overlaps,
        "containers_with_multiple_cold_marks": multiple_cold_marks,
        "empty_stage_container_id_rows": empty_stage_container_id,
        "warning_all_cold": bool(cold_rate == 1.0),
        "warning_all_warm": bool(cold_rate == 0.0),
    }


def audit_split(split_dir: Path, prefix: str) -> dict:
    split_map = pd.read_csv(split_dir / f"{prefix}_split.csv")
    train = pd.read_csv(split_dir / f"{prefix}_train.csv")
    test = pd.read_csv(split_dir / f"{prefix}_test.csv")
    train_ids = set(train["request_id"].unique())
    test_ids = set(test["request_id"].unique())
    manifest_ids = set(split_map["request_id"])
    return {
        "split_prefix": prefix,
        "train_requests": len(train_ids),
        "test_requests": len(test_ids),
        "split_overlap_requests": len(train_ids & test_ids),
        "split_missing_requests": len(manifest_ids - (train_ids | test_ids)),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent.parent
    trace_dir = root / args.trace_dir
    split_dir = root / args.split_dir

    trace_results = [audit_trace(path, root) for path in sorted(trace_dir.glob("*_synthetic.csv"))]
    split_results = [
        audit_split(split_dir, path.name.replace(".csv", ""))
        for path in sorted(trace_dir.glob("*_synthetic.csv"))
    ]

    trace_df = pd.DataFrame(trace_results)
    split_df = pd.DataFrame(split_results)

    print("TRACE_AUDIT")
    print(trace_df.to_string(index=False))
    print()
    print("SPLIT_AUDIT")
    print(split_df.to_string(index=False))


if __name__ == "__main__":
    main()

