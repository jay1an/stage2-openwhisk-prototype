import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .select_forecast_policy import candidate_score, load_characterization, parse_targets, select_for_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select forecast methods on validation folds and evaluate the selected "
            "methods on held-out folds."
        )
    )
    parser.add_argument("--detail", required=True, help="window-level detail CSV with fold_id")
    parser.add_argument("--characterization", default=None)
    parser.add_argument("--trace-type", default=None)
    parser.add_argument("--validation-folds", default="0,1,2")
    parser.add_argument("--test-folds", default="3")
    parser.add_argument("--targets", default="p90=0.98,p95=0.99")
    parser.add_argument(
        "--selection-mode",
        choices=["pooled", "fold-min"],
        default="fold-min",
        help="pooled selects from aggregate validation; fold-min requires robust per-fold coverage",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.0,
        help="extra validation coverage required above each target in fold-min mode",
    )
    parser.add_argument("--under-penalty", type=float, default=10.0)
    parser.add_argument("--over-penalty", type=float, default=1.0)
    parser.add_argument("--replica-second-penalty", type=float, default=0.0)
    parser.add_argument("--window-ms", type=int, default=5000)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def parse_fold_list(value: str) -> set[int]:
    folds = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        folds.add(int(item))
    if not folds:
        raise ValueError("fold list must not be empty")
    return folds


def summarize_detail(detail: pd.DataFrame, window_ms: int) -> pd.DataFrame:
    group_cols = ["workflow_name", "method_family", "count_calibration", "method", "policy"]
    rows = []
    for keys, group in detail.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        actual = group["actual_count"].astype(float)
        forecast = group["forecast_count"].astype(float)
        allocated = group["allocated_count"].astype(float)
        active = group[group["actual_count"] > 0]
        actual_total = float(actual.sum())
        allocated_total = float(allocated.sum())
        under_total = float(group["under_count"].sum())
        over_total = float(group["over_count"].sum())
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "folds": int(group["fold_id"].nunique()),
                "origins": int(group[["fold_id", "origin_window"]].drop_duplicates().shape[0]),
                "forecast_rows": int(len(group)),
                "active_rows": int(len(active)),
                "actual_total": int(actual_total),
                "allocated_replica_windows": int(allocated_total),
                "allocated_replica_seconds": float(allocated_total * window_ms / 1000.0),
                "under_total": int(under_total),
                "over_total": int(over_total),
                "coverage_rate": float((group["under_count"] == 0).mean()),
                "active_coverage_rate": float((active["under_count"] == 0).mean()) if len(active) else 1.0,
                "demand_coverage_rate": float(1.0 - under_total / actual_total) if actual_total > 0 else 1.0,
                "allocation_utilization": (
                    float((actual_total - under_total) / allocated_total)
                    if allocated_total > 0
                    else 0.0
                ),
                "over_allocation_ratio": (
                    float(over_total / allocated_total)
                    if allocated_total > 0
                    else 0.0
                ),
                "mae": float(np.mean(np.abs(actual - forecast))),
                "rmse": float(np.sqrt(np.mean((actual - forecast) ** 2))),
                "max_actual": int(actual.max()) if len(actual) else 0,
                "max_allocated": int(allocated.max()) if len(allocated) else 0,
            }
        )
    return pd.DataFrame(rows)


def add_method_family(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    if "count_calibration" not in out.columns:
        out["count_calibration"] = "none"
    if "method_family" not in out.columns:
        def family(method: str) -> str:
            if method.startswith("independent-"):
                return "per-stage-independent"
            if method.startswith("dag-twostage"):
                return "entry-two-stage-dag"
            if method.startswith("dag-"):
                return "entry-heuristic-dag"
            return "other"

        out["method_family"] = out["method"].map(family)
    return out


def select_for_policy_fold_min(
    validation_summary: pd.DataFrame,
    validation_fold_summary: pd.DataFrame,
    policy: str,
    target: float,
    args: argparse.Namespace,
) -> pd.Series:
    group = validation_summary[validation_summary["policy"] == policy].copy()
    fold_group = validation_fold_summary[validation_fold_summary["policy"] == policy].copy()
    if group.empty:
        raise ValueError(f"no validation rows for policy={policy}")

    key_cols = ["workflow_name", "method_family", "count_calibration", "method", "policy"]
    fold_stats = (
        fold_group.groupby(key_cols, as_index=False)
        .agg(
            validation_min_fold_coverage=("demand_coverage_rate", "min"),
            validation_mean_fold_coverage=("demand_coverage_rate", "mean"),
            validation_max_fold_shortfall=(
                "demand_coverage_rate",
                lambda values: float(np.maximum(0.0, target + args.safety_margin - values).max()),
            ),
            validation_mean_fold_seconds=("allocated_replica_seconds", "mean"),
            validation_mean_fold_over_allocation=("over_allocation_ratio", "mean"),
        )
    )
    scored = group.merge(fold_stats, on=key_cols, how="left")
    scored = candidate_score(scored, target, args)
    robust_target = target + args.safety_margin
    scored["robust_target_coverage"] = robust_target
    scored["meets_robust_fold_target"] = scored["validation_min_fold_coverage"] >= robust_target

    feasible = scored[scored["meets_robust_fold_target"]].copy()
    if not feasible.empty:
        feasible = feasible.sort_values(
            [
                "validation_mean_fold_seconds",
                "validation_mean_fold_over_allocation",
                "selection_cost",
                "under_total",
            ],
            ascending=[True, True, True, True],
        )
        chosen = feasible.iloc[0].copy()
        chosen["selection_reason"] = "fold-min-target-plus-margin-min-mean-seconds"
        return chosen

    fallback = scored.sort_values(
        [
            "validation_max_fold_shortfall",
            "selection_cost",
            "validation_mean_fold_seconds",
            "validation_mean_fold_over_allocation",
        ],
        ascending=[True, True, True, True],
    ).iloc[0].copy()
    fallback["selection_reason"] = "fallback-min-fold-shortfall-cost"
    return fallback


def main() -> None:
    args = parse_args()
    validation_folds = parse_fold_list(args.validation_folds)
    test_folds = parse_fold_list(args.test_folds)
    targets = parse_targets(args.targets)
    regime, regime_features = load_characterization(args.characterization, args.trace_type)

    detail = add_method_family(pd.read_csv(args.detail))
    if "fold_id" not in detail.columns:
        raise ValueError("--detail must contain fold_id")
    validation_detail = detail[detail["fold_id"].isin(validation_folds)].copy()
    test_detail = detail[detail["fold_id"].isin(test_folds)].copy()
    if validation_detail.empty:
        raise ValueError("validation folds produced no rows")
    if test_detail.empty:
        raise ValueError("test folds produced no rows")

    validation_summary = summarize_detail(validation_detail, args.window_ms)
    test_summary = summarize_detail(test_detail, args.window_ms)
    fold_frames = []
    for fold_id, group in validation_detail.groupby("fold_id"):
        fold_summary = summarize_detail(group.copy(), args.window_ms)
        fold_summary.insert(0, "fold_id", int(fold_id))
        fold_frames.append(fold_summary)
    validation_fold_summary = pd.concat(fold_frames, ignore_index=True)

    selected_rows = []
    test_rows = []
    for policy, target in targets.items():
        group = validation_summary[validation_summary["policy"] == policy].copy()
        if group.empty:
            continue
        if args.selection_mode == "fold-min":
            chosen = select_for_policy_fold_min(
                validation_summary,
                validation_fold_summary,
                policy,
                target,
                args,
            )
        else:
            chosen = select_for_policy(group, target, args)
        chosen["target_coverage"] = target
        chosen["regime"] = regime
        selected_rows.append(chosen)

        mask = (
            (test_summary["policy"] == policy)
            & (test_summary["method_family"] == chosen["method_family"])
            & (test_summary["count_calibration"] == chosen["count_calibration"])
            & (test_summary["method"] == chosen["method"])
        )
        heldout = test_summary[mask].copy()
        if not heldout.empty:
            row = heldout.iloc[0].copy()
            row["target_coverage"] = target
            row["regime"] = regime
            row["selected_on_validation_reason"] = chosen.get("selection_reason", "")
            row["validation_demand_coverage_rate"] = chosen["demand_coverage_rate"]
            row["validation_allocated_replica_seconds"] = chosen["allocated_replica_seconds"]
            row["validation_over_allocation_ratio"] = chosen["over_allocation_ratio"]
            row["meets_target_on_test"] = bool(row["demand_coverage_rate"] >= target)
            test_rows.append(row)

    selection = pd.DataFrame(selected_rows)
    heldout = pd.DataFrame(test_rows)
    scored_validation = []
    for policy, target in targets.items():
        group = validation_summary[validation_summary["policy"] == policy].copy()
        if not group.empty:
            scored = candidate_score(group, target, args)
            scored["target_coverage"] = target
            scored_validation.append(scored)
    scored_validation = pd.concat(scored_validation, ignore_index=True) if scored_validation else pd.DataFrame()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_path = out_dir / "validation_selection.csv"
    heldout_path = out_dir / "heldout_test_result.csv"
    validation_summary_path = out_dir / "validation_summary.csv"
    validation_fold_summary_path = out_dir / "validation_fold_summary.csv"
    test_summary_path = out_dir / "test_summary.csv"
    scored_path = out_dir / "validation_candidates_scored.csv"
    metadata_path = out_dir / "metadata.json"

    selection.to_csv(selection_path, index=False)
    heldout.to_csv(heldout_path, index=False)
    validation_summary.to_csv(validation_summary_path, index=False)
    validation_fold_summary.to_csv(validation_fold_summary_path, index=False)
    test_summary.to_csv(test_summary_path, index=False)
    scored_validation.to_csv(scored_path, index=False)
    metadata = {
        "detail": args.detail,
        "characterization": args.characterization,
        "trace_type": args.trace_type,
        "derived_regime": regime,
        "regime_features": regime_features,
        "validation_folds": sorted(validation_folds),
        "test_folds": sorted(test_folds),
        "targets": targets,
        "selection_mode": args.selection_mode,
        "safety_margin": args.safety_margin,
        "window_ms": args.window_ms,
        "under_penalty": args.under_penalty,
        "over_penalty": args.over_penalty,
        "replica_second_penalty": args.replica_second_penalty,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote {selection_path}")
    print(f"wrote {heldout_path}")
    print(f"wrote {validation_summary_path}")
    print(f"wrote {validation_fold_summary_path}")
    print(f"wrote {test_summary_path}")
    print(f"wrote {scored_path}")
    print(f"wrote {metadata_path}")
    if not heldout.empty:
        cols = [
            "regime",
            "policy",
            "target_coverage",
            "method_family",
            "count_calibration",
            "method",
            "validation_demand_coverage_rate",
            "demand_coverage_rate",
            "meets_target_on_test",
            "allocated_replica_seconds",
            "over_allocation_ratio",
            "allocation_utilization",
            "under_total",
            "over_total",
        ]
        print(heldout[[col for col in cols if col in heldout.columns]].to_string(index=False))


if __name__ == "__main__":
    main()

