import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TARGETS = {
    "p50": 0.95,
    "p90": 0.98,
    "p95": 0.99,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a forecast method/policy from validation summaries under "
            "coverage and allocation trade-offs."
        )
    )
    parser.add_argument("--summary", required=True, help="rolling summary CSV")
    parser.add_argument("--characterization", default=None, help="optional trace characterization CSV")
    parser.add_argument("--trace-type", default=None, help="optional trace_type to read from characterization")
    parser.add_argument(
        "--targets",
        default="p90=0.98,p95=0.99",
        help="comma-separated coverage targets, e.g. p90=0.98,p95=0.99",
    )
    parser.add_argument("--under-penalty", type=float, default=10.0)
    parser.add_argument("--over-penalty", type=float, default=1.0)
    parser.add_argument(
        "--replica-second-penalty",
        type=float,
        default=0.0,
        help="optional small penalty per allocated replica-second",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def parse_targets(value: str) -> dict[str, float]:
    if not value:
        return DEFAULT_TARGETS.copy()
    targets: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"target must be policy=value, got {item!r}")
        policy, raw = item.split("=", 1)
        policy = policy.strip()
        if policy not in DEFAULT_TARGETS:
            raise ValueError(f"unsupported policy {policy!r}; expected one of {sorted(DEFAULT_TARGETS)}")
        targets[policy] = float(raw)
    return targets


def infer_regime_from_characterization(row: pd.Series | None) -> tuple[str, dict[str, float]]:
    if row is None:
        return "unknown", {}

    active_ratio = float(row.get("scaled_5s_active_ratio", np.nan))
    cv = float(row.get("scaled_5s_cv", np.nan))
    p95 = float(row.get("scaled_5s_p95", np.nan))
    max_count = float(row.get("scaled_5s_max", np.nan))
    mean_all = float(row.get("scaled_5s_mean_all", np.nan))

    features = {
        "active_ratio": active_ratio,
        "cv": cv,
        "p95_count": p95,
        "max_count": max_count,
        "mean_all": mean_all,
    }
    if active_ratio >= 0.7 and cv < 1.0:
        return "continuous_moderate", features
    if active_ratio < 0.02:
        return "sparse", features
    if active_ratio < 0.10 and cv >= 5.0 and max_count >= max(50.0, 10.0 * max(mean_all, 1.0)):
        return "bursty", features
    if active_ratio < 0.20 and cv >= 3.0:
        return "mixed_or_drift", features
    return "general", features


def load_characterization(path: str | None, trace_type: str | None) -> tuple[str, dict[str, float]]:
    if not path:
        return "unknown", {}
    frame = pd.read_csv(path)
    if frame.empty:
        return "unknown", {}
    if trace_type and "trace_type" in frame.columns:
        match = frame[frame["trace_type"] == trace_type]
        row = match.iloc[0] if not match.empty else frame.iloc[0]
    else:
        row = frame.iloc[0]
    if trace_type and "trace_type" in row:
        # Preserve the user-facing trace label, while still recording derived features.
        _, features = infer_regime_from_characterization(row)
        return str(row["trace_type"]), features
    return infer_regime_from_characterization(row)


def candidate_score(group: pd.DataFrame, target: float, args: argparse.Namespace) -> pd.DataFrame:
    out = group.copy()
    out["meets_target"] = out["demand_coverage_rate"] >= target
    out["shortfall"] = np.maximum(0.0, target - out["demand_coverage_rate"].astype(float))
    out["selection_cost"] = (
        args.under_penalty * out["under_total"].astype(float)
        + args.over_penalty * out["over_total"].astype(float)
        + args.replica_second_penalty * out["allocated_replica_seconds"].astype(float)
    )
    return out


def select_for_policy(group: pd.DataFrame, target: float, args: argparse.Namespace) -> pd.Series:
    scored = candidate_score(group, target, args)
    feasible = scored[scored["meets_target"]].copy()
    if not feasible.empty:
        feasible = feasible.sort_values(
            [
                "allocated_replica_seconds",
                "over_allocation_ratio",
                "selection_cost",
                "under_total",
            ],
            ascending=[True, True, True, True],
        )
        chosen = feasible.iloc[0].copy()
        chosen["selection_reason"] = "meet-target-min-replica-seconds"
        return chosen

    fallback = scored.sort_values(
        [
            "shortfall",
            "selection_cost",
            "allocated_replica_seconds",
            "over_allocation_ratio",
        ],
        ascending=[True, True, True, True],
    ).iloc[0].copy()
    fallback["selection_reason"] = "fallback-min-shortfall-cost"
    return fallback


def build_pareto(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in summary.groupby("policy"):
        ordered = group.sort_values(
            ["allocated_replica_seconds", "over_allocation_ratio", "demand_coverage_rate"],
            ascending=[True, True, False],
        )
        best_coverage = -np.inf
        best_over = np.inf
        for _, row in ordered.iterrows():
            coverage = float(row["demand_coverage_rate"])
            over = float(row["over_allocation_ratio"])
            if coverage > best_coverage or over < best_over:
                rows.append(row.to_dict())
                best_coverage = max(best_coverage, coverage)
                best_over = min(best_over, over)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    targets = parse_targets(args.targets)
    summary = pd.read_csv(args.summary)
    required = {
        "workflow_name",
        "method_family",
        "method",
        "policy",
        "demand_coverage_rate",
        "allocated_replica_seconds",
        "over_allocation_ratio",
        "under_total",
        "over_total",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary is missing required columns: {missing}")

    if "count_calibration" not in summary.columns:
        summary["count_calibration"] = "none"

    regime, regime_features = load_characterization(args.characterization, args.trace_type)
    selected_rows = []
    candidate_rows = []
    for policy, target in targets.items():
        group = summary[summary["policy"] == policy].copy()
        if group.empty:
            continue
        scored = candidate_score(group, target, args)
        scored["target_coverage"] = target
        candidate_rows.append(scored)
        chosen = select_for_policy(group, target, args)
        chosen["target_coverage"] = target
        chosen["regime"] = regime
        selected_rows.append(chosen)

    selection = pd.DataFrame(selected_rows)
    candidates = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    pareto = build_pareto(summary)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_path = out_dir / "selection.csv"
    candidates_path = out_dir / "candidates_scored.csv"
    pareto_path = out_dir / "pareto_front.csv"
    metadata_path = out_dir / "metadata.json"

    selection.to_csv(selection_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    pareto.to_csv(pareto_path, index=False)
    metadata = {
        "summary": args.summary,
        "characterization": args.characterization,
        "trace_type": args.trace_type,
        "derived_regime": regime,
        "regime_features": regime_features,
        "targets": targets,
        "under_penalty": args.under_penalty,
        "over_penalty": args.over_penalty,
        "replica_second_penalty": args.replica_second_penalty,
        "selection_rule": (
            "For each policy, choose the feasible candidate with minimum "
            "allocated_replica_seconds, then lower over_allocation_ratio. "
            "If none meets target, choose minimum target shortfall and cost."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote {selection_path}")
    print(f"wrote {candidates_path}")
    print(f"wrote {pareto_path}")
    print(f"wrote {metadata_path}")
    if not selection.empty:
        cols = [
            "regime",
            "policy",
            "target_coverage",
            "method_family",
            "count_calibration",
            "method",
            "demand_coverage_rate",
            "allocated_replica_seconds",
            "over_allocation_ratio",
            "allocation_utilization",
            "under_total",
            "over_total",
            "selection_reason",
        ]
        print(selection[[col for col in cols if col in selection.columns]].to_string(index=False))


if __name__ == "__main__":
    main()

