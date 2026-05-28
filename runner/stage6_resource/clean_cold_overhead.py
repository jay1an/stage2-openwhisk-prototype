from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from runner.stage6_resource.fit_amdahl_model_extended import (
    DEFAULT_SWEEP,
    prepare_trace,
    project_root,
    resolve_path,
)


DEFAULT_OUT_DIR = "reports/stage6_resource_models_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean per-tier cold overhead estimates from the 9-tier sweep."
    )
    parser.add_argument("--sweep-csv", "--trace", dest="trace", default=DEFAULT_SWEEP)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def cleanse_cold_overhead(trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    warm = trace[trace["is_warm"]].copy()
    cold = trace[trace["is_cold"]].copy()

    warm_dispatch = (
        warm.groupby(["stage_name", "allocated_memory_mb"], as_index=False)[
            "dispatch_latency_ms"
        ]
        .mean()
        .rename(columns={"dispatch_latency_ms": "warm_dispatch_mean_ms"})
    )
    cold_dispatch = (
        cold.groupby(["stage_name", "allocated_memory_mb"], as_index=False)[
            "dispatch_latency_ms"
        ]
        .mean()
        .rename(columns={"dispatch_latency_ms": "cold_dispatch_mean_ms"})
    )

    merged = cold_dispatch.merge(
        warm_dispatch, on=["stage_name", "allocated_memory_mb"], how="inner"
    )
    merged["original_cold_overhead_ms"] = (
        merged["cold_dispatch_mean_ms"] - merged["warm_dispatch_mean_ms"]
    )
    merged = merged.rename(columns={"allocated_memory_mb": "tier_mb"})

    cleansed_parts = []
    summary_rows = []
    for stage, stage_rows in merged.groupby("stage_name", sort=True):
        stage_rows = stage_rows.sort_values("tier_mb").copy()
        values = stage_rows["original_cold_overhead_ms"].to_numpy(dtype=float)
        mean_before = float(np.mean(values))
        std_before = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        if std_before > 0.0:
            is_outlier = np.abs(values - mean_before) > 2.0 * std_before
        else:
            is_outlier = np.zeros(len(values), dtype=bool)

        non_outlier_values = values[~is_outlier]
        replacement = (
            float(np.median(non_outlier_values))
            if len(non_outlier_values) > 0
            else float(np.median(values))
        )
        cleansed = values.copy()
        cleansed[is_outlier] = replacement
        std_after = float(np.std(cleansed, ddof=1)) if len(cleansed) > 1 else 0.0
        mean_after = float(np.mean(cleansed))

        stage_rows["is_outlier"] = is_outlier
        stage_rows["cleansed_cold_overhead_ms"] = cleansed
        cleansed_parts.append(
            stage_rows[
                [
                    "stage_name",
                    "tier_mb",
                    "original_cold_overhead_ms",
                    "is_outlier",
                    "cleansed_cold_overhead_ms",
                ]
            ]
        )
        replaced = [
            f"{int(tier)}:{original:.1f}->{clean:.1f}"
            for tier, original, clean, outlier in zip(
                stage_rows["tier_mb"], values, cleansed, is_outlier
            )
            if outlier
        ]
        summary_rows.append(
            {
                "stage_name": stage,
                "n_outliers_detected": int(is_outlier.sum()),
                "mean_before": mean_before,
                "mean_after": mean_after,
                "std_before": std_before,
                "std_after": std_after,
                "replaced_values": "; ".join(replaced),
            }
        )

    cleansed_frame = pd.concat(cleansed_parts, ignore_index=True).sort_values(
        ["stage_name", "tier_mb"]
    )
    summary = pd.DataFrame(summary_rows).sort_values("stage_name")
    return cleansed_frame, summary


def main() -> None:
    args = parse_args()
    root = project_root()
    trace_path = resolve_path(root, args.trace)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = prepare_trace(trace_path)
    cleansed, summary = cleanse_cold_overhead(trace)
    cleansed.to_csv(out_dir / "cold_overhead_cleansed.csv", index=False)
    summary.to_csv(out_dir / "cold_overhead_cleansing_summary.csv", index=False)

    print("Cold overhead cleansing summary:")
    print(summary.to_string(index=False))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
