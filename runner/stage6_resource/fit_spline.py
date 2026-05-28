from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from runner.stage6_resource.fit_amdahl_model_extended import (
    DEFAULT_SWEEP,
    prepare_trace,
    project_root,
    resolve_path,
)


DEFAULT_OUT_DIR = "reports/stage6_resource_models_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit D3 natural cubic spline resource scaling models by stage."
    )
    parser.add_argument("--sweep-csv", "--trace", dest="trace", default=DEFAULT_SWEEP)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def fit_quality(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    rel = (predicted - actual) / np.maximum(actual, 1e-9)
    rms = float(np.sqrt(np.mean(np.square(rel))) * 100.0)
    max_err = float(np.max(np.abs(rel)) * 100.0)
    return rms, max_err


def warm_tier_means(trace: pd.DataFrame) -> pd.DataFrame:
    warm = trace[trace["is_warm"]].copy()
    means = (
        warm.groupby(["stage_name", "allocated_memory_mb", "allocated_cpu_cores"], as_index=False)
        .agg(
            warm_action_mean_ms=("action_duration_ms", "mean"),
            warm_action_std_ms=("action_duration_ms", "std"),
            sample_count=("action_duration_ms", "size"),
        )
        .rename(
            columns={
                "allocated_memory_mb": "tier_mb",
                "allocated_cpu_cores": "cpu_cores",
            }
        )
        .sort_values(["stage_name", "cpu_cores"])
    )
    return means


def fit_splines(means: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coeff_rows = []
    prediction_rows = []
    quality_rows = []
    for stage, stage_means in means.groupby("stage_name", sort=True):
        stage_means = stage_means.sort_values("cpu_cores")
        cpu = stage_means["cpu_cores"].to_numpy(dtype=float)
        y = stage_means["warm_action_mean_ms"].to_numpy(dtype=float)
        spline = CubicSpline(cpu, y, bc_type="natural")
        predicted = spline(cpu)
        rms, max_err = fit_quality(y, predicted)

        row: dict[str, float | str | bool] = {
            "stage_name": stage,
            "bc_type": "natural",
            "n_knots": len(cpu),
            "rms_error_pct": rms,
            "max_error_pct": max_err,
            "pass_3pct": rms < 3.0,
            "pass_8pct_max": max_err < 8.0,
        }
        for idx, value in enumerate(cpu):
            row[f"knot_cpu_{idx}"] = value
        for idx, value in enumerate(y):
            row[f"knot_value_{idx}"] = value
        for seg_idx in range(spline.c.shape[1]):
            for order_idx in range(spline.c.shape[0]):
                row[f"coeff_order{order_idx}_segment{seg_idx}"] = spline.c[order_idx, seg_idx]
        coeff_rows.append(row)

        quality_rows.append(
            {
                "stage_name": stage,
                "rms_error_pct": rms,
                "max_error_pct": max_err,
                "pass_3pct": rms < 3.0,
                "pass_8pct_max": max_err < 8.0,
                "note": "Exact at sweep knots; use only for interpolation within [0.4, 3.0] vCPU.",
            }
        )
        for (_, source), pred in zip(stage_means.iterrows(), predicted):
            prediction_rows.append(
                {
                    "stage_name": stage,
                    "tier_mb": source["tier_mb"],
                    "cpu_cores": source["cpu_cores"],
                    "actual_warm_action_ms": source["warm_action_mean_ms"],
                    "predicted_warm_action_ms": pred,
                    "relative_error_pct": (
                        (pred - source["warm_action_mean_ms"])
                        / max(source["warm_action_mean_ms"], 1e-9)
                        * 100.0
                    ),
                }
            )
    return (
        pd.DataFrame(coeff_rows).sort_values("stage_name"),
        pd.DataFrame(prediction_rows).sort_values(["stage_name", "cpu_cores"]),
        pd.DataFrame(quality_rows).sort_values("stage_name"),
    )


def main() -> None:
    args = parse_args()
    root = project_root()
    trace_path = resolve_path(root, args.trace)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = prepare_trace(trace_path)
    means = warm_tier_means(trace)
    coeffs, predictions, quality = fit_splines(means)
    coeffs.to_csv(out_dir / "d3_spline_coeffs.csv", index=False)
    predictions.to_csv(out_dir / "d3_spline_predictions.csv", index=False)
    quality.to_csv(out_dir / "d3_spline_fit_quality.csv", index=False)

    print("D3 spline fit quality:")
    print(quality.to_string(index=False))
    print("\nD3 spline is a fallback; recommend using only if D1 and D2 fail.")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
