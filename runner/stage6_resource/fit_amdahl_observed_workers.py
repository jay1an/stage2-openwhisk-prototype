from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from runner.stage6_resource.fit_amdahl_model_extended import (
    DEFAULT_SWEEP,
    prepare_trace,
    project_root,
    resolve_path,
)


DEFAULT_OUT_DIR = "reports/stage6_resource_models_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit D2 Amdahl model using observed workers per tier."
    )
    parser.add_argument("--sweep-csv", "--trace", dest="trace", default=DEFAULT_SWEEP)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def amdahl_observed(xdata: tuple[np.ndarray, np.ndarray], s: float, p: float, c: float) -> np.ndarray:
    cpu, workers = xdata
    serial_denom = np.minimum(cpu, 1.0)
    worker_denom = np.maximum(workers, 1e-9)
    return s / serial_denom + p / worker_denom + c


def fit_quality(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    rel = (predicted - actual) / np.maximum(actual, 1e-9)
    rms = float(np.sqrt(np.mean(np.square(rel))) * 100.0)
    max_err = float(np.max(np.abs(rel)) * 100.0)
    return rms, max_err


def observed_worker_means(trace: pd.DataFrame) -> pd.DataFrame:
    warm = trace[trace["is_warm"]].copy()
    means = (
        warm.groupby(["stage_name", "allocated_memory_mb", "allocated_cpu_cores"], as_index=False)
        .agg(
            warm_action_mean_ms=("action_duration_ms", "mean"),
            warm_action_std_ms=("action_duration_ms", "std"),
            parallel_workers_used=("parallel_workers_used", "mean"),
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


def fit_models(means: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    param_rows = []
    prediction_rows = []
    for stage, stage_means in means.groupby("stage_name", sort=True):
        stage_means = stage_means.sort_values("cpu_cores")
        cpu = stage_means["cpu_cores"].to_numpy(dtype=float)
        workers = stage_means["parallel_workers_used"].to_numpy(dtype=float)
        y = stage_means["warm_action_mean_ms"].to_numpy(dtype=float)
        params, _ = curve_fit(
            amdahl_observed,
            (cpu, workers),
            y,
            p0=[max(y.max() * 0.25, 1.0), max(y.max() * 0.5, 1.0), max(y.min() * 0.2, 1.0)],
            bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
            maxfev=100_000,
        )
        predicted = amdahl_observed((cpu, workers), *params)
        rms, max_err = fit_quality(y, predicted)
        param_rows.append(
            {
                "stage_name": stage,
                "S_ms": params[0],
                "P_ms": params[1],
                "C_ms": params[2],
                "rms_error_pct": rms,
                "max_error_pct": max_err,
                "pass_3pct": rms < 3.0,
                "pass_8pct_max": max_err < 8.0,
            }
        )
        for (_, row), pred in zip(stage_means.iterrows(), predicted):
            prediction_rows.append(
                {
                    "stage_name": stage,
                    "tier_mb": row["tier_mb"],
                    "cpu_cores": row["cpu_cores"],
                    "parallel_workers_used": row["parallel_workers_used"],
                    "actual_warm_action_ms": row["warm_action_mean_ms"],
                    "predicted_warm_action_ms": pred,
                    "relative_error_pct": (
                        (pred - row["warm_action_mean_ms"])
                        / max(row["warm_action_mean_ms"], 1e-9)
                        * 100.0
                    ),
                }
            )
    return (
        pd.DataFrame(param_rows).sort_values("stage_name"),
        pd.DataFrame(prediction_rows).sort_values(["stage_name", "cpu_cores"]),
    )


def main() -> None:
    args = parse_args()
    root = project_root()
    trace_path = resolve_path(root, args.trace)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = prepare_trace(trace_path)
    means = observed_worker_means(trace)
    params, predictions = fit_models(means)
    params.to_csv(out_dir / "d2_amdahl_observed_params.csv", index=False)
    predictions.to_csv(out_dir / "d2_amdahl_observed_predictions.csv", index=False)
    means[
        [
            "stage_name",
            "tier_mb",
            "cpu_cores",
            "parallel_workers_used",
            "sample_count",
        ]
    ].to_csv(out_dir / "d2_observed_workers_by_tier.csv", index=False)

    print("D2 observed-worker sanity table:")
    print(
        means[
            [
                "stage_name",
                "tier_mb",
                "cpu_cores",
                "parallel_workers_used",
                "sample_count",
            ]
        ].to_string(index=False)
    )
    print("\nD2 Amdahl observed-workers fit:")
    print(params.to_string(index=False))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
