from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from runner.stage6_resource.fit_amdahl_model_extended import (
    DEFAULT_OUT_DIR,
    DEFAULT_SWEEP,
    DEFAULT_WORKFLOW,
    load_stage_specs,
    prepare_trace,
    project_root,
    resolve_path,
    worker_breakpoints,
)


SEGMENTS = [
    ("le_1cpu", 0.0, 1.0),
    ("gt_1_le_2cpu", 1.0, 2.0),
    ("gt_2cpu", 2.0, np.inf),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit per-stage piecewise Amdahl models with CPU breakpoints at 1 and 2."
    )
    parser.add_argument("--sweep-csv", "--trace", dest="trace", default=DEFAULT_SWEEP)
    parser.add_argument("--workflow-config", default=DEFAULT_WORKFLOW)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def segment_mask(cpu: pd.Series, low: float, high: float) -> pd.Series:
    if low == 0.0:
        return cpu <= high
    if np.isinf(high):
        return cpu > low
    return (cpu > low) & (cpu <= high)


def segment_prediction(segment_name: str, params: np.ndarray, cpu: np.ndarray) -> np.ndarray:
    s, p, c = params
    if segment_name == "le_1cpu":
        return (s + p) / cpu + c
    return s + p / cpu + c


def fit_segment(segment_name: str, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cpu = data["allocated_cpu_cores"].to_numpy(dtype=float)
    y = data["action_duration_ms"].to_numpy(dtype=float)
    initial = np.array([max(y.mean() * 0.2, 1.0), max(y.mean() * cpu.mean() * 0.5, 1.0), max(y.min() * 0.2, 1.0)])

    def objective(params: np.ndarray) -> np.ndarray:
        return segment_prediction(segment_name, params, cpu) - y

    result = least_squares(objective, initial, bounds=(0.0, np.inf), max_nfev=50_000)
    return result.x, segment_prediction(segment_name, result.x, cpu)


def fit_quality(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    rel = (predicted - actual) / np.maximum(actual, 1e-9)
    return float(np.max(np.abs(rel)) * 100.0), float(np.sqrt(np.mean(rel**2)) * 100.0)


def plot_piecewise(out_path: Path, warm_rows: pd.DataFrame, means: pd.DataFrame, params: pd.DataFrame) -> None:
    stages = sorted(means["stage_name"].unique())
    fig, axes = plt.subplots(len(stages), 1, figsize=(8, 3.2 * len(stages)), sharex=True)
    if len(stages) == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        stage_rows = warm_rows[warm_rows["stage_name"] == stage]
        stage_means = means[means["stage_name"] == stage].sort_values("allocated_cpu_cores")
        ax.scatter(
            stage_rows["allocated_cpu_cores"],
            stage_rows["action_duration_ms"],
            alpha=0.45,
            label="actual warm samples",
        )
        ax.scatter(
            stage_means["allocated_cpu_cores"],
            stage_means["action_duration_ms"],
            color="black",
            s=35,
            label="tier means",
            zorder=3,
        )
        for segment_name, low, high in SEGMENTS:
            rows = params[(params["stage_name"] == stage) & (params["segment"] == segment_name)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            if np.isinf(high):
                cpu_line = np.linspace(max(low + 1e-6, stage_means["allocated_cpu_cores"].min()), stage_means["allocated_cpu_cores"].max(), 80)
            else:
                cpu_line = np.linspace(max(low + 1e-6, stage_means["allocated_cpu_cores"].min()), min(high, stage_means["allocated_cpu_cores"].max()), 80)
            cpu_line = cpu_line[(cpu_line >= stage_means["allocated_cpu_cores"].min()) & (cpu_line <= stage_means["allocated_cpu_cores"].max())]
            if len(cpu_line) == 0:
                continue
            y_line = segment_prediction(
                segment_name,
                np.array([row["S_ms"], row["P_ms"], row["C_ms"]], dtype=float),
                cpu_line,
            )
            ax.plot(cpu_line, y_line, label=segment_name)
        ax.axvline(1.0, color="gray", linestyle=":", linewidth=1)
        ax.axvline(2.0, color="gray", linestyle=":", linewidth=1)
        ax.set_title(stage)
        ax.set_ylabel("action_duration_ms")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("cpu_cores")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = project_root()
    trace_path = resolve_path(root, args.trace)
    workflow_path = resolve_path(root, args.workflow_config)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = load_stage_specs(workflow_path)
    trace = prepare_trace(trace_path)
    warm = trace[trace["is_warm"]].copy()

    param_rows = []
    residual_rows = []
    mean_rows = []
    segment_quality_rows = []

    for stage in sorted(specs):
        stage_warm = warm[warm["stage_name"] == stage].copy()
        workers = worker_breakpoints(stage_warm, specs[stage])
        means = (
            stage_warm.groupby(["allocated_memory_mb", "allocated_cpu_cores"], as_index=False)
            .agg(
                action_duration_ms=("action_duration_ms", "mean"),
                action_duration_std_ms=("action_duration_ms", "std"),
                samples=("action_duration_ms", "size"),
            )
            .merge(workers, on="allocated_cpu_cores", how="left")
            .sort_values("allocated_cpu_cores")
        )
        means["stage_name"] = stage
        mean_rows.extend(means.to_dict(orient="records"))

        for segment_name, low, high in SEGMENTS:
            segment_data = means[segment_mask(means["allocated_cpu_cores"], low, high)].copy()
            if segment_data.empty:
                continue
            params, predicted = fit_segment(segment_name, segment_data)
            actual = segment_data["action_duration_ms"].to_numpy(dtype=float)
            max_pct, rms_pct = fit_quality(actual, predicted)
            param_rows.append(
                {
                    "stage_name": stage,
                    "segment": segment_name,
                    "cpu_low_exclusive": low,
                    "cpu_high_inclusive": high,
                    "S_ms": params[0],
                    "P_ms": params[1],
                    "C_ms": params[2],
                    "n_tiers": len(segment_data),
                    "fit_residual_max_pct": max_pct,
                    "fit_residual_rms_pct": rms_pct,
                }
            )
            segment_quality_rows.append(
                {
                    "stage_name": stage,
                    "segment": segment_name,
                    "n_tiers": len(segment_data),
                    "rms_error_pct": rms_pct,
                    "max_error_pct": max_pct,
                }
            )
            segment_residuals = segment_data[
                [
                    "stage_name",
                    "allocated_memory_mb",
                    "allocated_cpu_cores",
                    "action_duration_ms",
                    "samples",
                    "parallel_denom",
                    "worker_cap",
                ]
            ].copy()
            segment_residuals["segment"] = segment_name
            segment_residuals["predicted_action_duration_ms"] = predicted
            segment_residuals["abs_error_ms"] = (
                segment_residuals["predicted_action_duration_ms"]
                - segment_residuals["action_duration_ms"]
            )
            segment_residuals["relative_error_pct"] = (
                segment_residuals["abs_error_ms"]
                / segment_residuals["action_duration_ms"].clip(lower=1e-9)
                * 100.0
            )
            residual_rows.extend(segment_residuals.to_dict(orient="records"))

    params = pd.DataFrame(param_rows).sort_values(["stage_name", "cpu_low_exclusive"])
    residuals = pd.DataFrame(residual_rows).sort_values(
        ["stage_name", "allocated_cpu_cores"]
    )
    means = pd.DataFrame(mean_rows).sort_values(["stage_name", "allocated_cpu_cores"])
    segment_quality = pd.DataFrame(segment_quality_rows).sort_values(
        ["stage_name", "segment"]
    )
    overall_quality = (
        residuals.groupby("stage_name", as_index=False)
        .agg(
            n_tiers=("allocated_cpu_cores", "size"),
            rms_error_pct=("relative_error_pct", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            max_error_pct=("relative_error_pct", lambda s: float(np.max(np.abs(s)))),
        )
        .sort_values("stage_name")
    )
    overall_quality["pass_3pct"] = overall_quality["rms_error_pct"] < 3.0

    params.to_csv(out_dir / "per_stage_piecewise_params.csv", index=False)
    segment_quality.to_csv(out_dir / "piecewise_segment_fit_quality.csv", index=False)
    overall_quality.to_csv(out_dir / "piecewise_fit_validation.csv", index=False)
    residuals.to_csv(out_dir / "piecewise_fit_residuals_by_tier.csv", index=False)
    means.to_csv(out_dir / "piecewise_warm_tier_means.csv", index=False)
    plot_piecewise(out_dir / "piecewise_predictions_vs_actual.png", warm, means, params)

    print("Piecewise parameters:")
    print(params.to_string(index=False))
    print("\nPiecewise per-segment quality:")
    print(segment_quality.to_string(index=False))
    print("\nPiecewise overall validation:")
    print(overall_quality.to_string(index=False))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
