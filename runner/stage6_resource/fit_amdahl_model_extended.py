from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares


DEFAULT_SWEEP = "reports/civic_memory_cpu_sweep_multinode_9tier/trace.csv"
DEFAULT_WORKFLOW = "configs/civic_alert_flow.yaml"
DEFAULT_OUT_DIR = "reports/stage6_amdahl_model_multinode_9tier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit per-stage Amdahl-style models on an extended memory/CPU sweep."
    )
    parser.add_argument("--sweep-csv", "--trace", dest="trace", default=DEFAULT_SWEEP)
    parser.add_argument("--workflow-config", default=DEFAULT_WORKFLOW)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def normalize_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def load_stage_specs(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    specs: dict[str, dict[str, float]] = {}
    for item in raw["nodes"]:
        cpu_iters = float(item.get("cpu_iters", 0.0) or 0.0)
        serial_fraction = float(item.get("serial_fraction", 0.25) or 0.25)
        specs[str(item["name"])] = {
            "cpu_iters": cpu_iters,
            "serial_fraction": serial_fraction,
            "io_wait_ms": float(item.get("io_wait_ms", item.get("sleep_ms", 0.0)) or 0.0),
            "max_parallel_workers": float(item.get("max_parallel_workers", 8) or 8),
        }
    return specs


def prepare_trace(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["stage_name"].astype(str) != "__entry__"].copy()
    if "status" in frame.columns:
        frame = frame[frame["status"].fillna("ok").astype(str) == "ok"].copy()

    numeric_columns = [
        "allocated_memory_mb",
        "allocated_cpu_cores",
        "dispatch_latency_ms",
        "action_duration_ms",
        "platform_overhead_ms",
        "parallel_workers_used",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")

    frame["is_warm"] = frame["cold_like"].map(normalize_bool) == False
    frame["is_cold"] = frame["cold_like"].map(normalize_bool) == True
    frame = frame.dropna(
        subset=["allocated_memory_mb", "allocated_cpu_cores", "action_duration_ms"]
    )
    return frame


def worker_breakpoints(warm_rows: pd.DataFrame, stage_spec: dict[str, float]) -> pd.DataFrame:
    grouped = (
        warm_rows.groupby("allocated_cpu_cores", as_index=False)["parallel_workers_used"]
        .mean()
        .sort_values("allocated_cpu_cores")
    )
    max_workers = float(stage_spec.get("max_parallel_workers", 8.0))
    grouped["worker_cap"] = grouped["parallel_workers_used"].round().clip(
        lower=1, upper=max_workers
    )
    grouped["parallel_denom"] = np.minimum(
        grouped["allocated_cpu_cores"], grouped["worker_cap"]
    )
    return grouped[["allocated_cpu_cores", "parallel_denom", "worker_cap"]]


def amdahl_prediction(params: np.ndarray, cpu: np.ndarray, parallel_denom: np.ndarray) -> np.ndarray:
    serial_denom = np.minimum(cpu, 1.0)
    return params[0] / serial_denom + params[1] / parallel_denom + params[2]


def powerlaw_prediction(params: np.ndarray, cpu: np.ndarray) -> np.ndarray:
    return params[0] * np.power(cpu, -params[1]) + params[2]


def residual_frame(
    stage_name: str,
    means: pd.DataFrame,
    model: str,
    predicted: np.ndarray,
) -> pd.DataFrame:
    out = means[
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
    out["model"] = model
    out["predicted_action_duration_ms"] = predicted
    out["abs_error_ms"] = out["predicted_action_duration_ms"] - out["action_duration_ms"]
    out["relative_error_pct"] = (
        out["abs_error_ms"] / out["action_duration_ms"].clip(lower=1e-9) * 100.0
    )
    out["stage_name"] = stage_name
    return out


def fit_quality(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    rel = (predicted - actual) / np.maximum(actual, 1e-9)
    return float(np.max(np.abs(rel)) * 100.0), float(np.sqrt(np.mean(rel**2)) * 100.0)


def fit_amdahl(means: pd.DataFrame, stage_spec: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    cpu = means["allocated_cpu_cores"].to_numpy(dtype=float)
    y = means["action_duration_ms"].to_numpy(dtype=float)
    parallel_denom = means["parallel_denom"].to_numpy(dtype=float)
    total_work = max(float(y.max() - stage_spec.get("io_wait_ms", 0.0)), 1.0)
    initial = np.array(
        [
            0.2 * total_work,
            0.8 * total_work,
            max(stage_spec.get("io_wait_ms", 0.0), 1.0),
        ]
    )

    def objective(params: np.ndarray) -> np.ndarray:
        return amdahl_prediction(params, cpu, parallel_denom) - y

    result = least_squares(objective, initial, bounds=(0.0, np.inf), max_nfev=50_000)
    return result.x, amdahl_prediction(result.x, cpu, parallel_denom)


def fit_powerlaw(means: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cpu = means["allocated_cpu_cores"].to_numpy(dtype=float)
    y = means["action_duration_ms"].to_numpy(dtype=float)
    initial = np.array([max(y.max() - y.min(), 1.0), 1.0, max(y.min() * 0.5, 1.0)])

    def objective(params: np.ndarray) -> np.ndarray:
        return powerlaw_prediction(params, cpu) - y

    result = least_squares(objective, initial, bounds=(0.0, np.inf), max_nfev=50_000)
    return result.x, powerlaw_prediction(result.x, cpu)


def plot_predictions(
    out_path: Path,
    warm_rows: pd.DataFrame,
    fit_means: pd.DataFrame,
    amdahl_params: pd.DataFrame,
    powerlaw_params: pd.DataFrame,
) -> None:
    stages = sorted(warm_rows["stage_name"].unique())
    fig, axes = plt.subplots(len(stages), 1, figsize=(8, 3.2 * len(stages)), sharex=True)
    if len(stages) == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        stage_rows = warm_rows[warm_rows["stage_name"] == stage]
        stage_means = fit_means[fit_means["stage_name"] == stage].sort_values(
            "allocated_cpu_cores"
        )
        arow = amdahl_params[amdahl_params["stage_name"] == stage].iloc[0]
        prow = powerlaw_params[powerlaw_params["stage_name"] == stage].iloc[0]
        cpu_line = np.linspace(
            stage_means["allocated_cpu_cores"].min(),
            stage_means["allocated_cpu_cores"].max(),
            200,
        )
        worker_cap = np.interp(
            cpu_line,
            stage_means["allocated_cpu_cores"].to_numpy(dtype=float),
            stage_means["worker_cap"].to_numpy(dtype=float),
        )
        parallel_denom = np.minimum(cpu_line, np.maximum(1.0, np.round(worker_cap)))
        amdahl_line = amdahl_prediction(
            np.array([arow["S_ms"], arow["P_ms"], arow["C_ms"]], dtype=float),
            cpu_line,
            parallel_denom,
        )
        power_line = powerlaw_prediction(
            np.array([prow["a"], prow["alpha"], prow["c"]], dtype=float),
            cpu_line,
        )
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
        ax.plot(cpu_line, amdahl_line, label="Amdahl fit")
        ax.plot(cpu_line, power_line, linestyle="--", label="Power-law fit")
        ax.set_title(stage)
        ax.set_ylabel("action_duration_ms")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("cpu_cores")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_readme(
    out_path: Path,
    amdahl_params: pd.DataFrame,
    powerlaw_params: pd.DataFrame,
    comparison: pd.DataFrame,
    warm_validation: pd.DataFrame,
    cold_validation: pd.DataFrame,
    specs: dict[str, dict[str, float]],
) -> None:
    amdahl_lines = []
    for row in amdahl_params.sort_values("stage_name").to_dict(orient="records"):
        spec = specs[row["stage_name"]]
        expected_serial = spec["cpu_iters"] * spec["serial_fraction"]
        expected_parallel = spec["cpu_iters"] * (1.0 - spec["serial_fraction"])
        amdahl_lines.append(
            "| {stage_name} | {S_ms:.1f} | {P_ms:.1f} | {C_ms:.1f} | {fit_residual_rms_pct:.3f}% | "
            "{fit_residual_max_pct:.3f}% | {cpu_iters:.0f} | {expected_serial:.0f} | {expected_parallel:.0f} |".format(
                **row,
                cpu_iters=spec["cpu_iters"],
                expected_serial=expected_serial,
                expected_parallel=expected_parallel,
            )
        )

    best = (
        comparison.sort_values(["stage_name", "rms_relative_error"])
        .groupby("stage_name", as_index=False)
        .first()
    )
    best_lines = [
        f"| {row.stage_name} | {row.model} | {row.rms_relative_error:.3f}% |"
        for _, row in best.sort_values("stage_name").iterrows()
    ]
    validation_lines = [
        f"| {row.stage_name} | {int(row.n_tiers)} | {row.rms_error_pct:.3f}% | {row.max_error_pct:.3f}% | {row.pass_3pct} |"
        for _, row in warm_validation.sort_values("stage_name").iterrows()
    ]
    cold_lines = [
        f"| {row.stage_name} | {row.cold_oh_mean_ms:.1f} | {row.cold_oh_std_ms:.1f} | {row.cv_pct:.2f}% | {row.pass_20pct} |"
        for _, row in cold_validation.sort_values("stage_name").iterrows()
    ]

    text = """# Extended 9-Tier Amdahl Fit

The fit uses warm action executions from the multi-node 9-tier sweep and fits
against the per-tier mean `action_duration_ms` for each stage.

## Amdahl Parameters

| stage | S_ms | P_ms | C_ms | RMS error | max error | cpu_iters | expected serial iters | expected parallel iters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{amdahl_rows}

## A1 Warm-Fit Validation

| stage | n_tiers | RMS error | max error | pass <3% |
| --- | ---: | ---: | ---: | --- |
{validation_rows}

## Model Comparison

| stage | best model | RMS relative error |
| --- | --- | ---: |
{best_rows}

## Cold Overhead Constancy

| stage | mean ms | std ms across tiers | CV | pass <20% |
| --- | ---: | ---: | ---: | --- |
{cold_rows}
""".format(
        amdahl_rows="\n".join(amdahl_lines),
        validation_rows="\n".join(validation_lines),
        best_rows="\n".join(best_lines),
        cold_rows="\n".join(cold_lines),
    )
    out_path.write_text(text, encoding="utf-8")


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
    cold = trace[trace["is_cold"]].copy()

    amdahl_rows = []
    power_rows = []
    comparison_rows = []
    mean_rows = []
    residual_rows = []

    for stage in sorted(specs):
        stage_warm = warm[warm["stage_name"] == stage].copy()
        if stage_warm.empty:
            raise ValueError(f"no warm rows for stage {stage}")
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

        actual = means["action_duration_ms"].to_numpy(dtype=float)
        amdahl_params, amdahl_pred = fit_amdahl(means, specs[stage])
        power_params, power_pred = fit_powerlaw(means)
        amdahl_max, amdahl_rms = fit_quality(actual, amdahl_pred)
        power_max, power_rms = fit_quality(actual, power_pred)
        breakpoint_values = means.loc[means["worker_cap"] > 1.0, "allocated_cpu_cores"]
        breakpoint_cpu = float(breakpoint_values.min()) if not breakpoint_values.empty else 1.0

        amdahl_rows.append(
            {
                "stage_name": stage,
                "S_ms": amdahl_params[0],
                "P_ms": amdahl_params[1],
                "C_ms": amdahl_params[2],
                "W_eff_breakpoint": breakpoint_cpu,
                "n_tiers": len(means),
                "fit_residual_max_pct": amdahl_max,
                "fit_residual_rms_pct": amdahl_rms,
            }
        )
        power_rows.append(
            {
                "stage_name": stage,
                "a": power_params[0],
                "alpha": power_params[1],
                "c": power_params[2],
                "n_tiers": len(means),
                "fit_residual_max_pct": power_max,
                "fit_residual_rms_pct": power_rms,
            }
        )
        comparison_rows.extend(
            [
                {
                    "stage_name": stage,
                    "model": "amdahl",
                    "max_relative_error": amdahl_max,
                    "rms_relative_error": amdahl_rms,
                },
                {
                    "stage_name": stage,
                    "model": "powerlaw",
                    "max_relative_error": power_max,
                    "rms_relative_error": power_rms,
                },
            ]
        )
        residual_rows.extend(
            residual_frame(stage, means, "amdahl", amdahl_pred).to_dict(orient="records")
        )
        residual_rows.extend(
            residual_frame(stage, means, "powerlaw", power_pred).to_dict(orient="records")
        )

    cold_overhead = (
        cold.assign(cold_overhead_ms=cold["dispatch_latency_ms"] - cold["action_duration_ms"])
        .groupby(["stage_name", "allocated_memory_mb"], as_index=False)
        .agg(
            cold_overhead_mean_ms=("cold_overhead_ms", "mean"),
            cold_overhead_std_ms=("cold_overhead_ms", "std"),
            sample_count=("cold_overhead_ms", "size"),
        )
        .rename(columns={"allocated_memory_mb": "memory_mb"})
        .sort_values(["stage_name", "memory_mb"])
    )

    amdahl_params = pd.DataFrame(amdahl_rows).sort_values("stage_name")
    powerlaw_params = pd.DataFrame(power_rows).sort_values("stage_name")
    comparison = pd.DataFrame(comparison_rows).sort_values(["stage_name", "model"])
    fit_means = pd.DataFrame(mean_rows).sort_values(["stage_name", "allocated_cpu_cores"])
    residuals = pd.DataFrame(residual_rows).sort_values(
        ["stage_name", "model", "allocated_cpu_cores"]
    )

    warm_validation = (
        residuals[residuals["model"] == "amdahl"]
        .groupby("stage_name", as_index=False)
        .agg(
            n_tiers=("allocated_cpu_cores", "size"),
            rms_error_pct=("relative_error_pct", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            max_error_pct=("relative_error_pct", lambda s: float(np.max(np.abs(s)))),
        )
    )
    warm_validation["pass_3pct"] = warm_validation["rms_error_pct"] < 3.0

    cold_validation = (
        cold_overhead.groupby("stage_name", as_index=False)["cold_overhead_mean_ms"]
        .agg(cold_oh_mean_ms="mean", cold_oh_std_ms="std")
    )
    cold_validation["cv_pct"] = (
        cold_validation["cold_oh_std_ms"]
        / cold_validation["cold_oh_mean_ms"].replace(0.0, np.nan)
        * 100.0
    )
    cold_validation["pass_20pct"] = cold_validation["cv_pct"] < 20.0

    amdahl_params.to_csv(out_dir / "per_stage_amdahl_params.csv", index=False)
    powerlaw_params.to_csv(out_dir / "per_stage_powerlaw_params.csv", index=False)
    cold_overhead.to_csv(out_dir / "per_stage_cold_overhead.csv", index=False)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)
    fit_means.to_csv(out_dir / "warm_tier_means.csv", index=False)
    residuals.to_csv(out_dir / "warm_fit_residuals_by_tier.csv", index=False)
    warm_validation.to_csv(out_dir / "warm_fit_validation.csv", index=False)
    cold_validation.to_csv(out_dir / "cold_overhead_validation.csv", index=False)
    plot_predictions(out_dir / "predictions_vs_actual.png", warm, fit_means, amdahl_params, powerlaw_params)
    write_readme(
        out_dir / "README.md",
        amdahl_params,
        powerlaw_params,
        comparison,
        warm_validation,
        cold_validation,
        specs,
    )

    print("Amdahl parameters:")
    print(amdahl_params.to_string(index=False))
    print("\nWarm A1 validation:")
    print(warm_validation.to_string(index=False))
    print("\nCold overhead validation:")
    print(cold_validation.to_string(index=False))
    print("\nBest model by RMS relative error:")
    best = comparison.sort_values(["stage_name", "rms_relative_error"]).groupby(
        "stage_name", as_index=False
    ).first()
    print(best[["stage_name", "model", "rms_relative_error"]].to_string(index=False))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
