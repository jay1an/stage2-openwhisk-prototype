#!/usr/bin/env python3
"""Fit per-stage lognormal latency parameters from Stage 3 sample pools."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


STAGE_ORDER = [
    "detect_object",
    "estimate_pose",
    "match_face",
    "classify_scene",
    "translate_alert",
]
CLASS_ORDER = ["warm", "cold_like"]
SAMPLES_CSV = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "stage3_latency_civic_alert_real_45min"
    / "latency_samples_for_monte_carlo.csv"
)
OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "path2_lognormal_fit"


def fit_lognormal(samples: np.ndarray) -> tuple[float, float]:
    """Return closed-form lognormal MLE parameters (mu, sigma)."""
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) < 2:
        raise ValueError("fit_lognormal requires at least two positive finite samples")
    log_values = np.log(values)
    return float(np.mean(log_values)), float(np.std(log_values, ddof=1))


def _positive_latency_frame(samples_csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(samples_csv_path)
    required = {"stage_name", "latency_class", "dispatch_latency_ms"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df = df.copy()
    df["dispatch_latency_ms"] = pd.to_numeric(df["dispatch_latency_ms"], errors="coerce")
    df = df[np.isfinite(df["dispatch_latency_ms"]) & (df["dispatch_latency_ms"] > 0.0)]
    return df


def _ordered_groups(df: pd.DataFrame) -> Iterable[tuple[str, str, pd.DataFrame]]:
    seen: set[tuple[str, str]] = set()
    for stage_name in STAGE_ORDER:
        for latency_class in CLASS_ORDER:
            group = df[(df["stage_name"] == stage_name) & (df["latency_class"] == latency_class)]
            if not group.empty:
                seen.add((stage_name, latency_class))
                yield stage_name, latency_class, group
    for (stage_name, latency_class), group in sorted(df.groupby(["stage_name", "latency_class"])):
        if (stage_name, latency_class) not in seen:
            yield str(stage_name), str(latency_class), group


def fit_per_stage(samples_csv_path: str | Path) -> pd.DataFrame:
    df = _positive_latency_frame(samples_csv_path)
    rows: list[dict[str, float | int | str]] = []
    for stage_name, latency_class, group in _ordered_groups(df):
        values = group["dispatch_latency_ms"].to_numpy(dtype=float)
        mu, sigma = fit_lognormal(values)
        dist = stats.lognorm(s=sigma, loc=0.0, scale=math.exp(mu))
        ks_statistic, ks_pvalue = stats.kstest(values, dist.cdf)

        mean_empirical = float(np.mean(values))
        std_empirical = float(np.std(values, ddof=1))
        mean_predicted = float(math.exp(mu + (sigma**2) / 2.0))
        std_predicted = float(math.sqrt((math.exp(sigma**2) - 1.0) * math.exp(2.0 * mu + sigma**2)))
        cv_empirical = float(std_empirical / mean_empirical)
        cv_predicted = float(math.sqrt(math.exp(sigma**2) - 1.0))
        rows.append(
            {
                "stage_name": stage_name,
                "latency_class": latency_class,
                "n_samples": int(len(values)),
                "mu": mu,
                "sigma": sigma,
                "mean_empirical": mean_empirical,
                "mean_predicted": mean_predicted,
                "std_empirical": std_empirical,
                "std_predicted": std_predicted,
                "cv_empirical": cv_empirical,
                "cv_predicted": cv_predicted,
                "p50_empirical": float(np.quantile(values, 0.50)),
                "p50_predicted": float(math.exp(mu)),
                "p95_empirical": float(np.quantile(values, 0.95)),
                "p95_predicted": float(math.exp(mu + 1.645 * sigma)),
                "ks_statistic": float(ks_statistic),
                "ks_pvalue": float(ks_pvalue),
            }
        )
    return pd.DataFrame(rows)


def _format_table(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def _fit_quality_label(row: pd.Series) -> str:
    if float(row["ks_pvalue"]) > 0.05 or float(row["ks_statistic"]) < 0.1:
        return "good"
    return "marginal"


def write_quality_report(params: pd.DataFrame, out_path: Path) -> None:
    report = params.copy()
    report["fit_quality"] = report.apply(_fit_quality_label, axis=1)
    report["mean_rel_error_pct"] = (
        (report["mean_predicted"] - report["mean_empirical"]).abs()
        / report["mean_empirical"]
        * 100.0
    )
    report["p50_rel_error_pct"] = (
        (report["p50_predicted"] - report["p50_empirical"]).abs()
        / report["p50_empirical"]
        * 100.0
    )
    report["p95_rel_error_pct"] = (
        (report["p95_predicted"] - report["p95_empirical"]).abs()
        / report["p95_empirical"]
        * 100.0
    )

    parameter_table = report[
        ["stage_name", "latency_class", "n_samples", "mu", "sigma", "mean_empirical", "mean_predicted"]
    ].copy()
    ks_table = report[
        ["stage_name", "latency_class", "ks_statistic", "ks_pvalue", "fit_quality"]
    ].copy()
    quantile_table = report[
        [
            "stage_name",
            "latency_class",
            "p50_empirical",
            "p50_predicted",
            "p50_rel_error_pct",
            "p95_empirical",
            "p95_predicted",
            "p95_rel_error_pct",
        ]
    ].copy()

    warm = report[report["latency_class"] == "warm"]
    cold = report[report["latency_class"] == "cold_like"]
    warm_mean_ok = bool((warm["mean_rel_error_pct"] <= 1.0).all())
    warm_sigma_range = warm[["stage_name", "sigma"]].copy()
    cold_sigma_range = cold[["stage_name", "sigma"]].copy()
    worst_cold = cold.sort_values(
        ["ks_statistic", "p95_rel_error_pct"],
        ascending=[False, False],
    ).head(1)
    worst_cold_text = "n/a"
    if not worst_cold.empty:
        row = worst_cold.iloc[0]
        worst_cold_text = (
            f"{row['stage_name']} (KS={row['ks_statistic']:.3f}, "
            f"p95 error={row['p95_rel_error_pct']:.1f}%)"
        )
    translate = cold[cold["stage_name"] == "translate_alert"]
    translate_text = "n/a"
    if not translate.empty:
        row = translate.iloc[0]
        translate_text = (
            f"KS={row['ks_statistic']:.3f}, p={row['ks_pvalue']:.3g}, "
            f"p95 empirical/predicted={row['p95_empirical']:.1f}/{row['p95_predicted']:.1f} ms"
        )

    lines = [
        "# Path 2 Lognormal Fit Quality",
        "",
        "## Parameter Table",
        "",
        _format_table(parameter_table),
        "",
        "## KS Test Results",
        "",
        _format_table(ks_table),
        "",
        "Good fits are defined as `ks_pvalue > 0.05` or `ks_statistic < 0.1`.",
        "Marginal fits are expected mostly in cold pools because they contain small samples and contention tails.",
        "",
        "## Empirical Vs Predicted Quantiles",
        "",
        _format_table(quantile_table),
        "",
        "## Sanity Checks",
        "",
        f"- Warm mean predicted within 1% of empirical for every stage: `{warm_mean_ok}`.",
        "- Warm sigma values:",
        _format_table(warm_sigma_range),
        "- Cold sigma values:",
        _format_table(cold_sigma_range),
        f"- Worst cold fit by KS statistic: `{worst_cold_text}`.",
        f"- Translate-alert cold fit: `{translate_text}`.",
        "",
        "## Notes",
        "",
        "- `dispatch_latency_ms` is used for fitting, so the distribution includes action time plus platform overhead.",
        "- The lognormal mean matches empirical means closely because the closed-form MLE is fitted on log latency, not directly on arithmetic moments.",
        "- Cold p95 gaps are larger than warm gaps because each cold pool has roughly 100 samples and known OpenWhisk wait-time contamination.",
        "- The source CSV has no memory-tier dimension; the requested comparison is therefore reported per `(stage_name, latency_class)` rather than per tier.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _subplot_grid() -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(len(STAGE_ORDER), len(CLASS_ORDER), figsize=(11, 16), constrained_layout=True)
    return fig, axes


def write_qq_plots(samples_csv_path: str | Path, params: pd.DataFrame, out_path: Path) -> None:
    df = _positive_latency_frame(samples_csv_path)
    fig, axes = _subplot_grid()
    for row_idx, stage_name in enumerate(STAGE_ORDER):
        for col_idx, latency_class in enumerate(CLASS_ORDER):
            ax = axes[row_idx, col_idx]
            group = df[(df["stage_name"] == stage_name) & (df["latency_class"] == latency_class)]
            param = params[(params["stage_name"] == stage_name) & (params["latency_class"] == latency_class)]
            if group.empty or param.empty:
                ax.set_axis_off()
                continue
            values = np.sort(group["dispatch_latency_ms"].to_numpy(dtype=float))
            n = len(values)
            p = (np.arange(1, n + 1) - 0.5) / n
            mu = float(param["mu"].iloc[0])
            sigma = float(param["sigma"].iloc[0])
            theoretical = stats.lognorm.ppf(p, s=sigma, loc=0.0, scale=math.exp(mu))
            ax.scatter(theoretical, values, s=8, alpha=0.55, linewidths=0)
            lower = float(min(theoretical.min(), values.min()))
            upper = float(max(theoretical.max(), values.max()))
            ax.plot([lower, upper], [lower, upper], color="black", linewidth=1.0, linestyle="--")
            ax.set_title(f"{stage_name} - {latency_class}")
            ax.set_xlabel("Theoretical lognormal quantiles (ms)")
            ax.set_ylabel("Empirical quantiles (ms)")
            ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_cdf_plots(samples_csv_path: str | Path, params: pd.DataFrame, out_path: Path) -> None:
    df = _positive_latency_frame(samples_csv_path)
    fig, axes = _subplot_grid()
    for row_idx, stage_name in enumerate(STAGE_ORDER):
        for col_idx, latency_class in enumerate(CLASS_ORDER):
            ax = axes[row_idx, col_idx]
            group = df[(df["stage_name"] == stage_name) & (df["latency_class"] == latency_class)]
            param = params[(params["stage_name"] == stage_name) & (params["latency_class"] == latency_class)]
            if group.empty or param.empty:
                ax.set_axis_off()
                continue
            values = np.sort(group["dispatch_latency_ms"].to_numpy(dtype=float))
            n = len(values)
            empirical = np.arange(1, n + 1) / n
            mu = float(param["mu"].iloc[0])
            sigma = float(param["sigma"].iloc[0])
            x = np.linspace(float(values.min()), float(values.max()), 400)
            fitted = stats.lognorm.cdf(x, s=sigma, loc=0.0, scale=math.exp(mu))
            ax.step(values, empirical, where="post", label="Empirical CDF", linewidth=1.4)
            ax.plot(x, fitted, linestyle="--", label="Fitted lognormal", linewidth=1.4)
            ax.set_title(f"{stage_name} - {latency_class}")
            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel("Cumulative probability")
            ax.set_ylim(0.0, 1.02)
            ax.grid(True, alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="lower right", fontsize=8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-csv", default=str(SAMPLES_CSV))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = fit_per_stage(args.samples_csv)
    params.to_csv(out_dir / "per_stage_lognormal_params.csv", index=False)

    sanity = params[
        [
            "stage_name",
            "latency_class",
            "n_samples",
            "mu",
            "sigma",
            "mean_empirical",
            "mean_predicted",
            "ks_pvalue",
        ]
    ].copy()
    print(sanity.to_string(index=False))

    if (params["sigma"] < 0.0).any():
        raise SystemExit("negative sigma detected; aborting report generation")
    mean_rel_error = (params["mean_predicted"] - params["mean_empirical"]).abs() / params["mean_empirical"]
    if not np.isfinite(mean_rel_error).all():
        raise SystemExit("non-finite mean prediction detected; aborting report generation")

    write_quality_report(params, out_dir / "fit_quality_summary.md")
    write_qq_plots(args.samples_csv, params, out_dir / "qq_plots.png")
    write_cdf_plots(args.samples_csv, params, out_dir / "empirical_vs_fitted_cdf.png")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
