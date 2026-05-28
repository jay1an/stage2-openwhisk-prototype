#!/usr/bin/env python3
"""R5 multi-node re-validation for the path 2 analytical risk model."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.stage4_risk.plan_risk import load_lognormal_params  # noqa: E402
from scripts.run_r4_path2_validation import (  # noqa: E402
    SLO_VALUES,
    STAGES,
    calibrate_transition_gap,
    cold_patterns,
    load_trace_tables,
    run_no_jit_validation,
    table_text,
)


OLD_TRACE = (
    ROOT
    / "reports"
    / "civic_azure_cand2_45min_1280mb_1cpu_keepalive20s_target20s_balanced_mi96"
    / "raw_trace.csv"
)
NEW_TRACE = (
    ROOT
    / "reports"
    / "civic_azure_cand2_60min_1280mb_1cpu_keepalive10s_target20s_2x_mi96"
    / "raw_trace.csv"
)
OLD_LOGNORMAL = ROOT / "reports" / "path2_lognormal_fit" / "per_stage_lognormal_params.csv"
NEW_LOGNORMAL = ROOT / "reports" / "path2_lognormal_fit_multinode" / "per_stage_lognormal_params.csv"
OLD_STAGE3_SAMPLES = ROOT / "reports" / "stage3_latency_civic_alert_real_45min" / "latency_samples_for_monte_carlo.csv"
NEW_STAGE3_SAMPLES = ROOT / "reports" / "stage3_latency_civic_alert_multinode_60min" / "latency_samples_for_monte_carlo.csv"
OLD_NO_JIT = ROOT / "reports" / "path2_no_jit_validation" / "no_jit_calibration.csv"
OUT_DIR = ROOT / "reports" / "path2_multinode_validation"


def pairwise_inter_stage_correlation(raw_trace_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    entry, dispatch, cold = load_trace_tables(raw_trace_path)
    del entry
    all_warm = dispatch[(cold[STAGES] == False).all(axis=1)].dropna()  # noqa: E712
    corr = all_warm[STAGES].corr()
    rows: list[dict[str, Any]] = []
    for i, stage_a in enumerate(STAGES):
        for stage_b in STAGES[i + 1 :]:
            rows.append(
                {
                    "stage_a": stage_a,
                    "stage_b": stage_b,
                    "pearson_r": float(corr.loc[stage_a, stage_b]),
                    "all_warm_workflows": int(len(all_warm)),
                }
            )
    pairwise = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {"metric": "all_warm_workflows", "value": float(len(all_warm))},
            {"metric": "mean_pairwise_pearson_r", "value": float(pairwise["pearson_r"].mean())},
            {"metric": "median_pairwise_pearson_r", "value": float(pairwise["pearson_r"].median())},
            {"metric": "max_pairwise_pearson_r", "value": float(pairwise["pearson_r"].max())},
            {"metric": "min_pairwise_pearson_r", "value": float(pairwise["pearson_r"].min())},
        ]
    )
    return pairwise, summary


def distribution_comparison() -> pd.DataFrame:
    old = pd.read_csv(OLD_LOGNORMAL)
    new = pd.read_csv(NEW_LOGNORMAL)
    merged = old.merge(
        new,
        on=["stage_name", "latency_class"],
        suffixes=("_old", "_new"),
    )
    merged["new_mean_rel_error_pct"] = (
        (merged["mean_predicted_new"] - merged["mean_empirical_new"]).abs()
        / merged["mean_empirical_new"]
        * 100.0
    )
    merged["sigma_ratio_new_over_old"] = merged["sigma_new"] / merged["sigma_old"]
    return merged[
        [
            "stage_name",
            "latency_class",
            "n_samples_old",
            "n_samples_new",
            "mu_old",
            "mu_new",
            "sigma_old",
            "sigma_new",
            "sigma_ratio_new_over_old",
            "mean_empirical_old",
            "mean_empirical_new",
            "mean_predicted_new",
            "new_mean_rel_error_pct",
            "p95_empirical_old",
            "p95_empirical_new",
            "p95_predicted_new",
        ]
    ]


def stage_std_comparison() -> pd.DataFrame:
    old = pd.read_csv(OLD_STAGE3_SAMPLES)
    new = pd.read_csv(NEW_STAGE3_SAMPLES)
    old_std = old.groupby(["stage_name", "latency_class"])["dispatch_latency_ms"].std().unstack()
    new_std = new.groupby(["stage_name", "latency_class"])["dispatch_latency_ms"].std().unstack()
    rows = []
    for stage in STAGES:
        rows.append(
            {
                "stage_name": stage,
                "old_warm_std": float(old_std.loc[stage, "warm"]),
                "new_warm_std": float(new_std.loc[stage, "warm"]),
                "warm_std_ratio": float(new_std.loc[stage, "warm"] / old_std.loc[stage, "warm"]),
                "old_cold_std": float(old_std.loc[stage, "cold_like"]),
                "new_cold_std": float(new_std.loc[stage, "cold_like"]),
                "cold_std_ratio": float(new_std.loc[stage, "cold_like"] / old_std.loc[stage, "cold_like"]),
            }
        )
    return pd.DataFrame(rows)


def old_new_accuracy_table(new_calibration: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(OLD_NO_JIT)
    rows = []
    for slo_ms in SLO_VALUES:
        old_row = old[old["slo_ms"] == slo_ms].iloc[0]
        new_row = new_calibration[new_calibration["slo_ms"] == slo_ms].iloc[0]
        rows.append(
            {
                "slo_ms": slo_ms,
                "observed_old": float(old_row["observed_violation_rate"]),
                "predicted_old": float(old_row["predicted_violation_rate"]),
                "abs_error_old": float(old_row["abs_error"]),
                "observed_new": float(new_row["observed_violation_rate"]),
                "predicted_new": float(new_row["predicted_violation_rate"]),
                "abs_error_new": float(new_row["abs_error"]),
                "passes_new_2pp": bool(float(new_row["abs_error"]) <= 0.02),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    *,
    out_path: Path,
    dist_cmp: pd.DataFrame,
    std_cmp: pd.DataFrame,
    old_corr_summary: pd.DataFrame,
    new_corr_summary: pd.DataFrame,
    new_corr_pairwise: pd.DataFrame,
    transition_summary: pd.DataFrame,
    no_jit_calibration: pd.DataFrame,
    accuracy_cmp: pd.DataFrame,
) -> None:
    old_rho = float(old_corr_summary.loc[old_corr_summary["metric"] == "mean_pairwise_pearson_r", "value"].iloc[0])
    new_rho = float(new_corr_summary.loc[new_corr_summary["metric"] == "mean_pairwise_pearson_r", "value"].iloc[0])
    new_20 = no_jit_calibration[no_jit_calibration["slo_ms"] == 20_000].iloc[0]
    pass_20 = bool(float(new_20["abs_error"]) <= 0.02)
    all_pass = bool((no_jit_calibration["abs_error"] <= 0.02).all())
    warm = dist_cmp[dist_cmp["latency_class"] == "warm"]
    cold = dist_cmp[dist_cmp["latency_class"] == "cold_like"]
    transition = dict(zip(transition_summary["metric"], transition_summary["value"], strict=False))

    lines = [
        "# R5 Multi-Node Path 2 Validation",
        "",
        "## Per-Stage Distribution Comparison",
        "",
        "Warm lognormal parameter comparison:",
        table_text(
            warm[
                [
                    "stage_name",
                    "mu_old",
                    "mu_new",
                    "sigma_old",
                    "sigma_new",
                    "sigma_ratio_new_over_old",
                    "new_mean_rel_error_pct",
                ]
            ]
        ),
        "",
        "Cold-like lognormal parameter comparison:",
        table_text(
            cold[
                [
                    "stage_name",
                    "mu_old",
                    "mu_new",
                    "sigma_old",
                    "sigma_new",
                    "sigma_ratio_new_over_old",
                    "new_mean_rel_error_pct",
                ]
            ]
        ),
        "",
        "Stage dispatch std comparison:",
        table_text(std_cmp),
        "",
        f"- Max new warm mean prediction error: `{warm['new_mean_rel_error_pct'].max():.3f}%`.",
        f"- Mean warm sigma ratio new/old: `{warm['sigma_ratio_new_over_old'].mean():.3f}`.",
        "",
        "## Inter-Stage Correlation",
        "",
        f"- Old mean pairwise Pearson rho: `{old_rho:.6f}`.",
        f"- New mean pairwise Pearson rho: `{new_rho:.6f}`.",
        f"- New rho < 0.01: `{abs(new_rho) < 0.01}`.",
        "",
        "New pairwise correlation table:",
        table_text(new_corr_pairwise),
        "",
        "## Transition Gap",
        "",
        table_text(transition_summary),
        f"- Multi-node per-edge transition overhead: `{transition['per_edge_overhead_ms']:.3f} ms`.",
        "",
        "## No-JIT Model Accuracy",
        "",
        table_text(accuracy_cmp),
        "",
        f"- New 20s observed violation: `{new_20['observed_violation_rate']:.6f}`.",
        f"- New 20s predicted violation: `{new_20['predicted_violation_rate']:.6f}`.",
        f"- New 20s absolute error: `{new_20['abs_error']:.6f}`.",
        f"- 20s acceptance ±2pp: `{pass_20}`.",
        f"- All SLO thresholds within ±2pp: `{all_pass}`.",
        "",
        "## Verdict",
        "",
    ]
    if pass_20:
        lines.extend(
            [
                "At the primary 20s SLO, the multi-node validation passes the ±2pp acceptance criterion.",
                "Path 2 is validated under this more production-like multi-node condition for the target SLO.",
            ]
        )
    else:
        lines.extend(
            [
                "At the primary 20s SLO, the multi-node validation does not pass the ±2pp acceptance criterion.",
                "Residual gap should be investigated before declaring path 2 fully validated.",
            ]
        )
    if all_pass:
        lines.append("All tested SLO thresholds also pass the ±2pp criterion.")
    else:
        failed = accuracy_cmp[~accuracy_cmp["passes_new_2pp"]]["slo_ms"].astype(int).tolist()
        lines.append(f"SLO thresholds failing ±2pp: `{failed}`.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("R5: loading multi-node trace")
    entry, dispatch, cold = load_trace_tables(NEW_TRACE)
    patterns = cold_patterns(entry, cold)
    lognormal_params = load_lognormal_params(NEW_LOGNORMAL)

    print("R5: transition gap and no-JIT validation")
    gap, transition_summary, transition_overhead_ms = calibrate_transition_gap(entry, dispatch, patterns)
    predictions, grouped, no_jit_calibration = run_no_jit_validation(
        patterns,
        lognormal_params,
        transition_overhead_ms,
    )

    gap_summary_out = transition_summary.copy()
    gap_summary_out.to_csv(OUT_DIR / "transition_gap_summary_multinode.csv", index=False)
    patterns.to_csv(OUT_DIR / "per_workflow_cold_pattern_multinode.csv", index=False)
    predictions.to_csv(OUT_DIR / "per_workflow_predictions_multinode.csv", index=False)
    grouped.to_csv(OUT_DIR / "grouped_validation_multinode.csv", index=False)
    no_jit_calibration.to_csv(OUT_DIR / "no_jit_calibration_multinode.csv", index=False)
    gap.to_csv(OUT_DIR / "transition_gap_analysis_multinode.csv", index=False)

    print("R5: correlation and distribution comparisons")
    old_corr_pairwise, old_corr_summary = pairwise_inter_stage_correlation(OLD_TRACE)
    new_corr_pairwise, new_corr_summary = pairwise_inter_stage_correlation(NEW_TRACE)
    old_corr_pairwise.to_csv(OUT_DIR / "inter_stage_correlation_old.csv", index=False)
    old_corr_summary.to_csv(OUT_DIR / "inter_stage_correlation_old_summary.csv", index=False)
    new_corr_pairwise.to_csv(OUT_DIR / "inter_stage_correlation_multinode.csv", index=False)
    new_corr_summary.to_csv(OUT_DIR / "inter_stage_correlation_multinode_summary.csv", index=False)

    dist_cmp = distribution_comparison()
    std_cmp = stage_std_comparison()
    accuracy_cmp = old_new_accuracy_table(no_jit_calibration)
    dist_cmp.to_csv(OUT_DIR / "lognormal_params_old_vs_multinode.csv", index=False)
    std_cmp.to_csv(OUT_DIR / "stage_std_old_vs_multinode.csv", index=False)
    accuracy_cmp.to_csv(OUT_DIR / "single_vs_multinode_no_jit_accuracy.csv", index=False)

    write_report(
        out_path=OUT_DIR / "r5_validation_report.md",
        dist_cmp=dist_cmp,
        std_cmp=std_cmp,
        old_corr_summary=old_corr_summary,
        new_corr_summary=new_corr_summary,
        new_corr_pairwise=new_corr_pairwise,
        transition_summary=transition_summary,
        no_jit_calibration=no_jit_calibration,
        accuracy_cmp=accuracy_cmp,
    )

    print("Transition gap summary:")
    print(transition_summary.to_string(index=False))
    print("\nNew no-JIT calibration:")
    print(no_jit_calibration.to_string(index=False))
    print("\nCorrelation summary:")
    print(pd.concat(
        [
            old_corr_summary.assign(trace="old_single_node"),
            new_corr_summary.assign(trace="new_multinode"),
        ],
        ignore_index=True,
    )[["trace", "metric", "value"]].to_string(index=False))
    print("\nOld vs new no-JIT accuracy:")
    print(accuracy_cmp.to_string(index=False))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
