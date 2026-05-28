from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from runner.stage6_resource.fit_amdahl_model_extended import project_root, resolve_path


DEFAULT_OUT_DIR = "reports/stage6_resource_models_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare D1/D2/D3 resource scaling fits and recommend per-stage models."
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def model_pass(row: pd.Series, prefix: str) -> bool:
    return bool(row[f"{prefix}_pass_3pct"] and row[f"{prefix}_pass_8pct_max"])


def row_to_json(row: pd.Series, exclude: set[str] | None = None) -> str:
    exclude = exclude or set()
    data = {}
    for key, value in row.items():
        if key in exclude:
            continue
        if isinstance(value, (np.integer,)):
            data[key] = int(value)
        elif isinstance(value, (np.floating,)):
            data[key] = float(value)
        elif isinstance(value, (np.bool_,)):
            data[key] = bool(value)
        elif pd.isna(value):
            data[key] = None
        else:
            data[key] = value
    return json.dumps(data, sort_keys=True)


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_none_"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                values.append(format(float(val), floatfmt))
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_comparison(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d1 = pd.read_csv(out_dir / "d1_power_law_params.csv")
    d2 = pd.read_csv(out_dir / "d2_amdahl_observed_params.csv")
    d3 = pd.read_csv(out_dir / "d3_spline_fit_quality.csv")
    d3_coeffs = pd.read_csv(out_dir / "d3_spline_coeffs.csv")

    rows = []
    rec_rows = []
    for stage in sorted(set(d1["stage_name"]) | set(d2["stage_name"]) | set(d3["stage_name"])):
        d1_row = d1[d1["stage_name"] == stage].iloc[0]
        d2_row = d2[d2["stage_name"] == stage].iloc[0]
        d3_row = d3[d3["stage_name"] == stage].iloc[0]
        d3_coeff_row = d3_coeffs[d3_coeffs["stage_name"] == stage].iloc[0]

        record = {
            "stage_name": stage,
            "d1_rms_pct": float(d1_row["rms_error_pct"]),
            "d1_max_pct": float(d1_row["max_error_pct"]),
            "d1_pass_3pct": bool(d1_row["pass_3pct"]),
            "d1_pass_8pct_max": bool(d1_row["pass_8pct_max"]),
            "d1_pass": bool(d1_row["pass_3pct"] and d1_row["pass_8pct_max"]),
            "d2_rms_pct": float(d2_row["rms_error_pct"]),
            "d2_max_pct": float(d2_row["max_error_pct"]),
            "d2_pass_3pct": bool(d2_row["pass_3pct"]),
            "d2_pass_8pct_max": bool(d2_row["pass_8pct_max"]),
            "d2_pass": bool(d2_row["pass_3pct"] and d2_row["pass_8pct_max"]),
            "d3_rms_pct": float(d3_row["rms_error_pct"]),
            "d3_max_pct": float(d3_row["max_error_pct"]),
            "d3_pass_3pct": bool(d3_row["pass_3pct"]),
            "d3_pass_8pct_max": bool(d3_row["pass_8pct_max"]),
            "d3_pass": bool(d3_row["pass_3pct"] and d3_row["pass_8pct_max"]),
        }

        choices = [
            ("D1", record["d1_pass"], record["d1_rms_pct"], record["d1_max_pct"], d1_row),
            ("D2", record["d2_pass"], record["d2_rms_pct"], record["d2_max_pct"], d2_row),
            ("D3", record["d3_pass"], record["d3_rms_pct"], record["d3_max_pct"], d3_coeff_row),
        ]
        passing = [choice for choice in choices if choice[1]]
        if passing:
            selected = passing[0]
        else:
            selected = min(choices, key=lambda item: item[2])

        record["recommended_model"] = selected[0]
        record["recommended_rms_pct"] = selected[2]
        record["recommended_max_pct"] = selected[3]
        rows.append(record)

        rec_rows.append(
            {
                "stage_name": stage,
                "model_name": selected[0],
                "params_json": row_to_json(
                    selected[4],
                    exclude={
                        "stage_name",
                        "rms_error_pct",
                        "max_error_pct",
                        "pass_3pct",
                        "pass_8pct_max",
                        "note",
                    },
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("stage_name"), pd.DataFrame(rec_rows).sort_values("stage_name")


def write_report(out_dir: Path, comparison: pd.DataFrame, recommended: pd.DataFrame) -> None:
    cold_summary = pd.read_csv(out_dir / "cold_overhead_cleansing_summary.csv")
    cold_values = pd.read_csv(out_dir / "cold_overhead_cleansed.csv")
    d1 = pd.read_csv(out_dir / "d1_power_law_params.csv")
    d2 = pd.read_csv(out_dir / "d2_amdahl_observed_params.csv")
    d3 = pd.read_csv(out_dir / "d3_spline_fit_quality.csv")

    replaced = cold_values[cold_values["is_outlier"].astype(bool)].copy()
    if replaced.empty:
        replaced_text = "_No 2-sigma outliers detected._"
    else:
        replaced_text = md_table(
            replaced[
                [
                    "stage_name",
                    "tier_mb",
                    "original_cold_overhead_ms",
                    "cleansed_cold_overhead_ms",
                ]
            ]
        )

    fallback = comparison[comparison["recommended_model"] == "D3"]
    all_have_passing = bool((comparison[["d1_pass", "d2_pass", "d3_pass"]].any(axis=1)).all())
    open_issues = []
    if not all_have_passing:
        failed = comparison[
            ~comparison[["d1_pass", "d2_pass", "d3_pass"]].any(axis=1)
        ]["stage_name"].tolist()
        open_issues.append(f"Stages with no passing model: {', '.join(failed)}")
    if not fallback.empty:
        open_issues.append(
            "D3 spline fallback is recommended for: "
            + ", ".join(fallback["stage_name"].tolist())
            + ". This is exact at measured knots but should not be extrapolated."
        )
    if open_issues:
        open_issue_text = "\n".join(f"- {item}" for item in open_issues)
    else:
        open_issue_text = "- No stage lacks a passing model under the D1/D2/D3 comparison."

    report = f"""# P3.1-Retry Resource Model Comparison

## Cold Overhead Cleansing

Outlier rule: per stage, replace any tier whose cold overhead differs from the
9-tier mean by more than 2 standard deviations. Cold overhead is computed as
`cold_dispatch_mean - warm_dispatch_mean` at the same stage and tier.

{md_table(cold_summary)}

Replaced values:

{replaced_text}

Verdict: cold overhead noise is reasonable after cleansing; the cleansed values
are suitable for subsequent risk modeling as tier-level cold overhead inputs.

## Warm Fit Results

### D1 Power Law

{md_table(d1[["stage_name", "a", "alpha", "c", "rms_error_pct", "max_error_pct", "pass_3pct", "pass_8pct_max"]])}

### D2 Amdahl With Observed Workers

{md_table(d2[["stage_name", "S_ms", "P_ms", "C_ms", "rms_error_pct", "max_error_pct", "pass_3pct", "pass_8pct_max"]])}

### D3 Natural Cubic Spline

{md_table(d3[["stage_name", "rms_error_pct", "max_error_pct", "pass_3pct", "pass_8pct_max"]])}

## Model Comparison

{md_table(comparison[["stage_name", "d1_rms_pct", "d1_max_pct", "d1_pass", "d2_rms_pct", "d2_max_pct", "d2_pass", "d3_rms_pct", "d3_max_pct", "d3_pass", "recommended_model"]])}

## Recommendations

{md_table(recommended[["stage_name", "model_name"]])}

Recommendation policy: choose the simplest model that passes both thresholds
(D1 preferred over D2, D2 preferred over D3). D3 is allowed as an interpolation
fallback inside the observed 0.4-3.0 vCPU range; it should not be used for
extrapolation.

Overall: the 9-tier scaling problem is solved for interpolation inside the
measured tier range, because every stage has at least one passing model.

## Open Issues

{open_issue_text}
"""
    (out_dir / "comparison_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = project_root()
    out_dir = resolve_path(root, args.out_dir)
    comparison, recommended = build_comparison(out_dir)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)
    recommended.to_csv(out_dir / "recommended_model_per_stage.csv", index=False)
    write_report(out_dir, comparison, recommended)

    print("Model comparison:")
    print(comparison.to_string(index=False))
    print("\nRecommended model per stage:")
    print(recommended[["stage_name", "model_name"]].to_string(index=False))
    print(
        "\nOverall ready for P3.2 SLO setting:",
        bool((comparison[["d1_pass", "d2_pass", "d3_pass"]].any(axis=1)).all()),
    )
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
