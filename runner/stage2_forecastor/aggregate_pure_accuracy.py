"""Aggregate forecast-detail CSVs into a pure forecast-accuracy leaderboard.

Computes three metrics from `actual_count` and `forecast_count` only — no
allocation / cost concept:

  error    = mean |actual - forecast|                                  (MAE)
  over_est = sum max(0, forecast - actual) / sum actual                (relative
             magnitude of overestimation)
  sMAPE    = mean(2|actual-forecast| / (|actual|+|forecast|)) * 100%   (with
             the convention that windows where actual=forecast=0 contribute 0)

By default it scans `data/entry_forecasts/<run_dir>/` and concatenates any
*_detail.csv it finds. It also reports peak10 (MAE on the top-10% actual
windows) as supplementary context.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--detail-globs", nargs="+", required=True,
                   help="one or more glob patterns of detail CSV files")
    p.add_argument("--filter-policy",
                   default="p50",
                   help=("for detail CSVs that have a `policy` column, keep only "
                         "this policy (p50 = point forecast); use 'none' to keep all rows"))
    p.add_argument("--out-csv", default=None,
                   help="optional output CSV path (omit to only print)")
    p.add_argument("--label", default=None,
                   help="optional label written into output rows")
    return p.parse_args()


def compute_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    abs_err = np.abs(actual - forecast)
    over = np.maximum(0.0, forecast - actual)
    actual_sum = actual.sum()
    # sMAPE with zero-zero convention
    denom = np.abs(actual) + np.abs(forecast)
    sm = np.where(denom > 0, 2.0 * abs_err / denom, 0.0)
    smape = float(sm.mean()) * 100.0
    # peak10
    active = actual > 0
    if active.any():
        thr = float(np.quantile(actual[active], 0.9))
        peak = actual >= thr
    else:
        peak = np.zeros(len(actual), dtype=bool)
    return {
        "error_mae": float(abs_err.mean()),
        "error_rmse": float(np.sqrt(((actual - forecast) ** 2).mean())),
        "over_est": float(over.sum() / max(actual_sum, 1e-9)),
        "sMAPE_percent": smape,
        "peak10_mae": float(abs_err[peak].mean()) if peak.any() else 0.0,
        "windows": int(len(actual)),
        "actual_total": int(actual_sum),
        "forecast_total": float(forecast.sum()),
        "zero_actual_frac": float((actual == 0).mean()),
    }


def main() -> None:
    args = parse_args()
    files: list[Path] = []
    for pattern in args.detail_globs:
        files.extend(Path(".").glob(pattern))
    if not files:
        raise SystemExit(f"no detail CSVs matched: {args.detail_globs}")

    rows = []
    for f in sorted(files):
        df = pd.read_csv(f)
        if args.filter_policy != "none" and "policy" in df.columns:
            df = df[df["policy"] == args.filter_policy]
        if "method" not in df.columns or "actual_count" not in df.columns or "forecast_count" not in df.columns:
            print(f"[skip] {f} missing required columns")
            continue
        for method, grp in df.groupby("method"):
            m = compute_metrics(
                grp["actual_count"].to_numpy(),
                grp["forecast_count"].to_numpy(),
            )
            m["method"] = str(method)
            m["source"] = str(f)
            if args.label:
                m["label"] = args.label
            rows.append(m)

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no rows produced")
    cols = ["method", "error_mae", "over_est", "sMAPE_percent",
            "peak10_mae", "error_rmse",
            "windows", "actual_total", "forecast_total", "zero_actual_frac",
            "source"]
    if args.label:
        cols.insert(0, "label")
    out = out[[c for c in cols if c in out.columns]].sort_values("error_mae")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f"wrote {args.out_csv}")

    # console leaderboard
    print("\n== pure forecast accuracy ==")
    print(out.assign(
        error_mae=out["error_mae"].round(3),
        over_est=out["over_est"].round(3),
        sMAPE_percent=out["sMAPE_percent"].round(2),
        peak10_mae=out["peak10_mae"].round(3),
        error_rmse=out["error_rmse"].round(3),
    )[["method", "error_mae", "over_est", "sMAPE_percent", "peak10_mae", "error_rmse"]
       ].to_string(index=False))


if __name__ == "__main__":
    main()
