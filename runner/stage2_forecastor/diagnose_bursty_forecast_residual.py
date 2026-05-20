"""Diagnose bursty entry-forecast residuals.

Split forecast windows into peak / shoulder / active_other / idle and compare
forecast counts vs actual counts in each bucket. Determines whether the bursty
trace failure is uniform over-prediction, peak under-prediction with shoulder
over-prediction, or active/idle misclassification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-trace", required=True)
    parser.add_argument("--forecast-detail", required=True)
    parser.add_argument("--window-ms", type=int, default=5000)
    parser.add_argument("--methods", nargs="+", default=["ewma", "burst-aware", "hazard-hurdle", "hurdle-ewma", "tsb"])
    parser.add_argument("--policy", default="p95")
    parser.add_argument("--peak-quantile", type=float, default=0.90)
    parser.add_argument("--near-radius-windows", type=int, default=3)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def build_actual_counts(entry_trace_path: Path, window_ms: int) -> pd.DataFrame:
    df = pd.read_csv(entry_trace_path)
    df = df[df["stage_name"] == "__entry__"].copy()
    df["window"] = (df["entry_ts_ms"].astype("int64") // window_ms).astype("int64")
    counts = df.groupby("window").size().rename("actual_count").reset_index()
    min_w = int(counts["window"].min())
    max_w = int(counts["window"].max())
    full = pd.DataFrame({"window": range(min_w, max_w + 1)})
    out = full.merge(counts, on="window", how="left")
    out["actual_count"] = out["actual_count"].fillna(0).astype(int)
    return out


def label_regions(actual: pd.DataFrame, peak_quantile: float, near_radius: int) -> pd.DataFrame:
    a = actual.copy().reset_index(drop=True)
    positive = a[a["actual_count"] > 0]
    if positive.empty:
        a["region"] = "idle"
        a["peak_threshold"] = 0
        return a
    thr = float(positive["actual_count"].quantile(peak_quantile))
    peak_idx = set(a.index[a["actual_count"] >= thr].tolist())
    near_idx = set()
    for idx in peak_idx:
        for d in range(-near_radius, near_radius + 1):
            j = idx + d
            if 0 <= j < len(a) and j not in peak_idx:
                near_idx.add(j)
    regions = []
    for i in range(len(a)):
        if i in peak_idx:
            regions.append("peak")
        elif i in near_idx:
            regions.append("shoulder")
        elif a.loc[i, "actual_count"] > 0:
            regions.append("active_other")
        else:
            regions.append("idle")
    a["region"] = regions
    a["peak_threshold"] = thr
    return a


def summarise(forecast: pd.DataFrame, labeled: pd.DataFrame, method: str, policy: str) -> pd.DataFrame:
    sub = forecast[(forecast["method"] == method) & (forecast["policy"] == policy)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.rename(columns={"target_window": "window"})
    merged = sub.merge(labeled[["window", "actual_count", "region"]], on="window", how="inner")
    if merged.empty:
        return pd.DataFrame()
    # forecast detail already has actual_count column; keep merged version distinct
    merged["actual"] = merged["actual_count_y"] if "actual_count_y" in merged.columns else merged["actual_count"]
    merged["alloc"] = merged["allocated_count"].astype(float)
    merged["forecast"] = merged["forecast_count"].astype(float)
    merged["residual_alloc"] = merged["alloc"] - merged["actual"]

    rows = []
    for region, g in merged.groupby("region"):
        rows.append(_row(method, policy, region, g))
    rows.append(_row(method, policy, "ALL", merged))
    return pd.DataFrame(rows)


def _row(method: str, policy: str, region: str, g: pd.DataFrame) -> dict:
    return {
        "method": method,
        "policy": policy,
        "region": region,
        "n_windows": int(len(g)),
        "actual_sum": float(g["actual"].sum()),
        "actual_max": float(g["actual"].max() if len(g) else 0),
        "actual_mean": float(g["actual"].mean() if len(g) else 0),
        "alloc_sum": float(g["alloc"].sum()),
        "alloc_max": float(g["alloc"].max() if len(g) else 0),
        "alloc_mean": float(g["alloc"].mean() if len(g) else 0),
        "forecast_mean": float(g["forecast"].mean() if len(g) else 0),
        "mean_residual": float(g["residual_alloc"].mean() if len(g) else 0),
        "median_residual": float(g["residual_alloc"].median() if len(g) else 0),
        "p90_residual": float(g["residual_alloc"].quantile(0.90) if len(g) else 0),
        "mae": float(g["residual_alloc"].abs().mean() if len(g) else 0),
        "under_sum": float(np.maximum(0, -g["residual_alloc"]).sum()),
        "over_sum": float(np.maximum(0, g["residual_alloc"]).sum()),
    }


def main() -> None:
    args = parse_args()
    entry_trace = Path(args.entry_trace)
    forecast_detail = Path(args.forecast_detail)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    actual = build_actual_counts(entry_trace, args.window_ms)
    labeled = label_regions(actual, args.peak_quantile, args.near_radius_windows)
    labeled.to_csv(out_dir / "actual_by_window_labeled.csv", index=False)

    forecast = pd.read_csv(forecast_detail)
    pieces = []
    for method in args.methods:
        s = summarise(forecast, labeled, method, args.policy)
        if not s.empty:
            pieces.append(s)
    summary = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    summary.to_csv(out_dir / "region_residual_summary.csv", index=False)

    region_counts = (
        labeled.groupby("region")
        .agg(n_windows=("window", "size"), actual_sum=("actual_count", "sum"), actual_max=("actual_count", "max"))
        .reset_index()
    )
    region_counts.to_csv(out_dir / "region_window_counts.csv", index=False)

    meta = {
        "entry_trace": str(entry_trace),
        "forecast_detail": str(forecast_detail),
        "window_ms": args.window_ms,
        "policy": args.policy,
        "methods": args.methods,
        "peak_quantile": args.peak_quantile,
        "near_radius_windows": args.near_radius_windows,
        "total_windows": int(len(actual)),
        "active_windows": int((actual["actual_count"] > 0).sum()),
        "max_actual": int(actual["actual_count"].max()),
        "total_actual": int(actual["actual_count"].sum()),
        "peak_threshold": float(labeled["peak_threshold"].iloc[0]) if len(labeled) else 0,
    }
    (out_dir / "diagnosis_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print()
    if not summary.empty:
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
            print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
