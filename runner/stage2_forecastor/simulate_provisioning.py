"""Lightweight Stage-2 -> provisioning simulator.

Given a forecast detail CSV (method, window, actual_count, forecast_count), this
script simulates a simple container-provisioning policy and reports:

  * over_cost  : provisioned-but-unused capacity over all windows
                 (sum of max(0, capacity - actual) per window)
  * miss_count : requests that arrived when capacity was insufficient
                 (sum of max(0, actual - capacity))
  * miss_rate  : miss_count / total_actual
  * cold_count : extra containers spun up vs the previous window
                 (sum of max(0, containers_t - containers_{t-1}))
  * util       : sum(min(actual, capacity)) / sum(capacity)
  * total_cost : alpha_keep * total_capacity + alpha_cold * cold_count
                 + alpha_miss * miss_count

The default coefficients (alpha_*) are deliberately simple and tunable.

Provisioning policy (single-knob):
  containers_t = ceil(forecast_count_t / batch_size + safety_margin)

This is consistent with how serverless ML inference systems (e.g. SMIless,
IceBreaker) translate a per-window invocation forecast into number of
container instances.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--detail-globs", nargs="+", required=True,
                   help="glob patterns matching forecast detail CSVs")
    p.add_argument("--batch-size", type=int, default=8,
                   help="how many requests one container handles per window")
    p.add_argument("--safety-margin", type=float, default=0.0,
                   help="extra fractional capacity added before ceil() (0.0 = exact)")
    p.add_argument("--alpha-keep", type=float, default=1.0,
                   help="cost per provisioned container-window")
    p.add_argument("--alpha-cold", type=float, default=3.0,
                   help="cost per cold-start container")
    p.add_argument("--alpha-miss", type=float, default=5.0,
                   help="cost per SLA-missed request")
    p.add_argument("--label", default=None)
    p.add_argument("--out-csv", default=None)
    return p.parse_args()


def simulate(actual: np.ndarray, forecast: np.ndarray, batch_size: int,
             safety_margin: float, alpha_keep: float, alpha_cold: float,
             alpha_miss: float) -> dict:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    # 1. provisioning decision (containers per window)
    needed = forecast * (1.0 + safety_margin) / batch_size
    containers = np.ceil(np.maximum(0.0, needed)).astype(int)
    # exception: zero forecast -> zero containers (no overhead)
    containers = np.where(forecast <= 0, 0, containers)

    capacity = containers * batch_size
    # 2. derive over / under
    over = np.maximum(0, capacity - actual).astype(float)
    under = np.maximum(0, actual - capacity).astype(float)

    # 3. cold-start events: containers brought up beyond what was already there
    prev = np.concatenate([[0], containers[:-1]])
    cold = np.maximum(0, containers - prev).astype(float)

    # 4. utilization (only meaningful when capacity > 0)
    sat_actual = np.minimum(actual, capacity)
    util = float(sat_actual.sum() / max(capacity.sum(), 1.0))

    miss_count = float(under.sum())
    miss_rate = miss_count / max(actual.sum(), 1.0)
    over_cost = float(over.sum())
    cold_count = float(cold.sum())
    total_capacity = float(capacity.sum())

    total_cost = (
        alpha_keep * total_capacity
        + alpha_cold * cold_count
        + alpha_miss * miss_count
    )

    return {
        "windows": int(len(actual)),
        "actual_total": int(actual.sum()),
        "forecast_total": float(forecast.sum()),
        "containers_avg": float(containers.mean()),
        "containers_p95": float(np.quantile(containers, 0.95)),
        "containers_max": int(containers.max()),
        "capacity_total": total_capacity,
        "over_cost": over_cost,
        "miss_count": miss_count,
        "miss_rate": miss_rate,
        "cold_count": cold_count,
        "util": util,
        "total_cost": total_cost,
    }


def main() -> None:
    args = parse_args()
    files: list[Path] = []
    for pat in args.detail_globs:
        files.extend(Path(".").glob(pat))
    if not files:
        raise SystemExit(f"no detail CSVs matched: {args.detail_globs}")

    rows = []
    for f in sorted(files):
        df = pd.read_csv(f)
        if "method" not in df.columns:
            print(f"[skip] {f} missing method column")
            continue
        if "policy" in df.columns:
            df = df[df["policy"] == "p50"]
        for method, grp in df.groupby("method"):
            r = simulate(
                grp["actual_count"].to_numpy(),
                grp["forecast_count"].to_numpy(),
                batch_size=args.batch_size,
                safety_margin=args.safety_margin,
                alpha_keep=args.alpha_keep,
                alpha_cold=args.alpha_cold,
                alpha_miss=args.alpha_miss,
            )
            r["method"] = str(method)
            r["source"] = str(f)
            if args.label:
                r["label"] = args.label
            rows.append(r)

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no rows produced")

    cols = ["method", "total_cost", "capacity_total", "over_cost", "miss_count",
            "miss_rate", "cold_count", "util",
            "containers_avg", "containers_p95", "containers_max",
            "windows", "actual_total", "forecast_total", "source"]
    if args.label:
        cols.insert(0, "label")
    out = out[[c for c in cols if c in out.columns]].sort_values("total_cost")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f"wrote {args.out_csv}")

    print(f"\n== provisioning simulation "
          f"(batch={args.batch_size}, margin={args.safety_margin}, "
          f"alphas keep={args.alpha_keep}/cold={args.alpha_cold}/miss={args.alpha_miss}) ==")
    show_cols = ["method", "total_cost", "over_cost", "miss_count",
                 "miss_rate", "cold_count", "util", "containers_avg"]
    print(out.assign(
        total_cost=out["total_cost"].round(1),
        over_cost=out["over_cost"].round(0),
        miss_count=out["miss_count"].round(0),
        miss_rate=out["miss_rate"].round(3),
        cold_count=out["cold_count"].round(0),
        util=out["util"].round(3),
        containers_avg=out["containers_avg"].round(2),
    )[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
