"""Per-window allocation inspector.

Given (app, window_sec, method, policy), opens the matching detail CSV and
prints:
  - full per-window table (window, actual, forecast, allocated, under, over)
  - top-N over-allocated windows (sorted by over_count desc)
  - top-N under-allocated windows (sorted by under_count desc)
  - rolled-up totals

Optionally writes a clean CSV via --out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


WORKFLOWS = {
    "periodic_dense": "sebs_video",
    "periodic_sparse": "civic_alert_flow",
    "bursty_dense": "sebs_video",
    "bursty_sparse": "civic_alert_flow",
    "drift": "spoken_dialog_flow",
}

FAMILY_FILE = {
    "classical": "{wf}_entry_classical_compare_detail.csv",
    "ml": "{wf}_entry_ml_compare_detail.csv",
    "twostage": "{wf}_entry_twostage_compare_detail.csv",
    "histogram": "{wf}_histogram_entry_compare_detail.csv",
    "pointprocess": "{wf}_entry_pointprocess_compare_detail.csv",
    "lstm": "{wf}_entry_lstm_compare_detail.csv",
    "gru": "{wf}_entry_lstm_compare_detail.csv",
    "hedge": "{wf}_entry_hedge_compare_detail.csv",
    "cp_classical": "{wf}_entry_cp_aci_classical_detail.csv",
    "cp_ml": "{wf}_entry_cp_aci_ml_detail.csv",
    "cp_twostage": "{wf}_entry_cp_aci_twostage_detail.csv",
    "cp_pointprocess": "{wf}_entry_cp_aci_pointprocess_detail.csv",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-window allocation inspector")
    p.add_argument("--app", required=True, choices=list(WORKFLOWS))
    p.add_argument("--window-sec", type=int, required=True, choices=[2, 5])
    p.add_argument("--family", required=True, choices=list(FAMILY_FILE))
    p.add_argument("--method", required=True,
                   help="e.g. hawkes-exp, arima-101, naive, hedge")
    p.add_argument("--policy", default="p95", choices=["p50", "p90", "p95"])
    p.add_argument("--top", type=int, default=15,
                   help="show top-N over/under windows")
    p.add_argument("--out", default=None,
                   help="optional path to dump filtered per-window CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wf = WORKFLOWS[args.app]
    src = Path(
        f"data/entry_forecasts/multiapp_{args.app}_{args.window_sec}s_{args.family}"
    ) / FAMILY_FILE[args.family].format(wf=wf)
    if not src.exists():
        raise SystemExit(f"detail file not found: {src}")
    df = pd.read_csv(src)
    if "window" not in df.columns and "target_window" in df.columns:
        df = df.rename(columns={"target_window": "window"})
    sub = df[(df["method"] == args.method) & (df["policy"] == args.policy)].copy()
    if sub.empty:
        methods = sorted(df["method"].unique())
        raise SystemExit(
            f"no rows for method={args.method} policy={args.policy} in {src}\n"
            f"available methods: {methods}"
        )
    sub = sub[["window", "actual_count", "forecast_count",
               "allocated_count", "under_count", "over_count"]].copy()
    sub = sub.sort_values("window").reset_index(drop=True)
    sub["status"] = sub.apply(
        lambda r: "under" if r["under_count"] > 0
        else ("over" if r["over_count"] > 0 else "exact"),
        axis=1,
    )

    print(f"=== {args.app} {args.window_sec}s | {args.method} | {args.policy} ===")
    print(f"source: {src}")
    print(f"total windows: {len(sub)}")
    print(f"  exact:      {(sub['status']=='exact').sum():4d}")
    print(f"  over-alloc: {(sub['status']=='over').sum():4d}  total over={int(sub['over_count'].sum())}")
    print(f"  under-alloc:{(sub['status']=='under').sum():4d}  total under={int(sub['under_count'].sum())}")
    print(f"  actual_total={int(sub['actual_count'].sum())}  allocated_total={int(sub['allocated_count'].sum())}")

    over = sub[sub["over_count"] > 0].sort_values("over_count", ascending=False).head(args.top)
    print(f"\n-- top {args.top} OVER-allocated windows (allocated > actual) --")
    print(over.to_string(index=False))

    under = sub[sub["under_count"] > 0].sort_values("under_count", ascending=False).head(args.top)
    print(f"\n-- top {args.top} UNDER-allocated windows (allocated < actual) --")
    print(under.to_string(index=False))

    print(f"\n-- top {args.top} windows by ACTUAL count (peak windows) --")
    peaks = sub.sort_values("actual_count", ascending=False).head(args.top)
    print(peaks.to_string(index=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_path, index=False)
        print(f"\nwrote per-window CSV to {out_path}")


if __name__ == "__main__":
    main()
