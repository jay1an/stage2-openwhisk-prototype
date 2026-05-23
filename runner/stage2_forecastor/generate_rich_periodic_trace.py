"""Generate a rich-periodic synthetic entry trace for forecast evaluation.

Produces a per-second arrival rate with strong multi-scale seasonality
(60-min main cycle + 10-min harmonic) plus mild Gaussian rate jitter, then
samples requests via inhomogeneous Poisson thinning. The result has a
clear repeating shape with realistic noise, so seasonal forecasters
(ETS / Theta / MSTL) can show real lift.

Output format matches `data/azure_multiapp/<label>/entry_trace_<label>.csv`
so it plugs straight into the existing compare_entry_* tooling.
"""

from __future__ import annotations

import argparse
import math
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


TRACE_COLUMNS = [
    "workflow_name", "request_id", "stage_name", "parent_stages",
    "entry_ts_ms", "dispatch_start_ms", "dispatch_end_ms",
    "dispatch_latency_ms", "action_start_ns", "action_end_ns",
    "action_duration_ms", "platform_overhead_ms", "container_id",
    "cold_like", "status", "error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="rich_periodic")
    p.add_argument("--workflow-name", default="civic_alert_flow")
    p.add_argument("--duration-sec", type=int, default=14400, help="4h by default")
    p.add_argument("--base-rate", type=float, default=2.0, help="mean requests per second")
    p.add_argument("--amp-main", type=float, default=1.6, help="amplitude of 60-min cycle (multiplicative)")
    p.add_argument("--amp-sub", type=float, default=0.5, help="amplitude of 10-min harmonic")
    p.add_argument("--noise-std", type=float, default=0.12, help="std of multiplicative log-Gaussian noise")
    p.add_argument("--main-period-sec", type=int, default=3600)
    p.add_argument("--sub-period-sec", type=int, default=600)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--base-entry-ts-ms", type=int, default=2_000_000_000_000)
    p.add_argument("--out-root", default="data/azure_multiapp")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    duration_ms = args.duration_sec * 1000
    n_sec = args.duration_sec
    t = np.arange(n_sec, dtype=float)
    rate = args.base_rate * (
        1.0
        + args.amp_main * np.sin(2 * math.pi * t / args.main_period_sec)
        + args.amp_sub * np.sin(2 * math.pi * t / args.sub_period_sec)
    )
    # multiplicative log-gaussian jitter, clipped to be positive
    jitter = np.exp(rng.normal(0.0, args.noise_std, size=n_sec))
    rate = np.maximum(0.02, rate * jitter)

    # inhomogeneous poisson via per-second binning
    counts_per_sec = rng.poisson(rate)
    request_ms = []
    for sec, n in enumerate(counts_per_sec):
        if n == 0:
            continue
        # spread n requests uniformly inside that second
        offsets = rng.uniform(0.0, 1000.0, size=int(n))
        for off in offsets:
            request_ms.append(args.base_entry_ts_ms + sec * 1000 + int(off))
    request_ms.sort()

    rows = []
    for ts in request_ms:
        rows.append({
            "workflow_name": args.workflow_name,
            "request_id": str(uuid.uuid4()),
            "stage_name": "__entry__",
            "parent_stages": "",
            "entry_ts_ms": int(ts),
            "dispatch_start_ms": int(ts),
            "dispatch_end_ms": int(ts),
            "dispatch_latency_ms": 0,
            "action_start_ns": "",
            "action_end_ns": "",
            "action_duration_ms": "",
            "platform_overhead_ms": "",
            "container_id": "",
            "cold_like": "",
            "status": "ok",
            "error": "",
        })

    df = pd.DataFrame(rows, columns=TRACE_COLUMNS)
    out_dir = Path(args.out_root) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"entry_trace_{args.label}.csv"
    df.to_csv(out_path, index=False)

    # 5s window count stats for sanity
    sec_idx = (df["entry_ts_ms"] - args.base_entry_ts_ms) // 5000
    counts = sec_idx.value_counts().sort_index()
    n_w = int(sec_idx.max()) + 1
    full = np.zeros(n_w, dtype=int)
    full[counts.index.astype(int)] = counts.values
    print(f"wrote {out_path}  rows={len(df)}  span={args.duration_sec}s")
    print(f"  5s windows: n={n_w} mean={full.mean():.2f} max={full.max()} p90={np.quantile(full,0.9):.1f} "
          f"p99={np.quantile(full,0.99):.1f} zero_frac={(full==0).mean():.2%}")

    # also write a 50:50 time split for downstream tools
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    cutoff = args.base_entry_ts_ms + duration_ms // 2
    sm = df[["request_id", "entry_ts_ms"]].copy()
    sm["split"] = np.where(sm["entry_ts_ms"] <= cutoff, "train", "test")
    sm["split_strategy"] = "time"
    sm["split_cutoff_ms"] = cutoff
    sm.to_csv(splits_dir / f"entry_{args.label}_split.csv", index=False)
    print(f"  split: train={(sm['split']=='train').sum()} test={(sm['split']=='test').sum()}")


if __name__ == "__main__":
    main()
