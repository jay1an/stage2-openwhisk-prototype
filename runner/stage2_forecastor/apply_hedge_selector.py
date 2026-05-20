"""Hedge / multiplicative-weights online expert selector.

Replaces the hysteresis-and-cooldown patchwork in
`online_adaptive_forecast_selector.py` with the textbook Hedge algorithm
(Cesa-Bianchi & Lugosi 2006).

  w_{i,t+1} = w_{i,t} * exp(-eta * L_{i,t})
  L_{i,t}   = under_cost * max(0, y_t - q_{i,t})
            + over_cost  * max(0, q_{i,t} - y_t)

At each window t and policy p, choose i_t = argmax w_{i,t} (deterministic).
No hysteresis, no cooldown, no risk-budget fallback - the regret bound
O(sqrt(T log K)) is what the algorithm gives you.

Input  : one or more forecast detail CSVs (long format with columns
         workflow_name, method, policy, window, actual_count, forecast_count,
         allocated_count, under_count, over_count).
Output : a single detail CSV under method="hedge".
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hedge online expert selector")
    p.add_argument("--detail", nargs="+", required=True, help="per-expert detail CSVs")
    p.add_argument("--out", required=True)
    p.add_argument("--policies", default="p50,p90,p95")
    p.add_argument("--under-cost", type=float, default=10.0)
    p.add_argument("--over-cost", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=None,
                   help="Hedge learning rate; default sqrt(8 log K / T)")
    p.add_argument("--sample", action="store_true",
                   help="randomized Hedge instead of argmax")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--activation-threshold", type=float, default=0.1)
    return p.parse_args()


def alloc_count(x: float, t: float) -> int:
    return 0 if x <= t else int(np.ceil(x))


def load_bank(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "window" not in df.columns and "target_window" in df.columns:
            df = df.rename(columns={"target_window": "window"})
        df["__source__"] = str(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pinball(forecast: float, actual: float, under: float, over: float) -> float:
    diff = float(actual) - float(forecast)
    return under * max(0.0, diff) + over * max(0.0, -diff)


def run_hedge_for_policy(bank: pd.DataFrame, policy: str,
                        args: argparse.Namespace) -> pd.DataFrame:
    sub = bank[bank["policy"] == policy].copy()
    if sub.empty:
        return pd.DataFrame()
    workflow = str(sub["workflow_name"].iloc[0])
    pivot = sub.pivot_table(index="window", columns="method", values="forecast_count",
                            aggfunc="first").sort_index()
    actual_map = sub.drop_duplicates("window").set_index("window")["actual_count"]
    experts = list(pivot.columns)
    K = len(experts)
    T = len(pivot.index)
    eta = args.eta if args.eta is not None else math.sqrt(8.0 * math.log(max(K, 2)) / max(T, 1))
    rng = np.random.default_rng(args.seed)
    weights = np.ones(K, dtype=float) / K
    rows = []
    seen = 0
    for w in pivot.index:
        forecasts = pivot.loc[w].values.astype(float)
        active = ~np.isnan(forecasts)
        if not active.any():
            continue
        active_w = weights * active
        if active_w.sum() <= 0:
            active_w = active.astype(float)
        probs = active_w / active_w.sum()
        if args.sample:
            chosen = int(rng.choice(K, p=probs))
        else:
            chosen = int(np.argmax(np.where(active, weights, -np.inf)))
        chosen_name = experts[chosen]
        chosen_forecast = float(forecasts[chosen])
        actual = float(actual_map.get(w, 0.0))
        allocated = alloc_count(chosen_forecast, args.activation_threshold)
        rows.append({
            "workflow_name": workflow,
            "method": "hedge",
            "policy": policy,
            "window": int(w),
            "actual_count": int(actual),
            "forecast_count": chosen_forecast,
            "allocated_count": allocated,
            "under_count": max(0, int(actual) - allocated),
            "over_count": max(0, allocated - int(actual)),
            "selected_expert": chosen_name,
        })
        seen += 1
        if seen <= args.warmup:
            continue
        losses = np.zeros(K, dtype=float)
        for i in range(K):
            if not active[i]:
                continue
            losses[i] = pinball(forecasts[i], actual, args.under_cost, args.over_cost)
        max_loss = float(losses.max()) if losses.size else 1.0
        if max_loss > 0:
            losses = losses / max_loss
        weights = weights * np.exp(-eta * losses)
        s = weights.sum()
        weights = weights / s if s > 0 else np.ones(K) / K
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.detail]
    bank = load_bank(paths)
    required = {"workflow_name", "method", "policy", "window", "actual_count", "forecast_count"}
    missing = required - set(bank.columns)
    if missing:
        raise SystemExit(f"detail missing columns: {missing}")
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    out_frames = [run_hedge_for_policy(bank, pol, args) for pol in policies]
    out = pd.concat([f for f in out_frames if not f.empty], ignore_index=True)
    if out.empty:
        raise SystemExit("no forecasts produced")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ["workflow_name", "method", "policy", "window", "actual_count",
                 "forecast_count", "allocated_count", "under_count", "over_count"]
    out[keep_cols].to_csv(out_path, index=False)
    usage = out.groupby(["policy", "selected_expert"]).size().unstack(fill_value=0)
    usage.to_csv(out_path.with_suffix(".expert_usage.csv"))
    meta = {
        "input_detail": [str(p) for p in paths],
        "policies": policies,
        "under_cost": args.under_cost,
        "over_cost": args.over_cost,
        "eta": args.eta,
        "sample": args.sample,
        "warmup": args.warmup,
        "seed": args.seed,
        "rows": int(len(out)),
        "windows": int(out["window"].nunique()),
        "experts": sorted(out["selected_expert"].dropna().unique().tolist()),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_path}  rows={len(out)}")
    print("\nexpert usage:")
    print(usage.to_string())


if __name__ == "__main__":
    main()
