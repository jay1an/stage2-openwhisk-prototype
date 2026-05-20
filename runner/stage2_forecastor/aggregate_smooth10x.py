"""Aggregate the smooth10x robustness-check sweep.

Reads detail CSVs from `data/entry_forecasts/smooth10x_*/` and produces a
cost-aware comparison alongside the original compression=30 results so the
report can show the robustness-check head-to-head.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


APPS = ["bursty_dense", "bursty_sparse"]
WORKFLOWS = {"bursty_dense": "sebs_video", "bursty_sparse": "civic_alert_flow"}
WINDOWS = [5, 2]
SUFFIXES = ["classical", "pointprocess", "cp_classical", "cp_pp", "hedge"]
DETAIL_PATTERN = {
    "classical":    "{wf}_entry_classical_compare_detail.csv",
    "pointprocess": "{wf}_entry_pointprocess_compare_detail.csv",
    "cp_classical": "{wf}_entry_cp_aci_classical_detail.csv",
    "cp_pp":        "{wf}_entry_cp_aci_pp_detail.csv",
    "hedge":        "{wf}_entry_hedge_compare_detail.csv",
}
COST_UNDER = 10.0
COST_OVER = 1.0
POLICY_ALPHA = {"p50": 0.50, "p90": 0.90, "p95": 0.95}


def cost_row(det: pd.DataFrame, app: str, win: int, family: str,
             method: str, policy: str) -> dict:
    actual = det["actual_count"].astype(float).to_numpy()
    forecast = det["forecast_count"].astype(float).to_numpy()
    allocated = det["allocated_count"].astype(float).to_numpy() if "allocated_count" in det.columns else forecast
    under = det["under_count"].astype(float).to_numpy() if "under_count" in det.columns else np.maximum(0, actual - allocated)
    over = det["over_count"].astype(float).to_numpy() if "over_count" in det.columns else np.maximum(0, allocated - actual)
    abs_err = np.abs(actual - forecast)
    alpha = POLICY_ALPHA.get(policy, 0.95)
    diff = actual - forecast
    pinball = alpha * np.maximum(0.0, diff) + (1.0 - alpha) * np.maximum(0.0, -diff)
    active = actual > 0
    if active.any():
        thr = float(np.quantile(actual[active], 0.9))
        peak = actual >= thr
    else:
        peak = np.zeros(len(actual), dtype=bool)
    actual_total = float(actual.sum())
    under_total = float(under.sum())
    over_total = float(over.sum())
    return {
        "app": app, "window_sec": win, "family": family,
        "method": method, "policy": policy,
        "weighted_cost": COST_UNDER * under_total + COST_OVER * over_total,
        "demand_coverage_rate": 1.0 - under_total / max(actual_total, 1e-9),
        "over_allocation_ratio": over_total / max(float(allocated.sum()), 1e-9),
        "mae": float(abs_err.mean()),
        "mae_peak10": float(abs_err[peak].mean()) if peak.any() else 0.0,
        "pinball_loss_mean": float(pinball.mean()),
        "tail_p95_abs_err": float(np.quantile(abs_err, 0.95)),
        "actual_total": int(actual_total), "under_total": int(under_total),
        "over_total": int(over_total), "windows_total": int(len(actual)),
    }


def main() -> None:
    root = Path("data/entry_forecasts")
    rows = []
    for app in APPS:
        wf = WORKFLOWS[app]
        for win in WINDOWS:
            for family in SUFFIXES:
                d = root / f"smooth10x_{app}_{win}s_{family}"
                pat = DETAIL_PATTERN[family].format(wf=wf)
                f = d / pat
                if not f.exists():
                    print(f"missing: {f}")
                    continue
                det = pd.read_csv(f)
                if "window" not in det.columns and "target_window" in det.columns:
                    det = det.rename(columns={"target_window": "window"})
                if "workflow_name" in det.columns:
                    det = det[det["workflow_name"] == wf]
                for (m, p), grp in det.groupby(["method", "policy"]):
                    rows.append(cost_row(grp, app, win, family, str(m), str(p)))
    if not rows:
        raise SystemExit("no smooth10x results found")
    df = pd.DataFrame(rows)
    out_root = Path("reports/multiapp_full_sweep")
    df.to_csv(out_root / "smooth10x_summary.csv", index=False)
    p95 = df[df["policy"] == "p95"]
    for app in APPS:
        for win in WINDOWS:
            sub = p95[(p95["app"] == app) & (p95["window_sec"] == win)].sort_values("weighted_cost")
            keep = ["method", "family", "weighted_cost", "demand_coverage_rate",
                    "mae_peak10", "pinball_loss_mean", "tail_p95_abs_err",
                    "mae", "over_allocation_ratio", "actual_total", "under_total", "over_total"]
            keep = [c for c in keep if c in sub.columns]
            (out_root / f"smooth10x_leaderboard_{app}_{win}s_p95.csv").write_text(
                sub[keep].to_csv(index=False), encoding="utf-8"
            )

    print("\n== smooth10x (compression=10) p95/5s — by weighted_cost ==")
    for app in APPS:
        sub = p95[(p95["app"] == app) & (p95["window_sec"] == 5)].sort_values("weighted_cost").head(10)
        print(f"\n--- {app} ---")
        print(sub[["method", "family", "weighted_cost", "demand_coverage_rate",
                   "mae_peak10", "mae"]].round(2).to_string(index=False))

    # quick side-by-side vs original
    ours = pd.read_csv(out_root / "ours_p95.csv")
    ours5 = ours[(ours["window_sec"] == 5) & (ours["app"].isin(APPS))]
    print("\n\n== original (compression=30) p95/5s — same methods, by weighted_cost ==")
    for app in APPS:
        sub = ours5[ours5["app"] == app]
        sub = sub[sub["method"].isin(p95["method"].unique())].sort_values("weighted_cost").head(10)
        print(f"\n--- {app} ---")
        cols = [c for c in ["method", "family", "weighted_cost", "demand_coverage_rate", "mae_peak10", "mae"] if c in sub.columns]
        print(sub[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
