"""Aggregate the multiapp sweep summary CSVs into a unified comparison table.

This version augments the per-family `*_summary.csv` with cost-aware metrics
recomputed from the `*_detail.csv` files:

  - weighted_cost     : c_under * sum(under) + c_over * sum(over)
                        (defaults c_under=10, c_over=1 — asymmetric SLO cost)
  - pinball_loss_mean : sum_t alpha*max(0,y-q) + (1-alpha)*max(0,q-y) / T
                        evaluated at the policy's nominal alpha
  - tail_p95_abs_err  : 95th percentile of |y - q| (tail risk, not mean)
  - mae_active        : MAE restricted to windows with actual_count > 0
  - mae_peak10        : MAE on the top-10% windows by actual_count
  - active_coverage   : fraction of active windows with actual <= allocated

It also separates the SMIless (`smiless-*`) and IceBreaker/FIP (`fip-*`)
families into a `baselines_summary.csv` so they can be used as published
reference points later — they are NOT shown in the main leaderboard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


APPS = ["periodic_dense", "periodic_sparse", "bursty_dense", "bursty_sparse", "drift"]
WORKFLOWS = {
    "periodic_dense": "sebs_video",
    "periodic_sparse": "civic_alert_flow",
    "bursty_dense": "sebs_video",
    "bursty_sparse": "civic_alert_flow",
    "drift": "spoken_dialog_flow",
}
WINDOWS = [5, 2]

# family -> (subdir suffix, summary file pattern, detail file pattern)
FAMILY_DIRS = {
    "rolling":      ("h1",              "{wf}_rolling_entry_compare_summary.csv",      "{wf}_rolling_entry_compare_detail.csv"),
    "classical":    ("classical",       "{wf}_entry_classical_compare_summary.csv",    "{wf}_entry_classical_compare_detail.csv"),
    "ml":           ("ml",              "{wf}_entry_ml_compare_summary.csv",           "{wf}_entry_ml_compare_detail.csv"),
    "twostage":     ("twostage",        "{wf}_entry_twostage_compare_summary.csv",     "{wf}_entry_twostage_compare_detail.csv"),
    "histogram":    ("histogram",       "{wf}_histogram_entry_compare_summary.csv",    "{wf}_histogram_entry_compare_detail.csv"),
    "pointprocess": ("pointprocess",    "{wf}_entry_pointprocess_compare_summary.csv", "{wf}_entry_pointprocess_compare_detail.csv"),
    "lstm":         ("lstm",            "{wf}_entry_lstm_compare_summary.csv",         "{wf}_entry_lstm_compare_detail.csv"),
    "gru":          ("gru",             "{wf}_entry_lstm_compare_summary.csv",         "{wf}_entry_lstm_compare_detail.csv"),
    "cp_classical": ("cp_classical",    "{wf}_entry_cp_aci_classical_summary.csv",     "{wf}_entry_cp_aci_classical_detail.csv"),
    "cp_ml":        ("cp_ml",           "{wf}_entry_cp_aci_ml_summary.csv",            "{wf}_entry_cp_aci_ml_detail.csv"),
    "cp_twostage":  ("cp_twostage",     "{wf}_entry_cp_aci_twostage_summary.csv",      "{wf}_entry_cp_aci_twostage_detail.csv"),
    "cp_pp":        ("cp_pointprocess", "{wf}_entry_cp_aci_pointprocess_summary.csv",  "{wf}_entry_cp_aci_pointprocess_detail.csv"),
    "hedge":        ("hedge",           "{wf}_entry_hedge_compare_summary.csv",        "{wf}_entry_hedge_compare_detail.csv"),
}

# Methods/families that are published baselines — we evaluate them with the
# same pipeline but report them separately so they don't pollute "our" leaderboard.
BASELINE_FAMILIES = {"lstm", "gru"}                    # SMIless paper
BASELINE_METHOD_PREFIXES = ("smiless-", "fip-")         # SMIless / IceBreaker

# Asymmetric cost: one dropped invocation hurts much more than one idle replica.
COST_UNDER = 10.0
COST_OVER = 1.0

POLICY_ALPHA = {"p50": 0.50, "p90": 0.90, "p95": 0.95}

SUMMARY_COLS = [
    "workflow_name", "method", "policy",
    "mae", "rmse",
    "actual_total", "allocated_replica_windows", "over_total", "under_total",
    "demand_coverage_rate", "allocation_utilization", "over_allocation_ratio",
    "active_coverage_rate", "max_actual", "max_allocated",
]


def cost_metrics(detail: pd.DataFrame, policy: str) -> dict[str, float]:
    """Compute cost-aware metrics from a detail slice (one (workflow, method, policy))."""
    actual = detail["actual_count"].astype(float).to_numpy()
    forecast = detail["forecast_count"].astype(float).to_numpy()
    allocated = detail["allocated_count"].astype(float).to_numpy() if "allocated_count" in detail.columns else forecast
    under = detail["under_count"].astype(float).to_numpy() if "under_count" in detail.columns else np.maximum(0.0, actual - allocated)
    over = detail["over_count"].astype(float).to_numpy() if "over_count" in detail.columns else np.maximum(0.0, allocated - actual)
    abs_err = np.abs(actual - forecast)
    alpha = POLICY_ALPHA.get(policy, 0.95)
    # pinball loss against forecast_count (the quantile prediction, before ceil)
    diff = actual - forecast
    pinball = alpha * np.maximum(0.0, diff) + (1.0 - alpha) * np.maximum(0.0, -diff)
    active_mask = actual > 0
    n = len(actual)
    if n == 0:
        return {}
    # peak-10% by actual size
    if active_mask.any():
        thr = float(np.quantile(actual[active_mask], 0.9))
    else:
        thr = float("inf")
    peak_mask = actual >= thr if np.isfinite(thr) else np.zeros(n, dtype=bool)
    weighted_cost = COST_UNDER * float(under.sum()) + COST_OVER * float(over.sum())
    return {
        "weighted_cost": weighted_cost,
        "weighted_cost_per_window": weighted_cost / n,
        "pinball_loss_mean": float(pinball.mean()),
        "tail_p95_abs_err": float(np.quantile(abs_err, 0.95)) if n else 0.0,
        "mae_active": float(abs_err[active_mask].mean()) if active_mask.any() else 0.0,
        "mae_peak10": float(abs_err[peak_mask].mean()) if peak_mask.any() else 0.0,
        "windows_total": int(n),
        "windows_active": int(active_mask.sum()),
        "windows_peak10": int(peak_mask.sum()),
    }


def load_family(root: Path, app: str, win: int, family: str) -> pd.DataFrame:
    suffix, sfile, dfile = FAMILY_DIRS[family]
    wf = WORKFLOWS[app]
    d = root / f"multiapp_{app}_{win}s_{suffix}"
    summary_path = d / sfile.format(wf=wf)
    detail_path = d / dfile.format(wf=wf)
    if not summary_path.exists():
        return pd.DataFrame()
    s = pd.read_csv(summary_path)
    keep = [c for c in SUMMARY_COLS if c in s.columns]
    s = s[keep].copy()
    s["app"] = app
    s["window_sec"] = win
    s["family"] = family

    if detail_path.exists():
        det = pd.read_csv(detail_path)
        if "window" not in det.columns and "target_window" in det.columns:
            det = det.rename(columns={"target_window": "window"})
        # for hedge there is no workflow_name filter issue, but other files
        # could carry multiple workflows; restrict to ours.
        if "workflow_name" in det.columns:
            det = det[det["workflow_name"] == wf]
        extra_rows = []
        for (m, p), grp in det.groupby(["method", "policy"]):
            extra_rows.append({"method": m, "policy": p, **cost_metrics(grp, str(p))})
        if extra_rows:
            extra = pd.DataFrame(extra_rows)
            s = s.merge(extra, on=["method", "policy"], how="left")
    return s


def is_baseline_row(row: pd.Series) -> bool:
    if row["family"] in BASELINE_FAMILIES:
        return True
    m = str(row["method"])
    return any(m.startswith(pref) for pref in BASELINE_METHOD_PREFIXES)


def write_pivots(df: pd.DataFrame, out_root: Path, tag: str) -> None:
    p95 = df[df["policy"] == "p95"]
    win5 = p95[p95["window_sec"] == 5]
    if win5.empty:
        return
    for metric, lower_better in [
        ("weighted_cost", True),
        ("mae", True),
        ("mae_peak10", True),
        ("pinball_loss_mean", True),
        ("demand_coverage_rate", False),
        ("over_allocation_ratio", True),
        ("tail_p95_abs_err", True),
    ]:
        if metric not in win5.columns:
            continue
        pivot = win5.pivot_table(index=["method", "family"], columns="app",
                                 values=metric, aggfunc="first")
        if metric in ("weighted_cost",):
            pivot = pivot.round(0)
        elif metric in ("demand_coverage_rate", "over_allocation_ratio"):
            pivot = pivot.round(3)
        else:
            pivot = pivot.round(2)
        pivot.to_csv(out_root / f"pivot_{tag}_{metric}_5s_p95.csv")


def write_leaderboards(df: pd.DataFrame, out_root: Path, tag: str) -> None:
    cols = ["method", "family", "weighted_cost", "demand_coverage_rate",
            "pinball_loss_mean", "mae_peak10", "tail_p95_abs_err",
            "over_allocation_ratio", "mae", "max_actual", "max_allocated"]
    p95 = df[df["policy"] == "p95"]
    for app in APPS:
        for win in WINDOWS:
            sub = p95[(p95["app"] == app) & (p95["window_sec"] == win)].copy()
            if sub.empty:
                continue
            sort_key = "weighted_cost" if "weighted_cost" in sub.columns else "mae"
            sub = sub.sort_values(sort_key)
            keep = [c for c in cols if c in sub.columns]
            (out_root / f"leaderboard_{tag}_{app}_{win}s_p95.csv").write_text(
                sub[keep].to_csv(index=False), encoding="utf-8"
            )


def main() -> None:
    root = Path("data/entry_forecasts")
    out_root = Path("reports/multiapp_full_sweep")
    out_root.mkdir(parents=True, exist_ok=True)

    frames = []
    for app in APPS:
        for win in WINDOWS:
            for family in FAMILY_DIRS:
                df = load_family(root, app, win, family)
                if not df.empty:
                    frames.append(df)
    if not frames:
        raise SystemExit("no summary files found")
    big = pd.concat(frames, ignore_index=True, sort=False)
    big.to_csv(out_root / "all_methods_summary.csv", index=False)

    big["is_baseline"] = big.apply(is_baseline_row, axis=1)
    ours = big[~big["is_baseline"]].copy()
    base = big[big["is_baseline"]].copy()

    ours.to_csv(out_root / "ours_summary.csv", index=False)
    base.to_csv(out_root / "baselines_summary.csv", index=False)

    ours[ours["policy"] == "p95"].to_csv(out_root / "ours_p95.csv", index=False)
    base[base["policy"] == "p95"].to_csv(out_root / "baselines_p95.csv", index=False)

    write_pivots(ours, out_root, "ours")
    write_pivots(base, out_root, "baselines")
    write_leaderboards(ours, out_root, "ours")
    write_leaderboards(base, out_root, "baselines")

    print(f"all rows: {len(big)}   ours: {len(ours)}   baselines: {len(base)}")
    print(f"ours families  : {sorted(ours['family'].unique())}")
    print(f"baseline families: {sorted(base['family'].unique())}")
    print(f"ours methods   : {len(ours['method'].unique())}")
    print(f"baseline methods: {sorted(base['method'].unique())}")

    win5_p95 = ours[(ours["policy"] == "p95") & (ours["window_sec"] == 5)]
    if "weighted_cost" in win5_p95.columns:
        pivot = win5_p95.pivot_table(index=["method", "family"], columns="app",
                                     values="weighted_cost", aggfunc="first").round(0)
        print("\n== p95 / 5s WEIGHTED COST pivot (under x10 + over x1, lower better) ==")
        print(pivot.to_string())

    if "demand_coverage_rate" in win5_p95.columns:
        pivot = win5_p95.pivot_table(index=["method", "family"], columns="app",
                                     values="demand_coverage_rate", aggfunc="first").round(3)
        print("\n== p95 / 5s DEMAND_COVERAGE_RATE pivot (higher better, target >=0.95) ==")
        print(pivot.to_string())


if __name__ == "__main__":
    main()
