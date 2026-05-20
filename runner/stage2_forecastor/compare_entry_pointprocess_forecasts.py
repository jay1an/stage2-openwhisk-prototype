"""Point-process entry forecasters: homogeneous Poisson and exp-kernel Hawkes.

These are the "principled" replacements for the patch-style hurdle / burst-aware
methods. Both fit a continuous-time intensity to training entry timestamps and
compute window counts as the integrated intensity, with quantiles drawn from
the resulting Poisson approximation.

For Hawkes-exp:
  lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i))
  MLE on training timestamps; predict each test window by integrating lambda
  over (window_start, window_end) given all events that fall before
  window_start (one-step-ahead causal conditioning).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .compare_entry_ml_forecasts import alloc_count, build_entry_counts, summarize
from .compare_stage_forecasts import load_split, resolve_window_ms
from ..workflow import load_workflow


POLICIES = {"p50": 0.50, "p90": 0.90, "p95": 0.95}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Point-process entry forecasters (homogeneous Poisson, Hawkes-exp)"
    )
    p.add_argument("--trace", required=True)
    p.add_argument("--workflow-config", required=True)
    p.add_argument("--split-map", required=True)
    p.add_argument("--window-sec", type=int, default=5)
    p.add_argument("--window-ms", type=int, default=None)
    p.add_argument("--methods", default="homogeneous-poisson,hawkes-exp")
    p.add_argument("--activation-threshold", type=float, default=0.1)
    p.add_argument("--hawkes-max-iter", type=int, default=200)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--write-detail", action="store_true")
    return p.parse_args()


def hawkes_neg_log_lik(params: np.ndarray, t: np.ndarray, T_obs: float) -> float:
    mu, alpha, beta = params
    if mu <= 0.0 or alpha < 0.0 or beta <= 0.0:
        return 1e12
    n = len(t)
    if n == 0:
        return mu * T_obs
    A = 0.0
    log_lik = 0.0
    for i in range(n):
        if i > 0:
            A = np.exp(-beta * (t[i] - t[i - 1])) * (A + 1.0)
        intensity = mu + alpha * A
        if intensity <= 0.0:
            return 1e12
        log_lik += np.log(intensity)
    compensator = mu * T_obs + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T_obs - t)))
    return -(log_lik - compensator)


def fit_hawkes_exp(t: np.ndarray, T_obs: float, max_iter: int) -> tuple[float, float, float] | None:
    n = len(t)
    if n < 20:
        return None
    diffs = np.diff(t)
    median_iat = float(np.median(diffs)) if len(diffs) > 0 else max(T_obs / max(n, 1), 1.0)
    inits = [
        (n / (2.0 * T_obs), n / (8.0 * T_obs), 1.0 / max(median_iat, 1e-3)),
        (n / (4.0 * T_obs), n / (4.0 * T_obs), 1.0 / max(median_iat, 1e-3) * 0.5),
        (n / (3.0 * T_obs), n / (10.0 * T_obs), 2.0 / max(median_iat, 1e-3)),
    ]
    best = None
    best_nll = np.inf
    for init in inits:
        try:
            res = minimize(
                hawkes_neg_log_lik,
                np.array(init, dtype=float),
                args=(t, T_obs),
                method="L-BFGS-B",
                bounds=[(1e-9, None), (0.0, None), (1e-6, None)],
                options={"maxiter": max_iter, "ftol": 1e-6},
            )
            if res.success and res.fun < best_nll:
                best_nll = float(res.fun)
                best = tuple(float(x) for x in res.x)
        except Exception:
            continue
    return best


def hawkes_window_integral(
    mu: float, alpha: float, beta: float, history: np.ndarray, a: float, b: float
) -> float:
    base = mu * (b - a)
    if len(history) == 0:
        return base
    h = history[history < b]
    if len(h) == 0:
        return base
    low = np.maximum(a, h)
    excit = float(np.sum(np.exp(-beta * (low - h)) - np.exp(-beta * (b - h)))) * (alpha / beta)
    return base + excit


def poisson_quantiles(lmbda: float) -> dict[str, float]:
    lmbda = max(0.0, float(lmbda))
    return {policy: float(poisson.ppf(q, mu=lmbda)) for policy, q in POLICIES.items()}


def build_forecast_homogeneous_poisson(
    train_times_ms: np.ndarray,
    T_train_ms: float,
    test_windows: list[tuple[int, int, int, int]],
) -> pd.DataFrame:
    n_train = len(train_times_ms)
    mu_per_ms = n_train / max(T_train_ms, 1.0)
    rows = []
    for window, win_start_ms, win_end_ms, actual in test_windows:
        lmbda = mu_per_ms * (win_end_ms - win_start_ms)
        q = poisson_quantiles(lmbda)
        rows.append(
            {
                "window": window,
                "actual_count": actual,
                "forecast_count": lmbda,
                "p50_count": q["p50"],
                "p90_count": q["p90"],
                "p95_count": q["p95"],
                "method": "homogeneous-poisson",
            }
        )
    return pd.DataFrame(rows)


def build_forecast_hawkes_exp(
    train_times_ms: np.ndarray,
    T_train_ms: float,
    full_times_ms: np.ndarray,
    test_windows: list[tuple[int, int, int, int]],
    hawkes_max_iter: int,
) -> tuple[pd.DataFrame, dict | None]:
    params = fit_hawkes_exp(train_times_ms, T_train_ms, hawkes_max_iter)
    if params is None:
        return pd.DataFrame(), None
    mu, alpha, beta = params
    rows = []
    for window, win_start_ms, win_end_ms, actual in test_windows:
        hist = full_times_ms[full_times_ms < win_start_ms]
        lmbda = hawkes_window_integral(mu, alpha, beta, hist, win_start_ms, win_end_ms)
        q = poisson_quantiles(lmbda)
        rows.append(
            {
                "window": window,
                "actual_count": actual,
                "forecast_count": lmbda,
                "p50_count": q["p50"],
                "p90_count": q["p90"],
                "p95_count": q["p95"],
                "method": "hawkes-exp",
            }
        )
    fit_info = {"mu_per_ms": mu, "alpha": alpha, "beta_per_ms": beta, "branching_ratio": alpha / beta if beta > 0 else None}
    return pd.DataFrame(rows), fit_info


def build_detail(
    forecast: pd.DataFrame, workflow_name: str, activation_threshold: float
) -> pd.DataFrame:
    rows = []
    method = str(forecast["method"].iloc[0])
    for _, row in forecast.iterrows():
        actual = int(row["actual_count"])
        for policy in POLICIES:
            forecast_count = float(row[f"{policy}_count"])
            allocated = alloc_count(forecast_count, activation_threshold)
            rows.append(
                {
                    "workflow_name": workflow_name,
                    "method": method,
                    "policy": policy,
                    "window": int(row["window"]),
                    "actual_count": actual,
                    "forecast_count": forecast_count,
                    "allocated_count": allocated,
                    "under_count": max(0, actual - allocated),
                    "over_count": max(0, allocated - actual),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    window_ms = resolve_window_ms(args)
    workflow = load_workflow(args.workflow_config)
    workflow_name = workflow.workflow_name
    trace = pd.read_csv(args.trace)

    _, _, split_map = load_split(
        trace, workflow_name, args.split_map, train_ratio=0.5, split_strategy="time"
    )
    split_map["entry_window"] = (split_map["entry_ts_ms"] // window_ms).astype(int)
    train_end_window = int(split_map["split_cutoff_ms"].dropna().iloc[0] // window_ms)
    eval_start_window = train_end_window + 1

    entry_rows = trace[
        (trace["workflow_name"] == workflow_name)
        & (trace["stage_name"] == "__entry__")
        & (trace["status"] == "ok")
    ][["request_id", "entry_ts_ms"]].drop_duplicates().sort_values("entry_ts_ms").reset_index(drop=True)
    all_times = entry_rows["entry_ts_ms"].astype("int64").values
    t0_ms = int(all_times[0])
    rel_times = all_times - t0_ms

    train_cutoff_rel_ms = int(split_map["split_cutoff_ms"].dropna().iloc[0]) - t0_ms
    train_times = rel_times[rel_times < train_cutoff_rel_ms].astype(float)
    T_train_ms = float(train_cutoff_rel_ms)

    counts = build_entry_counts(trace, workflow_name, window_ms)
    test_idx = counts.index[(counts.index >= eval_start_window)]
    test_idx = test_idx[test_idx <= int(split_map[split_map["split"] == "test"]["entry_window"].max())]
    test_windows: list[tuple[int, int, int, int]] = []
    for w in test_idx:
        win_start_ms = int(w) * window_ms - t0_ms
        win_end_ms = (int(w) + 1) * window_ms - t0_ms
        actual = int(counts.loc[int(w)])
        test_windows.append((int(w), int(win_start_ms), int(win_end_ms), int(actual)))

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    forecasts = []
    fit_info_all: dict = {}
    if "homogeneous-poisson" in methods:
        forecasts.append(
            build_forecast_homogeneous_poisson(train_times, T_train_ms, test_windows)
        )
    if "hawkes-exp" in methods:
        df, info = build_forecast_hawkes_exp(
            train_times,
            T_train_ms,
            rel_times.astype(float),
            test_windows,
            args.hawkes_max_iter,
        )
        if not df.empty:
            forecasts.append(df)
            fit_info_all["hawkes-exp"] = info
    if not forecasts:
        raise SystemExit("no forecasts produced")

    details = [build_detail(f, workflow_name, args.activation_threshold) for f in forecasts]
    detail = pd.concat(details, ignore_index=True)
    summary = summarize(detail, window_ms)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{workflow_name}_entry_pointprocess_compare_summary.csv"
    detail_path = out_dir / f"{workflow_name}_entry_pointprocess_compare_detail.csv"
    meta_path = out_dir / f"{workflow_name}_entry_pointprocess_compare_metadata.json"
    summary.to_csv(summary_path, index=False)
    if args.write_detail:
        detail.to_csv(detail_path, index=False)
    meta = {
        "workflow_name": workflow_name,
        "trace": args.trace,
        "workflow_config": args.workflow_config,
        "split_map": args.split_map,
        "window_ms": window_ms,
        "methods": methods,
        "train_windows": int(train_end_window - int(counts.index.min()) + 1),
        "test_windows": int(len(test_windows)),
        "train_events": int(len(train_times)),
        "hawkes_fit": fit_info_all.get("hawkes-exp"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
