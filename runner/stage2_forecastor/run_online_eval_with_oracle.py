"""Online 1-step-ahead evaluation harness for Stage-2 entry forecasting.

Walks every test window as an online origin, builds history up to that window,
runs each method's 1-step-ahead forecast, and records (actual, forecast, allocated)
per (method, policy, window). Adds reference rows:

- oracle: knows the true count for the next window; allocated = actual (best possible)
- selector: pinball-loss weighted online expert selection over a recent window
- hawkes-exp: exp-kernel Hawkes process, MLE-refit every K windows on history so far
- {base}+aci: any base method's p50 wrapped in an online ACI quantile calibrator

Outputs:
- {workflow}_online_eval_detail.csv  (all rows)
- {workflow}_online_eval_summary.csv (per-method aggregates)
- {workflow}_online_eval_metadata.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .compare_stage_forecasts import build_count_series, forecast_from_series, load_split
from ..workflow import load_workflow


POLICIES = ["p50", "p90", "p95"]
POLICY_Q = {"p50": 0.5, "p90": 0.9, "p95": 0.95}
POLICY_ALPHA = {"p50": 0.5, "p90": 0.1, "p95": 0.05}


# ---------------------------------------------------------------------------
# Hawkes-exp utilities (inlined from compare_entry_pointprocess_forecasts so the
# online harness has no cross-module wiring; works on entry timestamps in ms).
# ---------------------------------------------------------------------------


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
            A = math.exp(-beta * (t[i] - t[i - 1])) * (A + 1.0)
        intensity = mu + alpha * A
        if intensity <= 0.0:
            return 1e12
        log_lik += math.log(intensity)
    compensator = mu * T_obs + (alpha / beta) * float(np.sum(1.0 - np.exp(-beta * (T_obs - t))))
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
    """Integrate lambda(t) over [a, b] given event times strictly before b."""
    base = mu * (b - a)
    if len(history) == 0:
        return base
    h = history[history < b]
    if len(h) == 0:
        return base
    low = np.maximum(a, h)
    excit = float(np.sum(np.exp(-beta * (low - h)) - np.exp(-beta * (b - h)))) * (alpha / beta)
    return base + excit


# ---------------------------------------------------------------------------
# ACI (Adaptive Conformal Interval) online calibrator.
# Wraps a base point forecast; emits a quantile upper bound for one policy.
# Adapts alpha_t after each observed actual so empirical miss rate -> alpha_target.
# ---------------------------------------------------------------------------


class ACIState:
    """Per-method per-policy online ACI state."""

    def __init__(self, alpha_target: float, gamma: float = 0.05, warmup: int = 12) -> None:
        self.alpha_target = float(alpha_target)
        self.alpha_t = float(alpha_target)
        self.gamma = float(gamma)
        self.warmup = int(warmup)
        self.residuals: list[float] = []

    def upper(self, point: float) -> float:
        if len(self.residuals) <= self.warmup:
            return float(point)
        level = float(np.clip(1.0 - self.alpha_t, 0.0, 1.0))
        q_hat = float(np.quantile(np.asarray(self.residuals), level))
        return float(point) + q_hat

    def update(self, point: float, actual: float) -> None:
        """Observe `actual` after emitting forecast for this window; adapt alpha_t."""
        if len(self.residuals) > self.warmup:
            upper = self.upper(point)
            miss = 1.0 if float(actual) > upper else 0.0
            self.alpha_t = float(
                np.clip(
                    self.alpha_t + self.gamma * (self.alpha_target - miss),
                    1e-4,
                    1.0 - 1e-4,
                )
            )
        self.residuals.append(max(0.0, float(actual) - float(point)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", required=True)
    p.add_argument("--workflow-config", required=True)
    p.add_argument("--split-map", required=True)
    p.add_argument("--window-ms", type=int, default=5000)
    p.add_argument(
        "--methods",
        default="ewma,burst-aware,hurdle-ewma,tsb,hazard-hurdle,burst-localized,fip-fourier",
        help="streaming methods routed through forecast_from_series",
    )
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--residual-window", type=int, default=60)
    p.add_argument("--history-window", type=int, default=30)
    p.add_argument("--burst-threshold", type=float, default=2.0)
    p.add_argument("--burst-period-windows", type=int, default=None)
    p.add_argument("--burst-width-windows", type=int, default=0)
    p.add_argument("--background-count", type=float, default=None)
    p.add_argument("--idle-zero-ratio", type=float, default=0.8)
    p.add_argument("--activation-threshold", type=float, default=0.1)
    p.add_argument(
        "--selector-window",
        type=int,
        default=48,
        help="number of recent windows used to score experts for the selector",
    )
    p.add_argument(
        "--selector-min-history",
        type=int,
        default=12,
        help="number of warmup windows before the selector can switch",
    )
    p.add_argument(
        "--enable-hawkes",
        action="store_true",
        help="add hawkes-exp method (MLE refit on history every K windows)",
    )
    p.add_argument(
        "--hawkes-refit-every",
        type=int,
        default=12,
        help="refit Hawkes params every N origin windows (1 min @ 5s windows)",
    )
    p.add_argument(
        "--hawkes-max-iter",
        type=int,
        default=80,
        help="max L-BFGS iterations per Hawkes refit",
    )
    p.add_argument(
        "--hawkes-history-windows",
        type=int,
        default=720,
        help="cap Hawkes fit history to last K windows for speed; <=0 disables",
    )
    p.add_argument(
        "--enable-aci",
        action="store_true",
        help="add {base}+aci variant for each base method (online conformal upper bound on p95)",
    )
    p.add_argument(
        "--aci-gamma",
        type=float,
        default=0.05,
        help="ACI step size for alpha_t updates",
    )
    p.add_argument(
        "--aci-warmup",
        type=int,
        default=12,
        help="ACI warmup windows before emitting calibrated upper bound",
    )
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def pinball_loss(q: float, actual: float, predicted: float) -> float:
    delta = actual - predicted
    return max(q * delta, (q - 1.0) * delta)


def allocated_for_policy(forecast_row: pd.Series, policy: str) -> int:
    alloc_col = f"alloc_{policy}_count"
    if alloc_col in forecast_row and not pd.isna(forecast_row[alloc_col]):
        return int(forecast_row[alloc_col])
    ceil_col = f"ceil_{policy}_count"
    if ceil_col in forecast_row and not pd.isna(forecast_row[ceil_col]):
        return int(forecast_row[ceil_col])
    return int(math.ceil(max(0.0, float(forecast_row[f"{policy}_count"]))))


def alloc_count(forecast_value: float, activation_threshold: float) -> int:
    if forecast_value <= activation_threshold:
        return 0
    return int(math.ceil(forecast_value))


def hawkes_forecast(
    entry_times_ms: np.ndarray,
    target_window: int,
    window_ms: int,
    cached_params: tuple[float, float, float],
) -> dict[str, float]:
    """Compute Hawkes one-step-ahead p50/p90/p95 for `target_window`.

    Uses cached MLE params; refresh outside the hot path.
    """
    mu, alpha, beta = cached_params
    win_start = target_window * window_ms
    win_end = win_start + window_ms
    history = entry_times_ms[entry_times_ms < win_start]
    lam = hawkes_window_integral(mu, alpha, beta, history, float(win_start), float(win_end))
    lam = max(0.0, float(lam))
    return {
        "p50_count": float(poisson.ppf(0.50, mu=lam)) if lam > 0 else 0.0,
        "p90_count": float(poisson.ppf(0.90, mu=lam)) if lam > 0 else 0.0,
        "p95_count": float(poisson.ppf(0.95, mu=lam)) if lam > 0 else 0.0,
        "lambda": lam,
    }


def append_method_row(
    rows: list[dict],
    workflow_name: str,
    method: str,
    policy: str,
    origin_window: int,
    target_window: int,
    actual: int,
    point_forecast: float,
    upper_forecast: float,
    activation_threshold: float,
) -> tuple[float, int]:
    """Record a row and return (pinball_loss_for_quantile, allocated)."""
    alloc = alloc_count(upper_forecast, activation_threshold)
    pinball = pinball_loss(POLICY_Q[policy], actual, upper_forecast)
    rows.append(
        {
            "workflow_name": workflow_name,
            "method": method,
            "policy": policy,
            "origin_window": origin_window,
            "target_window": target_window,
            "actual_count": actual,
            "forecast_count": float(upper_forecast),
            "point_forecast": float(point_forecast),
            "allocated_count": alloc,
            "under_count": max(0, actual - alloc),
            "over_count": max(0, alloc - actual),
            "pinball_loss": pinball,
        }
    )
    return pinball, alloc


def main() -> None:
    args = parse_args()
    workflow = load_workflow(args.workflow_config)
    workflow_name = workflow.workflow_name
    trace = pd.read_csv(args.trace)

    train_ids, test_ids, split_map = load_split(
        trace,
        workflow_name,
        args.split_map,
        train_ratio=0.7,
        split_strategy="time",
    )
    split_map["entry_window"] = (split_map["entry_ts_ms"] // args.window_ms).astype(int)
    train_end_window = int(split_map[split_map["split"] == "train"]["entry_window"].max())
    eval_start_window = train_end_window + 1
    eval_end_window = int(split_map[split_map["split"] == "test"]["entry_window"].max())

    entries = trace[
        (trace["workflow_name"] == workflow_name) & (trace["stage_name"] == "__entry__")
    ].copy()
    entries["window"] = (entries["entry_ts_ms"] // args.window_ms).astype(int)
    first_window = int(entries["window"].min())
    actual_by_window = entries.groupby("window").size().astype(int).to_dict()

    # Pre-sort entry timestamps for Hawkes (we'll slice by < win_start each origin).
    entry_times_ms_all = entries["entry_ts_ms"].astype(np.int64).sort_values().to_numpy()

    streaming_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    all_methods: list[str] = list(streaming_methods)
    if args.enable_hawkes:
        all_methods.append("hawkes-exp")
    base_methods = list(all_methods)  # methods whose p50 we can wrap with ACI
    if args.enable_aci:
        all_methods.extend([f"{m}+aci" for m in base_methods])

    loss_buf: dict[str, dict[str, deque]] = {
        m: {p: deque(maxlen=args.selector_window) for p in POLICIES} for m in all_methods
    }

    aci_states: dict[str, dict[str, ACIState]] = {}
    if args.enable_aci:
        for m in base_methods:
            aci_states[m] = {
                p: ACIState(POLICY_ALPHA[p], gamma=args.aci_gamma, warmup=args.aci_warmup)
                for p in POLICIES
            }

    detail_rows: list[dict] = []
    selector_choices: list[dict] = []

    hawkes_params: tuple[float, float, float] | None = None
    hawkes_last_refit: int | None = None
    hawkes_fit_log: list[dict] = []

    for origin_window in range(train_end_window, eval_end_window):
        target_window = origin_window + 1
        actual = int(actual_by_window.get(target_window, 0))

        counts = (
            entries[entries["window"] <= origin_window]
            .groupby("window")
            .size()
            .reindex(range(first_window, origin_window + 1), fill_value=0)
            .astype(float)
        )

        # ---- 1. Streaming methods (forecast_from_series) ----
        per_method_point: dict[str, float] = {}
        per_method_upper: dict[str, dict[str, float]] = {}
        per_method_alloc: dict[str, dict[str, int]] = {}

        for method in streaming_methods:
            forecast = forecast_from_series(
                counts=counts,
                method=method,
                alpha=args.alpha,
                residual_window=args.residual_window,
                history_window=args.history_window,
                burst_threshold=args.burst_threshold,
                burst_period_windows=args.burst_period_windows,
                burst_width_windows=args.burst_width_windows,
                background_count=args.background_count,
                idle_zero_ratio=args.idle_zero_ratio,
                activation_threshold=args.activation_threshold,
                horizon=1,
                method_label=method,
            )
            row = forecast.iloc[0]
            per_method_point[method] = float(row["p50_count"])
            per_method_upper[method] = {p: float(row[f"{p}_count"]) for p in POLICIES}
            per_method_alloc[method] = {p: allocated_for_policy(row, p) for p in POLICIES}
            for policy in POLICIES:
                pinball = pinball_loss(POLICY_Q[policy], actual, per_method_upper[method][policy])
                detail_rows.append(
                    {
                        "workflow_name": workflow_name,
                        "method": method,
                        "policy": policy,
                        "origin_window": origin_window,
                        "target_window": target_window,
                        "actual_count": actual,
                        "forecast_count": per_method_upper[method][policy],
                        "point_forecast": per_method_point[method],
                        "allocated_count": per_method_alloc[method][policy],
                        "under_count": max(0, actual - per_method_alloc[method][policy]),
                        "over_count": max(0, per_method_alloc[method][policy] - actual),
                        "pinball_loss": pinball,
                    }
                )
                loss_buf[method][policy].append(pinball)

        # ---- 2. Hawkes-exp (optional) ----
        if args.enable_hawkes:
            target_win_start_ms = target_window * args.window_ms
            history_mask = entry_times_ms_all < target_win_start_ms
            history_ts = entry_times_ms_all[history_mask].astype(float)
            need_refit = (
                hawkes_params is None
                or hawkes_last_refit is None
                or (origin_window - hawkes_last_refit) >= args.hawkes_refit_every
            )
            if need_refit and len(history_ts) >= 20:
                if args.hawkes_history_windows > 0:
                    cutoff_ms = float(
                        max(0, (origin_window + 1 - args.hawkes_history_windows) * args.window_ms)
                    )
                    fit_ts = history_ts[history_ts >= cutoff_ms]
                else:
                    fit_ts = history_ts
                if len(fit_ts) >= 20:
                    t_rel = fit_ts - fit_ts[0]
                    T_obs = float(target_win_start_ms - fit_ts[0])
                    params = fit_hawkes_exp(t_rel, T_obs, args.hawkes_max_iter)
                    if params is not None:
                        hawkes_params = params
                        hawkes_last_refit = origin_window
                        hawkes_fit_log.append(
                            {
                                "origin_window": origin_window,
                                "n_events": int(len(fit_ts)),
                                "T_obs_ms": T_obs,
                                "mu_per_ms": params[0],
                                "alpha": params[1],
                                "beta_per_ms": params[2],
                                "branching_ratio": params[1] / params[2] if params[2] > 0 else None,
                            }
                        )

            if hawkes_params is not None:
                # Hawkes integral uses absolute timestamps (history before win_start)
                lam = hawkes_window_integral(
                    *hawkes_params,
                    history=history_ts,
                    a=float(target_win_start_ms),
                    b=float(target_win_start_ms + args.window_ms),
                )
                lam = max(0.0, float(lam))
                p50_lam = lam
                p90_lam = float(poisson.ppf(0.90, mu=lam)) if lam > 0 else 0.0
                p95_lam = float(poisson.ppf(0.95, mu=lam)) if lam > 0 else 0.0
            else:
                lam = 0.0
                p50_lam = p90_lam = p95_lam = 0.0
            per_method_point["hawkes-exp"] = p50_lam
            per_method_upper["hawkes-exp"] = {"p50": p50_lam, "p90": p90_lam, "p95": p95_lam}
            per_method_alloc["hawkes-exp"] = {
                p: alloc_count(per_method_upper["hawkes-exp"][p], args.activation_threshold)
                for p in POLICIES
            }
            for policy in POLICIES:
                pinball = pinball_loss(POLICY_Q[policy], actual, per_method_upper["hawkes-exp"][policy])
                detail_rows.append(
                    {
                        "workflow_name": workflow_name,
                        "method": "hawkes-exp",
                        "policy": policy,
                        "origin_window": origin_window,
                        "target_window": target_window,
                        "actual_count": actual,
                        "forecast_count": per_method_upper["hawkes-exp"][policy],
                        "point_forecast": p50_lam,
                        "allocated_count": per_method_alloc["hawkes-exp"][policy],
                        "under_count": max(0, actual - per_method_alloc["hawkes-exp"][policy]),
                        "over_count": max(0, per_method_alloc["hawkes-exp"][policy] - actual),
                        "pinball_loss": pinball,
                    }
                )
                loss_buf["hawkes-exp"][policy].append(pinball)

        # ---- 3. ACI calibration on each base method's p50 (per-policy) ----
        if args.enable_aci:
            for m in base_methods:
                point = per_method_point.get(m, 0.0)
                aci_label = f"{m}+aci"
                for policy in POLICIES:
                    state = aci_states[m][policy]
                    upper = state.upper(point)
                    upper = max(0.0, float(upper))
                    pinball, alloc = append_method_row(
                        detail_rows,
                        workflow_name,
                        aci_label,
                        policy,
                        origin_window,
                        target_window,
                        actual,
                        point,
                        upper,
                        args.activation_threshold,
                    )
                    loss_buf[aci_label][policy].append(pinball)

        # ---- 4. Oracle (knows actual) ----
        for policy in POLICIES:
            detail_rows.append(
                {
                    "workflow_name": workflow_name,
                    "method": "oracle",
                    "policy": policy,
                    "origin_window": origin_window,
                    "target_window": target_window,
                    "actual_count": actual,
                    "forecast_count": float(actual),
                    "point_forecast": float(actual),
                    "allocated_count": actual,
                    "under_count": 0,
                    "over_count": 0,
                    "pinball_loss": 0.0,
                }
            )

        # ---- 5. Selector ----
        windows_seen = origin_window - train_end_window + 1
        for policy in POLICIES:
            if windows_seen <= args.selector_min_history:
                chosen = "ewma"
                fc = per_method_upper["ewma"][policy]
                alloc = per_method_alloc["ewma"][policy]
                point = per_method_point["ewma"]
            else:
                scores = {
                    m: (np.mean(loss_buf[m][policy]) if loss_buf[m][policy] else float("inf"))
                    for m in all_methods
                }
                chosen = min(scores, key=scores.get)
                if chosen.endswith("+aci"):
                    base = chosen[: -len("+aci")]
                    point = per_method_point[base]
                    state = aci_states[base][policy]
                    fc = max(0.0, float(state.upper(point)))
                    alloc = alloc_count(fc, args.activation_threshold)
                else:
                    fc = per_method_upper[chosen][policy]
                    alloc = per_method_alloc[chosen][policy]
                    point = per_method_point[chosen]
            detail_rows.append(
                {
                    "workflow_name": workflow_name,
                    "method": "selector",
                    "policy": policy,
                    "origin_window": origin_window,
                    "target_window": target_window,
                    "actual_count": actual,
                    "forecast_count": float(fc),
                    "point_forecast": float(point),
                    "allocated_count": alloc,
                    "under_count": max(0, actual - alloc),
                    "over_count": max(0, alloc - actual),
                    "pinball_loss": pinball_loss(POLICY_Q[policy], actual, fc),
                }
            )
            selector_choices.append(
                {
                    "policy": policy,
                    "origin_window": origin_window,
                    "target_window": target_window,
                    "chosen_method": chosen,
                }
            )

        # ---- 6. Update ACI states with observed actual ----
        if args.enable_aci:
            for m in base_methods:
                point = per_method_point.get(m, 0.0)
                for policy in POLICIES:
                    aci_states[m][policy].update(point, actual)

    detail = pd.DataFrame(detail_rows)
    choices = pd.DataFrame(selector_choices)

    summary_rows = []
    for (method, policy), grp in detail.groupby(["method", "policy"]):
        actual_arr = grp["actual_count"].astype(float).values
        alloc_arr = grp["allocated_count"].astype(float).values
        covered = (alloc_arr >= actual_arr).mean()
        active_mask = actual_arr > 0
        active_coverage = (
            (alloc_arr[active_mask] >= actual_arr[active_mask]).mean() if active_mask.any() else 1.0
        )
        mae = float(np.mean(np.abs(alloc_arr - actual_arr)))
        rmse = float(np.sqrt(np.mean((alloc_arr - actual_arr) ** 2)))
        pinball = float(grp["pinball_loss"].mean())
        summary_rows.append(
            {
                "method": method,
                "policy": policy,
                "windows": len(grp),
                "actual_total": int(actual_arr.sum()),
                "allocated_total": int(alloc_arr.sum()),
                "under_total": int(grp["under_count"].sum()),
                "over_total": int(grp["over_count"].sum()),
                "coverage_rate": float(covered),
                "active_coverage_rate": float(active_coverage),
                "mae": mae,
                "rmse": rmse,
                "pinball_loss_mean": pinball,
                "over_allocation_ratio": float(grp["over_count"].sum() / max(1, alloc_arr.sum())),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["policy", "mae"])

    selector_usage = (
        choices.groupby(["policy", "chosen_method"]).size().reset_index(name="count")
    )
    selector_usage["pct"] = selector_usage.groupby("policy")["count"].transform(
        lambda x: x / x.sum() * 100
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"{workflow_name}_online_eval_detail.csv"
    summary_path = out_dir / f"{workflow_name}_online_eval_summary.csv"
    selector_path = out_dir / f"{workflow_name}_online_eval_selector_usage.csv"
    metadata_path = out_dir / f"{workflow_name}_online_eval_metadata.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    selector_usage.to_csv(selector_path, index=False)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace": args.trace,
        "split_map": args.split_map,
        "workflow_name": workflow_name,
        "window_ms": args.window_ms,
        "streaming_methods": streaming_methods,
        "all_methods": all_methods,
        "eval_start_window": eval_start_window,
        "eval_end_window": eval_end_window,
        "selector_window": args.selector_window,
        "selector_min_history": args.selector_min_history,
        "hawkes_enabled": bool(args.enable_hawkes),
        "hawkes_refit_every": args.hawkes_refit_every if args.enable_hawkes else None,
        "hawkes_history_windows": args.hawkes_history_windows if args.enable_hawkes else None,
        "hawkes_n_fits": len(hawkes_fit_log) if args.enable_hawkes else 0,
        "aci_enabled": bool(args.enable_aci),
        "aci_gamma": args.aci_gamma if args.enable_aci else None,
        "aci_warmup": args.aci_warmup if args.enable_aci else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    if hawkes_fit_log:
        hawkes_log_path = out_dir / f"{workflow_name}_online_eval_hawkes_fits.csv"
        pd.DataFrame(hawkes_fit_log).to_csv(hawkes_log_path, index=False)

    print("===== Summary (p95 only, sorted by MAE) =====")
    p95_summary = summary[summary["policy"] == "p95"].copy()
    print(p95_summary.to_string(index=False))
    print(f"\n===== Selector usage (p95) =====")
    print(selector_usage[selector_usage["policy"] == "p95"].to_string(index=False))
    print(f"\nWrote: {summary_path}\nWrote: {detail_path}")


if __name__ == "__main__":
    main()
