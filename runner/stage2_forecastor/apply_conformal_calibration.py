"""Conformal Prediction calibration wrapper for entry-window forecasts.

Implements two principled calibration methods that replace the ad-hoc
`*-calibrated` / `*-gated` / `burst-aware` patches:

  split-cp:  Romano et al. 2019 split conformal (one-sided upper bound).
              residuals e_t = max(0, y_t - q_hat_t) on a calibration window.
              Predict y_{t+1} <= q_hat_{t+1} + Q_{1-alpha}(e_cal).
              Coverage guarantee on exchangeable data.

  aci:       Gibbs & Candes 2021 adaptive conformal inference.
              alpha_t <- alpha_t + gamma (alpha_target - err_t).
              No exchangeability required; tracks coverage under drift.

Input  : forecast detail CSV with the standard
         (workflow_name, method, policy, window, actual_count, forecast_count,
          allocated_count, under_count, over_count) schema.
Output : same schema with forecast_count replaced by the CP-inflated upper
         bound and method renamed to "<method>+<cp>".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


POLICY_TO_ALPHA = {"p50": 0.50, "p90": 0.10, "p95": 0.05}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conformal calibration wrapper")
    p.add_argument("--detail", required=True, help="forecast detail CSV (long format)")
    p.add_argument("--out", required=True, help="output calibrated detail CSV")
    p.add_argument("--method", choices=["split-cp", "aci"], default="aci")
    p.add_argument("--calib-frac", type=float, default=0.3,
                   help="split-CP calibration set fraction of the series")
    p.add_argument("--warmup", type=int, default=20,
                   help="windows held out before predictions are emitted")
    p.add_argument("--gamma", type=float, default=0.05,
                   help="ACI step size")
    p.add_argument("--activation-threshold", type=float, default=0.1)
    return p.parse_args()


def alloc_count(forecast_count: float, threshold: float) -> int:
    if forecast_count <= threshold:
        return 0
    return int(np.ceil(forecast_count))


def split_conformal_upper(point: np.ndarray, actual: np.ndarray, alpha: float,
                          calib_idx: np.ndarray, eval_idx: np.ndarray) -> np.ndarray:
    residuals = np.maximum(0.0, actual[calib_idx] - point[calib_idx])
    n = len(residuals)
    out = point.copy().astype(float)
    if n == 0:
        return out
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q_hat = float(np.quantile(residuals, q_level))
    out[eval_idx] = point[eval_idx] + q_hat
    return out


def aci_upper(point: np.ndarray, actual: np.ndarray, alpha_target: float,
              gamma: float, warmup: int) -> np.ndarray:
    n = len(point)
    out = point.copy().astype(float)
    alpha_t = alpha_target
    residuals: list[float] = []
    for t in range(n):
        if t < warmup:
            residuals.append(max(0.0, float(actual[t] - point[t])))
            continue
        recent = np.asarray(residuals, dtype=float)
        if recent.size == 0:
            q_hat = 0.0
        else:
            level = float(np.clip(1.0 - alpha_t, 0.0, 1.0))
            q_hat = float(np.quantile(recent, level))
        out[t] = float(point[t]) + q_hat
        miss = 1.0 if float(actual[t]) > out[t] else 0.0
        alpha_t = float(np.clip(alpha_t + gamma * (alpha_target - miss), 1e-4, 1.0 - 1e-4))
        residuals.append(max(0.0, float(actual[t]) - float(point[t])))
    return out


def calibrate_one(grp: pd.DataFrame, method: str, args: argparse.Namespace) -> pd.DataFrame:
    grp = grp.sort_values("window").reset_index(drop=True)
    pol = str(grp["policy"].iloc[0])
    alpha = POLICY_TO_ALPHA.get(pol)
    if alpha is None:
        return grp
    point = grp["forecast_count"].astype(float).values
    actual = grp["actual_count"].astype(float).values
    n = len(grp)
    if method == "split-cp":
        cal_size = max(1, int(n * args.calib_frac))
        calib_idx = np.arange(min(cal_size, n))
        eval_idx = np.arange(len(calib_idx), n)
        ub = split_conformal_upper(point, actual, alpha, calib_idx, eval_idx)
    else:
        ub = aci_upper(point, actual, alpha, args.gamma, args.warmup)
    ub = np.maximum(0.0, ub)
    out = grp.copy()
    out["forecast_count"] = ub
    out["allocated_count"] = [alloc_count(float(x), args.activation_threshold) for x in ub]
    out["under_count"] = np.maximum(0, out["actual_count"].astype(int) - out["allocated_count"].astype(int))
    out["over_count"] = np.maximum(0, out["allocated_count"].astype(int) - out["actual_count"].astype(int))
    base_method = str(grp["method"].iloc[0])
    out["method"] = f"{base_method}+{method}"
    return out


def main() -> None:
    args = parse_args()
    detail = pd.read_csv(args.detail)
    if "window" not in detail.columns and "target_window" in detail.columns:
        detail = detail.rename(columns={"target_window": "window"})
    required = {"workflow_name", "method", "policy", "window", "actual_count", "forecast_count"}
    missing = required - set(detail.columns)
    if missing:
        raise SystemExit(f"detail missing columns: {missing}")
    parts = []
    for keys, grp in detail.groupby(["workflow_name", "method", "policy"], sort=False):
        parts.append(calibrate_one(grp, args.method, args))
    calibrated = pd.concat(parts, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(out_path, index=False)
    meta = {
        "input": args.detail,
        "method": args.method,
        "calib_frac": args.calib_frac,
        "warmup": args.warmup,
        "gamma": args.gamma,
        "rows_in": int(len(detail)),
        "rows_out": int(len(calibrated)),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_path}  rows={len(calibrated)}  cp_method={args.method}")


if __name__ == "__main__":
    main()
