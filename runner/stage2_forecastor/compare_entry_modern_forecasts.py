"""Modern forecasting baselines (Stage-2, entry-only, *no* allocation logic).

Implements rolling one-step forecasts using:
  - AutoETS         (Hyndman exponential smoothing family)
  - AutoTheta       (M3 competition winner family)
  - AutoARIMA       (auto-selected ARIMA)
  - MSTL            (multiple seasonal-trend decomposition + ETS residual)
  - CrostonSBA      (Syntetos-Boylan intermittent demand)
  - TSB             (Teunter-Syntetos-Babai intermittent demand)
  - fourier-reg     (sinusoidal seasonality features + HistGradientBoosting)

All forecasts are point estimates (no quantile / safety-buffer / allocation).
The output detail CSV only has `actual_count` and `forecast_count`; the
aggregator computes error / over_est / sMAPE entirely from those two columns.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("NIXTLA_NUMBA_RELEASE_GIL", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--workflow-name", required=True)
    p.add_argument("--split-cutoff-ms", type=int, required=True,
                   help="entry_ts_ms cutoff between train and test")
    p.add_argument("--window-sec", type=int, default=5)
    p.add_argument("--season-length", type=int, default=720,
                   help="seasonality period in windows (e.g. 720 = 1h at 5s windows)")
    p.add_argument(
        "--methods",
        default="auto-ets,auto-theta,auto-arima,mstl-ets,croston-sba,tsb,fourier-reg",
        help="comma-separated list of methods",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--write-detail", action="store_true")
    return p.parse_args()


def build_counts(trace: pd.DataFrame, workflow_name: str, window_ms: int) -> pd.Series:
    e = trace[
        (trace["workflow_name"] == workflow_name)
        & (trace["stage_name"] == "__entry__")
        & (trace["status"] == "ok")
    ].copy()
    e["window"] = (e["entry_ts_ms"] // window_ms).astype(int)
    first, last = int(e["window"].min()), int(e["window"].max())
    return (
        e.groupby("window").size().reindex(range(first, last + 1), fill_value=0).astype(float)
    )


def split_train_test(counts: pd.Series, cutoff_window: int) -> tuple[pd.Series, pd.Series]:
    train = counts[counts.index <= cutoff_window]
    test = counts[counts.index > cutoff_window]
    if train.empty or test.empty:
        raise ValueError("train or test is empty after split")
    return train, test


# ---- statsforecast methods ------------------------------------------------

def rolling_statsforecast(
    method_name: str,
    model,
    train: pd.Series,
    test: pd.Series,
) -> pd.DataFrame:
    """Use statsforecast.cross_validation to get rolling 1-step forecasts.

    statsforecast expects long-format dataframe (unique_id, ds, y).
    We treat each window-index as integer ds.
    """
    from statsforecast import StatsForecast

    all_counts = pd.concat([train, test])
    df = pd.DataFrame({
        "unique_id": "entry",
        "ds": all_counts.index.to_numpy(),
        "y": all_counts.to_numpy(dtype=float),
    })
    sf = StatsForecast(models=[model], freq=1, n_jobs=1)
    cv = sf.cross_validation(
        df=df, h=1, step_size=1, n_windows=len(test), refit=False,
    )
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    # statsforecast 2.x: column for model named after class
    model_col = [c for c in cv.columns if c not in {"unique_id", "ds", "cutoff", "y"}][0]
    rows = []
    for _, row in cv.iterrows():
        rows.append({
            "method": method_name,
            "window": int(row["ds"]),
            "actual_count": float(row["y"]),
            "forecast_count": max(0.0, float(row[model_col])),
        })
    return pd.DataFrame(rows)


def predict_auto_ets(train, test, season_length):
    """AutoETS struggles with very long seasonal periods (>~150); cap it."""
    from statsforecast.models import AutoETS
    sl = min(season_length, 120)
    return rolling_statsforecast(f"auto-ets-sl{sl}", AutoETS(season_length=sl), train, test)


def predict_auto_theta(train, test, season_length):
    from statsforecast.models import AutoTheta
    return rolling_statsforecast("auto-theta", AutoTheta(season_length=season_length), train, test)


def predict_auto_arima(train, test, season_length):
    from statsforecast.models import AutoARIMA
    return rolling_statsforecast(
        "auto-arima",
        AutoARIMA(season_length=season_length, max_p=3, max_q=3, max_P=1, max_Q=1,
                  approximation=True, stepwise=True),
        train, test,
    )


def predict_mstl_ets(train, test, season_length):
    from statsforecast.models import MSTL, AutoETS
    inner = AutoETS(model="ZZN", season_length=1)
    return rolling_statsforecast(
        "mstl-ets",
        MSTL(season_length=[season_length // 6, season_length], trend_forecaster=inner),
        train, test,
    )


def predict_croston_sba(train, test, alpha: float = 0.1):
    """Native online Croston SBA: works as a single-pass O(n) recurrence."""
    y_train = train.to_numpy(dtype=float)
    y_test = test.to_numpy(dtype=float)
    # Initialize from first nonzero in training
    nz = np.where(y_train > 0)[0]
    if len(nz) == 0:
        a = 0.0
        p = 1.0
        intv = 1.0
    else:
        a = float(y_train[nz[0]])
        p = float(nz[1] - nz[0]) if len(nz) > 1 else 1.0
        intv = float(len(y_train) - nz[-1])
    # walk through training to converge a,p
    for i, y in enumerate(y_train):
        if y > 0:
            a = alpha * y + (1 - alpha) * a
            p = alpha * intv + (1 - alpha) * p
            intv = 1.0
        else:
            intv += 1.0
    # one-step-ahead rolling forecast on test
    rows = []
    for i, w in enumerate(test.index):
        forecast = (1.0 - alpha / 2.0) * a / max(p, 1e-9)
        actual = float(y_test[i])
        rows.append({
            "method": "croston-sba",
            "window": int(w),
            "actual_count": actual,
            "forecast_count": max(0.0, forecast),
        })
        if actual > 0:
            a = alpha * actual + (1 - alpha) * a
            p = alpha * intv + (1 - alpha) * p
            intv = 1.0
        else:
            intv += 1.0
    return pd.DataFrame(rows)


def predict_tsb(train, test, alpha_d: float = 0.1, alpha_p: float = 0.1):
    """Native online TSB (Teunter-Syntetos-Babai) intermittent demand forecast."""
    y_train = train.to_numpy(dtype=float)
    y_test = test.to_numpy(dtype=float)
    # initial demand-size, prob-of-demand
    nz_vals = y_train[y_train > 0]
    z = float(nz_vals.mean()) if len(nz_vals) else 0.0
    pi = float((y_train > 0).mean()) if len(y_train) else 0.0
    for y in y_train:
        if y > 0:
            z = alpha_d * y + (1 - alpha_d) * z
            pi = alpha_p * 1.0 + (1 - alpha_p) * pi
        else:
            pi = (1 - alpha_p) * pi
    rows = []
    for i, w in enumerate(test.index):
        forecast = pi * z
        actual = float(y_test[i])
        rows.append({
            "method": "tsb",
            "window": int(w),
            "actual_count": actual,
            "forecast_count": max(0.0, forecast),
        })
        if actual > 0:
            z = alpha_d * actual + (1 - alpha_d) * z
            pi = alpha_p * 1.0 + (1 - alpha_p) * pi
        else:
            pi = (1 - alpha_p) * pi
    return pd.DataFrame(rows)


# ---- Fourier-feature regression ------------------------------------------

def _fourier_features(idx: np.ndarray, period_windows: float, n_harmonics: int) -> np.ndarray:
    feats = []
    for k in range(1, n_harmonics + 1):
        feats.append(np.sin(2 * np.pi * k * idx / period_windows))
        feats.append(np.cos(2 * np.pi * k * idx / period_windows))
    return np.column_stack(feats)


def predict_fourier_reg(train, test, season_length):
    from sklearn.ensemble import HistGradientBoostingRegressor

    all_counts = pd.concat([train, test])
    idx_all = all_counts.index.to_numpy(dtype=float)
    y_all = all_counts.to_numpy(dtype=float)

    # build feature matrix: fourier (main + half + 6-fold harmonics) + lag1, lag2, lag5, mean10
    fourier_main = _fourier_features(idx_all, season_length, n_harmonics=4)
    fourier_sub = _fourier_features(idx_all, season_length / 6.0, n_harmonics=3)

    def lag(arr, k):
        out = np.full_like(arr, fill_value=np.nan, dtype=float)
        out[k:] = arr[:-k]
        return out

    feat = np.column_stack([
        fourier_main, fourier_sub,
        lag(y_all, 1), lag(y_all, 2), lag(y_all, 5), lag(y_all, 12),
        pd.Series(y_all).rolling(10, min_periods=1).mean().to_numpy(),
        pd.Series(y_all).rolling(30, min_periods=1).mean().to_numpy(),
    ])

    n_train = len(train)
    # drop initial rows with NaN lags
    valid = ~np.isnan(feat).any(axis=1)
    train_mask = valid & (np.arange(len(feat)) < n_train)
    X_train = feat[train_mask]
    y_train = y_all[train_mask]

    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=400, l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    test_idx = np.arange(n_train, len(feat))
    X_test = feat[test_idx]
    y_pred = np.maximum(0.0, model.predict(X_test))

    rows = []
    for i, w in enumerate(test.index):
        rows.append({
            "method": "fourier-reg",
            "window": int(w),
            "actual_count": float(test.iloc[i]),
            "forecast_count": float(y_pred[i]),
        })
    return pd.DataFrame(rows)


# ---- main -----------------------------------------------------------------

def main() -> None:
    args = parse_args()
    window_ms = args.window_sec * 1000
    trace = pd.read_csv(args.trace)
    counts = build_counts(trace, args.workflow_name, window_ms)
    cutoff_window = int(args.split_cutoff_ms // window_ms)
    train, test = split_train_test(counts, cutoff_window)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    results = []
    timings = {}
    import time
    for m in methods:
        t0 = time.time()
        try:
            if m == "auto-ets":
                r = predict_auto_ets(train, test, args.season_length)
            elif m == "auto-theta":
                r = predict_auto_theta(train, test, args.season_length)
            elif m == "auto-arima":
                r = predict_auto_arima(train, test, args.season_length)
            elif m == "mstl-ets":
                r = predict_mstl_ets(train, test, args.season_length)
            elif m == "croston-sba":
                r = predict_croston_sba(train, test)
            elif m == "tsb":
                r = predict_tsb(train, test)
            elif m == "fourier-reg":
                r = predict_fourier_reg(train, test, args.season_length)
            else:
                print(f"[skip] unknown method: {m}")
                continue
            r.insert(0, "workflow_name", args.workflow_name)
            results.append(r)
            dt = time.time() - t0
            timings[m] = dt
            err_mae = float((r["actual_count"] - r["forecast_count"]).abs().mean())
            print(f"  [{m:14s}] OK  {dt:6.1f}s  MAE={err_mae:.3f}")
        except Exception as e:
            print(f"  [{m:14s}] FAIL  {type(e).__name__}: {e}")

    if not results:
        raise SystemExit("no methods produced output")

    detail = pd.concat(results, ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_detail:
        out_path = out_dir / f"{args.workflow_name}_entry_modern_compare_detail.csv"
        detail.to_csv(out_path, index=False)
        print(f"\nwrote {out_path}  rows={len(detail)}")

    metadata = {
        "trace": args.trace,
        "workflow_name": args.workflow_name,
        "split_cutoff_ms": args.split_cutoff_ms,
        "window_sec": args.window_sec,
        "season_length": args.season_length,
        "methods": methods,
        "train_windows": int(len(train)),
        "test_windows": int(len(test)),
        "timings_sec": timings,
    }
    (out_dir / f"{args.workflow_name}_entry_modern_compare_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
