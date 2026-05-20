"""SRB-Ensemble forecaster for Stage-2 entry-window prediction.

The novelty is composing three ingredients carefully:

1. Fourier seasonal decomposition via Ridge regression.
   Only learns the *deterministic* seasonal shape s(t), no noise. Closed-form
   solution -> stable, no overfit.

2. LightGBM residual booster with L1 (MAE) loss.
   Learns r(t) = y(t) - s(t) using lag-1..6 of y, lag-1..3 of residuals,
   rolling means, seasonal-phase sin/cos, and a burst flag. L1 objective
   directly aligns with the MAE evaluation metric -- this matters because
   default L2 produces mean-biased predictions that hurt MAE on Poisson-like
   counts.

3. Scalar calibration + soft zero-threshold.
   Fit on the last 20% of training: a multiplicative shrinkage beta that
   minimises MAE, plus a threshold theta below which we output 0. This
   addresses the 27% zero-window mass that bloats sMAPE.

We also output an ensemble forecast srb-ens that is a simple equal-weight
average of srb-base, auto-arima, and auto-ets. Simple equal-weight ensembles
are remarkably robust and beat learned-weight ensembles when validation
windows are scarce (the "wisdom of crowds" effect on noisy data).

The output detail CSV is in the same format as compare_entry_modern_forecasts.py
so the existing pure-accuracy aggregator works without changes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("NIXTLA_NUMBA_RELEASE_GIL", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="statsforecast")

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--workflow-name", required=True)
    p.add_argument("--split-cutoff-ms", type=int, required=True)
    p.add_argument("--window-sec", type=int, default=5)
    p.add_argument("--season-length", type=int, default=720)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--write-detail", action="store_true")
    p.add_argument("--methods",
                   default="srb-base,srb-ens,auto-arima,auto-ets,fourier-reg,auto-theta,naive",
                   help="methods to run. srb-ens triggers running its sub-models.")
    p.add_argument("--arima-season-length", type=int, default=60,
                   help="capped seasonality for AutoARIMA (long sl hangs the solver)")
    p.add_argument("--ets-season-length", type=int, default=120,
                   help="capped seasonality for AutoETS")
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


# ===================================================================
# Helper feature builders
# ===================================================================

def fourier_features(idx: np.ndarray, period: float, n_harmonics: int) -> np.ndarray:
    feats = []
    for k in range(1, n_harmonics + 1):
        feats.append(np.sin(2 * np.pi * k * idx / period))
        feats.append(np.cos(2 * np.pi * k * idx / period))
    return np.column_stack(feats)


# ===================================================================
# SRB base model
# ===================================================================

class SRBBase:
    """Fourier seasonal + LightGBM-L1 residual + calibration."""

    def __init__(self, season_length: int, sub_period_div: int = 6,
                 n_main_harmonics: int = 4, n_sub_harmonics: int = 3,
                 ridge_alpha: float = 1.0,
                 lgbm_params: dict | None = None,
                 calib_frac: float = 0.2):
        self.season_length = float(season_length)
        self.sub_period = float(season_length) / sub_period_div
        self.n_main = n_main_harmonics
        self.n_sub = n_sub_harmonics
        self.ridge_alpha = ridge_alpha
        self.calib_frac = calib_frac
        self.lgbm_params = lgbm_params or {
            "objective": "regression_l1",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "lambda_l2": 0.1,
            "n_estimators": 400,
            "verbose": -1,
        }
        self.ridge_coef_ = None
        self.ridge_intercept_ = None
        self.lgbm_ = None
        self.calib_beta_ = 1.0
        self.calib_theta_ = 0.5

    # -- seasonal -----
    def _seasonal_features(self, idx: np.ndarray) -> np.ndarray:
        return np.column_stack([
            fourier_features(idx, self.season_length, self.n_main),
            fourier_features(idx, self.sub_period, self.n_sub),
        ])

    def _fit_seasonal(self, idx: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import Ridge
        X = self._seasonal_features(idx)
        m = Ridge(alpha=self.ridge_alpha)
        m.fit(X, y)
        self.ridge_coef_ = m.coef_
        self.ridge_intercept_ = m.intercept_
        return m.predict(X)

    def _seasonal_predict(self, idx: np.ndarray) -> np.ndarray:
        X = self._seasonal_features(idx)
        return X @ self.ridge_coef_ + self.ridge_intercept_

    # -- residual features (autoregressive-safe) -----
    @staticmethod
    def _build_residual_features(y_hist: np.ndarray, r_hist: np.ndarray,
                                 t_idx: int, season_length: float, sub_period: float) -> np.ndarray:
        """Build features for predicting r at position t_idx in y_hist.
        y_hist and r_hist are arrays of length >= t_idx (only indices < t_idx
        are observed)."""
        def at(arr, k):
            j = t_idx - k
            return float(arr[j]) if j >= 0 else 0.0

        last5 = y_hist[max(0, t_idx - 5):t_idx]
        last10 = y_hist[max(0, t_idx - 10):t_idx]
        last30 = y_hist[max(0, t_idx - 30):t_idx]
        last60 = y_hist[max(0, t_idx - 60):t_idx]
        rm10 = float(last10.mean()) if len(last10) else 0.0
        rm30 = float(last30.mean()) if len(last30) else 0.0
        rm60 = float(last60.mean()) if len(last60) else 0.0
        burst = float(last5.max() > 2.0 * rm30) if len(last5) and rm30 > 0 else 0.0

        phase_main_s = np.sin(2 * np.pi * t_idx / season_length)
        phase_main_c = np.cos(2 * np.pi * t_idx / season_length)
        phase_sub_s = np.sin(2 * np.pi * t_idx / sub_period)
        phase_sub_c = np.cos(2 * np.pi * t_idx / sub_period)

        return np.array([
            at(y_hist, 1), at(y_hist, 2), at(y_hist, 3),
            at(y_hist, 4), at(y_hist, 5), at(y_hist, 6),
            at(r_hist, 1), at(r_hist, 2), at(r_hist, 3),
            rm10, rm30, rm60, burst,
            phase_main_s, phase_main_c, phase_sub_s, phase_sub_c,
        ], dtype=float)

    def fit(self, y_train: np.ndarray, idx_train: np.ndarray) -> None:
        import lightgbm as lgb

        # 1. fit seasonal
        s_train = self._fit_seasonal(idx_train.astype(float), y_train)
        r_train = y_train - s_train

        # 2. build training matrix for residual booster (use ACTUAL y/r history)
        n = len(y_train)
        X_rows, y_targets = [], []
        for t in range(6, n):
            feats = self._build_residual_features(
                y_train, r_train, t, self.season_length, self.sub_period
            )
            X_rows.append(feats)
            y_targets.append(r_train[t])
        X = np.asarray(X_rows)
        y = np.asarray(y_targets)

        # 3. fit LightGBM with L1 objective
        self.lgbm_ = lgb.LGBMRegressor(**self.lgbm_params)
        self.lgbm_.fit(X, y)

        # 4. calibration: re-predict on last calib_frac of train using rolling-style
        #    (predictions use actual past r, just like at test time)
        n_calib = max(20, int(n * self.calib_frac))
        calib_start = n - n_calib
        y_pred_cal = np.zeros(n_calib)
        for i, t in enumerate(range(calib_start, n)):
            feats = self._build_residual_features(
                y_train, r_train, t, self.season_length, self.sub_period
            )
            r_hat = float(self.lgbm_.predict(feats.reshape(1, -1))[0])
            y_pred_cal[i] = max(0.0, s_train[t] + r_hat)
        y_true_cal = y_train[calib_start:n]

        # beta = scalar that minimises MAE -> closed form weighted median of (y/ŷ)
        eps = 1e-9
        ratios = y_true_cal / (y_pred_cal + eps)
        weights = y_pred_cal  # weight by predicted magnitude
        if weights.sum() > 0:
            order = np.argsort(ratios)
            ratios_s = ratios[order]
            weights_s = weights[order]
            cum = np.cumsum(weights_s)
            half = weights_s.sum() / 2.0
            k = np.searchsorted(cum, half)
            self.calib_beta_ = float(np.clip(ratios_s[min(k, len(ratios_s) - 1)], 0.5, 1.5))
        else:
            self.calib_beta_ = 1.0

        # theta = soft zero threshold (small values get clipped to 0)
        # Choose theta in {0.1, 0.2, ..., 1.0} that minimises calibrated MAE on calib set
        cal_pred_beta = self.calib_beta_ * y_pred_cal
        best_theta, best_mae = 0.0, float("inf")
        for theta in np.linspace(0.0, 1.0, 11):
            pred = np.where(cal_pred_beta < theta, 0.0, cal_pred_beta)
            mae = float(np.mean(np.abs(y_true_cal - pred)))
            if mae < best_mae:
                best_mae = mae
                best_theta = float(theta)
        self.calib_theta_ = best_theta

    def predict_rolling(self, y_train: np.ndarray, idx_train: np.ndarray,
                        y_test: np.ndarray, idx_test: np.ndarray) -> np.ndarray:
        """1-step rolling predictions on test windows.
        After predicting window t, we get actual y(t) and append it to history."""
        # Concatenate histories (observed truthful y/r for all past windows)
        y_hist = list(y_train)
        # residuals are recomputed because we now know y_hist and seasonal s
        s_train = self._seasonal_predict(idx_train.astype(float))
        r_hist = list(y_train - s_train)

        s_test = self._seasonal_predict(idx_test.astype(float))
        n_test = len(y_test)
        preds = np.zeros(n_test)
        for i in range(n_test):
            t_global = len(y_hist)  # next index to predict
            feats = self._build_residual_features(
                np.asarray(y_hist), np.asarray(r_hist),
                t_global, self.season_length, self.sub_period,
            )
            r_hat = float(self.lgbm_.predict(feats.reshape(1, -1))[0])
            y_hat_raw = max(0.0, s_test[i] + r_hat)
            y_hat = self.calib_beta_ * y_hat_raw
            if y_hat < self.calib_theta_:
                y_hat = 0.0
            preds[i] = y_hat
            # observe truth, append to history for next step
            y_hist.append(y_test[i])
            r_hist.append(y_test[i] - s_test[i])
        return preds


# ===================================================================
# Baseline predictors (reused / inlined; same as compare_entry_modern_forecasts.py)
# ===================================================================

def rolling_statsforecast(method_name: str, model, train: pd.Series, test: pd.Series) -> pd.DataFrame:
    from statsforecast import StatsForecast
    all_counts = pd.concat([train, test])
    df = pd.DataFrame({
        "unique_id": "entry",
        "ds": all_counts.index.to_numpy(),
        "y": all_counts.to_numpy(dtype=float),
    })
    sf = StatsForecast(models=[model], freq=1, n_jobs=1)
    cv = sf.cross_validation(df=df, h=1, step_size=1, n_windows=len(test), refit=False)
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
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


def predict_auto_ets(train, test, sl):
    from statsforecast.models import AutoETS
    return rolling_statsforecast(f"auto-ets-sl{sl}", AutoETS(season_length=sl), train, test)


def predict_auto_theta(train, test, sl):
    from statsforecast.models import AutoTheta
    return rolling_statsforecast("auto-theta", AutoTheta(season_length=sl), train, test)


def predict_auto_arima(train, test, sl):
    from statsforecast.models import AutoARIMA
    return rolling_statsforecast(
        f"auto-arima-sl{sl}",
        AutoARIMA(season_length=sl, max_p=3, max_q=3, max_P=1, max_Q=1,
                  approximation=True, stepwise=True),
        train, test,
    )


def predict_fourier_reg(train, test, season_length):
    from sklearn.ensemble import HistGradientBoostingRegressor
    all_counts = pd.concat([train, test])
    idx_all = all_counts.index.to_numpy(dtype=float)
    y_all = all_counts.to_numpy(dtype=float)
    fmain = fourier_features(idx_all, season_length, n_harmonics=4)
    fsub = fourier_features(idx_all, season_length / 6.0, n_harmonics=3)

    def lag(arr, k):
        out = np.full_like(arr, fill_value=np.nan, dtype=float)
        out[k:] = arr[:-k]
        return out

    feat = np.column_stack([
        fmain, fsub,
        lag(y_all, 1), lag(y_all, 2), lag(y_all, 5), lag(y_all, 12),
        pd.Series(y_all).rolling(10, min_periods=1).mean().to_numpy(),
        pd.Series(y_all).rolling(30, min_periods=1).mean().to_numpy(),
    ])
    n_train = len(train)
    valid = ~np.isnan(feat).any(axis=1)
    train_mask = valid & (np.arange(len(feat)) < n_train)
    X_train = feat[train_mask]
    y_train = y_all[train_mask]
    model = HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05,
                                          max_iter=400, l2_regularization=0.1, random_state=42)
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


def predict_naive(train, test):
    last_val = float(train.iloc[-1])
    rows = []
    prev = last_val
    for w, y in test.items():
        rows.append({
            "method": "naive",
            "window": int(w),
            "actual_count": float(y),
            "forecast_count": max(0.0, prev),
        })
        prev = float(y)  # one-step-ahead = previous actual
    return pd.DataFrame(rows)


def predict_srb_base(train: pd.Series, test: pd.Series, season_length: int) -> pd.DataFrame:
    y_tr = train.to_numpy(dtype=float)
    idx_tr = train.index.to_numpy()
    y_te = test.to_numpy(dtype=float)
    idx_te = test.index.to_numpy()
    m = SRBBase(season_length=season_length)
    m.fit(y_tr, idx_tr)
    preds = m.predict_rolling(y_tr, idx_tr, y_te, idx_te)
    rows = []
    for i, w in enumerate(test.index):
        rows.append({
            "method": "srb-base",
            "window": int(w),
            "actual_count": float(y_te[i]),
            "forecast_count": float(preds[i]),
        })
    return pd.DataFrame(rows), preds, m


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    args = parse_args()
    window_ms = args.window_sec * 1000
    trace = pd.read_csv(args.trace)
    counts = build_counts(trace, args.workflow_name, window_ms)
    cutoff_window = int(args.split_cutoff_ms // window_ms)
    train = counts[counts.index <= cutoff_window]
    test = counts[counts.index > cutoff_window]
    if train.empty or test.empty:
        raise SystemExit("train or test empty")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"trace={args.trace}  workflow={args.workflow_name}  window={args.window_sec}s")
    print(f"train_windows={len(train)}  test_windows={len(test)}  cutoff_window={cutoff_window}")
    print(f"methods={methods}")
    print(f"arima_sl={args.arima_season_length}  ets_sl={args.ets_season_length}")

    results = []
    timings = {}
    srb_preds = None
    arima_preds = None
    ets_preds = None
    srb_model = None

    # We may need srb-base + auto-arima + auto-ets even if user only asked
    # for srb-ens.
    needed = set(methods)
    if "srb-ens" in needed:
        needed.update({"srb-base", "auto-arima", "auto-ets"})

    test_index = test.index.to_numpy()

    for m in methods if "srb-ens" not in methods else [x for x in methods if x != "srb-ens"] + ["srb-ens"]:
        if m == "srb-ens" and not (srb_preds is not None and arima_preds is not None and ets_preds is not None):
            print(f"  [srb-ens       ] FAIL  sub-model predictions missing")
            continue
        t0 = time.time()
        try:
            if m == "srb-base":
                df, preds, srb_model = predict_srb_base(train, test, args.season_length)
                srb_preds = preds
                r = df
            elif m == "auto-arima":
                r = predict_auto_arima(train, test, args.arima_season_length)
                # extract forecast aligned with test_index
                r_sorted = r.sort_values("window").reset_index(drop=True)
                arima_preds = r_sorted["forecast_count"].to_numpy()
            elif m == "auto-ets":
                r = predict_auto_ets(train, test, args.ets_season_length)
                r_sorted = r.sort_values("window").reset_index(drop=True)
                ets_preds = r_sorted["forecast_count"].to_numpy()
            elif m == "auto-theta":
                r = predict_auto_theta(train, test, args.season_length)
            elif m == "fourier-reg":
                r = predict_fourier_reg(train, test, args.season_length)
            elif m == "naive":
                r = predict_naive(train, test)
            elif m == "srb-ens":
                # equal-weight ensemble of srb-base + auto-arima + auto-ets
                if not (len(srb_preds) == len(arima_preds) == len(ets_preds) == len(test)):
                    raise RuntimeError("ensemble component length mismatch")
                ens = (srb_preds + arima_preds + ets_preds) / 3.0
                ens = np.maximum(0.0, ens)
                rows = []
                for i, w in enumerate(test.index):
                    rows.append({
                        "method": "srb-ens",
                        "window": int(w),
                        "actual_count": float(test.iloc[i]),
                        "forecast_count": float(ens[i]),
                    })
                r = pd.DataFrame(rows)
            else:
                print(f"  [{m:14s}] SKIP unknown")
                continue
            r.insert(0, "workflow_name", args.workflow_name)
            results.append(r)
            dt = time.time() - t0
            timings[m] = dt
            mae = float((r["actual_count"] - r["forecast_count"]).abs().mean())
            print(f"  [{m:14s}] OK  {dt:6.1f}s  MAE={mae:.3f}")
        except Exception as e:
            import traceback
            print(f"  [{m:14s}] FAIL  {type(e).__name__}: {e}")
            traceback.print_exc()

    if not results:
        raise SystemExit("no methods produced output")

    detail = pd.concat(results, ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_detail:
        out_path = out_dir / f"{args.workflow_name}_entry_srb_compare_detail.csv"
        detail.to_csv(out_path, index=False)
        print(f"\nwrote {out_path}  rows={len(detail)}")

    # metadata
    meta = {
        "trace": args.trace,
        "workflow_name": args.workflow_name,
        "split_cutoff_ms": args.split_cutoff_ms,
        "window_sec": args.window_sec,
        "season_length": args.season_length,
        "arima_season_length": args.arima_season_length,
        "ets_season_length": args.ets_season_length,
        "methods": methods,
        "train_windows": int(len(train)),
        "test_windows": int(len(test)),
        "timings_sec": timings,
    }
    if srb_model is not None:
        meta["srb_calib"] = {
            "beta": float(srb_model.calib_beta_),
            "theta": float(srb_model.calib_theta_),
        }
    (out_dir / f"{args.workflow_name}_entry_srb_compare_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
