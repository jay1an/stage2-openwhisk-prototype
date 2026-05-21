"""Online ML wrappers for Stage-2 entry forecasting.

Three "trained" model families adapted to one-step-ahead online use:

- AutoArimaOnline: rolling refit every K windows on the history seen so far,
  using statsforecast.AutoARIMA. Direct p50/p90/p95 from the model's prediction
  intervals.
- LightGBMOnline: rolling refit every K windows; lag + cyclical-period features
  feed three quantile regressors (alpha = 0.5 / 0.9 / 0.95).
- LSTMOnline: pretrain once on warmup history, then freeze the network and run
  a residual quantile calibrator online (rolling residual quantile gives the
  p90/p95 padding).

All three expose the same one-step-ahead interface:

    wrapper.predict(counts: pd.Series, target_window: int) -> dict
        returns {"p50_count", "p90_count", "p95_count", "lambda" (optional)}

`counts` MUST already be a per-window count series with index = window indices,
covering [first_window, origin_window]. The wrapper does NOT see future data.

The harness owns refit cadence via these wrappers' internal counters; the
wrappers do not log to disk.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Quiet statsforecast / lightgbm noise.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# 1. Auto-ARIMA (rolling refit)
# ---------------------------------------------------------------------------


@dataclass
class AutoArimaOnline:
    """Seasonal ARIMA with fixed orders, refit every K windows.

    Uses fixed (p,d,q)(P,D,Q)[season_length] rather than full Auto-ARIMA stepwise
    search because the latter costs >100 sec per fit on >1k-window history, which
    is impractical when refitting every 60 windows over 432 eval windows.
    """

    season_length: int = 60
    refit_every: int = 60
    min_history: int = 120
    order: tuple[int, int, int] = (2, 0, 1)
    seasonal_order: tuple[int, int, int] = (1, 0, 0)
    _model: object = None
    _last_refit_target: int | None = None

    def predict(self, counts: pd.Series, target_window: int) -> dict[str, float]:
        n = len(counts)
        if n < self.min_history:
            mean = float(counts.mean()) if n else 0.0
            return {
                "p50_count": max(0.0, mean),
                "p90_count": max(0.0, mean),
                "p95_count": max(0.0, mean),
            }
        need_refit = (
            self._model is None
            or self._last_refit_target is None
            or (target_window - self._last_refit_target) >= self.refit_every
        )
        if need_refit:
            from statsforecast.models import ARIMA

            self._model = ARIMA(
                order=self.order,
                seasonal_order=self.seasonal_order,
                season_length=self.season_length,
            )
            try:
                self._model.fit(counts.to_numpy(dtype=np.float64))
                self._last_refit_target = target_window
            except Exception:
                self._model = None
        if self._model is None:
            mean = float(counts.mean())
            return {
                "p50_count": max(0.0, mean),
                "p90_count": max(0.0, mean),
                "p95_count": max(0.0, mean),
            }
        try:
            fc = self._model.predict(h=1, level=[90, 95])
        except Exception:
            mean = float(counts.mean())
            return {
                "p50_count": max(0.0, mean),
                "p90_count": max(0.0, mean),
                "p95_count": max(0.0, mean),
            }
        p50 = float(fc["mean"][0])
        p90 = float(fc["hi-90"][0])
        p95 = float(fc["hi-95"][0])
        return {
            "p50_count": max(0.0, p50),
            "p90_count": max(0.0, p90),
            "p95_count": max(0.0, p95),
        }


# ---------------------------------------------------------------------------
# 2. LightGBM (rolling refit, quantile regression on lag + cyclical features)
# ---------------------------------------------------------------------------


def _build_lgb_features(
    counts: np.ndarray,
    target_windows: np.ndarray,
    history_windows: np.ndarray,
    n_lags: int,
    main_period: int,
    sub_period: int,
) -> np.ndarray:
    """One row per target window, lags relative to that window's index in counts."""
    n = len(counts)
    rows = []
    for t_abs, t_in_counts in zip(target_windows, history_windows):
        row: list[float] = []
        for lag in range(1, n_lags + 1):
            idx = t_in_counts - lag
            row.append(counts[idx] if 0 <= idx < n else 0.0)
        row.extend(
            [
                math.sin(2.0 * math.pi * t_abs / main_period),
                math.cos(2.0 * math.pi * t_abs / main_period),
                math.sin(2.0 * math.pi * t_abs / sub_period),
                math.cos(2.0 * math.pi * t_abs / sub_period),
            ]
        )
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


@dataclass
class LightGBMOnline:
    refit_every: int = 60
    n_lags: int = 20
    main_period: int = 60
    sub_period: int = 12
    min_history: int = 240
    n_estimators: int = 200
    num_leaves: int = 15
    learning_rate: float = 0.05
    _q50: object = None
    _q90: object = None
    _q95: object = None
    _last_refit_target: int | None = None

    def predict(self, counts: pd.Series, target_window: int) -> dict[str, float]:
        n = len(counts)
        if n < self.min_history:
            mean = float(counts.mean()) if n else 0.0
            return {
                "p50_count": max(0.0, mean),
                "p90_count": max(0.0, mean),
                "p95_count": max(0.0, mean),
            }
        need_refit = (
            self._q50 is None
            or self._last_refit_target is None
            or (target_window - self._last_refit_target) >= self.refit_every
        )
        counts_arr = counts.to_numpy(dtype=np.float64)
        # `counts` index is absolute window indices.
        abs_idx = counts.index.to_numpy(dtype=np.int64)
        if need_refit:
            import lightgbm as lgb

            in_counts_idx = np.arange(self.n_lags, n)
            target_abs = abs_idx[in_counts_idx]
            X = _build_lgb_features(
                counts_arr,
                target_abs,
                in_counts_idx,
                self.n_lags,
                self.main_period,
                self.sub_period,
            )
            y = counts_arr[self.n_lags:]
            common = dict(
                n_estimators=self.n_estimators,
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                min_data_in_leaf=5,
                verbose=-1,
            )
            try:
                self._q50 = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **common).fit(X, y)
                self._q90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **common).fit(X, y)
                self._q95 = lgb.LGBMRegressor(objective="quantile", alpha=0.95, **common).fit(X, y)
                self._last_refit_target = target_window
            except Exception:
                self._q50 = self._q90 = self._q95 = None
        if self._q50 is None:
            mean = float(counts.mean())
            return {
                "p50_count": max(0.0, mean),
                "p90_count": max(0.0, mean),
                "p95_count": max(0.0, mean),
            }
        # Feature for the target window: lags relative to position n in counts.
        target_row = _build_lgb_features(
            counts_arr,
            np.array([target_window], dtype=np.int64),
            np.array([n], dtype=np.int64),
            self.n_lags,
            self.main_period,
            self.sub_period,
        )
        p50 = float(self._q50.predict(target_row)[0])
        p90 = float(self._q90.predict(target_row)[0])
        p95 = float(self._q95.predict(target_row)[0])
        # Quantile crossing guard.
        p90 = max(p90, p50)
        p95 = max(p95, p90)
        return {
            "p50_count": max(0.0, p50),
            "p90_count": max(0.0, p90),
            "p95_count": max(0.0, p95),
        }


# ---------------------------------------------------------------------------
# 3. LSTM (pretrain + frozen network + online residual quantile calibrator)
# ---------------------------------------------------------------------------


@dataclass
class LSTMOnline:
    seq_len: int = 30
    hidden: int = 32
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    residual_window: int = 60
    main_period: int = 60
    pretrain_min_history: int = 240
    _net: object = None
    _device: object = None
    _residuals: list[float] = field(default_factory=list)

    def _ensure_pretrained(self, counts: np.ndarray) -> bool:
        if self._net is not None:
            return True
        if len(counts) < self.pretrain_min_history + self.seq_len:
            return False
        import torch
        from torch import nn

        self._device = torch.device("cpu")

        # Build supervised pairs: (counts[t-seq_len:t]) -> counts[t]
        idx = np.arange(self.seq_len, len(counts))
        X = np.stack([counts[t - self.seq_len : t] for t in idx]).astype(np.float32)
        y = counts[idx].astype(np.float32)
        X_t = torch.from_numpy(X).unsqueeze(-1)  # (N, seq_len, 1)
        y_t = torch.from_numpy(y)

        class TinyLSTM(nn.Module):
            def __init__(self, hidden: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        net = TinyLSTM(self.hidden).to(self._device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        n = X_t.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                batch = perm[start : start + self.batch_size]
                yb = y_t[batch].to(self._device)
                xb = X_t[batch].to(self._device)
                opt.zero_grad()
                pred = net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        net.eval()
        self._net = net

        # Warm the residual buffer with training residuals over the last window.
        with __import__("torch").no_grad():
            preds = net(X_t.to(self._device)).cpu().numpy()
        residuals = (y - preds).tolist()
        self._residuals = residuals[-self.residual_window :]
        return True

    def predict(self, counts: pd.Series, target_window: int) -> dict[str, float]:
        counts_arr = counts.to_numpy(dtype=np.float64)
        if not self._ensure_pretrained(counts_arr):
            mean = float(counts.mean()) if len(counts) else 0.0
            return {
                "p50_count": max(0.0, mean),
                "p90_count": max(0.0, mean),
                "p95_count": max(0.0, mean),
            }
        import torch

        tail = counts_arr[-self.seq_len :].astype(np.float32)
        with torch.no_grad():
            x = torch.from_numpy(tail).unsqueeze(0).unsqueeze(-1).to(self._device)
            point = float(self._net(x).cpu().numpy()[0])
        point = max(0.0, point)
        if len(self._residuals) == 0:
            pad_90 = pad_95 = 0.0
        else:
            arr = np.asarray(self._residuals)
            pad_90 = float(np.quantile(arr, 0.90))
            pad_95 = float(np.quantile(arr, 0.95))
        return {
            "p50_count": point,
            "p90_count": max(0.0, point + max(0.0, pad_90)),
            "p95_count": max(0.0, point + max(0.0, pad_95)),
        }

    def observe(self, point: float, actual: float) -> None:
        """Append residual after observing actual; harness calls per window."""
        self._residuals.append(float(actual) - float(point))
        if len(self._residuals) > self.residual_window:
            self._residuals = self._residuals[-self.residual_window :]
