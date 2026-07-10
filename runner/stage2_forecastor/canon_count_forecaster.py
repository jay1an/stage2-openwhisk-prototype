#!/usr/bin/env python3
"""CANON entry-arrival count forecaster.

This module turns the previously report-local CANON forecaster into reusable
version-controlled code.  The primary model predicts 300s arrival counts with
causal lag/seasonal features and a HistGradientBoostingRegressor mean layer.
Poisson and Negative-Binomial count layers turn that mean into pool sizes; a
60s top-up layer adds a recent residual quantile for faster burst response.

CAVEATS (read before citing the offline pool_frontier numbers):
- The offline ``simulate_oracle`` assumes a warm container is available exactly
  when needed, so it reports 0% entry cold. The REAL cluster cannot: the CANON
  oracle-static replay measured ~7% (premium) / ~13% (free) entry cold despite
  oracle prewarm (warmup ~1.7s vs a 2.5s lead, container-reuse steal, bursts).
  Treat the offline oracle as an idealized floor; the real oracle floor is ~10%.
  Use this proxy for ORDERING/trends, not absolute cold rates. Ground truth is
  the Step-6 three-arm cluster run.
- On this sparse/bursty CANON trace the useful operational point is the 60s
  HGBR direct-quantile (hgb_q90_60s). The NB-on-mean K-sizer mis-allocates
  (over-provisions high-mean bins, under-provisions low-mean burst bins) and is
  dominated by hgb_q90; the 300s versions are dominated by static keepalive.
  Keep NB/Fano only as an overdispersion descriptor, not as the K-sizer.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from runner.stage2_forecastor.forecast_entry import recent_residual_quantile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARRIVALS = ROOT / "reports" / "trace_selection_v3" / "regular_aggregate_FINAL_scaled_arrivals.csv"
DEFAULT_CANON_SCHEDULE = (
    ROOT
    / "reports"
    / "trace_selection_v3"
    / "schedule_regular_aggregate_CANON_20h_peak40_15_FINAL_eval.csv"
)
DEFAULT_OUT_DIR = ROOT / "reports" / "step4_forecaster"
DAY_S = 86400.0
TRAIN_END_S = 12.0 * DAY_S
SEED = 20260615
LEAD_SEC = 2.5


@dataclass(frozen=True)
class CountData:
    bin_s: int
    edges: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    indices: np.ndarray
    features: np.ndarray
    train_mask: np.ndarray
    eval_mask: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class CountModel:
    bin_s: int
    starts: np.ndarray
    actual: np.ndarray
    train_actual: np.ndarray
    mean_pred: np.ndarray
    q90_pred: np.ndarray
    poisson_k90: np.ndarray
    nb_k90: np.ndarray
    persistence: np.ndarray
    seasonal_only: np.ndarray
    global_mean: np.ndarray
    nb_alpha: float
    nb_k: float
    train_fano: float


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def load_inputs(
    arrivals_path: str | Path = DEFAULT_ARRIVALS,
    canon_schedule_path: str | Path = DEFAULT_CANON_SCHEDULE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    arrivals = pd.read_csv(resolve(arrivals_path))
    canon = pd.read_csv(resolve(canon_schedule_path))
    arrivals_required = {"arrival_s", "split"}
    canon_required = {"source_start_s", "target_offset_ms"}
    missing_arrivals = sorted(arrivals_required.difference(arrivals.columns))
    missing_canon = sorted(canon_required.difference(canon.columns))
    if missing_arrivals:
        raise ValueError(f"arrivals file missing columns: {missing_arrivals}")
    if missing_canon:
        raise ValueError(f"CANON schedule missing columns: {missing_canon}")
    return arrivals, canon


def validate_split(arrivals: pd.DataFrame, canon: pd.DataFrame) -> dict[str, Any]:
    train = arrivals[arrivals["split"].eq("train")]["arrival_s"].to_numpy(float)
    eval_all = arrivals[arrivals["split"].eq("eval")]["arrival_s"].to_numpy(float)
    canon_times = canon["source_start_s"].to_numpy(float)
    if len(train) == 0 or len(eval_all) == 0 or len(canon_times) == 0:
        raise ValueError("train/eval/CANON arrivals must all be non-empty")
    return {
        "train_n": int(len(train)),
        "train_start_day": float(train.min() / DAY_S),
        "train_end_day": float(train.max() / DAY_S),
        "eval_n": int(len(eval_all)),
        "eval_start_day": float(eval_all.min() / DAY_S),
        "eval_end_day": float(eval_all.max() / DAY_S),
        "canon_n": int(len(canon_times)),
        "canon_start_day": float(canon_times.min() / DAY_S),
        "canon_end_day": float(canon_times.max() / DAY_S),
        "canon_span_h": float((canon_times.max() - canon_times.min()) / 3600.0),
        "train_ends_before_canon": bool(train.max() < canon_times.min()),
    }


def phase_aligned_edges(start_s: float, end_s: float, bin_s: int) -> np.ndarray:
    phase = float(start_s) % float(bin_s)
    first = phase
    if first > 0.0:
        first -= float(bin_s)
    edges = np.arange(first, math.ceil((end_s - first) / bin_s) * bin_s + first + bin_s, bin_s)
    return edges


def make_count_data(
    arrival_s: np.ndarray,
    canon_start_s: float,
    canon_end_s: float,
    *,
    bin_s: int,
    lag_k: int,
) -> CountData:
    total_end = max(float(arrival_s.max()), float(canon_end_s)) + float(bin_s)
    edges = phase_aligned_edges(float(canon_start_s), total_end, int(bin_s))
    counts, _ = np.histogram(arrival_s, bins=edges)
    starts = edges[:-1]
    rows: list[list[float]] = []
    indices: list[int] = []
    feature_names = (
        tuple(f"lag_{i}" for i in range(lag_k, 0, -1))
        + (
            "roll3_mean",
            "roll3_max",
            "roll6_mean",
            "roll6_max",
            "roll12_mean",
            "roll12_max",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        )
    )
    for i in range(lag_k, len(counts)):
        recent = counts[i - lag_k : i].astype(float)
        roll3 = recent[-min(3, len(recent)) :]
        roll6 = recent[-min(6, len(recent)) :]
        roll12 = recent[-min(12, len(recent)) :]
        hour = (starts[i] / 3600.0) % 24.0
        dow = (starts[i] / DAY_S) % 7.0
        rows.append(
            [
                *recent.tolist(),
                float(roll3.mean()),
                float(roll3.max()),
                float(roll6.mean()),
                float(roll6.max()),
                float(roll12.mean()),
                float(roll12.max()),
                math.sin(2.0 * math.pi * hour / 24.0),
                math.cos(2.0 * math.pi * hour / 24.0),
                math.sin(2.0 * math.pi * dow / 7.0),
                math.cos(2.0 * math.pi * dow / 7.0),
            ]
        )
        indices.append(i)
    idx = np.asarray(indices, dtype=int)
    train_mask = starts[idx] < TRAIN_END_S
    eval_mask = (starts[idx] >= canon_start_s) & (starts[idx] < canon_end_s)
    return CountData(
        bin_s=int(bin_s),
        edges=edges,
        starts=starts,
        counts=counts.astype(float),
        indices=idx,
        features=np.asarray(rows, dtype=float),
        train_mask=train_mask,
        eval_mask=eval_mask,
        feature_names=feature_names,
    )


def fit_nb_alpha(y: np.ndarray, mu: np.ndarray) -> tuple[float, float]:
    y = y.astype(float)
    mu = np.maximum(mu.astype(float), 1e-9)
    numerator = float(np.sum((y - mu) ** 2 - mu))
    denominator = float(np.sum(mu**2))
    alpha = max(0.0, numerator / max(denominator, 1e-9))
    if alpha <= 1e-9:
        return 0.0, math.inf
    return alpha, 1.0 / alpha


def _chronological_holdout_nb_alpha(data: CountData, y: np.ndarray) -> float:
    train_positions = np.flatnonzero(data.train_mask)
    if train_positions.size < 30:
        return 0.0
    cut = max(10, int(train_positions.size * 0.8))
    if cut >= train_positions.size:
        return 0.0
    fit_pos = train_positions[:cut]
    val_pos = train_positions[cut:]
    model = HistGradientBoostingRegressor(
        random_state=SEED,
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=0.01,
    )
    model.fit(data.features[fit_pos], y[fit_pos])
    val_mu = np.maximum(0.0, model.predict(data.features[val_pos]))
    alpha, _ = fit_nb_alpha(y[val_pos], val_mu)
    return float(alpha)


def poisson_quantile(mu: np.ndarray, q: float) -> np.ndarray:
    return poisson.ppf(q, np.maximum(mu, 1e-9)).astype(float)


def nb_quantile(mu: np.ndarray, alpha: float, q: float) -> np.ndarray:
    mu = np.maximum(mu.astype(float), 1e-9)
    if alpha <= 1e-9:
        return poisson_quantile(mu, q)
    size = 1.0 / alpha
    prob = size / (size + mu)
    return nbinom.ppf(q, size, prob).astype(float)


def fit_count_model(data: CountData, *, quantile: float = 0.90) -> CountModel:
    if not data.train_mask.any() or not data.eval_mask.any():
        raise ValueError("both train and eval masks must contain at least one bin")
    y = data.counts[data.indices]
    X_train = data.features[data.train_mask]
    y_train = y[data.train_mask]
    X_eval = data.features[data.eval_mask]
    eval_idx = data.indices[data.eval_mask]
    model_mean = HistGradientBoostingRegressor(
        random_state=SEED,
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=0.01,
    )
    model_q90 = HistGradientBoostingRegressor(
        random_state=SEED,
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=0.01,
        loss="quantile",
        quantile=float(quantile),
    )
    model_mean.fit(X_train, y_train)
    model_q90.fit(X_train, y_train)
    train_mean = np.maximum(0.0, model_mean.predict(X_train))
    train_fano = float(np.var(y_train, ddof=1) / max(np.mean(y_train), 1e-9))
    train_global_alpha = max(
        0.0,
        (train_fano - 1.0) / max(float(np.mean(y_train)), 1e-9),
    )
    in_sample_alpha, _ = fit_nb_alpha(y_train, train_mean)
    holdout_alpha = _chronological_holdout_nb_alpha(data, y)
    alpha = max(in_sample_alpha, holdout_alpha)
    if alpha <= 1e-9 and train_fano > 1.0:
        alpha = train_global_alpha
    nb_k = math.inf if alpha <= 1e-9 else 1.0 / alpha
    mean_pred = np.maximum(0.0, model_mean.predict(X_eval))
    q90_pred = np.maximum(mean_pred, model_q90.predict(X_eval))
    actual = data.counts[eval_idx]
    persistence = np.asarray(
        [data.counts[i - 1] if i > 0 else 0.0 for i in eval_idx],
        dtype=float,
    )
    season_lag = int(round(DAY_S / data.bin_s))
    seasonal = np.asarray(
        [
            data.counts[i - season_lag]
            if i >= season_lag
            else float(y_train.mean())
            for i in eval_idx
        ],
        dtype=float,
    )
    global_mean = np.full(len(eval_idx), float(y_train.mean()))
    return CountModel(
        bin_s=data.bin_s,
        starts=data.starts[eval_idx],
        actual=actual,
        train_actual=y_train,
        mean_pred=mean_pred,
        q90_pred=q90_pred,
        poisson_k90=poisson_quantile(mean_pred, quantile),
        nb_k90=nb_quantile(mean_pred, alpha, quantile),
        persistence=persistence,
        seasonal_only=seasonal,
        global_mean=global_mean,
        nb_alpha=float(alpha),
        nb_k=float(nb_k),
        train_fano=train_fano,
    )


def metric_row(
    *,
    bin_s: int,
    method: str,
    actual: np.ndarray,
    pred: np.ndarray,
    kind: str,
) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return {
        "bin_s": int(bin_s),
        "method": method,
        "kind": kind,
        "n_bins": int(len(actual)),
        "actual_sum": float(actual.sum()),
        "pred_sum": float(pred.sum()),
        "actual_mean": float(actual.mean()),
        "pred_mean": float(pred.mean()),
        "r2": float(r2_score(actual, pred)) if len(actual) > 1 else math.nan,
        "mae": float(mean_absolute_error(actual, pred)),
        "coverage_actual_le_pred": float(np.mean(actual <= pred)),
    }


def accuracy_rows(model: CountModel) -> list[dict[str, Any]]:
    rows = [
        metric_row(
            bin_s=model.bin_s,
            method="hgb_mean",
            actual=model.actual,
            pred=model.mean_pred,
            kind="mean",
        ),
        metric_row(
            bin_s=model.bin_s,
            method="hgb_q90",
            actual=model.actual,
            pred=model.q90_pred,
            kind="quantile",
        ),
        metric_row(
            bin_s=model.bin_s,
            method="poisson_k90",
            actual=model.actual,
            pred=model.poisson_k90,
            kind="quantile",
        ),
        metric_row(
            bin_s=model.bin_s,
            method="nb_k90",
            actual=model.actual,
            pred=model.nb_k90,
            kind="quantile",
        ),
        metric_row(
            bin_s=model.bin_s,
            method="persistence",
            actual=model.actual,
            pred=model.persistence,
            kind="baseline",
        ),
        metric_row(
            bin_s=model.bin_s,
            method="seasonal_only",
            actual=model.actual,
            pred=model.seasonal_only,
            kind="baseline",
        ),
        metric_row(
            bin_s=model.bin_s,
            method="global_mean",
            actual=model.actual,
            pred=model.global_mean,
            kind="baseline",
        ),
    ]
    return rows


def expand_300_to_60(model300: CountModel, starts60: np.ndarray, *, values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(starts60), dtype=float)
    for start, value in zip(model300.starts, values):
        mask = (starts60 >= start) & (starts60 < start + model300.bin_s)
        out[mask] = float(value) / max(1.0, float(mask.sum()))
    return out


def topup_60s_from_300(
    *,
    data60: CountData,
    model300: CountModel,
    quantile: float = 0.90,
    residual_window: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eval_idx = data60.indices[data60.eval_mask]
    starts60 = data60.starts[eval_idx]
    actual60 = data60.counts[eval_idx]
    base_mu60 = expand_300_to_60(model300, starts60, values=model300.mean_pred)
    base_nb60 = nb_quantile(base_mu60, model300.nb_alpha, quantile)
    all_starts = data60.starts[data60.indices]
    all_base = np.zeros_like(all_starts, dtype=float)
    eval_start = float(starts60[0])
    train_like = all_starts < eval_start
    if train_like.any():
        all_base[train_like] = np.mean(data60.counts[data60.indices][train_like])
    all_base[data60.eval_mask] = base_mu60
    history_counts = data60.counts[data60.indices]
    topup = np.zeros_like(actual60, dtype=float)
    for pos, idx in enumerate(np.flatnonzero(data60.eval_mask)):
        prior_residuals = history_counts[:idx] - all_base[:idx]
        residual_q = recent_residual_quantile(
            prior_residuals,
            base=0.0,
            quantile=quantile,
            residual_window=residual_window,
        )
        topup[pos] = max(base_nb60[pos], math.ceil(max(0.0, base_mu60[pos] + residual_q)))
    return actual60, base_mu60, topup


def simulate_static(times: np.ndarray, keepalive_s: float) -> tuple[int, float]:
    free_expiries: list[float] = []
    cold = 0
    container_seconds = 0.0
    for t in sorted(times):
        free_expiries = [expiry for expiry in free_expiries if expiry >= t]
        if free_expiries:
            free_expiries.pop(0)
        else:
            cold += 1
        free_expiries.append(float(t) + keepalive_s)
        container_seconds += keepalive_s
    return cold, container_seconds


def simulate_oracle(times: np.ndarray, keepalive_s: float) -> tuple[int, float, int]:
    free_expiries: list[float] = []
    warmups = 0
    container_seconds = 0.0
    for t in sorted(times):
        free_expiries = [expiry for expiry in free_expiries if expiry >= t]
        if not free_expiries:
            warmups += 1
            free_expiries.append(float(t) + keepalive_s)
            container_seconds += LEAD_SEC + keepalive_s
        free_expiries.pop(0)
        free_expiries.append(float(t) + keepalive_s)
    return 0, container_seconds, warmups


def simulate_forecaster(
    times_abs: np.ndarray,
    bin_starts: np.ndarray,
    desired_k: np.ndarray,
    *,
    bin_s: int,
    keepalive_s: float,
) -> tuple[int, float, int]:
    live_expiries: list[float] = []
    warmups = 0
    container_seconds = 0.0
    events: list[tuple[float, str, int | None]] = []
    for start, k in zip(bin_starts, desired_k):
        events.append((float(start) - LEAD_SEC, "prewarm", int(math.ceil(max(0.0, k)))))
    for t in times_abs:
        events.append((float(t), "arrival", None))
    cold = 0
    for t, kind, k in sorted(events, key=lambda item: (item[0], 0 if item[1] == "prewarm" else 1)):
        live_expiries = [expiry for expiry in live_expiries if expiry >= t]
        if kind == "prewarm":
            have = len(live_expiries)
            add = max(0, int(k or 0) - have)
            warmups += add
            live_expiries.extend([t + float(bin_s) + LEAD_SEC + keepalive_s] * add)
            container_seconds += add * (float(bin_s) + LEAD_SEC + keepalive_s)
        else:
            if live_expiries:
                live_expiries.pop(0)
            else:
                cold += 1
            live_expiries.append(t + keepalive_s)
    return cold, container_seconds, warmups


def pool_frontier_rows(
    *,
    canon_times_abs: np.ndarray,
    canon_times_rel: np.ndarray,
    model300: CountModel,
    starts60: np.ndarray,
    topup60: np.ndarray,
    hgb_q90_60: np.ndarray,
    poisson_60: np.ndarray,
    nb_60: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for keepalive in [10.0, 60.0, 300.0]:
        cold, seconds = simulate_static(canon_times_rel, keepalive)
        rows.append(
            {
                "arm": f"static_keepalive_{int(keepalive)}s",
                "bin_s": 0,
                "entry_cold": cold,
                "entry_cold_rate": cold / len(canon_times_rel),
                "container_seconds": seconds,
                "warmups": 0,
            }
        )
    for name, starts, k, bin_s in [
        ("hgb_q90_300s", model300.starts, model300.q90_pred, model300.bin_s),
        ("poisson_k90_300s", model300.starts, model300.poisson_k90, model300.bin_s),
        ("nb_k90_300s", model300.starts, model300.nb_k90, model300.bin_s),
        ("hgb_q90_60s", starts60, hgb_q90_60, 60),
        ("poisson_k90_60s", starts60, poisson_60, 60),
        ("nb_k90_60s", starts60, nb_60, 60),
        ("nb_k90_60s_topup", starts60, topup60, 60),
    ]:
        cold, seconds, warmups = simulate_forecaster(
            canon_times_abs,
            starts,
            k,
            bin_s=int(bin_s),
            keepalive_s=10.0,
        )
        rows.append(
            {
                "arm": name,
                "bin_s": int(bin_s),
                "entry_cold": cold,
                "entry_cold_rate": cold / len(canon_times_abs),
                "container_seconds": seconds,
                "warmups": warmups,
            }
        )
    cold, seconds, warmups = simulate_oracle(canon_times_rel, 10.0)
    rows.append(
        {
            "arm": "oracle",
            "bin_s": 0,
            "entry_cold": cold,
            "entry_cold_rate": 0.0,
            "container_seconds": seconds,
            "warmups": warmups,
        }
    )
    return rows


def fano_rows(data: CountData, model: CountModel) -> list[dict[str, Any]]:
    train = model.train_actual
    values, counts = np.unique(train.astype(int), return_counts=True)
    rows = [
        {
            "bin_s": data.bin_s,
            "metric": "summary",
            "count_value": math.nan,
            "frequency": int(len(train)),
            "mean": float(np.mean(train)),
            "variance": float(np.var(train, ddof=1)),
            "fano": model.train_fano,
            "nb_alpha": model.nb_alpha,
            "nb_k": model.nb_k,
        }
    ]
    for value, frequency in zip(values, counts):
        rows.append(
            {
                "bin_s": data.bin_s,
                "metric": "histogram",
                "count_value": int(value),
                "frequency": int(frequency),
                "mean": math.nan,
                "variance": math.nan,
                "fano": math.nan,
                "nb_alpha": model.nb_alpha,
                "nb_k": model.nb_k,
            }
        )
    return rows


def run_evaluation(
    *,
    arrivals_path: str | Path = DEFAULT_ARRIVALS,
    canon_schedule_path: str | Path = DEFAULT_CANON_SCHEDULE,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    lag_k: int = 10,
    quantile: float = 0.90,
) -> dict[str, pd.DataFrame]:
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrivals, canon = load_inputs(arrivals_path, canon_schedule_path)
    metadata = validate_split(arrivals, canon)
    all_times = arrivals["arrival_s"].to_numpy(float)
    canon_abs = canon["source_start_s"].to_numpy(float)
    canon_rel = canon["target_offset_ms"].to_numpy(float) / 1000.0
    canon_start = float(canon_abs.min())
    canon_end = float(canon_abs.max()) + 1.0
    data300 = make_count_data(all_times, canon_start, canon_end, bin_s=300, lag_k=lag_k)
    data60 = make_count_data(all_times, canon_start, canon_end, bin_s=60, lag_k=lag_k)
    model300 = fit_count_model(data300, quantile=quantile)
    model60 = fit_count_model(data60, quantile=quantile)
    actual60, base_mu60, topup60 = topup_60s_from_300(
        data60=data60,
        model300=model300,
        quantile=quantile,
    )
    starts60 = data60.starts[data60.indices[data60.eval_mask]]
    accuracy = pd.DataFrame(accuracy_rows(model300) + accuracy_rows(model60))
    accuracy = pd.concat(
        [
            accuracy,
            pd.DataFrame(
                [
                    metric_row(
                        bin_s=60,
                        method="nb_k90_60s_topup",
                        actual=actual60,
                        pred=topup60,
                        kind="quantile_topup",
                    ),
                    metric_row(
                        bin_s=60,
                        method="base_mu_from_300",
                        actual=actual60,
                        pred=base_mu60,
                        kind="mean_topdown",
                    ),
                ]
            ),
        ],
        ignore_index=True,
    )
    pool = pd.DataFrame(
        pool_frontier_rows(
            canon_times_abs=canon_abs,
            canon_times_rel=canon_rel,
            model300=model300,
            starts60=starts60,
            topup60=topup60,
            hgb_q90_60=model60.q90_pred,
            poisson_60=model60.poisson_k90,
            nb_60=model60.nb_k90,
        )
    )
    fano = pd.DataFrame(fano_rows(data300, model300) + fano_rows(data60, model60))
    metadata.update(
        {
            "bin300_actual_sum": float(model300.actual.sum()),
            "bin60_actual_sum": float(model60.actual.sum()),
            "bin300_train_fano": model300.train_fano,
            "bin300_nb_alpha": model300.nb_alpha,
            "bin300_nb_k": model300.nb_k,
            "bin60_train_fano": model60.train_fano,
            "bin60_nb_alpha": model60.nb_alpha,
            "bin60_nb_k": model60.nb_k,
            "source_arrivals": str(resolve(arrivals_path)),
            "canon_schedule": str(resolve(canon_schedule_path)),
        }
    )
    accuracy.to_csv(out / "accuracy.csv", index=False)
    pool.to_csv(out / "pool_frontier.csv", index=False)
    fano.to_csv(out / "fano_and_count_dist.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    write_report(out, metadata, accuracy, pool)
    return {
        "accuracy": accuracy,
        "pool_frontier": pool,
        "fano_and_count_dist": fano,
        "metadata": pd.DataFrame([metadata]),
    }


def write_report(
    out: Path,
    metadata: dict[str, Any],
    accuracy: pd.DataFrame,
    pool: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Step 4 CANON Count Forecaster\n")
    lines.append("\n## Data Split\n")
    lines.append(
        f"- Train n={metadata['train_n']} day {metadata['train_start_day']:.4f}-{metadata['train_end_day']:.4f}.\n"
    )
    lines.append(
        f"- Eval full n={metadata['eval_n']} day {metadata['eval_start_day']:.4f}-{metadata['eval_end_day']:.4f}.\n"
    )
    lines.append(
        f"- CANON n={metadata['canon_n']} day {metadata['canon_start_day']:.4f}-{metadata['canon_end_day']:.4f}, span={metadata['canon_span_h']:.2f}h.\n"
    )
    lines.append(f"- Train ends before CANON: `{metadata['train_ends_before_canon']}`.\n")
    lines.append("\n## Count Dispersion\n")
    lines.append(
        "| bin_s | train_fano | nb_alpha | nb_k |\n|---:|---:|---:|---:|\n"
        f"| 300 | {metadata['bin300_train_fano']:.4f} | {metadata['bin300_nb_alpha']:.6f} | {metadata['bin300_nb_k']:.4f} |\n"
        f"| 60 | {metadata['bin60_train_fano']:.4f} | {metadata['bin60_nb_alpha']:.6f} | {metadata['bin60_nb_k']:.4f} |\n"
    )
    lines.append("\n## Accuracy\n\n")
    lines.append(accuracy.round(6).to_markdown(index=False))
    lines.append("\n\n## Entry Pool Frontier\n\n")
    lines.append(pool.round(6).to_markdown(index=False))
    nb = pool[pool["arm"].eq("nb_k90_300s")].iloc[0]
    pois = pool[pool["arm"].eq("poisson_k90_300s")].iloc[0]
    hgb = pool[pool["arm"].eq("hgb_q90_300s")].iloc[0]
    nb_top = pool[pool["arm"].eq("nb_k90_60s_topup")].iloc[0]
    lines.append("\n\n## Verdict\n")
    lines.append(
        f"- 300s NB-K cold={nb.entry_cold_rate:.2%}, seconds={nb.container_seconds:.1f}; "
        f"Poisson-K cold={pois.entry_cold_rate:.2%}, seconds={pois.container_seconds:.1f}; "
        f"HGBR-q90 cold={hgb.entry_cold_rate:.2%}, seconds={hgb.container_seconds:.1f}.\n"
    )
    lines.append(
        f"- 60s NB top-up cold={nb_top.entry_cold_rate:.2%}, seconds={nb_top.container_seconds:.1f}. "
        "This is the online-friendly candidate because it keeps the 300s signal but reacts to recent residuals.\n"
    )
    lines.append(
        "- NB is preferable when it improves coverage/cold rate at comparable or lower container-seconds; "
        "otherwise the table exposes the cost of over-dispersion explicitly.\n"
    )
    (out / "report.md").write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrivals", default=str(DEFAULT_ARRIVALS))
    parser.add_argument("--canon-schedule", default=str(DEFAULT_CANON_SCHEDULE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--lag-k", type=int, default=10)
    parser.add_argument("--quantile", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_evaluation(
        arrivals_path=args.arrivals,
        canon_schedule_path=args.canon_schedule,
        out_dir=args.out_dir,
        lag_k=args.lag_k,
        quantile=args.quantile,
    )
    metadata = outputs["metadata"].iloc[0].to_dict()
    print("SPLIT")
    for key in [
        "train_n",
        "train_start_day",
        "train_end_day",
        "eval_n",
        "canon_n",
        "canon_start_day",
        "canon_end_day",
        "train_ends_before_canon",
    ]:
        print(f"{key}={metadata[key]}")
    print("\nFANO")
    print(
        outputs["fano_and_count_dist"]
        .query("metric == 'summary'")
        .round(6)
        .to_string(index=False)
    )
    print("\nACCURACY")
    print(outputs["accuracy"].round(6).to_string(index=False))
    print("\nPOOL FRONTIER")
    print(outputs["pool_frontier"].round(6).to_string(index=False))
    print(f"\nWrote {resolve(args.out_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
