"""Entry-stage cold probability helpers."""

from __future__ import annotations

import math

import pandas as pd


def entry_cold_probability(
    predicted_arrivals: float,
    entry_prewarm_count: float,
    p_baseline_floor: float = 0.01,
) -> float:
    """
    Naive formula with floor:
      p = max(p_baseline_floor, 1 - prewarm / max(predicted, eps))
    """
    if p_baseline_floor < 0.0 or p_baseline_floor > 1.0:
        raise ValueError(f"p_baseline_floor must be in [0, 1], got {p_baseline_floor}")
    if predicted_arrivals < 0.0 or entry_prewarm_count < 0.0:
        raise ValueError("predicted_arrivals and entry_prewarm_count must be non-negative")
    eps = 1e-9
    uncovered = 1.0 - float(entry_prewarm_count) / max(float(predicted_arrivals), eps)
    return float(min(1.0, max(float(p_baseline_floor), uncovered)))


def calibrated_entry_cold_probability(
    predicted_arrivals: float,
    entry_prewarm_count: float,
    zero_prewarm_cold_rate: float,
    residual_floor: float = 0.01,
) -> float:
    """Cold probability calibrated to observed zero-prewarm natural reuse."""
    if predicted_arrivals < 0.0 or entry_prewarm_count < 0.0:
        raise ValueError("predicted_arrivals and entry_prewarm_count must be non-negative")
    if not 0.0 <= zero_prewarm_cold_rate <= 1.0:
        raise ValueError(f"zero_prewarm_cold_rate must be in [0, 1], got {zero_prewarm_cold_rate}")
    if not 0.0 <= residual_floor <= 1.0:
        raise ValueError(f"residual_floor must be in [0, 1], got {residual_floor}")
    eps = 1e-9
    coverage = min(1.0, float(entry_prewarm_count) / max(float(predicted_arrivals), eps))
    natural_reuse_rate = float(zero_prewarm_cold_rate) * (1.0 - coverage)
    return float(min(1.0, max(float(residual_floor), natural_reuse_rate)))


def calibrate_p_baseline(trace_csv_path: str, entry_stage_name: str = "detect_object") -> float:
    """
    From a real trace, compute observed cold rate when prewarm=0:
      p_baseline = cold_count / total_invocations for entry stage.
    """
    df = pd.read_csv(trace_csv_path)
    required = {"stage_name", "cold_like"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"trace missing required columns: {missing}")
    entry = df[df["stage_name"] == entry_stage_name].copy()
    if entry.empty:
        raise ValueError(f"entry stage {entry_stage_name!r} not found in trace")
    cold = entry["cold_like"].astype(str).str.strip().str.lower().eq("true")
    rate = float(cold.mean())
    if not math.isfinite(rate):
        raise ValueError("computed non-finite p_baseline")
    return rate
