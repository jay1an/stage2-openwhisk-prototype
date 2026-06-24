"""Resource scaling helpers for path-2 risk estimates.

The current production scaling path uses the D3 cubic-spline model selected in
P3.1-retry.  The spline predicts per-stage warm action duration from the
extended 9-tier sweep.  Cold dispatch latency is modeled as warm action time
plus the cleansed per-tier cold overhead.
"""

from __future__ import annotations

import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .dag_aggregation import LogNormalParams


DEFAULT_WARM_SPLINES = "reports/stage6_resource_models_v2/per_stage_warm_splines.pkl"
DEFAULT_COLD_OVERHEAD = "reports/stage6_resource_models_v2/cold_overhead_cleansed.csv"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _project_root() / candidate


def memory_to_cpu_cores(
    memory_mb: int,
    base_millicpus: int = 200,
    step_memory_mb: int = 256,
    max_millicpus: int = 3200,
) -> float:
    """OpenWhisk CPU scaling: cpu = min(max, base * ceil(memory/step)) / 1000."""

    if memory_mb <= 0:
        raise ValueError(f"memory_mb must be positive, got {memory_mb}")
    if base_millicpus <= 0 or step_memory_mb <= 0 or max_millicpus <= 0:
        raise ValueError("CPU scaling constants must be positive")

    steps = math.ceil(memory_mb / step_memory_mb)
    millicpus = min(max_millicpus, base_millicpus * steps)
    return millicpus / 1000.0


@lru_cache(maxsize=None)
def load_warm_splines(path: str | Path = DEFAULT_WARM_SPLINES) -> dict[str, Any]:
    """Load per-stage warm CubicSpline objects from the P3.1-retry artifact."""

    resolved = _resolve_path(path)
    with resolved.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "splines" in payload:
        return payload["splines"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported warm spline artifact format: {resolved}")


@lru_cache(maxsize=None)
def load_cleansed_cold_overhead(
    path: str | Path = DEFAULT_COLD_OVERHEAD,
) -> dict[tuple[str, int], float]:
    """Load cleansed cold overhead values keyed by ``(stage_name, tier_mb)``."""

    resolved = _resolve_path(path)
    df = pd.read_csv(resolved)
    required = {"stage_name", "tier_mb", "cleansed_cold_overhead_ms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{resolved} is missing columns: {sorted(missing)}")
    return {
        (str(row["stage_name"]), int(row["tier_mb"])): float(
            row["cleansed_cold_overhead_ms"]
        )
        for _, row in df.iterrows()
    }


def spline_predict_warm_mean(
    stage_name: str,
    cpu_cores: float,
    splines: dict[str, Any],
) -> float:
    """Predict warm action duration in ms using the selected D3 spline."""

    if stage_name not in splines:
        raise KeyError(f"No warm spline found for stage {stage_name!r}")
    spline = splines[stage_name]
    if hasattr(spline, "x"):
        min_cpu = float(spline.x[0])
        max_cpu = float(spline.x[-1])
        if cpu_cores < min_cpu or cpu_cores > max_cpu:
            raise ValueError(
                f"cpu_cores={cpu_cores:.3f} for {stage_name} is outside "
                f"the fitted spline range [{min_cpu:.3f}, {max_cpu:.3f}]"
            )
    prediction = float(spline(cpu_cores))
    if prediction <= 0:
        raise ValueError(
            f"Spline predicted non-positive warm mean for {stage_name}: {prediction}"
        )
    return prediction


def cold_overhead_for_tier(
    stage_name: str,
    memory_mb: int,
    cold_overhead_table: dict[tuple[str, int], float],
) -> float:
    """Look up cleansed cold overhead in ms for the exact stage/tier pair."""

    key = (stage_name, int(memory_mb))
    if key not in cold_overhead_table:
        available = sorted(tier for stage, tier in cold_overhead_table if stage == stage_name)
        raise KeyError(
            f"No cleansed cold overhead for {stage_name!r} at {memory_mb} MB; "
            f"available tiers: {available}"
        )
    value = float(cold_overhead_table[key])
    if value < 0:
        raise ValueError(f"Cold overhead for {stage_name} at {memory_mb} MB is negative")
    return value


def amdahl_predict_mean(
    stage_name: str,
    cpu_cores: float,
    amdahl_params: pd.DataFrame,
) -> float:
    """Deprecated: use :func:`spline_predict_warm_mean` instead."""

    row = amdahl_params.loc[amdahl_params["stage_name"] == stage_name]
    if row.empty:
        raise KeyError(f"No Amdahl params found for stage {stage_name!r}")
    params = row.iloc[0]
    s_ms = float(params["S_ms"])
    p_ms = float(params["P_ms"])
    c_ms = float(params["C_ms"])
    w_eff = float(params.get("W_eff_breakpoint", params.get("W_eff", 1.0)))
    return s_ms / min(cpu_cores, 1.0) + p_ms / min(cpu_cores, w_eff) + c_ms


def scale_lognormal_warm(
    base_params: LogNormalParams,
    target_warm_mean: float,
) -> LogNormalParams:
    """Scale a lognormal to a target mean while keeping CV constant."""

    if target_warm_mean <= 0:
        raise ValueError(f"target_warm_mean must be positive, got {target_warm_mean}")
    sigma = float(base_params.sigma)
    mu_new = math.log(target_warm_mean) - sigma**2 / 2.0
    return LogNormalParams(mu=mu_new, sigma=sigma)


def scale_lognormal(base_params: LogNormalParams, target_mean: float) -> LogNormalParams:
    """Backward-compatible alias for CV-constant lognormal scaling."""

    return scale_lognormal_warm(base_params, target_mean)


def _require_params(
    *,
    stage_name: str,
    latency_class: str,
    base_params: LogNormalParams | None,
    base_params_warm: LogNormalParams | None,
    base_params_cold: LogNormalParams | None,
) -> LogNormalParams:
    if latency_class == "warm":
        params = base_params_warm or base_params
    else:
        params = base_params_cold or base_params
    if params is None:
        raise ValueError(
            f"Missing base lognormal params for {stage_name!r} latency class "
            f"{latency_class!r}"
        )
    return params


def scale_stage_for_memory_tier(
    stage_name: str,
    latency_class: str,
    target_memory_mb: int,
    base_memory_mb: int = 1280,
    base_params: LogNormalParams | None = None,
    amdahl_params: pd.DataFrame | None = None,
    cold_overhead_ms: float | None = None,
    *,
    base_params_warm: LogNormalParams | None = None,
    base_params_cold: LogNormalParams | None = None,
    splines: dict[str, Any] | None = None,
    cold_overhead_table: dict[tuple[str, int], float] | None = None,
    contention_factor: float = 1.0,
) -> LogNormalParams:
    """Produce a stage lognormal at ``target_memory_mb`` using D3 scaling.

    ``contention_factor`` multiplies the spline-predicted warm mean to account
    for concurrency contention (measured ~+10% on real Azure replay vs the
    isolated sweep the spline is fit on). ``1.0`` keeps the isolated mean. It is
    applied to the warm execution mean only; the cold overhead is unscaled.

    ``latency_class`` accepts ``warm``, ``cold``, or ``cold_like``.  The legacy
    Amdahl arguments are retained for callers that still pass the old R3 API;
    they are intentionally ignored by the D3 path.
    """

    del base_memory_mb, amdahl_params, cold_overhead_ms

    class_key = "cold" if latency_class == "cold_like" else latency_class
    if class_key not in {"warm", "cold"}:
        raise ValueError(f"Unknown latency_class: {latency_class}")

    splines = splines or load_warm_splines()
    cold_overhead_table = cold_overhead_table or load_cleansed_cold_overhead()

    if contention_factor <= 0.0 or not math.isfinite(contention_factor):
        raise ValueError(f"contention_factor must be positive, got {contention_factor}")
    target_cpu = memory_to_cpu_cores(target_memory_mb)
    target_warm_mean = spline_predict_warm_mean(stage_name, target_cpu, splines) * contention_factor

    if class_key == "warm":
        params = _require_params(
            stage_name=stage_name,
            latency_class=class_key,
            base_params=base_params,
            base_params_warm=base_params_warm,
            base_params_cold=base_params_cold,
        )
        return scale_lognormal_warm(params, target_warm_mean)

    params = _require_params(
        stage_name=stage_name,
        latency_class=class_key,
        base_params=base_params,
        base_params_warm=base_params_warm,
        base_params_cold=base_params_cold,
    )
    cold_oh = cold_overhead_for_tier(stage_name, target_memory_mb, cold_overhead_table)
    return scale_lognormal_warm(params, target_warm_mean + cold_oh)
