"""Resource scaling helpers for analytical plan-risk evaluation."""

from __future__ import annotations

import math

import pandas as pd

from runner.stage4_risk.dag_aggregation import LogNormalParams


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
    steps = math.ceil(float(memory_mb) / float(step_memory_mb))
    millicpus = min(int(max_millicpus), int(base_millicpus) * steps)
    return float(millicpus) / 1000.0


def amdahl_predict_mean(stage_name: str, cpu_cores: float, amdahl_params: pd.DataFrame) -> float:
    """Predict mean action_duration_ms at a CPU tier using fitted Amdahl params."""
    if cpu_cores <= 0.0 or not math.isfinite(cpu_cores):
        raise ValueError(f"cpu_cores must be positive and finite, got {cpu_cores}")
    required = {"stage_name", "S_ms", "P_ms", "C_ms"}
    missing = sorted(required.difference(amdahl_params.columns))
    if missing:
        raise ValueError(f"amdahl_params missing required columns: {missing}")

    row = amdahl_params[amdahl_params["stage_name"] == stage_name]
    if row.empty:
        raise ValueError(f"missing Amdahl params for stage={stage_name}")
    if len(row) > 1:
        raise ValueError(f"duplicate Amdahl params for stage={stage_name}")

    row0 = row.iloc[0]
    s_ms = float(row0["S_ms"])
    p_ms = float(row0["P_ms"])
    c_ms = float(row0["C_ms"])
    if "W_eff_breakpoint" in amdahl_params.columns:
        w_eff = float(row0["W_eff_breakpoint"])
    elif "W_eff" in amdahl_params.columns:
        w_eff = float(row0["W_eff"])
    else:
        w_eff = max(1.0, math.floor(cpu_cores))
    w_eff = max(1.0, w_eff)

    serial_divisor = min(cpu_cores, 1.0)
    parallel_divisor = min(cpu_cores, w_eff)
    predicted = s_ms / serial_divisor + p_ms / parallel_divisor + c_ms
    if predicted <= 0.0 or not math.isfinite(predicted):
        raise ValueError(f"invalid Amdahl prediction for stage={stage_name}: {predicted}")
    return float(predicted)


def scale_lognormal(base_params: LogNormalParams, target_mean: float) -> LogNormalParams:
    """Scale a lognormal to a target mean while keeping CV constant."""
    if target_mean <= 0.0 or not math.isfinite(target_mean):
        raise ValueError(f"target_mean must be positive and finite, got {target_mean}")
    sigma = base_params.sigma
    mu = math.log(float(target_mean)) - (sigma**2) / 2.0
    return LogNormalParams(mu=mu, sigma=sigma)


def scale_stage_for_memory_tier(
    stage_name: str,
    latency_class: str,
    target_memory_mb: int,
    base_memory_mb: int,
    base_params: LogNormalParams,
    amdahl_params: pd.DataFrame,
    cold_overhead_ms: float | None = None,
) -> LogNormalParams:
    """Scale a stage lognormal from base memory to target memory.

    Warm dispatch latency is scaled by the action-duration Amdahl ratio.
    Cold dispatch latency is split into a CPU-scaled action component and a
    CPU-independent cold-overhead component when cold_overhead_ms is supplied.
    """
    base_cpu = memory_to_cpu_cores(base_memory_mb)
    target_cpu = memory_to_cpu_cores(target_memory_mb)
    base_action_mean = amdahl_predict_mean(stage_name, base_cpu, amdahl_params)
    target_action_mean = amdahl_predict_mean(stage_name, target_cpu, amdahl_params)
    ratio = target_action_mean / base_action_mean

    base_dispatch_mean = base_params.mean
    if str(latency_class) == "warm" or cold_overhead_ms is None:
        target_dispatch_mean = base_dispatch_mean * ratio
    else:
        overhead = max(0.0, float(cold_overhead_ms))
        overhead = min(overhead, base_dispatch_mean * 0.95)
        base_action_part = max(1e-9, base_dispatch_mean - overhead)
        target_dispatch_mean = overhead + base_action_part * ratio
    return scale_lognormal(base_params, target_dispatch_mean)
