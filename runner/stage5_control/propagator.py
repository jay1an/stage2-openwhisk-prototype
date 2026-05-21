from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class _PreparedInputs:
    entry_lookup: dict[tuple[str, int, str], float]
    kernel_lookup: dict[tuple[str, str, str], tuple[tuple[int, float], ...]]


_CACHE: dict[tuple[int, int], _PreparedInputs] = {}


def clear_propagator_cache() -> None:
    _CACHE.clear()


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")


def _prepared(entry_forecast: pd.DataFrame, delay_kernel: pd.DataFrame) -> _PreparedInputs:
    key = (id(entry_forecast), id(delay_kernel))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    _require_columns(
        entry_forecast,
        {"workflow_name", "target_window", "policy", "forecast_count"},
        "entry_forecast",
    )
    _require_columns(
        delay_kernel,
        {"workflow_name", "stage_name", "prev_state", "offset_windows", "probability"},
        "delay_kernel",
    )

    entries = entry_forecast.copy()
    entries["target_window"] = pd.to_numeric(entries["target_window"], errors="coerce")
    entries["forecast_count"] = pd.to_numeric(entries["forecast_count"], errors="coerce")
    entries = entries.dropna(subset=["target_window", "forecast_count"])
    entries["target_window"] = entries["target_window"].astype(int)
    entry_lookup = {
        (str(row["workflow_name"]), int(row["target_window"]), str(row["policy"])): float(row["forecast_count"])
        for row in entries.to_dict(orient="records")
    }

    kernels = delay_kernel.copy()
    kernels["offset_windows"] = pd.to_numeric(kernels["offset_windows"], errors="coerce")
    kernels["probability"] = pd.to_numeric(kernels["probability"], errors="coerce")
    kernels = kernels.dropna(subset=["offset_windows", "probability"])
    kernels["offset_windows"] = kernels["offset_windows"].astype(int)
    kernel_lookup: dict[tuple[str, str, str], tuple[tuple[int, float], ...]] = {}
    for key_cols, group in kernels.groupby(["workflow_name", "stage_name", "prev_state"], dropna=False):
        workflow_name, stage_name, prev_state = (str(value) for value in key_cols)
        items = tuple(
            (int(row["offset_windows"]), float(row["probability"]))
            for row in group.sort_values("offset_windows").to_dict(orient="records")
        )
        kernel_lookup[(workflow_name, stage_name, prev_state)] = items

    prepared = _PreparedInputs(entry_lookup=entry_lookup, kernel_lookup=kernel_lookup)
    _CACHE[key] = prepared
    return prepared


def propagate_entry_to_stage(
    entry_forecast: pd.DataFrame,
    delay_kernel: pd.DataFrame,
    *,
    workflow_name: str,
    stage_name: str,
    target_window: int,
    policy: str,
    prev_warm: bool,
) -> float:
    """Return the propagated count for one stage/window/policy.

    `prev_warm=True` selects the warm delay kernel; `False` selects the cold
    delay kernel. If a state-specific kernel is absent, the legacy `any` kernel
    is used as a compatibility fallback.
    """

    prepared = _prepared(entry_forecast, delay_kernel)
    prev_state = "warm" if prev_warm else "cold"
    kernel_items = prepared.kernel_lookup.get((workflow_name, stage_name, prev_state))
    if kernel_items is None:
        kernel_items = prepared.kernel_lookup.get((workflow_name, stage_name, "any"))
    if kernel_items is None:
        raise ValueError(
            f"delay_kernel has no rows for workflow={workflow_name}, stage={stage_name}, "
            f"prev_state={prev_state} or any"
        )

    total = 0.0
    for offset_windows, probability in kernel_items:
        source_window = int(target_window) - int(offset_windows)
        if source_window < 0:
            continue
        total += float(probability) * prepared.entry_lookup.get(
            (workflow_name, source_window, policy),
            0.0,
        )
    return max(0.0, float(total))
