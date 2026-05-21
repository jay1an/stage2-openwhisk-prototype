import pandas as pd

from runner.stage5_control.propagator import (
    clear_propagator_cache,
    propagate_entry_to_stage,
)


def test_propagate_entry_to_stage_uses_weighted_offsets():
    clear_propagator_cache()
    entry = pd.DataFrame(
        [
            {"workflow_name": "wf", "method": "m", "target_window": 8, "policy": "p95", "forecast_count": 10},
            {"workflow_name": "wf", "method": "m", "target_window": 9, "policy": "p95", "forecast_count": 20},
            {"workflow_name": "wf", "method": "m", "target_window": 10, "policy": "p95", "forecast_count": 30},
        ]
    )
    kernel = pd.DataFrame(
        [
            {"workflow_name": "wf", "stage_name": "stage_a", "prev_state": "warm", "offset_windows": 0, "probability": 0.25},
            {"workflow_name": "wf", "stage_name": "stage_a", "prev_state": "warm", "offset_windows": 1, "probability": 0.75},
            {"workflow_name": "wf", "stage_name": "stage_a", "prev_state": "cold", "offset_windows": 2, "probability": 1.0},
        ]
    )

    value = propagate_entry_to_stage(
        entry,
        kernel,
        workflow_name="wf",
        stage_name="stage_a",
        target_window=10,
        policy="p95",
        prev_warm=True,
    )

    assert value == 0.25 * 30 + 0.75 * 20


def test_propagate_entry_to_stage_switches_on_prev_warm_state():
    clear_propagator_cache()
    entry = pd.DataFrame(
        [
            {"workflow_name": "wf", "method": "m", "target_window": 8, "policy": "p95", "forecast_count": 10},
            {"workflow_name": "wf", "method": "m", "target_window": 10, "policy": "p95", "forecast_count": 30},
        ]
    )
    kernel = pd.DataFrame(
        [
            {"workflow_name": "wf", "stage_name": "stage_a", "prev_state": "warm", "offset_windows": 0, "probability": 1.0},
            {"workflow_name": "wf", "stage_name": "stage_a", "prev_state": "cold", "offset_windows": 2, "probability": 1.0},
        ]
    )

    warm_value = propagate_entry_to_stage(
        entry,
        kernel,
        workflow_name="wf",
        stage_name="stage_a",
        target_window=10,
        policy="p95",
        prev_warm=True,
    )
    cold_value = propagate_entry_to_stage(
        entry,
        kernel,
        workflow_name="wf",
        stage_name="stage_a",
        target_window=10,
        policy="p95",
        prev_warm=False,
    )

    assert warm_value == 30
    assert cold_value == 10
