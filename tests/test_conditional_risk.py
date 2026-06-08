from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from runner.stage4_risk.dag_aggregation import (
    LogNormalParams,
    aggregate_dag,
    conditional_risk,
)
from runner.workflow import load_workflow


ROOT = Path(__file__).resolve().parents[1]


def mean_lognormal(mean: float, sigma: float) -> LogNormalParams:
    return LogNormalParams(mu=math.log(mean) - sigma**2 / 2.0, sigma=sigma)


def civic_workflow():
    return load_workflow(str(ROOT / "configs" / "civic_alert_flow.yaml"))


def civic_stage_dists() -> dict[str, LogNormalParams]:
    return {
        "detect_object": mean_lognormal(1800.0, 0.05),
        "estimate_pose": mean_lognormal(1600.0, 0.05),
        "match_face": mean_lognormal(2200.0, 0.06),
        "classify_scene": mean_lognormal(1650.0, 0.05),
        "translate_alert": mean_lognormal(1450.0, 0.05),
    }


def test_conditional_risk_matches_mc_for_partial_civic_alert(capsys):
    workflow = civic_workflow()
    dists = civic_stage_dists()
    completed = {
        "detect_object": 1500.0,
        "estimate_pose": 3300.0,
    }
    analytical_dist = aggregate_dag(workflow, dists, fixed_finish=completed)
    slos = [analytical_dist.quantile(p) for p in (0.20, 0.50, 0.80)]

    rng = np.random.default_rng(20260608)
    samples = {
        stage: rng.lognormal(mean=dist.mu, sigma=dist.sigma, size=200_000)
        for stage, dist in dists.items()
    }
    match_finish = completed["estimate_pose"] + samples["match_face"]
    classify_start = np.maximum(completed["detect_object"], match_finish)
    e2e_samples = classify_start + samples["classify_scene"] + samples["translate_alert"]

    print("R2 conditional MC validation")
    for slo in slos:
        analytical = conditional_risk(workflow, dists, completed, slo)
        empirical = float(np.mean(e2e_samples > slo))
        abs_error = abs(analytical - empirical)
        print(
            f"slo_ms={slo:.3f} analytical={analytical:.6f} "
            f"mc={empirical:.6f} abs_error={abs_error:.6f}"
        )
        assert 0.02 > abs_error

    # Keep pytest from swallowing the table unless this test is run without -s.
    captured = capsys.readouterr()
    print(captured.out, end="")


def test_empty_completed_matches_full_aggregate_survival():
    workflow = civic_workflow()
    dists = civic_stage_dists()
    slo_ms = 9000.0
    expected = aggregate_dag(workflow, dists).survival(slo_ms)
    actual = conditional_risk(workflow, dists, {}, slo_ms)
    print(f"R3 empty completed: expected={expected:.12f} actual={actual:.12f}")
    assert actual == expected


def test_all_nodes_completed_is_deterministic_zero_or_one():
    workflow = civic_workflow()
    dists = civic_stage_dists()
    completed = {
        "detect_object": 1000.0,
        "estimate_pose": 2100.0,
        "match_face": 3300.0,
        "classify_scene": 4500.0,
        "translate_alert": 5600.0,
    }
    e2e = aggregate_dag(workflow, dists, fixed_finish=completed)
    print(f"R3 all completed: mean={e2e.mean:.3f} sigma={e2e.sigma:.3f}")
    assert e2e.mean == pytest.approx(5600.0)
    assert e2e.sigma == 0.0
    assert conditional_risk(workflow, dists, completed, 5000.0) == 1.0
    assert conditional_risk(workflow, dists, completed, 6000.0) == 0.0


def test_fixed_source_feeding_fan_in_remains_finite():
    workflow = civic_workflow()
    dists = civic_stage_dists()
    e2e = aggregate_dag(
        workflow,
        dists,
        fixed_finish={"detect_object": 1500.0},
    )
    risk = conditional_risk(
        workflow,
        dists,
        {"detect_object": 1500.0},
        slo_ms=8500.0,
    )
    print(
        f"R3 fixed source fan-in: mu={e2e.mu:.6f} sigma={e2e.sigma:.6f} "
        f"mean={e2e.mean:.3f} risk={risk:.6f}"
    )
    assert math.isfinite(e2e.mu)
    assert math.isfinite(e2e.sigma)
    assert e2e.mean > 0.0
    assert 0.0 <= risk <= 1.0


def test_fixed_finish_input_validation():
    workflow = civic_workflow()
    dists = civic_stage_dists()

    cases = [
        ("unknown fixed node", {"not_a_stage": 1000.0}),
        ("negative fixed finish", {"detect_object": -1.0}),
        ("zero fixed finish", {"detect_object": 0.0}),
        ("nan fixed finish", {"detect_object": math.nan}),
        ("fixed parent incomplete", {"estimate_pose": 2500.0}),
    ]
    for label, fixed_finish in cases:
        with pytest.raises(ValueError) as excinfo:
            aggregate_dag(workflow, dists, fixed_finish=fixed_finish)
        print(f"R4 {label}: {excinfo.value}")
