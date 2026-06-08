#!/usr/bin/env python3
"""Verify online UP-only dynamic upgrade decisions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from runner.stage5_control.multi_slo_planner import (
    DEFAULT_SAFETY_FACTORS,
    DEFAULT_TIERS,
    STAGES,
    PlannerConfig,
    _dynamic_conditional_risk,
    dynamic_upgrade,
    load_reference_data,
)
from runner.workflow import load_workflow


PREMIUM_PLAN = {
    "detect_object": 1536,
    "estimate_pose": 1280,
    "match_face": 2048,
    "classify_scene": 3072,
    "translate_alert": 1024,
}


@dataclass(frozen=True)
class FixtureData:
    workflow: object
    ref_data: object


@pytest.fixture(scope="module")
def fixture_data() -> FixtureData:
    return FixtureData(
        workflow=load_workflow("configs/civic_alert_flow.yaml"),
        ref_data=load_reference_data(),
    )


def make_config(slo_ms: float, max_violation_rate: float = 0.05) -> PlannerConfig:
    return PlannerConfig(
        slo_ms=float(slo_ms),
        max_violation_rate=float(max_violation_rate),
        predicted_arrivals=5.0,
        tiers=list(DEFAULT_TIERS),
        safety_factors=list(DEFAULT_SAFETY_FACTORS),
        stages=list(STAGES),
    )


def risk_for(
    fixture_data: FixtureData,
    config: PlannerConfig,
    plan: dict[str, int],
    completed: dict[str, float],
    cold_upgrades: set[str] | None = None,
) -> float:
    return _dynamic_conditional_risk(
        config=config,
        ref_data=fixture_data.ref_data,
        workflow=fixture_data.workflow,
        memory_tier_per_stage=plan,
        completed_finish_ms=completed,
        cold_upgrade_stages=cold_upgrades or set(),
    )


def apply_changes(plan: dict[str, int], changes: dict[str, int] | None) -> dict[str, int]:
    out = dict(plan)
    if changes:
        out.update(changes)
    return out


def test_u1_noop_when_state_is_healthy(fixture_data: FixtureData) -> None:
    config = make_config(15000.0)
    completed: dict[str, float] = {}
    pending = list(STAGES)
    r0 = risk_for(fixture_data, config, PREMIUM_PLAN, completed)
    changes = dynamic_upgrade(
        config,
        fixture_data.ref_data,
        fixture_data.workflow,
        PREMIUM_PLAN,
        completed,
        pending,
    )
    print(
        "U1 no-op: "
        f"r0={r0:.12f} target={config.max_violation_rate:.6f} changes={changes}"
    )
    assert r0 <= config.max_violation_rate
    assert changes is None


def test_u2_recovery_improves_slow_state(fixture_data: FixtureData) -> None:
    config = make_config(25000.0)
    current = {stage: 512 for stage in STAGES}
    completed = {"detect_object": 3500.0, "estimate_pose": 5500.0}
    pending = [stage for stage in STAGES if stage not in completed]
    r0 = risk_for(fixture_data, config, current, completed)
    changes = dynamic_upgrade(
        config,
        fixture_data.ref_data,
        fixture_data.workflow,
        current,
        completed,
        pending,
    )
    upgraded = apply_changes(current, changes)
    r1 = risk_for(fixture_data, config, upgraded, completed, set(changes or {}))
    print(
        "U2 recovery: "
        f"r0={r0:.12f} r1={r1:.12f} target={config.max_violation_rate:.6f} "
        f"changes={changes}"
    )
    assert r0 > config.max_violation_rate
    assert changes is not None
    assert r1 < r0
    assert r1 <= config.max_violation_rate
    assert set(changes).issubset(pending)
    for stage_name, new_tier in changes.items():
        assert new_tier > current[stage_name]


def test_u3_decision_a_rejects_cold_worse_upgrade(fixture_data: FixtureData) -> None:
    config = make_config(15000.0)
    completed = {"detect_object": 4200.0}
    pending = [stage for stage in STAGES if stage not in completed]
    candidate_stage = "estimate_pose"
    candidate = dict(PREMIUM_PLAN)
    candidate[candidate_stage] = 1536

    r0 = risk_for(fixture_data, config, PREMIUM_PLAN, completed)
    warm_only_risk = risk_for(fixture_data, config, candidate, completed)
    cold_accounted_risk = risk_for(
        fixture_data,
        config,
        candidate,
        completed,
        {candidate_stage},
    )
    changes = dynamic_upgrade(
        config,
        fixture_data.ref_data,
        fixture_data.workflow,
        PREMIUM_PLAN,
        completed,
        pending,
    )
    print(
        "U3 decision-a: "
        f"stage={candidate_stage} r0={r0:.12f} "
        f"warm_only_risk={warm_only_risk:.12f} "
        f"cold_accounted_risk={cold_accounted_risk:.12f} "
        f"warm_delta={r0 - warm_only_risk:.12f} "
        f"cold_delta={r0 - cold_accounted_risk:.12f} changes={changes}"
    )
    assert warm_only_risk < r0
    assert cold_accounted_risk > r0
    assert changes is None or candidate_stage not in changes


def test_u4_boundaries_empty_pending_and_top_tier(fixture_data: FixtureData) -> None:
    config = make_config(15000.0)
    empty_pending_changes = dynamic_upgrade(
        config,
        fixture_data.ref_data,
        fixture_data.workflow,
        PREMIUM_PLAN,
        {"detect_object": 4200.0},
        [],
    )

    top_plan = {stage: 3840 for stage in STAGES}
    completed = {"detect_object": 6000.0}
    pending = [stage for stage in STAGES if stage not in completed]
    top_risk = risk_for(fixture_data, config, top_plan, completed)
    top_changes = dynamic_upgrade(
        config,
        fixture_data.ref_data,
        fixture_data.workflow,
        top_plan,
        completed,
        pending,
    )
    print(
        "U4 boundaries: "
        f"empty_pending_changes={empty_pending_changes} "
        f"top_tier_risk={top_risk:.12f} top_changes={top_changes}"
    )
    assert empty_pending_changes is None
    assert top_changes is None


def test_u5_up_only_never_downgrades_loose_state(fixture_data: FixtureData) -> None:
    config = make_config(60000.0)
    current = {stage: 512 for stage in STAGES}
    completed: dict[str, float] = {}
    pending = list(STAGES)
    r0 = risk_for(fixture_data, config, current, completed)
    changes = dynamic_upgrade(
        config,
        fixture_data.ref_data,
        fixture_data.workflow,
        current,
        completed,
        pending,
    )
    print(f"U5 UP-only loose state: r0={r0:.12f} changes={changes}")
    assert r0 <= config.max_violation_rate
    if changes:
        for stage_name, new_tier in changes.items():
            assert new_tier >= current[stage_name]
