#!/usr/bin/env python3
"""Verify runtime dynamic-upgrade integration points."""

from __future__ import annotations

from dataclasses import dataclass

from runner import run_workflow as run_workflow_module
from runner.run_workflow import _runtime_upgrade_decision, run_one_workflow
from runner.stage5_control.multi_slo_planner import (
    DEFAULT_SAFETY_FACTORS,
    DEFAULT_TIERS,
    STAGES,
    PlannerConfig,
    load_reference_data,
)
from runner.workflow import NodeSpec, WorkflowSpec, load_workflow, suffix_action_name


PREMIUM_PLAN = {
    "detect_object": 1536,
    "estimate_pose": 1280,
    "match_face": 2048,
    "classify_scene": 3072,
    "translate_alert": 1024,
}


@dataclass(frozen=True)
class FixtureData:
    workflow: WorkflowSpec
    ref_data: object


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def invoke_activation(self, action_name: str, payload: dict) -> dict:
        self.calls.append((action_name, dict(payload)))
        allocated_memory_mb = payload.get("allocated_memory_mb", "")
        allocated_cpu_cores = payload.get("allocated_cpu_cores", "")
        return {
            "response": {
                "status": "success",
                "result": {
                    "action_duration_ms": 1.0,
                    "allocated_memory_mb": allocated_memory_mb,
                    "allocated_cpu_cores": allocated_cpu_cores,
                    "cold_like": False,
                },
            },
            "annotations": [
                {"key": "waitTime", "value": 0},
                {"key": "limits", "value": {"memory": allocated_memory_mb}},
            ],
            "duration": 1,
            "activationId": "fake-activation-1",
            "version": "0.0.1",
        }


def make_config(slo_ms: float, max_violation_rate: float = 0.05) -> PlannerConfig:
    return PlannerConfig(
        slo_ms=float(slo_ms),
        max_violation_rate=float(max_violation_rate),
        predicted_arrivals=5.0,
        tiers=list(DEFAULT_TIERS),
        safety_factors=list(DEFAULT_SAFETY_FACTORS),
        stages=list(STAGES),
    )


def fixture_data() -> FixtureData:
    return FixtureData(
        workflow=load_workflow("configs/civic_alert_flow.yaml"),
        ref_data=load_reference_data(),
    )


def test_v1_runtime_upgrade_decision_slow_and_healthy_states() -> None:
    data = fixture_data()
    workflow_start = 1000.0

    slow_config = make_config(25000.0)
    slow_plan = {stage: 512 for stage in STAGES}
    slow_completed = {"detect_object", "estimate_pose"}
    slow_measured_completion_at = {
        "detect_object": workflow_start + 3.5,
        "estimate_pose": workflow_start + 5.5,
    }
    slow_changes = _runtime_upgrade_decision(
        workflow=data.workflow,
        normalized_plan=slow_plan,
        completed_names=slow_completed,
        running_names=set(),
        started_names=slow_completed,
        measured_completion_at=slow_measured_completion_at,
        workflow_start_monotonic=workflow_start,
        dynamic_config=slow_config,
        dynamic_ref_data=data.ref_data,
    )
    print(
        "V1 slow helper: "
        f"completed_finish_ms={{'detect_object': 3500.0, 'estimate_pose': 5500.0}} "
        f"changes={slow_changes}"
    )
    assert slow_changes is not None
    assert set(slow_changes).isdisjoint(slow_completed)
    for stage_name, new_tier in slow_changes.items():
        assert new_tier > slow_plan[stage_name]

    healthy_config = make_config(15000.0)
    healthy_changes = _runtime_upgrade_decision(
        workflow=data.workflow,
        normalized_plan=dict(PREMIUM_PLAN),
        completed_names=set(),
        running_names=set(),
        started_names=set(),
        measured_completion_at={},
        workflow_start_monotonic=workflow_start,
        dynamic_config=healthy_config,
        dynamic_ref_data=data.ref_data,
    )
    print(f"V1 healthy helper: changes={healthy_changes}")
    assert healthy_changes is None


def test_v2_dynamic_off_path_is_inert(monkeypatch) -> None:
    def fail_if_reached(*args, **kwargs):
        raise AssertionError("_runtime_upgrade_decision should not be reached")

    monkeypatch.setattr(
        run_workflow_module,
        "_runtime_upgrade_decision",
        fail_if_reached,
    )
    workflow = WorkflowSpec(
        workflow_name="tiny",
        namespace="guest",
        entry="A",
        nodes={"A": NodeSpec(name="A", action="wf_tiny_a", parents=[])},
    )
    plan = {"A": 1280}
    original_plan = dict(plan)
    client = FakeClient()

    rows = run_one_workflow(
        workflow=workflow,
        client=client,
        max_workers=1,
        plan=plan,
        enable_dynamic=False,
        dynamic_config=object(),
        dynamic_ref_data=object(),
    )
    print(
        "V2 off path: "
        f"rows={len(rows)} plan_after={plan} "
        f"action={client.calls[0][0]} dynamic_column_present="
        f"{any('dynamic_upgrades' in row for row in rows)}"
    )
    assert plan == original_plan
    assert len(rows) == 2
    assert client.calls[0][0] == "wf_tiny_a_1280"
    assert not any("dynamic_upgrades" in row for row in rows)


def test_v3_apply_changes_updates_tier_and_action_suffix(monkeypatch) -> None:
    helper_calls: list[dict] = []

    def fake_upgrade_decision(**kwargs):
        helper_calls.append(kwargs)
        if len(helper_calls) == 1:
            return {"B": 2048}
        return None

    monkeypatch.setattr(
        run_workflow_module,
        "_runtime_upgrade_decision",
        fake_upgrade_decision,
    )
    workflow = WorkflowSpec(
        workflow_name="tiny_dynamic",
        namespace="guest",
        entry="A",
        nodes={
            "A": NodeSpec(name="A", action="wf_tiny_a", parents=[]),
            "B": NodeSpec(name="B", action="wf_tiny_b", parents=["A"]),
        },
    )
    client = FakeClient()
    rows = run_one_workflow(
        workflow=workflow,
        client=client,
        max_workers=1,
        plan={"A": 512, "B": 512},
        enable_dynamic=True,
        dynamic_config=object(),
        dynamic_ref_data=object(),
    )
    old_action = suffix_action_name(workflow.nodes["B"].action, "_512")
    new_action = suffix_action_name(workflow.nodes["B"].action, "_2048")
    invoked_actions = [action_name for action_name, _ in client.calls]
    stage_a_row = next(row for row in rows if row["stage_name"] == "A")
    print(
        "V3 apply semantics: "
        f"helper_calls={len(helper_calls)} old_action={old_action} "
        f"new_action={new_action} invoked_actions={invoked_actions} "
        f"dynamic_upgrades={stage_a_row.get('dynamic_upgrades')}"
    )
    assert invoked_actions == ["wf_tiny_a_512", "wf_tiny_b_2048"]
    assert old_action != new_action
    assert stage_a_row["dynamic_upgrades"] == '{"B":2048}'
