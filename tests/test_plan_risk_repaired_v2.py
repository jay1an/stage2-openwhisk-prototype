from __future__ import annotations

import pytest

from runner.stage4_risk.plan_risk import PlanInput, compute_plan_risk
from runner.stage5_control.multi_slo_planner import load_reference_data


PREMIUM_18 = {
    "detect_object": 2048,
    "estimate_pose": 1280,
    "match_face": 1792,
    "classify_scene": 1280,
    "translate_alert": 1792,
}
FREE_22 = {
    "detect_object": 1024,
    "estimate_pose": 1280,
    "match_face": 1024,
    "classify_scene": 1024,
    "translate_alert": 1024,
}


def make_plan(memory: dict[str, int]) -> PlanInput:
    ref = load_reference_data()
    return PlanInput(
        memory_tier_per_stage=dict(memory),
        entry_prewarm_count=0.0,
        predicted_arrivals=5.0,
        lognormal_params=ref.lognormal_params,
        amdahl_params=ref.amdahl_params,
        cold_overhead_per_stage=ref.cold_overhead_per_stage,
        p_baseline=ref.p_baseline,
    )


def test_legacy_explicit_path_matches_default() -> None:
    plan = make_plan(PREMIUM_18)
    default = compute_plan_risk(plan, 18000.0, rho=0.67, contention_factor=1.10)
    explicit = compute_plan_risk(
        plan,
        18000.0,
        rho=0.67,
        contention_factor=1.10,
        risk_model="legacy",
    )
    print(
        "legacy equality: "
        f"default_warm_p95={default.e2e_warm_params.quantile(.95):.6f} "
        f"explicit_warm_p95={explicit.e2e_warm_params.quantile(.95):.6f} "
        f"default_violation={default.p_violation_total:.12f} "
        f"explicit_violation={explicit.p_violation_total:.12f}"
    )
    assert explicit.p_entry_cold == pytest.approx(default.p_entry_cold)
    assert explicit.e2e_warm_params.mu == pytest.approx(default.e2e_warm_params.mu)
    assert explicit.e2e_warm_params.sigma == pytest.approx(default.e2e_warm_params.sigma)
    assert explicit.e2e_cold_entry_params.mu == pytest.approx(default.e2e_cold_entry_params.mu)
    assert explicit.e2e_cold_entry_params.sigma == pytest.approx(default.e2e_cold_entry_params.sigma)
    assert explicit.p_violation_total == pytest.approx(default.p_violation_total)


def test_repaired_v2_reproduces_validation_level_p95s() -> None:
    # repaired_v2 uses deterministic sync shifts plus a lognormal cold-overhead
    # distribution fit by moment matching from sweep cold-minus-warm-median
    # samples. It is therefore close to, but slightly more conservative than,
    # the empirical-sampling validation rows.
    cases = [
        ("premium", 18000.0, PREMIUM_18, 16951.384, 18520.415),
        ("free", 22000.0, FREE_22, 21770.338, 23454.807),
    ]
    for slo_class, slo_ms, memory, expected_warm_p95, expected_cold_p95 in cases:
        plan = make_plan(memory)
        result = compute_plan_risk(
            plan,
            slo_ms,
            rho=0.67,
            contention_factor=1.0,
            risk_model="repaired_v2",
            slo_class=slo_class,
            p_entry_cold=0.1,
        )
        warm_p95 = result.e2e_warm_params.quantile(0.95)
        cold_p95 = result.e2e_cold_entry_params.quantile(0.95)
        print(
            "repaired_v2 p95: "
            f"class={slo_class} warm_p95={warm_p95:.3f} cold_p95={cold_p95:.3f} "
            f"warm_survival={result.p_violation_warm:.6f} "
            f"cold_survival={result.p_violation_cold_entry:.6f} "
            f"total={result.p_violation_total:.6f}"
        )
        assert warm_p95 == pytest.approx(expected_warm_p95, abs=1.0)
        assert cold_p95 == pytest.approx(expected_cold_p95, abs=1.0)


def test_repaired_v2_p_entry_cold_boundaries() -> None:
    plan = make_plan(PREMIUM_18)
    warm_only = compute_plan_risk(
        plan,
        18000.0,
        rho=0.67,
        contention_factor=1.0,
        risk_model="repaired_v2",
        slo_class="premium",
        p_entry_cold=0.0,
    )
    cold_only = compute_plan_risk(
        plan,
        18000.0,
        rho=0.67,
        contention_factor=1.0,
        risk_model="repaired_v2",
        slo_class="premium",
        p_entry_cold=1.0,
    )
    print(
        "p_entry boundaries: "
        f"warm_total={warm_only.p_violation_total:.12f} "
        f"warm_survival={warm_only.p_violation_warm:.12f} "
        f"cold_total={cold_only.p_violation_total:.12f} "
        f"cold_survival={cold_only.p_violation_cold_entry:.12f}"
    )
    assert warm_only.p_violation_total == pytest.approx(warm_only.p_violation_warm)
    assert cold_only.p_violation_total == pytest.approx(cold_only.p_violation_cold_entry)


def test_repaired_v2_rejects_invalid_p_entry_cold() -> None:
    plan = make_plan(PREMIUM_18)
    for value in [None, -0.01, 1.01]:
        with pytest.raises(ValueError) as excinfo:
            compute_plan_risk(
                plan,
                18000.0,
                rho=0.67,
                contention_factor=1.0,
                risk_model="repaired_v2",
                slo_class="premium",
                p_entry_cold=value,
            )
        print(f"invalid p_entry={value}: {excinfo.value}")
