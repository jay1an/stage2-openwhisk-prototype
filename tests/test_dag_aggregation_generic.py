#!/usr/bin/env python3
"""Verify generic DAG aggregation against civic oracle and MC checks."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runner.stage4_risk.dag_aggregation import (  # noqa: E402
    PERCENTILES,
    LogNormalParams,
    aggregate_civic_alert,
    aggregate_dag,
)
from runner.workflow import NodeSpec, WorkflowSpec, load_workflow  # noqa: E402


def random_dist(rng: np.random.Generator) -> LogNormalParams:
    mean = float(rng.uniform(400.0, 3500.0))
    cv = float(rng.uniform(0.03, 0.20))
    sigma_sq = math.log1p(cv**2)
    return LogNormalParams(mu=math.log(mean) - sigma_sq / 2.0, sigma=math.sqrt(sigma_sq))


def assert_close(name: str, actual: float, expected: float, tol: float = 1e-12) -> None:
    diff = abs(actual - expected)
    print(
        f"{name}: actual={actual:.16g} expected={expected:.16g} "
        f"abs_diff={diff:.3e} tol={tol:.1e}"
    )
    if diff > tol:
        raise AssertionError(f"{name} differs by {diff}, tolerance {tol}")


def g1_civic_regression() -> None:
    print("G1 civic_alert regression")
    workflow = load_workflow(str(ROOT / "configs" / "civic_alert_flow.yaml"))
    rng = np.random.default_rng(20260608)
    for trial in range(12):
        dists = {stage: random_dist(rng) for stage in workflow.nodes}
        for overhead in [0.0, 250.0]:
            generic = aggregate_dag(workflow, dists, transition_overhead_ms=overhead)
            oracle = aggregate_civic_alert(dists, transition_overhead_ms=overhead)
            prefix = f"trial={trial} overhead={overhead}"
            assert_close(f"{prefix} mu", generic.mu, oracle.mu)
            assert_close(f"{prefix} sigma", generic.sigma, oracle.sigma)
    print("G1 PASS\n")


def chain_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        workflow_name="chain_abc",
        namespace="guest",
        entry="A",
        nodes={
            "A": NodeSpec(name="A", action="a", parents=[]),
            "B": NodeSpec(name="B", action="b", parents=["A"]),
            "C": NodeSpec(name="C", action="c", parents=["B"]),
        },
    )


def diamond_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        workflow_name="diamond_abcd",
        namespace="guest",
        entry="A",
        nodes={
            "A": NodeSpec(name="A", action="a", parents=[]),
            "B": NodeSpec(name="B", action="b", parents=["A"]),
            "C": NodeSpec(name="C", action="c", parents=["A"]),
            "D": NodeSpec(name="D", action="d", parents=["B", "C"]),
        },
    )


def draw_samples(
    dists: dict[str, LogNormalParams],
    n: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    return {
        name: rng.lognormal(mean=dist.mu, sigma=dist.sigma, size=n)
        for name, dist in dists.items()
    }


def mc_chain(samples: dict[str, np.ndarray]) -> np.ndarray:
    return samples["A"] + samples["B"] + samples["C"]


def mc_diamond(samples: dict[str, np.ndarray]) -> np.ndarray:
    return samples["A"] + np.maximum(samples["B"], samples["C"]) + samples["D"]


def g2_case(
    name: str,
    workflow: WorkflowSpec,
    dists: dict[str, LogNormalParams],
    mc_func,
    *,
    mc_samples: int,
    seed: int,
) -> None:
    print(f"G2 {name} MC validation")
    rng = np.random.default_rng(seed)
    analytical = aggregate_dag(workflow, dists)
    samples = draw_samples(dists, mc_samples, rng)
    mc_values = mc_func(samples)
    max_rel_error = 0.0
    for p in PERCENTILES:
        analytical_q = analytical.quantile(p)
        mc_q = float(np.quantile(mc_values, p))
        rel_error_pct = abs(analytical_q - mc_q) / mc_q * 100.0
        max_rel_error = max(max_rel_error, rel_error_pct)
        print(
            f"{name} p{int(round(p * 100)):02d}: analytical={analytical_q:.3f} "
            f"mc={mc_q:.3f} rel_error_pct={rel_error_pct:.3f}"
        )
        if rel_error_pct >= 3.0:
            raise AssertionError(
                f"{name} p={p} rel_error_pct={rel_error_pct:.3f} >= 3.0"
            )
    print(f"{name} max_rel_error_pct={max_rel_error:.3f}")
    print(f"G2 {name} PASS\n")


def g2_generality() -> None:
    chain_dists = {
        "A": LogNormalParams(mu=math.log(900.0), sigma=0.04),
        "B": LogNormalParams(mu=math.log(700.0), sigma=0.05),
        "C": LogNormalParams(mu=math.log(500.0), sigma=0.04),
    }
    diamond_dists = {
        "A": LogNormalParams(mu=math.log(800.0), sigma=0.04),
        "B": LogNormalParams(mu=math.log(760.0), sigma=0.05),
        "C": LogNormalParams(mu=math.log(720.0), sigma=0.05),
        "D": LogNormalParams(mu=math.log(600.0), sigma=0.04),
    }
    g2_case(
        "chain",
        chain_workflow(),
        chain_dists,
        mc_chain,
        mc_samples=100_000,
        seed=101,
    )
    g2_case(
        "diamond",
        diamond_workflow(),
        diamond_dists,
        mc_diamond,
        mc_samples=100_000,
        seed=202,
    )


def expect_value_error(label: str, func) -> None:
    try:
        func()
    except ValueError as exc:
        print(f"{label}: PASS ValueError: {exc}")
        return
    raise AssertionError(f"{label}: expected ValueError")


def g3_validation() -> None:
    print("G3 input validation")
    workflow = chain_workflow()
    valid = {
        "A": LogNormalParams(mu=math.log(1.0), sigma=0.1),
        "B": LogNormalParams(mu=math.log(1.0), sigma=0.1),
        "C": LogNormalParams(mu=math.log(1.0), sigma=0.1),
    }
    expect_value_error("missing dist", lambda: aggregate_dag(workflow, {"A": valid["A"], "B": valid["B"]}))
    expect_value_error(
        "unknown dist",
        lambda: aggregate_dag(workflow, {**valid, "Z": LogNormalParams(mu=0.0, sigma=0.1)}),
    )
    cyclic = WorkflowSpec(
        workflow_name="cycle",
        namespace="guest",
        entry="A",
        nodes={
            "A": NodeSpec(name="A", action="a", parents=["C"]),
            "B": NodeSpec(name="B", action="b", parents=["A"]),
            "C": NodeSpec(name="C", action="c", parents=["B"]),
        },
    )
    expect_value_error("cycle", lambda: aggregate_dag(cyclic, valid))
    bad_parent = replace(workflow, nodes={**workflow.nodes, "D": NodeSpec(name="D", action="d", parents=["NOPE"])})
    bad_parent_dists = {**valid, "D": LogNormalParams(mu=0.0, sigma=0.1)}
    expect_value_error("unknown parent", lambda: aggregate_dag(bad_parent, bad_parent_dists))
    expect_value_error("negative overhead", lambda: aggregate_dag(workflow, valid, -1.0))
    expect_value_error("nan overhead", lambda: aggregate_dag(workflow, valid, math.nan))
    print("G3 PASS\n")


def main() -> None:
    g1_civic_regression()
    g2_generality()
    g3_validation()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
