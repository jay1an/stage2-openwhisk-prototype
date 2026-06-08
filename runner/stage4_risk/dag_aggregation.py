#!/usr/bin/env python3
"""Closed-form DAG aggregation for lognormal stage latency models."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from runner.workflow import WorkflowSpec, load_workflow


STAGE_ORDER = [
    "detect_object",
    "estimate_pose",
    "match_face",
    "classify_scene",
    "translate_alert",
]
PERCENTILES = [0.50, 0.75, 0.90, 0.95, 0.99]
CIVIC_ALERT_CRITICAL_PATH_EDGES = 4
DEFAULT_PARAMS = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "path2_lognormal_fit"
    / "per_stage_lognormal_params.csv"
)
DEFAULT_WORKFLOW = Path(__file__).resolve().parents[2] / "configs" / "civic_alert_flow.yaml"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "path2_dag_aggregation"


@dataclass(frozen=True)
class LogNormalParams:
    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu):
            raise ValueError(f"mu must be finite, got {self.mu}")
        if not math.isfinite(self.sigma) or self.sigma < 0.0:
            raise ValueError(f"sigma must be finite and non-negative, got {self.sigma}")

    @property
    def mean(self) -> float:
        return float(math.exp(self.mu + (self.sigma**2) / 2.0))

    @property
    def variance(self) -> float:
        if self.sigma == 0.0:
            return 0.0
        return float(math.expm1(self.sigma**2) * math.exp(2.0 * self.mu + self.sigma**2))

    @property
    def cv(self) -> float:
        if self.sigma == 0.0:
            return 0.0
        return float(math.sqrt(math.expm1(self.sigma**2)))

    def quantile(self, p: float) -> float:
        if not 0.0 < p < 1.0:
            raise ValueError(f"quantile probability must be in (0, 1), got {p}")
        return float(math.exp(self.mu + norm.ppf(p) * self.sigma))

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        if self.sigma == 0.0:
            return 1.0 if x >= math.exp(self.mu) else 0.0
        return float(norm.cdf((math.log(x) - self.mu) / self.sigma))

    def survival(self, x: float) -> float:
        if x <= 0.0:
            return 1.0
        if self.sigma == 0.0:
            return 0.0 if x >= math.exp(self.mu) else 1.0
        return float(norm.sf((math.log(x) - self.mu) / self.sigma))


def fenton_wilkinson_sum(distributions: list[LogNormalParams]) -> LogNormalParams:
    """Approximate the sum of independent lognormals as a lognormal."""
    if not distributions:
        raise ValueError("fenton_wilkinson_sum requires at least one distribution")
    if len(distributions) == 1:
        return distributions[0]

    mean_sum = float(sum(dist.mean for dist in distributions))
    var_sum = float(sum(dist.variance for dist in distributions))
    if mean_sum <= 0.0 or not math.isfinite(mean_sum):
        raise ValueError(f"invalid summed mean: {mean_sum}")
    if var_sum < 0.0 or not math.isfinite(var_sum):
        raise ValueError(f"invalid summed variance: {var_sum}")
    if var_sum == 0.0:
        return LogNormalParams(mu=math.log(mean_sum), sigma=0.0)

    sigma_sq = math.log1p(var_sum / (mean_sum**2))
    mu = math.log(mean_sum) - sigma_sq / 2.0
    return LogNormalParams(mu=mu, sigma=math.sqrt(max(0.0, sigma_sq)))


def clark_max(a: LogNormalParams, b: LogNormalParams, rho: float = 0.0) -> LogNormalParams:
    """Approximate max of two lognormals as a lognormal via Clark in log-space."""
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], got {rho}")

    mu_x, sigma_x = a.mu, a.sigma
    mu_y, sigma_y = b.mu, b.sigma
    a_sq = sigma_x**2 + sigma_y**2 - 2.0 * rho * sigma_x * sigma_y
    a_gap = math.sqrt(max(0.0, a_sq))

    if a_gap < 1e-12:
        return a if mu_x >= mu_y else b

    alpha = (mu_x - mu_y) / a_gap
    phi = float(norm.pdf(alpha))
    cdf_alpha = float(norm.cdf(alpha))
    cdf_neg_alpha = float(norm.cdf(-alpha))

    mean_log_max = mu_x * cdf_alpha + mu_y * cdf_neg_alpha + a_gap * phi
    second_log_max = (
        (mu_x**2 + sigma_x**2) * cdf_alpha
        + (mu_y**2 + sigma_y**2) * cdf_neg_alpha
        + (mu_x + mu_y) * a_gap * phi
    )
    var_log_max = max(0.0, second_log_max - mean_log_max**2)
    return LogNormalParams(mu=float(mean_log_max), sigma=math.sqrt(var_log_max))


def add_deterministic_shift(dist: LogNormalParams, shift_ms: float) -> LogNormalParams:
    """Approximate L + constant as lognormal by preserving variance."""
    if shift_ms < 0.0 or not math.isfinite(shift_ms):
        raise ValueError(f"shift_ms must be finite and non-negative, got {shift_ms}")
    if shift_ms == 0.0:
        return dist

    new_mean = dist.mean + float(shift_ms)
    new_var = dist.variance
    if new_var == 0.0:
        return LogNormalParams(mu=math.log(new_mean), sigma=0.0)
    sigma_sq = math.log1p(new_var / (new_mean**2))
    mu = math.log(new_mean) - sigma_sq / 2.0
    return LogNormalParams(mu=mu, sigma=math.sqrt(max(0.0, sigma_sq)))


def aggregate_civic_alert(
    stage_dists: dict[str, LogNormalParams],
    transition_overhead_ms: float = 0.0,
) -> LogNormalParams:
    """Compute civic_alert workflow E2E distribution from per-stage dists.

    transition_overhead_ms is a deterministic delay per critical-path DAG edge.
    civic_alert has four critical-path transitions, so the final distribution
    is shifted by 4 * transition_overhead_ms with variance preserved.
    """
    missing = sorted(set(STAGE_ORDER).difference(stage_dists))
    if missing:
        raise ValueError(f"missing stage distributions: {missing}")
    if transition_overhead_ms < 0.0 or not math.isfinite(transition_overhead_ms):
        raise ValueError(
            f"transition_overhead_ms must be finite and non-negative, got {transition_overhead_ms}"
        )

    detect = stage_dists["detect_object"]
    estimate = stage_dists["estimate_pose"]
    match = stage_dists["match_face"]
    classify = stage_dists["classify_scene"]
    translate = stage_dists["translate_alert"]

    path1_partial = fenton_wilkinson_sum([detect, estimate, match])
    path2_partial = detect
    classify_scene_start = clark_max(path1_partial, path2_partial, rho=0.0)
    classify_scene_end = fenton_wilkinson_sum([classify_scene_start, classify])
    e2e = fenton_wilkinson_sum([classify_scene_end, translate])
    shift_ms = CIVIC_ALERT_CRITICAL_PATH_EDGES * float(transition_overhead_ms)
    return add_deterministic_shift(e2e, shift_ms)


def aggregate_dag(
    workflow: WorkflowSpec,
    stage_dists: dict[str, LogNormalParams],
    transition_overhead_ms: float = 0.0,
    fixed_finish: dict[str, float] | None = None,
) -> LogNormalParams:
    """Aggregate arbitrary DAG latency from per-stage lognormal distributions."""

    if transition_overhead_ms < 0.0 or not math.isfinite(transition_overhead_ms):
        raise ValueError(
            f"transition_overhead_ms must be finite and non-negative, got {transition_overhead_ms}"
        )

    node_names = set(workflow.nodes)
    dist_names = set(stage_dists)
    missing = sorted(node_names - dist_names)
    unknown = sorted(dist_names - node_names)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing stage distributions: {missing}")
        if unknown:
            parts.append(f"unknown stage distributions: {unknown}")
        raise ValueError("; ".join(parts))
    if not workflow.nodes:
        raise ValueError("workflow must contain at least one node")

    fixed_finish = fixed_finish or {}
    unknown_fixed = sorted(set(fixed_finish) - node_names)
    if unknown_fixed:
        raise ValueError(f"unknown fixed finish nodes: {unknown_fixed}")
    normalized_fixed_finish: dict[str, float] = {}
    for node_name, value in fixed_finish.items():
        try:
            finish_ms = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"fixed finish for {node_name!r} must be finite and positive, got {value}"
            ) from exc
        if finish_ms <= 0.0 or not math.isfinite(finish_ms):
            raise ValueError(
                f"fixed finish for {node_name!r} must be finite and positive, got {value}"
            )
        normalized_fixed_finish[node_name] = finish_ms
    fixed_finish = normalized_fixed_finish

    children: dict[str, list[str]] = {name: [] for name in workflow.nodes}
    indegree: dict[str, int] = {name: 0 for name in workflow.nodes}
    for node_name, node in workflow.nodes.items():
        for parent in node.parents:
            if parent not in workflow.nodes:
                raise ValueError(f"node {node_name!r} references unknown parent {parent!r}")
            children[parent].append(node_name)
            indegree[node_name] += 1

    fixed_nodes = set(fixed_finish)
    incomplete_fixed_parents = {
        node_name: [
            parent
            for parent in workflow.nodes[node_name].parents
            if parent not in fixed_nodes
        ]
        for node_name in fixed_nodes
    }
    incomplete_fixed_parents = {
        node_name: parents
        for node_name, parents in incomplete_fixed_parents.items()
        if parents
    }
    if incomplete_fixed_parents:
        raise ValueError(
            "fixed finish nodes require fixed parents; "
            f"incomplete parents: {incomplete_fixed_parents}"
        )

    ready = [name for name in workflow.nodes if indegree[name] == 0]
    topo_order: list[str] = []
    while ready:
        node_name = ready.pop(0)
        topo_order.append(node_name)
        for child in children[node_name]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(topo_order) != len(workflow.nodes):
        cycle_nodes = sorted(name for name, degree in indegree.items() if degree > 0)
        raise ValueError(f"workflow graph contains a cycle involving: {cycle_nodes}")

    finish: dict[str, LogNormalParams] = {}
    longest_edges: dict[str, int] = {}
    for node_name in topo_order:
        parents = workflow.nodes[node_name].parents
        if parents:
            longest_edges[node_name] = max(longest_edges[parent] + 1 for parent in parents)
        else:
            longest_edges[node_name] = 0

        if node_name in fixed_finish:
            finish[node_name] = LogNormalParams(
                mu=math.log(float(fixed_finish[node_name])),
                sigma=0.0,
            )
        elif parents:
            arrival = finish[parents[0]]
            for parent in parents[1:]:
                arrival = clark_max(arrival, finish[parent], rho=0.0)
            finish[node_name] = fenton_wilkinson_sum([arrival, stage_dists[node_name]])
        else:
            finish[node_name] = stage_dists[node_name]

    sinks = [name for name in topo_order if not children[name]]
    e2e = finish[sinks[0]]
    for sink in sinks[1:]:
        e2e = clark_max(e2e, finish[sink], rho=0.0)

    critical_path_edges = max(longest_edges[sink] for sink in sinks)
    return add_deterministic_shift(
        e2e,
        critical_path_edges * float(transition_overhead_ms),
    )


def conditional_risk(
    workflow: WorkflowSpec,
    stage_dists: dict[str, LogNormalParams],
    completed_finish_ms: dict[str, float],
    slo_ms: float,
    transition_overhead_ms: float = 0.0,
) -> float:
    e2e = aggregate_dag(
        workflow,
        stage_dists,
        transition_overhead_ms=transition_overhead_ms,
        fixed_finish=completed_finish_ms,
    )
    return e2e.survival(slo_ms)


def _load_stage_params(params_csv: str | Path, scenario: str) -> dict[str, LogNormalParams]:
    df = pd.read_csv(params_csv)
    required = {"stage_name", "latency_class", "mu", "sigma"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"missing required columns in lognormal params: {missing}")

    if scenario == "entry_warm":
        class_by_stage = {stage: "warm" for stage in STAGE_ORDER}
    elif scenario == "entry_cold":
        class_by_stage = {stage: "warm" for stage in STAGE_ORDER}
        class_by_stage["detect_object"] = "cold_like"
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    out: dict[str, LogNormalParams] = {}
    for stage_name, latency_class in class_by_stage.items():
        row = df[(df["stage_name"] == stage_name) & (df["latency_class"] == latency_class)]
        if row.empty:
            raise ValueError(f"missing params for stage={stage_name} class={latency_class}")
        if len(row) > 1:
            raise ValueError(f"duplicate params for stage={stage_name} class={latency_class}")
        out[stage_name] = LogNormalParams(mu=float(row["mu"].iloc[0]), sigma=float(row["sigma"].iloc[0]))
    return out


def _validate_civic_workflow(workflow: WorkflowSpec) -> None:
    node_set = set(workflow.nodes)
    expected = set(STAGE_ORDER)
    if node_set != expected:
        raise ValueError(f"this module currently validates civic_alert only; got nodes={sorted(node_set)}")
    parents = {name: sorted(workflow.nodes[name].parents) for name in STAGE_ORDER}
    expected_parents = {
        "detect_object": [],
        "estimate_pose": ["detect_object"],
        "match_face": ["estimate_pose"],
        "classify_scene": ["detect_object", "match_face"],
        "translate_alert": ["classify_scene"],
    }
    if parents != expected_parents:
        raise ValueError(f"unexpected civic_alert topology: {parents}")


def _draw_stage_samples(
    stage_dists: dict[str, LogNormalParams],
    mc_samples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    return {
        stage_name: rng.lognormal(mean=dist.mu, sigma=dist.sigma, size=mc_samples)
        for stage_name, dist in stage_dists.items()
    }


def _aggregate_civic_alert_samples(samples: dict[str, np.ndarray]) -> np.ndarray:
    detect = samples["detect_object"]
    path1_partial = detect + samples["estimate_pose"] + samples["match_face"]
    path2_partial = detect
    classify_start = np.maximum(path1_partial, path2_partial)
    classify_end = classify_start + samples["classify_scene"]
    return classify_end + samples["translate_alert"]


def _distribution_row(scenario: str, dist: LogNormalParams) -> dict[str, float | str]:
    return {
        "scenario": scenario,
        "mu": dist.mu,
        "sigma": dist.sigma,
        "mean": dist.mean,
        "p50": dist.quantile(0.50),
        "p90": dist.quantile(0.90),
        "p95": dist.quantile(0.95),
        "p99": dist.quantile(0.99),
    }


def _validate_scenario(
    scenario: str,
    analytical: LogNormalParams,
    mc_values: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for p in PERCENTILES:
        analytical_value = analytical.quantile(p)
        mc_value = float(np.quantile(mc_values, p))
        abs_error = abs(analytical_value - mc_value)
        rel_error_pct = abs_error / mc_value * 100.0 if mc_value > 0.0 else math.nan
        rows.append(
            {
                "scenario": scenario,
                "percentile": f"p{int(round(p * 100))}",
                "analytical_value": analytical_value,
                "mc_value": mc_value,
                "abs_error": abs_error,
                "rel_error_pct": rel_error_pct,
            }
        )
    return pd.DataFrame(rows)


def _path_sanity(stage_dists: dict[str, LogNormalParams]) -> dict[str, float]:
    path1 = fenton_wilkinson_sum(
        [
            stage_dists["detect_object"],
            stage_dists["estimate_pose"],
            stage_dists["match_face"],
        ]
    )
    path2 = stage_dists["detect_object"]
    clark = clark_max(path1, path2, rho=0.0)
    return {
        "path1_mean": path1.mean,
        "path2_mean": path2.mean,
        "clark_mean": clark.mean,
        "clark_vs_path1_rel_diff_pct": abs(clark.mean - path1.mean) / path1.mean * 100.0,
    }


def _acceptance_threshold(scenario: str, percentile: str) -> float:
    if scenario == "entry_warm":
        return 5.0 if percentile == "p99" else 3.0
    if scenario == "entry_cold":
        return 10.0 if percentile == "p99" else 5.0
    raise ValueError(f"unknown scenario: {scenario}")


def _with_acceptance(validation: pd.DataFrame) -> pd.DataFrame:
    out = validation.copy()
    out["threshold_pct"] = [
        _acceptance_threshold(str(row.scenario), str(row.percentile))
        for row in out.itertuples(index=False)
    ]
    out["passes"] = out["rel_error_pct"] <= out["threshold_pct"]
    return out


def _table_text(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def _write_cdf_plot(
    out_path: Path,
    analytical: dict[str, LogNormalParams],
    mc_by_scenario: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, scenario in zip(axes, ["entry_warm", "entry_cold"], strict=True):
        mc_values = np.sort(mc_by_scenario[scenario])
        y = np.arange(1, len(mc_values) + 1) / len(mc_values)
        upper = float(np.quantile(mc_values, 0.999))
        x = np.linspace(float(mc_values.min()), upper, 500)
        cdf = np.array([analytical[scenario].cdf(value) for value in x])

        ax.step(mc_values, y, where="post", linewidth=1.5, label="MC empirical CDF")
        ax.plot(x, cdf, linestyle="--", linewidth=1.8, label="Analytical lognormal CDF")
        ax.set_title(scenario)
        ax.set_xlabel("Workflow E2E latency (ms)")
        ax.set_ylabel("Cumulative probability")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(left=max(0.0, float(mc_values.min()) * 0.95), right=upper)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_report(
    out_path: Path,
    e2e: pd.DataFrame,
    validation: pd.DataFrame,
    sanity_rows: pd.DataFrame,
    transition_overhead_ms: float,
) -> None:
    accepted = _with_acceptance(validation)
    verdict = bool(accepted["passes"].all())
    all_within_5pct = bool((validation["rel_error_pct"] <= 5.0).all())
    lines = [
        "# Path 2 DAG Aggregation Validation",
        "",
        f"- Transition overhead per edge: `{transition_overhead_ms:.6f} ms`.",
        f"- Total deterministic E2E shift: `{CIVIC_ALERT_CRITICAL_PATH_EDGES * transition_overhead_ms:.6f} ms`.",
        "",
        "## Analytical E2E Distributions",
        "",
        _table_text(e2e),
        "",
        "## MC Validation",
        "",
        _table_text(accepted),
        "",
        f"- All percentile errors within scenario-specific acceptance thresholds: `{verdict}`.",
        f"- All percentile errors within 5% relative error: `{all_within_5pct}`.",
        "- Entry-warm acceptance: p50/p75/p90/p95 < 3%, p99 < 5%.",
        "- Entry-cold acceptance: p50/p75/p90/p95 < 5%, p99 < 10%.",
        "",
        "## Clark Dominance Sanity",
        "",
        _table_text(sanity_rows),
        "",
        "For civic_alert, the long path into `classify_scene` dominates the direct `detect_object` branch. "
        "The Clark max should therefore have a mean very close to `path1_mean`, not the average of the two path means.",
        "",
        "## Assumptions",
        "",
        "- Stage distributions are independent.",
        "- The Clark fan-in step treats the long path and direct branch as independent even though they share `detect_object`; this first version intentionally overestimates variance for safety.",
        "- Validation MC draws from the fitted per-stage lognormal distributions, aggregates the raw samples through the real civic_alert max/sum DAG, and applies the same deterministic transition shift.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_validation(
    *,
    lognormal_params: str | Path,
    workflow_config: str | Path,
    out_dir: str | Path,
    mc_samples: int,
    seed: int,
    transition_overhead_ms: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive")
    if transition_overhead_ms < 0.0 or not math.isfinite(transition_overhead_ms):
        raise ValueError("transition_overhead_ms must be finite and non-negative")

    workflow = load_workflow(str(workflow_config))
    _validate_civic_workflow(workflow)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    analytical: dict[str, LogNormalParams] = {}
    mc_by_scenario: dict[str, np.ndarray] = {}
    e2e_rows: list[dict[str, float | str]] = []
    validation_frames: list[pd.DataFrame] = []
    sanity_rows: list[dict[str, float | str]] = []

    for scenario in ["entry_warm", "entry_cold"]:
        stage_dists = _load_stage_params(lognormal_params, scenario)
        e2e_dist = aggregate_civic_alert(
            stage_dists,
            transition_overhead_ms=transition_overhead_ms,
        )
        analytical[scenario] = e2e_dist
        e2e_rows.append(_distribution_row(scenario, e2e_dist))

        stage_samples = _draw_stage_samples(stage_dists, mc_samples, rng)
        mc_values = _aggregate_civic_alert_samples(stage_samples)
        mc_values = mc_values + CIVIC_ALERT_CRITICAL_PATH_EDGES * float(transition_overhead_ms)
        mc_by_scenario[scenario] = mc_values
        validation_frames.append(_validate_scenario(scenario, e2e_dist, mc_values))

        sanity = _path_sanity(stage_dists)
        sanity_rows.append({"scenario": scenario, **sanity})

    e2e = pd.DataFrame(e2e_rows)
    validation = pd.concat(validation_frames, ignore_index=True)
    sanity_df = pd.DataFrame(sanity_rows)

    e2e.to_csv(out / "e2e_distributions.csv", index=False)
    validation.to_csv(out / "mc_validation.csv", index=False)
    sanity_df.to_csv(out / "clark_sanity.csv", index=False)
    _write_cdf_plot(out / "cdf_comparison.png", analytical, mc_by_scenario)
    _write_report(out / "aggregation_report.md", e2e, validation, sanity_df, transition_overhead_ms)
    return e2e, validation, sanity_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lognormal-params", default=str(DEFAULT_PARAMS))
    parser.add_argument("--workflow-config", default=str(DEFAULT_WORKFLOW))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--mc-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--transition-overhead-ms", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    e2e, validation, sanity = run_validation(
        lognormal_params=args.lognormal_params,
        workflow_config=args.workflow_config,
        out_dir=args.out_dir,
        mc_samples=args.mc_samples,
        seed=args.seed,
        transition_overhead_ms=args.transition_overhead_ms,
    )
    accepted = _with_acceptance(validation)
    print("Analytical E2E distributions:")
    print(e2e.to_string(index=False))
    print("\nMC validation:")
    print(accepted.to_string(index=False))
    print("\nClark dominance sanity:")
    print(sanity.to_string(index=False))
    print(f"\nAll acceptance checks passed: {bool(accepted['passes'].all())}")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
