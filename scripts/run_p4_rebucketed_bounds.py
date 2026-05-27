#!/usr/bin/env python3
"""Re-bucket Stage 3 cold samples and re-check Monte Carlo bounds."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.workflow import WorkflowSpec, load_workflow  # noqa: E402


RAW_TRACE = (
    ROOT
    / "reports"
    / "civic_azure_cand2_45min_1280mb_1cpu_keepalive20s_target20s_balanced_mi96"
    / "raw_trace.csv"
)
P3_BOUNDS = ROOT / "reports" / "stage4_p3_first_pass" / "step3_mc_bounds.csv"
WORKFLOW_CONFIG = ROOT / "configs" / "civic_alert_flow.yaml"
OUT_DIR = ROOT / "reports" / "stage3_cold_bucketed"
SLO_VALUES = [15_000, 20_000, 25_000, 30_000]
BUCKETS = ["warm", "partial_cold_cascade", "cold_with_contention", "clean_cold"]
SAMPLE_COLUMNS = [
    "trace_label",
    "workflow_name",
    "stage_name",
    "latency_class",
    "dispatch_latency_ms",
    "platform_overhead_ms",
    "action_duration_ms",
    "stage_start_offset_ms",
    "stage_completion_offset_ms",
    "cold_like_normalized",
    "latency_class_v2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-trace", default=str(RAW_TRACE))
    parser.add_argument("--workflow-config", default=str(WORKFLOW_CONFIG))
    parser.add_argument("--p3-bounds", default=str(P3_BOUNDS))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--simulations-per-request", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260527)
    return parser.parse_args()


def truthy(value: object) -> bool:
    return str(value).strip().lower() == "true"


def topological_nodes(workflow: WorkflowSpec) -> list[str]:
    remaining = set(workflow.nodes)
    ordered: list[str] = []
    while remaining:
        ready = sorted(
            name
            for name in remaining
            if all(parent in ordered for parent in workflow.nodes[name].parents)
        )
        if not ready:
            raise ValueError("workflow DAG contains a cycle or missing parent")
        ordered.extend(ready)
        remaining.difference_update(ready)
    return ordered


def build_bucketed_samples(raw_trace: pd.DataFrame, workflow: WorkflowSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage = raw_trace[
        (raw_trace["stage_name"] != "__entry__")
        & (raw_trace["stage_name"].isin(workflow.nodes))
    ].copy()
    stage["cold_like_normalized"] = stage["cold_like"].map(truthy)
    cold_counts = (
        stage.groupby("request_id")["cold_like_normalized"]
        .sum()
        .astype(int)
        .rename("workflow_cold_count")
    )
    stage = stage.merge(cold_counts, on="request_id", how="left")
    stage["ow_wait_ms_num"] = pd.to_numeric(stage["ow_wait_ms"], errors="coerce").fillna(0.0)

    stage["latency_class"] = np.where(stage["cold_like_normalized"], "cold_like", "warm")
    conditions = [
        ~stage["cold_like_normalized"],
        stage["cold_like_normalized"] & (stage["workflow_cold_count"] >= 2),
        stage["cold_like_normalized"] & (stage["workflow_cold_count"] == 1) & (stage["ow_wait_ms_num"] >= 200.0),
        stage["cold_like_normalized"] & (stage["workflow_cold_count"] == 1) & (stage["ow_wait_ms_num"] < 200.0),
    ]
    stage["latency_class_v2"] = np.select(
        conditions,
        ["warm", "partial_cold_cascade", "cold_with_contention", "clean_cold"],
        default="unlabeled",
    )

    for column in [
        "dispatch_latency_ms",
        "platform_overhead_ms",
        "action_duration_ms",
        "dispatch_start_ms",
        "dispatch_end_ms",
        "entry_ts_ms",
    ]:
        stage[column] = pd.to_numeric(stage[column], errors="coerce")

    samples = pd.DataFrame(
        {
            "trace_label": "raw_trace",
            "workflow_name": stage["workflow_name"],
            "stage_name": stage["stage_name"],
            "latency_class": stage["latency_class"],
            "dispatch_latency_ms": stage["dispatch_latency_ms"],
            "platform_overhead_ms": stage["platform_overhead_ms"],
            "action_duration_ms": stage["action_duration_ms"],
            "stage_start_offset_ms": stage["dispatch_start_ms"] - stage["entry_ts_ms"],
            "stage_completion_offset_ms": stage["dispatch_end_ms"] - stage["entry_ts_ms"],
            "cold_like_normalized": stage["cold_like_normalized"],
            "latency_class_v2": stage["latency_class_v2"],
        }
    )[SAMPLE_COLUMNS]

    counts = (
        samples.groupby(["stage_name", "latency_class_v2"])
        .size()
        .rename("count")
        .reset_index()
    )
    full_index = pd.MultiIndex.from_product(
        [sorted(workflow.nodes), BUCKETS],
        names=["stage_name", "latency_class_v2"],
    )
    counts = (
        counts.set_index(["stage_name", "latency_class_v2"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    requests = raw_trace[raw_trace["stage_name"] == "__entry__"][
        ["request_id", "entry_ts_ms", "workflow_e2e_ms"]
    ].copy()
    requests = requests.merge(cold_counts, on="request_id", how="inner")
    requests["entry_ts_ms"] = pd.to_numeric(requests["entry_ts_ms"], errors="coerce")
    requests["workflow_e2e_ms"] = pd.to_numeric(requests["workflow_e2e_ms"], errors="coerce")
    requests = requests.dropna(subset=["entry_ts_ms", "workflow_e2e_ms"])
    requests = requests.sort_values(["entry_ts_ms", "request_id"]).reset_index(drop=True)
    return samples, counts, requests


class BucketSampler:
    def __init__(self, samples: pd.DataFrame, rng: np.random.Generator) -> None:
        self.rng = rng
        self.pools: dict[tuple[str, str], np.ndarray] = {}
        numeric = samples.copy()
        numeric["dispatch_latency_ms"] = pd.to_numeric(numeric["dispatch_latency_ms"], errors="coerce")
        numeric = numeric.dropna(subset=["dispatch_latency_ms"])
        for (stage_name, bucket), group in numeric.groupby(["stage_name", "latency_class_v2"]):
            self.pools[(str(stage_name), str(bucket))] = group["dispatch_latency_ms"].to_numpy(dtype=float)

    def has_pool(self, stage_name: str, bucket_names: tuple[str, ...]) -> bool:
        return any(len(self.pools.get((stage_name, bucket), [])) > 0 for bucket in bucket_names)

    def sample(self, stage_name: str, bucket_names: tuple[str, ...]) -> float:
        arrays = [
            self.pools[(stage_name, bucket)]
            for bucket in bucket_names
            if len(self.pools.get((stage_name, bucket), [])) > 0
        ]
        if not arrays:
            raise ValueError(f"missing latency pool for stage={stage_name} buckets={bucket_names}")
        pool = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
        return float(pool[int(self.rng.integers(0, len(pool)))])


def simulate_scenario(
    *,
    requests: pd.DataFrame,
    workflow: WorkflowSpec,
    ordered: list[str],
    samples: pd.DataFrame,
    scenario: str,
    bucket_names: tuple[str, ...],
    simulations_per_request: int,
    seed: int,
) -> pd.DataFrame:
    sampler = BucketSampler(samples, np.random.default_rng(seed))
    missing = [stage for stage in ordered if not sampler.has_pool(stage, bucket_names)]
    if missing:
        raise ValueError(f"{scenario} has no sample pool for stages: {', '.join(missing)}")

    n = len(requests) * simulations_per_request
    latencies = np.empty(n, dtype=float)
    cold_counts = np.empty(n, dtype=int)
    row = 0
    stage_count = len(ordered)
    cold_count = 0 if bucket_names == ("warm",) else stage_count
    for _sim_id in range(simulations_per_request):
        for _request in requests.itertuples(index=False):
            completions: dict[str, float] = {}
            for stage_name in ordered:
                ready = max(
                    (completions[parent] for parent in workflow.nodes[stage_name].parents),
                    default=0.0,
                )
                completions[stage_name] = ready + sampler.sample(stage_name, bucket_names)
            latencies[row] = max(completions.values())
            cold_counts[row] = cold_count
            row += 1
    return pd.DataFrame(
        {
            "predicted_latency_ms": latencies,
            "cold_like_stage_count": cold_counts,
        }
    )


def empty_summary_rows(scenario: str, slo_values: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "slo_ms": slo_ms,
                "predicted_violation_rate": math.nan,
                "p50_ms": math.nan,
                "p90_ms": math.nan,
                "p95_ms": math.nan,
                "mc_partial_cold_rate": math.nan,
                "mc_all_cold_rate": math.nan,
            }
            for slo_ms in slo_values
        ]
    )


def summarize_mc(instances: pd.DataFrame, scenario: str, slo_values: list[int], stage_count: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lat = instances["predicted_latency_ms"]
    cold = instances["cold_like_stage_count"]
    for slo_ms in slo_values:
        rows.append(
            {
                "scenario": scenario,
                "slo_ms": slo_ms,
                "predicted_violation_rate": float((lat > slo_ms).mean()),
                "p50_ms": float(lat.quantile(0.50)),
                "p90_ms": float(lat.quantile(0.90)),
                "p95_ms": float(lat.quantile(0.95)),
                "mc_partial_cold_rate": float(((cold > 0) & (cold < stage_count)).mean()),
                "mc_all_cold_rate": float((cold == stage_count).mean()),
            }
        )
    return pd.DataFrame(rows)


def real_trace_facts(requests: pd.DataFrame, stage_count: int) -> dict[str, float]:
    all_warm = requests[requests["workflow_cold_count"] == 0]["workflow_e2e_ms"]
    all_cold = requests[requests["workflow_cold_count"] == stage_count]["workflow_e2e_ms"]
    return {
        "real_all_warm_p95_ms": float(all_warm.quantile(0.95)),
        "real_all_cold_p95_ms": float(all_cold.quantile(0.95)),
        "real_all_cold_count": float(len(all_cold)),
        "real_overall_20s_violation_rate": float((requests["workflow_e2e_ms"] > 20_000).mean()),
    }


def fmt_ms(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.1f}"


def table_text(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def write_report(
    *,
    out_dir: Path,
    counts: pd.DataFrame,
    bounds: pd.DataFrame,
    old_bounds: pd.DataFrame | None,
    facts: dict[str, float],
    sanity: dict[str, Any],
    scenario_errors: dict[str, str],
) -> None:
    comparison_rows: list[dict[str, str]] = []
    old_scale_p95 = math.nan
    old_warm_p95 = math.nan
    if old_bounds is not None and not old_bounds.empty:
        old_scale = old_bounds[old_bounds["baseline"] == "scale_to_zero"]
        old_warm = old_bounds[old_bounds["baseline"] == "always_warm"]
        if not old_scale.empty:
            old_scale_p95 = float(old_scale["p95_ms"].iloc[0])
        if not old_warm.empty:
            old_warm_p95 = float(old_warm["p95_ms"].iloc[0])

    def v2_p95(scenario: str) -> float:
        rows = bounds[bounds["scenario"] == scenario]
        return float(rows["p95_ms"].iloc[0]) if not rows.empty else math.nan

    comparison_rows.append(
        {
            "scenario": "scale_to_zero_refined",
            "old_mc_p95_ms": fmt_ms(old_scale_p95),
            "v2_mc_p95_ms": fmt_ms(v2_p95("scale_to_zero_refined")),
            "real_p95_ms": fmt_ms(facts["real_all_cold_p95_ms"]),
        }
    )
    comparison_rows.append(
        {
            "scenario": "scale_to_zero_realistic",
            "old_mc_p95_ms": "n/a",
            "v2_mc_p95_ms": fmt_ms(v2_p95("scale_to_zero_realistic")),
            "real_p95_ms": "n/a",
        }
    )
    comparison_rows.append(
        {
            "scenario": "always_warm",
            "old_mc_p95_ms": fmt_ms(old_warm_p95),
            "v2_mc_p95_ms": fmt_ms(v2_p95("always_warm_v2")),
            "real_p95_ms": fmt_ms(facts["real_all_warm_p95_ms"]),
        }
    )
    comparison = pd.DataFrame(comparison_rows)

    refined_p95 = v2_p95("scale_to_zero_refined")
    real_cold_p95 = facts["real_all_cold_p95_ms"]
    old_error = abs(old_scale_p95 - real_cold_p95) / real_cold_p95 if not pd.isna(old_scale_p95) else math.nan
    new_error = abs(refined_p95 - real_cold_p95) / real_cold_p95 if not pd.isna(refined_p95) else math.nan
    in_range = bool(25_000 <= refined_p95 <= 35_000) if not pd.isna(refined_p95) else False
    brackets = bool(refined_p95 <= real_cold_p95 <= old_scale_p95) if not pd.isna(refined_p95) and not pd.isna(old_scale_p95) else False
    violation_20s = bounds[bounds["slo_ms"] == 20_000][["scenario", "predicted_violation_rate"]]

    lines = [
        "# P4 Re-Bucketed Cold Samples and MC Bounds",
        "",
        "## Bucket Sample Counts",
        "",
        table_text(counts.pivot(index="stage_name", columns="latency_class_v2", values="count").reset_index()),
        "",
        "## Sanity Checks",
        "",
        f"- Total cold samples: `{sanity['cold_total']}` (expected `539`).",
        f"- Total warm samples: `{sanity['warm_total']}` (expected `19516`).",
        f"- Per-stage clean_cold counts below 10: `{sanity['low_clean_cold_stages']}`.",
        f"- Scenario errors: `{scenario_errors}`.",
        "",
        "## MC Bounds V2",
        "",
        table_text(bounds),
        "",
        "## MC Bounds Comparison",
        "",
        table_text(comparison),
        "",
        "## Verdict On Issue (1)",
        "",
        f"- Does `scale_to_zero_refined` p95 fall in [25000, 35000] ms? `{in_range}`.",
        f"- Does it bracket the real all_cold p95 `{real_cold_p95:.1f} ms`? `{brackets}`.",
        f"- Old relative p95 error vs real all_cold: `{old_error:.3f}`.",
        f"- New relative p95 error vs real all_cold: `{'n/a' if pd.isna(new_error) else f'{new_error:.3f}'}`.",
        "- Verdict: issue (1) is not fixed under the literal P4 bucket rule because `clean_cold` is empty.",
        "",
        "## Observation On Issue (2)",
        "",
        table_text(violation_20s),
        "",
        f"- Real 20s violation rate: `{facts['real_overall_20s_violation_rate']:.6f}`.",
        "- The 20s calibration gap remains a separate correlation problem and is deferred to path 2.",
        "",
        "## Notes",
        "",
        "- The literal rule labels any workflow with two or more cold stages as `partial_cold_cascade`, including the 14 observed all-cold workflows.",
        "- All singleton cold samples in this trace have `ow_wait_ms >= 200`, so none qualify as `clean_cold`.",
    ]
    (out_dir / "v2_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    workflow = load_workflow(str(args.workflow_config))
    ordered = topological_nodes(workflow)
    raw_trace = pd.read_csv(args.raw_trace)

    print("Step 1/2: building refined cold buckets")
    samples, counts, requests = build_bucketed_samples(raw_trace, workflow)
    samples.to_csv(out_dir / "latency_samples_for_monte_carlo_v2.csv", index=False)
    counts.to_csv(out_dir / "sample_counts.csv", index=False)
    print(counts.pivot(index="stage_name", columns="latency_class_v2", values="count").to_string())

    cold_total = int(samples["cold_like_normalized"].sum())
    warm_total = int((~samples["cold_like_normalized"]).sum())
    clean_counts = counts[counts["latency_class_v2"] == "clean_cold"]
    low_clean = clean_counts.loc[clean_counts["count"] < 10, "stage_name"].tolist()
    sanity = {
        "cold_total": cold_total,
        "warm_total": warm_total,
        "low_clean_cold_stages": low_clean,
    }
    print(f"cold_total={cold_total}; warm_total={warm_total}; low_clean_cold_stages={low_clean}")

    print("\nStep 3: running MC bounds with refined buckets")
    scenarios = [
        ("scale_to_zero_refined", ("clean_cold",)),
        ("scale_to_zero_realistic", ("clean_cold", "partial_cold_cascade")),
        ("always_warm_v2", ("warm",)),
    ]
    summary_frames: list[pd.DataFrame] = []
    scenario_errors: dict[str, str] = {}
    for index, (scenario, bucket_names) in enumerate(scenarios, start=1):
        try:
            instances = simulate_scenario(
                requests=requests,
                workflow=workflow,
                ordered=ordered,
                samples=samples,
                scenario=scenario,
                bucket_names=bucket_names,
                simulations_per_request=args.simulations_per_request,
                seed=args.seed + index,
            )
            summary = summarize_mc(instances, scenario, SLO_VALUES, len(ordered))
        except ValueError as exc:
            scenario_errors[scenario] = str(exc)
            summary = empty_summary_rows(scenario, SLO_VALUES)
        summary_frames.append(summary)
        print(summary.to_string(index=False))
    bounds = pd.concat(summary_frames, ignore_index=True)
    bounds.to_csv(out_dir / "step3_mc_bounds_v2.csv", index=False)

    print("\nStep 4: writing comparison report")
    old_bounds = pd.read_csv(args.p3_bounds) if Path(args.p3_bounds).exists() else None
    facts = real_trace_facts(requests, len(ordered))
    write_report(
        out_dir=out_dir,
        counts=counts,
        bounds=bounds,
        old_bounds=old_bounds,
        facts=facts,
        sanity=sanity,
        scenario_errors=scenario_errors,
    )
    print(f"wrote {out_dir / 'v2_comparison_report.md'}")


if __name__ == "__main__":
    main()
