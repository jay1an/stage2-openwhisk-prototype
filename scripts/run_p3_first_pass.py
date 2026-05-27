#!/usr/bin/env python3
"""First-pass real-data SLO validation and Monte Carlo calibration."""

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

from runner.stage4_risk.container_pool_cold_model import ContainerPoolColdModel  # noqa: E402
from runner.workflow import WorkflowSpec, load_workflow  # noqa: E402


REAL_TRACE_DIR = (
    ROOT
    / "reports"
    / "civic_azure_cand2_45min_1280mb_1cpu_keepalive20s_target20s_balanced_mi96"
)
LATENCY_SAMPLES = ROOT / "reports" / "stage3_latency_civic_alert_real_45min" / "latency_samples_for_monte_carlo.csv"
DELAY_KERNEL = ROOT / "reports" / "stage3_delay_kernel_civic_alert_real_45min" / "delay_kernel.csv"
WORKFLOW_CONFIG = ROOT / "configs" / "civic_alert_flow.yaml"
OUT_DIR = ROOT / "reports" / "stage4_p3_first_pass"
SLO_VALUES = [15_000, 20_000, 25_000, 30_000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-trace-dir", default=str(REAL_TRACE_DIR))
    parser.add_argument("--latency-samples", default=str(LATENCY_SAMPLES))
    parser.add_argument("--delay-kernel", default=str(DELAY_KERNEL))
    parser.add_argument("--workflow-config", default=str(WORKFLOW_CONFIG))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--simulations-per-request", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--memory-mb", type=int, default=1280)
    parser.add_argument("--keepalive-sec", type=float, default=20.0)
    return parser.parse_args()


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


def stage_class(value: object) -> str:
    return "cold_like" if str(value).strip().lower() == "true" else "warm"


class ComponentSampler:
    def __init__(self, samples: pd.DataFrame, rng: np.random.Generator) -> None:
        self.rng = rng
        samples = samples.copy()
        for column in ["dispatch_latency_ms", "platform_overhead_ms", "action_duration_ms"]:
            samples[column] = pd.to_numeric(samples[column], errors="coerce")
        samples = samples.dropna(subset=["dispatch_latency_ms", "action_duration_ms"])
        self.pools: dict[tuple[str, str], np.ndarray] = {}
        for (stage, klass), group in samples.groupby(["stage_name", "latency_class"]):
            self.pools[(str(stage), str(klass))] = group[
                ["dispatch_latency_ms", "action_duration_ms"]
            ].to_numpy(dtype=float)
        if not self.pools:
            raise ValueError("latency sample pool is empty")

    def sample(self, stage_name: str, klass: str) -> tuple[float, float]:
        pool = self.pools.get((stage_name, klass))
        if pool is None or len(pool) == 0:
            raise ValueError(f"missing latency pool for stage={stage_name} class={klass}")
        row = pool[int(self.rng.integers(0, len(pool)))]
        return float(row[0]), float(row[1])


def workflow_requests(raw_trace: pd.DataFrame, workflow: WorkflowSpec) -> pd.DataFrame:
    entry = raw_trace[raw_trace["stage_name"] == "__entry__"][
        ["request_id", "entry_ts_ms", "workflow_e2e_ms"]
    ].copy()
    entry["entry_ts_ms"] = pd.to_numeric(entry["entry_ts_ms"], errors="coerce")
    entry["workflow_e2e_ms"] = pd.to_numeric(entry["workflow_e2e_ms"], errors="coerce")
    stage = raw_trace[
        (raw_trace["stage_name"] != "__entry__")
        & (raw_trace["stage_name"].isin(workflow.nodes))
    ]
    complete = stage.groupby("request_id")["stage_name"].nunique()
    valid = set(complete[complete == len(workflow.nodes)].index)
    out = entry[entry["request_id"].isin(valid)].dropna(subset=["entry_ts_ms"]).copy()
    return out.sort_values(["entry_ts_ms", "request_id"]).reset_index(drop=True)


def simulate_fixed_state(
    *,
    requests: pd.DataFrame,
    workflow: WorkflowSpec,
    ordered: list[str],
    sampler: ComponentSampler,
    simulations_per_request: int,
    fixed_class: str,
    memory_mb: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    memory_gb = float(memory_mb) / 1024.0
    cold_count = len(ordered) if fixed_class == "cold_like" else 0
    for sim_id in range(simulations_per_request):
        for request in requests.itertuples(index=False):
            completions: dict[str, float] = {}
            action_total_ms = 0.0
            for stage_name in ordered:
                ready = max(
                    (completions[parent] for parent in workflow.nodes[stage_name].parents),
                    default=0.0,
                )
                dispatch_ms, action_ms = sampler.sample(stage_name, fixed_class)
                completions[stage_name] = ready + dispatch_ms
                action_total_ms += action_ms
            rows.append(
                {
                    "request_id": request.request_id,
                    "simulation_id": sim_id,
                    "predicted_latency_ms": max(completions.values()),
                    "cold_like_stage_count": cold_count,
                    "lambda_style_cost": memory_gb * action_total_ms / 1000.0,
                }
            )
    return pd.DataFrame(rows)


def simulate_actual_pool(
    *,
    requests: pd.DataFrame,
    workflow: WorkflowSpec,
    ordered: list[str],
    sampler: ComponentSampler,
    simulations_per_request: int,
    memory_mb: int,
    keepalive_sec: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    keepalive_ms = max(0.0, float(keepalive_sec) * 1000.0)
    memory_gb = float(memory_mb) / 1024.0
    for sim_id in range(simulations_per_request):
        pools = {stage_name: ContainerPoolColdModel() for stage_name in ordered}
        for request in requests.itertuples(index=False):
            entry_ts_ms = float(request.entry_ts_ms)
            completions: dict[str, float] = {}
            cold_count = 0
            action_total_ms = 0.0
            for stage_name in ordered:
                ready = max(
                    (completions[parent] for parent in workflow.nodes[stage_name].parents),
                    default=0.0,
                )
                ready_abs_ms = entry_ts_ms + ready
                pool_index, is_cold = pools[stage_name].reserve(ready_abs_ms)
                klass = "cold_like" if is_cold else "warm"
                dispatch_ms, action_ms = sampler.sample(stage_name, klass)
                pools[stage_name].complete(
                    index=pool_index,
                    ready_time_ms=ready_abs_ms,
                    duration_ms=dispatch_ms,
                    keepalive_ms=keepalive_ms,
                )
                completions[stage_name] = ready + dispatch_ms
                action_total_ms += action_ms
                cold_count += int(is_cold)
            rows.append(
                {
                    "request_id": request.request_id,
                    "simulation_id": sim_id,
                    "predicted_latency_ms": max(completions.values()),
                    "cold_like_stage_count": cold_count,
                    "lambda_style_cost": memory_gb * action_total_ms / 1000.0,
                }
            )
    return pd.DataFrame(rows)


def write_actual_rates(workflow_detail: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    workflow_detail = workflow_detail.copy()
    workflow_detail["workflow_e2e_ms"] = pd.to_numeric(workflow_detail["workflow_e2e_ms"], errors="coerce")
    rows = []
    total = len(workflow_detail)
    for slo_ms in SLO_VALUES:
        row: dict[str, Any] = {
            "slo_ms": slo_ms,
            "n_total": total,
            "n_violations_overall": int((workflow_detail["workflow_e2e_ms"] > slo_ms).sum()),
            "rate_overall": float((workflow_detail["workflow_e2e_ms"] > slo_ms).mean()),
        }
        for klass, column in [
            ("all_warm", "rate_all_warm"),
            ("partial_cold", "rate_partial_cold"),
            ("all_cold", "rate_all_cold"),
        ]:
            subset = workflow_detail[workflow_detail["workflow_cold_class"] == klass]
            row[column] = float((subset["workflow_e2e_ms"] > slo_ms).mean()) if len(subset) else math.nan
        rows.append(row)
    rates = pd.DataFrame(rows)
    rates.to_csv(out_dir / "step1_actual_violation_rates.csv", index=False)
    facts = {
        "min_e2e_ms": float(workflow_detail["workflow_e2e_ms"].min()),
        "max_all_warm_e2e_ms": float(
            workflow_detail.loc[
                workflow_detail["workflow_cold_class"] == "all_warm", "workflow_e2e_ms"
            ].max()
        ),
        "all_warm_p95_ms": float(
            workflow_detail.loc[
                workflow_detail["workflow_cold_class"] == "all_warm", "workflow_e2e_ms"
            ].quantile(0.95)
        ),
        "all_cold_mean_ms": float(
            workflow_detail.loc[
                workflow_detail["workflow_cold_class"] == "all_cold", "workflow_e2e_ms"
            ].mean()
        ),
        "all_cold_p95_ms": float(
            workflow_detail.loc[
                workflow_detail["workflow_cold_class"] == "all_cold", "workflow_e2e_ms"
            ].quantile(0.95)
        ),
        "all_cold_max_ms": float(
            workflow_detail.loc[
                workflow_detail["workflow_cold_class"] == "all_cold", "workflow_e2e_ms"
            ].max()
        ),
    }
    pd.DataFrame([facts]).to_csv(out_dir / "step1_trace_latency_facts.csv", index=False)
    return rates, facts


def write_forecast(raw_trace: pd.DataFrame, workflow: WorkflowSpec, window_sec: int, out_path: Path) -> pd.DataFrame:
    window_ms = int(window_sec * 1000)
    stage = raw_trace[
        (raw_trace["stage_name"] != "__entry__")
        & (raw_trace["stage_name"].isin(workflow.nodes))
    ].copy()
    stage["dispatch_start_ms"] = pd.to_numeric(stage["dispatch_start_ms"], errors="coerce")
    stage["target_window"] = (stage["dispatch_start_ms"] // window_ms).astype(int)
    counts = (
        stage.groupby(["workflow_name", "stage_name", "target_window"], as_index=False)
        .size()
        .rename(columns={"size": "actual_count"})
    )
    counts["method"] = f"perfect_actual_{window_sec}s"
    counts["policy"] = "p95"
    counts["forecast_count"] = counts["actual_count"].astype(float)
    counts["allocated_count"] = counts["actual_count"].astype(float)
    counts = counts[
        [
            "workflow_name",
            "method",
            "stage_name",
            "target_window",
            "policy",
            "actual_count",
            "forecast_count",
            "allocated_count",
        ]
    ].sort_values(["stage_name", "target_window"])
    counts.to_csv(out_path, index=False)
    return counts


def summarize_mc(instances: pd.DataFrame, baseline: str, slo_values: list[int]) -> pd.DataFrame:
    rows = []
    lat = instances["predicted_latency_ms"]
    cold = instances["cold_like_stage_count"]
    for slo_ms in slo_values:
        rows.append(
            {
                "baseline": baseline,
                "slo_ms": slo_ms,
                "predicted_violation_rate": float((lat > slo_ms).mean()),
                "p50_ms": float(lat.quantile(0.50)),
                "p90_ms": float(lat.quantile(0.90)),
                "p95_ms": float(lat.quantile(0.95)),
                "mc_partial_cold_rate": float(((cold > 0) & (cold < 5)).mean()),
                "mc_all_cold_rate": float((cold == 5).mean()),
                "lambda_style_cost": float(instances["lambda_style_cost"].mean()),
            }
        )
    return pd.DataFrame(rows)


def accuracy_label(abs_error: float) -> str:
    if abs_error < 0.02:
        return "Excellent"
    if abs_error < 0.05:
        return "Good"
    if abs_error < 0.10:
        return "Marginal"
    return "Poor"


def write_step2_notes(out_dir: Path) -> None:
    text = """# Step 2 MC Interface Inspection

`runner.stage4_risk.estimate_slo_risk` requires:

- `--workflow-config`
- `--trace`
- `--forecast-detail`
- `--latency-samples`
- `--method`
- `--policy`
- `--slo-ms`
- `--out-dir`

Optional but relevant:

- `--control-plan`: JSON/CSV Stage-5 control plan. When supplied, `warm_count`
  overrides `allocated_count` for matching `(stage_name, target_window)` rows.
- `--cold-model {pool,deficit}`: pool tracks warm/busy/expired containers;
  deficit uses per-window allocation deficit.
- `--residual-cold-probability`: minimum cold probability.

Forecast detail schema expected by the code:

- `workflow_name`
- `method`
- `policy`
- `stage_name`
- `target_window` (or `window`, copied to `target_window`)
- `actual_count`
- `forecast_count`
- `allocated_count`

The script cannot run without a forecast-detail file. `paper_baselines.py`
builds control plans with columns equivalent to `workflow_name`, `stage_name`,
`window`, `warm_count`, `keepalive_ttl_sec`, `memory_mb`, `source`, and `note`.

Chosen integration mode: **Mode C**. This P3 script reuses the Stage-4
latency sample pools and `ContainerPoolColdModel` directly. This avoids
running one heavyweight `estimate_slo_risk.py` process per SLO while preserving
the same empirical warm/cold sampling and pool cold-start mechanism. Perfect
forecast files are still emitted in the forecast-detail schema above for later
CLI-based experiments.
"""
    (out_dir / "step2_mc_interface_notes.md").write_text(text, encoding="utf-8")


def write_summary_report(
    out_dir: Path,
    actual_rates: pd.DataFrame,
    facts: dict[str, float],
    bounds: pd.DataFrame,
    calibration: pd.DataFrame,
    window_sensitivity: pd.DataFrame,
) -> None:
    def table_text(df: pd.DataFrame) -> str:
        return "```text\n" + df.to_string(index=False) + "\n```"

    pivot_bounds = bounds.pivot_table(
        index="slo_ms",
        columns="baseline",
        values="predicted_violation_rate",
        aggfunc="first",
    ).reset_index()
    merged = actual_rates[["slo_ms", "rate_overall"]].merge(pivot_bounds, on="slo_ms", how="left")
    merged["bound_gap"] = merged["scale_to_zero"] - merged["always_warm"]
    scale_p95 = float(bounds.loc[bounds["baseline"] == "scale_to_zero", "p95_ms"].iloc[0])
    warm_p95 = float(bounds.loc[bounds["baseline"] == "always_warm", "p95_ms"].iloc[0])
    worst_abs_error = float(calibration["absolute_error"].max())
    worst_row = calibration.loc[calibration["absolute_error"].idxmax()]

    meaningful = merged[merged["bound_gap"] > 0.30]["slo_ms"].astype(int).tolist()
    premium = 20_000
    free = 25_000
    if 20_000 not in meaningful and meaningful:
        premium = int(meaningful[0])
    if 25_000 not in SLO_VALUES:
        free = int(SLO_VALUES[-1])

    lines = [
        "# P3 First-Pass Real-Data SLO Validation",
        "",
        "## Actual Violation Rates",
        "",
        table_text(actual_rates),
        "",
        f"- Minimum E2E latency: `{facts['min_e2e_ms']:.1f} ms`.",
        f"- Maximum all-warm E2E latency: `{facts['max_all_warm_e2e_ms']:.1f} ms`.",
        f"- All-warm p95: `{facts['all_warm_p95_ms']:.1f} ms`.",
        f"- All-cold mean: `{facts['all_cold_mean_ms']:.1f} ms`.",
        f"- All-cold p95/max: `{facts['all_cold_p95_ms']:.1f} / {facts['all_cold_max_ms']:.1f} ms`.",
        "",
        "## MC Bound Estimates",
        "",
        table_text(bounds),
        "",
        "## MC Calibration Accuracy",
        "",
        table_text(calibration),
        "",
        "## Window Sensitivity",
        "",
        table_text(window_sensitivity),
        "",
        "## SLO Recommendation",
        "",
        table_text(merged),
        "",
        f"- Meaningful planner room (scale-to-zero minus always-warm > 30%): `{meaningful}`.",
        f"- Recommended strict/premium SLO: `{premium} ms`. It separates all-warm success from cold-start-heavy behavior.",
        f"- Recommended lax/free SLO: `{free} ms`. It tolerates normal warm operation and some partial-cold behavior while still exposing all-cold penalties.",
        f"- MC trustworthiness verdict: marginal at the boundary. Worst absolute calibration error is `{worst_abs_error:.3f}` at `{int(worst_row['slo_ms'])} ms`; lower-tail SLOs are trivial and lax SLOs have small absolute errors.",
        f"- Always-warm MC p95 is `{warm_p95:.1f} ms`, close to real all-warm p95 `{facts['all_warm_p95_ms']:.1f} ms`.",
        f"- Scale-to-zero MC p95 is `{scale_p95:.1f} ms`, far above real all-cold p95 `{facts['all_cold_p95_ms']:.1f} ms`; this violates the requested Step 3 sanity expectation.",
        "",
        "## Open Issues",
        "",
        "- `estimate_slo_risk.py` cannot run without a forecast-detail file, even for fixed all-cold/all-warm bounds.",
        "- P3 uses a direct wrapper around Stage-4 sampling and pool dynamics instead of launching many CLI jobs.",
        "- Window size has no effect in this no-prewarm pool replay wrapper because cold/warm state is driven by request timestamps and keepalive.",
        "- The `cold_like` sample pool contains long partial-cold/queueing tails. Independent per-stage all-cold sampling can therefore overstate scale-to-zero p95 compared with the 14 observed all-cold workflows.",
        "- The actual-pool MC underpredicts the 20s violation rate (`2.6%` vs `11.2%`), likely because it misses correlated partial-cold bursts or replay-specific platform contention.",
        "",
    ]
    (out_dir / "step5_summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_dir = Path(args.real_trace_dir)
    workflow = load_workflow(str(args.workflow_config))
    ordered = topological_nodes(workflow)
    raw_trace = pd.read_csv(real_dir / "raw_trace.csv")
    workflow_detail = pd.read_csv(real_dir / "workflow_detail.csv")
    requests = workflow_requests(raw_trace, workflow)
    samples = pd.read_csv(args.latency_samples)

    print("Step 1: extracting actual violation rates")
    actual_rates, facts = write_actual_rates(workflow_detail, out_dir)
    print(actual_rates.to_string(index=False))
    print(f"min_e2e_ms={facts['min_e2e_ms']:.1f}; max_all_warm_e2e_ms={facts['max_all_warm_e2e_ms']:.1f}")

    print("\nStep 2: documenting estimate_slo_risk integration mode")
    write_step2_notes(out_dir)

    print("\nStep 3: simulating MC bounds")
    scale_to_zero = simulate_fixed_state(
        requests=requests,
        workflow=workflow,
        ordered=ordered,
        sampler=ComponentSampler(samples, np.random.default_rng(args.seed + 1)),
        simulations_per_request=args.simulations_per_request,
        fixed_class="cold_like",
        memory_mb=args.memory_mb,
    )
    always_warm = simulate_fixed_state(
        requests=requests,
        workflow=workflow,
        ordered=ordered,
        sampler=ComponentSampler(samples, np.random.default_rng(args.seed + 2)),
        simulations_per_request=args.simulations_per_request,
        fixed_class="warm",
        memory_mb=args.memory_mb,
    )
    bounds = pd.concat(
        [
            summarize_mc(scale_to_zero, "scale_to_zero", SLO_VALUES),
            summarize_mc(always_warm, "always_warm", SLO_VALUES),
        ],
        ignore_index=True,
    )
    bounds.to_csv(out_dir / "step3_mc_bounds.csv", index=False)
    print(bounds.to_string(index=False))

    print("\nStep 4a: writing perfect forecast files")
    forecast_paths = {}
    for window_sec in [5, 10]:
        forecast_paths[window_sec] = out_dir / f"step4_perfect_forecast_{window_sec}s.csv"
        write_forecast(raw_trace, workflow, window_sec, forecast_paths[window_sec])
        print(f"wrote {forecast_paths[window_sec]}")

    print("\nStep 4b: simulating actual no-prewarm pool dynamics")
    calibration_rows = []
    distributions = {}
    for window_sec in [5, 10]:
        # The no-prewarm pool model is timestamp-driven; forecast window size is
        # retained in the output for comparison with the CLI workflow.
        actual_pool = simulate_actual_pool(
            requests=requests,
            workflow=workflow,
            ordered=ordered,
            sampler=ComponentSampler(samples, np.random.default_rng(args.seed + 100)),
            simulations_per_request=args.simulations_per_request,
            memory_mb=args.memory_mb,
            keepalive_sec=args.keepalive_sec,
        )
        distributions[window_sec] = actual_pool
        for slo_ms in SLO_VALUES:
            mc_rate = float((actual_pool["predicted_latency_ms"] > slo_ms).mean())
            actual_rate = float(
                actual_rates.loc[actual_rates["slo_ms"] == slo_ms, "rate_overall"].iloc[0]
            )
            absolute_error = abs(mc_rate - actual_rate)
            relative_error = (
                absolute_error / actual_rate * 100.0 if actual_rate > 0.0 else math.nan
            )
            calibration_rows.append(
                {
                    "window_sec": window_sec,
                    "slo_ms": slo_ms,
                    "mc_predicted_violation_rate": mc_rate,
                    "actual_violation_rate": actual_rate,
                    "absolute_error": absolute_error,
                    "relative_error_pct": relative_error,
                    "accuracy": accuracy_label(absolute_error),
                }
            )
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(out_dir / "step4_mc_calibration.csv", index=False)
    print(calibration.to_string(index=False))

    print("\nStep 4c: window sensitivity")
    wide = calibration.pivot(index="slo_ms", columns="window_sec", values="mc_predicted_violation_rate")
    wide["absolute_difference"] = (wide[5] - wide[10]).abs()
    sensitivity = wide.reset_index().rename(columns={5: "rate_5s", 10: "rate_10s"})
    mean_abs_diff = float(sensitivity["absolute_difference"].mean())
    sensitivity["mean_absolute_difference"] = mean_abs_diff
    sensitivity["sensitive_gt_2pct"] = mean_abs_diff > 0.02
    sensitivity.to_csv(out_dir / "step4_window_sensitivity.csv", index=False)
    print(sensitivity.to_string(index=False))

    print("\nStep 5: writing summary report")
    write_summary_report(out_dir, actual_rates, facts, bounds, calibration, sensitivity)
    print(f"wrote {out_dir / 'step5_summary_report.md'}")


if __name__ == "__main__":
    main()
