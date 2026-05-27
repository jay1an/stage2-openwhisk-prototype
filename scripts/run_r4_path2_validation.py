#!/usr/bin/env python3
"""R4 end-to-end validation, no-JIT calibration, and plan-grid evaluation."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.stage4_risk.dag_aggregation import (  # noqa: E402
    CIVIC_ALERT_CRITICAL_PATH_EDGES,
    LogNormalParams,
    aggregate_civic_alert,
)
from runner.stage4_risk.entry_cold import calibrated_entry_cold_probability, calibrate_p_baseline  # noqa: E402
from runner.stage4_risk.plan_risk import (  # noqa: E402
    PlanInput,
    compute_cold_overhead_per_stage,
    load_lognormal_params,
)
from runner.stage4_risk.scaling import scale_stage_for_memory_tier  # noqa: E402


STAGES = [
    "detect_object",
    "estimate_pose",
    "match_face",
    "classify_scene",
    "translate_alert",
]
SLO_VALUES = [15_000, 20_000, 25_000, 30_000]
TARGET_VIOLATION_RATES = [0.01, 0.05, 0.10]
MEMORY_TIERS = [768, 1280, 2048, 2560]
PLAN_WINDOW_SEC = 5.0
RAW_TRACE = (
    ROOT
    / "reports"
    / "civic_azure_cand2_45min_1280mb_1cpu_keepalive20s_target20s_balanced_mi96"
    / "raw_trace.csv"
)
LOGNORMAL_PARAMS = ROOT / "reports" / "path2_lognormal_fit" / "per_stage_lognormal_params.csv"
AMDAHL_PARAMS = ROOT / "reports" / "stage6_amdahl_model" / "per_stage_amdahl_params.csv"
CALIBRATION_DIR = ROOT / "reports" / "path2_calibration"
NO_JIT_DIR = ROOT / "reports" / "path2_no_jit_validation"
PLAN_GRID_DIR = ROOT / "reports" / "path2_plan_grid"


def table_text(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def truthy(value: object) -> bool:
    return str(value).strip().lower() == "true"


def load_trace_tables(raw_trace_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(raw_trace_path)
    entry = raw[raw["stage_name"] == "__entry__"][
        ["request_id", "workflow_e2e_ms"]
    ].copy()
    entry["workflow_e2e_ms"] = pd.to_numeric(entry["workflow_e2e_ms"], errors="coerce")

    stage = raw[raw["stage_name"].isin(STAGES)].copy()
    stage["dispatch_latency_ms"] = pd.to_numeric(stage["dispatch_latency_ms"], errors="coerce")
    stage["cold_bool"] = stage["cold_like"].map(truthy)
    dispatch = stage.pivot_table(
        index="request_id",
        columns="stage_name",
        values="dispatch_latency_ms",
        aggfunc="first",
    )
    cold = stage.pivot_table(
        index="request_id",
        columns="stage_name",
        values="cold_bool",
        aggfunc="first",
    )
    dispatch = dispatch.reindex(columns=STAGES).dropna()
    cold = cold.reindex(columns=STAGES).dropna()
    common = dispatch.index.intersection(cold.index)
    entry = entry[entry["request_id"].isin(common)].dropna(subset=["workflow_e2e_ms"])
    common = common.intersection(entry["request_id"])
    dispatch = dispatch.loc[common]
    cold = cold.loc[common].astype(bool)
    entry = entry.set_index("request_id").loc[common].reset_index()
    return entry, dispatch, cold


def cold_patterns(entry: pd.DataFrame, cold: pd.DataFrame) -> pd.DataFrame:
    patterns = cold.copy()
    out = pd.DataFrame(index=patterns.index)
    out["cold_pattern"] = patterns.apply(
        lambda row: "".join("1" if bool(row[stage]) else "0" for stage in STAGES),
        axis=1,
    )
    out["n_cold_stages"] = patterns[STAGES].sum(axis=1).astype(int)
    out = out.reset_index().rename(columns={"index": "request_id"})
    out = out.merge(entry, on="request_id", how="left")
    out = out.rename(columns={"workflow_e2e_ms": "observed_e2e_ms"})
    return out[["request_id", "observed_e2e_ms", "cold_pattern", "n_cold_stages"]]


def calibrate_transition_gap(entry: pd.DataFrame, dispatch: pd.DataFrame, patterns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    predicted_sum = dispatch[STAGES].sum(axis=1).rename("predicted_sum")
    gap = (
        patterns[patterns["cold_pattern"] == "00000"][["request_id", "observed_e2e_ms"]]
        .merge(predicted_sum, left_on="request_id", right_index=True, how="left")
        .dropna(subset=["predicted_sum"])
        .rename(columns={"request_id": "workflow_id", "predicted_sum": "predicted_sum"})
    )
    gap["gap_ms"] = gap["observed_e2e_ms"] - gap["predicted_sum"]
    gap = gap.rename(columns={"observed_e2e_ms": "observed_e2e"})
    gap = gap[["workflow_id", "observed_e2e", "predicted_sum", "gap_ms"]]

    summary_values = {
        "all_warm_workflow_count": float(len(gap)),
        "gap_mean_ms": float(gap["gap_ms"].mean()),
        "gap_std_ms": float(gap["gap_ms"].std(ddof=1)),
        "gap_p50_ms": float(gap["gap_ms"].quantile(0.50)),
        "gap_p95_ms": float(gap["gap_ms"].quantile(0.95)),
    }
    summary_values["per_edge_overhead_ms"] = summary_values["gap_mean_ms"] / CIVIC_ALERT_CRITICAL_PATH_EDGES
    summary = pd.DataFrame(
        [{"metric": metric, "value": value} for metric, value in summary_values.items()]
    )
    return gap, summary, summary_values["per_edge_overhead_ms"]


def stage_dist_for_pattern(
    pattern: str,
    lognormal_params: dict[str, dict[str, LogNormalParams]],
    memory_config: dict[str, int] | None = None,
    amdahl_params: pd.DataFrame | None = None,
    cold_overhead: dict[str, float] | None = None,
) -> dict[str, LogNormalParams]:
    if len(pattern) != len(STAGES):
        raise ValueError(f"pattern must have {len(STAGES)} bits, got {pattern!r}")
    out: dict[str, LogNormalParams] = {}
    for stage_name, bit in zip(STAGES, pattern, strict=True):
        latency_class = "cold_like" if bit == "1" else "warm"
        base = lognormal_params[stage_name][latency_class]
        if memory_config is None:
            out[stage_name] = base
        else:
            if amdahl_params is None or cold_overhead is None:
                raise ValueError("amdahl_params and cold_overhead are required for memory scaling")
            out[stage_name] = scale_stage_for_memory_tier(
                stage_name=stage_name,
                latency_class=latency_class,
                target_memory_mb=int(memory_config[stage_name]),
                base_memory_mb=1280,
                base_params=base,
                amdahl_params=amdahl_params,
                cold_overhead_ms=cold_overhead.get(stage_name),
            )
    return out


def aggregate_pattern(
    pattern: str,
    lognormal_params: dict[str, dict[str, LogNormalParams]],
    transition_overhead_ms: float,
    memory_config: dict[str, int] | None = None,
    amdahl_params: pd.DataFrame | None = None,
    cold_overhead: dict[str, float] | None = None,
) -> LogNormalParams:
    stage_dists = stage_dist_for_pattern(
        pattern,
        lognormal_params,
        memory_config=memory_config,
        amdahl_params=amdahl_params,
        cold_overhead=cold_overhead,
    )
    return aggregate_civic_alert(stage_dists, transition_overhead_ms=transition_overhead_ms)


def transition_model_calibration(
    patterns: pd.DataFrame,
    lognormal_params: dict[str, dict[str, LogNormalParams]],
    transition_overhead_ms: float,
) -> pd.DataFrame:
    rows = []
    for scenario, pattern in [("entry_warm", "00000"), ("entry_cold", "10000")]:
        real = patterns[patterns["cold_pattern"] == pattern]["observed_e2e_ms"]
        model = aggregate_pattern(pattern, lognormal_params, transition_overhead_ms)
        rows.append(
            {
                "scenario": scenario,
                "p50_real": float(real.quantile(0.50)) if len(real) else math.nan,
                "p50_model": model.quantile(0.50),
                "p95_real": float(real.quantile(0.95)) if len(real) else math.nan,
                "p95_model": model.quantile(0.95),
                "p99_real": float(real.quantile(0.99)) if len(real) else math.nan,
                "p99_model": model.quantile(0.99),
            }
        )
    return pd.DataFrame(rows)


def transition_mc_validation(
    lognormal_params: dict[str, dict[str, LogNormalParams]],
    transition_overhead_ms: float,
    mc_samples: int = 100_000,
    seed: int = 20260527,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for scenario, pattern in [("entry_warm", "00000"), ("entry_cold", "10000")]:
        model = aggregate_pattern(pattern, lognormal_params, transition_overhead_ms)
        stage_dists = stage_dist_for_pattern(pattern, lognormal_params)
        samples = {
            stage_name: rng.lognormal(dist.mu, dist.sigma, size=mc_samples)
            for stage_name, dist in stage_dists.items()
        }
        raw = (
            np.maximum(
                samples["detect_object"] + samples["estimate_pose"] + samples["match_face"],
                samples["detect_object"],
            )
            + samples["classify_scene"]
            + samples["translate_alert"]
            + CIVIC_ALERT_CRITICAL_PATH_EDGES * transition_overhead_ms
        )
        for p in [0.50, 0.75, 0.90, 0.95, 0.99]:
            analytical = model.quantile(p)
            mc = float(np.quantile(raw, p))
            rows.append(
                {
                    "scenario": scenario,
                    "percentile": f"p{int(p * 100)}",
                    "analytical_value": analytical,
                    "mc_value": mc,
                    "abs_error": abs(analytical - mc),
                    "rel_error_pct": abs(analytical - mc) / mc * 100.0,
                }
            )
    return pd.DataFrame(rows)


def run_no_jit_validation(
    patterns: pd.DataFrame,
    lognormal_params: dict[str, dict[str, LogNormalParams]],
    transition_overhead_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dist_cache = {
        pattern: aggregate_pattern(pattern, lognormal_params, transition_overhead_ms)
        for pattern in sorted(patterns["cold_pattern"].unique())
    }
    predictions = patterns.copy()
    predictions["predicted_e2e_mean"] = predictions["cold_pattern"].map(
        lambda pattern: dist_cache[pattern].mean
    )
    predictions["predicted_e2e_p50"] = predictions["cold_pattern"].map(
        lambda pattern: dist_cache[pattern].quantile(0.50)
    )
    predictions["predicted_e2e_p95"] = predictions["cold_pattern"].map(
        lambda pattern: dist_cache[pattern].quantile(0.95)
    )
    predictions = predictions.rename(columns={"observed_e2e_ms": "observed_e2e"})

    grouped_rows = []
    for pattern, group in predictions.groupby("cold_pattern", sort=True):
        dist = dist_cache[pattern]
        observed_mean = float(group["observed_e2e"].mean())
        predicted_mean = dist.mean
        observed_p95 = float(group["observed_e2e"].quantile(0.95))
        predicted_p95 = dist.quantile(0.95)
        grouped_rows.append(
            {
                "pattern": pattern,
                "n_cold_stages": int(group["n_cold_stages"].iloc[0]),
                "count": int(len(group)),
                "observed_e2e_mean": observed_mean,
                "predicted_e2e_mean": predicted_mean,
                "abs_error": abs(predicted_mean - observed_mean),
                "rel_error_pct": abs(predicted_mean - observed_mean) / observed_mean * 100.0,
                "observed_e2e_p95": observed_p95,
                "predicted_e2e_p95": predicted_p95,
            }
        )
    grouped = pd.DataFrame(grouped_rows).sort_values(["n_cold_stages", "pattern"])

    calibration_rows = []
    for slo_ms in SLO_VALUES:
        observed = float((predictions["observed_e2e"] > slo_ms).mean())
        predicted = float(
            np.mean([dist_cache[pattern].survival(slo_ms) for pattern in predictions["cold_pattern"]])
        )
        calibration_rows.append(
            {
                "slo_ms": slo_ms,
                "observed_violation_rate": observed,
                "predicted_violation_rate": predicted,
                "abs_error": abs(predicted - observed),
                "rel_error_pct": abs(predicted - observed) / observed * 100.0 if observed > 0.0 else math.nan,
            }
        )
    calibration = pd.DataFrame(calibration_rows)
    return predictions, grouped, calibration


def all_memory(memory_mb: int) -> dict[str, int]:
    return {stage: memory_mb for stage in STAGES}


def plan_memory_configs() -> list[tuple[str, str, dict[str, int]]]:
    configs: list[tuple[str, str, dict[str, int]]] = []
    for tier in MEMORY_TIERS:
        configs.append((f"uniform_{tier}", f"all stages {tier}MB", all_memory(tier)))
    for tier in [768, 2048, 2560]:
        memory = all_memory(1280)
        memory["detect_object"] = tier
        configs.append((f"entry_only_{tier}", f"entry {tier}MB, downstream 1280MB", memory))
    for tier in [768, 2048, 2560]:
        memory = all_memory(1280)
        for stage in STAGES[1:]:
            memory[stage] = tier
        configs.append((f"downstream_{tier}", f"entry 1280MB, downstream {tier}MB", memory))
    for tier in [768, 2048, 2560]:
        memory = all_memory(1280)
        for stage in ["detect_object", "estimate_pose", "match_face"]:
            memory[stage] = tier
        configs.append((f"path1_{tier}", f"detect/pose/match {tier}MB, tail 1280MB", memory))
    return configs


def scenario_stage_dists(
    memory_config: dict[str, int],
    pattern: str,
    lognormal_params: dict[str, dict[str, LogNormalParams]],
    amdahl_params: pd.DataFrame,
    cold_overhead: dict[str, float],
) -> dict[str, LogNormalParams]:
    return stage_dist_for_pattern(
        pattern,
        lognormal_params,
        memory_config=memory_config,
        amdahl_params=amdahl_params,
        cold_overhead=cold_overhead,
    )


def expected_lambda_cost_gb_seconds(
    *,
    memory_config: dict[str, int],
    warm_stage_dists: dict[str, LogNormalParams],
    cold_entry_stage_dists: dict[str, LogNormalParams],
    p_entry_cold: float,
    predicted_arrivals: float,
    entry_prewarm_count: float,
) -> float:
    def workflow_gb_s(stage_dists: dict[str, LogNormalParams]) -> float:
        return sum(
            float(memory_config[stage]) / 1024.0 * stage_dists[stage].mean / 1000.0
            for stage in STAGES
        )

    expected_per_workflow = (
        (1.0 - p_entry_cold) * workflow_gb_s(warm_stage_dists)
        + p_entry_cold * workflow_gb_s(cold_entry_stage_dists)
    )
    execution_cost = float(predicted_arrivals) * expected_per_workflow
    prewarm_cost = (
        float(entry_prewarm_count)
        * float(memory_config["detect_object"])
        / 1024.0
        * PLAN_WINDOW_SEC
    )
    return execution_cost + prewarm_cost


def evaluate_plan_grid(
    lognormal_params: dict[str, dict[str, LogNormalParams]],
    amdahl_params: pd.DataFrame,
    cold_overhead: dict[str, float],
    p_baseline: float,
    transition_overhead_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for memory_label, memory_description, memory_config in plan_memory_configs():
        for entry_prewarm in [0, 1, 2, 5, 10]:
            for predicted_arrivals in [5, 10]:
                plan_id = f"{memory_label}_prewarm{entry_prewarm}_arr{predicted_arrivals}"
                p_entry_cold = calibrated_entry_cold_probability(
                    predicted_arrivals=float(predicted_arrivals),
                    entry_prewarm_count=float(entry_prewarm),
                    zero_prewarm_cold_rate=p_baseline,
                    residual_floor=0.01,
                )
                warm_stage_dists = scenario_stage_dists(
                    memory_config, "00000", lognormal_params, amdahl_params, cold_overhead
                )
                cold_entry_stage_dists = scenario_stage_dists(
                    memory_config, "10000", lognormal_params, amdahl_params, cold_overhead
                )
                e2e_warm = aggregate_civic_alert(
                    warm_stage_dists,
                    transition_overhead_ms=transition_overhead_ms,
                )
                e2e_cold = aggregate_civic_alert(
                    cold_entry_stage_dists,
                    transition_overhead_ms=transition_overhead_ms,
                )
                lambda_cost = expected_lambda_cost_gb_seconds(
                    memory_config=memory_config,
                    warm_stage_dists=warm_stage_dists,
                    cold_entry_stage_dists=cold_entry_stage_dists,
                    p_entry_cold=p_entry_cold,
                    predicted_arrivals=float(predicted_arrivals),
                    entry_prewarm_count=float(entry_prewarm),
                )
                for slo_ms in SLO_VALUES:
                    p_warm = e2e_warm.survival(float(slo_ms))
                    p_cold = e2e_cold.survival(float(slo_ms))
                    p_total = (1.0 - p_entry_cold) * p_warm + p_entry_cold * p_cold
                    rows.append(
                        {
                            "plan_id": plan_id,
                            "memory_config_label": memory_label,
                            "memory_config_description": memory_description,
                            "memory_tier_per_stage": json.dumps(memory_config, sort_keys=True),
                            "entry_prewarm": entry_prewarm,
                            "predicted_arrivals": predicted_arrivals,
                            "slo_ms": slo_ms,
                            "p_entry_cold": p_entry_cold,
                            "e2e_warm_p95": e2e_warm.quantile(0.95),
                            "e2e_cold_p95": e2e_cold.quantile(0.95),
                            "p_violation_warm": p_warm,
                            "p_violation_cold_entry": p_cold,
                            "p_violation_total": p_total,
                            "lambda_cost": lambda_cost,
                        }
                    )
    full = pd.DataFrame(rows)

    best_rows = []
    for (slo_ms, predicted_arrivals), group in full.groupby(["slo_ms", "predicted_arrivals"], sort=True):
        for target in TARGET_VIOLATION_RATES:
            feasible = group[group["p_violation_total"] <= target].copy()
            if feasible.empty:
                best_rows.append(
                    {
                        "slo_ms": slo_ms,
                        "predicted_arrivals": predicted_arrivals,
                        "target_violation_rate": target,
                        "min_cost_plan_id": "",
                        "plan_config": "",
                        "achieved_cost": math.nan,
                        "achieved_violation": math.nan,
                    }
                )
            else:
                feasible = feasible.sort_values(
                    ["lambda_cost", "p_violation_total", "entry_prewarm", "plan_id"]
                )
                best = feasible.iloc[0]
                best_rows.append(
                    {
                        "slo_ms": slo_ms,
                        "predicted_arrivals": predicted_arrivals,
                        "target_violation_rate": target,
                        "min_cost_plan_id": best["plan_id"],
                        "plan_config": best["memory_tier_per_stage"],
                        "achieved_cost": best["lambda_cost"],
                        "achieved_violation": best["p_violation_total"],
                    }
                )
    best_plan = pd.DataFrame(best_rows)

    pareto_rows = []
    for (slo_ms, predicted_arrivals), group in full.groupby(["slo_ms", "predicted_arrivals"], sort=True):
        group = group.copy()
        is_pareto = []
        for row in group.itertuples(index=False):
            dominated = group[
                (group["lambda_cost"] <= row.lambda_cost)
                & (group["p_violation_total"] <= row.p_violation_total)
                & (
                    (group["lambda_cost"] < row.lambda_cost)
                    | (group["p_violation_total"] < row.p_violation_total)
                )
            ]
            is_pareto.append(dominated.empty)
        group["is_pareto_optimal"] = is_pareto
        pareto_rows.append(
            group[
                [
                    "slo_ms",
                    "predicted_arrivals",
                    "plan_id",
                    "memory_config_label",
                    "entry_prewarm",
                    "lambda_cost",
                    "p_violation_total",
                    "is_pareto_optimal",
                ]
            ]
        )
    pareto = pd.concat(pareto_rows, ignore_index=True)
    return full, best_plan, pareto


def write_final_report(
    *,
    out_path: Path,
    transition_summary: pd.DataFrame,
    transition_calibration: pd.DataFrame,
    no_jit_calibration: pd.DataFrame,
    plan_grid: pd.DataFrame,
    best_plan: pd.DataFrame,
    pareto: pd.DataFrame,
) -> None:
    summary = dict(zip(transition_summary["metric"], transition_summary["value"], strict=False))
    nojit_20 = no_jit_calibration[no_jit_calibration["slo_ms"] == 20_000].iloc[0]
    warm_cal = transition_calibration[transition_calibration["scenario"] == "entry_warm"].iloc[0]
    cold_cal = transition_calibration[transition_calibration["scenario"] == "entry_cold"].iloc[0]
    plan20 = plan_grid[plan_grid["slo_ms"] == 20_000]
    under_1 = plan20[plan20["p_violation_total"] < 0.01]
    under_5 = plan20[plan20["p_violation_total"] < 0.05]
    uniform_1280 = plan20[
        (plan20["memory_config_label"] == "uniform_1280")
        & (plan20["predicted_arrivals"] == 5)
    ].sort_values("entry_prewarm")
    memory_only = plan20[
        (plan20["entry_prewarm"] == 0)
        & (plan20["predicted_arrivals"] == 5)
        & (plan20["memory_config_label"].isin(["uniform_1280", "uniform_2048"]))
    ][["memory_config_label", "p_violation_total", "lambda_cost"]]

    lines = [
        "# R4 Path 2 Final Validation Report",
        "",
        "## Transition Overhead Calibration",
        "",
        f"- All-warm trace gap mean: `{summary['gap_mean_ms']:.3f} ms`.",
        f"- All-warm trace gap p95: `{summary['gap_p95_ms']:.3f} ms`.",
        f"- Calibrated per-edge overhead: `{summary['per_edge_overhead_ms']:.3f} ms`.",
        "- This direct trace calibration does not explain a 1.15s p95 model-vs-real gap; the observed inter-stage timing gap is only a few milliseconds.",
        f"- After applying this overhead, entry-warm p95 remains `{warm_cal['p95_model']:.1f} ms` vs real `{warm_cal['p95_real']:.1f} ms`.",
        f"- Entry-cold-only p95 remains `{cold_cal['p95_model']:.1f} ms` vs real `{cold_cal['p95_real']:.1f} ms`.",
        "",
        "Transition calibration table:",
        table_text(transition_calibration),
        "",
        "## No-JIT Validation",
        "",
        table_text(no_jit_calibration),
        "",
        f"- 20s observed violation: `{nojit_20['observed_violation_rate']:.6f}`.",
        f"- 20s predicted violation: `{nojit_20['predicted_violation_rate']:.6f}`.",
        f"- 20s absolute error: `{nojit_20['abs_error']:.6f}`.",
        f"- Acceptance [9.2%, 13.2%]: `{0.092 <= nojit_20['predicted_violation_rate'] <= 0.132}`.",
        "- This misses the 2pp acceptance band. The dominant visible cause is warm-tail underfit: the all-warm p95 stays about 1.15s below the real all-warm p95 even after transition calibration.",
        "- A second likely cause is missing per-workflow stage-latency correlation, which makes real workflows jointly slow more often than independent per-stage lognormal draws.",
        "",
        "## Plan Evaluation Key Findings",
        "",
        f"- Plans achieving <1% violation at 20s: `{len(under_1)}` rows.",
        f"- Plans achieving <5% violation at 20s: `{len(under_5)}` rows.",
        f"- Cost range across grid: `{plan_grid['lambda_cost'].min():.6f}` to `{plan_grid['lambda_cost'].max():.6f}` GB-s/window.",
        "",
        "Entry prewarm sweep at uniform 1280MB, arrivals=5, SLO=20s:",
        table_text(uniform_1280[["entry_prewarm", "p_entry_cold", "p_violation_total", "lambda_cost"]]),
        "",
        "Memory-only comparison at arrivals=5, SLO=20s:",
        table_text(memory_only),
        "",
        "Best plans per budget sample:",
        table_text(best_plan.head(12)),
        "",
        "Pareto rows flagged:",
        f"- `{int(pareto['is_pareto_optimal'].sum())}` of `{len(pareto)}` evaluated rows are Pareto-optimal within their `(SLO, arrivals)` groups.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    NO_JIT_DIR.mkdir(parents=True, exist_ok=True)
    PLAN_GRID_DIR.mkdir(parents=True, exist_ok=True)

    entry, dispatch, cold = load_trace_tables(RAW_TRACE)
    patterns = cold_patterns(entry, cold)
    lognormal_params = load_lognormal_params(LOGNORMAL_PARAMS)
    amdahl_params = pd.read_csv(AMDAHL_PARAMS)
    cold_overhead = compute_cold_overhead_per_stage(lognormal_params)
    p_baseline = calibrate_p_baseline(str(RAW_TRACE), "detect_object")

    print("Part 1: transition overhead calibration")
    gap, gap_summary, transition_overhead_ms = calibrate_transition_gap(entry, dispatch, patterns)
    gap.to_csv(CALIBRATION_DIR / "transition_gap_analysis.csv", index=False)
    gap_summary.to_csv(CALIBRATION_DIR / "transition_gap_summary.csv", index=False)
    transition_calibration = transition_model_calibration(
        patterns,
        lognormal_params,
        transition_overhead_ms,
    )
    transition_calibration.to_csv(
        CALIBRATION_DIR / "transition_overhead_calibration.csv",
        index=False,
    )
    transition_mc = transition_mc_validation(lognormal_params, transition_overhead_ms)
    transition_mc.to_csv(CALIBRATION_DIR / "transition_mc_validation.csv", index=False)
    print(gap_summary.to_string(index=False))
    print(transition_calibration.to_string(index=False))
    print(transition_mc.to_string(index=False))

    print("\nPart 2: no-JIT validation")
    patterns.to_csv(NO_JIT_DIR / "per_workflow_cold_pattern.csv", index=False)
    predictions, grouped, no_jit_calibration = run_no_jit_validation(
        patterns,
        lognormal_params,
        transition_overhead_ms,
    )
    predictions.to_csv(NO_JIT_DIR / "per_workflow_predictions.csv", index=False)
    grouped.to_csv(NO_JIT_DIR / "grouped_validation.csv", index=False)
    no_jit_calibration.to_csv(NO_JIT_DIR / "no_jit_calibration.csv", index=False)
    print(patterns["cold_pattern"].value_counts().head(15).to_string())
    print(no_jit_calibration.to_string(index=False))
    print(grouped.head(20).to_string(index=False))

    print("\nPart 3: plan grid")
    full_plan, best_plan, pareto = evaluate_plan_grid(
        lognormal_params,
        amdahl_params,
        cold_overhead,
        p_baseline,
        transition_overhead_ms,
    )
    full_plan.to_csv(PLAN_GRID_DIR / "full_plan_evaluation.csv", index=False)
    best_plan.to_csv(PLAN_GRID_DIR / "best_plan_per_budget.csv", index=False)
    pareto.to_csv(PLAN_GRID_DIR / "pareto_frontier.csv", index=False)
    write_final_report(
        out_path=PLAN_GRID_DIR / "r4_final_report.md",
        transition_summary=gap_summary,
        transition_calibration=transition_calibration,
        no_jit_calibration=no_jit_calibration,
        plan_grid=full_plan,
        best_plan=best_plan,
        pareto=pareto,
    )
    print(full_plan[full_plan["slo_ms"] == 20_000].head(20).to_string(index=False))
    print(best_plan.head(20).to_string(index=False))
    print(f"wrote {CALIBRATION_DIR}, {NO_JIT_DIR}, {PLAN_GRID_DIR}")


if __name__ == "__main__":
    main()
