#!/usr/bin/env python3
"""Offline validation for the repaired entry-cold lognormal model.

This script intentionally does not change the production risk model.  It
compares the current `compute_plan_risk` model against a repaired validation
model on the real CANON replay, using explicit JIT sync-wait distributions and
the entry-cold "replace warm sync with cold sync" overlap rule.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from runner.stage4_risk.dag_aggregation import LogNormalParams, aggregate_civic_alert
from runner.stage4_risk.plan_risk import BASE_MEMORY_MB, ENTRY_STAGE, PlanInput, compute_plan_risk
from runner.stage4_risk.scaling import scale_stage_for_memory_tier
from runner.stage5_control.multi_slo_planner import STAGES, load_reference_data


RHO = 0.67
OLD_CONTENTION = 1.10
NEW_CONTENTION = 1.0
DEFAULT_N_SAMPLES = 500_000
DEFAULT_SEED = 20260709


@dataclass(frozen=True)
class Inputs:
    workflow_detail: Path
    stage_detail: Path
    plan_csv: Path
    sweep_trace: Path
    out_dir: Path
    premium_slo_ms: float
    free_slo_ms: float
    n_samples: int
    seed: int


def _require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {description}: {path}")


def parse_memory_config(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in str(text).split(","):
        if not item.strip():
            continue
        key, value = item.split(":", 1)
        out[key.strip()] = int(value)
    if sorted(out) != sorted(STAGES):
        raise ValueError(f"memory_config does not cover all stages: {text}")
    return out


def quantiles(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "p50": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "max": math.nan,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def fit_lognormal_from_samples(samples: np.ndarray) -> LogNormalParams:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples) & (samples > 0.0)]
    if samples.size == 0:
        raise ValueError("cannot fit lognormal to empty samples")
    if samples.size == 1:
        return LogNormalParams(mu=math.log(float(samples[0])), sigma=0.0)
    logs = np.log(samples)
    return LogNormalParams(mu=float(np.mean(logs)), sigma=float(np.std(logs, ddof=1)))


def sample_lognormal(dist: LogNormalParams, rng: np.random.Generator, n: int) -> np.ndarray:
    if dist.sigma == 0.0:
        return np.full(n, dist.mean, dtype=float)
    return rng.lognormal(mean=dist.mu, sigma=dist.sigma, size=n)


def empirical_sample(values: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot sample from empty empirical distribution")
    return rng.choice(values, size=n, replace=True)


def load_plans(path: Path) -> dict[str, dict[str, int]]:
    plans = pd.read_csv(path)
    required = {"slo_class", "memory_config"}
    missing = required.difference(plans.columns)
    if missing:
        raise ValueError(f"plan csv missing columns: {sorted(missing)}")
    out: dict[str, dict[str, int]] = {}
    for row in plans.itertuples(index=False):
        cls = str(getattr(row, "slo_class"))
        out[cls] = parse_memory_config(str(getattr(row, "memory_config")))
    for cls in ["premium", "free"]:
        if cls not in out:
            raise ValueError(f"plan csv missing slo_class={cls}")
    return out


def build_plan_input(ref, memory: dict[str, int]) -> PlanInput:
    return PlanInput(
        memory_tier_per_stage=memory,
        entry_prewarm_count=0.0,
        predicted_arrivals=5.0,
        lognormal_params=ref.lognormal_params,
        amdahl_params=ref.amdahl_params,
        cold_overhead_per_stage=ref.cold_overhead_per_stage,
        p_baseline=ref.p_baseline,
    )


def scaled_stage(ref, memory: dict[str, int], stage: str, latency_class: str, contention: float) -> LogNormalParams:
    return scale_stage_for_memory_tier(
        stage_name=stage,
        latency_class=latency_class,
        target_memory_mb=int(memory[stage]),
        base_memory_mb=BASE_MEMORY_MB,
        base_params=ref.lognormal_params[stage][latency_class],
        amdahl_params=ref.amdahl_params,
        splines=ref.warm_splines,
        contention_factor=contention,
    )


def warm_dag_dist(ref, memory: dict[str, int], contention: float) -> LogNormalParams:
    return aggregate_civic_alert(
        {stage: scaled_stage(ref, memory, stage, "warm", contention) for stage in STAGES},
        rho=RHO,
    )


def load_replay(workflow_path: Path, stage_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    wf = pd.read_csv(workflow_path)
    st = pd.read_csv(stage_path)
    wf_required = {"workflow_e2e_ms", "slo_class", "request_id"}
    st_required = {
        "request_id",
        "stage_name",
        "cold_like",
        "dispatch_latency_ms",
        "jit_sync_waited_ms",
        "stage_latency_class",
    }
    wf_missing = wf_required.difference(wf.columns)
    st_missing = st_required.difference(st.columns)
    if wf_missing:
        raise ValueError(f"workflow_detail missing columns: {sorted(wf_missing)}")
    if st_missing:
        raise ValueError(f"stage_detail missing columns: {sorted(st_missing)}")
    st = st.copy()
    st["cold_bool"] = st["cold_like"].astype(str).str.lower().eq("true")
    detect = (
        st[st["stage_name"] == ENTRY_STAGE][["request_id", "cold_bool"]]
        .rename(columns={"cold_bool": "entry_cold"})
        .drop_duplicates("request_id")
    )
    wf = wf.merge(detect, on="request_id", how="left")
    if wf["entry_cold"].isna().any():
        raise ValueError("some workflow rows lack detect_object stage rows")
    wf["entry_state"] = np.where(wf["entry_cold"], "entry_cold", "entry_warm")

    sync = (
        st[st["stage_name"] != ENTRY_STAGE]
        .groupby("request_id", as_index=False)["jit_sync_waited_ms"]
        .sum()
        .rename(columns={"jit_sync_waited_ms": "downstream_sync_wait_ms"})
    )
    wf = wf.merge(sync, on="request_id", how="left")
    wf["downstream_sync_wait_ms"] = wf["downstream_sync_wait_ms"].fillna(0.0)
    wf["workflow_minus_downstream_sync_ms"] = wf["workflow_e2e_ms"] - wf["downstream_sync_wait_ms"]
    return wf, st


def cold_overhead_samples(sweep_trace: Path, stage: str, tier: int) -> np.ndarray:
    df = pd.read_csv(sweep_trace)
    required = {"stage_name", "allocated_memory_mb", "cold_like", "dispatch_latency_ms"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"sweep trace missing columns: {sorted(missing)}")
    sub = df[(df["stage_name"] == stage) & (df["allocated_memory_mb"] == int(tier))]
    if sub.empty:
        raise ValueError(f"no sweep rows for stage={stage} tier={tier}")
    cold = sub[sub["cold_like"].astype(str).str.lower().eq("true")]["dispatch_latency_ms"].dropna().to_numpy(float)
    warm = sub[sub["cold_like"].astype(str).str.lower().eq("false")]["dispatch_latency_ms"].dropna().to_numpy(float)
    if cold.size == 0 or warm.size == 0:
        raise ValueError(f"missing cold/warm sweep rows for stage={stage} tier={tier}")
    warm_median = float(np.median(warm))
    return np.maximum(cold - warm_median, 1.0)


def summarize_actual_by_state(wf: pd.DataFrame, slo_by_class: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cls in ["premium", "free"]:
        cls_df = wf[wf["slo_class"] == cls]
        for state in ["entry_warm", "entry_cold"]:
            sub = cls_df[cls_df["entry_state"] == state]
            q = quantiles(sub["workflow_e2e_ms"])
            rows.append(
                {
                    "slo_class": cls,
                    "entry_state": state,
                    "actual_n": q["n"],
                    "actual_mean_ms": q["mean"],
                    "actual_p50_ms": q["p50"],
                    "actual_p95_ms": q["p95"],
                    "actual_p99_ms": q["p99"],
                    "actual_violation": float(sub["workflow_e2e_ms"].gt(slo_by_class[cls]).mean()) if len(sub) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def model_rows(inputs: Inputs, wf: pd.DataFrame, st: pd.DataFrame, plans: dict[str, dict[str, int]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ref = load_reference_data()
    rng = np.random.default_rng(inputs.seed)
    slo_by_class = {"premium": inputs.premium_slo_ms, "free": inputs.free_slo_ms}
    actual = summarize_actual_by_state(wf, slo_by_class)
    rows: list[dict[str, object]] = []
    total_rows: list[dict[str, object]] = []
    selfcheck_rows: list[dict[str, object]] = []
    overhead_rows: list[dict[str, object]] = []

    for cls in ["premium", "free"]:
        memory = plans[cls]
        slo = slo_by_class[cls]
        plan = build_plan_input(ref, memory)
        old = compute_plan_risk(plan, slo_ms=slo, rho=RHO, contention_factor=OLD_CONTENTION)
        no_sync_dist = warm_dag_dist(ref, memory, contention=NEW_CONTENTION)

        cls_wf = wf[wf["slo_class"] == cls]
        warm_wf = cls_wf[cls_wf["entry_state"] == "entry_warm"]
        cold_wf = cls_wf[cls_wf["entry_state"] == "entry_cold"]
        p_entry_actual = float(len(cold_wf) / len(cls_wf)) if len(cls_wf) else math.nan
        sync_warm = warm_wf["downstream_sync_wait_ms"].dropna().to_numpy(float)
        sync_cold = cold_wf["downstream_sync_wait_ms"].dropna().to_numpy(float)
        overhead = cold_overhead_samples(inputs.sweep_trace, ENTRY_STAGE, memory[ENTRY_STAGE])
        overhead_stats = quantiles(overhead)
        overhead_ln = fit_lognormal_from_samples(overhead)
        overhead_rows.append(
            {
                "slo_class": cls,
                "entry_tier": memory[ENTRY_STAGE],
                "stage": ENTRY_STAGE,
                "n": overhead_stats["n"],
                "mean_ms": overhead_stats["mean"],
                "p50_ms": overhead_stats["p50"],
                "p95_ms": overhead_stats["p95"],
                "p99_ms": overhead_stats["p99"],
                "max_ms": overhead_stats["max"],
                "lognormal_sigma": overhead_ln.sigma,
            }
        )

        n = inputs.n_samples
        # Shared no-sync execution samples for the repaired validation model.
        no_sync_samples = sample_lognormal(no_sync_dist, rng, n)
        sync_warm_samples = empirical_sample(sync_warm, rng, n)
        sync_cold_samples = empirical_sample(sync_cold, rng, n)
        overhead_samples = empirical_sample(overhead, rng, n)

        new_warm_samples = no_sync_samples + sync_warm_samples
        new_cold_samples = no_sync_samples + overhead_samples + sync_cold_samples

        models = {
            ("entry_warm", "old_current_api"): sample_lognormal(old.e2e_warm_params, rng, n),
            ("entry_cold", "old_current_api"): sample_lognormal(old.e2e_cold_entry_params, rng, n),
            ("entry_warm", "new_no_contention_plus_empirical_warm_sync"): new_warm_samples,
            ("entry_cold", "new_no_contention_plus_overhead_plus_cold_sync"): new_cold_samples,
        }
        for (state, model_name), samples in models.items():
            q = quantiles(samples)
            actual_match = actual[(actual["slo_class"] == cls) & (actual["entry_state"] == state)].iloc[0]
            rows.append(
                {
                    "slo_class": cls,
                    "slo_ms": slo,
                    "entry_state": state,
                    "model": model_name,
                    "actual_n": int(actual_match["actual_n"]),
                    "actual_p95_ms": float(actual_match["actual_p95_ms"]),
                    "actual_p99_ms": float(actual_match["actual_p99_ms"]),
                    "model_mean_ms": q["mean"],
                    "model_p50_ms": q["p50"],
                    "model_p95_ms": q["p95"],
                    "model_p99_ms": q["p99"],
                    "p95_error_ms": q["p95"] - float(actual_match["actual_p95_ms"]),
                    "p99_error_ms": q["p99"] - float(actual_match["actual_p99_ms"]),
                    "model_conditional_violation": float(np.mean(samples > slo)),
                    "actual_conditional_violation": float(actual_match["actual_violation"]),
                }
            )

        # Fork §22 style self-check: actual no-sync base + overhead + cold-sync.
        actual_warm_no_sync = warm_wf["workflow_minus_downstream_sync_ms"].dropna().to_numpy(float)
        selfcheck_old = empirical_sample(actual_warm_no_sync, rng, n) + overhead_samples + sync_warm_samples
        selfcheck_new = empirical_sample(actual_warm_no_sync, rng, n) + overhead_samples + sync_cold_samples
        cold_actual = actual[(actual["slo_class"] == cls) & (actual["entry_state"] == "entry_cold")].iloc[0]
        for model_name, samples in {
            "selfcheck_keep_warm_sync": selfcheck_old,
            "selfcheck_replace_with_cold_sync": selfcheck_new,
        }.items():
            q = quantiles(samples)
            selfcheck_rows.append(
                {
                    "slo_class": cls,
                    "slo_ms": slo,
                    "model": model_name,
                    "actual_cold_p95_ms": float(cold_actual["actual_p95_ms"]),
                    "model_p95_ms": q["p95"],
                    "p95_error_ms": q["p95"] - float(cold_actual["actual_p95_ms"]),
                    "model_mean_ms": q["mean"],
                    "actual_cold_mean_ms": float(cold_actual["actual_mean_ms"]),
                }
            )

        old_total_api = old.p_violation_total
        old_total_actual_pentry = (1.0 - p_entry_actual) * old.e2e_warm_params.survival(slo) + p_entry_actual * old.e2e_cold_entry_params.survival(slo)
        new_warm_survival = float(np.mean(new_warm_samples > slo))
        new_cold_survival = float(np.mean(new_cold_samples > slo))
        new_total_actual_pentry = (1.0 - p_entry_actual) * new_warm_survival + p_entry_actual * new_cold_survival
        real_total = float(cls_wf["workflow_e2e_ms"].gt(slo).mean()) if len(cls_wf) else math.nan
        total_rows.extend(
            [
                {
                    "slo_class": cls,
                    "slo_ms": slo,
                    "model": "old_current_api_pbaseline",
                    "p_entry_cold_used": old.p_entry_cold,
                    "warm_survival": old.e2e_warm_params.survival(slo),
                    "cold_survival": old.e2e_cold_entry_params.survival(slo),
                    "total_violation": old_total_api,
                    "real_total_violation": real_total,
                },
                {
                    "slo_class": cls,
                    "slo_ms": slo,
                    "model": "old_current_api_actual_pentry",
                    "p_entry_cold_used": p_entry_actual,
                    "warm_survival": old.e2e_warm_params.survival(slo),
                    "cold_survival": old.e2e_cold_entry_params.survival(slo),
                    "total_violation": old_total_actual_pentry,
                    "real_total_violation": real_total,
                },
                {
                    "slo_class": cls,
                    "slo_ms": slo,
                    "model": "new_actual_pentry",
                    "p_entry_cold_used": p_entry_actual,
                    "warm_survival": new_warm_survival,
                    "cold_survival": new_cold_survival,
                    "total_violation": new_total_actual_pentry,
                    "real_total_violation": real_total,
                },
            ]
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(total_rows),
        pd.DataFrame(overhead_rows),
        pd.DataFrame(selfcheck_rows),
    )


def stage_warm_p95(ref, st: pd.DataFrame, plans: dict[str, dict[str, int]], wf: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cls in ["premium", "free"]:
        request_ids = set(wf[wf["slo_class"] == cls]["request_id"])
        memory = plans[cls]
        for stage in STAGES:
            sub = st[
                (st["stage_name"] == stage)
                & (st["request_id"].isin(request_ids))
                & (~st["cold_bool"])
            ]
            actual = quantiles(sub["dispatch_latency_ms"])
            old_dist = scaled_stage(ref, memory, stage, "warm", OLD_CONTENTION)
            new_dist = scaled_stage(ref, memory, stage, "warm", NEW_CONTENTION)
            rows.append(
                {
                    "slo_class": cls,
                    "stage_name": stage,
                    "tier": memory[stage],
                    "actual_n": actual["n"],
                    "actual_warm_p95_ms": actual["p95"],
                    "old_model_p95_ms": old_dist.quantile(0.95),
                    "new_execution_model_p95_ms": new_dist.quantile(0.95),
                    "old_p95_error_ms": old_dist.quantile(0.95) - actual["p95"],
                    "new_p95_error_ms": new_dist.quantile(0.95) - actual["p95"],
                }
            )
    return pd.DataFrame(rows)


def write_report(
    inputs: Inputs,
    old_new: pd.DataFrame,
    total: pd.DataFrame,
    overhead: pd.DataFrame,
    selfcheck: pd.DataFrame,
    stage_p95: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Step 1 Model Validation: old vs repaired lognormal model")
    lines.append("")
    lines.append("This is an offline validation only. It does not modify `plan_risk.py` or any runner logic.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- workflow detail: `{inputs.workflow_detail}`")
    lines.append(f"- stage detail: `{inputs.stage_detail}`")
    lines.append(f"- plan csv: `{inputs.plan_csv}`")
    lines.append(f"- sweep trace for cold overhead: `{inputs.sweep_trace}`")
    lines.append(f"- SLO: premium={inputs.premium_slo_ms:.0f} ms, free={inputs.free_slo_ms:.0f} ms")
    lines.append(f"- MC samples for empirical convolution: {inputs.n_samples}")
    lines.append("")
    lines.append("## Conditional E2E P95/P99")
    lines.append("")
    show = old_new[
        [
            "slo_class",
            "entry_state",
            "model",
            "actual_n",
            "actual_p95_ms",
            "model_p95_ms",
            "p95_error_ms",
            "actual_p99_ms",
            "model_p99_ms",
            "p99_error_ms",
            "model_conditional_violation",
            "actual_conditional_violation",
        ]
    ].copy()
    lines.append(show.round(3).to_markdown(index=False))
    lines.append("")
    lines.append("## Total violation at class SLO")
    lines.append("")
    lines.append(total.round(5).to_markdown(index=False))
    lines.append("")
    lines.append("## Per-stage warm P95")
    lines.append("")
    lines.append(stage_p95.round(3).to_markdown(index=False))
    lines.append("")
    lines.append("## Entry cold overhead from sweep")
    lines.append("")
    lines.append(overhead.round(3).to_markdown(index=False))
    lines.append("")
    lines.append("## Sync-overlap self-check")
    lines.append("")
    lines.append(selfcheck.round(3).to_markdown(index=False))
    lines.append("")

    # Automated checks and conclusion.
    warm_new = old_new[
        (old_new["entry_state"] == "entry_warm")
        & (old_new["model"] == "new_no_contention_plus_empirical_warm_sync")
    ].copy()
    warm_new["abs_pct_error"] = (warm_new["p95_error_ms"].abs() / warm_new["actual_p95_ms"]) * 100.0
    max_warm_pct = float(warm_new["abs_pct_error"].max())
    prem_self = selfcheck[
        (selfcheck["slo_class"] == "premium")
        & (selfcheck["model"] == "selfcheck_replace_with_cold_sync")
    ].iloc[0]
    prem_self_err = float(prem_self["p95_error_ms"])
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- New warm model max conditional p95 error: {max_warm_pct:.2f}%.")
    if max_warm_pct <= 5.0:
        lines.append("- Warm E2E passes the <~5% check: contention=1.0 + explicit sync_wait is sufficient on this replay.")
    else:
        lines.append("- Warm E2E fails the <~5% check: execution still needs residual contention beyond explicit sync_wait.")
    lines.append(
        f"- Premium sync-overlap self-check p95 error: {prem_self_err:.1f} ms "
        "(target is roughly +50 ms from the fork prototype)."
    )
    lines.append(
        "- The deployable repaired validation model and the fork self-check are both reported because the self-check uses actual no-sync replay base to isolate the sync-overlap mechanism, while the deployable model uses the spline/lognormal DAG execution distribution."
    )
    lines.append("")
    (inputs.out_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(inputs: Inputs) -> None:
    for path, desc in [
        (inputs.workflow_detail, "workflow detail"),
        (inputs.stage_detail, "stage detail"),
        (inputs.plan_csv, "plan csv"),
        (inputs.sweep_trace, "sweep trace"),
    ]:
        _require(path, desc)
    inputs.out_dir.mkdir(parents=True, exist_ok=True)
    wf, st = load_replay(inputs.workflow_detail, inputs.stage_detail)
    plans = load_plans(inputs.plan_csv)
    ref = load_reference_data()
    old_new, total, overhead, selfcheck = model_rows(inputs, wf, st, plans)
    stage_p95 = stage_warm_p95(ref, st, plans, wf)
    old_new.to_csv(inputs.out_dir / "old_vs_new_p95.csv", index=False)
    total.to_csv(inputs.out_dir / "total_violation.csv", index=False)
    overhead.to_csv(inputs.out_dir / "cold_overhead_summary.csv", index=False)
    selfcheck.to_csv(inputs.out_dir / "sync_overlap_selfcheck.csv", index=False)
    stage_p95.to_csv(inputs.out_dir / "stage_warm_p95_actual_vs_model.csv", index=False)
    write_report(inputs, old_new, total, overhead, selfcheck, stage_p95)

    print("OLD VS NEW CONDITIONAL P95/P99")
    print(old_new.round(3).to_string(index=False))
    print("\nTOTAL VIOLATION")
    print(total.round(5).to_string(index=False))
    print("\nPER-STAGE WARM P95")
    print(stage_p95.round(3).to_string(index=False))
    print("\nCOLD OVERHEAD SUMMARY")
    print(overhead.round(3).to_string(index=False))
    print("\nSYNC-OVERLAP SELFCHECK")
    print(selfcheck.round(3).to_string(index=False))
    print(f"\nWrote {inputs.out_dir / 'validation_report.md'}")


def parse_args() -> Inputs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow-detail",
        type=Path,
        default=Path("reports/eval_canon_1824_oracle_static/full/workflow_detail.csv"),
    )
    parser.add_argument(
        "--stage-detail",
        type=Path,
        default=Path("reports/eval_canon_1824_oracle_static/full/stage_detail.csv"),
    )
    parser.add_argument(
        "--plan-csv",
        type=Path,
        default=Path("reports/eval_canon_1824_oracle_static/plans/risk_price_1824.csv"),
    )
    parser.add_argument(
        "--sweep-trace",
        type=Path,
        default=Path("reports/sweep_14tier_ceil_model/merged_trace.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/step1_model_validation"),
    )
    parser.add_argument("--slo-premium-ms", type=float, default=18000.0)
    parser.add_argument("--slo-free-ms", type=float, default=22000.0)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    return Inputs(
        workflow_detail=args.workflow_detail,
        stage_detail=args.stage_detail,
        plan_csv=args.plan_csv,
        sweep_trace=args.sweep_trace,
        out_dir=args.out_dir,
        premium_slo_ms=args.slo_premium_ms,
        free_slo_ms=args.slo_free_ms,
        n_samples=args.n_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    run(parse_args())
