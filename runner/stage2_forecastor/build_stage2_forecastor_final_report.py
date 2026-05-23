import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .analyze_forecast_calibration import refresh_counts, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact Stage 2 Workflow Forecastor final report."
    )
    parser.add_argument(
        "--calibration-after-warmup",
        default=(
            "reports/calibration_lstm_ctx30_h30_scale20_stage_scope/"
            "calibration_after_warmup_by_policy.csv"
        ),
    )
    parser.add_argument(
        "--online-after-warmup",
        default=(
            "reports/online_adaptive_selector_azure_periodic_drift_scaled30_riskbudget/"
            "online_selected_after_warmup_summary.csv"
        ),
    )
    parser.add_argument(
        "--online-latency",
        default=(
            "reports/online_adaptive_selector_azure_periodic_drift_scaled30_riskbudget/"
            "online_selector_latency.csv"
        ),
    )
    parser.add_argument(
        "--online-usage",
        default=(
            "reports/online_adaptive_selector_azure_periodic_drift_scaled30_riskbudget/"
            "online_expert_usage.csv"
        ),
    )
    parser.add_argument(
        "--lightgbm-stage-detail",
        default=(
            "reports/stage2_lightgbm_quantile_azure_periodic_drift_challenge_scaled30/"
            "stage_detail.csv"
        ),
    )
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument(
        "--policies",
        default="p90,p95",
        help="comma-separated policies to include",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/stage2_workflow_forecastor_final",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def policy_set(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def load_calibration_summary(path: Path, policies: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[
        (frame["detail_level"] == "stage")
        & (frame["policy"].isin(policies))
        & (frame["method_family"].isin(["entry-lstm-dag", "per-stage-independent-lstm"]))
        & (frame["calibration_method"].isin(["raw", "rolling_conformal_stage"]))
    ].copy()
    frame["report_group"] = frame["calibration_method"].map(
        {
            "raw": "raw-lstm",
            "rolling_conformal_stage": "rolling-conformal-lstm",
        }
    )
    frame["source_report"] = str(path)
    return frame


def load_online_summary(path: Path, policies: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["policy"].isin(policies)].copy()
    frame["report_group"] = "online-riskbudget-selector"
    frame["source_report"] = str(path)
    if "calibration_method" not in frame.columns:
        frame["calibration_method"] = "online_adaptive_riskbudget"
    return frame


def load_lightgbm_after_warmup(
    path: Path,
    policies: set[str],
    activation_threshold: float,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    frame["detail_level"] = "stage"
    frame["calibration_method"] = "raw"
    frame = refresh_counts(frame, activation_threshold)
    first_fold = int(frame["fold_id"].min())
    warm = frame[(frame["fold_id"] > first_fold) & (frame["policy"].isin(policies))].copy()
    if warm.empty:
        return pd.DataFrame()
    summary = summarize(warm)
    is_conformal = summary["method"].astype(str).str.contains("conformal", case=False)
    summary["report_group"] = "lightgbm-raw"
    summary.loc[is_conformal, "report_group"] = "lightgbm-conformal-variant"
    summary.loc[is_conformal, "calibration_method"] = "method_conformal_variant"
    summary["source_report"] = str(path)
    return summary


def compact_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "report_group",
        "detail_level",
        "workflow_name",
        "method_family",
        "method",
        "calibration_method",
        "policy",
        "nominal_quantile",
        "demand_coverage_rate",
        "allocated_replica_seconds",
        "over_allocation_ratio",
        "allocation_utilization",
        "empirical_quantile_coverage",
        "quantile_calibration_error",
        "pinball_loss_mean",
        "active_ratio",
        "predicted_active_ratio",
        "source_report",
    ]
    out = frame[[col for col in keep if col in frame.columns]].copy()
    return out.sort_values(["policy", "report_group", "method_family", "calibration_method"])


def best_tradeoffs(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in comparison.groupby("policy"):
        group = group.copy()
        group["coverage_shortfall"] = (
            group["nominal_quantile"].astype(float)
            - group["empirical_quantile_coverage"].astype(float)
        ).clip(lower=0.0)
        for label, sort_cols in [
            (
                "closest-calibration",
                ["quantile_calibration_error", "allocated_replica_seconds"],
            ),
            (
                "lowest-cost-above-demand-coverage-0.98",
                ["allocated_replica_seconds", "quantile_calibration_error"],
            ),
            (
                "lowest-overallocation",
                ["over_allocation_ratio", "quantile_calibration_error"],
            ),
        ]:
            subset = group
            if label == "lowest-cost-above-demand-coverage-0.98":
                subset = group[group["demand_coverage_rate"] >= 0.98]
                if subset.empty:
                    subset = group
            chosen = subset.sort_values(sort_cols, ascending=True).iloc[0].copy()
            chosen["tradeoff_label"] = label
            rows.append(chosen)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, max_rows: int = 24) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy()
    try:
        return display.to_markdown(index=False)
    except Exception:
        return display.to_csv(index=False)


def write_report(
    out_dir: Path,
    comparison: pd.DataFrame,
    tradeoffs: pd.DataFrame,
    latency: pd.DataFrame,
    usage: pd.DataFrame,
) -> None:
    main_cols = [
        "report_group",
        "method_family",
        "calibration_method",
        "policy",
        "demand_coverage_rate",
        "allocated_replica_seconds",
        "over_allocation_ratio",
        "empirical_quantile_coverage",
        "quantile_calibration_error",
        "pinball_loss_mean",
    ]
    trade_cols = ["tradeoff_label"] + main_cols
    usage_cols = [
        "policy",
        "source_calibration_method",
        "source_method_family",
        "selected_rows",
        "allocated_replica_windows",
        "under_total",
        "over_total",
    ]
    latency_cols = [
        "policy",
        "mean_selector_decision_ms",
        "p95_selector_decision_ms",
        "max_selector_decision_ms",
    ]
    lines = [
        "# Stage 2 Workflow Forecastor Final Report",
        "",
        "## Scope",
        "",
        "- Workload: Azure-derived periodic/drift challenge trace (default: `visual_qa_flow`), scaled to 5s windows.",
        "- Evaluation summary here uses warm-up-excluded rows where available.",
        "- Compared methods include raw LSTM, rolling conformal LSTM, LightGBM, and online risk-budget selector.",
        "- `empirical_quantile_coverage` is the strict p90/p95 calibration check.",
        "- `demand_coverage_rate` measures total demand covered by integer allocation.",
        "",
        "## Method Comparison",
        "",
        markdown_table(comparison[[col for col in main_cols if col in comparison.columns]], 40),
        "",
        "## Selected Tradeoffs",
        "",
        markdown_table(tradeoffs[[col for col in trade_cols if col in tradeoffs.columns]], 20),
        "",
        "## Online Selector Latency",
        "",
        markdown_table(latency[[col for col in latency_cols if col in latency.columns]], 10),
        "",
        "## Online Expert Usage",
        "",
        markdown_table(usage[[col for col in usage_cols if col in usage.columns]], 30),
        "",
        "## Interpretation",
        "",
        "- Raw LSTM is cheaper but under-calibrated at the strict p90/p95 quantile-hit level.",
        "- Rolling conformal LSTM is the safer calibration layer but costs more replica-seconds.",
        "- The online risk-budget selector uses recent coverage to activate safer experts and reaches a better reliability-cost compromise than always using conformal experts.",
        "- Selector decision latency is millisecond-scale in this offline replay; a deployed daemon should avoid CSV reloads and keep models resident.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = project_root()
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = policy_set(args.policies)

    calibration = load_calibration_summary(
        resolve_path(root, args.calibration_after_warmup), policies
    )
    online = load_online_summary(resolve_path(root, args.online_after_warmup), policies)
    lightgbm = load_lightgbm_after_warmup(
        resolve_path(root, args.lightgbm_stage_detail),
        policies,
        args.activation_threshold,
    )
    comparison = compact_comparison(pd.concat([calibration, lightgbm, online], ignore_index=True))
    tradeoffs = best_tradeoffs(comparison)

    latency_path = resolve_path(root, args.online_latency)
    latency = pd.read_csv(latency_path) if latency_path.exists() else pd.DataFrame()
    usage_path = resolve_path(root, args.online_usage)
    usage = pd.read_csv(usage_path) if usage_path.exists() else pd.DataFrame()

    comparison.to_csv(out_dir / "stage2_method_comparison.csv", index=False)
    tradeoffs.to_csv(out_dir / "stage2_selected_tradeoffs.csv", index=False)
    latency.to_csv(out_dir / "stage2_online_selector_latency.csv", index=False)
    usage.to_csv(out_dir / "stage2_online_expert_usage.csv", index=False)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_after_warmup": str(resolve_path(root, args.calibration_after_warmup)),
        "online_after_warmup": str(resolve_path(root, args.online_after_warmup)),
        "online_latency": str(latency_path),
        "online_usage": str(usage_path),
        "lightgbm_stage_detail": str(resolve_path(root, args.lightgbm_stage_detail)),
        "policies": sorted(policies),
        "activation_threshold": args.activation_threshold,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(out_dir, comparison, tradeoffs, latency, usage)
    print(f"wrote Stage 2 final report to {out_dir}")


if __name__ == "__main__":
    main()

