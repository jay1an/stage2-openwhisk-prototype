import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "fold_id",
    "workflow_name",
    "method_family",
    "method",
    "policy",
    "nominal_quantile",
    "stage_name",
    "target_window",
    "actual_count",
    "forecast_count",
    "window_ms",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze quantile calibration for rolling forecast detail CSVs and "
            "apply no-lookahead rolling conformal post-calibration."
        )
    )
    parser.add_argument(
        "--detail",
        nargs="+",
        required=True,
        help="One or more detail CSVs, e.g. entry_detail.csv and stage_detail.csv.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument(
        "--conformal-scope",
        choices=["global", "method", "stage"],
        default="stage",
        help=(
            "Scope used to collect previous-fold residuals. stage is strictest: "
            "level + method + policy + stage_name."
        ),
    )
    parser.add_argument("--min-calibration-rows", type=int, default=30)
    parser.add_argument(
        "--allow-negative-shift",
        action="store_true",
        help="Allow conformal correction to reduce forecasts; default is safety-only.",
    )
    parser.add_argument(
        "--active-gate-thresholds",
        default="auto,0.1,0.2,0.5",
        help=(
            "Comma-separated thresholds for the diagnostic P(active) gate. "
            "Use 'auto' for the hurdle quantile threshold 1 - nominal_quantile."
        ),
    )
    parser.add_argument(
        "--risk-bins",
        type=int,
        default=8,
        help="Number of forecast-count bins for reliability/risk tables.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def infer_level(frame: pd.DataFrame, source_name: str) -> str:
    if "entry" in source_name.lower():
        return "entry"
    if "stage" in source_name.lower():
        return "stage"
    stages = set(frame["stage_name"].astype(str).unique())
    return "entry" if stages == {"__entry__"} else "stage"


def load_detail(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        frame = frame.copy()
        frame["source_file"] = str(path)
        frame["detail_level"] = infer_level(frame, path.name)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["fold_id"] = out["fold_id"].astype(int)
    out["target_window"] = out["target_window"].astype(int)
    out["actual_count"] = out["actual_count"].astype(float)
    out["forecast_count"] = out["forecast_count"].astype(float)
    out["nominal_quantile"] = out["nominal_quantile"].astype(float)
    out["window_ms"] = out["window_ms"].astype(int)
    return out


def alloc_count(value: float, activation_threshold: float) -> int:
    value = max(0.0, float(value))
    if value <= activation_threshold:
        return 0
    return int(math.ceil(value))


def pinball_loss(actual: float, forecast: float, quantile: float) -> float:
    error = float(actual) - float(forecast)
    return float(max(quantile * error, (quantile - 1.0) * error))


def refresh_counts(frame: pd.DataFrame, activation_threshold: float) -> pd.DataFrame:
    out = frame.copy()
    out["forecast_count"] = out["forecast_count"].clip(lower=0.0)
    out["allocated_count"] = out["forecast_count"].map(
        lambda value: alloc_count(value, activation_threshold)
    )
    out["under_count"] = np.maximum(0.0, out["actual_count"] - out["allocated_count"])
    out["over_count"] = np.maximum(0.0, out["allocated_count"] - out["actual_count"])
    out["quantile_hit"] = out["actual_count"] <= out["forecast_count"]
    out["pinball_loss"] = [
        pinball_loss(actual, forecast, quantile)
        for actual, forecast, quantile in zip(
            out["actual_count"], out["forecast_count"], out["nominal_quantile"]
        )
    ]
    return out


def conformal_group_columns(scope: str) -> list[str]:
    if scope == "global":
        return ["detail_level", "policy"]
    if scope == "method":
        return ["detail_level", "method_family", "method", "policy"]
    return ["detail_level", "method_family", "method", "policy", "stage_name"]


def rolling_conformal(
    raw: pd.DataFrame,
    activation_threshold: float,
    scope: str,
    min_rows: int,
    allow_negative_shift: bool,
) -> pd.DataFrame:
    group_cols = conformal_group_columns(scope)
    outputs = []
    for _, group in raw.groupby(group_cols, dropna=False):
        group = group.sort_values(["fold_id", "target_window"]).copy()
        calibrated_parts = []
        for fold_id in sorted(group["fold_id"].unique()):
            current = group[group["fold_id"] == fold_id].copy()
            history = group[group["fold_id"] < fold_id]
            if len(history) >= min_rows:
                residuals = history["actual_count"].to_numpy(dtype=float) - history[
                    "forecast_count"
                ].to_numpy(dtype=float)
                nominal = float(current["nominal_quantile"].iloc[0])
                shift = float(np.quantile(residuals, nominal))
                if not allow_negative_shift:
                    shift = max(0.0, shift)
            else:
                shift = 0.0
            current["conformal_shift"] = shift
            current["forecast_count"] = current["forecast_count"] + shift
            calibrated_parts.append(current)
        outputs.append(pd.concat(calibrated_parts, ignore_index=True))
    out = pd.concat(outputs, ignore_index=True) if outputs else raw.copy()
    out["calibration_method"] = f"rolling_conformal_{scope}"
    return refresh_counts(out, activation_threshold)


def parse_active_thresholds(text: str) -> list[str | float]:
    thresholds: list[str | float] = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item == "auto":
            thresholds.append("auto")
        else:
            thresholds.append(float(item))
    return thresholds


def add_rolling_p_active(frame: pd.DataFrame, min_rows: int = 10) -> pd.DataFrame:
    keys = ["detail_level", "method_family", "method", "stage_name"]
    outputs = []
    dedup_cols = keys + ["fold_id", "target_window"]
    base = frame.drop_duplicates(dedup_cols).copy()
    base["actual_active"] = (base["actual_count"] > 0).astype(float)
    for _, group in base.groupby(keys, dropna=False):
        group = group.sort_values(["fold_id", "target_window"]).copy()
        p_rows = []
        for fold_id in sorted(group["fold_id"].unique()):
            current = group[group["fold_id"] == fold_id].copy()
            history = group[group["fold_id"] < fold_id]
            if len(history) >= min_rows:
                probability = float(history["actual_active"].mean())
            else:
                # Safety-first default: do not gate until we have history.
                probability = 1.0
            current["rolling_p_active"] = probability
            p_rows.append(current[dedup_cols + ["rolling_p_active"]])
        outputs.append(pd.concat(p_rows, ignore_index=True))
    p_active = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    out = frame.merge(p_active, on=dedup_cols, how="left")
    out["rolling_p_active"] = out["rolling_p_active"].fillna(1.0)
    return out


def apply_active_gate(
    raw: pd.DataFrame,
    threshold: str | float,
    activation_threshold: float,
) -> pd.DataFrame:
    out = add_rolling_p_active(raw)
    if threshold == "auto":
        gate_threshold = 1.0 - out["nominal_quantile"].astype(float)
        label = "raw_active_gate_auto"
    else:
        gate_threshold = float(threshold)
        label = f"raw_active_gate_t{float(threshold):.2f}".replace(".", "p")
    gated = out["rolling_p_active"] <= gate_threshold
    out["active_gate_threshold"] = gate_threshold
    out["active_gate_applied"] = gated
    out.loc[gated, "forecast_count"] = 0.0
    out["calibration_method"] = label
    return refresh_counts(out, activation_threshold)


def summarize(frame: pd.DataFrame, extra_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = [
        "detail_level",
        "calibration_method",
        "workflow_name",
        "method_family",
        "method",
        "policy",
        "nominal_quantile",
    ]
    if extra_cols:
        group_cols.extend(extra_cols)
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        actual = group["actual_count"].astype(float)
        forecast = group["forecast_count"].astype(float)
        allocated = group["allocated_count"].astype(float)
        active_mask = actual > 0
        predicted_active_mask = allocated > 0
        tp = int((active_mask & predicted_active_mask).sum())
        fp = int((~active_mask & predicted_active_mask).sum())
        fn = int((active_mask & ~predicted_active_mask).sum())
        tn = int((~active_mask & ~predicted_active_mask).sum())
        actual_total = float(actual.sum())
        allocated_total = float(allocated.sum())
        under_total = float(group["under_count"].sum())
        over_total = float(group["over_count"].sum())
        empirical = float(group["quantile_hit"].mean()) if len(group) else 1.0
        nominal = float(group["nominal_quantile"].iloc[0])
        row = {
            **dict(zip(group_cols, keys)),
            "folds": int(group["fold_id"].nunique()),
            "forecast_rows": int(len(group)),
            "active_rows": int(active_mask.sum()),
            "active_ratio": float(active_mask.mean()) if len(group) else 0.0,
            "predicted_active_rows": int(predicted_active_mask.sum()),
            "predicted_active_ratio": (
                float(predicted_active_mask.mean()) if len(group) else 0.0
            ),
            "true_positive_active_rows": tp,
            "false_positive_active_rows": fp,
            "false_negative_active_rows": fn,
            "true_negative_active_rows": tn,
            "active_precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0,
            "active_recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0,
            "active_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0,
            "actual_total": int(actual_total),
            "allocated_replica_windows": int(allocated_total),
            "allocated_replica_seconds": float(
                allocated_total * float(group["window_ms"].iloc[0]) / 1000.0
            ),
            "under_total": int(under_total),
            "over_total": int(over_total),
            "demand_coverage_rate": (
                float(1.0 - under_total / actual_total) if actual_total > 0 else 1.0
            ),
            "allocation_utilization": (
                float((actual_total - under_total) / allocated_total)
                if allocated_total > 0
                else 0.0
            ),
            "over_allocation_ratio": (
                float(over_total / allocated_total) if allocated_total > 0 else 0.0
            ),
            "empirical_quantile_coverage": empirical,
            "signed_quantile_calibration_error": empirical - nominal,
            "quantile_calibration_error": abs(empirical - nominal),
            "brier_score_quantile_hit": float(
                np.mean((group["quantile_hit"].astype(float) - nominal) ** 2)
            ),
            "pinball_loss_mean": float(group["pinball_loss"].mean()),
            "mae": float(np.mean(np.abs(actual - forecast))),
            "rmse": float(np.sqrt(np.mean((actual - forecast) ** 2))),
            "mean_forecast_count": float(forecast.mean()),
            "max_actual": int(actual.max()) if len(actual) else 0,
            "max_allocated": int(allocated.max()) if len(allocated) else 0,
        }
        if "conformal_shift" in group.columns:
            row["mean_conformal_shift"] = float(group["conformal_shift"].mean())
            row["max_conformal_shift"] = float(group["conformal_shift"].max())
        if "rolling_p_active" in group.columns:
            row["mean_rolling_p_active"] = float(group["rolling_p_active"].mean())
            row["active_gate_rows"] = int(group.get("active_gate_applied", False).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_overview(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "detail_level",
        "calibration_method",
        "workflow_name",
        "method_family",
        "method",
    ]
    for keys, group in summary.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        weights = group["forecast_rows"].astype(float)
        abs_error = group["quantile_calibration_error"].astype(float)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "policies": ",".join(group["policy"].astype(str)),
                "weighted_quantile_ece": float(np.average(abs_error, weights=weights)),
                "max_quantile_calibration_error": float(abs_error.max()),
                "mean_demand_coverage_rate": float(
                    np.average(group["demand_coverage_rate"], weights=weights)
                ),
                "total_allocated_replica_seconds": float(
                    group["allocated_replica_seconds"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def risk_bins(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    rows = []
    group_cols = [
        "detail_level",
        "calibration_method",
        "workflow_name",
        "method_family",
        "method",
        "policy",
        "nominal_quantile",
    ]
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = group.copy()
        unique = group["forecast_count"].nunique()
        if unique <= 1:
            group["_bin"] = 0
        else:
            q = min(int(bins), int(unique))
            try:
                group["_bin"] = pd.qcut(
                    group["forecast_count"], q=q, labels=False, duplicates="drop"
                )
            except ValueError:
                group["_bin"] = 0
        for bin_id, part in group.groupby("_bin", dropna=False):
            nominal = float(part["nominal_quantile"].iloc[0])
            empirical = float(part["quantile_hit"].mean()) if len(part) else 1.0
            rows.append(
                {
                    **dict(zip(group_cols, keys)),
                    "bin_id": int(bin_id) if pd.notna(bin_id) else 0,
                    "rows": int(len(part)),
                    "forecast_min": float(part["forecast_count"].min()),
                    "forecast_max": float(part["forecast_count"].max()),
                    "forecast_mean": float(part["forecast_count"].mean()),
                    "actual_mean": float(part["actual_count"].mean()),
                    "allocated_mean": float(part["allocated_count"].mean()),
                    "empirical_quantile_coverage": empirical,
                    "empirical_exceedance_rate": float(1.0 - empirical),
                    "nominal_exceedance_rate": float(1.0 - nominal),
                    "signed_coverage_error": empirical - nominal,
                    "under_total": int(part["under_count"].sum()),
                    "over_total": int(part["over_count"].sum()),
                }
            )
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy()
    try:
        return display.to_markdown(index=False)
    except Exception:
        return display.to_csv(index=False)


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    overview: pd.DataFrame,
    warm_summary: pd.DataFrame,
    warm_overview: pd.DataFrame,
) -> None:
    focus_cols = [
        "detail_level",
        "calibration_method",
        "method_family",
        "policy",
        "demand_coverage_rate",
        "allocated_replica_seconds",
        "over_allocation_ratio",
        "empirical_quantile_coverage",
        "quantile_calibration_error",
        "active_ratio",
        "predicted_active_ratio",
    ]
    focus = summary[
        [col for col in focus_cols if col in summary.columns]
    ].sort_values(["detail_level", "method_family", "policy", "calibration_method"])
    warm_focus = warm_summary[
        [col for col in focus_cols if col in warm_summary.columns]
    ].sort_values(["detail_level", "method_family", "policy", "calibration_method"])

    lines = [
        "# Rolling Calibration Analysis",
        "",
        "## Scope",
        "",
        "- `raw`: original quantile forecasts from the detail CSVs.",
        "- `rolling_conformal_*`: post-hoc conformal shift using only previous folds.",
        "- `raw_active_gate_*`: diagnostic P(active) gate using previous-fold active frequency.",
        "- Quantile calibration checks whether `actual_count <= forecast_count` at p50/p90/p95.",
        "- Allocation metrics use integer `allocated_count` after activation threshold and ceiling.",
        "",
        "## Calibration Overview",
        "",
        md_table(overview.sort_values(["detail_level", "weighted_quantile_ece"])),
        "",
        "## Calibration Overview After Warm-Up",
        "",
        "The first rolling fold has no previous-fold residuals, so it is reported separately here.",
        "",
        md_table(warm_overview.sort_values(["detail_level", "weighted_quantile_ece"])),
        "",
        "## Policy Summary",
        "",
        md_table(focus, max_rows=24),
        "",
        "## Policy Summary After Warm-Up",
        "",
        md_table(warm_focus, max_rows=24),
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = project_root()
    detail_paths = [resolve_path(root, value) for value in args.detail]
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_detail(detail_paths)
    raw = refresh_counts(raw, args.activation_threshold)
    raw["calibration_method"] = "raw"
    raw["conformal_shift"] = 0.0

    frames = [raw]
    frames.append(
        rolling_conformal(
            raw,
            activation_threshold=args.activation_threshold,
            scope=args.conformal_scope,
            min_rows=max(1, args.min_calibration_rows),
            allow_negative_shift=args.allow_negative_shift,
        )
    )

    for threshold in parse_active_thresholds(args.active_gate_thresholds):
        frames.append(
            apply_active_gate(
                raw,
                threshold=threshold,
                activation_threshold=args.activation_threshold,
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    summary = summarize(combined)
    by_fold = summarize(combined, extra_cols=["fold_id"])
    by_stage = summarize(combined, extra_cols=["stage_name"])
    overview = calibration_overview(summary)
    bins = risk_bins(combined, bins=max(1, args.risk_bins))
    first_fold = int(combined["fold_id"].min())
    warm_combined = combined[combined["fold_id"] > first_fold].copy()
    warm_summary = summarize(warm_combined) if not warm_combined.empty else pd.DataFrame()
    warm_overview = (
        calibration_overview(warm_summary) if not warm_summary.empty else pd.DataFrame()
    )

    combined.to_csv(out_dir / "calibrated_detail.csv", index=False)
    summary.to_csv(out_dir / "calibration_by_policy.csv", index=False)
    by_fold.to_csv(out_dir / "calibration_by_fold.csv", index=False)
    by_stage.to_csv(out_dir / "calibration_by_stage.csv", index=False)
    overview.to_csv(out_dir / "calibration_overview.csv", index=False)
    bins.to_csv(out_dir / "risk_bin_table.csv", index=False)
    warm_summary.to_csv(out_dir / "calibration_after_warmup_by_policy.csv", index=False)
    warm_overview.to_csv(out_dir / "calibration_after_warmup_overview.csv", index=False)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "detail_files": [str(path) for path in detail_paths],
        "out_dir": str(out_dir),
        "activation_threshold": args.activation_threshold,
        "conformal_scope": args.conformal_scope,
        "min_calibration_rows": args.min_calibration_rows,
        "allow_negative_shift": bool(args.allow_negative_shift),
        "active_gate_thresholds": args.active_gate_thresholds,
        "risk_bins": args.risk_bins,
        "warmup_excluded_fold_id": first_fold,
        "notes": [
            "Rolling conformal shifts use only folds strictly earlier than the evaluated fold.",
            "The first fold is uncalibrated for previous-fold conformal analysis and is summarized separately as warm-up.",
            "The active gate is diagnostic; final P(active) should be a learned classifier or two-head model.",
        ],
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(out_dir, summary, overview, warm_summary, warm_overview)
    print(f"wrote calibration analysis to {out_dir}")


if __name__ == "__main__":
    main()

