from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Stage-2 window-count forecasts as count predictions. "
            "This treats forecast_count as the predicted number of invocations "
            "in a window; it is separate from SLO/risk upper-bound selection."
        )
    )
    parser.add_argument("--forecast-detail", required=True)
    parser.add_argument("--workflow", default=None)
    parser.add_argument("--stage-name", default=None)
    parser.add_argument("--selection-scope", choices=["global", "per-stage"], default="global")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_detail(path: Path, workflow: str | None, stage_name: str | None) -> pd.DataFrame:
    rows = pd.read_csv(path)
    if workflow is not None and "workflow_name" in rows.columns:
        rows = rows[rows["workflow_name"].astype(str) == str(workflow)].copy()
    if stage_name is not None and "stage_name" in rows.columns:
        rows = rows[rows["stage_name"].astype(str) == str(stage_name)].copy()
    required = {"method", "policy", "actual_count", "forecast_count"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"forecast detail is missing columns: {sorted(missing)}")
    if "target_window" not in rows.columns:
        if "window" in rows.columns:
            rows = rows.rename(columns={"window": "target_window"})
        else:
            raise ValueError("forecast detail must contain target_window or window")
    if "stage_name" not in rows.columns:
        rows["stage_name"] = "__entry__"
    for column in ["target_window", "actual_count", "forecast_count", "allocated_count"]:
        if column in rows.columns:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.dropna(subset=["target_window", "actual_count", "forecast_count"]).copy()
    rows["target_window"] = rows["target_window"].astype(int)
    return rows


def metric_summary(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    data = rows.copy()
    data["error"] = data["forecast_count"].astype(float) - data["actual_count"].astype(float)
    data["absolute_error"] = data["error"].abs()
    data["squared_error"] = data["error"] ** 2
    if "allocated_count" in data.columns:
        data["covered"] = data["actual_count"].astype(float) <= data["allocated_count"].astype(float)
    else:
        data["covered"] = data["actual_count"].astype(float) <= np.ceil(data["forecast_count"].astype(float))

    summary = (
        data.groupby(group_cols, as_index=False)
        .agg(
            windows=("target_window", "nunique"),
            active_windows=("actual_count", lambda s: int((s.astype(float) > 0).sum())),
            actual_total=("actual_count", "sum"),
            forecast_total=("forecast_count", "sum"),
            mae=("absolute_error", "mean"),
            rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s.astype(float))))),
            bias=("error", "mean"),
            max_actual=("actual_count", "max"),
            max_forecast=("forecast_count", "max"),
            count_hit_rate=("covered", "mean"),
        )
        .reset_index(drop=True)
    )
    summary["over_forecast_ratio"] = (
        (summary["forecast_total"] - summary["actual_total"]).clip(lower=0.0)
        / summary["actual_total"].clip(lower=1.0)
    )
    return summary.sort_values(["mae", "rmse", "over_forecast_ratio"]).reset_index(drop=True)


def select_rows(rows: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scope == "global":
        summary = metric_summary(rows, ["method", "policy"])
        best = summary.iloc[0]
        selected = rows[
            (rows["method"].astype(str) == str(best["method"]))
            & (rows["policy"].astype(str) == str(best["policy"]))
        ].copy()
        selected["selection_scope"] = "global"
        return selected, summary.head(1).copy()

    selected_parts = []
    best_rows = []
    for stage, group in rows.groupby("stage_name"):
        summary = metric_summary(group, ["stage_name", "method", "policy"])
        best = summary.iloc[0]
        selected = group[
            (group["method"].astype(str) == str(best["method"]))
            & (group["policy"].astype(str) == str(best["policy"]))
        ].copy()
        selected["selection_scope"] = "per-stage"
        selected_parts.append(selected)
        best_rows.append(best)
    return pd.concat(selected_parts, ignore_index=True), pd.DataFrame(best_rows)


def main() -> None:
    args = parse_args()
    root = project_root()
    forecast_detail = resolve_path(root, args.forecast_detail)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_detail(forecast_detail, args.workflow, args.stage_name)
    method_summary = metric_summary(rows, ["method", "policy"])
    stage_summary = metric_summary(rows, ["stage_name", "method", "policy"])
    selected, selected_summary = select_rows(rows, args.selection_scope)

    method_summary.to_csv(out_dir / "count_forecast_method_summary.csv", index=False)
    stage_summary.to_csv(out_dir / "count_forecast_stage_summary.csv", index=False)
    selected_summary.to_csv(out_dir / "selected_count_forecast_summary.csv", index=False)
    selected.sort_values(["stage_name", "target_window"]).to_csv(
        out_dir / "selected_count_forecast_detail.csv",
        index=False,
    )
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_detail": str(forecast_detail),
        "workflow": args.workflow,
        "stage_name": args.stage_name,
        "selection_scope": args.selection_scope,
        "note": (
            "Offline diagnostic selection using realized counts; use this to "
            "debug count-prediction behavior, not as online evidence."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")
    print(method_summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()

