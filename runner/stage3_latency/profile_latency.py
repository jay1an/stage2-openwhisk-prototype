import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "entry_ts_ms",
    "dispatch_start_ms",
    "dispatch_end_ms",
    "dispatch_latency_ms",
    "action_start_ns",
    "action_end_ns",
    "action_duration_ms",
    "platform_overhead_ms",
]

LATENCY_COLUMNS = [
    "dispatch_latency_ms",
    "platform_overhead_ms",
    "action_duration_ms",
    "stage_start_offset_ms",
    "stage_completion_offset_ms",
]

QUANTILES = {
    "p50": 0.50,
    "p90": 0.90,
    "p95": 0.95,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a coarse Stage-3 latency profile from workflow trace CSVs. "
            "The profile keeps warm/cold-like paths separate and exports empirical "
            "sample pools for later SLO-risk Monte Carlo."
        )
    )
    parser.add_argument(
        "--traces",
        nargs="+",
        required=True,
        help="one or more trace CSV files, relative to the project root unless absolute",
    )
    parser.add_argument(
        "--trace-labels",
        nargs="*",
        default=None,
        help="optional labels for trace groups; length must match --traces when provided",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/stage3_latency_profile",
        help="directory for profile tables, plots, and metadata",
    )
    parser.add_argument(
        "--cold-overhead-threshold-ms",
        type=float,
        default=500.0,
        help="fallback threshold when cold_like is missing",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="include non-ok rows if present; by default only ok stage rows are profiled",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def normalize_bool(value) -> object:
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return np.nan


def classify_latency(row: pd.Series, cold_threshold_ms: float) -> str:
    cold_like = normalize_bool(row.get("cold_like", np.nan))
    if cold_like is True:
        return "cold_like"
    if cold_like is False:
        return "warm"

    overhead = row.get("platform_overhead_ms", np.nan)
    if pd.notna(overhead):
        return "cold_like" if float(overhead) >= cold_threshold_ms else "warm"
    return "unknown"


def load_trace(
    root: Path,
    trace_path: str,
    trace_label: str,
    cold_threshold_ms: float,
    include_failed: bool,
) -> pd.DataFrame:
    resolved = resolve_path(root, trace_path)
    frame = pd.read_csv(resolved)
    frame["trace_file"] = str(resolved)
    frame["trace_label"] = trace_label

    for col in NUMERIC_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "dispatch_latency_ms" not in frame.columns or frame["dispatch_latency_ms"].isna().all():
        frame["dispatch_latency_ms"] = frame["dispatch_end_ms"] - frame["dispatch_start_ms"]
    if "platform_overhead_ms" not in frame.columns or frame["platform_overhead_ms"].isna().all():
        frame["platform_overhead_ms"] = frame["dispatch_latency_ms"] - frame["action_duration_ms"]

    stage_rows = frame[frame["stage_name"] != "__entry__"].copy()
    if not include_failed and "status" in stage_rows.columns:
        stage_rows = stage_rows[stage_rows["status"].fillna("ok") == "ok"].copy()

    stage_rows["latency_class"] = stage_rows.apply(
        lambda row: classify_latency(row, cold_threshold_ms), axis=1
    )
    stage_rows["stage_start_offset_ms"] = (
        stage_rows["dispatch_start_ms"] - stage_rows["entry_ts_ms"]
    )
    stage_rows["stage_completion_offset_ms"] = (
        stage_rows["dispatch_end_ms"] - stage_rows["entry_ts_ms"]
    )
    stage_rows["decomposition_residual_ms"] = (
        stage_rows["dispatch_latency_ms"]
        - stage_rows["platform_overhead_ms"]
        - stage_rows["action_duration_ms"]
    )
    stage_rows["cold_like_normalized"] = stage_rows["latency_class"] == "cold_like"
    return stage_rows


def quantile_or_nan(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(q))


def summarize_group(group: pd.DataFrame, columns: Iterable[str]) -> dict:
    out = {"rows": int(len(group))}
    for col in columns:
        if col not in group.columns:
            continue
        values = pd.to_numeric(group[col], errors="coerce").dropna()
        prefix = col.replace("_ms", "")
        out[f"{prefix}_mean_ms"] = float(values.mean()) if not values.empty else float("nan")
        out[f"{prefix}_std_ms"] = float(values.std(ddof=0)) if not values.empty else float("nan")
        out[f"{prefix}_min_ms"] = float(values.min()) if not values.empty else float("nan")
        for name, q in QUANTILES.items():
            out[f"{prefix}_{name}_ms"] = quantile_or_nan(values, q)
        out[f"{prefix}_max_ms"] = float(values.max()) if not values.empty else float("nan")
    return out


def grouped_summary(
    frame: pd.DataFrame,
    group_cols: list[str],
    latency_cols: list[str] | None = None,
) -> pd.DataFrame:
    latency_cols = latency_cols or LATENCY_COLUMNS
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(summarize_group(group, latency_cols))
        if "latency_class" in group.columns:
            row["cold_like_count"] = int((group["latency_class"] == "cold_like").sum())
            row["warm_count"] = int((group["latency_class"] == "warm").sum())
            row["unknown_count"] = int((group["latency_class"] == "unknown").sum())
            denom = max(1, int(len(group)))
            row["cold_like_rate"] = row["cold_like_count"] / denom
        rows.append(row)
    return pd.DataFrame(rows)


def build_workflow_instances(stage_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (trace_label, trace_file, workflow_name, request_id), group in stage_rows.groupby(
        ["trace_label", "trace_file", "workflow_name", "request_id"], dropna=False
    ):
        entry_ts = float(group["entry_ts_ms"].iloc[0])
        end_idx = group["dispatch_end_ms"].idxmax()
        critical_stage = str(group.loc[end_idx, "stage_name"])
        workflow_latency = float(group["dispatch_end_ms"].max() - entry_ts)
        rows.append(
            {
                "trace_label": trace_label,
                "trace_file": trace_file,
                "workflow_name": workflow_name,
                "request_id": request_id,
                "entry_ts_ms": entry_ts,
                "workflow_latency_ms": workflow_latency,
                "critical_stage": critical_stage,
                "stage_count": int(len(group)),
                "cold_like_stage_count": int((group["latency_class"] == "cold_like").sum()),
                "warm_stage_count": int((group["latency_class"] == "warm").sum()),
                "total_dispatch_latency_ms": float(group["dispatch_latency_ms"].sum()),
                "total_platform_overhead_ms": float(group["platform_overhead_ms"].sum()),
                "total_action_duration_ms": float(group["action_duration_ms"].sum()),
                "max_stage_dispatch_latency_ms": float(group["dispatch_latency_ms"].max()),
            }
        )
    instances = pd.DataFrame(rows)
    summary = grouped_summary(
        instances,
        ["trace_label", "workflow_name"],
        latency_cols=[
            "workflow_latency_ms",
            "total_dispatch_latency_ms",
            "total_platform_overhead_ms",
            "total_action_duration_ms",
            "max_stage_dispatch_latency_ms",
        ],
    )
    if not instances.empty:
        critical = (
            instances.groupby(["trace_label", "workflow_name", "critical_stage"])
            .size()
            .reset_index(name="critical_count")
        )
        totals = (
            instances.groupby(["trace_label", "workflow_name"])
            .size()
            .reset_index(name="workflow_count")
        )
        critical = critical.merge(totals, on=["trace_label", "workflow_name"], how="left")
        critical["critical_share"] = critical["critical_count"] / critical["workflow_count"]
    else:
        critical = pd.DataFrame()
    return pd.concat([summary], ignore_index=True), critical


def write_plots(out_dir: Path, stage_by_class: pd.DataFrame, workflow_instances: pd.DataFrame) -> None:
    plot_frame = stage_by_class[stage_by_class["latency_class"].isin(["warm", "cold_like"])].copy()
    if not plot_frame.empty:
        plot_frame["stage_label"] = (
            plot_frame["workflow_name"].astype(str)
            + "\n"
            + plot_frame["stage_name"].astype(str)
        )

        for metric, title, filename in [
            (
                "dispatch_latency_p95_ms",
                "P95 dispatch latency by stage and warm/cold-like class",
                "stage_dispatch_latency_p95_by_class.png",
            ),
            (
                "platform_overhead_p95_ms",
                "P95 platform overhead by stage and warm/cold-like class",
                "stage_platform_overhead_p95_by_class.png",
            ),
        ]:
            if metric not in plot_frame.columns:
                continue
            pivot = plot_frame.pivot_table(
                index="stage_label",
                columns="latency_class",
                values=metric,
                aggfunc="mean",
            ).fillna(0.0)
            pivot = pivot[[col for col in ["warm", "cold_like"] if col in pivot.columns]]
            fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 0.7), 5.5), dpi=180)
            pivot.plot(kind="bar", ax=ax, color=["#2f855a", "#c53030"][: len(pivot.columns)])
            ax.set_title(title)
            ax.set_ylabel("milliseconds")
            ax.set_xlabel("stage")
            ax.grid(True, axis="y", linestyle=":", alpha=0.7)
            ax.tick_params(axis="x", labelrotation=55)
            fig.tight_layout()
            fig.savefig(out_dir / filename)
            plt.close(fig)

        mean_cols = ["platform_overhead_mean_ms", "action_duration_mean_ms"]
        if all(col in plot_frame.columns for col in mean_cols):
            stacked = (
                plot_frame.assign(
                    label=lambda x: x["workflow_name"].astype(str)
                    + "\n"
                    + x["stage_name"].astype(str)
                    + "\n"
                    + x["latency_class"].astype(str)
                )
                .set_index("label")[mean_cols]
                .rename(
                    columns={
                        "platform_overhead_mean_ms": "platform overhead",
                        "action_duration_mean_ms": "action duration",
                    }
                )
            )
            fig, ax = plt.subplots(figsize=(max(10, len(stacked) * 0.55), 5.8), dpi=180)
            stacked.plot(kind="bar", stacked=True, ax=ax, color=["#dd6b20", "#3182ce"])
            ax.set_title("Mean dispatch latency decomposition")
            ax.set_ylabel("milliseconds")
            ax.set_xlabel("stage / class")
            ax.grid(True, axis="y", linestyle=":", alpha=0.7)
            ax.tick_params(axis="x", labelrotation=65)
            fig.tight_layout()
            fig.savefig(out_dir / "mean_latency_decomposition_by_stage_class.png")
            plt.close(fig)

    if not workflow_instances.empty:
        labels = sorted(workflow_instances["trace_label"].unique())
        data = [
            workflow_instances[workflow_instances["trace_label"] == label][
                "workflow_latency_ms"
            ].dropna()
            for label in labels
        ]
        if any(len(values) for values in data):
            fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 5.2), dpi=180)
            ax.boxplot(data, labels=labels, showfliers=False)
            ax.set_title("Workflow end-to-end latency distribution")
            ax.set_ylabel("milliseconds")
            ax.grid(True, axis="y", linestyle=":", alpha=0.7)
            ax.tick_params(axis="x", labelrotation=20)
            fig.tight_layout()
            fig.savefig(out_dir / "workflow_latency_distribution.png")
            plt.close(fig)


def write_readme(
    out_dir: Path,
    traces: list[str],
    stage_overall: pd.DataFrame,
    workflow_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Stage 3-A Latency Profile Pack",
        "",
        "## Scope",
        "",
        "- This pack is an offline coarse latency profiler for the current Stage-3 minimum viable step.",
        "- It separates `warm` and `cold_like` paths when `cold_like` is available.",
        "- It uses the coarse decomposition `dispatch_latency_ms = platform_overhead_ms + action_duration_ms`.",
        "- It is intended to feed the later workflow SLO-risk estimator; it is not yet a full OpenWhisk internal breakdown.",
        "",
        "## Inputs",
        "",
    ]
    lines.extend([f"- `{trace}`" for trace in traces])
    lines.extend(
        [
            "",
            "## Main Outputs",
            "",
            "- `latency_observations.csv`: cleaned per-stage latency rows.",
            "- `stage_latency_profile_by_class.csv`: stage-level warm/cold-like latency quantiles.",
            "- `stage_latency_profile_overall.csv`: stage-level overall cold-like rate and latency quantiles.",
            "- `stage_offset_profile.csv`: entry-to-stage start/completion offset distributions.",
            "- `workflow_latency_instances.csv`: one row per workflow request.",
            "- `workflow_latency_profile.csv`: workflow end-to-end latency quantiles.",
            "- `latency_samples_for_monte_carlo.csv`: empirical samples for Stage-4 Monte Carlo.",
            "",
            "## Stage-Level Overall Profile",
            "",
        ]
    )
    compact_stage_cols = [
        "trace_label",
        "workflow_name",
        "stage_name",
        "rows",
        "cold_like_rate",
        "dispatch_latency_p50_ms",
        "dispatch_latency_p90_ms",
        "dispatch_latency_p95_ms",
        "platform_overhead_p95_ms",
        "action_duration_p95_ms",
    ]
    compact_stage = stage_overall[[col for col in compact_stage_cols if col in stage_overall.columns]]
    lines.append(table_text(compact_stage))
    lines.extend(["", "## Workflow Latency Profile", ""])
    compact_workflow_cols = [
        "trace_label",
        "workflow_name",
        "rows",
        "workflow_latency_p50_ms",
        "workflow_latency_p90_ms",
        "workflow_latency_p95_ms",
        "total_platform_overhead_p95_ms",
        "total_action_duration_p95_ms",
    ]
    compact_workflow = workflow_summary[
        [col for col in compact_workflow_cols if col in workflow_summary.columns]
    ]
    lines.append(table_text(compact_workflow))
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Real OpenWhisk pilot traces are small and currently mainly cover `sebs_trip_booking`.",
            "- The `sebs_video` continuous-moderate trace is synthetic-stage data calibrated from pilot traces.",
            "- Use this pack to design Stage-4 risk estimation, but do not present it as final real-cluster latency evidence.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def table_text(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def main() -> None:
    args = parse_args()
    root = project_root()
    labels = args.trace_labels
    if labels is None or len(labels) == 0:
        labels = [Path(path).stem for path in args.traces]
    if len(labels) != len(args.traces):
        raise ValueError("--trace-labels length must match --traces length")

    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        load_trace(
            root=root,
            trace_path=trace,
            trace_label=label,
            cold_threshold_ms=args.cold_overhead_threshold_ms,
            include_failed=args.include_failed,
        )
        for trace, label in zip(args.traces, labels)
    ]
    stage_rows = pd.concat(frames, ignore_index=True)

    stage_by_class = grouped_summary(
        stage_rows,
        ["trace_label", "workflow_name", "stage_name", "latency_class"],
    ).sort_values(["trace_label", "workflow_name", "stage_name", "latency_class"])
    stage_overall = grouped_summary(
        stage_rows,
        ["trace_label", "workflow_name", "stage_name"],
    ).sort_values(["trace_label", "workflow_name", "stage_name"])
    offset_profile = grouped_summary(
        stage_rows,
        ["trace_label", "workflow_name", "stage_name"],
        latency_cols=["stage_start_offset_ms", "stage_completion_offset_ms"],
    ).sort_values(["trace_label", "workflow_name", "stage_name"])
    workflow_instances = build_workflow_instances(stage_rows)[0]

    # The helper above returns the profile in [0] and critical-stage profile in [1].
    workflow_rows = []
    for (trace_label, trace_file, workflow_name, request_id), group in stage_rows.groupby(
        ["trace_label", "trace_file", "workflow_name", "request_id"], dropna=False
    ):
        entry_ts = float(group["entry_ts_ms"].iloc[0])
        end_idx = group["dispatch_end_ms"].idxmax()
        workflow_rows.append(
            {
                "trace_label": trace_label,
                "trace_file": trace_file,
                "workflow_name": workflow_name,
                "request_id": request_id,
                "entry_ts_ms": entry_ts,
                "workflow_latency_ms": float(group["dispatch_end_ms"].max() - entry_ts),
                "critical_stage": str(group.loc[end_idx, "stage_name"]),
                "stage_count": int(len(group)),
                "cold_like_stage_count": int((group["latency_class"] == "cold_like").sum()),
                "warm_stage_count": int((group["latency_class"] == "warm").sum()),
                "total_dispatch_latency_ms": float(group["dispatch_latency_ms"].sum()),
                "total_platform_overhead_ms": float(group["platform_overhead_ms"].sum()),
                "total_action_duration_ms": float(group["action_duration_ms"].sum()),
                "max_stage_dispatch_latency_ms": float(group["dispatch_latency_ms"].max()),
            }
        )
    workflow_instances = pd.DataFrame(workflow_rows)
    workflow_summary, critical_stage_profile = build_workflow_instances(stage_rows)

    sample_cols = [
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
    ]
    monte_carlo_samples = stage_rows[[col for col in sample_cols if col in stage_rows.columns]].copy()

    stage_rows.to_csv(out_dir / "latency_observations.csv", index=False)
    stage_by_class.to_csv(out_dir / "stage_latency_profile_by_class.csv", index=False)
    stage_overall.to_csv(out_dir / "stage_latency_profile_overall.csv", index=False)
    offset_profile.to_csv(out_dir / "stage_offset_profile.csv", index=False)
    workflow_instances.to_csv(out_dir / "workflow_latency_instances.csv", index=False)
    workflow_summary.to_csv(out_dir / "workflow_latency_profile.csv", index=False)
    critical_stage_profile.to_csv(out_dir / "workflow_critical_stage_profile.csv", index=False)
    monte_carlo_samples.to_csv(out_dir / "latency_samples_for_monte_carlo.csv", index=False)

    write_plots(out_dir, stage_by_class, workflow_instances)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_traces": [str(resolve_path(root, trace)) for trace in args.traces],
        "trace_labels": labels,
        "out_dir": str(out_dir),
        "rows": int(len(stage_rows)),
        "workflow_instances": int(len(workflow_instances)),
        "cold_overhead_threshold_ms": args.cold_overhead_threshold_ms,
        "include_failed": bool(args.include_failed),
        "notes": [
            "Coarse Stage-3 profile: warm/cold-like, platform overhead, action duration.",
            "Not a full OpenWhisk internal component decomposition.",
            "Use latency_samples_for_monte_carlo.csv as the Stage-4 empirical latency pool.",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_readme(out_dir, args.traces, stage_overall, workflow_summary)

    print(f"wrote {out_dir}")
    print(f"stage rows: {len(stage_rows)}")
    print(f"workflow instances: {len(workflow_instances)}")
    print("stage profile:")
    compact_cols = [
        "trace_label",
        "workflow_name",
        "stage_name",
        "rows",
        "cold_like_rate",
        "dispatch_latency_p50_ms",
        "dispatch_latency_p95_ms",
    ]
    print(stage_overall[[col for col in compact_cols if col in stage_overall.columns]].to_string(index=False))


if __name__ == "__main__":
    main()

