import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create memory-tier summary tables from a profile_latency report pack."
    )
    parser.add_argument(
        "--profile-dir",
        required=True,
        help="directory produced by runner.profile_latency",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory; defaults to --profile-dir",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def memory_from_label(label: object) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*mb", str(label).lower())
    if not match:
        match = re.search(r"mem[_-]?(\d+(?:\.\d+)?)", str(label).lower())
    return float(match.group(1)) if match else np.nan


def add_memory_column(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["memory_mb"] = out["trace_label"].map(memory_from_label)
    return out.sort_values(["workflow_name", "memory_mb", "trace_label"])


def add_speedup_columns(workflow: pd.DataFrame) -> pd.DataFrame:
    out = workflow.copy()
    metric_cols = [
        "workflow_latency_p50_ms",
        "workflow_latency_p90_ms",
        "workflow_latency_p95_ms",
        "total_platform_overhead_p95_ms",
        "total_action_duration_p95_ms",
    ]
    for col in metric_cols:
        if col not in out.columns:
            continue
        speedup_col = col.replace("_ms", "_speedup_vs_min_memory")
        out[speedup_col] = np.nan
        for workflow_name, group in out.groupby("workflow_name", dropna=False):
            ordered = group.sort_values(["memory_mb", "trace_label"])
            baseline_values = pd.to_numeric(ordered[col], errors="coerce").dropna()
            if baseline_values.empty:
                continue
            baseline = float(baseline_values.iloc[0])
            if baseline <= 0:
                continue
            idx = ordered.index
            current = pd.to_numeric(out.loc[idx, col], errors="coerce")
            out.loc[idx, speedup_col] = baseline / current.replace(0, np.nan)
    return out


def keep_existing(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[col for col in columns if col in frame.columns]].copy()


def table_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def write_readme(
    out_dir: Path,
    stage_summary: pd.DataFrame,
    workflow_summary: pd.DataFrame,
) -> None:
    workflow_cols = [
        "memory_mb",
        "trace_label",
        "workflow_name",
        "rows",
        "workflow_latency_p50_ms",
        "workflow_latency_p95_ms",
        "total_platform_overhead_p95_ms",
        "total_action_duration_p95_ms",
        "workflow_latency_p95_speedup_vs_min_memory",
    ]
    stage_cols = [
        "memory_mb",
        "trace_label",
        "workflow_name",
        "stage_name",
        "rows",
        "cold_like_rate",
        "dispatch_latency_p95_ms",
        "platform_overhead_p95_ms",
        "action_duration_p95_ms",
    ]

    lines = [
        "# Memory Sweep Summary",
        "",
        "## Scope",
        "",
        "- This pack compares OpenWhisk action memory tiers on the same workflow.",
        "- `action_duration_ms` is measured inside the Python action.",
        "- `platform_overhead_ms = dispatch_latency_ms - action_duration_ms`.",
        "- The overhead is a coarse dispatch/platform proxy, not pure Kubernetes scheduler time.",
        "",
        "## Main Outputs",
        "",
        "- `memory_workflow_summary.csv`: workflow-level latency by memory tier.",
        "- `memory_stage_summary.csv`: stage-level latency by memory tier.",
        "- `memory_sweep_metadata.json`: generation metadata.",
        "",
        "## Workflow-Level View",
        "",
        table_text(keep_existing(workflow_summary, workflow_cols)),
        "",
        "## Stage-Level View",
        "",
        table_text(keep_existing(stage_summary, stage_cols)),
        "",
        "## Interpretation Guardrails",
        "",
        "- Compare memory tiers only within the same cluster run and similar background load.",
        "- First calls after action updates are useful cold-like samples, but OpenWhisk reuse is still controlled by the platform.",
        "- For paper claims, call this a memory-tier latency profile or platform-overhead profile, not a full scheduler decomposition.",
        "",
    ]
    (out_dir / "README_memory_sweep.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = project_root()
    profile_dir = resolve_path(root, args.profile_dir)
    out_dir = resolve_path(root, args.out_dir or args.profile_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_path = profile_dir / "stage_latency_profile_overall.csv"
    workflow_path = profile_dir / "workflow_latency_profile.csv"
    if not stage_path.exists():
        raise FileNotFoundError(stage_path)
    if not workflow_path.exists():
        raise FileNotFoundError(workflow_path)

    stage_summary = add_memory_column(pd.read_csv(stage_path))
    workflow_summary = add_speedup_columns(add_memory_column(pd.read_csv(workflow_path)))

    stage_summary.to_csv(out_dir / "memory_stage_summary.csv", index=False)
    workflow_summary.to_csv(out_dir / "memory_workflow_summary.csv", index=False)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile_dir": str(profile_dir),
        "out_dir": str(out_dir),
        "stage_rows": int(len(stage_summary)),
        "workflow_rows": int(len(workflow_summary)),
        "notes": [
            "Summaries are derived from runner.profile_latency outputs.",
            "platform_overhead_ms is dispatch latency minus in-action duration.",
            "It is not a pure Kubernetes scheduler timing breakdown.",
        ],
    }
    (out_dir / "memory_sweep_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_readme(out_dir, stage_summary, workflow_summary)

    print(f"wrote {out_dir}")
    print("workflow memory summary:")
    compact_cols = [
        "memory_mb",
        "workflow_name",
        "rows",
        "workflow_latency_p50_ms",
        "workflow_latency_p95_ms",
        "total_platform_overhead_p95_ms",
        "total_action_duration_p95_ms",
        "workflow_latency_p95_speedup_vs_min_memory",
    ]
    print(keep_existing(workflow_summary, compact_cols).to_string(index=False))


if __name__ == "__main__":
    main()

