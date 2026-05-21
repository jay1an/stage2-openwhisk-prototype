from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..workflow import load_workflow


STATE_COLUMNS = ("was_cold_start", "cold_like", "is_cold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build state-conditional DAG delay kernels from stage traces. "
            "The output separates warm, cold, and legacy unconditional offsets."
        )
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def normalize_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y", "cold", "cold_like"}:
        return True
    if text in {"false", "f", "0", "no", "n", "warm"}:
        return False
    return None


def find_state_column(trace: pd.DataFrame) -> str:
    for column in STATE_COLUMNS:
        if column in trace.columns:
            return column
    if "latency_class" in trace.columns:
        return "latency_class"
    raise ValueError(
        "trace must contain a cold-state column. Checked: "
        + ", ".join((*STATE_COLUMNS, "latency_class"))
    )


def state_from_column(row: pd.Series, column: str) -> bool | None:
    if column == "latency_class":
        text = str(row.get(column, "")).strip().lower()
        if text.startswith("cold"):
            return True
        if text == "warm":
            return False
        return None
    return normalize_bool(row.get(column))


def offset_distribution(rows: pd.DataFrame, window_ms: float) -> pd.DataFrame:
    delay_ms = (
        pd.to_numeric(rows["dispatch_start_ms"], errors="coerce")
        - pd.to_numeric(rows["entry_ts_ms"], errors="coerce")
    )
    offsets = (delay_ms.clip(lower=0.0) // window_ms).astype("Int64")
    offsets = offsets.dropna().astype(int)
    counts = offsets.value_counts().sort_index()
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("cannot build delay kernel from empty offset distribution")
    return pd.DataFrame(
        {
            "offset_windows": counts.index.astype(int),
            "probability": counts.to_numpy(dtype=float) / total,
        }
    )


def build_delay_kernel(trace: pd.DataFrame, workflow_name: str, stages: list[str], window_ms: float) -> pd.DataFrame:
    required = {"workflow_name", "stage_name", "entry_ts_ms", "dispatch_start_ms"}
    missing = required - set(trace.columns)
    if missing:
        raise ValueError(f"trace missing required columns: {sorted(missing)}")

    state_column = find_state_column(trace)
    rows = trace[
        (trace["workflow_name"].astype(str) == workflow_name)
        & (trace["stage_name"].astype(str) != "__entry__")
    ].copy()
    if "status" in rows.columns:
        rows = rows[rows["status"].astype(str) == "ok"].copy()
    if rows.empty:
        raise ValueError(f"no stage rows found for workflow {workflow_name}")

    rows["prev_cold"] = rows.apply(lambda row: state_from_column(row, state_column), axis=1)
    rows = rows[rows["prev_cold"].notna()].copy()
    if rows.empty:
        raise ValueError(f"state column {state_column} did not contain usable warm/cold values")

    out_rows: list[dict[str, Any]] = []
    for stage in stages:
        stage_rows = rows[rows["stage_name"].astype(str) == stage].copy()
        if stage_rows.empty:
            raise ValueError(f"no trace rows found for stage {stage}")
        state_groups = {
            "warm": stage_rows[stage_rows["prev_cold"] == False],
            "cold": stage_rows[stage_rows["prev_cold"] == True],
            "any": stage_rows,
        }
        for state, state_rows in state_groups.items():
            if state_rows.empty:
                raise ValueError(
                    f"no {state} rows for stage {stage}; cannot build state-conditional kernel"
                )
            dist = offset_distribution(state_rows, window_ms)
            for record in dist.to_dict(orient="records"):
                out_rows.append(
                    {
                        "workflow_name": workflow_name,
                        "stage_name": stage,
                        "prev_state": state,
                        "offset_windows": int(record["offset_windows"]),
                        "probability": float(record["probability"]),
                    }
                )
    return pd.DataFrame(out_rows)


def kernel_summary(kernel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (workflow, stage, state), group in kernel.groupby(
        ["workflow_name", "stage_name", "prev_state"], as_index=False
    ):
        probability = pd.to_numeric(group["probability"], errors="coerce")
        offsets = pd.to_numeric(group["offset_windows"], errors="coerce")
        rows.append(
            {
                "workflow_name": workflow,
                "stage_name": stage,
                "prev_state": state,
                "rows": int(len(group)),
                "probability_sum": float(probability.sum()),
                "mean_offset_windows": float((offsets * probability).sum()),
                "max_offset_windows": int(offsets.max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.window_sec <= 0:
        raise ValueError("--window-sec must be positive")
    root = project_root()
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    workflow = load_workflow(str(resolve_path(root, args.workflow_config)))
    trace = pd.read_csv(resolve_path(root, args.trace))
    kernel = build_delay_kernel(
        trace,
        workflow_name=workflow.workflow_name,
        stages=list(workflow.nodes.keys()),
        window_ms=float(args.window_sec) * 1000.0,
    )
    kernel.to_csv(out_dir / "delay_kernel.csv", index=False)
    summary = kernel_summary(kernel)
    summary.to_csv(out_dir / "delay_kernel_summary.csv", index=False)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace": str(resolve_path(root, args.trace)),
        "workflow_config": str(resolve_path(root, args.workflow_config)),
        "workflow_name": workflow.workflow_name,
        "window_sec": args.window_sec,
        "rows": int(len(kernel)),
        "states": sorted(kernel["prev_state"].unique().tolist()),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'delay_kernel.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
