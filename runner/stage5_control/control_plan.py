from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_STAGE = "*"
DEFAULT_WINDOW = -1


@dataclass(frozen=True)
class PlanRow:
    workflow_name: str | None
    stage_name: str
    window: int
    warm_count: float
    keepalive_ttl_sec: float
    memory_mb: int
    source: str = "manual"
    note: str = ""

    def validate(self) -> None:
        if not self.stage_name:
            raise ValueError("stage_name must be non-empty")
        if self.window < DEFAULT_WINDOW:
            raise ValueError(f"window must be >= {DEFAULT_WINDOW}, got {self.window}")
        if self.warm_count < 0:
            raise ValueError(f"warm_count must be non-negative, got {self.warm_count}")
        if self.keepalive_ttl_sec < 0:
            raise ValueError(
                f"keepalive_ttl_sec must be non-negative, got {self.keepalive_ttl_sec}"
            )
        if self.memory_mb <= 0:
            raise ValueError(f"memory_mb must be positive, got {self.memory_mb}")


@dataclass
class ControlPlan:
    rows: list[PlanRow]
    window_sec: float
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.window_sec <= 0:
            raise ValueError(f"window_sec must be positive, got {self.window_sec}")
        for row in self.rows:
            row.validate()

    def lookup(self, stage_name: str, window: int) -> PlanRow | None:
        by_key = self._lookup_map()
        for key in (
            (stage_name, window),
            (stage_name, DEFAULT_WINDOW),
            (DEFAULT_STAGE, window),
            (DEFAULT_STAGE, DEFAULT_WINDOW),
        ):
            if key in by_key:
                return by_key[key]
        return None

    def _lookup_map(self) -> dict[tuple[str, int], PlanRow]:
        lookup: dict[tuple[str, int], PlanRow] = {}
        for row in self.rows:
            lookup[(row.stage_name, row.window)] = row
        return lookup


def _coerce_float(value: Any, name: str, default: float | None = None) -> float:
    if value is None or value == "":
        if default is None:
            raise ValueError(f"{name} is required")
        return float(default)
    return float(value)


def _coerce_int(value: Any, name: str, default: int | None = None) -> int:
    if value is None or value == "":
        if default is None:
            raise ValueError(f"{name} is required")
        return int(default)
    return int(float(value))


def frame_to_plan(
    frame: pd.DataFrame,
    *,
    window_sec: float,
    default_workflow_name: str | None = None,
    default_memory_mb: int = 256,
    metadata: dict[str, Any] | None = None,
) -> ControlPlan:
    rows: list[PlanRow] = []
    for record in frame.to_dict(orient="records"):
        stage_name = str(record.get("stage_name", "")).strip()
        warm_default = record.get("allocated_count", 0.0)
        row = PlanRow(
            workflow_name=record.get("workflow_name") or default_workflow_name,
            stage_name=stage_name,
            window=_coerce_int(record.get("window", DEFAULT_WINDOW), "window", DEFAULT_WINDOW),
            warm_count=_coerce_float(record.get("warm_count", warm_default), "warm_count", 0.0),
            keepalive_ttl_sec=_coerce_float(
                record.get("keepalive_ttl_sec", 0.0), "keepalive_ttl_sec", 0.0
            ),
            memory_mb=_coerce_int(record.get("memory_mb", default_memory_mb), "memory_mb"),
            source=str(record.get("source", "manual")),
            note=str(record.get("note", "")),
        )
        rows.append(row)
    return ControlPlan(rows=rows, window_sec=window_sec, metadata=metadata or {})


def plan_to_frame(plan: ControlPlan) -> pd.DataFrame:
    rows = [asdict(row) for row in plan.rows]
    if not rows:
        return pd.DataFrame(
            columns=[
                "workflow_name",
                "stage_name",
                "window",
                "warm_count",
                "keepalive_ttl_sec",
                "memory_mb",
                "source",
                "note",
            ]
        )
    return pd.DataFrame(rows)


def load_control_plan(path: str | Path, *, default_window_sec: float = 5.0) -> ControlPlan:
    plan_path = Path(path)
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)

    if plan_path.suffix.lower() == ".json":
        data = json.loads(plan_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows_data = data
            metadata: dict[str, Any] = {}
            window_sec = default_window_sec
            workflow_name = None
            default_memory_mb = 256
        elif isinstance(data, dict):
            rows_data = data.get("rows") or data.get("plan") or []
            metadata = {k: v for k, v in data.items() if k not in {"rows", "plan"}}
            window_sec = float(data.get("window_sec", default_window_sec))
            workflow_name = data.get("workflow_name")
            default_memory_mb = int(data.get("default_memory_mb", 256))
        else:
            raise ValueError(f"Unsupported JSON control plan shape in {plan_path}")
        frame = pd.DataFrame(rows_data)
        return frame_to_plan(
            frame,
            window_sec=window_sec,
            default_workflow_name=workflow_name,
            default_memory_mb=default_memory_mb,
            metadata=metadata,
        )

    frame = pd.read_csv(plan_path)
    return frame_to_plan(frame, window_sec=default_window_sec, metadata={"source_path": str(plan_path)})


def save_control_plan(plan: ControlPlan, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".json":
        payload = {
            "window_sec": plan.window_sec,
            **plan.metadata,
            "rows": [asdict(row) for row in plan.rows],
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return
    plan_to_frame(plan).to_csv(out_path, index=False)


def expand_control_plan(
    plan: ControlPlan,
    *,
    stages: list[str],
    windows: list[int],
) -> pd.DataFrame:
    expanded: list[dict[str, Any]] = []
    for stage_name in stages:
        for window in windows:
            row = plan.lookup(stage_name, int(window))
            if row is None:
                continue
            expanded.append(
                {
                    "workflow_name": row.workflow_name,
                    "stage_name": stage_name,
                    "window": int(window),
                    "warm_count": float(row.warm_count),
                    "keepalive_ttl_sec": float(row.keepalive_ttl_sec),
                    "memory_mb": int(row.memory_mb),
                    "source": row.source,
                    "note": row.note,
                }
            )
    return pd.DataFrame(expanded)


def build_baseline_plan_from_forecast(
    forecast_detail: pd.DataFrame,
    *,
    workflow_name: str | None,
    window_sec: float,
    keepalive_ttl_sec: float,
    memory_mb: int,
    allocation_column: str = "allocated_count",
) -> ControlPlan:
    required = {"stage_name", "target_window", allocation_column}
    missing = required.difference(forecast_detail.columns)
    if missing:
        raise ValueError(f"forecast_detail is missing columns: {sorted(missing)}")

    grouped = (
        forecast_detail.groupby(["stage_name", "target_window"], as_index=False)[allocation_column]
        .max()
        .rename(columns={"target_window": "window", allocation_column: "warm_count"})
    )
    grouped["workflow_name"] = workflow_name
    grouped["keepalive_ttl_sec"] = keepalive_ttl_sec
    grouped["memory_mb"] = memory_mb
    grouped["source"] = f"baseline:{allocation_column}"
    return frame_to_plan(
        grouped,
        window_sec=window_sec,
        default_workflow_name=workflow_name,
        default_memory_mb=memory_mb,
        metadata={
            "workflow_name": workflow_name,
            "allocation_column": allocation_column,
            "plan_type": "forecast_baseline",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and normalize Stage 5 control plans.")
    parser.add_argument("--control-plan", required=True, help="Input JSON or CSV control plan.")
    parser.add_argument("--out", required=True, help="Output normalized JSON or CSV path.")
    parser.add_argument("--default-window-sec", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_control_plan(args.control_plan, default_window_sec=args.default_window_sec)
    save_control_plan(plan, args.out)
    print(
        f"normalized {len(plan.rows)} plan rows to {args.out} "
        f"(window_sec={plan.window_sec})"
    )


if __name__ == "__main__":
    main()
