#!/usr/bin/env python3
"""Regenerate replay cost summary from an existing raw trace."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.replay_civic_azure_schedule import (  # noqa: E402
    build_container_idle_records,
    stage_cost,
    stage_latency_class,
    summarize_cost,
    workflow_cold_class,
    write_csv,
)


DEFAULT_REPORT_DIR = (
    ROOT
    / "reports"
    / "civic_azure_cand2_45min_1280mb_1cpu_keepalive20s_target20s_balanced_mi96"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute cost_summary.csv from an existing raw_trace.csv."
    )
    parser.add_argument("--raw-trace", default=str(DEFAULT_REPORT_DIR / "raw_trace.csv"))
    parser.add_argument("--metadata", default=str(DEFAULT_REPORT_DIR / "run_metadata.json"))
    parser.add_argument("--old-cost-summary", default=str(DEFAULT_REPORT_DIR / "cost_summary.csv"))
    parser.add_argument("--out", default=str(DEFAULT_REPORT_DIR / "cost_summary_fixed.csv"))
    parser.add_argument("--memory-mb", type=int, default=None)
    parser.add_argument("--cpu-cores", type=float, default=None)
    parser.add_argument("--keepalive-sec", type=float, default=None)
    parser.add_argument("--price-per-gb-second", type=float, default=0.0)
    parser.add_argument("--price-per-vcpu-second", type=float, default=0.0)
    parser.add_argument("--price-per-request", type=float, default=0.0)
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("stage_name") != "__entry__"]


def entry_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row.get("stage_name") == "__entry__":
            return row
    return {}


def build_records(
    rows: list[dict[str, Any]],
    *,
    memory_mb: int,
    cpu_cores: float,
    price_per_gb_second: float,
    price_per_vcpu_second: float,
    price_per_request: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("request_id") or "")].append(row)

    workflow_records: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    for request_id, group in grouped.items():
        stages = stage_rows(group)
        if not request_id or not stages:
            continue
        entry = entry_row(group)
        cold_class = workflow_cold_class(stages)
        totals = {
            "gb_seconds": 0.0,
            "vcpu_seconds": 0.0,
            "memory_cost": 0.0,
            "cpu_cost": 0.0,
            "request_cost": 0.0,
            "total_cost": 0.0,
        }
        for row in stages:
            costs = stage_cost(
                row,
                memory_mb=memory_mb,
                cpu_cores=cpu_cores,
                price_per_gb_second=price_per_gb_second,
                price_per_vcpu_second=price_per_vcpu_second,
                price_per_request=price_per_request,
            )
            totals["gb_seconds"] += costs["execution_gb_seconds"]
            totals["vcpu_seconds"] += costs["execution_vcpu_seconds"]
            totals["memory_cost"] += costs["memory_cost"]
            totals["cpu_cost"] += costs["cpu_cost"]
            totals["request_cost"] += costs["request_cost"]
            totals["total_cost"] += costs["total_cost"]
            stage_records.append(
                {
                    **row,
                    "workflow_cold_class": cold_class,
                    "stage_latency_class": stage_latency_class(row),
                    "execution_gb_seconds": costs["execution_gb_seconds"],
                    "execution_vcpu_seconds": costs["execution_vcpu_seconds"],
                    "total_cost": costs["total_cost"],
                }
            )

        workflow_records.append(
            {
                "workflow_name": entry.get("workflow_name", stages[0].get("workflow_name", "")),
                "request_id": request_id,
                "status": entry.get("status", "ok"),
                "workflow_cold_class": cold_class,
                "stage_count": len(stages),
                "cold_stage_count": sum(str(row.get("cold_like")).lower() == "true" for row in stages),
                "execution_gb_seconds": totals["gb_seconds"],
                "execution_vcpu_seconds": totals["vcpu_seconds"],
                "request_count": len(stages),
                "memory_cost": totals["memory_cost"],
                "cpu_cost": totals["cpu_cost"],
                "request_cost": totals["request_cost"],
                "total_cost": totals["total_cost"],
            }
        )
    return workflow_records, stage_records


def read_single_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def main() -> None:
    args = parse_args()
    raw_trace = Path(args.raw_trace)
    metadata = load_metadata(Path(args.metadata))
    memory_mb = int(args.memory_mb if args.memory_mb is not None else metadata.get("memory_mb", 1280))
    cpu_cores = float(args.cpu_cores if args.cpu_cores is not None else metadata.get("cpu_cores", 1.0))
    keepalive_sec = float(args.keepalive_sec if args.keepalive_sec is not None else metadata.get("keepalive_sec", 20.0))

    rows = load_rows(raw_trace)
    workflow_records, stage_records = build_records(
        rows,
        memory_mb=memory_mb,
        cpu_cores=cpu_cores,
        price_per_gb_second=args.price_per_gb_second,
        price_per_vcpu_second=args.price_per_vcpu_second,
        price_per_request=args.price_per_request,
    )
    idle_records = build_container_idle_records(
        stage_records,
        keepalive_sec=keepalive_sec,
        memory_mb=memory_mb,
        cpu_cores=cpu_cores,
    )
    summary = summarize_cost(
        workflow_records,
        idle_records,
        memory_mb=memory_mb,
        cpu_cores=cpu_cores,
        keepalive_sec=keepalive_sec,
        price_per_gb_second=args.price_per_gb_second,
        price_per_vcpu_second=args.price_per_vcpu_second,
        price_per_request=args.price_per_request,
    )
    out_path = Path(args.out)
    write_csv(out_path, list(summary[0]), summary)

    old = read_single_csv(Path(args.old_cost_summary))
    new = summary[0]
    fields = [
        "execution_vcpu_seconds_total",
        "idle_vcpu_seconds_total",
        "total_vcpu_seconds_including_idle",
        "idle_gb_seconds_total",
        "lambda_style_gb_seconds",
        "lambda_style_cost",
        "provider_style_total_cost",
        "total_cost",
    ]
    print(f"wrote {out_path}")
    print("old vs new cost summary:")
    print("field,old,new")
    for field in fields:
        print(f"{field},{old.get(field, '')},{new.get(field, '')}")


if __name__ == "__main__":
    main()
