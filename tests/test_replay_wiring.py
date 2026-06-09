#!/usr/bin/env python3
"""Verify multi-SLO replay wiring helpers without a live cluster."""

from __future__ import annotations

import csv
import random

from scripts import replay_civic_azure_schedule as replay


def test_v1_assign_slo_class_is_seeded_and_ratio_accurate() -> None:
    n = 10_000
    premium_ratio = 0.37
    seed = 20260609
    first = [
        replay.assign_slo_class(random.Random(seed), premium_ratio)
        for _ in range(1)
    ]
    rng_a = random.Random(seed)
    rng_b = random.Random(seed)
    seq_a = [replay.assign_slo_class(rng_a, premium_ratio) for _ in range(n)]
    seq_b = [replay.assign_slo_class(rng_b, premium_ratio) for _ in range(n)]
    observed = seq_a.count("premium") / n
    print(
        "V1 SLO assignment: "
        f"seed={seed} n={n} premium_ratio={premium_ratio:.3f} "
        f"observed={observed:.4f} first_draw={first[0]} reproducible={seq_a == seq_b}"
    )
    assert seq_a == seq_b
    assert abs(observed - premium_ratio) <= 0.02


def test_v2_load_plan_by_class_matches_plan_summary_csv() -> None:
    plans = replay.load_plan_by_class(replay.DEFAULT_PLAN_CSV)
    expected_premium = {
        "detect_object": 1536,
        "estimate_pose": 2560,
        "match_face": 1280,
        "classify_scene": 3072,
        "translate_alert": 1280,
    }
    expected_free = {
        "detect_object": 768,
        "estimate_pose": 1280,
        "match_face": 1280,
        "classify_scene": 1280,
        "translate_alert": 1024,
    }
    print(
        "V2 plan CSV: "
        f"premium={plans['premium']} free={plans['free']}"
    )
    assert plans["premium"] == expected_premium
    assert plans["free"] == expected_free


def test_v3_summarize_per_class_and_csv_columns(tmp_path) -> None:
    workflows = [
        {
            "request_id": "p1",
            "slo_class": "premium",
            "workflow_e2e_ms": 14000,
            "execution_gb_seconds": 10.0,
            "dynamic_upgraded": True,
            "dynamic_upgrade_count": 2,
        },
        {
            "request_id": "p2",
            "slo_class": "premium",
            "workflow_e2e_ms": 16000,
            "execution_gb_seconds": 20.0,
            "dynamic_upgraded": False,
            "dynamic_upgrade_count": 0,
        },
        {
            "request_id": "f1",
            "slo_class": "free",
            "workflow_e2e_ms": 19000,
            "execution_gb_seconds": 5.0,
            "dynamic_upgraded": False,
            "dynamic_upgrade_count": 0,
        },
    ]
    stages = [
        {"request_id": "p1", "stage_name": "detect_object", "stage_latency_class": "cold"},
        {"request_id": "p1", "stage_name": "estimate_pose", "stage_latency_class": "warm"},
        {"request_id": "p2", "stage_name": "detect_object", "stage_latency_class": "warm"},
        {"request_id": "p2", "stage_name": "estimate_pose", "stage_latency_class": "cold"},
        {"request_id": "p2", "stage_name": "match_face", "stage_latency_class": "warm"},
        {"request_id": "f1", "stage_name": "detect_object", "stage_latency_class": "warm"},
        {"request_id": "f1", "stage_name": "translate_alert", "stage_latency_class": "cold"},
    ]
    rows = replay.summarize_per_class(
        workflows,
        stages,
        {"premium": 15000.0, "free": 20000.0},
    )
    out = tmp_path / "per_class_summary.csv"
    replay.write_csv(out, replay.PER_CLASS_SUMMARY_COLUMNS, rows)
    with out.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)

    premium = next(row for row in rows if row["slo_class"] == "premium")
    free = next(row for row in rows if row["slo_class"] == "free")
    print(
        "V3 per-class summary: "
        f"columns={header} premium={premium} free={free}"
    )
    assert header == replay.PER_CLASS_SUMMARY_COLUMNS
    assert premium["count"] == 2
    assert premium["slo_satisfaction_rate"] == 0.5
    assert premium["dynamic_trigger_count"] == 1
    assert premium["total_upgrades"] == 2
    assert abs(premium["entry_cold_rate"] - 0.5) < 1e-12
    assert abs(premium["downstream_cold_rate"] - (1.0 / 3.0)) < 1e-12
    assert premium["cost_gbsec"] == 30.0
    assert free["count"] == 1
    assert free["slo_satisfaction_rate"] == 1.0
    assert free["dynamic_trigger_count"] == 0
    assert free["downstream_cold_rate"] == 1.0


def test_v4_import_check() -> None:
    print(
        "V4 import check: "
        f"module={replay.__name__} default_plan_csv={replay.DEFAULT_PLAN_CSV}"
    )
    assert replay.DEFAULT_PLAN_CSV.name == "plan_summary.csv"
