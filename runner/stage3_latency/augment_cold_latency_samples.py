import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .profile_latency import LATENCY_COLUMNS, grouped_summary, load_trace


# Optional calibration pilot traces (pass via --calibration-traces).
# Historically defaulted to SeBS trip-booking pilots that have since been
# removed from this repo; on a real server, pass your own pilot CSVs.
DEFAULT_CALIBRATION_TRACES: list[str] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pilot-calibrated augmented cold-like latency sample pool. "
            "The generated samples are for sensitivity analysis and Monte Carlo, "
            "not a replacement for real OpenWhisk cold-start experiments."
        )
    )
    parser.add_argument("--target-trace", required=True)
    parser.add_argument("--target-label", default="target_trace")
    parser.add_argument("--calibration-traces", nargs="*", default=DEFAULT_CALIBRATION_TRACES)
    parser.add_argument("--calibration-label", default="real_trip_booking_pilot")
    parser.add_argument("--out-dir", default="reports/stage3_latency_augmented_cold")
    parser.add_argument("--min-cold-samples-per-stage", type=int, default=500)
    parser.add_argument("--cold-overhead-threshold-ms", type=float, default=500.0)
    parser.add_argument("--min-stage-cold-source", type=int, default=20)
    parser.add_argument("--overhead-jitter-sigma", type=float, default=0.08)
    parser.add_argument("--action-jitter-sigma", type=float, default=0.04)
    parser.add_argument("--memory-mb", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def clean_positive(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return arr[np.isfinite(arr) & (arr > 0)]


def sample_with_lognormal_jitter(
    pool: np.ndarray,
    size: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(pool) == 0:
        raise ValueError("cannot sample from an empty pool")
    sampled = rng.choice(pool, size=size, replace=True)
    if sigma <= 0:
        return sampled.astype(float)
    noise = rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=size)
    return np.maximum(1.0, sampled.astype(float) * noise)


def load_target(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    return load_trace(
        root=root,
        trace_path=args.target_trace,
        trace_label=args.target_label,
        cold_threshold_ms=args.cold_overhead_threshold_ms,
        include_failed=False,
    )


def load_calibration(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for idx, trace in enumerate(args.calibration_traces):
        label = f"{args.calibration_label}_{idx}"
        frames.append(
            load_trace(
                root=root,
                trace_path=trace,
                trace_label=label,
                cold_threshold_ms=args.cold_overhead_threshold_ms,
                include_failed=False,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_augmented_rows(
    target: pd.DataFrame,
    calibration: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    global_cold_overhead = clean_positive(
        calibration[calibration["latency_class"] == "cold_like"]["platform_overhead_ms"]
    )
    if len(global_cold_overhead) == 0:
        global_cold_overhead = clean_positive(
            target[target["latency_class"] == "cold_like"]["platform_overhead_ms"]
        )
    if len(global_cold_overhead) == 0:
        raise ValueError("no cold-like overhead samples found in calibration or target traces")

    rows = []
    for (workflow_name, stage_name), stage in target.groupby(["workflow_name", "stage_name"]):
        if stage_name == "__entry__":
            continue
        existing_cold = stage[stage["latency_class"] == "cold_like"].copy()
        needed = max(0, args.min_cold_samples_per_stage - len(existing_cold))
        if needed == 0:
            continue

        stage_cold_overhead = clean_positive(existing_cold["platform_overhead_ms"])
        if len(stage_cold_overhead) >= args.min_stage_cold_source:
            overhead_pool = stage_cold_overhead
            source_pool = "target_stage_cold"
        else:
            # Use real pilot cold overhead when target stage cold samples are too sparse.
            overhead_pool = global_cold_overhead
            source_pool = "real_pilot_global_cold"

        action_pool = clean_positive(stage["action_duration_ms"])
        if len(action_pool) == 0:
            action_pool = clean_positive(target["action_duration_ms"])
        if len(action_pool) == 0:
            raise ValueError(f"no action_duration_ms samples available for stage {stage_name}")

        start_pool = clean_positive(stage["stage_start_offset_ms"])
        if len(start_pool) == 0:
            start_pool = np.asarray([0.0])

        overhead = sample_with_lognormal_jitter(
            overhead_pool,
            needed,
            args.overhead_jitter_sigma,
            rng,
        )
        action = sample_with_lognormal_jitter(
            action_pool,
            needed,
            args.action_jitter_sigma,
            rng,
        )
        start_offset = rng.choice(start_pool, size=needed, replace=True).astype(float)
        dispatch_latency = overhead + action
        completion_offset = start_offset + dispatch_latency

        for i in range(needed):
            rows.append(
                {
                    "trace_label": f"{args.target_label}_augmented_cold",
                    "workflow_name": workflow_name,
                    "stage_name": stage_name,
                    "latency_class": "cold_like",
                    "dispatch_latency_ms": float(dispatch_latency[i]),
                    "platform_overhead_ms": float(overhead[i]),
                    "action_duration_ms": float(action[i]),
                    "stage_start_offset_ms": float(start_offset[i]),
                    "stage_completion_offset_ms": float(completion_offset[i]),
                    "cold_like_normalized": True,
                    "memory_mb": int(args.memory_mb),
                    "sample_origin": "augmented_cold",
                    "source_pool": source_pool,
                    "source_overhead_pool_size": int(len(overhead_pool)),
                    "source_action_pool_size": int(len(action_pool)),
                    "synthetic_sample_id": f"{workflow_name}:{stage_name}:cold_aug:{i}",
                }
            )
    return pd.DataFrame(rows)


def prepare_original_samples(target: pd.DataFrame, calibration: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    keep = [
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
    target_out = target[[col for col in keep if col in target.columns]].copy()
    target_out["memory_mb"] = int(args.memory_mb)
    target_out["sample_origin"] = "target_observed"
    target_out["source_pool"] = "target_trace"
    target_out["source_overhead_pool_size"] = np.nan
    target_out["source_action_pool_size"] = np.nan

    cal_out = calibration[[col for col in keep if col in calibration.columns]].copy()
    cal_out["memory_mb"] = int(args.memory_mb)
    cal_out["sample_origin"] = "real_pilot_observed"
    cal_out["source_pool"] = "real_pilot_trace"
    cal_out["source_overhead_pool_size"] = np.nan
    cal_out["source_action_pool_size"] = np.nan
    return pd.concat([target_out, cal_out], ignore_index=True)


def main() -> None:
    args = parse_args()
    root = project_root()
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = load_target(root, args)
    calibration = load_calibration(root, args)
    augmented = build_augmented_rows(target, calibration, args)
    original = prepare_original_samples(target, calibration, args)
    combined = pd.concat([original, augmented], ignore_index=True)

    target_summary = grouped_summary(
        target,
        ["trace_label", "workflow_name", "stage_name", "latency_class"],
        LATENCY_COLUMNS,
    )
    calibration_summary = grouped_summary(
        calibration,
        ["trace_label", "workflow_name", "stage_name", "latency_class"],
        LATENCY_COLUMNS,
    )
    augmented_summary = grouped_summary(
        augmented,
        ["trace_label", "workflow_name", "stage_name", "latency_class", "source_pool"],
        LATENCY_COLUMNS,
    )
    combined_summary = grouped_summary(
        combined,
        ["sample_origin", "workflow_name", "stage_name", "latency_class"],
        LATENCY_COLUMNS,
    )

    target_summary.to_csv(out_dir / "target_latency_summary.csv", index=False)
    calibration_summary.to_csv(out_dir / "calibration_latency_summary.csv", index=False)
    augmented_summary.to_csv(out_dir / "augmented_cold_summary.csv", index=False)
    combined_summary.to_csv(out_dir / "combined_latency_summary.csv", index=False)
    augmented.to_csv(out_dir / "augmented_cold_samples.csv", index=False)
    combined.to_csv(out_dir / "latency_samples_for_monte_carlo_augmented.csv", index=False)

    source_counts = {
        "target_rows": int(len(target)),
        "target_cold_like_rows": int((target["latency_class"] == "cold_like").sum()),
        "target_warm_rows": int((target["latency_class"] == "warm").sum()),
        "calibration_rows": int(len(calibration)),
        "calibration_cold_like_rows": int((calibration["latency_class"] == "cold_like").sum()),
        "calibration_warm_rows": int((calibration["latency_class"] == "warm").sum()),
        "augmented_cold_rows": int(len(augmented)),
        "combined_rows": int(len(combined)),
    }
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_trace": str(resolve_path(root, args.target_trace)),
        "target_label": args.target_label,
        "calibration_traces": [str(resolve_path(root, t)) for t in args.calibration_traces],
        "out_dir": str(out_dir),
        "min_cold_samples_per_stage": args.min_cold_samples_per_stage,
        "cold_overhead_threshold_ms": args.cold_overhead_threshold_ms,
        "min_stage_cold_source": args.min_stage_cold_source,
        "overhead_jitter_sigma": args.overhead_jitter_sigma,
        "action_jitter_sigma": args.action_jitter_sigma,
        "memory_mb": args.memory_mb,
        "seed": args.seed,
        "source_counts": source_counts,
        "interpretation": [
            "Augmented cold samples are bootstrap/jitter samples calibrated from real pilot cold overheads.",
            "They are intended for sensitivity analysis and Monte Carlo system plumbing.",
            "They are not a substitute for full real OpenWhisk cold-start measurements.",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "# Stage 3 Augmented Cold Latency Samples",
        "",
        "## Purpose",
        "",
        "Current main trace has too few cold-like samples for stable Monte Carlo tail analysis.",
        "This report builds a pilot-calibrated cold-like sample pool by bootstrapping real pilot cold overheads",
        "and combining them with target-stage action-duration distributions.",
        "",
        "## Source Counts",
        "",
        pd.DataFrame([source_counts]).to_markdown(index=False),
        "",
        "## Target Summary",
        "",
        target_summary.to_markdown(index=False),
        "",
        "## Augmented Cold Summary",
        "",
        augmented_summary.to_markdown(index=False),
        "",
        "## Important Caveat",
        "",
        "These augmented rows are for sensitivity/error analysis and Stage-4 Monte Carlo plumbing only.",
        "Paper claims about cold-start latency should be replaced by real OpenWhisk measurements once available.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote augmented cold latency samples to {out_dir}")


if __name__ == "__main__":
    main()

