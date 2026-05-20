import argparse
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..workflow import load_workflow


TRACE_COLUMNS = [
    "workflow_name",
    "request_id",
    "stage_name",
    "parent_stages",
    "entry_ts_ms",
    "dispatch_start_ms",
    "dispatch_end_ms",
    "dispatch_latency_ms",
    "action_start_ns",
    "action_end_ns",
    "action_duration_ms",
    "platform_overhead_ms",
    "container_id",
    "cold_like",
    "status",
    "error",
]


DEFAULT_KEY = (
    "d27353c8ad7c924a609457eb5a53333a7e519bcf8efd884dcca7ffb908ca3fa6::"
    "905e6674359f6487df567fa2c8ca1c8641e7740f2e32d9fd26e9fe1ff7a4670d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a two-hour scaled challenge entry trace from Azure minute-count data. "
            "One source minute is mapped to two seconds by default."
        )
    )
    parser.add_argument(
        "--candidate-characterization",
        default="notebooks/data/azure_analysis/azure2021_candidate_characterization.csv",
    )
    parser.add_argument(
        "--candidate-minute-counts",
        default="notebooks/data/azure_analysis/azure2021_candidate_minute_counts.csv",
    )
    parser.add_argument("--workflow-config", default="configs/sebs_video.yaml")
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--source-label", default="azure_periodic_drift_challenge_scaled30")
    parser.add_argument("--source-span-minutes", type=int, default=3600)
    parser.add_argument("--time-compression", type=float, default=30.0)
    parser.add_argument("--base-entry-ts-ms", type=int, default=1_910_000_000_000)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--out-schedule", required=True)
    parser.add_argument("--out-trace", required=True)
    parser.add_argument("--out-analysis-dir", required=True)
    parser.add_argument("--metadata-out", required=True)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def dense_minute_counts(minute_counts: pd.DataFrame, key: str) -> pd.Series:
    rows = minute_counts[minute_counts["key"] == key].copy()
    if rows.empty:
        raise ValueError(f"no minute counts found for key={key}")
    rows["bin"] = pd.to_numeric(rows["bin"], errors="coerce").astype(int)
    rows["count"] = pd.to_numeric(rows["count"], errors="coerce").fillna(0).astype(int)
    first_bin = int(rows["bin"].min())
    last_bin = int(rows["bin"].max())
    return (
        rows.groupby("bin")["count"]
        .sum()
        .reindex(range(first_bin, last_bin + 1), fill_value=0)
        .astype(int)
    )


def choose_segment(counts: pd.Series, span_minutes: int) -> pd.Series:
    if len(counts) < span_minutes:
        raise ValueError(
            f"source has only {len(counts)} minutes, but --source-span-minutes={span_minutes}"
        )
    values = counts.to_numpy(dtype=float)
    cumsum = np.concatenate([[0.0], np.cumsum(values)])
    best_score = -1e18
    best_start = 0
    for start in range(0, len(values) - span_minutes + 1):
        end = start + span_minutes
        segment = values[start:end]
        total = cumsum[end] - cumsum[start]
        active_ratio = float(np.mean(segment > 0))
        mean = float(np.mean(segment))
        std = float(np.std(segment))
        # Prefer a nontrivial but not pure-burst segment: enough traffic, variation,
        # and active minutes for train/test learning.
        score = total + 120.0 * std + 600.0 * active_ratio - 30.0 * abs(active_ratio - 0.35)
        if mean <= 0:
            score -= 1e6
        if score > best_score:
            best_score = score
            best_start = start
    return counts.iloc[best_start : best_start + span_minutes].copy()


def split_key(key: str) -> tuple[str, str]:
    if "::" not in key:
        return "", key
    app, func = key.split("::", 1)
    return app, func


def scaled_offset_ms(source_minute_offset: int, within_source_minute_ms: float, time_compression: float) -> int:
    source_offset_ms = source_minute_offset * 60_000.0 + within_source_minute_ms
    return int(round(source_offset_ms / time_compression))


def build_schedule(
    segment: pd.Series,
    workflow_name: str,
    key: str,
    source_label: str,
    time_compression: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    app, func = split_key(key)
    rows = []
    first_source_bin = int(segment.index.min())
    index = 0
    for source_bin, count in segment.items():
        count = int(count)
        if count <= 0:
            continue
        # The source table is minute-granular. Spread calls inside the minute so the
        # scaled schedule preserves count intensity without inventing extra bursts.
        within = np.sort(rng.uniform(0.0, 60_000.0, size=count))
        source_minute_offset = int(source_bin - first_source_bin)
        for within_ms in within:
            offset_ms = scaled_offset_ms(source_minute_offset, float(within_ms), time_compression)
            source_start_s = float(source_bin * 60 + within_ms / 1000.0)
            rows.append(
                {
                    "workflow_name": workflow_name,
                    "index": index,
                    "target_offset_ms": offset_ms,
                    "source_label": source_label,
                    "source_app": app,
                    "source_func": func,
                    "source_start_s": source_start_s,
                    "source_end_s": source_start_s,
                    "source_duration_ms": 0,
                }
            )
            index += 1
    schedule = pd.DataFrame(rows)
    if schedule.empty:
        raise ValueError("selected segment generated an empty schedule")
    return schedule.sort_values(["target_offset_ms", "index"]).reset_index(drop=True)


def build_entry_trace(schedule: pd.DataFrame, workflow_name: str, base_entry_ts_ms: int) -> pd.DataFrame:
    rows = []
    for _, row in schedule.iterrows():
        entry_ts_ms = int(base_entry_ts_ms + int(row["target_offset_ms"]))
        request_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{workflow_name}:{row['source_label']}:{int(row['index'])}:{entry_ts_ms}",
            )
        )
        rows.append(
            {
                "workflow_name": workflow_name,
                "request_id": request_id,
                "stage_name": "__entry__",
                "parent_stages": "",
                "entry_ts_ms": entry_ts_ms,
                "dispatch_start_ms": entry_ts_ms,
                "dispatch_end_ms": entry_ts_ms,
                "dispatch_latency_ms": 0,
                "action_start_ns": "",
                "action_end_ns": "",
                "action_duration_ms": "",
                "platform_overhead_ms": "",
                "container_id": "",
                "cold_like": "",
                "status": "ok",
                "error": "",
            }
        )
    return pd.DataFrame(rows, columns=TRACE_COLUMNS)


def characterize(schedule: pd.DataFrame, segment: pd.Series, time_compression: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scaled = schedule.copy()
    scaled["scaled_5s_window"] = (scaled["target_offset_ms"] // 5000).astype(int)
    scaled_counts = (
        scaled.groupby("scaled_5s_window")
        .size()
        .reindex(range(0, int(math.ceil((segment.size * 60_000 / time_compression) / 5000))), fill_value=0)
        .reset_index(name="count")
    )
    source_counts = segment.reset_index()
    source_counts.columns = ["source_minute_bin", "count"]

    intervals = schedule["target_offset_ms"].diff().dropna()
    scaled_count_values = scaled_counts["count"].astype(float)
    source_values = segment.astype(float)
    metadata = {
        "source_minutes": int(segment.size),
        "scaled_span_ms": int(segment.size * 60_000 / time_compression),
        "scaled_span_hours": float((segment.size * 60_000 / time_compression) / 3_600_000),
        "source_total": int(source_values.sum()),
        "source_active_ratio": float((source_values > 0).mean()),
        "source_mean_all": float(source_values.mean()),
        "source_mean_active": float(source_values[source_values > 0].mean()) if (source_values > 0).any() else 0.0,
        "source_max": int(source_values.max()),
        "source_cv": float(source_values.std(ddof=0) / max(1e-9, source_values.mean())),
        "scaled_5s_windows": int(len(scaled_counts)),
        "scaled_5s_active_ratio": float((scaled_count_values > 0).mean()),
        "scaled_5s_mean_all": float(scaled_count_values.mean()),
        "scaled_5s_mean_active": float(scaled_count_values[scaled_count_values > 0].mean()) if (scaled_count_values > 0).any() else 0.0,
        "scaled_5s_p50": float(scaled_count_values.quantile(0.50)),
        "scaled_5s_p90": float(scaled_count_values.quantile(0.90)),
        "scaled_5s_p95": float(scaled_count_values.quantile(0.95)),
        "scaled_5s_max": int(scaled_count_values.max()),
        "scaled_5s_cv": float(scaled_count_values.std(ddof=0) / max(1e-9, scaled_count_values.mean())),
        "interarrival_scaled_mean_ms": float(intervals.mean()) if len(intervals) else 0.0,
        "interarrival_scaled_p95_ms": float(intervals.quantile(0.95)) if len(intervals) else 0.0,
        "interarrival_scaled_max_ms": float(intervals.max()) if len(intervals) else 0.0,
    }
    return source_counts, scaled_counts, metadata


def main() -> None:
    args = parse_args()
    root = project_root()
    workflow = load_workflow(str(resolve_path(root, args.workflow_config)))
    char_path = resolve_path(root, args.candidate_characterization)
    counts_path = resolve_path(root, args.candidate_minute_counts)
    out_schedule = resolve_path(root, args.out_schedule)
    out_trace = resolve_path(root, args.out_trace)
    out_analysis_dir = resolve_path(root, args.out_analysis_dir)
    metadata_out = resolve_path(root, args.metadata_out)

    characterization = pd.read_csv(char_path)
    minute_counts = pd.read_csv(counts_path)
    counts = dense_minute_counts(minute_counts, args.key)
    segment = choose_segment(counts, args.source_span_minutes)

    rng = np.random.default_rng(args.seed)
    schedule = build_schedule(
        segment=segment,
        workflow_name=workflow.workflow_name,
        key=args.key,
        source_label=args.source_label,
        time_compression=args.time_compression,
        rng=rng,
    )
    entry_trace = build_entry_trace(schedule, workflow.workflow_name, args.base_entry_ts_ms)
    source_counts, scaled_counts, trace_metadata = characterize(
        schedule,
        segment,
        args.time_compression,
    )

    out_schedule.parent.mkdir(parents=True, exist_ok=True)
    out_trace.parent.mkdir(parents=True, exist_ok=True)
    out_analysis_dir.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)

    schedule.to_csv(out_schedule, index=False)
    entry_trace.to_csv(out_trace, index=False)
    source_counts.to_csv(out_analysis_dir / "source_minute_counts.csv", index=False)
    scaled_counts.to_csv(out_analysis_dir / "scaled_5s_counts.csv", index=False)

    app, func = split_key(args.key)
    cand = characterization[characterization["key"] == args.key].copy()
    selected_source_start_bin = int(segment.index.min())
    selected_source_end_bin = int(segment.index.max())
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_name": workflow.workflow_name,
        "workflow_config": str(resolve_path(root, args.workflow_config)),
        "key": args.key,
        "source_app": app,
        "source_func": func,
        "source_label": args.source_label,
        "seed": args.seed,
        "time_compression": args.time_compression,
        "mapping": "one source minute to two scaled seconds when time_compression=30",
        "selected_source_start_bin": selected_source_start_bin,
        "selected_source_end_bin": selected_source_end_bin,
        "candidate_characterization": cand.iloc[0].to_dict() if not cand.empty else {},
        "trace_characterization": trace_metadata,
        "outputs": {
            "schedule": str(out_schedule),
            "entry_trace": str(out_trace),
            "analysis_dir": str(out_analysis_dir),
        },
        "notes": [
            "This trace is Azure-minute-count-derived, not exact invocation-second replay.",
            "Minute-level counts are spread uniformly within each source minute before compression.",
            "It is intended as a challenging forecasting trace with periodic/drift structure.",
        ],
    }
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote {out_schedule}")
    print(f"wrote {out_trace}")
    print(f"wrote {metadata_out}")
    print(pd.DataFrame([trace_metadata]).to_string(index=False))


if __name__ == "__main__":
    main()

