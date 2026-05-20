"""Extract entry-only traces for multiple Azure apps in a single pass.

Each app keeps its original Azure continuous timestamps (sub-second precision)
and is scaled by --time-compression (1 minute -> 2 scaled seconds when
compression=30). For each app the script picks the densest source window that
covers --scaled-hours of scaled time and shifts that window to start at t=0
so all apps "operate simultaneously" in scaled time.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--azure-trace", required=True)
    p.add_argument("--app-list", required=True, help="csv with columns: label, app, func, workflow_config")
    p.add_argument("--time-compression", type=float, default=30.0)
    p.add_argument("--scaled-hours", type=float, default=2.0)
    p.add_argument("--base-entry-ts-ms", type=int, default=1_920_000_000_000)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--source-window-stride-min", type=int, default=15,
                   help="stride (source-minutes) when scanning for the densest window")
    return p.parse_args()


def load_app_list(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"label", "app", "func", "workflow_config"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"app-list missing columns: {missing}")
    return df


def scan_azure(azure_trace: Path, app_func_pairs: list[tuple[str, str]], chunksize: int) -> pd.DataFrame:
    """Filter the 305 MB Azure file in one pass for the requested (app, func) pairs."""
    target_pairs = {(a, f) for a, f in app_func_pairs}
    frames = []
    for chunk in pd.read_csv(azure_trace, chunksize=chunksize, usecols=["app", "func", "end_timestamp", "duration"]):
        mask = pd.Series(False, index=chunk.index)
        # vectorised pair-match: build a tuple-key
        chunk_keys = list(zip(chunk["app"].astype(str), chunk["func"].astype(str)))
        keep = [k in target_pairs for k in chunk_keys]
        sub = chunk.loc[keep].copy()
        if sub.empty:
            continue
        sub["end_timestamp"] = sub["end_timestamp"].astype(float)
        sub["duration"] = sub["duration"].astype(float)
        sub["source_start_s"] = sub["end_timestamp"] - sub["duration"]
        sub["source_end_s"] = sub["end_timestamp"]
        sub["source_duration_ms"] = (sub["duration"] * 1000.0).round().astype("int64")
        frames.append(sub[["app", "func", "source_start_s", "source_end_s", "source_duration_ms"]])
    if not frames:
        raise RuntimeError("no rows matched any (app, func) pair")
    return pd.concat(frames, ignore_index=True).sort_values(["app", "func", "source_start_s"]).reset_index(drop=True)


def pick_dense_window(rows: pd.DataFrame, source_window_s: float, stride_s: float) -> tuple[float, float]:
    """Pick a [start, end] window of size source_window_s that maximises count."""
    if rows.empty:
        raise ValueError("empty rows")
    starts = rows["source_start_s"].values
    t_lo = float(starts.min())
    t_hi = float(starts.max())
    if (t_hi - t_lo) <= source_window_s:
        return t_lo, t_lo + source_window_s
    best_start = t_lo
    best_count = -1
    scan = np.arange(t_lo, t_hi - source_window_s + 1.0, stride_s)
    for s in scan:
        cnt = int(np.sum((starts >= s) & (starts < s + source_window_s)))
        if cnt > best_count:
            best_count = cnt
            best_start = s
    return float(best_start), float(best_start) + source_window_s


def build_per_app(
    app_rows: pd.DataFrame,
    label: str,
    workflow_name: str,
    workflow_config: str,
    time_compression: float,
    scaled_hours: float,
    base_entry_ts_ms: int,
    source_window_stride_min: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    source_window_s = scaled_hours * 3600.0 * time_compression
    stride_s = max(60.0, float(source_window_stride_min) * 60.0)
    window_start_s, window_end_s = pick_dense_window(app_rows, source_window_s, stride_s)

    window = app_rows[(app_rows["source_start_s"] >= window_start_s) & (app_rows["source_start_s"] < window_end_s)].copy()
    if window.empty:
        raise ValueError(f"no rows in selected window for {label}")
    window = window.sort_values("source_start_s").reset_index(drop=True)

    window["target_offset_ms"] = (
        ((window["source_start_s"] - window_start_s) * 1000.0) / time_compression
    ).round().astype("int64")
    window["index"] = np.arange(len(window))

    schedule = window.assign(
        workflow_name=workflow_name,
        source_label=f"azure_multiapp_{label}",
        source_app=window["app"],
        source_func=window["func"],
    )[
        ["workflow_name", "index", "target_offset_ms", "source_label", "source_app", "source_func",
         "source_start_s", "source_end_s", "source_duration_ms"]
    ]

    entry_rows = []
    for _, row in schedule.iterrows():
        entry_ts_ms = int(base_entry_ts_ms + int(row["target_offset_ms"]))
        rid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{workflow_name}:{label}:{int(row['index'])}:{entry_ts_ms}"))
        entry_rows.append({
            "workflow_name": workflow_name,
            "request_id": rid,
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
        })
    entry_trace = pd.DataFrame(entry_rows, columns=TRACE_COLUMNS)

    intervals_ms = schedule["target_offset_ms"].diff().dropna()
    duration_minutes = (window["source_start_s"].iloc[-1] - window["source_start_s"].iloc[0]) / 60.0
    # per-minute bucket from window start
    bucket = ((window["source_start_s"] - window_start_s) // 60).astype(int)
    per_min = bucket.value_counts()
    span_min = int(math.ceil(source_window_s / 60.0))
    per_min = per_min.reindex(range(0, span_min), fill_value=0)
    meta = {
        "label": label,
        "workflow_name": workflow_name,
        "workflow_config": workflow_config,
        "source_app": str(window["app"].iloc[0]),
        "source_func": str(window["func"].iloc[0]),
        "rows": int(len(window)),
        "time_compression": float(time_compression),
        "scaled_hours": float(scaled_hours),
        "scaled_span_s": float((window_end_s - window_start_s) / time_compression),
        "base_entry_ts_ms": int(base_entry_ts_ms),
        "source_window_start_s": float(window_start_s),
        "source_window_end_s": float(window_end_s),
        "source_window_minutes": int(source_window_s / 60.0),
        "first_source_start_s": float(window["source_start_s"].iloc[0]),
        "last_source_start_s": float(window["source_start_s"].iloc[-1]),
        "source_duration_minutes": float(duration_minutes),
        "source_per_min_counts": {
            "active_ratio": float((per_min > 0).mean()),
            "mean": float(per_min.mean()),
            "max": int(per_min.max()),
            "p95": float(per_min.quantile(0.95)),
            "p99": float(per_min.quantile(0.99)),
            "cv": float(per_min.std(ddof=0) / max(1e-9, per_min.mean())),
        },
        "interval_ms": {
            "count": int(len(intervals_ms)),
            "mean": float(intervals_ms.mean()) if len(intervals_ms) else 0.0,
            "median": float(intervals_ms.median()) if len(intervals_ms) else 0.0,
            "p95": float(intervals_ms.quantile(0.95)) if len(intervals_ms) else 0.0,
            "max": float(intervals_ms.max()) if len(intervals_ms) else 0.0,
        },
        "notes": [
            "Entry-only trace using Azure continuous timestamps (end_timestamp - duration).",
            "No within-minute uniform spreading.",
            "scaled time begins at t=0 for every app, so apps run simultaneously when replayed.",
        ],
    }
    return schedule, entry_trace, meta


def main() -> None:
    args = parse_args()
    azure_trace = Path(args.azure_trace)
    app_list_path = Path(args.app_list)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apps = load_app_list(app_list_path)
    pairs = [(str(a), str(f)) for a, f in zip(apps["app"], apps["func"])]
    print(f"scanning Azure trace for {len(pairs)} (app, func) pairs ...")
    azure_rows = scan_azure(azure_trace, pairs, args.chunksize)
    print(f"  matched rows: {len(azure_rows):,}")
    by_pair = azure_rows.groupby(["app", "func"])

    meta_all = {}
    for _, app_row in apps.iterrows():
        label = str(app_row["label"])
        app = str(app_row["app"])
        func = str(app_row["func"])
        wf_config = str(app_row["workflow_config"])

        if (app, func) not in by_pair.groups:
            print(f"[{label}] NO matched rows; skip")
            continue
        rows = by_pair.get_group((app, func)).copy()

        workflow_name = Path(wf_config).stem
        schedule, entry_trace, meta = build_per_app(
            app_rows=rows,
            label=label,
            workflow_name=workflow_name,
            workflow_config=wf_config,
            time_compression=args.time_compression,
            scaled_hours=args.scaled_hours,
            base_entry_ts_ms=args.base_entry_ts_ms,
            source_window_stride_min=args.source_window_stride_min,
        )

        app_dir = out_dir / label
        app_dir.mkdir(parents=True, exist_ok=True)
        schedule.to_csv(app_dir / f"schedule_{label}.csv", index=False)
        entry_trace.to_csv(app_dir / f"entry_trace_{label}.csv", index=False)
        (app_dir / f"metadata_{label}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        meta_all[label] = meta

        print(f"[{label}] rows={meta['rows']}, scaled_span_s={meta['scaled_span_s']:.0f}, "
              f"active_ratio={meta['source_per_min_counts']['active_ratio']:.3f}, "
              f"per_min_max={meta['source_per_min_counts']['max']}")

    (out_dir / "multiapp_metadata.json").write_text(json.dumps(meta_all, indent=2), encoding="utf-8")
    print(f"wrote summary -> {out_dir / 'multiapp_metadata.json'}")


if __name__ == "__main__":
    main()
