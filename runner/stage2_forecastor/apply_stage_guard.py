import argparse
import math
from pathlib import Path

import pandas as pd


POLICIES = ["p50", "p90", "p95", "p99"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window-sec", type=int, default=5)
    parser.add_argument(
        "--window-ms",
        type=int,
        default=None,
        help="override --window-sec with a millisecond-level window",
    )
    parser.add_argument(
        "--stage-guards",
        required=True,
        help="comma-separated guards, for example decode=0,resize=1,classify=1",
    )
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument(
        "--method-suffix",
        default="stageguard",
        help="suffix appended to the method column",
    )
    return parser.parse_args()


def resolve_window_ms(args: argparse.Namespace) -> int:
    if args.window_ms is not None:
        if args.window_ms <= 0:
            raise ValueError("--window-ms must be positive")
        return args.window_ms
    if args.window_sec <= 0:
        raise ValueError("--window-sec must be positive")
    return args.window_sec * 1000


def parse_stage_guards(value: str) -> dict[str, int]:
    guards = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid stage guard: {item}")
        stage, guard = item.split("=", 1)
        guard_value = int(guard)
        if guard_value < 0:
            raise ValueError(f"stage guard must be non-negative: {item}")
        guards[stage.strip()] = guard_value
    return guards


def ceil_count(value: float) -> int:
    return int(math.ceil(max(0.0, value)))


def alloc_count(value: float, activation_threshold: float) -> int:
    if value < activation_threshold:
        return 0
    return ceil_count(value)


def main() -> None:
    args = parse_args()
    window_ms = resolve_window_ms(args)
    guards = parse_stage_guards(args.stage_guards)
    forecast = pd.read_csv(args.forecast)

    rows = []
    for _, row in forecast.iterrows():
        stage_name = row["stage_name"]
        guard = guards.get(stage_name, 0)
        for offset in range(-guard, guard + 1):
            target_window = int(row["window"]) + offset
            if target_window < 0:
                continue
            item = row.to_dict()
            item["window"] = target_window
            item["window_start_ms"] = target_window * window_ms
            if "method" in item:
                item["method"] = f"{item['method']}+{args.method_suffix}"
            rows.append(item)

    if not rows:
        raise SystemExit("no forecast rows produced after applying stage guards")

    guarded = pd.DataFrame(rows)
    group_cols = [
        "workflow_name",
        "method",
        "stage_name",
        "window",
        "window_start_ms",
    ]
    value_cols = [f"{policy}_count" for policy in POLICIES if f"{policy}_count" in guarded]
    guarded = guarded.groupby(group_cols, as_index=False)[value_cols].max()

    for policy in POLICIES:
        count_col = f"{policy}_count"
        if count_col not in guarded:
            continue
        guarded[f"ceil_{policy}_count"] = guarded[count_col].map(ceil_count)
        guarded[f"alloc_{policy}_count"] = guarded[count_col].map(
            lambda value: alloc_count(value, args.activation_threshold)
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    guarded.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

