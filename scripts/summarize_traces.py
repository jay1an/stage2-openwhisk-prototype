import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def stats(vals):
    nums = [v for v in vals if v is not None]
    if not nums:
        return None
    if len(nums) == 1:
        return (nums[0], nums[0], nums[0], 1)
    return (statistics.mean(nums), min(nums), max(nums), len(nums))


def fmt_stat(s):
    if s is None:
        return "     -"
    if s[3] == 1:
        return f"{s[0]:8.1f}"
    return f"{s[0]:8.1f} [{s[1]:.0f}..{s[2]:.0f}]"


def summarize(csv_path: Path):
    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r["stage_name"] == "__entry__":
                continue
            rows.append(r)
    if not rows:
        return

    workflow_name = rows[0]["workflow_name"]
    by_stage = defaultdict(lambda: {"cold": [], "warm": []})
    for r in rows:
        bucket = "cold" if str(r.get("cold_like", "")).lower() == "true" else "warm"
        d = {
            "action_duration_ms": to_float(r["action_duration_ms"]),
            "dispatch_latency_ms": to_float(r["dispatch_latency_ms"]),
            "platform_overhead_ms": to_float(r["platform_overhead_ms"]),
            "cpu_user_ms": to_float(r.get("cpu_user_ms")),
            "cpu_system_ms": to_float(r.get("cpu_system_ms")),
            "cpu_process_ms": to_float(r.get("cpu_process_ms")),
            "mem_rss_kb": to_float(r.get("mem_rss_kb")),
            "mem_peak_kb": to_float(r.get("mem_peak_kb")),
        }
        by_stage[r["stage_name"]][bucket].append(d)

    per_request_total = defaultdict(float)
    per_request_kind = defaultdict(lambda: "warm")
    for r in rows:
        rid = r["request_id"]
        per_request_total[rid] = max(
            per_request_total[rid],
            to_float(r["dispatch_end_ms"]) - to_float(r["entry_ts_ms"]),
        )
        if str(r.get("cold_like", "")).lower() == "true":
            per_request_kind[rid] = "cold"

    print(f"\n=== {workflow_name}  ({csv_path.name}) ===")
    cold_e2e = [v for rid, v in per_request_total.items() if per_request_kind[rid] == "cold"]
    warm_e2e = [v for rid, v in per_request_total.items() if per_request_kind[rid] == "warm"]
    if cold_e2e:
        print(f"  end-to-end COLD request: mean={statistics.mean(cold_e2e):7.1f} ms  "
              f"min={min(cold_e2e):.0f}  max={max(cold_e2e):.0f}  n={len(cold_e2e)}")
    if warm_e2e:
        print(f"  end-to-end WARM request: mean={statistics.mean(warm_e2e):7.1f} ms  "
              f"min={min(warm_e2e):.0f}  max={max(warm_e2e):.0f}  n={len(warm_e2e)}")

    print()
    hdr = (f"  {'stage':<20} {'k':<4} {'n':>2}  "
           f"{'wall_ms':<22}  {'platform_ms':<22}  "
           f"{'cpu_user_ms':<22}  {'cpu_sys_ms':<18}  "
           f"{'rss_kb':<18}  {'peak_kb':<18}")
    print(hdr)
    for stage, buckets in by_stage.items():
        for kind in ("cold", "warm"):
            vals = buckets[kind]
            if not vals:
                continue
            keys = ["action_duration_ms", "platform_overhead_ms",
                    "cpu_user_ms", "cpu_system_ms", "mem_rss_kb", "mem_peak_kb"]
            cells = [fmt_stat(stats([v[k] for v in vals])) for k in keys]
            print(f"  {stage:<20} {kind:<4} {len(vals):>2}  "
                  f"{cells[0]:<22}  {cells[1]:<22}  "
                  f"{cells[2]:<22}  {cells[3]:<18}  "
                  f"{cells[4]:<18}  {cells[5]:<18}")


def main():
    for path in sys.argv[1:]:
        summarize(Path(path))


if __name__ == "__main__":
    main()
