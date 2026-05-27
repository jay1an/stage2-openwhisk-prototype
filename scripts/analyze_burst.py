"""Analyze the 2-min uneven burst test: container create/reuse, cold/warm
overhead, and overlap with the pod-watcher lifecycle log."""

import csv
import os
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TRACE = os.path.join(ROOT, "data", "burst_civic_trace.csv")
TIMELINE = os.path.join(ROOT, "data", "burst_civic_timeline.log")
POD_LOG = os.path.join(ROOT, "data", "pod_events.log")


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    with open(TRACE) as f:
        rows = list(csv.DictReader(f))

    stage_rows = [r for r in rows if r["stage_name"] != "__entry__"]
    container_ids = sorted({r["container_id"] for r in stage_rows if r["container_id"]})
    request_ids = sorted({r["request_id"] for r in rows})

    print("=" * 72)
    print("BURST TEST — civic_alert_flow, 2 minutes, uneven, fire-and-forget")
    print("=" * 72)
    print(f"workflow invocations: {len(request_ids)}")
    print(f"stage executions:     {len(stage_rows)}")
    print(f"distinct containers:  {len(container_ids)}  ← unique pods used")
    cold_n = sum(1 for r in stage_rows if r["cold_like"] == "True")
    warm_n = sum(1 for r in stage_rows if r["cold_like"] == "False")
    print(f"cold invocations:     {cold_n}   (= new container created)")
    print(f"warm invocations:     {warm_n}   (= existing container reused)")
    print(f"reuse ratio:          {warm_n/len(stage_rows):.1%}")

    # Per-stage cold vs warm distinct containers
    print()
    print("--- Per-stage container creation vs reuse ---")
    print(f"{'stage':<18} {'cold':>5} {'warm':>5} {'containers':>11}  {'reuses/container':>18}")
    by_stage = defaultdict(list)
    for r in stage_rows:
        by_stage[r["stage_name"]].append(r)
    for stage, srows in by_stage.items():
        cids = {r["container_id"] for r in srows if r["container_id"]}
        c = sum(1 for r in srows if r["cold_like"] == "True")
        w = sum(1 for r in srows if r["cold_like"] == "False")
        per = len(srows) / max(1, len(cids))
        print(f"{stage:<18} {c:>5} {w:>5} {len(cids):>11}  {per:>17.2f}x")

    # Cold vs warm platform overhead per stage
    print()
    print("--- Per-stage timings (cold vs warm) ---")
    print(f"{'stage':<18} {'kind':>5} {'n':>3} {'wall_ms (mean)':>17} {'platform_ms':>13} {'action_dur_ms':>15} {'cpu_user_ms':>13}")
    for stage, srows in by_stage.items():
        for kind, flag in (("cold", "True"), ("warm", "False")):
            ss = [r for r in srows if r["cold_like"] == flag]
            if not ss:
                continue
            wall = [to_float(r["dispatch_latency_ms"]) for r in ss]
            wall = [v for v in wall if v is not None]
            plat = [to_float(r["platform_overhead_ms"]) for r in ss]
            plat = [v for v in plat if v is not None]
            adur = [to_float(r["action_duration_ms"]) for r in ss]
            adur = [v for v in adur if v is not None]
            cpu = [to_float(r["cpu_user_ms"]) for r in ss]
            cpu = [v for v in cpu if v is not None]
            def m(xs):
                return f"{statistics.mean(xs):.1f}" if xs else "—"
            print(f"{stage:<18} {kind:>5} {len(ss):>3} {m(wall):>17} {m(plat):>13} {m(adur):>15} {m(cpu):>13}")

    # Per-workflow cold count over schedule order
    print()
    print("--- Per-workflow cold count (schedule order) ---")
    # Order by entry_ts_ms of __entry__
    entries = [r for r in rows if r["stage_name"] == "__entry__"]
    entries.sort(key=lambda r: int(r["entry_ts_ms"]))
    for idx, e in enumerate(entries):
        rid = e["request_id"]
        wf_rows = [r for r in stage_rows if r["request_id"] == rid]
        c = sum(1 for r in wf_rows if r["cold_like"] == "True")
        w = sum(1 for r in wf_rows if r["cold_like"] == "False")
        e2e_start = int(e["entry_ts_ms"])
        e2e_end = max(int(r["dispatch_end_ms"]) for r in wf_rows)
        print(f"#{idx:02d} req={rid[:8]}  entry_ts={e2e_start}  cold={c}/{len(wf_rows)}  warm={w}/{len(wf_rows)}  e2e={e2e_end - e2e_start}ms")

    # Pod lifecycle from pod_events.log
    print()
    print("--- Pod lifecycle (from kubectl watcher) ---")
    if os.path.exists(POD_LOG):
        with open(POD_LOG) as f:
            lines = f.readlines()
        new_pods = [l for l in lines if " NEW " in l]
        gone_pods = [l for l in lines if " GONE " in l]
        # Count by stage suffix
        stage_creates = Counter()
        for l in new_pods:
            for stage in ("detect-object", "estimate-pose", "match-face", "classify-scene", "translate-alert"):
                if stage in l:
                    stage_creates[stage] += 1
                    break
        print(f"total NEW pod events:    {len(new_pods)}")
        print(f"total GONE pod events:   {len(gone_pods)}")
        print("new pods per stage:")
        for stage, n in stage_creates.most_common():
            print(f"  {stage:<18} {n}")
        first_ts = new_pods[0].split()[0] if new_pods else "-"
        last_ts = new_pods[-1].split()[0] if new_pods else "-"
        print(f"first pod NEW: {first_ts}")
        print(f"last  pod NEW: {last_ts}")
    else:
        print(f"({POD_LOG} not found)")

    # Memory / CPU snapshot
    print()
    print("--- Per-stage resource usage (warm only) ---")
    print(f"{'stage':<18} {'rss_kb (mean)':>15} {'peak_kb (mean)':>16} {'cpu_user_ms':>13}")
    for stage, srows in by_stage.items():
        warm = [r for r in srows if r["cold_like"] == "False"]
        if not warm:
            continue
        rss = [to_float(r["mem_rss_kb"]) for r in warm]
        rss = [v for v in rss if v is not None]
        peak = [to_float(r["mem_peak_kb"]) for r in warm]
        peak = [v for v in peak if v is not None]
        cpu = [to_float(r["cpu_user_ms"]) for r in warm]
        cpu = [v for v in cpu if v is not None]
        def m(xs):
            return f"{statistics.mean(xs):.1f}" if xs else "—"
        print(f"{stage:<18} {m(rss):>15} {m(peak):>16} {m(cpu):>13}")


if __name__ == "__main__":
    main()
