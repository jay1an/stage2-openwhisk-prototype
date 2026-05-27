"""Uneven 2-minute burst test against civic_alert_flow.

Fires invocations at uneven precomputed timestamps over 120 s, fire-and-forget
(does not wait for the previous one to finish). Caps in-flight workflows so the
OpenWhisk userMemory pool (20 GB / 256 MiB-per-action) cannot be exhausted.
Each per-container concurrency stays at 1 (no `--concurrency`).
"""

import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from runner.openwhisk_client import OpenWhiskClient
from runner.run_workflow import run_one_workflow
from runner.trace_store import CsvTraceStore
from runner.workflow import load_workflow


APIHOST = "https://192.168.123.17:31001"
AUTH = "2de2c2ac-36cf-4770-91ec-e5e0101a6e1d:2f929ea9c9ffb1548a5edd9aa24a9b9946e6c65f220b0393ce1c80c1fe6ef9ec"
WORKFLOW_YAML = os.path.join(ROOT, "configs", "civic_alert_flow.yaml")
TRACE_PATH = os.path.join(ROOT, "data", "burst_civic_trace.csv")
TIMELINE_PATH = os.path.join(ROOT, "data", "burst_civic_timeline.log")

# 5 stages per civic invocation. 20 GB / 256 MiB ≈ 80 container slots.
# Cap in-flight workflows so 6 * 5 = 30 distinct containers max (~7.5 GiB).
MAX_IN_FLIGHT = 6
TOTAL_DURATION_S = 120.0

# Uneven schedule (seconds from start). Mix of bursts and quiet periods.
# Phase 1 (0..25 s):   5 calls (early burst)
# Phase 2 (25..55 s):  2 calls (quiet)
# Phase 3 (55..90 s):  6 calls (heaviest burst, 3 within 4 s)
# Phase 4 (90..120 s): 3 calls (trickle)
SCHEDULE = [
    1.0, 2.5, 5.0, 9.0, 18.0,
    32.0, 47.0,
    58.0, 60.0, 61.5, 70.0, 80.0, 88.0,
    95.0, 108.0, 117.0,
]


def main() -> None:
    workflow = load_workflow(WORKFLOW_YAML)
    client = OpenWhiskClient(apihost=APIHOST, auth=AUTH, namespace=workflow.namespace)

    if os.path.exists(TRACE_PATH):
        os.remove(TRACE_PATH)
    store = CsvTraceStore(TRACE_PATH)

    store_lock = threading.Lock()
    timeline_lock = threading.Lock()
    timeline_fp = open(TIMELINE_PATH, "w")

    in_flight = threading.Semaphore(MAX_IN_FLIGHT)
    t0 = time.time()

    def log(msg: str) -> None:
        line = f"[T+{time.time() - t0:7.3f}s] {msg}"
        with timeline_lock:
            timeline_fp.write(line + "\n")
            timeline_fp.flush()
        print(line, flush=True)

    def fire(idx: int, scheduled_offset: float) -> None:
        with in_flight:
            log(f"#{idx:02d} START (sched +{scheduled_offset:.2f}s, drift {time.time() - t0 - scheduled_offset:+.2f}s)")
            try:
                rows = run_one_workflow(workflow, client, max_workers=8)
                with store_lock:
                    store.append_many(rows)
                request_id = rows[0]["request_id"]
                # Summarize cold/warm counts inside this workflow's rows
                stage_rows = [r for r in rows if r["stage_name"] != "__entry__"]
                cold = sum(1 for r in stage_rows if str(r.get("cold_like")) == "True")
                log(f"#{idx:02d} END req={request_id} stages={len(stage_rows)} cold={cold}/{len(stage_rows)}")
            except Exception as exc:
                log(f"#{idx:02d} ERROR {exc}")

    log(f"==== burst test start, MAX_IN_FLIGHT={MAX_IN_FLIGHT}, total={len(SCHEDULE)} calls over {TOTAL_DURATION_S}s ====")
    with ThreadPoolExecutor(max_workers=len(SCHEDULE) + 2) as pool:
        for idx, offset in enumerate(SCHEDULE):
            # Sleep until scheduled offset
            now = time.time() - t0
            wait = offset - now
            if wait > 0:
                time.sleep(wait)
            log(f"#{idx:02d} SCHEDULED (offset {offset:.2f}s)")
            pool.submit(fire, idx, offset)
        # Wait for stragglers
        log("==== schedule exhausted, waiting for stragglers ====")
    log("==== burst test finished ====")
    timeline_fp.close()


if __name__ == "__main__":
    main()
