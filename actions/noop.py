import os
import time
import uuid


CONTAINER_ID = str(uuid.uuid4())
IS_FIRST_CALL = True


def main(args):
    global IS_FIRST_CALL

    stage_name = args.get("stage_name", "unknown")
    sleep_ms = int(args.get("sleep_ms", 0))
    payload = args.get("payload", {})

    start_ns = time.time_ns()
    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)
    end_ns = time.time_ns()

    was_cold_like = IS_FIRST_CALL
    IS_FIRST_CALL = False

    return {
        "workflow_name": args.get("workflow_name"),
        "request_id": args.get("request_id"),
        "entry_ts_ms": args.get("entry_ts_ms"),
        "stage_name": stage_name,
        "parent_stages": args.get("parent_stages", []),
        "container_id": CONTAINER_ID,
        "cold_like": was_cold_like,
        "pid": os.getpid(),
        "action_start_ns": start_ns,
        "action_end_ns": end_ns,
        "action_duration_ms": (end_ns - start_ns) / 1_000_000.0,
        "payload_size": len(str(payload)),
    }

