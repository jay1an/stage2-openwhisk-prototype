import os
import time
import uuid


CONTAINER_ID = str(uuid.uuid4())
IS_FIRST_CALL = True


def cpu_work(iterations, seed):
    value = seed & 0xFFFFFFFF
    for idx in range(iterations):
        value = (value * 1664525 + 1013904223 + idx) & 0xFFFFFFFF
    return value


def main(args):
    global IS_FIRST_CALL

    iterations = int(args.get("iterations", 8_000_000))
    repeat = int(args.get("repeat", 1))
    request_id = args.get("request_id", "")
    stage_name = args.get("stage_name", "cpu_probe")

    wall_start_ns = time.time_ns()
    process_start_ns = time.process_time_ns()
    checksum = 0
    for item in range(max(1, repeat)):
        checksum ^= cpu_work(iterations, item + len(str(request_id)))
    process_end_ns = time.process_time_ns()
    wall_end_ns = time.time_ns()

    was_cold_like = IS_FIRST_CALL
    IS_FIRST_CALL = False

    return {
        "workflow_name": args.get("workflow_name", "cpu_probe"),
        "request_id": request_id,
        "stage_name": stage_name,
        "container_id": CONTAINER_ID,
        "cold_like": was_cold_like,
        "pid": os.getpid(),
        "iterations": iterations,
        "repeat": repeat,
        "checksum": checksum,
        "action_start_ns": wall_start_ns,
        "action_end_ns": wall_end_ns,
        "action_duration_ms": (wall_end_ns - wall_start_ns) / 1_000_000.0,
        "process_cpu_ms": (process_end_ns - process_start_ns) / 1_000_000.0,
    }
