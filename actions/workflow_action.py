import hashlib
import math
import os
import re
import resource
import time
import uuid
from concurrent.futures import ProcessPoolExecutor


CONTAINER_ID = str(uuid.uuid4())
CONTAINER_CREATED_NS = time.time_ns()
IS_FIRST_CALL = True
LAST_ACTION_END_NS = None
CONTAINER_INVOCATION_COUNT = 0


DEFAULT_PROFILE = {
    "cpu_iters": 40_000,
    "serial_fraction": 0.25,
    "io_wait_ms": 0,
    "parallel_workers": "auto",
    "max_parallel_workers": 8,
    "memory_kb": 64,
    "memory_passes": 1,
    "memory_stride": 0,
    "output_items": 2,
}


PROFILES = {
    ("civic_alert_flow", "detect_object"): {
        "cpu_iters": 21_000_000,
        "serial_fraction": 0.20,
        "io_wait_ms": 800,
        "memory_kb": 16_384,
        "memory_passes": 2,
        "memory_stride": 256,
        "output_items": 8,
    },
    ("civic_alert_flow", "estimate_pose"): {
        "cpu_iters": 18_800_000,
        "serial_fraction": 0.25,
        "io_wait_ms": 700,
        "memory_kb": 12_288,
        "memory_passes": 2,
        "memory_stride": 256,
        "output_items": 6,
    },
    ("civic_alert_flow", "match_face"): {
        "cpu_iters": 24_100_000,
        "serial_fraction": 0.25,
        "io_wait_ms": 900,
        "memory_kb": 24_576,
        "memory_passes": 2,
        "memory_stride": 256,
        "output_items": 4,
    },
    ("civic_alert_flow", "classify_scene"): {
        "cpu_iters": 20_800_000,
        "serial_fraction": 0.20,
        "io_wait_ms": 800,
        "memory_kb": 16_384,
        "memory_passes": 2,
        "memory_stride": 256,
        "output_items": 5,
    },
    ("civic_alert_flow", "translate_alert"): {
        "cpu_iters": 19_300_000,
        "serial_fraction": 0.35,
        "io_wait_ms": 750,
        "memory_kb": 8_192,
        "memory_passes": 1,
        "memory_stride": 256,
        "output_items": 2,
    },
    ("visual_qa_flow", "image_embed"): {
        "cpu_iters": 3_600_000,
        "memory_kb": 8_192,
        "memory_passes": 2,
        "memory_stride": 256,
        "output_items": 8,
    },
    ("visual_qa_flow", "text_embed"): {
        "cpu_iters": 2_800_000,
        "memory_kb": 4_096,
        "memory_passes": 1,
        "memory_stride": 256,
        "output_items": 4,
    },
    ("visual_qa_flow", "answer_question"): {
        "cpu_iters": 4_200_000,
        "memory_kb": 6_144,
        "memory_passes": 2,
        "memory_stride": 256,
        "output_items": 2,
    },
    ("spoken_dialog_flow", "speech_decode"): {
        "cpu_iters": 4_000_000,
        "memory_kb": 6_144,
        "memory_passes": 2,
        "memory_stride": 256,
        "output_items": 6,
    },
    ("spoken_dialog_flow", "topic_route"): {
        "cpu_iters": 1_800_000,
        "memory_kb": 3_072,
        "memory_passes": 1,
        "memory_stride": 256,
        "output_items": 3,
    },
    ("spoken_dialog_flow", "entity_extract"): {
        "cpu_iters": 2_600_000,
        "memory_kb": 4_096,
        "memory_passes": 1,
        "memory_stride": 256,
        "output_items": 4,
    },
    ("spoken_dialog_flow", "response_generate"): {
        "cpu_iters": 4_800_000,
        "memory_kb": 8_192,
        "memory_passes": 2,
        "memory_stride": 256,
        "output_items": 3,
    },
    ("spoken_dialog_flow", "speech_synthesize"): {
        "cpu_iters": 3_600_000,
        "memory_kb": 6_144,
        "memory_passes": 2,
        "memory_stride": 256,
        "output_items": 2,
    },
}


def normalize_stage(stage_name):
    """Collapse statically expanded map stages such as map_0 into map."""
    return re.sub(r"_\d+$", "", str(stage_name))


def stable_seed(*parts):
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def cpu_work(iterations, seed):
    value = seed & 0xFFFFFFFF
    for idx in range(iterations):
        value = (value * 1664525 + 1013904223 + idx) & 0xFFFFFFFF
    return value


def memory_work(memory_kb, seed, passes=1, stride=0):
    size = max(0, int(memory_kb)) * 1024
    if size == 0:
        return 0
    data = bytearray(size)
    step = max(1, int(stride) if int(stride or 0) > 0 else size // 64)
    checksum = 0
    for pass_id in range(max(1, int(passes))):
        for idx in range(0, size, step):
            data[idx] = (seed + idx + pass_id) & 0xFF
            checksum = (checksum + data[idx]) & 0xFFFFFFFF
    return checksum


def optional_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_int(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def detect_cgroup_cpu_cores():
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as fh:
            quota_text, period_text = fh.read().strip().split()[:2]
        if quota_text != "max":
            quota = float(quota_text)
            period = float(period_text)
            if quota > 0 and period > 0:
                return quota / period
    except (OSError, ValueError):
        pass

    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="utf-8") as fh:
            quota = float(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="utf-8") as fh:
            period = float(fh.read().strip())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass

    return None


def detect_pod_name():
    hostname = os.environ.get("HOSTNAME", "")
    if hostname:
        return hostname
    try:
        with open("/etc/hostname", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def resolve_allocated_cpu_cores(args):
    explicit = optional_float(args.get("allocated_cpu_cores"))
    if explicit is not None and explicit > 0:
        return explicit

    memory_mb = optional_float(args.get("allocated_memory_mb"))
    if memory_mb is not None and memory_mb > 0:
        return memory_mb / 1280.0

    detected = detect_cgroup_cpu_cores()
    if detected is not None and detected > 0:
        return detected

    return None


def resolve_parallel_workers(profile, allocated_cpu_cores):
    max_workers = optional_int(profile.get("max_parallel_workers")) or 8
    max_workers = max(1, max_workers)

    requested = profile.get("parallel_workers", "auto")
    if requested in (None, "", "auto"):
        workers = (
            max(1, math.ceil(allocated_cpu_cores))
            if allocated_cpu_cores is not None
            else 1
        )
    else:
        workers = optional_int(requested) or 1

    return min(max_workers, max(1, workers))


def split_iterations(iterations, workers):
    iterations = max(0, int(iterations))
    workers = max(1, int(workers))
    if iterations == 0:
        return []
    active_workers = min(workers, iterations)
    base = iterations // active_workers
    remainder = iterations % active_workers
    return [
        base + (1 if idx < remainder else 0)
        for idx in range(active_workers)
    ]


def parallel_cpu_work(iterations, seed, workers):
    chunks = split_iterations(iterations, workers)
    if not chunks:
        return 0, 0
    if len(chunks) == 1:
        return cpu_work(chunks[0], seed), 1

    checksums = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [
            pool.submit(cpu_work, chunk, seed + idx * 104729)
            for idx, chunk in enumerate(chunks)
        ]
        for future in futures:
            checksums.append(future.result())

    checksum = 0
    for value in checksums:
        checksum ^= value
    return checksum, len(chunks)


def resolve_cpu_plan(profile):
    cpu_iters = max(0, int(profile.get("cpu_iters", 0)))
    serial_raw = optional_int(profile.get("serial_cpu_iters"))
    parallel_raw = optional_int(profile.get("parallel_cpu_iters"))

    if serial_raw is None and parallel_raw is None:
        serial_fraction = optional_float(profile.get("serial_fraction"))
        if serial_fraction is None:
            serial_fraction = 0.25
        serial_fraction = min(1.0, max(0.0, serial_fraction))
        serial_iters = int(cpu_iters * serial_fraction)
        parallel_iters = max(0, cpu_iters - serial_iters)
    elif serial_raw is None:
        parallel_iters = max(0, int(parallel_raw))
        serial_iters = max(0, cpu_iters - parallel_iters)
    elif parallel_raw is None:
        serial_iters = max(0, int(serial_raw))
        parallel_iters = max(0, cpu_iters - serial_iters)
    else:
        serial_iters = max(0, int(serial_raw))
        parallel_iters = max(0, int(parallel_raw))

    return serial_iters, parallel_iters


def apply_profile_overrides(profile, args):
    out = dict(profile)
    int_keys = [
        "cpu_iters",
        "serial_cpu_iters",
        "parallel_cpu_iters",
        "io_wait_ms",
        "max_parallel_workers",
        "memory_kb",
        "memory_passes",
        "memory_stride",
        "output_items",
    ]
    for key in int_keys:
        value = args.get(key)
        if value not in (None, ""):
            out[key] = int(value)
    for key in ["serial_fraction"]:
        value = args.get(key)
        if value not in (None, ""):
            out[key] = float(value)
    for key in ["parallel_workers"]:
        value = args.get(key)
        if value not in (None, ""):
            out[key] = value
    return out


def build_output(stage_name, profile, checksum):
    output_items = int(profile.get("output_items", 1))
    return [
        {
            "item_id": f"{stage_name}-{idx}",
            "score": (checksum + idx * 17) % 1000,
        }
        for idx in range(output_items)
    ]


def main(args):
    global CONTAINER_INVOCATION_COUNT, IS_FIRST_CALL, LAST_ACTION_END_NS

    workflow_name = args.get("workflow_name", "unknown")
    request_id = args.get("request_id", "")
    stage_name = args.get("stage_name", "unknown")
    base_stage = normalize_stage(stage_name)
    pod_name = detect_pod_name()
    start_ns = time.time_ns()
    previous_action_end_ns = LAST_ACTION_END_NS
    idle_since_prev_ms = (
        ""
        if previous_action_end_ns is None
        else (start_ns - previous_action_end_ns) / 1_000_000.0
    )
    CONTAINER_INVOCATION_COUNT += 1
    container_invocation_index = CONTAINER_INVOCATION_COUNT
    container_uptime_ms = (start_ns - CONTAINER_CREATED_NS) / 1_000_000.0
    if args.get("__warmup", False):
        was_cold_like = IS_FIRST_CALL
        IS_FIRST_CALL = False
        warmup_hold_ms = int(float(args.get("warmup_hold_ms", 0) or 0))
        if warmup_hold_ms > 0:
            time.sleep(warmup_hold_ms / 1000.0)
        end_ns = time.time_ns()
        LAST_ACTION_END_NS = end_ns
        return {
            "workflow_name": workflow_name,
            "request_id": request_id,
            "stage_name": stage_name,
            "base_stage": base_stage,
            "container_id": CONTAINER_ID,
            "container_invocation_index": container_invocation_index,
            "container_uptime_ms": container_uptime_ms,
            "previous_action_end_ns": previous_action_end_ns or "",
            "idle_since_prev_ms": idle_since_prev_ms,
            "cold_like": was_cold_like,
            "pod_name": pod_name,
            "pid": os.getpid(),
            "warmup": True,
            "warmed": True,
            "warmup_hold_ms": warmup_hold_ms,
            "action_start_ns": start_ns,
            "action_end_ns": end_ns,
            "action_duration_ms": (end_ns - start_ns) / 1_000_000.0,
        }

    sleep_ms = int(args.get("sleep_ms", 0))
    payload = args.get("payload", {})
    parent_payload = payload.get("parents", {}) if isinstance(payload, dict) else {}

    profile = apply_profile_overrides(
        PROFILES.get((workflow_name, base_stage), DEFAULT_PROFILE),
        args,
    )
    seed = stable_seed(workflow_name, request_id, stage_name)
    allocated_memory_mb = optional_int(args.get("allocated_memory_mb"))
    allocated_cpu_cores = resolve_allocated_cpu_cores(args)
    detected_cpu_cores = detect_cgroup_cpu_cores()
    serial_cpu_iters, parallel_cpu_iters = resolve_cpu_plan(profile)
    parallel_workers = resolve_parallel_workers(profile, allocated_cpu_cores)
    io_wait_ms = int(profile.get("io_wait_ms", 0) or 0)
    if io_wait_ms <= 0 and sleep_ms > 0:
        io_wait_ms = sleep_ms

    ru_before = resource.getrusage(resource.RUSAGE_SELF)
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    pt_before = time.process_time_ns()

    serial_start_ns = time.time_ns()
    serial_checksum = cpu_work(serial_cpu_iters, seed)
    serial_end_ns = time.time_ns()

    io_start_ns = time.time_ns()
    if io_wait_ms > 0:
        time.sleep(io_wait_ms / 1000.0)
    io_end_ns = time.time_ns()

    parallel_child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    parallel_start_ns = time.time_ns()
    parallel_checksum, parallel_workers_used = parallel_cpu_work(
        parallel_cpu_iters,
        seed ^ 0xA5A5A5A5,
        parallel_workers,
    )
    parallel_end_ns = time.time_ns()
    parallel_child_after = resource.getrusage(resource.RUSAGE_CHILDREN)

    memory_start_ns = time.time_ns()
    memory_checksum = memory_work(
        int(profile["memory_kb"]),
        seed,
        int(profile.get("memory_passes", 1)),
        int(profile.get("memory_stride", 0)),
    )
    memory_end_ns = time.time_ns()

    pt_after = time.process_time_ns()
    ru_after = resource.getrusage(resource.RUSAGE_SELF)
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    checksum = (serial_checksum ^ parallel_checksum ^ memory_checksum) & 0xFFFFFFFF
    end_ns = time.time_ns()

    mem_rss_kb = ""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    mem_rss_kb = int(line.split()[1])
                    break
    except OSError:
        pass

    action_duration_ms = (end_ns - start_ns) / 1_000_000.0
    serial_wall_ms = (serial_end_ns - serial_start_ns) / 1_000_000.0
    io_wall_ms = (io_end_ns - io_start_ns) / 1_000_000.0
    parallel_wall_ms = (parallel_end_ns - parallel_start_ns) / 1_000_000.0
    memory_wall_ms = (memory_end_ns - memory_start_ns) / 1_000_000.0

    cpu_user_ms = (ru_after.ru_utime - ru_before.ru_utime) * 1000.0
    cpu_system_ms = (ru_after.ru_stime - ru_before.ru_stime) * 1000.0
    cpu_self_ms = cpu_user_ms + cpu_system_ms
    cpu_self_process_ms = (pt_after - pt_before) / 1_000_000.0
    cpu_children_user_ms = (child_after.ru_utime - child_before.ru_utime) * 1000.0
    cpu_children_system_ms = (child_after.ru_stime - child_before.ru_stime) * 1000.0
    cpu_children_ms = cpu_children_user_ms + cpu_children_system_ms
    cpu_total_ms = cpu_self_ms + cpu_children_ms
    cpu_process_ms = cpu_total_ms
    parallel_children_user_ms = (
        parallel_child_after.ru_utime - parallel_child_before.ru_utime
    ) * 1000.0
    parallel_children_system_ms = (
        parallel_child_after.ru_stime - parallel_child_before.ru_stime
    ) * 1000.0
    parallel_cpu_ms = parallel_children_user_ms + parallel_children_system_ms
    observed_effective_cores = (
        cpu_total_ms / action_duration_ms if action_duration_ms > 0 else ""
    )
    observed_parallel_cores = (
        parallel_cpu_ms / parallel_wall_ms if parallel_wall_ms > 0 else ""
    )
    mem_peak_kb = ru_after.ru_maxrss

    was_cold_like = IS_FIRST_CALL
    IS_FIRST_CALL = False
    LAST_ACTION_END_NS = end_ns

    return {
        "workflow_name": workflow_name,
        "request_id": request_id,
        "entry_ts_ms": args.get("entry_ts_ms"),
        "stage_name": stage_name,
        "base_stage": base_stage,
        "parent_stages": args.get("parent_stages", []),
        "container_id": CONTAINER_ID,
        "container_invocation_index": container_invocation_index,
        "container_uptime_ms": container_uptime_ms,
        "previous_action_end_ns": previous_action_end_ns or "",
        "idle_since_prev_ms": idle_since_prev_ms,
        "cold_like": was_cold_like,
        "pod_name": pod_name,
        "pid": os.getpid(),
        "action_start_ns": start_ns,
        "action_end_ns": end_ns,
        "action_duration_ms": action_duration_ms,
        "workload_mode": "mixed_serial_io_parallel",
        "allocated_memory_mb": allocated_memory_mb or "",
        "allocated_cpu_cores": allocated_cpu_cores or "",
        "detected_cpu_cores": detected_cpu_cores or "",
        "serial_cpu_iters": serial_cpu_iters,
        "parallel_cpu_iters": parallel_cpu_iters,
        "io_wait_ms": io_wait_ms,
        "parallel_workers": parallel_workers,
        "parallel_workers_used": parallel_workers_used,
        "serial_wall_ms": serial_wall_ms,
        "io_wall_ms": io_wall_ms,
        "parallel_wall_ms": parallel_wall_ms,
        "memory_wall_ms": memory_wall_ms,
        "cpu_user_ms": cpu_user_ms,
        "cpu_system_ms": cpu_system_ms,
        "cpu_self_ms": cpu_self_ms,
        "cpu_self_process_ms": cpu_self_process_ms,
        "cpu_children_user_ms": cpu_children_user_ms,
        "cpu_children_system_ms": cpu_children_system_ms,
        "cpu_children_ms": cpu_children_ms,
        "cpu_process_ms": cpu_process_ms,
        "cpu_total_ms": cpu_total_ms,
        "parallel_cpu_ms": parallel_cpu_ms,
        "observed_effective_cores": observed_effective_cores,
        "observed_parallel_cores": observed_parallel_cores,
        "mem_rss_kb": mem_rss_kb,
        "mem_peak_kb": mem_peak_kb,
        "mock_profile": profile,
        "parent_count": len(parent_payload),
        "payload_size": len(str(payload)),
        "checksum": checksum,
        "items": build_output(stage_name, profile, checksum),
    }
