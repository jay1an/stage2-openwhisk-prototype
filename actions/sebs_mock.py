import hashlib
import os
import re
import time
import uuid


CONTAINER_ID = str(uuid.uuid4())
IS_FIRST_CALL = True


DEFAULT_PROFILE = {
    "cpu_iters": 40_000,
    "memory_kb": 64,
    "memory_passes": 1,
    "memory_stride": 0,
    "output_items": 2,
}


PROFILES = {
    ("sebs_trip_booking", "reserve_hotel"): {"cpu_iters": 60_000, "memory_kb": 96, "output_items": 2},
    ("sebs_trip_booking", "reserve_rental"): {"cpu_iters": 65_000, "memory_kb": 96, "output_items": 2},
    ("sebs_trip_booking", "reserve_flight"): {"cpu_iters": 75_000, "memory_kb": 128, "output_items": 2},
    ("sebs_trip_booking", "confirm"): {"cpu_iters": 45_000, "memory_kb": 64, "output_items": 1},
    ("sebs_video", "decode"): {"cpu_iters": 150_000, "memory_kb": 512, "output_items": 4},
    ("sebs_video", "analyse"): {"cpu_iters": 180_000, "memory_kb": 384, "output_items": 2},
    ("sebs_video", "summarize"): {"cpu_iters": 80_000, "memory_kb": 192, "output_items": 1},
    ("sebs_map_reduce", "split"): {"cpu_iters": 70_000, "memory_kb": 128, "output_items": 5},
    ("sebs_map_reduce", "map"): {"cpu_iters": 130_000, "memory_kb": 256, "output_items": 2},
    ("sebs_map_reduce", "shuffle"): {"cpu_iters": 90_000, "memory_kb": 192, "output_items": 3},
    ("sebs_map_reduce", "reduce"): {"cpu_iters": 120_000, "memory_kb": 256, "output_items": 1},
    ("sebs_ml", "generate"): {"cpu_iters": 100_000, "memory_kb": 256, "output_items": 4},
    ("sebs_ml", "train"): {"cpu_iters": 260_000, "memory_kb": 768, "output_items": 1},
    ("civic_alert_flow", "detect_object"): {
        "cpu_iters": 4_800_000,
        "memory_kb": 8_192,
        "memory_passes": 2,
        "memory_stride": 256,
        "output_items": 8,
    },
    ("civic_alert_flow", "estimate_pose"): {
        "cpu_iters": 3_500_000,
        "memory_kb": 6_144,
        "memory_passes": 2,
        "memory_stride": 256,
        "output_items": 6,
    },
    ("civic_alert_flow", "match_face"): {
        "cpu_iters": 5_200_000,
        "memory_kb": 12_288,
        "memory_passes": 2,
        "memory_stride": 256,
        "output_items": 4,
    },
    ("civic_alert_flow", "classify_scene"): {
        "cpu_iters": 4_000_000,
        "memory_kb": 10_240,
        "memory_passes": 2,
        "memory_stride": 256,
        "output_items": 5,
    },
    ("civic_alert_flow", "translate_alert"): {
        "cpu_iters": 2_400_000,
        "memory_kb": 4_096,
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


def apply_profile_overrides(profile, args):
    out = dict(profile)
    for key in ["cpu_iters", "memory_kb", "memory_passes", "memory_stride", "output_items"]:
        value = args.get(key)
        if value not in (None, ""):
            out[key] = int(value)
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
    global IS_FIRST_CALL

    workflow_name = args.get("workflow_name", "unknown")
    request_id = args.get("request_id", "")
    stage_name = args.get("stage_name", "unknown")
    base_stage = normalize_stage(stage_name)
    start_ns = time.time_ns()
    if args.get("__warmup", False):
        was_cold_like = IS_FIRST_CALL
        IS_FIRST_CALL = False
        end_ns = time.time_ns()
        return {
            "workflow_name": workflow_name,
            "request_id": request_id,
            "stage_name": stage_name,
            "base_stage": base_stage,
            "container_id": CONTAINER_ID,
            "cold_like": was_cold_like,
            "pid": os.getpid(),
            "warmup": True,
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

    cpu_checksum = cpu_work(int(profile["cpu_iters"]), seed)
    memory_checksum = memory_work(
        int(profile["memory_kb"]),
        seed,
        int(profile.get("memory_passes", 1)),
        int(profile.get("memory_stride", 0)),
    )
    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)
    checksum = (cpu_checksum ^ memory_checksum) & 0xFFFFFFFF
    end_ns = time.time_ns()

    was_cold_like = IS_FIRST_CALL
    IS_FIRST_CALL = False

    return {
        "workflow_name": workflow_name,
        "request_id": request_id,
        "entry_ts_ms": args.get("entry_ts_ms"),
        "stage_name": stage_name,
        "base_stage": base_stage,
        "parent_stages": args.get("parent_stages", []),
        "container_id": CONTAINER_ID,
        "cold_like": was_cold_like,
        "pid": os.getpid(),
        "action_start_ns": start_ns,
        "action_end_ns": end_ns,
        "action_duration_ms": (end_ns - start_ns) / 1_000_000.0,
        "mock_profile": profile,
        "parent_count": len(parent_payload),
        "payload_size": len(str(payload)),
        "checksum": checksum,
        "items": build_output(stage_name, profile, checksum),
    }
