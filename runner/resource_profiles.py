from __future__ import annotations


HUAWEI_FUNCTIONGRAPH_CPU_PER_MEMORY_MB = 1.0 / 1280.0
LEGACY_256MB_250M_CPU_PER_MEMORY_MB = 1.0 / 1024.0


def memory_to_cpu_cores(
    memory_mb: float | int | None,
    *,
    profile: str = "huawei_functiongraph",
    cpu_per_memory_mb: float | None = None,
) -> float | None:
    if memory_mb in (None, ""):
        return None

    memory = float(memory_mb)
    if memory <= 0:
        return None

    normalized = str(profile or "huawei_functiongraph").strip().lower()
    if normalized in {"huawei", "huawei_functiongraph", "functiongraph"}:
        ratio = HUAWEI_FUNCTIONGRAPH_CPU_PER_MEMORY_MB
    elif normalized in {"legacy_256mb_250m", "openwhisk_256mb_250m"}:
        ratio = LEGACY_256MB_250M_CPU_PER_MEMORY_MB
    elif normalized == "custom":
        if cpu_per_memory_mb is None:
            raise ValueError("cpu_per_memory_mb is required when profile='custom'")
        ratio = float(cpu_per_memory_mb)
    else:
        raise ValueError(f"unsupported resource profile: {profile}")

    return memory * ratio
