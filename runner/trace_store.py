import csv
from pathlib import Path
from typing import Dict, Iterable


TRACE_COLUMNS = [
    "workflow_name",
    "request_id",
    "stage_name",
    "parent_stages",
    "slo_class",
    "entry_ts_ms",
    "workflow_start_ms",
    "workflow_end_ms",
    "workflow_e2e_ms",
    "dispatch_start_ms",
    "dispatch_end_ms",
    "dispatch_latency_ms",
    "action_start_ns",
    "action_end_ns",
    "action_duration_ms",
    "platform_overhead_ms",
    "container_id",
    "container_invocation_index",
    "container_uptime_ms",
    "previous_action_end_ns",
    "idle_since_prev_ms",
    "cold_like",
    "pod_name",
    "activation_id",
    "action_version",
    "ow_cold_start",
    "ow_memory_mb",
    "allocated_memory_mb",
    "allocated_cpu_cores",
    "detected_cpu_cores",
    "ow_wait_ms",
    "ow_init_ms",
    "ow_duration_ms",
    "ow_runtime_overhead_ms",
    "client_gateway_overhead_ms",
    "workload_mode",
    "serial_cpu_iters",
    "parallel_cpu_iters",
    "io_wait_ms",
    "parallel_workers",
    "parallel_workers_used",
    "serial_wall_ms",
    "io_wall_ms",
    "parallel_wall_ms",
    "memory_wall_ms",
    "cpu_user_ms",
    "cpu_system_ms",
    "cpu_self_ms",
    "cpu_self_process_ms",
    "cpu_children_user_ms",
    "cpu_children_system_ms",
    "cpu_children_ms",
    "cpu_process_ms",
    "cpu_total_ms",
    "parallel_cpu_ms",
    "observed_effective_cores",
    "observed_parallel_cores",
    "mem_rss_kb",
    "mem_peak_kb",
    "status",
    "error",
]


class CsvTraceStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = TRACE_COLUMNS
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
            if existing_header:
                self.fieldnames = existing_header
        else:
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def append_many(self, rows: Iterable[Dict[str, object]]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            for row in rows:
                normalized = {key: row.get(key, "") for key in self.fieldnames}
                writer.writerow(normalized)
