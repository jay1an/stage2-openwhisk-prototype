import csv
from pathlib import Path
from typing import Dict, Iterable


TRACE_COLUMNS = [
    "workflow_name",
    "request_id",
    "stage_name",
    "parent_stages",
    "entry_ts_ms",
    "dispatch_start_ms",
    "dispatch_end_ms",
    "dispatch_latency_ms",
    "action_start_ns",
    "action_end_ns",
    "action_duration_ms",
    "platform_overhead_ms",
    "container_id",
    "cold_like",
    "status",
    "error",
]


class CsvTraceStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=TRACE_COLUMNS)
                writer.writeheader()

    def append_many(self, rows: Iterable[Dict[str, object]]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRACE_COLUMNS)
            for row in rows:
                normalized = {key: row.get(key, "") for key in TRACE_COLUMNS}
                writer.writerow(normalized)
