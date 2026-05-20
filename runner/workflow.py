from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


@dataclass(frozen=True)
class NodeSpec:
    name: str
    action: str
    parents: List[str]
    sleep_ms: int = 0
    cpu_iters: int | None = None
    memory_kb: int | None = None
    memory_passes: int | None = None
    memory_stride: int | None = None
    output_items: int | None = None
    warm_overhead_ms: float | None = None
    cold_overhead_ms: float | None = None


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_name: str
    namespace: str
    entry: str
    nodes: Dict[str, NodeSpec]

    def children_of(self, node_name: str) -> List[str]:
        return [
            node.name
            for node in self.nodes.values()
            if node_name in node.parents
        ]

    def ready_nodes(self, completed: Iterable[str], running: Iterable[str]) -> List[NodeSpec]:
        completed_set = set(completed)
        running_set = set(running)
        ready = []
        for node in self.nodes.values():
            if node.name in completed_set or node.name in running_set:
                continue
            if all(parent in completed_set for parent in node.parents):
                ready.append(node)
        return ready


def load_workflow(path: str) -> WorkflowSpec:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    nodes = {}
    for item in raw["nodes"]:
        node = NodeSpec(
            name=item["name"],
            action=item["action"],
            parents=list(item.get("parents", [])),
            sleep_ms=int(item.get("sleep_ms", 0)),
            cpu_iters=(
                int(item["cpu_iters"]) if item.get("cpu_iters") is not None else None
            ),
            memory_kb=(
                int(item["memory_kb"]) if item.get("memory_kb") is not None else None
            ),
            memory_passes=(
                int(item["memory_passes"]) if item.get("memory_passes") is not None else None
            ),
            memory_stride=(
                int(item["memory_stride"]) if item.get("memory_stride") is not None else None
            ),
            output_items=(
                int(item["output_items"]) if item.get("output_items") is not None else None
            ),
            warm_overhead_ms=(
                float(item["warm_overhead_ms"]) if item.get("warm_overhead_ms") is not None else None
            ),
            cold_overhead_ms=(
                float(item["cold_overhead_ms"]) if item.get("cold_overhead_ms") is not None else None
            ),
        )
        nodes[node.name] = node

    if raw["entry"] not in nodes:
        raise ValueError(f"entry node {raw['entry']} is not defined in nodes")

    return WorkflowSpec(
        workflow_name=raw["workflow_name"],
        namespace=raw.get("namespace", "guest"),
        entry=raw["entry"],
        nodes=nodes,
    )
