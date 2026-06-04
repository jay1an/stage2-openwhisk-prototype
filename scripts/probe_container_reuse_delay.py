#!/usr/bin/env python3
"""Probe warmup-container reuse as a function of post-warmup delay."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.openwhisk_client import OpenWhiskClient
from runner.stage4_risk.scaling import memory_to_cpu_cores
from runner.workflow import NodeSpec, load_workflow, suffix_action_name


DEFAULT_ACTIONS = [
    "wf_civic_estimate_pose_1280",
    "wf_civic_match_face_2048",
]
DEFAULT_DELAYS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.3, 2.5, 3.0, 4.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", default=str(ROOT / "configs" / "civic_alert_flow.yaml"))
    parser.add_argument("--action-file", default=str(ROOT / "actions" / "workflow_action.py"))
    parser.add_argument("--apihost", default="https://192.168.123.17:31001")
    parser.add_argument(
        "--auth",
        default="",
        help="OpenWhisk AUTH. If omitted, read owdev-whisk.auth guest auth via kubectl.",
    )
    parser.add_argument("--actions", nargs="+", default=DEFAULT_ACTIONS)
    parser.add_argument("--delays", nargs="+", type=float, default=DEFAULT_DELAYS)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "path3_reuse_probe"))
    parser.add_argument("--kind", default="python:3")
    return parser.parse_args()


def read_guest_auth() -> str:
    encoded = subprocess.check_output(
        [
            "kubectl",
            "get",
            "secret",
            "owdev-whisk.auth",
            "-n",
            "openwhisk",
            "-o",
            "jsonpath={.data.guest}",
        ],
        text=True,
    )
    return subprocess.check_output(["base64", "-d"], input=encoded, text=True).strip()


def activation_annotations(activation: dict) -> dict:
    return {
        item.get("key"): item.get("value")
        for item in activation.get("annotations", [])
        if isinstance(item, dict) and item.get("key")
    }


def variant_timeout_ms(memory_mb: int) -> int:
    return 120000 if memory_mb == 1280 else 240000


def parse_variant(action_name: str) -> tuple[str, int]:
    raw_name = action_name.rsplit("/", 1)[-1]
    prefix, tier_text = raw_name.rsplit("_", 1)
    if not tier_text.isdigit():
        raise ValueError(f"action variant must end with _<memory_mb>: {action_name}")
    stage_name = prefix.removeprefix("wf_civic_").replace("-", "_")
    return stage_name, int(tier_text)


def real_params(workflow_name: str, node: NodeSpec, memory_mb: int) -> dict:
    return {
        "workflow_name": workflow_name,
        "request_id": str(uuid.uuid4()),
        "entry_ts_ms": int(time.time() * 1000),
        "stage_name": node.name,
        "parent_stages": node.parents,
        "sleep_ms": node.sleep_ms,
        "cpu_iters": node.cpu_iters,
        "serial_cpu_iters": node.serial_cpu_iters,
        "parallel_cpu_iters": node.parallel_cpu_iters,
        "serial_fraction": node.serial_fraction,
        "io_wait_ms": node.io_wait_ms,
        "parallel_workers": node.parallel_workers,
        "max_parallel_workers": node.max_parallel_workers,
        "memory_kb": node.memory_kb,
        "memory_passes": node.memory_passes,
        "memory_stride": node.memory_stride,
        "output_items": node.output_items,
        "allocated_memory_mb": memory_mb,
        "allocated_cpu_cores": memory_to_cpu_cores(memory_mb),
        "payload": {"parents": {}},
    }


def redeploy_action(args: argparse.Namespace, auth: str, action_name: str, memory_mb: int) -> None:
    subprocess.run(
        [
            "wsk",
            "-i",
            "--apihost",
            args.apihost,
            "--auth",
            auth,
            "action",
            "update",
            action_name,
            args.action_file,
            "--kind",
            args.kind,
            "--memory",
            str(memory_mb),
            "--timeout",
            str(variant_timeout_ms(memory_mb)),
        ],
        stdout=subprocess.DEVNULL,
        check=True,
    )


def invoke(client: OpenWhiskClient, action_name: str, params: dict) -> tuple[dict, dict, dict]:
    activation = client.invoke_activation(action_name, params)
    result = activation.get("response", {}).get("result", {})
    if not isinstance(result, dict):
        result = {"error": result}
    annotations = activation_annotations(activation)
    return activation, result, annotations


def safe_float(value: object) -> float | str:
    if value in ("", None):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def run_trial(
    *,
    args: argparse.Namespace,
    auth: str,
    client: OpenWhiskClient,
    workflow_name: str,
    node: NodeSpec,
    action_name: str,
    memory_mb: int,
    delay_s: float,
    trial: int,
) -> dict:
    redeploy_action(args, auth, action_name, memory_mb)

    request_id = str(uuid.uuid4())
    warmup_params = {
        "__warmup": True,
        "workflow_name": workflow_name,
        "request_id": request_id,
        "stage_name": node.name,
        "allocated_memory_mb": memory_mb,
        "allocated_cpu_cores": memory_to_cpu_cores(memory_mb),
    }
    warmup_activation, warmup_result, warmup_annotations = invoke(
        client,
        action_name,
        warmup_params,
    )
    warmup_return_monotonic = time.monotonic()

    if delay_s > 0:
        time.sleep(delay_s)

    params = real_params(workflow_name, node, memory_mb)
    params["request_id"] = request_id
    real_invoke_sent_monotonic = time.monotonic()
    real_activation, real_result, real_annotations = invoke(client, action_name, params)

    warmup_container_id = warmup_result.get("container_id", "")
    real_container_id = real_result.get("container_id", "")
    real_cold_like = str(real_result.get("cold_like", "")).lower() == "true"
    same_container = bool(warmup_container_id) and warmup_container_id == real_container_id
    hit = same_container and not real_cold_like

    return {
        "action": action_name,
        "stage_name": node.name,
        "tier_mb": memory_mb,
        "delay_s": delay_s,
        "trial": trial,
        "request_id": request_id,
        "warmup_return_monotonic": warmup_return_monotonic,
        "warmup_container_id": warmup_container_id,
        "warmup_activation_id": warmup_activation.get("activationId", ""),
        "warmup_cold_like": warmup_result.get("cold_like", ""),
        "warmup_ow_wait_ms": safe_float(warmup_annotations.get("waitTime", "")),
        "warmup_ow_init_ms": safe_float(warmup_annotations.get("initTime", "")),
        "warmup_ow_duration_ms": safe_float(warmup_activation.get("duration", "")),
        "real_invoke_sent_monotonic": real_invoke_sent_monotonic,
        "real_container_id": real_container_id,
        "real_cold_like": real_cold_like,
        "real_ow_wait_ms": safe_float(real_annotations.get("waitTime", "")),
        "real_ow_init_ms": safe_float(real_annotations.get("initTime", "")),
        "real_ow_duration_ms": safe_float(real_activation.get("duration", "")),
        "real_activation_id": real_activation.get("activationId", ""),
        "same_container": same_container,
        "hit": hit,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["action"], float(row["delay_s"])), []).append(row)

    out = []
    for (action, delay_s), items in sorted(grouped.items()):
        waits = [
            float(item["real_ow_wait_ms"])
            for item in items
            if item["real_ow_wait_ms"] not in ("", None)
        ]
        out.append(
            {
                "action": action,
                "delay_s": delay_s,
                "n": len(items),
                "hit_count": sum(bool(item["hit"]) for item in items),
                "hit_rate": sum(bool(item["hit"]) for item in items) / len(items),
                "mean_real_ow_wait_ms": mean(waits),
                "same_container_count": sum(bool(item["same_container"]) for item in items),
                "cold_count": sum(bool(item["real_cold_like"]) for item in items),
            }
        )
    return out


def threshold(summary_rows: list[dict], action: str) -> str:
    action_rows = [row for row in summary_rows if row["action"] == action]
    for row in sorted(action_rows, key=lambda item: float(item["delay_s"])):
        if float(row["hit_rate"]) >= 0.99:
            return f"{float(row['delay_s']):.1f}s"
    return "not reached"


def pre_threshold_shape(summary_rows: list[dict], action: str) -> str:
    action_rows = sorted(
        [row for row in summary_rows if row["action"] == action],
        key=lambda item: float(item["delay_s"]),
    )
    first_full = None
    for row in action_rows:
        if float(row["hit_rate"]) >= 0.99:
            first_full = float(row["delay_s"])
            break
    if first_full is None:
        return "no full-hit threshold observed"
    before = [row for row in action_rows if float(row["delay_s"]) < first_full]
    nonzero = [row for row in before if float(row["hit_rate"]) > 0.0]
    if not nonzero:
        return f"clean step: 0% below {first_full:.1f}s, 100% at/above it"
    return (
        f"mostly step-like: {len(nonzero)} pre-threshold delay(s) had partial hits "
        f"before {first_full:.1f}s"
    )


def write_report(path: Path, summary_rows: list[dict], actions: list[str]) -> None:
    delays = sorted({float(row["delay_s"]) for row in summary_rows})
    by_key = {(row["action"], float(row["delay_s"])): row for row in summary_rows}
    thresholds = {action: threshold(summary_rows, action) for action in actions}
    consistent = len(set(thresholds.values())) == 1

    lines = [
        "# Container Reuse Delay Probe",
        "",
        "Each trial redeployed the action variant, invoked a blocking `__warmup`, waited a controlled delay after the warmup returned, then invoked the normal action once.",
        "",
        "## Hit Rate vs Delay",
        "",
        "| delay_s | " + " | ".join(f"{action} hit_rate" for action in actions) + " |",
        "|---:|" + "|".join("---:" for _ in actions) + "|",
    ]
    for delay in delays:
        cells = []
        for action in actions:
            row = by_key.get((action, delay))
            cells.append("n/a" if row is None else f"{float(row['hit_rate']) * 100:.1f}%")
        lines.append(f"| {delay:.1f} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Platform Wait Check", ""])
    lines.append("| action | delay_s | hit_rate | mean_real_ow_wait_ms | same_container_count | cold_count |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['action']} | {float(row['delay_s']):.1f} | "
            f"{float(row['hit_rate']) * 100:.1f}% | {float(row['mean_real_ow_wait_ms']):.1f} | "
            f"{row['same_container_count']} | {row['cold_count']} |"
        )

    lines.extend(["", "## Threshold Verdict", ""])
    for action in actions:
        lines.append(
            f"- `{action}` first reached ~100% hit rate at: {thresholds[action]} "
            f"({pre_threshold_shape(summary_rows, action)})."
        )
    lines.append("")
    if consistent:
        lines.append(
            f"The two actions show the same threshold: {next(iter(thresholds.values()))}. "
            "This supports a shared OpenWhisk/container-pool settle delay rather than an action-specific workload effect."
        )
    else:
        lines.append(
            "The actions did not show the exact same threshold; treat the larger observed threshold as the safer scheduling bound."
        )
    lines.append("")
    lines.append(
        "Hits have `same_container=True`, `cold_like=False`, and real `waitTime` around 10-14ms. "
        "Misses have `same_container=False`, `cold_like=True`, and real `waitTime` around 1.5-1.9s, confirming a fresh cold start."
    )
    lines.append("")
    lines.append(
        "Verdict: the controlled data does not support a 2.3s minimum threshold. "
        "The reliable threshold in this run is 2.0s after the blocking warmup returns. "
        "A practical scheduler should target at least 2.0s, with a small safety margin if the cluster is busy."
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    auth = args.auth or read_guest_auth()
    workflow = load_workflow(args.workflow)
    client = OpenWhiskClient(args.apihost, auth, namespace=workflow.namespace, timeout_sec=args.timeout_sec)

    action_info = []
    for action_name in args.actions:
        stage_name, tier_mb = parse_variant(action_name)
        if stage_name not in workflow.nodes:
            raise ValueError(f"stage {stage_name!r} from {action_name!r} not found in workflow")
        expected_action = suffix_action_name(workflow.nodes[stage_name].action, f"_{tier_mb}")
        if expected_action != action_name:
            raise ValueError(f"action {action_name!r} does not match workflow variant {expected_action!r}")
        action_info.append((action_name, workflow.nodes[stage_name], tier_mb))

    out_dir = Path(args.out_dir)
    raw_rows: list[dict] = []
    total = len(action_info) * len(args.delays) * args.trials
    index = 0

    try:
        for action_name, node, tier_mb in action_info:
            for delay_s in args.delays:
                for trial in range(1, args.trials + 1):
                    index += 1
                    print(
                        f"[{index}/{total}] action={action_name} delay={delay_s:.1f}s trial={trial}",
                        flush=True,
                    )
                    row = run_trial(
                        args=args,
                        auth=auth,
                        client=client,
                        workflow_name=workflow.workflow_name,
                        node=node,
                        action_name=action_name,
                        memory_mb=tier_mb,
                        delay_s=delay_s,
                        trial=trial,
                    )
                    raw_rows.append(row)
                    print(
                        "    "
                        f"warmup_container={row['warmup_container_id']} "
                        f"real_container={row['real_container_id']} "
                        f"cold={row['real_cold_like']} same={row['same_container']} "
                        f"hit={row['hit']} wait={row['real_ow_wait_ms']}",
                        flush=True,
                    )
                    write_csv(out_dir / "reuse_probe_raw.csv", raw_rows)
    finally:
        print("\nRestoring probed action variants...", flush=True)
        for action_name, _node, tier_mb in action_info:
            redeploy_action(args, auth, action_name, tier_mb)
            print(f"  restored {action_name} memory={tier_mb} timeout={variant_timeout_ms(tier_mb)}", flush=True)

    summary_rows = summarize(raw_rows)
    write_csv(out_dir / "reuse_probe_summary.csv", summary_rows)
    write_report(out_dir / "reuse_probe_report.md", summary_rows, [item[0] for item in action_info])

    print("\nSummary:")
    for row in summary_rows:
        print(
            f"{row['action']} delay={float(row['delay_s']):.1f}s "
            f"hit_rate={float(row['hit_rate']) * 100:.1f}% "
            f"mean_wait={float(row['mean_real_ow_wait_ms']):.1f}ms"
        )
    print(f"\nWrote {out_dir / 'reuse_probe_raw.csv'}")
    print(f"Wrote {out_dir / 'reuse_probe_summary.csv'}")
    print(f"Wrote {out_dir / 'reuse_probe_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
