#!/usr/bin/env python3
"""Deploy resource-suffixed OpenWhisk action variants for a workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.workflow import load_workflow, with_action_suffix
from scripts.replay_civic_azure_schedule import auth_from_args, update_actions


DEFAULT_WORKFLOW = ROOT / "configs" / "civic_alert_flow.yaml"
DEFAULT_ACTION_FILE = ROOT / "actions" / "workflow_action.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create or update workflow action variants named like "
            "wf_stage_1280, wf_stage_2048, etc. Existing matching variants are skipped."
        )
    )
    parser.add_argument("--apihost", required=True)
    parser.add_argument(
        "--auth",
        default="",
        help="OpenWhisk AUTH; when omitted, read owdev-whisk.auth guest auth via kubectl.",
    )
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    parser.add_argument("--action-file", default=str(DEFAULT_ACTION_FILE))
    parser.add_argument("--kind", default="python:3")
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--wsk-cli", default="wsk")
    parser.add_argument(
        "--memory-tiers",
        nargs="+",
        type=int,
        required=True,
        metavar="MB",
        help="memory tiers to deploy as action suffixes, e.g. 512 1280 2048 2560",
    )
    parser.add_argument(
        "--force-deploy",
        action="store_true",
        help="update variants even when memory/timeout/kind/source already match",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auth = auth_from_args(args.auth)
    base_workflow = load_workflow(args.workflow)

    for memory_mb in args.memory_tiers:
        args.memory_mb = memory_mb
        suffix = f"_{memory_mb}"
        workflow = with_action_suffix(base_workflow, suffix)
        action_names = [node.action for node in workflow.nodes.values()]
        print(
            f"\n== memory={memory_mb}MiB suffix={suffix} "
            f"actions={len(dict.fromkeys(action_names))} ==",
            flush=True,
        )
        update_actions(args, auth, action_names)

    print("\nAction variants are ready.")


if __name__ == "__main__":
    main()
