"""Compare two ways of propagating an entry forecast to downstream stage counts.

Method A (current `propagate.py`): build a per-stage empirical delay kernel from
training stage observations, multiply each entry forecast row by the kernel
probability for every (offset, stage) pair, sum by (stage, window).

Method B (entry + Stage3 latency model): compute deterministic per-stage
offset from the entry node using `warm_overhead_ms`, `cold_overhead_ms`,
`cpu_iters`, `memory_kb` from the workflow YAML, propagate the entry forecast
by shifting whole-count to the stage's expected window.

Ground truth: actual per-stage per-window counts from the test portion of the
profiled stage trace.

For each (stage, method, policy), report:
- MAE / RMSE between allocated counts and actual counts
- coverage_rate (fraction of windows with allocated >= actual)
- demand_coverage_rate (sum allocated / sum actual)
- over_allocation_ratio (sum over / sum allocated)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..workflow import NodeSpec, WorkflowSpec, load_workflow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--workflow-config", required=True)
    p.add_argument("--stage-trace", required=True, help="full profiled stage trace CSV")
    p.add_argument("--entry-forecast-detail", required=True, help="entry rolling forecast detail CSV")
    p.add_argument("--train-ratio", type=float, default=0.5)
    p.add_argument("--window-ms", type=int, default=5000)
    p.add_argument("--method", default="hazard-hurdle", help="entry forecast method to evaluate")
    p.add_argument("--policy", default="p95")
    p.add_argument("--activation-threshold", type=float, default=0.1)
    p.add_argument("--cpu-iters-per-ms", type=float, default=8_000.0)
    p.add_argument("--memory-ops-per-ms", type=float, default=12_000.0)
    p.add_argument("--warm-slow-probability", type=float, default=0.07)
    p.add_argument("--warm-slow-multiplier", type=float, default=7.0)
    p.add_argument("--cold-probability", type=float, default=0.0,
                   help="probability a stage is cold-like when computing expected delay (default warm)")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def topo_order(workflow: WorkflowSpec) -> List[str]:
    indeg = {n: len(node.parents) for n, node in workflow.nodes.items()}
    children: Dict[str, List[str]] = defaultdict(list)
    for n, node in workflow.nodes.items():
        for parent in node.parents:
            children[parent].append(n)
    order: List[str] = []
    q = deque([n for n, d in indeg.items() if d == 0])
    while q:
        n = q.popleft()
        order.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(order) != len(workflow.nodes):
        raise ValueError("workflow has a cycle")
    return order


def expected_action_ms(node: NodeSpec, cpu_iters_per_ms: float, memory_ops_per_ms: float) -> float:
    cpu_iters = float(node.cpu_iters or 40_000.0)
    memory_kb = float(node.memory_kb or 64.0)
    memory_passes = float(node.memory_passes or 1.0)
    memory_stride = max(1.0, float(node.memory_stride or 256.0))
    cpu_ms = cpu_iters / max(1.0, cpu_iters_per_ms)
    memory_ops = memory_kb * 1024.0 / memory_stride * memory_passes
    memory_ms = memory_ops / max(1.0, memory_ops_per_ms)
    return max(1.0, cpu_ms + memory_ms)


def expected_overhead_ms(
    node: NodeSpec,
    cold_probability: float,
    warm_slow_probability: float,
    warm_slow_multiplier: float,
) -> float:
    warm = float(node.warm_overhead_ms or 50.0)
    cold = float(node.cold_overhead_ms or 1500.0)
    warm_mean = warm * ((1 - warm_slow_probability) * 1.0 + warm_slow_probability * warm_slow_multiplier)
    return cold_probability * cold + (1.0 - cold_probability) * warm_mean


def per_stage_expected_delay_ms(
    workflow: WorkflowSpec,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Expected delay (ms) from workflow entry to the *start* of each stage."""
    delays: Dict[str, float] = {}
    for name in topo_order(workflow):
        node = workflow.nodes[name]
        if not node.parents:
            delays[name] = 0.0
            continue
        # Stage starts after the latest parent completes
        parent_completion = []
        for p in node.parents:
            p_node = workflow.nodes[p]
            p_overhead = expected_overhead_ms(p_node, args.cold_probability,
                                              args.warm_slow_probability, args.warm_slow_multiplier)
            p_action = expected_action_ms(p_node, args.cpu_iters_per_ms, args.memory_ops_per_ms)
            parent_completion.append(delays[p] + p_overhead + p_action)
        delays[name] = max(parent_completion)
    return delays


def build_actual_stage_counts(
    stage_trace_path: Path,
    workflow_name: str,
    window_ms: int,
    train_until_ms: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(stage_trace_path)
    df = df[(df["workflow_name"] == workflow_name) & (df["status"] == "ok")].copy()
    df["entry_ts_ms"] = df["entry_ts_ms"].astype("int64")
    df["dispatch_start_ms"] = df["dispatch_start_ms"].astype("int64")
    df["window"] = (df["dispatch_start_ms"] // window_ms).astype("int64")
    train = df[df["entry_ts_ms"] <= train_until_ms].copy()
    test = df[df["entry_ts_ms"] > train_until_ms].copy()
    return train, test


def kernel_propagate(
    forecast: pd.DataFrame,
    train: pd.DataFrame,
    workflow: WorkflowSpec,
    window_ms: int,
    method: str,
    policy: str,
) -> pd.DataFrame:
    rows = []
    policy_col = f"forecast_count"
    actual_col = "actual_count"
    sub = forecast[(forecast["method"] == method) & (forecast["policy"] == policy)].copy()
    if sub.empty:
        return pd.DataFrame()
    for stage in workflow.nodes:
        stage_train = train[train["stage_name"] == stage]
        if stage_train.empty:
            continue
        delay_ms = stage_train["dispatch_start_ms"].astype(float) - stage_train["entry_ts_ms"].astype(float)
        offsets = np.maximum(0, np.floor(delay_ms / window_ms)).astype(int)
        counts = pd.Series(offsets).value_counts().sort_index()
        total = counts.sum()
        kernel = {int(o): float(c / total) for o, c in counts.items()}
        for _, fr in sub.iterrows():
            for off, prob in kernel.items():
                target_window = int(fr["target_window"]) + off
                rows.append({
                    "stage_name": stage,
                    "method": "kernel",
                    "target_window": target_window,
                    "allocated": float(fr["allocated_count"]) * prob,
                    "forecast": float(fr["forecast_count"]) * prob,
                })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby(["stage_name", "method", "target_window"], as_index=False)[["allocated", "forecast"]].sum()


def latency_propagate(
    forecast: pd.DataFrame,
    workflow: WorkflowSpec,
    window_ms: int,
    method: str,
    policy: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    sub = forecast[(forecast["method"] == method) & (forecast["policy"] == policy)].copy()
    if sub.empty:
        return pd.DataFrame()
    delays_ms = per_stage_expected_delay_ms(workflow, args)
    rows = []
    for stage, dly_ms in delays_ms.items():
        offset = int(math.floor(dly_ms / window_ms))
        for _, fr in sub.iterrows():
            rows.append({
                "stage_name": stage,
                "method": "latency",
                "target_window": int(fr["target_window"]) + offset,
                "allocated": float(fr["allocated_count"]),
                "forecast": float(fr["forecast_count"]),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby(["stage_name", "method", "target_window"], as_index=False)[["allocated", "forecast"]].sum()


def evaluate(
    propagation: pd.DataFrame,
    test: pd.DataFrame,
    workflow: WorkflowSpec,
    window_ms: int,
    activation_threshold: float,
) -> pd.DataFrame:
    if propagation.empty:
        return pd.DataFrame()
    actual = (
        test.groupby(["stage_name", "window"])
        .size()
        .rename("actual_count")
        .reset_index()
    )
    test_window_min = int(test["window"].min())
    test_window_max = int(test["window"].max())
    rows = []
    for stage in workflow.nodes:
        prop_s = propagation[propagation["stage_name"] == stage].copy()
        if prop_s.empty:
            continue
        prop_s = prop_s[(prop_s["target_window"] >= test_window_min) & (prop_s["target_window"] <= test_window_max)].copy()
        actual_s = actual[actual["stage_name"] == stage].copy().rename(columns={"window": "target_window"})
        full_windows = pd.DataFrame({"target_window": range(test_window_min, test_window_max + 1)})
        m = full_windows.merge(prop_s, on="target_window", how="left").merge(actual_s, on="target_window", how="left")
        m["allocated"] = m["allocated"].fillna(0.0)
        m["forecast"] = m["forecast"].fillna(0.0)
        m["actual_count"] = m["actual_count"].fillna(0).astype(int)
        m["alloc_ceil"] = m["allocated"].apply(lambda v: int(math.ceil(max(0.0, v))) if v >= activation_threshold else 0)
        # If propagation has multiple methods rows for this stage, treat each separately
        for method_label, g in m.groupby("method", dropna=True):
            actual_vec = g["actual_count"].astype(float)
            alloc_vec = g["alloc_ceil"].astype(float)
            rows.append({
                "stage_name": stage,
                "method": method_label,
                "n_windows": int(len(g)),
                "actual_sum": int(actual_vec.sum()),
                "actual_max": int(actual_vec.max() if len(actual_vec) else 0),
                "alloc_sum": int(alloc_vec.sum()),
                "alloc_max": int(alloc_vec.max() if len(alloc_vec) else 0),
                "mae": float((alloc_vec - actual_vec).abs().mean()),
                "rmse": float(np.sqrt(((alloc_vec - actual_vec) ** 2).mean())),
                "coverage_rate": float((alloc_vec >= actual_vec).mean()),
                "demand_coverage_rate": float(min(1.0, alloc_vec.sum() / max(1.0, actual_vec.sum()))),
                "over_sum": float(np.maximum(0.0, alloc_vec - actual_vec).sum()),
                "under_sum": float(np.maximum(0.0, actual_vec - alloc_vec).sum()),
                "over_allocation_ratio": float(
                    np.maximum(0.0, alloc_vec - actual_vec).sum() / max(1.0, alloc_vec.sum())
                ),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    workflow = load_workflow(args.workflow_config)
    workflow_name = workflow.workflow_name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    forecast = pd.read_csv(args.entry_forecast_detail)
    # train cutoff from the entry-forecast origins: use the smallest test target_window
    sub_method = forecast[(forecast["method"] == args.method) & (forecast["policy"] == args.policy)]
    if sub_method.empty:
        raise SystemExit(f"no rows for method={args.method} policy={args.policy}")
    train_until_window = int(sub_method["target_window"].min()) - 1
    train_until_ms = train_until_window * args.window_ms

    train, test = build_actual_stage_counts(Path(args.stage_trace), workflow_name, args.window_ms, train_until_ms)
    if test.empty:
        raise SystemExit("no test rows in stage trace")

    propagation_k = kernel_propagate(forecast, train, workflow, args.window_ms, args.method, args.policy)
    propagation_l = latency_propagate(forecast, workflow, args.window_ms, args.method, args.policy, args)
    propagation = pd.concat([propagation_k, propagation_l], ignore_index=True)
    propagation.to_csv(out_dir / "propagation_predictions.csv", index=False)

    eval_df = evaluate(propagation, test, workflow, args.window_ms, args.activation_threshold)
    eval_df.to_csv(out_dir / "propagation_eval.csv", index=False)

    delays_ms = per_stage_expected_delay_ms(workflow, args)
    meta = {
        "workflow_name": workflow_name,
        "workflow_config": args.workflow_config,
        "stage_trace": args.stage_trace,
        "entry_forecast_detail": args.entry_forecast_detail,
        "method": args.method,
        "policy": args.policy,
        "window_ms": args.window_ms,
        "train_until_window": train_until_window,
        "train_until_ms": train_until_ms,
        "expected_stage_delay_ms": delays_ms,
    }
    (out_dir / "propagation_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if eval_df.empty:
        print("EMPTY eval")
        return
    pivot_mae = eval_df.pivot(index="stage_name", columns="method", values="mae")
    pivot_over = eval_df.pivot(index="stage_name", columns="method", values="over_allocation_ratio")
    pivot_cov = eval_df.pivot(index="stage_name", columns="method", values="demand_coverage_rate")
    print(f"\n[{workflow_name}] method={args.method} policy={args.policy}")
    print("\n=== MAE per stage ===")
    print(pivot_mae.to_string())
    print("\n=== over_allocation_ratio ===")
    print(pivot_over.to_string())
    print("\n=== demand_coverage_rate ===")
    print(pivot_cov.to_string())


if __name__ == "__main__":
    main()
