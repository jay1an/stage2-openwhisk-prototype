import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_TARGETS = {"p90": 0.98, "p95": 0.99}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-facing tables and figures for the Stage-2 continuous_moderate slice."
    )
    parser.add_argument(
        "--stage-summary",
        default="data/stage_forecasts/rolling/h1_countcalib_summary/summary.csv",
    )
    parser.add_argument(
        "--heldout",
        default=(
            "data/stage_forecasts/rolling/"
            "h1_countcalib_selector_continuous_holdout_robust_m003/"
            "heldout_test_result.csv"
        ),
    )
    parser.add_argument(
        "--validation-selection",
        default=(
            "data/stage_forecasts/rolling/"
            "h1_countcalib_selector_continuous_holdout_robust_m003/"
            "validation_selection.csv"
        ),
    )
    parser.add_argument(
        "--regime-sanity",
        default="data/entry_forecasts/rolling/regime_sanity/trace_regime_sanity_summary.csv",
    )
    parser.add_argument(
        "--characterization",
        default="data/analysis/azure_scaled30_2h_trace_set/trace_set_characterization.csv",
    )
    parser.add_argument("--out-dir", default="reports/stage2_continuous_moderate")
    return parser.parse_args()


def method_label(row: pd.Series) -> str:
    method = str(row["method"])
    method = method.replace("independent-", "ind-")
    method = method.replace("dag-twostage-", "dag-2s-")
    method = method.replace("dag-", "dag-")
    calib = str(row.get("count_calibration", "none"))
    if calib and calib != "none":
        method = f"{method}+{calib}"
    return method


def select_compact_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "policy",
        "method_family",
        "count_calibration",
        "method",
        "demand_coverage_rate",
        "allocated_replica_seconds",
        "over_allocation_ratio",
        "allocation_utilization",
        "under_total",
        "over_total",
    ]
    return frame[[col for col in cols if col in frame.columns]].copy()


def build_frontier_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, target in DEFAULT_TARGETS.items():
        group = summary[summary["policy"] == policy].copy()
        group = group.sort_values(
            ["allocated_replica_seconds", "over_allocation_ratio", "demand_coverage_rate"],
            ascending=[True, True, False],
        )
        best_coverage = -1.0
        best_over = 1e18
        for _, row in group.iterrows():
            coverage = float(row["demand_coverage_rate"])
            over = float(row["over_allocation_ratio"])
            if coverage > best_coverage or over < best_over:
                out = row.to_dict()
                out["target_coverage"] = target
                out["meets_target"] = coverage >= target
                rows.append(out)
                best_coverage = max(best_coverage, coverage)
                best_over = min(best_over, over)
    return pd.DataFrame(rows)


def build_best_feasible_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, target in DEFAULT_TARGETS.items():
        group = summary[summary["policy"] == policy].copy()
        feasible = group[group["demand_coverage_rate"] >= target].copy()
        if feasible.empty:
            chosen = group.sort_values(
                ["demand_coverage_rate", "allocated_replica_seconds"],
                ascending=[False, True],
            ).iloc[0].copy()
            reason = "fallback-max-coverage"
        else:
            chosen = feasible.sort_values(
                ["allocated_replica_seconds", "over_allocation_ratio"],
                ascending=[True, True],
            ).iloc[0].copy()
            reason = "min-replica-seconds-among-feasible"
        chosen["target_coverage"] = target
        chosen["selection_reason"] = reason
        rows.append(chosen)
    return pd.DataFrame(rows)


def plot_frontier(summary: pd.DataFrame, heldout: pd.DataFrame, out_dir: Path) -> None:
    colors = {
        "entry-heuristic-dag": "#2b6cb0",
        "entry-two-stage-dag": "#9f7aea",
        "per-stage-independent": "#2f855a",
    }
    for policy, target in DEFAULT_TARGETS.items():
        group = summary[summary["policy"] == policy].copy()
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
        for family, sub in group.groupby("method_family"):
            ax.scatter(
                sub["allocated_replica_seconds"],
                sub["demand_coverage_rate"],
                s=52,
                alpha=0.85,
                label=family,
                color=colors.get(family, "#4a5568"),
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    method_label(row),
                    (row["allocated_replica_seconds"], row["demand_coverage_rate"]),
                    fontsize=6,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
        selected = heldout[heldout["policy"] == policy]
        if not selected.empty:
            row = selected.iloc[0]
            ax.scatter(
                [row["allocated_replica_seconds"]],
                [row["demand_coverage_rate"]],
                marker="*",
                s=220,
                color="#dd6b20",
                label="held-out selected",
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
            )
        ax.axhline(target, color="#c53030", linestyle="--", linewidth=1.2, label=f"target={target:.2f}")
        ax.set_title(f"Stage-level coverage-cost frontier ({policy})")
        ax.set_xlabel("Allocated replica-seconds")
        ax.set_ylabel("Demand coverage")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / f"coverage_cost_frontier_{policy}.png")
        plt.close(fig)


def plot_over_allocation(summary: pd.DataFrame, out_dir: Path) -> None:
    compact = summary[summary["policy"].isin(DEFAULT_TARGETS)].copy()
    compact = compact.sort_values(["policy", "over_allocation_ratio"])
    labels = [f"{row.policy}\\n{method_label(row)}" for _, row in compact.iterrows()]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.35), 5.8), dpi=180)
    ax.bar(range(len(compact)), compact["over_allocation_ratio"], color="#4c78a8", alpha=0.85)
    ax.set_xticks(range(len(compact)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("Over-allocation ratio")
    ax.set_title("Over-allocation comparison across p90/p95 candidates")
    ax.grid(True, axis="y", linestyle=":", alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "over_allocation_candidates.png")
    plt.close(fig)


def write_report(
    out_dir: Path,
    characterization: pd.DataFrame,
    regime_sanity: pd.DataFrame,
    best_feasible: pd.DataFrame,
    heldout: pd.DataFrame,
) -> None:
    continuous = characterization[characterization["trace_type"] == "continuous_moderate"]
    continuous_row = continuous.iloc[0].to_dict() if not continuous.empty else {}
    lines = [
        "# Stage 2 Continuous Moderate Result Pack",
        "",
        "## Scope",
        "",
        "- Workflow: `sebs_video`.",
        "- Arrival source: Azure-derived `continuous_moderate`.",
        "- Time compression: one Azure minute to two seconds.",
        "- Evaluation window: 5 seconds.",
        "- Main held-out protocol: validation folds 0,1,2 and held-out fold 3.",
        "- This is an offline/synthetic-stage result pack, not a fresh OpenWhisk replay.",
        "",
        "## Trace Characterization",
        "",
        f"- 5s active ratio: {continuous_row.get('scaled_5s_active_ratio', float('nan')):.4f}.",
        f"- 5s mean count: {continuous_row.get('scaled_5s_mean_all', float('nan')):.4f}.",
        f"- 5s p95 count: {continuous_row.get('scaled_5s_p95', float('nan')):.4f}.",
        f"- 5s max count: {continuous_row.get('scaled_5s_max', float('nan')):.4f}.",
        f"- 5s coefficient of variation: {continuous_row.get('scaled_5s_cv', float('nan')):.4f}.",
        "",
        "## Regime Sanity Check",
        "",
        "Only `continuous_moderate` meets usable p90/p95 coverage targets with the current forecaster.",
        "Sparse, bursty, and mixed/drift are kept as limitations or future work for now.",
        "",
        regime_sanity.to_markdown(index=False),
        "",
        "## Best Feasible Candidates on All Rolling Folds",
        "",
        select_compact_columns(best_feasible).to_markdown(index=False),
        "",
        "## Robust Held-out Selection",
        "",
        select_compact_columns(heldout).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- The current Stage-2 slice is solid enough to freeze as the main forecasting experiment for now.",
        "- It is not the final proposal-level Workflow Forecastor across all workload regimes.",
        "- The strongest honest claim is coverage-cost aware model selection under a continuous moderate regime.",
        "- Do not claim that DAG propagation always wins; strict p95 targets can prefer per-stage independent baselines.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.stage_summary)
    heldout = pd.read_csv(args.heldout)
    validation_selection = pd.read_csv(args.validation_selection)
    regime_sanity = pd.read_csv(args.regime_sanity)
    characterization = pd.read_csv(args.characterization)

    p90_p95_summary = summary[summary["policy"].isin(DEFAULT_TARGETS)].copy()
    p90_p95_summary["label"] = p90_p95_summary.apply(method_label, axis=1)
    frontier = build_frontier_table(p90_p95_summary)
    best_feasible = build_best_feasible_table(p90_p95_summary)

    p90_p95_summary.to_csv(out_dir / "stage_candidate_summary_p90_p95.csv", index=False)
    frontier.to_csv(out_dir / "coverage_cost_frontier.csv", index=False)
    best_feasible.to_csv(out_dir / "best_feasible_by_policy.csv", index=False)
    heldout.to_csv(out_dir / "robust_heldout_selection.csv", index=False)
    validation_selection.to_csv(out_dir / "robust_validation_selection.csv", index=False)
    regime_sanity.to_csv(out_dir / "regime_sanity_summary.csv", index=False)

    plot_frontier(p90_p95_summary, heldout, out_dir)
    plot_over_allocation(p90_p95_summary, out_dir)
    write_report(out_dir, characterization, regime_sanity, best_feasible, heldout)

    metadata = {
        "stage_summary": args.stage_summary,
        "heldout": args.heldout,
        "validation_selection": args.validation_selection,
        "regime_sanity": args.regime_sanity,
        "characterization": args.characterization,
        "targets": DEFAULT_TARGETS,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote {out_dir / 'README.md'}")
    print(f"wrote {out_dir / 'coverage_cost_frontier_p90.png'}")
    print(f"wrote {out_dir / 'coverage_cost_frontier_p95.png'}")
    print(f"wrote {out_dir / 'over_allocation_candidates.png'}")
    print(select_compact_columns(heldout).to_string(index=False))


if __name__ == "__main__":
    main()

