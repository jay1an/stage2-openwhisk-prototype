import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .estimate_slo_risk import run_estimation


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in parse_csv_list(value)]


def slugify(value: str) -> str:
    value = value.replace(".", "p")
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value)
    return value.strip("-").lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Stage-4 Monte Carlo SLO-risk suite across policies, SLOs, "
            "and residual cold-like probabilities."
        )
    )
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--forecast-detail", required=True)
    parser.add_argument("--latency-samples", required=True)
    parser.add_argument("--method", default="online-adaptive-expert-bank")
    parser.add_argument("--policies", default="p90,p95")
    parser.add_argument("--fold-id", type=int, default=None)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--slo-ms", default="2000,2500,3000")
    parser.add_argument("--residual-cold-probabilities", default="0,0.01,0.05")
    parser.add_argument("--simulations-per-request", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--risk-bins", type=int, default=10)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def write_readme(out_dir: Path, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    best = summary.sort_values(
        ["policy", "slo_ms", "residual_cold_probability", "predicted_violation_probability"]
    )
    lines = [
        "# Stage 4 Monte Carlo SLO-Risk Suite",
        "",
        "## Scope",
        "",
        "- Runs the offline Monte Carlo risk estimator over multiple risk scenarios.",
        "- Intended to connect Stage-2 probabilistic allocation forecasts and Stage-3 warm/cold latency samples.",
        "- Current latency samples may be synthetic or pilot-calibrated; keep `sample_origin` in the latency CSV for traceability.",
        "",
        "## Inputs",
        "",
        f"- Workflow config: `{args.workflow_config}`",
        f"- Stage trace: `{args.trace}`",
        f"- Forecast detail: `{args.forecast_detail}`",
        f"- Latency samples: `{args.latency_samples}`",
        f"- Method: `{args.method}`",
        f"- Policies: `{args.policies}`",
        f"- SLOs: `{args.slo_ms}`",
        f"- Residual cold probabilities: `{args.residual_cold_probabilities}`",
        "",
        "## Outputs",
        "",
        "- `risk_suite_summary.csv`: all scenario summaries.",
        "- `metadata.json`: reproducibility metadata.",
        "- One subdirectory per scenario with request-level risk, calibration bins, and stage contribution.",
        "",
        "## Quick View",
        "",
        "```text",
        best.to_string(index=False) if not best.empty else "No scenario summary rows.",
        "```",
        "",
        "## Interpretation Guardrail",
        "",
        "This suite is meant to validate the complete risk-estimation pipeline before full real-platform data are available. "
        "Do not present pilot-calibrated or synthetic latency samples as final real OpenWhisk measurements.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent.parent.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = parse_csv_list(args.policies)
    slo_values = parse_float_list(args.slo_ms)
    residual_values = parse_float_list(args.residual_cold_probabilities)

    rows = []
    scenario_index = 0
    for policy in policies:
        for slo_ms in slo_values:
            for residual in residual_values:
                scenario_index += 1
                scenario_id = (
                    f"{scenario_index:03d}_"
                    f"{slugify(args.method)}_{policy}_"
                    f"slo{slugify(str(slo_ms))}_cold{slugify(str(residual))}"
                )
                scenario_out = out_dir / scenario_id
                scenario_args = argparse.Namespace(
                    workflow_config=args.workflow_config,
                    trace=args.trace,
                    forecast_detail=args.forecast_detail,
                    latency_samples=args.latency_samples,
                    method=args.method,
                    policy=policy,
                    fold_id=args.fold_id,
                    window_sec=args.window_sec,
                    slo_ms=slo_ms,
                    simulations_per_request=args.simulations_per_request,
                    seed=args.seed + scenario_index,
                    out_dir=str(scenario_out),
                    residual_cold_probability=residual,
                    risk_bins=args.risk_bins,
                    write_stage_samples=False,
                )
                summary = run_estimation(scenario_args).copy()
                calibration_path = scenario_out / "risk_calibration_summary.csv"
                if calibration_path.exists():
                    calibration = pd.read_csv(calibration_path)
                    if not calibration.empty:
                        for col in [
                            "brier_score",
                            "expected_calibration_error",
                            "max_calibration_error",
                        ]:
                            summary[col] = calibration[col].iloc[0]
                summary.insert(0, "scenario_id", scenario_id)
                summary["residual_cold_probability"] = residual
                rows.append(summary)

    suite_summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    suite_summary.to_csv(out_dir / "risk_suite_summary.csv", index=False)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_config": args.workflow_config,
        "trace": args.trace,
        "forecast_detail": args.forecast_detail,
        "latency_samples": args.latency_samples,
        "method": args.method,
        "policies": policies,
        "slo_ms": slo_values,
        "residual_cold_probabilities": residual_values,
        "simulations_per_request": args.simulations_per_request,
        "risk_bins": args.risk_bins,
        "seed": args.seed,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_readme(out_dir, suite_summary, args)
    print(f"wrote {out_dir}")
    if not suite_summary.empty:
        print(suite_summary.to_string(index=False))


if __name__ == "__main__":
    main()

