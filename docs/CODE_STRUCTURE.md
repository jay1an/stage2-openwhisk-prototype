# Code Structure By Research Stage

The active experiment code is under `runner/`. It is now organized by the
research stages in the proposal:

```text
runner/
  stage2_forecastor/   # entry forecasting, DAG propagation, calibration, selector
  stage3_latency/      # warm/cold-like latency profiler
  stage4_risk/         # workflow SLO risk Monte Carlo estimator
  stage5_control/      # fast prewarming / keep-alive planning prototype
  stage6_resource/     # slow memory/resource characterization prototype
  workflow.py          # shared DAG utilities
  openwhisk_client.py  # shared OpenWhisk client
  trace_store.py       # shared trace output
```

Files directly under `runner/` are shared runtime modules. Stage-specific
commands live inside the stage subpackages.

## Project Root Name

`stage2-openwhisk-prototype` is now a historical project-root name. It contains
Stage 2 through Stage 6 prototype code because many VM commands, notes, and
paths already point to this directory. Do not rename it casually; use the
stage-specific subdirectories inside it. A later cleanup can rename the root to
something like `serverless-dag-optimizer-prototype` once all VM paths and docs
are updated together.

## Stage Responsibilities

Stage 2, `runner/stage2_forecastor/`:

- Build Azure-derived workflow entry traces.
- Forecast workflow entry arrivals.
- Propagate entry forecasts through the fixed DAG into stage-level demand.
- Compare entry+DAG forecasting with per-stage independent baselines.
- Evaluate quantile calibration and online expert selection.

Stage 3, `runner/stage3_latency/`:

- Convert workflow traces into warm/cold-like latency profiles.
- Separate platform overhead and action duration at a coarse level.
- Export empirical latency samples for Stage 4.

Stage 4, `runner/stage4_risk/`:

- Combine Stage 2 allocation forecasts and Stage 3 latency samples.
- Simulate workflow latency through the DAG.
- Estimate SLO violation probability.
- Optionally evaluate an explicit Stage 5 control plan through
  `estimate_slo_risk --control-plan`.

Stage 5, `runner/stage5_control/`:

- Prototype fast prewarming and keep-alive planning.
- Normalize control plans with `control_plan.py`.
- Estimate GB-second proxy cost with `cost_model.py`.
- Search risk-budgeted Pareto plans with `risk_budgeted_pareto_planner.py`.
- Generate scale-to-zero, always-warm, ORION-style, and StepConf-style
  comparison plans with `paper_baselines.py`.
- This stage should eventually execute warmup invocations, but current code is
  still offline planning.

Stage 6, `runner/stage6_resource/`:

- Prototype memory-tier and resource-size profiling.
- This stage should eventually feed slow resource configuration decisions.

## Command Style

Use stage-specific paths:

```bash
# Stage 2 forecastor final report
python -m runner.stage2_forecastor.build_stage2_forecastor_final_report --help

# Stage 3 latency profile
python -m runner.stage3_latency.profile_latency --help

# Stage 4 SLO risk estimation
python -m runner.stage4_risk.estimate_slo_risk --help

# Stage 5 risk-budgeted Pareto planning
python -m runner.stage5_control.risk_budgeted_pareto_planner --help
```
