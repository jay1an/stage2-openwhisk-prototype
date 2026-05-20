# Runner Code Structure

This package is organized by research stage. New code should go into the
stage-specific subpackages below. The files directly under `runner/` are shared
runtime modules used by multiple stages.

## Shared Runtime

- `workflow.py`: DAG config parsing and workflow topology helpers.
- `openwhisk_client.py`: minimal OpenWhisk REST client.
- `trace_store.py`: CSV trace writer.
- `workload.py`: synthetic workload event generation.
- `run_workflow.py`: execute one workflow invocation.
- `run_workload.py`: closed-loop workload runner.
- `run_open_loop_workload.py`: open-loop workload runner.
- `replay_schedule.py`: replay an external arrival schedule on OpenWhisk.

## Stage 2: Workflow Forecastor

Directory: `runner/stage2_forecastor/`

Purpose:

- Azure-derived entry trace extraction and challenge trace generation.
- Entry arrival forecasting.
- DAG delay-kernel propagation from entry forecasts to stage demand.
- Per-stage independent forecasting baselines.
- Quantile LightGBM, LSTM, rolling conformal calibration, and online selector.
- Stage-2 report builders and forecast policy selection.

Representative entry points:

- `python -m runner.stage2_forecastor.build_azure_challenge_trace`
- `python -m runner.stage2_forecastor.compare_stage_lstm_rolling`
- `python -m runner.stage2_forecastor.compare_stage_lightgbm_quantile_rolling`
- `python -m runner.stage2_forecastor.analyze_forecast_calibration`
- `python -m runner.stage2_forecastor.online_adaptive_forecast_selector`
- `python -m runner.stage2_forecastor.build_stage2_forecastor_final_report`
- `python -m runner.stage2_forecastor.simulate_profiled_stage_trace`

## Stage 3: Latency Profiler

Directory: `runner/stage3_latency/`

Purpose:

- Build coarse warm/cold-like latency profiles from workflow traces.
- Export stage-level and workflow-level latency quantiles.
- Export empirical latency sample pools for Stage 4 Monte Carlo.
- Build pilot-calibrated augmented cold-like sample pools for sensitivity tests.

Representative entry points:

- `python -m runner.stage3_latency.profile_latency`
- `python -m runner.stage3_latency.augment_cold_latency_samples`

## Stage 4: Workflow SLO Risk Estimator

Directory: `runner/stage4_risk/`

Purpose:

- Combine Stage 2 stage allocation forecasts with Stage 3 latency samples.
- Run offline Monte Carlo simulations through the workflow DAG.
- Estimate `P(workflow_latency > SLO)`.
- Compare risk under different forecast policies and SLO thresholds.

Representative entry points:

- `python -m runner.stage4_risk.estimate_slo_risk`
- `python -m runner.stage4_risk.run_stage4_monte_carlo_suite`

`estimate_slo_risk` can now consume a Stage 5 control plan through
`--control-plan`, so risk can be evaluated from explicit `warm_count`,
`keepalive_ttl_sec`, and `memory_mb` decisions rather than only the older
forecast `allocated_count` column.

## Stage 5: Fast Warm Manager

Directory: `runner/stage5_control/`

Purpose:

- Prototype short-loop prewarming / keep-alive planning.
- Convert forecast and latency profiles into warm-count and priority hints.

Representative entry point:

- `python -m runner.stage5_control.plan_joint_control`
- `python -m runner.stage5_control.risk_budgeted_pareto_planner`
- `python -m runner.stage5_control.paper_baselines`
- `python -m runner.stage5_control.cost_model`
- `python -m runner.stage5_control.control_plan`

## Stage 6: Slow Resource Configurator

Directory: `runner/stage6_resource/`

Purpose:

- Prototype memory-tier and CPU-proxy characterization.
- Summarize resource-size latency/cost trade-offs.

Representative entry points:

- `python -m runner.stage6_resource.benchmark_cpu_scaling`
- `python -m runner.stage6_resource.summarize_memory_sweep`

## Command Rule

Use the stage-organized module path for experiment commands. For example:

```bash
python -m runner.stage4_risk.estimate_slo_risk --help
```
