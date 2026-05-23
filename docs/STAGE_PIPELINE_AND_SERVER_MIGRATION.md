# Stage Pipeline and Server Migration Notes

## Goal

This project is being built as a complete research prototype first, using
synthetic and pilot-calibrated data where real OpenWhisk data are not yet
available. Later, the same pipeline should run on a server with real OpenWhisk
replay traces.

## Data Classes

Keep data provenance explicit in every CSV whenever possible.

- `synthetic`: generated from a controlled model.
- `pilot_calibrated` or `augmented_cold`: simulated from small real pilot traces.
- `real_measured`: collected from actual OpenWhisk workflow executions.

Do not mix these labels in the paper narrative. Synthetic and pilot-calibrated
data are valid for system bring-up, sensitivity analysis, and ablation, but final
claims should be repeated with `real_measured` data when the cluster is stable.

## Current Offline Pipeline

### Stage 2: Workflow Forecastor

Input:

- workflow entry schedule or simulated stage trace
- workflow DAG config

Main outputs:

- stage-level forecast detail
- p90/p95 allocation detail
- calibration and coverage metrics

Important current report:

```bash
reports/stage2_workflow_forecastor_final
```

### Stage 3: Latency Profiler

Input:

- stage-level trace with `dispatch_latency_ms`, `platform_overhead_ms`,
  `action_duration_ms`, and `cold_like`

Main outputs:

- warm/cold latency profiles
- latency sample pools for Monte Carlo

Current augmented sample pool:

```bash
reports/stage3_latency_augmented_cold_visual_qa_flow_periodic_drift/latency_samples_for_monte_carlo_augmented.csv
```

### Stage 4: SLO Risk Estimator

Input:

- Stage-2 forecast detail
- Stage-3 latency sample pool
- workflow DAG config
- SLO target

Main outputs:

- workflow-level SLO violation probability
- per-request risk
- risk calibration table
- stage-level cold/critical-path contribution

Single scenario example:

```bash
# Purpose: estimate workflow-level p95 SLO risk using the online selected forecast.
python -m runner.stage4_risk.estimate_slo_risk \
  --workflow-config configs/visual_qa_flow.yaml \
  --trace data/stage_synthetic/visual_qa_flow_azure_periodic_drift_challenge_scaled30_stage_trace.csv \
  --forecast-detail reports/online_adaptive_selector_azure_periodic_drift_scaled30_riskbudget/online_selected_detail.csv \
  --latency-samples reports/stage3_latency_augmented_cold_visual_qa_flow_periodic_drift/latency_samples_for_monte_carlo_augmented.csv \
  --method online-adaptive-expert-bank \
  --policy p95 \
  --window-sec 5 \
  --slo-ms 2500 \
  --simulations-per-request 200 \
  --residual-cold-probability 0.01 \
  --out-dir reports/stage4_mc_risk_p95
```

Suite example:

```bash
# Purpose: run several SLO and residual cold-risk settings in one reproducible suite.
python -m runner.stage4_risk.run_stage4_monte_carlo_suite \
  --workflow-config configs/visual_qa_flow.yaml \
  --trace data/stage_synthetic/visual_qa_flow_azure_periodic_drift_challenge_scaled30_stage_trace.csv \
  --forecast-detail reports/online_adaptive_selector_azure_periodic_drift_scaled30_riskbudget/online_selected_detail.csv \
  --latency-samples reports/stage3_latency_augmented_cold_visual_qa_flow_periodic_drift/latency_samples_for_monte_carlo_augmented.csv \
  --method online-adaptive-expert-bank \
  --policies p90,p95 \
  --slo-ms 2000,2500,3000 \
  --residual-cold-probabilities 0,0.01,0.05 \
  --simulations-per-request 200 \
  --out-dir reports/stage4_mc_risk_suite
```

### Stage 5/6: Joint Control Planner

Input:

- Stage-2 forecast detail
- Stage-3 latency sample pool
- workflow DAG config
- workflow SLO

Main outputs:

- fast cold-start control plan: `prewarm_target`, `keepalive_sec`
- slow resource sizing plan: `selected_memory_mb`
- DAG slack-aware priority: `slack_ms`, `urgency_score`, `slack_priority_rank`

Example:

```bash
# Purpose: produce an offline joint control plan without invoking OpenWhisk.
python -m runner.stage5_control.plan_joint_control \
  --workflow-config configs/visual_qa_flow.yaml \
  --forecast-detail reports/online_adaptive_selector_azure_periodic_drift_scaled30_riskbudget/online_selected_detail.csv \
  --latency-samples reports/stage3_latency_augmented_cold_visual_qa_flow_periodic_drift/latency_samples_for_monte_carlo_augmented.csv \
  --method online-adaptive-expert-bank \
  --policy p95 \
  --slo-ms 2500 \
  --memory-tiers-mb 128,256,512,1024,2048 \
  --prewarm-safety 1.0 \
  --min-keepalive-sec 5 \
  --max-keepalive-sec 60 \
  --out-dir reports/stage5_6_joint_control_plan_p95
```

Important outputs:

```bash
reports/stage5_6_joint_control_plan_p95/control_plan.csv
reports/stage5_6_joint_control_plan_p95/stage_resource_plan.csv
reports/stage5_6_joint_control_plan_p95/dag_slack_profile.csv
reports/stage5_6_joint_control_plan_p95/apply_memory_plan_template.sh
reports/stage5_6_joint_control_plan_p95/warmup_invoke_template.sh
```

The planner intentionally does not execute `wsk` commands. Review generated
templates before applying them on the cluster.

## Server Migration Checklist

### 1. Environment

```bash
# Purpose: create an isolated Python environment on the server.
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If LightGBM or PyTorch is installed separately on the server, verify them:

```bash
# Purpose: verify optional forecasting dependencies.
python -c "import lightgbm as lgb; print('lightgbm', lgb.__version__)"
python -c "import torch; print('torch', torch.__version__)"
```

### 2. OpenWhisk Connectivity

```bash
# Purpose: read current guest auth after OpenWhisk rebuild/redeploy.
GUEST_AUTH="$(kubectl get secret owdev-whisk.auth -n openwhisk -o jsonpath='{.data.guest}' | base64 -d)"
APIHOST="https://MASTER_IP:31001"
```

```bash
# Purpose: verify OpenWhisk API is reachable.
wsk -i --apihost "$APIHOST" --auth "$GUEST_AUTH" action list
```

### 3. Deploy Actions

```bash
# Purpose: deploy profile-driven workflow actions (civic_alert / spoken_dialog / visual_qa).
./scripts/deploy_actions.sh
```

### 4. Real Trace Collection

Run only small validation first. Do not jump directly to burst or high-concurrency
replay.

```bash
# Purpose: run one workflow entry to validate action deployment and trace schema.
python -m runner.replay_schedule \
  --workflow configs/visual_qa_flow.yaml \
  --schedule data/azure_schedules_scaled/schedule_azure_periodic_drift_challenge_scaled30_visual_qa_flow.csv \
  --limit 1 \
  --time-scale 1 \
  --min-gap-ms 500 \
  --max-gap-ms 3000 \
  --max-inflight 1 \
  --stage-max-workers 2 \
  --invoke-timeout-sec 60 \
  --trace data/real_traces/server_probe_visual_qa_flow_l1.csv \
  --schedule-out data/real_schedules/server_probe_visual_qa_flow_l1.csv \
  --apihost "$APIHOST" \
  --auth "$GUEST_AUTH"
```

### 5. Replace Offline Samples

Once enough real traces exist, regenerate Stage-3 latency samples from
`real_measured` data and point Stage 4 to the new sample file. The Stage-4 Monte
Carlo code should not need structural changes.

### 6. Apply Control Decisions Later

The first deployable control path should remain simple:

- Fast loop: warm action containers by issuing lightweight `__warmup=true`
  invocations until the observed warm container count matches `prewarm_target`.
- Slow loop: apply selected memory tiers with `wsk action update ... --memory`.
- Slack-aware scheduler: when several ready DAG stages compete for local
  dispatch workers, dispatch lower-slack stages first.

The current planner emits the target state. A future online daemon can read
`control_plan.csv` or receive equivalent rows from the forecast/risk pipeline.

## OpenWhisk Resource Note

In the current deployment, action pods have explicit memory request/limit but no
explicit CPU request/limit. Therefore:

- Treat memory tier as the directly controllable resource.
- Treat CPU as an implicit or memory-correlated proxy unless the deployment is
  changed to set action CPU limits.
- Record action pod resources with:

```bash
# Purpose: inspect actual Kubernetes resource requests/limits for an action pod.
kubectl get pod -n openwhisk POD_NAME -o jsonpath='{.spec.containers[0].resources}{"\n"}'
```
