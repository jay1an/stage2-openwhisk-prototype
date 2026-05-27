#!/usr/bin/env bash
set -euo pipefail

# Run this script on the VM that can reach your OpenWhisk API host.
# It updates all actions referenced by one workflow to each memory tier,
# executes the workflow repeatedly, and builds latency profiles per tier.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Workflow under test. Must be one of the supported DAGs:
#   configs/civic_alert_flow.yaml
#   configs/spoken_dialog_flow.yaml
#   configs/visual_qa_flow.yaml
# Override via WORKFLOW=configs/<name>.yaml.
WORKFLOW="${WORKFLOW:-configs/civic_alert_flow.yaml}"

# OpenWhisk endpoint and auth. APIHOST MUST be set for your real cluster, e.g.
#   APIHOST=https://<your-openwhisk-host>:31001
# AUTH is normally read from the cluster's Kubernetes secret below.
APIHOST="${APIHOST:?APIHOST must be set, e.g. https://<openwhisk-host>:31001}"
AUTH="${AUTH:-}"
if [[ -z "$AUTH" ]]; then
  # Read the current OpenWhisk guest auth from the Kubernetes secret.
  AUTH="$(kubectl get secret owdev-whisk.auth -n openwhisk -o jsonpath='{.data.guest}' | base64 -d)"
fi

# OpenWhisk runtime/action settings.
OW_KIND="${OW_KIND:-python:3}"
ACTION_FILE="${ACTION_FILE:-actions/workflow_action.py}"
ACTION_TIMEOUT_MS="${ACTION_TIMEOUT_MS:-60000}"
PYTHON="${PYTHON:-python3}"
WSK_CLI="${WSK_CLI:-wsk}"
WSK_ARGS=(-i --apihost "$APIHOST" --auth "$AUTH")

# Experiment knobs.
MEMORY_TIERS="${MEMORY_TIERS:-128 256 512 1024}"
COLD_COUNT="${COLD_COUNT:-3}"
WARM_COUNT="${WARM_COUNT:-25}"
COLD_INTERVAL_MS="${COLD_INTERVAL_MS:-1000}"
WARM_INTERVAL_MS="${WARM_INTERVAL_MS:-250}"
MAX_WORKERS="${MAX_WORKERS:-8}"
CPU_PROFILE="${CPU_PROFILE:-huawei_functiongraph}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TRACE_DIR="${TRACE_DIR:-data/traces/memory_sweep_${RUN_ID}}"
OUT_DIR="${OUT_DIR:-reports/memory_sweep_${RUN_ID}}"
mkdir -p "$TRACE_DIR" "$OUT_DIR"

workflow_name="$(
  "$PYTHON" -c "from runner.workflow import load_workflow; print(load_workflow('$WORKFLOW').workflow_name)"
)"

mapfile -t actions < <(
  "$PYTHON" -c "from runner.workflow import load_workflow; wf=load_workflow('$WORKFLOW'); print('\n'.join(sorted({n.action for n in wf.nodes.values()})))"
)

if [[ "${#actions[@]}" -eq 0 ]]; then
  echo "No actions found in workflow: $WORKFLOW" >&2
  exit 1
fi

echo "workflow=${workflow_name}"
echo "actions=${actions[*]}"
echo "memory_tiers=${MEMORY_TIERS}"
echo "trace_dir=${TRACE_DIR}"
echo "out_dir=${OUT_DIR}"

trace_args=()
label_args=()

for memory_mb in ${MEMORY_TIERS}; do
  echo
  echo "=== memory ${memory_mb} MB ==="

  for action in "${actions[@]}"; do
    # Create or update the action with the requested memory tier.
    # Updating the action also gives us a useful cold-like first run for this tier.
    if "$WSK_CLI" "${WSK_ARGS[@]}" action get "$action" >/dev/null 2>&1; then
      "$WSK_CLI" "${WSK_ARGS[@]}" action update "$action" "$ACTION_FILE" \
        --kind "$OW_KIND" \
        --memory "$memory_mb" \
        --timeout "$ACTION_TIMEOUT_MS"
    else
      "$WSK_CLI" "${WSK_ARGS[@]}" action create "$action" "$ACTION_FILE" \
        --kind "$OW_KIND" \
        --memory "$memory_mb" \
        --timeout "$ACTION_TIMEOUT_MS"
    fi
  done

  trace_file="${TRACE_DIR}/${workflow_name}_mem${memory_mb}.csv"

  # Cold-like sample block after action update.
  "$PYTHON" -m runner.run_workflow \
    --workflow "$WORKFLOW" \
    --apihost "$APIHOST" \
    --auth "$AUTH" \
    --trace "$trace_file" \
    --count "$COLD_COUNT" \
    --interval-ms "$COLD_INTERVAL_MS" \
    --max-workers "$MAX_WORKERS" \
    --allocated-memory-mb "$memory_mb" \
    --cpu-profile "$CPU_PROFILE"

  # Warm sample block. Consecutive calls should mostly reuse containers.
  "$PYTHON" -m runner.run_workflow \
    --workflow "$WORKFLOW" \
    --apihost "$APIHOST" \
    --auth "$AUTH" \
    --trace "$trace_file" \
    --count "$WARM_COUNT" \
    --interval-ms "$WARM_INTERVAL_MS" \
    --max-workers "$MAX_WORKERS" \
    --allocated-memory-mb "$memory_mb" \
    --cpu-profile "$CPU_PROFILE"

  trace_args+=("$trace_file")
  label_args+=("mem_${memory_mb}mb")
done

# Build the existing Stage-3 latency profile grouped by memory label.
"$PYTHON" -m runner.profile_latency \
  --traces "${trace_args[@]}" \
  --trace-labels "${label_args[@]}" \
  --out-dir "$OUT_DIR"

# Build memory-focused summary tables from the profile pack.
"$PYTHON" -m runner.summarize_memory_sweep \
  --profile-dir "$OUT_DIR" \
  --out-dir "$OUT_DIR"

echo
echo "Memory sweep complete."
echo "Trace CSVs: $TRACE_DIR"
echo "Report pack: $OUT_DIR"
