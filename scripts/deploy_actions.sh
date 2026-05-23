#!/usr/bin/env bash
set -euo pipefail

# Resolve the stage2-openwhisk-prototype root directory.
# This lets the script find actions/*.py no matter where it is launched.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# OpenWhisk Python runtime kind.
# Override with OW_KIND=python:3.11 if your cluster uses another runtime name.
OW_KIND="${OW_KIND:-python:3}"

# Workflow stage actions for the three supported DAGs:
#   - civic_alert_flow
#   - spoken_dialog_flow
#   - visual_qa_flow
#
# Each action wraps actions/workflow_action.py, which selects the right CPU /
# memory profile via (workflow_name, stage_name) on invocation.
WORKFLOW_ACTIONS=(
  wf_civic_detect_object
  wf_civic_estimate_pose
  wf_civic_match_face
  wf_civic_classify_scene
  wf_civic_translate_alert
  wf_visual_image_embed
  wf_visual_text_embed
  wf_visual_answer_question
  wf_spoken_speech_decode
  wf_spoken_topic_route
  wf_spoken_entity_extract
  wf_spoken_response_generate
  wf_spoken_speech_synthesize
)

for action in "${WORKFLOW_ACTIONS[@]}"; do
  if wsk action get "$action" -i >/dev/null 2>&1; then
    wsk action update "$action" "$ROOT_DIR/actions/workflow_action.py" --kind "$OW_KIND" -i
  else
    wsk action create "$action" "$ROOT_DIR/actions/workflow_action.py" --kind "$OW_KIND" -i
  fi
done

# Print the final action list for verification.
wsk action list -i
