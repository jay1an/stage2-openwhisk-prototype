#!/usr/bin/env bash
set -euo pipefail

# Resolve the stage2-openwhisk-prototype root directory.
# This lets the script find actions/*.py no matter where it is launched.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# OpenWhisk Python runtime kind.
# Override with OW_KIND=python:3.11 if your cluster uses another runtime name.
OW_KIND="${OW_KIND:-python:3}"

# Basic smoke-test actions for linear3 and parallel_join4.
NOOP_ACTIONS=(
  wf_decode
  wf_resize
  wf_classify
  wf_detect
  wf_caption
  wf_merge
)

for action in "${NOOP_ACTIONS[@]}"; do
  # Create or update the noop action.
  if wsk action get "$action" -i >/dev/null 2>&1; then
    wsk action update "$action" "$ROOT_DIR/actions/noop.py" --kind "$OW_KIND" -i
  else
    wsk action create "$action" "$ROOT_DIR/actions/noop.py" --kind "$OW_KIND" -i
  fi
done

# SeBS-Flow Level-2 compute mock actions.
# These actions preserve the workflow DAG shape and add CPU/memory/JSON overhead.
SEBS_MOCK_ACTIONS=(
  wf_sebs_trip_reserve_hotel
  wf_sebs_trip_reserve_rental
  wf_sebs_trip_reserve_flight
  wf_sebs_trip_confirm
  wf_sebs_video_decode
  wf_sebs_video_analyse
  wf_sebs_video_summarize
  wf_sebs_mr_split
  wf_sebs_mr_map
  wf_sebs_mr_shuffle
  wf_sebs_mr_reduce
  wf_sebs_ml_generate
  wf_sebs_ml_train
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

for action in "${SEBS_MOCK_ACTIONS[@]}"; do
  # Create or update the SeBS mock action.
  if wsk action get "$action" -i >/dev/null 2>&1; then
    wsk action update "$action" "$ROOT_DIR/actions/sebs_mock.py" --kind "$OW_KIND" -i
  else
    wsk action create "$action" "$ROOT_DIR/actions/sebs_mock.py" --kind "$OW_KIND" -i
  fi
done

# Print the final action list for verification.
wsk action list -i
