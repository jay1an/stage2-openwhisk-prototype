#!/usr/bin/env bash
# Poll civic wf-* pods every 0.5s and log create / terminate transitions
# with millisecond timestamps. Output: data/pod_events.log
set -u

OUT="${1:-data/pod_events.log}"
mkdir -p "$(dirname "$OUT")"
: > "$OUT"

declare -A PREV

while true; do
  ts=$(date +%H:%M:%S.%3N)
  declare -A CURR=()
  while IFS= read -r line; do
    name=$(awk '{print $1}' <<<"$line")
    status=$(awk '{print $3}' <<<"$line")
    # Only track status (not age — age changes every second and would spam CHANGE)
    CURR["$name"]="$status"
  done < <(kubectl -n openwhisk get pods --no-headers 2>/dev/null | grep "wf-civic" || true)

  # New pods
  for name in "${!CURR[@]}"; do
    if [[ -z "${PREV[$name]:-}" ]]; then
      echo "$ts NEW $name ${CURR[$name]}" >> "$OUT"
    elif [[ "${PREV[$name]}" != "${CURR[$name]}" ]]; then
      echo "$ts CHANGE $name ${PREV[$name]} -> ${CURR[$name]}" >> "$OUT"
    fi
  done
  # Removed pods
  for name in "${!PREV[@]}"; do
    if [[ -z "${CURR[$name]:-}" ]]; then
      echo "$ts GONE $name (was ${PREV[$name]})" >> "$OUT"
    fi
  done

  # Snapshot for next iteration
  unset PREV
  declare -A PREV
  for name in "${!CURR[@]}"; do
    PREV["$name"]="${CURR[$name]}"
  done

  sleep 0.5
done
