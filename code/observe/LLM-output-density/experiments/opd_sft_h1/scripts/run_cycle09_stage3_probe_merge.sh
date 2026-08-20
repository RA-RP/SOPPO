#!/usr/bin/env bash
# Merge PROBE-CORE only after both disjoint GPU partitions validate.
set -euo pipefail

REPO=/root/LLM-output-density
SCRIPTS="$REPO/experiments/opd_sft_h1/scripts"
PYTHON=/root/miniconda3/envs/density/bin/python
ROOT=/root/autodl-tmp/cycle09_stage3_followup/probe_merge
STATUS="$ROOT/status.json"
GPU0_PID_FILE=/root/autodl-tmp/cycle09_stage3_followup/gpu0_core/supervisor.pid
GPU1_PID_FILE=/root/autodl-tmp/cycle09_stage3_followup/gpu1_probe/supervisor.pid

mkdir -p "$ROOT"
printf '{"status":"waiting_for_probe_partitions","auto_shutdown":false}\n' > "$STATUS"

finish() {
  local rc=$?
  if [[ "$rc" -eq 0 ]]; then
    printf '{"status":"complete","auto_shutdown":false}\n' > "$STATUS"
  else
    printf '{"status":"failed","returncode":%s,"auto_shutdown":false}\n' "$rc" > "$STATUS"
  fi
}
trap finish EXIT

for pid_file in "$GPU0_PID_FILE" "$GPU1_PID_FILE"; do
  pid=$(cat "$pid_file")
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
done
"$PYTHON" -c "import json; assert json.load(open('/root/autodl-tmp/cycle09_stage3_followup/gpu0_core/status.json'))['status']=='complete'; assert json.load(open('/root/autodl-tmp/cycle09_stage3_followup/gpu1_probe/status.json'))['status']=='complete'"

printf '{"status":"merging_probe_partitions","auto_shutdown":false}\n' > "$STATUS"
cd "$REPO"
"$PYTHON" "$SCRIPTS/cycle09_stage3_probe_core_merge.py" --phase merge \
  > "$ROOT/merge.log" 2>&1
