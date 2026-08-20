#!/usr/bin/env bash
# Continue GPU1 with the disjoint Llama PROBE-CORE partition after T-SUB.
set -euo pipefail

REPO=/root/LLM-output-density
SCRIPTS="$REPO/experiments/opd_sft_h1/scripts"
PYTHON=/root/miniconda3/envs/density/bin/python
ROOT=/root/autodl-tmp/cycle09_stage3_followup/gpu1_probe
LOGS="$ROOT/logs"
STATUS="$ROOT/status.json"
UPSTREAM_PID_FILE=/root/autodl-tmp/cycle09_stage3_followup/gpu1_core/supervisor.pid

mkdir -p "$LOGS"
printf '{"status":"waiting_for_llama_tsub","auto_shutdown":false}\n' > "$STATUS"

finish() {
  local rc=$?
  if [[ "$rc" -eq 0 ]]; then
    printf '{"status":"complete","auto_shutdown":false}\n' > "$STATUS"
  else
    printf '{"status":"failed","returncode":%s,"auto_shutdown":false}\n' "$rc" > "$STATUS"
  fi
}
trap finish EXIT

upstream_pid=$(cat "$UPSTREAM_PID_FILE")
while kill -0 "$upstream_pid" 2>/dev/null; do
  sleep 20
done
"$PYTHON" -c "import json; p=json.load(open('/root/autodl-tmp/cycle09_stage3_followup/gpu1_core/status.json')); assert p['status']=='complete'"

printf '{"status":"running_probe_core_llama_partition","auto_shutdown":false}\n' > "$STATUS"
cd "$REPO"
env CYCLE09_STAGE3_SCOPE=partition_probe_llama_20260723 CUDA_VISIBLE_DEVICES=1 \
  "$PYTHON" "$SCRIPTS/cycle09_stage3_probe_core.py" \
  --families llama3_2_3b --phase all --device cuda:0 \
  > "$LOGS/probe_core_llama.log" 2>&1
