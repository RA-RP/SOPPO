#!/usr/bin/env bash
# Continue the formal Llama H2/H3 lane after the active T-WHITE process.
set -euo pipefail

REPO=/root/LLM-output-density
SCRIPTS="$REPO/experiments/opd_sft_h1/scripts"
PYTHON=/root/miniconda3/envs/density/bin/python
ROOT=/root/autodl-tmp/cycle09_stage3_followup/gpu1_core
LOGS="$ROOT/logs"
STATUS="$ROOT/status.json"
UPSTREAM_PID_FILE=/root/autodl-tmp/cycle09_stage3_followup/H2_white/llama_formal.pid

mkdir -p "$LOGS"
printf '{"status":"waiting_for_llama_twhite","auto_shutdown":false}\n' > "$STATUS"

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

"$PYTHON" -c "import json; p=json.load(open('/root/autodl-tmp/cycle09_stage3_followup/H2_white/T_WHITE_llama3_2_3b_manifest.json')); assert p['status']=='complete'"
printf '{"status":"running_llama_tsub","auto_shutdown":false}\n' > "$STATUS"
cd "$REPO"
env CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$SCRIPTS/cycle09_stage3_tsub.py" \
  --family llama3_2_3b --arms opd,offkd --steps 5,20,40,80,160,320 \
  --layer 14 --probes E_ood --rank-fraction 0.05 --device cuda:0 \
  > "$LOGS/llama_tsub.log" 2>&1
