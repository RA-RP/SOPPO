#!/usr/bin/env bash
# Continue formal Qwen metric/probe work after the active H0 endpoint package.
set -euo pipefail

REPO=/root/LLM-output-density
SCRIPTS="$REPO/experiments/opd_sft_h1/scripts"
PYTHON=/root/miniconda3/envs/density/bin/python
ROOT=/root/autodl-tmp/cycle09_stage3_followup/gpu0_core
LOGS="$ROOT/logs"
STATUS="$ROOT/status.json"
H0_PID_FILE=/root/autodl-tmp/cycle09_stage3_followup/H0_q1_finalize/supervisor.pid
SMOKE_SCOPE=smoke_twhite_cache_qwen_20260723

mkdir -p "$LOGS"
printf '{"status":"waiting_for_H0","auto_shutdown":false}\n' > "$STATUS"

finish() {
  local rc=$?
  if [[ "$rc" -eq 0 ]]; then
    printf '{"status":"complete","auto_shutdown":false}\n' > "$STATUS"
  else
    printf '{"status":"failed","returncode":%s,"auto_shutdown":false}\n' "$rc" > "$STATUS"
  fi
}
trap finish EXIT

h0_pid=$(cat "$H0_PID_FILE")
while kill -0 "$h0_pid" 2>/dev/null; do
  sleep 20
done
"$PYTHON" -c "import json; p=json.load(open('/root/autodl-tmp/cycle09_stage3_followup/H0_q1_finalize/status.json')); assert p['status']=='complete'"

cd "$REPO"
printf '{"status":"validating_qwen_twhite_cache","auto_shutdown":false}\n' > "$STATUS"
env CYCLE09_STAGE3_SCOPE="$SMOKE_SCOPE" CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
  "$SCRIPTS/cycle09_stage3_twhite.py" --family qwen3_4b --arms alpha05 \
  --steps 20 --layer 18 --probes E_ood --sample-limit 2 --device cuda:0 \
  > "$LOGS/qwen_twhite_cache_smoke.log" 2>&1
"$PYTHON" - <<'PY'
import pandas as pd
old = pd.read_csv('/root/autodl-tmp/cycle09_stage3_followup/smoke/smoke_20260723/H2_white/T_WHITE_qwen3_4b.csv')
new = pd.read_csv('/root/autodl-tmp/cycle09_stage3_followup/smoke/smoke_twhite_cache_qwen_20260723/H2_white/T_WHITE_qwen3_4b.csv')
keys = ['family','arm','step','probe','layer','module','epsilon','track']
values = ['r_epsilon','base_r_epsilon','delta_from_base']
for frame in (old, new):
    frame.sort_values(keys, inplace=True, kind='stable')
assert old[keys + values].reset_index(drop=True).equals(new[keys + values].reset_index(drop=True)), 'T-WHITE smoke numeric drift'
PY

printf '{"status":"running_qwen_twhite","auto_shutdown":false}\n' > "$STATUS"
env CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$SCRIPTS/cycle09_stage3_twhite.py" \
  --family qwen3_4b --arms opd,sft,offkd,seqkd,alpha05 \
  --steps 0,5,20,40,80,160,320 --layer 18 --device cuda:0 \
  > "$LOGS/qwen_twhite.log" 2>&1

printf '{"status":"running_probe_core_qwen_partition","auto_shutdown":false}\n' > "$STATUS"
env CYCLE09_STAGE3_SCOPE=partition_probe_qwen_20260723 CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON" "$SCRIPTS/cycle09_stage3_probe_core.py" \
  --families qwen3_4b --phase all --device cuda:0 \
  > "$LOGS/probe_core_qwen.log" 2>&1
