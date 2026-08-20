#!/usr/bin/env bash
# Complete the already-trained Q1 alpha=.5 endpoint and freeze Stage3 H0.
set -euo pipefail

REPO=/root/LLM-output-density
SCRIPTS="$REPO/experiments/opd_sft_h1/scripts"
PYTHON=/root/miniconda3/envs/density/bin/python
ROOT=/root/autodl-tmp/cycle09_stage3_followup/H0_q1_finalize
STATUS="$ROOT/status.json"
LOGS="$ROOT/logs"

mkdir -p "$LOGS"
printf '{"status":"running","formal":true,"training_started":false,"auto_shutdown":false}\n' > "$STATUS"

finish() {
  local rc=$?
  if [[ "$rc" -eq 0 ]]; then
    printf '{"status":"complete","formal":true,"training_started":false,"auto_shutdown":false}\n' > "$STATUS"
  else
    printf '{"status":"failed","returncode":%s,"formal":true,"training_started":false,"auto_shutdown":false}\n' "$rc" > "$STATUS"
  fi
}
trap finish EXIT

cd "$REPO"
"$PYTHON" "$SCRIPTS/cycle09_q1_stageb_postprocess.py" --phase validate > "$LOGS/validate.log" 2>&1
"$PYTHON" "$SCRIPTS/cycle09_q1_stageb_postprocess.py" --phase export > "$LOGS/export.log" 2>&1
"$PYTHON" "$SCRIPTS/cycle09_q1_stageb_postprocess.py" --phase support-stats > "$LOGS/support.log" 2>&1

env CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$SCRIPTS/cycle09_q1_behavior.py" \
  --phase cell --step 320 > "$LOGS/behavior_320.log" 2>&1 &
behavior_pid=$!

(
  for probe in S_math E_math E_math_hard_v2 E_ood E_if E_general; do
    env CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$SCRIPTS/cycle09_q1_geometry.py" \
      --phase cell --step 320 --probe "$probe" --device cuda:0 --no-retain-factor \
      > "$LOGS/geometry_320_${probe}.log" 2>&1
  done
) &
geometry_pid=$!

wait "$behavior_pid"
wait "$geometry_pid"

"$PYTHON" "$SCRIPTS/cycle09_q1_behavior.py" --phase finalize > "$LOGS/behavior_finalize.log" 2>&1
"$PYTHON" "$SCRIPTS/cycle09_q1_geometry.py" --phase finalize > "$LOGS/geometry_finalize.log" 2>&1
"$PYTHON" "$SCRIPTS/cycle09_q1_stageb_handoff.py" > "$LOGS/q1_handoff.log" 2>&1
"$PYTHON" "$SCRIPTS/cycle09_stage3_handoff.py" --gate H0 > "$LOGS/stage3_h0_handoff.log" 2>&1
