#!/usr/bin/env bash
# Complete formal T-SUPPORT and freeze H1 without depending on an IDE session.
set -euo pipefail

REPO=/root/LLM-output-density
SCRIPTS="$REPO/experiments/opd_sft_h1/scripts"
PYTHON=/root/miniconda3/envs/density/bin/python
ROOT=/root/autodl-tmp/cycle09_stage3_followup/cpu_h1
LOGS="$ROOT/logs"
STATUS="$ROOT/status.json"
INPUT=/root/autodl-tmp/cycle09_stage3_followup/contracts/support_inputs.json

mkdir -p "$LOGS"
printf '{"status":"running_support_inventory","auto_shutdown":false}\n' > "$STATUS"

finish() {
  local rc=$?
  if [[ "$rc" -eq 0 ]]; then
    printf '{"status":"complete","auto_shutdown":false}\n' > "$STATUS"
  else
    printf '{"status":"failed","returncode":%s,"auto_shutdown":false}\n' "$rc" > "$STATUS"
  fi
}
trap finish EXIT

cd "$REPO"
"$PYTHON" "$SCRIPTS/cycle09_stage3_support_inventory.py" --phase write \
  > "$LOGS/support_inventory.log" 2>&1

printf '{"status":"running_t_support","auto_shutdown":false}\n' > "$STATUS"
"$PYTHON" "$SCRIPTS/cycle09_stage3_support.py" --phase run --input-manifest "$INPUT" \
  > "$LOGS/t_support.log" 2>&1

printf '{"status":"writing_h1_handoff","auto_shutdown":false}\n' > "$STATUS"
"$PYTHON" "$SCRIPTS/cycle09_stage3_handoff.py" --gate H1 \
  > "$LOGS/h1_handoff.log" 2>&1
