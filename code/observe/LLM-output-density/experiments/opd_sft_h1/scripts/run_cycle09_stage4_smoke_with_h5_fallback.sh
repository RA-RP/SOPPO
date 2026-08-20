#!/usr/bin/env bash
# Foreground Stage-4 smoke.  On a nonzero exit it immediately detaches H5 resume.
set -euo pipefail

REPO=/root/LLM-output-density
ROOT=/root/autodl-tmp/cycle09_stage4_state_displacement
PY=/root/miniconda3/envs/density/bin/python
SUPERVISOR=$REPO/experiments/opd_sft_h1/scripts/cycle09_stage4_supervisor.py
H5_RESUME=$REPO/experiments/opd_sft_h1/scripts/run_cycle09_h5_postprocess_resume.sh
LOG_DIR=$ROOT/logs
mkdir -p "$LOG_DIR"

success=0
fallback() {
  rc=$?
  if [[ "$success" -eq 0 ]]; then
    printf '%s Stage-4 smoke failed/interrupted rc=%s; starting H5 resume.\\n' \
      "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG_DIR/smoke_with_h5_fallback.log"
    "$H5_RESUME" >> "$LOG_DIR/smoke_with_h5_fallback.log" 2>&1 || true
  fi
  exit "$rc"
}
trap fallback EXIT
trap 'exit 130' INT TERM

printf '%s Stage-4 foreground smoke started.\\n' "$(date -u +%FT%TZ)" | tee -a "$LOG_DIR/smoke_with_h5_fallback.log"
"$PY" "$SUPERVISOR" --phase smoke 2>&1 | tee -a "$LOG_DIR/smoke_with_h5_fallback.log"
success=1
trap - EXIT INT TERM
printf '%s Stage-4 smoke completed successfully; H5 remains paused by design.\\n' \
  "$(date -u +%FT%TZ)" | tee -a "$LOG_DIR/smoke_with_h5_fallback.log"
