#!/usr/bin/env bash
# Detached, restart-safe Stage-4 formal launcher. It never shuts down the instance.
set -euo pipefail

REPO=/root/LLM-output-density
ROOT=/root/autodl-tmp/cycle09_stage4_state_displacement
PY=/root/miniconda3/envs/density/bin/python
SUPERVISOR=$REPO/experiments/opd_sft_h1/scripts/cycle09_stage4_supervisor.py
LOG_DIR=$ROOT/logs
mkdir -p "$LOG_DIR"

if (( $# > 0 )) && [[ "$1" == "--dry-run" ]]; then
  "$PY" "$SUPERVISOR" --phase dry-run
  exit 0
fi

if pgrep -f 'cycle09_stage3_frozen_self_postprocess.py --phase worker' >/dev/null; then
  echo 'H5 workers are active. Pause H5 before starting Stage-4 formal.' >&2
  exit 2
fi
if pgrep -f 'cycle09_stage4_supervisor.py --phase formal' >/dev/null; then
  echo 'Stage-4 formal supervisor is already active; it is resumable and will skip completed cells.' >&2
  exit 2
fi

setsid /root/miniconda3/envs/density/bin/python /root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_stage4_supervisor.py --phase formal > "$LOG_DIR/stage4_formal.log" 2>&1 < /dev/null &

pid=$!
printf 'stage4_formal_supervisor_pid=%s\\nstarted_utc=%s\\nno_auto_shutdown=true\\n' \
  "$pid" "$(date -u +%FT%TZ)" > "$LOG_DIR/stage4_formal_launcher.pid"
echo "Stage-4 formal detached supervisor PID=$pid"
