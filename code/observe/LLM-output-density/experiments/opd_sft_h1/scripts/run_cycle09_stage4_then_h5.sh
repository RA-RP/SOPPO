#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/cycle09_stage4_state_displacement
PY=/root/miniconda3/envs/density/bin/python
SCRIPT=/root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_stage4_then_h5.py
LOG_DIR=$ROOT/logs
mkdir -p "$LOG_DIR"

if pgrep -f 'cycle09_stage4_then_h5.py' >/dev/null; then
  echo 'Stage4-to-H5 controller is already active.' >&2
  exit 2
fi

setsid "$PY" "$SCRIPT" > "$LOG_DIR/stage4_then_h5.log" 2>&1 < /dev/null &
pid=$!
printf 'pid=%s\\nstarted_utc=%s\\npoll_seconds=60\\n' "$pid" "$(date -u +%FT%TZ)" > "$LOG_DIR/stage4_then_h5.pid"
echo "Stage4-to-H5 controller PID=$pid"
