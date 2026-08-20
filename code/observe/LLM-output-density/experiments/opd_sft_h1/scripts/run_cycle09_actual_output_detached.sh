#!/usr/bin/env bash
# Launch the approved O0--O5 run independently of VS Code.  No automatic shutdown.
set -euo pipefail

REPO=/root/LLM-output-density
ROOT=/root/autodl-tmp/cycle09_actual_output_trajectory
PY=/root/miniconda3/envs/density/bin/python
SUPERVISOR=$REPO/experiments/opd_sft_h1/scripts/cycle09_actual_output_supervisor.py
LOG_DIR=$ROOT/logs
mkdir -p "$LOG_DIR"

if pgrep -f '[p]ython .*cycle09_actual_output_supervisor.py --phase formal' >/dev/null; then
  echo 'Actual-output formal supervisor is already active.' >&2
  exit 2
fi

setsid "$PY" "$SUPERVISOR" --phase formal > "$LOG_DIR/formal.log" 2>&1 < /dev/null &
pid=$!
printf 'actual_output_supervisor_pid=%s\nstarted_utc=%s\nno_auto_shutdown=true\nstops_after=O5\n' \
  "$pid" "$(date -u +%FT%TZ)" > "$ROOT/formal_launcher.pid"
echo "Actual-output O0--O5 supervisor detached: PID=$pid"
