#!/usr/bin/env bash
# Launch the authorized 160->320 delivery independently of VS Code/SSH.
set -euo pipefail

REPO=/root/LLM-output-density
PYTHON=/root/miniconda3/envs/density/bin/python
SUPERVISOR="$REPO/experiments/opd_sft_h1/scripts/cycle09_block3_stageb_supervisor.py"
ROOT=/root/autodl-tmp/cycle09_block3/stageb_320/watchdog
LOG="$ROOT/watchdog.log"
PID="$ROOT/watchdog.pid"
mkdir -p "$ROOT"

if [[ "${1:-}" != "--run" ]]; then
  if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
    echo "watchdog already running: $(cat "$PID")"
    exit 0
  fi
  setsid nohup bash "$0" --run >>"$LOG" 2>&1 < /dev/null &
  echo "$!" > "$PID"
  echo "started detached Stage-B watchdog pid=$! log=$LOG"
  exit 0
fi

echo "[$(date -Is)] current early delivery already verified; starting Stage-B supervisor"
exec "$PYTHON" "$SUPERVISOR"
