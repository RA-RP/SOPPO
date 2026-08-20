#!/usr/bin/env bash
# Detached, fail-stop sequence for the early six-checkpoint Llama completion.
set -euo pipefail

REPO=/root/LLM-output-density
PYTHON=/root/miniconda3/envs/density/bin/python
SUPERVISOR="$REPO/experiments/opd_sft_h1/scripts/cycle09_block3_p1_supervisor.py"
ROOT=/root/autodl-tmp/cycle09_block3/p1_early_watchdog
LOG="$ROOT/watchdog.log"
PID_FILE="$ROOT/watchdog.pid"

detach() {
  mkdir -p "$ROOT"
  setsid nohup bash "$0" --run >>"$LOG" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" >"$PID_FILE"
  printf '{"pid":%s,"log":"%s","shutdown_policy":"never"}\n' "$pid" "$LOG"
}

run() {
  mkdir -p "$ROOT"
  printf '[%s] smoke\n' "$(date --iso-8601=seconds)" >>"$LOG"
  "$PYTHON" "$SUPERVISOR" --mode smoke

  printf '[%s] paired cap pilot\n' "$(date --iso-8601=seconds)" >>"$LOG"
  "$PYTHON" "$SUPERVISOR" --mode cap-pilot

  printf '[%s] Q1 extension plus Llama OPD early grid\n' "$(date --iso-8601=seconds)" >>"$LOG"
  "$PYTHON" "$SUPERVISOR" --mode formal --scope opd_early

  printf '[%s] Llama offline early grid\n' "$(date --iso-8601=seconds)" >>"$LOG"
  "$PYTHON" "$SUPERVISOR" --mode formal --scope offline_early

  printf '[%s] complete\n' "$(date --iso-8601=seconds)" >>"$LOG"
}

case "${1:---detach}" in
  --detach) detach ;;
  --run) run ;;
  *) echo "usage: $0 [--detach|--run]" >&2; exit 2 ;;
esac
