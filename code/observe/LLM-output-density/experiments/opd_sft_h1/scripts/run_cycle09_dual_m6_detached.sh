#!/usr/bin/env bash
# Detached, resumable launcher.  It deliberately never shuts down the instance.
set -euo pipefail

REPO=/root/LLM-output-density
SCRIPT="$REPO/experiments/opd_sft_h1/scripts/cycle09_dual_m6_campaign.py"
ROOT=/root/autodl-tmp/cycle09_dual_m6
LOG="$ROOT/logs/supervisor.log"
PIDFILE="$ROOT/supervisor.pid"

mkdir -p "$ROOT/logs"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running: pid=$(cat "$PIDFILE") log=$LOG"
  exit 0
fi

nohup setsid /root/miniconda3/envs/density/bin/python "$SCRIPT" supervisor >>"$LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PIDFILE"
echo "launched pid=$pid log=$LOG"
