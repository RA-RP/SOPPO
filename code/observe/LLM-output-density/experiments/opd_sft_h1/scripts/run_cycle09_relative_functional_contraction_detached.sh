#!/usr/bin/env bash
# Restart-safe detached launcher; it intentionally never powers off the instance.
set -euo pipefail

PYTHON=/root/miniconda3/envs/density/bin/python
SCRIPT=/root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_relative_functional_contraction_supervisor.py
ROOT=/root/autodl-tmp/cycle09_relative_functional_contraction
LOG="$ROOT/supervisor.log"
PIDFILE="$ROOT/supervisor.pid"
SESSION=cycle09_relative_contraction

mkdir -p "$ROOT"
if screen -ls | grep -q "\.${SESSION}[[:space:]]"; then
  echo "relative functional contraction screen session is already running" >&2
  exit 0
fi
if pgrep -f '[c]ycle09_relative_functional_contraction_supervisor.py --mode formal' >/dev/null; then
  echo "relative functional contraction supervisor is already running" >&2
  exit 0
fi

screen -dmS "$SESSION" /bin/bash -lc "exec '$PYTHON' '$SCRIPT' --mode formal >>'$LOG' 2>&1"
sleep 1
pid=$(pgrep -f '[c]ycle09_relative_functional_contraction_supervisor.py --mode formal' | head -n1 || true)
printf '%s\n' "$pid" > "$PIDFILE"
echo "started screen=$SESSION pid=$pid log=$LOG"
