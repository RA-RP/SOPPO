#!/usr/bin/env bash
# Attach CPU-only D5/D6/D7 tables after the detached D5 GPU queue finishes.
set -euo pipefail

PYTHON=/root/miniconda3/envs/density/bin/python
ROOT=/root/autodl-tmp/cycle09_relative_functional_contraction/d5_fairness
PIDFILE="$ROOT/formal.pid"
PROGRESS="$ROOT/formal/progress.json"
SCRIPT=/root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_d5_d7_tables.py
LOG="$ROOT/d5_d7_tables.log"
SESSION=cycle09_d5_d7_tables

if screen -ls | grep -q "\\.${SESSION}[[:space:]]"; then
  echo "D5/D6/D7 waiter already running" >&2
  exit 0
fi

screen -dmS "$SESSION" /bin/bash -lc "
  set -euo pipefail
  while [ ! -s '$PIDFILE' ]; do sleep 20; done
  pid=\$(cat '$PIDFILE')
  while kill -0 \"\$pid\" 2>/dev/null; do sleep 120; done
  '$PYTHON' -c \"import json,os,sys; p='$PROGRESS'; d=json.load(open(p)) if os.path.isfile(p) else {}; sys.exit(0 if d.get('status') == 'complete' else 1)\"
  exec '$PYTHON' '$SCRIPT' --phase formal
" >>'$LOG' 2>&1

echo "started D5/D6/D7 waiter screen=$SESSION log=$LOG"
