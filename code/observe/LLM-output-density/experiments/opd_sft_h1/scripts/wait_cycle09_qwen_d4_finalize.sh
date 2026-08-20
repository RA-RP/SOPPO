#!/usr/bin/env bash
# Wait for the detached D4 run, then finalize only after a complete progress record.
set -euo pipefail

PYTHON=/root/miniconda3/envs/density/bin/python
ROOT=/root/autodl-tmp/cycle09_relative_functional_contraction/d4_merged_state
FORMAL_PIDFILE="$ROOT/formal_memory.pid"
PROGRESS="$ROOT/formal/formal_memory_progress.json"
FINALIZER=/root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_qwen_d4_finalize.py
LOG="$ROOT/finalize_waiter.log"
SESSION=cycle09_qwen_d4_finalize

if screen -ls | grep -q "\\.${SESSION}[[:space:]]"; then
  echo "D4 finalizer waiter already running" >&2
  exit 0
fi

screen -dmS "$SESSION" /bin/bash -lc "
  set -euo pipefail
  while [ ! -s '$FORMAL_PIDFILE' ]; do sleep 20; done
  pid=\$(cat '$FORMAL_PIDFILE')
  while kill -0 \"\$pid\" 2>/dev/null; do sleep 120; done
  '$PYTHON' -c \"import json,sys; p='$PROGRESS'; d=json.load(open(p)) if __import__('os').path.isfile(p) else {}; sys.exit(0 if d.get('status') == 'complete' else 1)\"
  exec '$PYTHON' '$FINALIZER' --phase finalize
" >>'$LOG' 2>&1

echo "started finalizer waiter screen=$SESSION log=$LOG"
