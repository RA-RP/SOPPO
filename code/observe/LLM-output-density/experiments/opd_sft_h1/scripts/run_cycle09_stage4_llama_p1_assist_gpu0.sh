#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/cycle09_stage4_state_displacement
PY=/root/miniconda3/envs/density/bin/python
SCRIPT=/root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_stage4_llama_p1_assist.py
LOG_DIR=$ROOT/logs
mkdir -p "$LOG_DIR"

if pgrep -f 'cycle09_stage4_llama_p1_assist.py' >/dev/null; then
  echo 'Llama P1 GPU0 assistant is already active.' >&2
  exit 2
fi

setsid "$PY" "$SCRIPT" > "$LOG_DIR/llama_p1_gpu0_assist.log" 2>&1 < /dev/null &
pid=$!
printf 'pid=%s\\nstarted_utc=%s\\ndevice=cuda:0\\n' "$pid" "$(date -u +%FT%TZ)" > "$LOG_DIR/llama_p1_gpu0_assist.pid"
echo "Llama P1 GPU0 assistant PID=$pid"
