#!/usr/bin/env bash
# Restart-safe, VSCode-independent runner for D4.1 on one 32G GPU.
set -euo pipefail

PYTHON=/root/miniconda3/envs/density/bin/python
SCRIPT=/root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_qwen_d4_merged_state.py
ROOT=/root/autodl-tmp/cycle09_relative_functional_contraction/d4_merged_state
LOG="$ROOT/formal_memory.log"
PIDFILE="$ROOT/formal_memory.pid"
SESSION=cycle09_qwen_d4_formal

mkdir -p "$ROOT"
if screen -ls | grep -q "\\.${SESSION}[[:space:]]"; then
  echo "D4.1 formal screen session already running" >&2
  exit 0
fi
if pgrep -f '[c]ycle09_qwen_d4_merged_state.py --phase formal-memory' >/dev/null; then
  echo "D4.1 formal process already running" >&2
  exit 0
fi

screen -dmS "$SESSION" /bin/bash -lc \
  "exec env CUDA_VISIBLE_DEVICES=0 '$PYTHON' '$SCRIPT' --phase formal-memory --tag formal --device cuda:0 --forward-batch-size 1 --max-batch-tokens 8192 >>'$LOG' 2>&1"
sleep 1
pid=$(pgrep -f '[c]ycle09_qwen_d4_merged_state.py --phase formal-memory' | head -n1 || true)
printf '%s\n' "$pid" > "$PIDFILE"
echo "started screen=$SESSION pid=$pid log=$LOG"
