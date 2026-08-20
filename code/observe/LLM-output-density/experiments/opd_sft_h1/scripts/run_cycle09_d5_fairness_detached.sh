#!/usr/bin/env bash
# Restart-safe D5 runner for one 32G GPU; the Python formal phase finalizes its CSVs.
set -euo pipefail

PYTHON=/root/miniconda3/envs/density/bin/python
SCRIPT=/root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_d5_fairness.py
ROOT=/root/autodl-tmp/cycle09_relative_functional_contraction/d5_fairness
LOG="$ROOT/formal.log"
PIDFILE="$ROOT/formal.pid"
SESSION=cycle09_d5_fairness

mkdir -p "$ROOT"
if screen -ls | grep -q "\\.${SESSION}[[:space:]]"; then
  echo "D5 fairness screen session already running" >&2
  exit 0
fi
if pgrep -f '[p]ython.*cycle09_d5_fairness.py --phase formal' >/dev/null; then
  echo "D5 fairness process already running" >&2
  exit 0
fi

screen -dmS "$SESSION" /bin/bash -lc \
  "exec env CUDA_VISIBLE_DEVICES=0 '$PYTHON' '$SCRIPT' --phase formal --tag formal --device cuda:0 --forward-batch-size 1 --max-batch-tokens 8192 >>'$LOG' 2>&1"
sleep 1
pid=$(pgrep -f '[p]ython.*cycle09_d5_fairness.py --phase formal' | head -n1 || true)
printf '%s\n' "$pid" > "$PIDFILE"
echo "started screen=$SESSION pid=$pid log=$LOG"
