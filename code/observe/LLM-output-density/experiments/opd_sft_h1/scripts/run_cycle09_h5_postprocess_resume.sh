#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/cycle09_stage3_followup/H5_frozen_self
SCRIPT=/root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_stage3_frozen_self_postprocess.py
PY=/root/miniconda3/envs/density/bin/python
GPU0_PLAN='behavior:5,geometry:5,behavior:40,geometry:40,behavior:160,geometry:80'
GPU1_PLAN='geometry_reference,behavior:20,geometry:20,behavior:80,geometry:160,behavior:320,geometry:320'

if (( $# > 0 )) && [[ "$1" == "--dry-run" ]]; then
  printf 'gpu0=%s\n' "$GPU0_PLAN"
  printf 'gpu1=%s\n' "$GPU1_PLAN"
  exit 0
fi

if pgrep -f 'cycle09_stage3_frozen_self_postprocess.py --phase worker' >/dev/null; then
  echo 'H5 worker already active; refusing duplicate resume.' >&2
  exit 2
fi

mkdir -p "$ROOT/logs"
setsid /bin/bash -lc '
set -uo pipefail
ROOT=/root/autodl-tmp/cycle09_stage3_followup/H5_frozen_self
SCRIPT=/root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_stage3_frozen_self_postprocess.py
PY=/root/miniconda3/envs/density/bin/python
$PY "$SCRIPT" --phase worker --device cuda:0 --plan "behavior:5,geometry:5,behavior:40,geometry:40,behavior:160,geometry:80" > "$ROOT/logs/postprocess_resume_gpu0.log" 2>&1 &
P0=$!
$PY "$SCRIPT" --phase worker --device cuda:1 --plan "geometry_reference,behavior:20,geometry:20,behavior:80,geometry:160,behavior:320,geometry:320" > "$ROOT/logs/postprocess_resume_gpu1.log" 2>&1 &
P1=$!
printf "gpu0_pid=%s\ngpu1_pid=%s\n" "$P0" "$P1" > "$ROOT/logs/postprocess_resume_pids.txt"
wait "$P0"; R0=$?
wait "$P1"; R1=$?
printf "gpu0_rc=%s\ngpu1_rc=%s\n" "$R0" "$R1" >> "$ROOT/logs/postprocess_resume_pids.txt"
if [[ "$R0" -ne 0 || "$R1" -ne 0 ]]; then exit 1; fi
$PY "$SCRIPT" --phase behavior_finalize --device cuda:0 >> "$ROOT/logs/postprocess_resume_finalize.log" 2>&1
$PY "$SCRIPT" --phase geometry_finalize --device cuda:1 >> "$ROOT/logs/postprocess_resume_finalize.log" 2>&1
$PY "$SCRIPT" --phase total_effect --device cuda:0 >> "$ROOT/logs/postprocess_resume_finalize.log" 2>&1
' > "$ROOT/logs/postprocess_resume_supervisor.log" 2>&1 < /dev/null &

echo "H5 resume supervisor PID=$!"
