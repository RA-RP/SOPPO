#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init
DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
ROOT="$RUN_ROOT/$EXPERIMENT_ID/c_epsilon"
python -m src.evaluation.c_epsilon prepare --data-dir "$DATA_DIR" --output "$ROOT/inputs"
