#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init
ROOT="$RUN_ROOT/$EXPERIMENT_ID/c_epsilon"
python -m src.evaluation.c_epsilon derive \
    --base "$ROOT/raw/base" \
    --methods-root "$ROOT/raw/methods" \
    --output "$ROOT/aggregate"
