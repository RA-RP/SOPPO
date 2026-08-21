#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init
EXPERIMENT_ROOT="$RUN_ROOT/$EXPERIMENT_ID"
python -m src.training.selectors lambda \
    --main-root "$EXPERIMENT_ROOT/main" \
    --headroom "$EXPERIMENT_ROOT/preexperiment/selection/headroom_selection.json" \
    --output "$EXPERIMENT_ROOT/lambda_search/selection"
