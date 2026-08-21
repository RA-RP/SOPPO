#!/bin/bash
set -euo pipefail
SCRIPT_DIR="${SOPPO_CLUSTER_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init
EXPERIMENT_ROOT="$RUN_ROOT/$EXPERIMENT_ID"
python -m src.training.selectors headroom \
    --main-root "$EXPERIMENT_ROOT/main" \
    --output "$EXPERIMENT_ROOT/preexperiment/selection"
