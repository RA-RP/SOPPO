#!/usr/bin/env bash
# Produce only the sample-free Round3 aggregate export.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
OUTPUT_DIR="$EXPORT_ROOT/round3/$SOPPO_ROUND3_EXPERIMENT_ID"
mkdir -p "$OUTPUT_DIR"
"$ROUND3_TRAIN_PYTHON" -m src.round3.aggregate \
    --evaluations-root "$ROUND3_RUN_ROOT/evaluations" \
    --output "$OUTPUT_DIR/round3_aggregate.json"

