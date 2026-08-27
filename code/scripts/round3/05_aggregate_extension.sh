#!/usr/bin/env bash
# Merge legacy five-method evidence with the two-method DPO-reward extension.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
round3_require_baseline_experiment_id
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
OUTPUT_DIR="$EXPORT_ROOT/round3/$SOPPO_ROUND3_EXPERIMENT_ID"
mkdir -p "$OUTPUT_DIR"
"$ROUND3_TRAIN_PYTHON" -m src.round3.aggregate_extension \
    --repo-root "$SOPPO_ROOT" \
    --baseline-run-root "$ROUND3_BASELINE_RUN_ROOT" \
    --extension-run-root "$ROUND3_RUN_ROOT" \
    --output "$OUTPUT_DIR/round3_aggregate_v1.6.json"

