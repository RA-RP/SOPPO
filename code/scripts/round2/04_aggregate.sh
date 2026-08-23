#!/usr/bin/env bash
# Build a sample-free Round2 export; first-round merge remains a later read-only step.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"

EXPORT_DIR="$EXPORT_ROOT/$ROUND2_EXPERIMENT_ID"
"$ROUND2_TRAIN_PYTHON" -m src.round2.aggregate \
    --experiment "$ROUND2_RUN_ROOT" \
    --export "$EXPORT_DIR"

echo "Round2 aggregate ready: $EXPORT_DIR"
