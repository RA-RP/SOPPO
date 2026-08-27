#!/usr/bin/env bash
# Precompute the common frozen-reference cache once on GPU0.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
CONFIG_METHOD="${1:-dpo_1k}"
CONFIG="$ROUND3_RUN_ROOT/resolved/strong_smoke/$CONFIG_METHOD.yaml"
[[ -f "$CONFIG" ]] || { echo "ERROR: resolve strong-smoke configs first" >&2; exit 1; }
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
if [[ -f "$ROUND3_REFERENCE_DIR/manifest.json" ]]; then
    echo "Reference cache exists and will be re-audited by each method preflight: $ROUND3_REFERENCE_DIR"
    exit 0
fi
INPUT_PREFLIGHT="$ROUND3_RUN_ROOT/control/reference_cache_input_preflight.json"
CUDA_VISIBLE_DEVICES=0,1,2 "$ROUND3_TRAIN_PYTHON" -m src.round3.preflight \
    --config "$CONFIG" --repo-root "$SOPPO_ROOT" --output "$INPUT_PREFLIGHT" --phase inputs
CUDA_VISIBLE_DEVICES=0 "$ROUND3_TRAIN_PYTHON" -m src.round3.reference_cache \
    --config "$CONFIG" --batch-size "${SOPPO_ROUND3_REFERENCE_BATCH_SIZE:-1}"
