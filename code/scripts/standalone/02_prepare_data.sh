#!/bin/bash
# Download, split, and audit the immutable 30k UltraFeedback input.
set -euo pipefail

if [[ "${RUN_CONTEXT:-}" != "standalone" ]]; then
    echo "ERROR: RUN_CONTEXT=standalone is required" >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/server_paths.sh"
source "$SCRIPT_DIR/runtime_env.sh"
soppo_activate_env "$ENV_ROOT/youc"
soppo_set_cache_env
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
if [[ -f "$DATA_DIR/manifest_public.json" ]]; then
    python -m src.data.audit_prepared_data --data-dir "$DATA_DIR"
    echo "Verified existing immutable 30k dataset: $DATA_DIR"
    exit 0
fi
if [[ -e "$DATA_DIR" ]] && find "$DATA_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR: Incomplete non-empty data directory exists: $DATA_DIR" >&2
    exit 1
fi
mkdir -p "$DATA_DIR"

python "$CODE_ROOT/src/data/prepare_ultrafeedback.py" \
    --output_dir "$DATA_DIR" \
    --total_samples 30000 \
    --labeled_train_samples 2700 \
    --labeled_val_samples 300 \
    --unlabeled_samples 24000 \
    --test_samples 3000 \
    --seed 42 \
    --dataset_name openbmb/UltraFeedback \
    --max_seq_len 2048
python -m src.data.audit_prepared_data --data-dir "$DATA_DIR"
echo "Immutable 30k dataset ready: $DATA_DIR"
