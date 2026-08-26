#!/usr/bin/env bash
# Five one-step production-path smokes, followed by storage projection.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
bash "$SCRIPT_DIR/01_resolve_all.sh" strong_smoke
bash "$SCRIPT_DIR/02_server_tests.sh"
bash "$SCRIPT_DIR/02_prepare_reference_cache.sh"
for method in \
    dpo_1k \
    sspo_code_loss_stratified_ultrachat_2df9e9a \
    dpo_8k \
    dpo_pe_sft_rollout \
    dpo_pe_rollout_only; do
    bash "$SCRIPT_DIR/run_method.sh" "$method" strong_smoke
    CUDA_VISIBLE_DEVICES=0 "$ROUND3_TRAIN_PYTHON" -m src.round3.verify_checkpoint \
        --config "$ROUND3_RUN_ROOT/resolved/strong_smoke/$method.yaml" \
        --checkpoint "$ROUND3_RUN_ROOT/strong_smoke/$method/smoke_checkpoint/step_000001" \
        --output "$ROUND3_RUN_ROOT/control/strong_smoke/$method/checkpoint_verified.json"
done
"$ROUND3_TRAIN_PYTHON" -m src.round3.project_storage \
    --smoke-root "$ROUND3_RUN_ROOT/strong_smoke" \
    --model-dir "$ROUND3_MODEL_DIR" \
    --data-dir "$ROUND3_DATA_DIR" \
    --reference-dir "$ROUND3_REFERENCE_DIR" \
    --train-env "$ROUND3_TRAIN_ENV" \
    --rollout-env "$ROUND3_ROLLOUT_ENV" \
    --platform-log-root "$ROUND3_LOG_ROOT" \
    --output "$ROUND3_RUN_ROOT/storage_projection.json"
