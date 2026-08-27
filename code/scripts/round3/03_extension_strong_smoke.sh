#!/usr/bin/env bash
# Two DPO-reward production-path smokes after the legacy formal reaches terminal.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
round3_require_baseline_experiment_id
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
[[ -f "$ROUND3_RUN_ROOT/baseline_link.json" ]] || {
    echo "ERROR: run 00_reuse_baseline_revisions.sh first" >&2
    exit 1
}
bash "$SCRIPT_DIR/01_resolve_all.sh" strong_smoke extension
bash "$SCRIPT_DIR/02_server_tests.sh"
bash "$SCRIPT_DIR/02_prepare_reference_cache.sh" dpo_pe_dpo_reward_sft_rollout

for method in \
    dpo_pe_dpo_reward_sft_rollout \
    dpo_pe_dpo_reward_rollout_only; do
    config="$ROUND3_RUN_ROOT/resolved/strong_smoke/$method.yaml"
    bash "$SCRIPT_DIR/run_method.sh" "$method" strong_smoke
    train_gpu="$(round3_resolved_value "$config" training.train_gpu)"
    CUDA_VISIBLE_DEVICES="$train_gpu" "$ROUND3_TRAIN_PYTHON" -m src.round3.verify_checkpoint \
        --config "$config" \
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
    --methods dpo_pe_dpo_reward_sft_rollout dpo_pe_dpo_reward_rollout_only \
    --output "$ROUND3_RUN_ROOT/storage_projection.json"

