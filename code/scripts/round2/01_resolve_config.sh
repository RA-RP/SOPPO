#!/usr/bin/env bash
# Resolve one round2 config without mutating the source YAML.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/round2_env.sh"

CONFIG_NAME="${1:-soppo_pe_sft_rollout_exp.yaml}"
CONFIG_PATH="$ROUND2_CONFIG_DIR/$CONFIG_NAME"
[[ -f "$CONFIG_PATH" ]] || { echo "ERROR: config not found: $CONFIG_PATH" >&2; exit 1; }
[[ -x "$ROUND2_TRAIN_PYTHON" ]] || {
    echo "ERROR: Round2 training Python missing: $ROUND2_TRAIN_PYTHON" >&2
    exit 1
}
if [[ -n "$(git -C "$SOPPO_ROOT" status --porcelain)" ]]; then
    echo "ERROR: resolve round2 only from a clean, reviewed SOPPO commit" >&2
    exit 1
fi
GIT_COMMIT="$(git -C "$SOPPO_ROOT" rev-parse HEAD)"

MODEL_DIR="${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}"
DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
SFT_DATA_FILE="${SOPPO_ROUND2_SFT_DATA_FILE:-$DATA_ROOT/round2/mvp-v0.5-30k/sft_anchor_response_a/sft_anchor.jsonl}"
TRAIN_GPU_IDS="${SOPPO_ROUND2_TRAIN_GPU_IDS:-0,1}"
ROLLOUT_GPU_IDS="${SOPPO_ROUND2_ROLLOUT_GPU_IDS:-2}"

[[ "$SFT_DATA_FILE" = /* ]] || {
    echo "ERROR: SOPPO_ROUND2_SFT_DATA_FILE must be an absolute server path" >&2
    exit 1
}
[[ -f "$SFT_DATA_FILE" ]] || {
    echo "ERROR: SFT data file is missing: $SFT_DATA_FILE" >&2
    exit 1
}

METHOD_NAME="${CONFIG_NAME%.yaml}"
RUN_DIR="${SOPPO_ROUND2_RUN_DIR:-$ROUND2_RUN_ROOT/$METHOD_NAME}"
ARTIFACT_DIR="$RUN_DIR/rollouts"
RESOLVED="$RUN_DIR/config.resolved.yaml"
if [[ -e "$RESOLVED" ]]; then
    echo "ERROR: Refuse to overwrite resolved config: $RESOLVED" >&2
    exit 1
fi

OVERRIDES=(
    --override "provenance.git_commit=$GIT_COMMIT"
    --override "model.name_or_path=$MODEL_DIR"
    --override "model.manifest_path=$MODEL_DIR/model_manifest.json"
    --override "data.data_dir=$DATA_DIR"
    --override "output.run_dir=$RUN_DIR"
    --override "tensor_parallel.gpu_ids=$TRAIN_GPU_IDS"
    --override "rollout.gpu_ids=$ROLLOUT_GPU_IDS"
    --override "rollout.artifact_dir=$ARTIFACT_DIR"
    --override "rollout.sft_data_file=$SFT_DATA_FILE"
)
if [[ -n "${SOPPO_ROUND2_MAX_STEPS:-}" ]]; then
    OVERRIDES+=(--override "training.max_steps=$SOPPO_ROUND2_MAX_STEPS")
fi
if [[ -n "${SOPPO_ROUND2_SMOKE_MODE:-}" ]]; then
    OVERRIDES+=(--override "training.smoke_mode=$SOPPO_ROUND2_SMOKE_MODE")
fi
if [[ -n "${SOPPO_ROUND2_EVAL_MAX_SAMPLES:-}" ]]; then
    OVERRIDES+=(--override "training.eval_max_samples=$SOPPO_ROUND2_EVAL_MAX_SAMPLES")
fi
if [[ -n "${SOPPO_ROUND2_MIN_NEW_TOKENS:-}" ]]; then
    OVERRIDES+=(--override "rollout.min_new_tokens=$SOPPO_ROUND2_MIN_NEW_TOKENS")
fi
if [[ -n "${SOPPO_ROUND2_IGNORE_EOS:-}" ]]; then
    OVERRIDES+=(--override "rollout.ignore_eos=$SOPPO_ROUND2_IGNORE_EOS")
fi
if [[ -n "${SOPPO_ROUND2_SMOKE_OBJECTIVE_STEP:-}" ]]; then
    OVERRIDES+=(--override "training.smoke_objective_step=$SOPPO_ROUND2_SMOKE_OBJECTIVE_STEP")
fi

export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
# Validate first so a malformed contract does not leave a fake resolved config.
"$ROUND2_TRAIN_PYTHON" -m src.round2.validate_config \
    --config "$CONFIG_PATH" "${OVERRIDES[@]}" >/dev/null
mkdir -p "$RUN_DIR"
"$ROUND2_TRAIN_PYTHON" -m src.round2.validate_config \
    --config "$CONFIG_PATH" "${OVERRIDES[@]}" \
    --write-resolved "$RESOLVED"

echo "Resolved round2 config: $RESOLVED"
