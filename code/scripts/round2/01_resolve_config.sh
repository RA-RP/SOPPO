#!/usr/bin/env bash
# Resolve one round2 config without mutating the source YAML.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/round2_env.sh"

CONFIG_NAME="${1:-soppo_pe_sft_rollout_exp.yaml}"
CONFIG_PATH="$ROUND2_CONFIG_DIR/$CONFIG_NAME"
[[ -f "$CONFIG_PATH" ]] || { echo "ERROR: config not found: $CONFIG_PATH" >&2; exit 1; }
[[ -x "$ROUND2_PYTHON" ]] || { echo "ERROR: Python missing: $ROUND2_PYTHON" >&2; exit 1; }

MODEL_DIR="${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}"
DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
MEGATRON_ENTRYPOINT="${SOPPO_MEGATRON_ENTRYPOINT:-}"
MEGATRON_WORKING_DIR="${SOPPO_MEGATRON_WORKING_DIR:-}"
ROLLOUT_ENTRYPOINT="${SOPPO_ROLLOUT_ENTRYPOINT:-}"
ROLLOUT_WORKING_DIR="${SOPPO_ROLLOUT_WORKING_DIR:-}"
TRAIN_GPU_IDS="${SOPPO_ROUND2_TRAIN_GPU_IDS:-0,1}"
ROLLOUT_GPU_IDS="${SOPPO_ROUND2_ROLLOUT_GPU_IDS:-2}"

for pair in \
    "SOPPO_MEGATRON_ENTRYPOINT:$MEGATRON_ENTRYPOINT" \
    "SOPPO_MEGATRON_WORKING_DIR:$MEGATRON_WORKING_DIR" \
    "SOPPO_ROLLOUT_ENTRYPOINT:$ROLLOUT_ENTRYPOINT" \
    "SOPPO_ROLLOUT_WORKING_DIR:$ROLLOUT_WORKING_DIR"; do
    name="${pair%%:*}"
    value="${pair#*:}"
    if [[ -z "$value" ]]; then
        echo "ERROR: set $name before resolving round2 config" >&2
        exit 1
    fi
done

RUN_DIR="$ROUND2_RUN_ROOT/$CONFIG_NAME"
ARTIFACT_DIR="$RUN_DIR/rollouts"
ARTIFACT_PATH="$ARTIFACT_DIR/rollout.jsonl"
mkdir -p "$RUN_DIR"
RESOLVED="$RUN_DIR/config.resolved.yaml"
if [[ -e "$RESOLVED" ]]; then
    echo "ERROR: Refuse to overwrite resolved config: $RESOLVED" >&2
    exit 1
fi

export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
"$ROUND2_PYTHON" -m src.round2.validate_config \
    --config "$CONFIG_PATH" \
    --override "model.name_or_path=$MODEL_DIR" \
    --override "model.manifest_path=$MODEL_DIR/model_manifest.json" \
    --override "data.data_dir=$DATA_DIR" \
    --override "output.run_dir=$RUN_DIR" \
    --override "megatron.entrypoint=$MEGATRON_ENTRYPOINT" \
    --override "megatron.working_dir=$MEGATRON_WORKING_DIR" \
    --override "megatron.gpu_ids=$TRAIN_GPU_IDS" \
    --override "megatron.data_parallel_size=$(awk -F, '{print NF}' <<< "$TRAIN_GPU_IDS")" \
    --override "rollout.entrypoint=$ROLLOUT_ENTRYPOINT" \
    --override "rollout.working_dir=$ROLLOUT_WORKING_DIR" \
    --override "rollout.gpu_ids=$ROLLOUT_GPU_IDS" \
    --override "rollout.artifact_dir=$ARTIFACT_DIR" \
    --override "rollout.artifact_path=$ARTIFACT_PATH" \
    --override "rollout.policy_checkpoint=$RUN_DIR/checkpoints/final" \
    --write-resolved "$RESOLVED"

echo "Resolved round2 config: $RESOLVED"
