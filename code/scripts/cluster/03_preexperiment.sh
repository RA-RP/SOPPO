#!/bin/bash
# Final DPO trajectories; DPO-10 records its frozen-base validation baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init

INDEX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[[ "$INDEX" =~ ^[0-1]$ ]] || { echo "ERROR: DPO array index must be 0..1" >&2; exit 2; }
DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
MODEL_DIR="${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}"
MANIFEST="$MODEL_DIR/model_manifest.json"
CACHE_DIR="${SOPPO_REFERENCE_CACHE:-$CACHE_ROOT/soppo/reference/qwen3-4b-mvp-v0.5-30k}/targets"
ROOT="$RUN_ROOT/$EXPERIMENT_ID/main"
soppo_hardware_gate "$RUN_ROOT/$EXPERIMENT_ID/pipeline/hardware/dpo_${INDEX}.csv"

case "$INDEX" in
    0) CONFIG=dpo10; RUN_NAME=dpo10; EXTRA=() ;;
    1) CONFIG=dpo100; RUN_NAME=dpo100; EXTRA=(--set "data.train_file=$DATA_DIR/oracle_train.private.jsonl") ;;
esac
soppo_torchrun -m src.training.trainer \
    --config "$CODE_ROOT/configs/mvp/$CONFIG.yaml" \
    --set "model.name_or_path=$MODEL_DIR" --set "model.manifest_path=$MANIFEST" \
    --set "data.data_dir=$DATA_DIR" --set "data.reference_cache=$CACHE_DIR" \
    --set training.eval_steps=20 --set training.save_steps=20 \
    --set "output.run_dir=$ROOT/$RUN_NAME" "${EXTRA[@]}"
