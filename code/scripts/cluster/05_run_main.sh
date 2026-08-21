#!/bin/bash
# Two dynamic trajectories; together with stages 03/04 these form eight final runs.
set -euo pipefail
SCRIPT_DIR="${SOPPO_CLUSTER_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init

INDEX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[[ "$INDEX" =~ ^[0-1]$ ]] || { echo "ERROR: dynamic array index must be 0..1" >&2; exit 2; }
DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
MODEL_DIR="${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}"
MANIFEST="$MODEL_DIR/model_manifest.json"
ROOT="$RUN_ROOT/$EXPERIMENT_ID/main"
soppo_hardware_gate "$RUN_ROOT/$EXPERIMENT_ID/pipeline/hardware/dynamic_${INDEX}.csv"

case "$INDEX" in
    0) CONFIG=sspo_hard_exp; RUN_NAME=sspo_hard_exp ;;
    1) CONFIG=soppo_pe_exp; RUN_NAME=soppo_pe_exp ;;
esac
soppo_torchrun -m src.training.trainer \
    --config "$CODE_ROOT/configs/mvp/$CONFIG.yaml" \
    --set "model.name_or_path=$MODEL_DIR" --set "model.manifest_path=$MANIFEST" \
    --set "data.data_dir=$DATA_DIR" \
    --set training.eval_steps=40 --set training.save_steps=40 \
    --set "output.run_dir=$ROOT/$RUN_NAME"
