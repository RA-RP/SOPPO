#!/bin/bash
# Four preregistered normalized fixed-lambda SOPPO-PE trajectories.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init

INDEX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[[ "$INDEX" =~ ^[0-3]$ ]] || { echo "ERROR: static-lambda array index must be 0..3" >&2; exit 2; }
LAMBDAS=(0.1 0.3 0.5 1.0)
LAMBDA="${LAMBDAS[$INDEX]}"
DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
MODEL_DIR="${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}"
MANIFEST="$MODEL_DIR/model_manifest.json"
ROOT="$RUN_ROOT/$EXPERIMENT_ID/main"
soppo_hardware_gate "$RUN_ROOT/$EXPERIMENT_ID/pipeline/hardware/pe_static_${INDEX}.csv"

soppo_torchrun -m src.training.trainer \
    --config "$CODE_ROOT/configs/mvp/soppo_pe_static.yaml" \
    --set "model.name_or_path=$MODEL_DIR" --set "model.manifest_path=$MANIFEST" \
    --set "data.data_dir=$DATA_DIR" --set "method.fixed_lambda=$LAMBDA" \
    --set training.eval_steps=40 --set training.save_steps=40 \
    --set "output.run_dir=$ROOT/soppo_pe_static_lambda_$LAMBDA"
