#!/bin/bash
# Strong two-rank GPU smoke; routing is selected by submit_all.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init

DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
MODEL_DIR="${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}"
MODEL_MANIFEST="$MODEL_DIR/model_manifest.json"
SMOKE_ROOT="$RUN_ROOT/$EXPERIMENT_ID/pipeline/smoke"
soppo_hardware_gate "$RUN_ROOT/$EXPERIMENT_ID/pipeline/hardware/smoke.csv"
python -m src.model.model_manifest --model-dir "$MODEL_DIR" --verify
python -m src.training.smoke_fixture --data-dir "$DATA_DIR" --output "$SMOKE_ROOT/data"

mkdir -p "$SMOKE_ROOT/reference/targets"
python -m src.training.cache_tools combine \
    --input "$SMOKE_ROOT/data/labeled_train.jsonl" \
    --input "$SMOKE_ROOT/data/labeled_val.jsonl" \
    --input "$SMOKE_ROOT/data/oracle_train.private.jsonl" \
    --output "$SMOKE_ROOT/reference/cache_source.jsonl"
soppo_torchrun -m src.training.reference_cache \
    --model "$MODEL_DIR" --model-manifest "$MODEL_MANIFEST" \
    --input "$SMOKE_ROOT/reference/cache_source.jsonl" \
    --output "$SMOKE_ROOT/reference/combined.ref.jsonl" \
    --max-length 256 --batch-size 2 --dtype float16
python -m src.training.cache_tools split \
    --combined-cache "$SMOKE_ROOT/reference/combined.ref.jsonl" \
    --target "$SMOKE_ROOT/data/labeled_train.jsonl" \
    --target "$SMOKE_ROOT/data/labeled_val.jsonl" \
    --target "$SMOKE_ROOT/data/oracle_train.private.jsonl" \
    --output-dir "$SMOKE_ROOT/reference/targets" \
    --model-manifest "$MODEL_MANIFEST" --max-length 256

run_dpo_smoke() {
    local config_name="${1:?config required}"
    local run_name="${2:?run name required}"
    shift 2
    soppo_torchrun -m src.training.trainer \
        --config "$CODE_ROOT/configs/mvp/$config_name.yaml" \
        --set "model.name_or_path=$MODEL_DIR" --set "model.manifest_path=$MODEL_MANIFEST" \
        --set model.torch_dtype=float16 --set model.max_seq_len=256 \
        --set "data.data_dir=$SMOKE_ROOT/data" \
        --set "data.reference_cache=$SMOKE_ROOT/reference/targets" --set data.num_workers=0 \
        --set training.epochs=1 --set training.max_steps=1 \
        --set training.dpo_batch_size_per_device=1 \
        --set training.gradient_accumulation_steps=2 --set training.global_batch_size=4 \
        --set training.smoke_mode=true --set training.eval_steps=1 --set training.save_steps=1 \
        --set "output.run_dir=$SMOKE_ROOT/runs/$run_name" "$@"
}

run_joint_smoke() {
    local config_name="${1:?config required}"
    local run_name="${2:?run name required}"
    shift 2
    soppo_torchrun -m src.training.trainer \
        --config "$CODE_ROOT/configs/mvp/$config_name.yaml" \
        --set "model.name_or_path=$MODEL_DIR" --set "model.manifest_path=$MODEL_MANIFEST" \
        --set model.torch_dtype=float16 --set model.max_seq_len=256 \
        --set "data.data_dir=$SMOKE_ROOT/data" --set data.num_workers=0 \
        --set training.epochs=1 --set training.max_steps=1 \
        --set training.gradient_accumulation_steps=2 --set training.global_batch_size=6 \
        --set training.joint_labeled_batch_size_per_device=1 \
        --set 'training.joint_labeled_microsteps=[0]' \
        --set 'training.joint_unlabeled_microbatch_pattern=[1,1]' \
        --set training.joint_labeled_global_batch_size=2 \
        --set training.joint_unlabeled_global_batch_size=4 \
        --set training.smoke_mode=true --set training.eval_steps=1 --set training.save_steps=1 \
        --set "output.run_dir=$SMOKE_ROOT/runs/$run_name" "$@"
}

run_dpo_smoke dpo10 dpo10 --set output.save_checkpoints=false
run_dpo_smoke dpo100 dpo100 --set "data.train_file=$SMOKE_ROOT/data/oracle_train.private.jsonl" \
    --set output.save_checkpoints=false
run_joint_smoke sspo_hard_exp sspo_hard_exp --set output.save_checkpoints=false
run_joint_smoke soppo_pe_exp soppo_pe_exp --set output.save_checkpoints=true
run_joint_smoke soppo_pe_static soppo_pe_static_lambda_0.5 \
    --set method.fixed_lambda=0.5 --set output.save_checkpoints=false

PE_CHECKPOINT="$SMOKE_ROOT/runs/soppo_pe_exp/checkpoints/step_000001"
soppo_torchrun -m src.training.trainer \
    --config "$CODE_ROOT/configs/mvp/soppo_pe_exp.yaml" --init-checkpoint "$PE_CHECKPOINT" \
    --set "model.name_or_path=$MODEL_DIR" --set "model.manifest_path=$MODEL_MANIFEST" \
    --set model.torch_dtype=float16 --set model.max_seq_len=256 \
    --set "data.data_dir=$SMOKE_ROOT/data" --set data.num_workers=0 \
    --set training.epochs=1 --set training.max_steps=1 \
    --set training.gradient_accumulation_steps=2 --set training.global_batch_size=6 \
    --set training.joint_labeled_batch_size_per_device=1 \
    --set 'training.joint_labeled_microsteps=[0]' \
    --set 'training.joint_unlabeled_microbatch_pattern=[1,1]' \
    --set training.joint_labeled_global_batch_size=2 \
    --set training.joint_unlabeled_global_batch_size=4 \
    --set training.smoke_mode=true --set training.eval_steps=1 --set training.save_steps=1 \
    --set output.save_checkpoints=false \
    --set "output.run_dir=$SMOKE_ROOT/runs/checkpoint_roundtrip"

for required in dpo10 dpo100 sspo_hard_exp soppo_pe_exp soppo_pe_static_lambda_0.5 checkpoint_roundtrip; do
    test -f "$SMOKE_ROOT/runs/$required/complete.json"
done
test -f "$PE_CHECKPOINT/adapter_config.json"
touch "$SMOKE_ROOT/SMOKE_COMPLETE"
echo "Strong v0.6 smoke completed successfully: $SMOKE_ROOT"
