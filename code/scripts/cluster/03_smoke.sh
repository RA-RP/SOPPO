#!/bin/bash
# Strong 1/2/4-rank GPU smoke; routing is selected by the pipeline launcher.
set -euo pipefail

SCRIPT_DIR="${SOPPO_CLUSTER_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init

DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
MODEL_DIR="${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}"
MODEL_MANIFEST="$MODEL_DIR/model_manifest.json"
SMOKE_ROOT="$RUN_ROOT/$EXPERIMENT_ID/pipeline/smoke"
WORLD_SIZE="${SOPPO_NPROC_PER_NODE:-2}"
case "$WORLD_SIZE" in
    1|2|4) ;;
    *)
        echo "ERROR: smoke supports exactly 1, 2, or 4 ranks; got $WORLD_SIZE" >&2
        exit 1
        ;;
esac
DPO_SMOKE_GLOBAL_BATCH=$((4 * WORLD_SIZE))
JOINT_SMOKE_GLOBAL_BATCH=$((8 * WORLD_SIZE))
JOINT_SMOKE_LABELED_GLOBAL_BATCH=$WORLD_SIZE
JOINT_SMOKE_UNLABELED_GLOBAL_BATCH=$((7 * WORLD_SIZE))
soppo_hardware_gate "$RUN_ROOT/$EXPERIMENT_ID/pipeline/hardware/smoke.csv"
python -m src.model.model_manifest --model-dir "$MODEL_DIR" --verify
python -m src.training.smoke_fixture \
    --data-dir "$DATA_DIR" --output "$SMOKE_ROOT/data" \
    --model "$MODEL_DIR" --max-length 2048 --world-size "$WORLD_SIZE"

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
    --max-length 2048 --batch-size 1 --dtype bfloat16
python -m src.training.cache_tools split \
    --combined-cache "$SMOKE_ROOT/reference/combined.ref.jsonl" \
    --target "$SMOKE_ROOT/data/labeled_train.jsonl" \
    --target "$SMOKE_ROOT/data/labeled_val.jsonl" \
    --target "$SMOKE_ROOT/data/oracle_train.private.jsonl" \
    --output-dir "$SMOKE_ROOT/reference/targets" \
    --model-manifest "$MODEL_MANIFEST" --max-length 2048

run_dpo_smoke() {
    local config_name="${1:?config required}"
    local run_name="${2:?run name required}"
    shift 2
    soppo_torchrun -m src.training.trainer \
        --config "$CODE_ROOT/configs/mvp/$config_name.yaml" \
        --set "model.name_or_path=$MODEL_DIR" --set "model.manifest_path=$MODEL_MANIFEST" \
        --set "data.data_dir=$SMOKE_ROOT/data" \
        --set "data.reference_cache=$SMOKE_ROOT/reference/targets" --set data.num_workers=0 \
        --set "training.num_devices=$WORLD_SIZE" \
        --set training.epochs=1 --set training.max_steps=2 \
        --set training.dpo_batch_size_per_device=4 \
        --set training.gradient_accumulation_steps=1 \
        --set "training.global_batch_size=$DPO_SMOKE_GLOBAL_BATCH" \
        --set training.backward_subbatch_size_per_device=2 \
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
        --set "data.data_dir=$SMOKE_ROOT/data" --set data.num_workers=0 \
        --set "training.num_devices=$WORLD_SIZE" \
        --set training.epochs=1 --set training.max_steps=1 \
        --set training.dpo_batch_size_per_device=4 \
        --set training.gradient_accumulation_steps=2 \
        --set "training.global_batch_size=$JOINT_SMOKE_GLOBAL_BATCH" \
        --set training.backward_subbatch_size_per_device=2 \
        --set training.joint_labeled_batch_size_per_device=1 \
        --set 'training.joint_labeled_microsteps=[0]' \
        --set 'training.joint_unlabeled_microbatch_pattern=[3,4]' \
        --set "training.joint_labeled_global_batch_size=$JOINT_SMOKE_LABELED_GLOBAL_BATCH" \
        --set "training.joint_unlabeled_global_batch_size=$JOINT_SMOKE_UNLABELED_GLOBAL_BATCH" \
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
    --set "data.data_dir=$SMOKE_ROOT/data" --set data.num_workers=0 \
    --set "training.num_devices=$WORLD_SIZE" \
    --set training.epochs=1 --set training.max_steps=1 \
    --set training.dpo_batch_size_per_device=4 \
    --set training.gradient_accumulation_steps=2 \
    --set "training.global_batch_size=$JOINT_SMOKE_GLOBAL_BATCH" \
    --set training.backward_subbatch_size_per_device=2 \
    --set training.joint_labeled_batch_size_per_device=1 \
    --set 'training.joint_labeled_microsteps=[0]' \
    --set 'training.joint_unlabeled_microbatch_pattern=[3,4]' \
    --set "training.joint_labeled_global_batch_size=$JOINT_SMOKE_LABELED_GLOBAL_BATCH" \
    --set "training.joint_unlabeled_global_batch_size=$JOINT_SMOKE_UNLABELED_GLOBAL_BATCH" \
    --set training.smoke_mode=true --set training.eval_steps=1 --set training.save_steps=1 \
    --set output.save_checkpoints=false \
    --set "output.run_dir=$SMOKE_ROOT/runs/checkpoint_roundtrip"

for required in dpo10 dpo100 sspo_hard_exp soppo_pe_exp soppo_pe_static_lambda_0.5 checkpoint_roundtrip; do
    test -f "$SMOKE_ROOT/runs/$required/complete.json"
done
test -f "$PE_CHECKPOINT/adapter_config.json"
touch "$SMOKE_ROOT/SMOKE_COMPLETE"
echo "Strong v0.6 smoke completed successfully: $SMOKE_ROOT"
