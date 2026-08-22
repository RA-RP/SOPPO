#!/bin/bash
# Run the complete authorized experiment sequentially on one dedicated 2-GPU server.
set -Eeuo pipefail

if [[ "${RUN_CONTEXT:-}" != "standalone" ]]; then
    echo "ERROR: RUN_CONTEXT=standalone is required" >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_STAGE_DIR="$(cd "$SCRIPT_DIR/../cluster" && pwd)"
source "$SCRIPT_DIR/server_paths.sh"
source "$SCRIPT_DIR/job_env.sh"

DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
MODEL_DIR="${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}"
PIPELINE_DIR="$RUN_ROOT/$EXPERIMENT_ID/pipeline"
REGISTRY="$PIPELINE_DIR/task_registry.json"
DRIVER_PID_FILE="$PIPELINE_DIR/driver.pid"
TRAIN_GPU_IDS="${SOPPO_TRAIN_GPU_IDS:-0,1}"
POST_GPU_ID="${SOPPO_POST_GPU_ID:-${TRAIN_GPU_IDS%%,*}}"
MINIMUM_MEMORY_MIB="${SOPPO_MIN_GPU_MEMORY_MIB:-79000}"

IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "$TRAIN_GPU_IDS"
if (( ${#TRAIN_GPU_ARRAY[@]} != 2 )); then
    echo "ERROR: v0.6 training contract requires exactly two GPU IDs" >&2
    exit 1
fi
for gpu_id in "${TRAIN_GPU_ARRAY[@]}" "$POST_GPU_ID"; do
    [[ "$gpu_id" =~ ^[0-9]+$ ]] || {
        echo "ERROR: GPU IDs must be non-negative integers: $gpu_id" >&2
        exit 1
    }
done
if [[ "${TRAIN_GPU_ARRAY[0]}" == "${TRAIN_GPU_ARRAY[1]}" ]]; then
    echo "ERROR: SOPPO_TRAIN_GPU_IDS contains a duplicate GPU ID" >&2
    exit 1
fi
[[ "$MINIMUM_MEMORY_MIB" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: SOPPO_MIN_GPU_MEMORY_MIB must be a positive integer" >&2
    exit 1
}

if [[ -e "$PIPELINE_DIR" ]]; then
    echo "ERROR: Refuse to overwrite pipeline directory: $PIPELINE_DIR" >&2
    exit 1
fi
for existing in main preexperiment lambda_search c_epsilon evaluation; do
    if [[ -e "$RUN_ROOT/$EXPERIMENT_ID/$existing" ]]; then
        echo "ERROR: Refuse to mix a new pipeline with existing output: $existing" >&2
        exit 1
    fi
done
if [[ -e "$EXPORT_ROOT/$EXPERIMENT_ID" ]]; then
    echo "ERROR: Refuse to overwrite existing export: $EXPORT_ROOT/$EXPERIMENT_ID" >&2
    exit 1
fi
if [[ -n "$(git -C "$SOPPO_ROOT" status --porcelain)" ]]; then
    echo "ERROR: Server SOPPO checkout must be clean before pipeline start" >&2
    exit 1
fi
GIT_COMMIT="$(git -C "$SOPPO_ROOT" rev-parse HEAD)"
if [[ -n "${SOPPO_EXPECTED_GIT_COMMIT:-}" && "$SOPPO_EXPECTED_GIT_COMMIT" != "$GIT_COMMIT" ]]; then
    echo "ERROR: Checkout changed between launcher and pipeline process" >&2
    exit 1
fi
export SOPPO_EXPECTED_GIT_COMMIT="$GIT_COMMIT"
export SOPPO_CLUSTER_SCRIPT_DIR="$SCRIPT_DIR"
export SOPPO_DATA_DIR="$DATA_DIR"
export SOPPO_MODEL_DIR="$MODEL_DIR"
soppo_job_init

test -f "$MODEL_DIR/model_manifest.json" || {
    echo "ERROR: Qwen3 model is not prepared: $MODEL_DIR" >&2
    exit 1
}
test -f "$DATA_DIR/manifest_public.json" || {
    echo "ERROR: 30k data is not prepared: $DATA_DIR" >&2
    exit 1
}
python -m src.model.model_manifest --model-dir "$MODEL_DIR" --verify
python -m src.data.audit_prepared_data --data-dir "$DATA_DIR"

mkdir -p "$PIPELINE_DIR/logs" "$PIPELINE_DIR/hardware"
printf '%s\n' "$$" > "$DRIVER_PID_FILE"
PIPELINE_PGID="$(ps -o pgid= -p $$ | tr -d ' ')"
STAGE_ORDER="preflight,tests,smoke,reference_cache,dpo_0,dpo_1,headroom_gate,static_0,static_1,static_2,static_3,static_select,dynamic_0,dynamic_1,c_epsilon_prepare,c_epsilon_0,c_epsilon_1,c_epsilon_2,c_epsilon_3,c_epsilon_4,c_epsilon_5,c_epsilon_6,c_epsilon_7,c_epsilon_8,c_epsilon_derive,evaluation_0,evaluation_1,evaluation_2,evaluation_3,evaluation_4,evaluation_5,evaluation_6,evaluation_7,aggregate"
python "$SCRIPT_DIR/pipeline_state.py" init \
    --path "$REGISTRY" --experiment "$EXPERIMENT_ID" --commit "$GIT_COMMIT" \
    --pid "$$" --pgid "$PIPELINE_PGID" --training-gpus "$TRAIN_GPU_IDS" \
    --post-gpu "$POST_GPU_ID" --minimum-memory-mib "$MINIMUM_MEMORY_MIB" \
    --stages "$STAGE_ORDER"

CURRENT_STAGE="preflight"
PIPELINE_FINISHED=0
finish_on_exit() {
    local status=$?
    trap - EXIT INT TERM
    if (( PIPELINE_FINISHED == 0 )) && [[ -f "$REGISTRY" ]]; then
        local final_state=failed
        if (( status == 130 || status == 143 )); then
            final_state=interrupted
        fi
        if [[ -n "$CURRENT_STAGE" ]]; then
            python "$SCRIPT_DIR/pipeline_state.py" set-stage \
                --path "$REGISTRY" --stage "$CURRENT_STAGE" --state "$final_state" \
                --exit-code "$status" 2>/dev/null || true
        fi
        python "$SCRIPT_DIR/pipeline_state.py" set-pipeline \
            --path "$REGISTRY" --state "$final_state" --current-stage "$CURRENT_STAGE" \
            2>/dev/null || true
        echo "ERROR: Standalone pipeline stopped at $CURRENT_STAGE with exit code $status" >&2
        if [[ -n "$CURRENT_STAGE" ]]; then
            echo "Inspect: $PIPELINE_DIR/logs/$CURRENT_STAGE.log" >&2
        fi
    fi
    exit "$status"
}
trap finish_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_stage() {
    local stage="${1:?stage name required}"
    local gpu_ids="${2-}"
    local processes="${3:?process count required}"
    local script_name="${4:?script name required}"
    shift 4
    local log_file="$PIPELINE_DIR/logs/$stage.log"
    CURRENT_STAGE="$stage"
    python "$SCRIPT_DIR/pipeline_state.py" set-stage \
        --path "$REGISTRY" --stage "$stage" --state running
    echo "=== Starting $stage at $(date -Iseconds) ==="
    set +e
    (
        export CUDA_VISIBLE_DEVICES="$gpu_ids"
        export SOPPO_NPROC_PER_NODE="$processes"
        export SOPPO_MIN_GPU_MEMORY_MIB="$MINIMUM_MEMORY_MIB"
        bash "$SHARED_STAGE_DIR/$script_name" "$@"
    ) 2>&1 | tee "$log_file"
    local status=${PIPESTATUS[0]}
    set -e
    if (( status != 0 )); then
        python "$SCRIPT_DIR/pipeline_state.py" set-stage \
            --path "$REGISTRY" --stage "$stage" --state failed --exit-code "$status"
        return "$status"
    fi
    python "$SCRIPT_DIR/pipeline_state.py" set-stage \
        --path "$REGISTRY" --stage "$stage" --state completed --exit-code 0
    echo "=== Completed $stage at $(date -Iseconds) ==="
    CURRENT_STAGE=""
}

echo "Standalone GPU routing: training=$TRAIN_GPU_IDS, postprocess=$POST_GPU_ID"
python "$SCRIPT_DIR/pipeline_state.py" set-stage \
    --path "$REGISTRY" --stage preflight --state running
set +e
(
    export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS"
    export SOPPO_NPROC_PER_NODE=2
    export SOPPO_MIN_GPU_MEMORY_MIB="$MINIMUM_MEMORY_MIB"
    soppo_hardware_gate "$PIPELINE_DIR/hardware/preflight_training.csv"
    export CUDA_VISIBLE_DEVICES="$POST_GPU_ID"
    export SOPPO_NPROC_PER_NODE=1
    soppo_hardware_gate "$PIPELINE_DIR/hardware/preflight_postprocess.csv"
) 2>&1 | tee "$PIPELINE_DIR/logs/preflight.log"
PREFLIGHT_STATUS=${PIPESTATUS[0]}
set -e
if (( PREFLIGHT_STATUS != 0 )); then
    python "$SCRIPT_DIR/pipeline_state.py" set-stage \
        --path "$REGISTRY" --stage preflight --state failed --exit-code "$PREFLIGHT_STATUS"
    exit "$PREFLIGHT_STATUS"
fi
python "$SCRIPT_DIR/pipeline_state.py" set-stage \
    --path "$REGISTRY" --stage preflight --state completed --exit-code 0
CURRENT_STAGE=""

run_stage tests "" 1 01_server_tests.sh
run_stage smoke "$TRAIN_GPU_IDS" 2 03_smoke.sh
run_stage reference_cache "$TRAIN_GPU_IDS" 2 02_finalize_inputs.sh
for index in 0 1; do
    run_stage "dpo_$index" "$TRAIN_GPU_IDS" 2 03_preexperiment.sh "$index"
done
run_stage headroom_gate "" 1 03_select_preexperiment.sh
for index in 0 1 2 3; do
    run_stage "static_$index" "$TRAIN_GPU_IDS" 2 04_lambda_search.sh "$index"
done
run_stage static_select "" 1 04_select_lambda.sh
for index in 0 1; do
    run_stage "dynamic_$index" "$TRAIN_GPU_IDS" 2 05_run_main.sh "$index"
done
run_stage c_epsilon_prepare "" 1 06_prepare_c_epsilon.sh
for index in 0 1 2 3 4 5 6 7 8; do
    run_stage "c_epsilon_$index" "$POST_GPU_ID" 1 06_c_epsilon.sh "$index"
done
run_stage c_epsilon_derive "" 1 06_derive_c_epsilon.sh
for index in 0 1 2 3 4 5 6 7; do
    run_stage "evaluation_$index" "$POST_GPU_ID" 1 07_evaluate.sh "$index"
done
run_stage aggregate "" 1 08_aggregate.sh

CURRENT_STAGE=""
python "$SCRIPT_DIR/pipeline_state.py" set-pipeline \
    --path "$REGISTRY" --state completed --current-stage ""
cp "$REGISTRY" "$EXPORT_ROOT/$EXPERIMENT_ID/task_registry.json"
PIPELINE_FINISHED=1
trap - EXIT INT TERM
echo "Standalone pipeline completed successfully."
echo "Export: $EXPORT_ROOT/$EXPERIMENT_ID"
