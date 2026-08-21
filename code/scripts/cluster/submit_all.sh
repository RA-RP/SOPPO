#!/bin/bash
# Submit the complete fail-closed Slurm DAG once. Server execution authorization is required.
set -Eeuo pipefail

if [[ "${RUN_CONTEXT:-}" != "cluster" ]]; then
    echo "ERROR: RUN_CONTEXT=cluster is required" >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init

DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
MODEL_DIR="${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}"
PIPELINE_DIR="$RUN_ROOT/$EXPERIMENT_ID/pipeline"
REGISTRY="$PIPELINE_DIR/task_registry.json"
GPU_PARTITION="${SOPPO_GPU_PARTITION:-gpu}"
SMOKE_PARTITION="${SOPPO_SMOKE_PARTITION:-$GPU_PARTITION}"
CPU_PARTITION="${SOPPO_CPU_PARTITION:-$GPU_PARTITION}"
GPU2_GRES="${SOPPO_GPU2_GRES:-gpu:tesla:2}"
GPU1_GRES="${SOPPO_GPU1_GRES:-gpu:tesla:1}"
AUX_GRES="${SOPPO_AUX_GRES:-$GPU1_GRES}"
ARRAY_LIMIT="${SOPPO_ARRAY_PARALLELISM:-4}"

command -v sbatch >/dev/null || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
command -v scancel >/dev/null || { echo "ERROR: scancel is unavailable" >&2; exit 1; }
test -f "$MODEL_DIR/model_manifest.json" || { echo "ERROR: Qwen3 is not downloaded/frozen" >&2; exit 1; }
test -f "$DATA_DIR/manifest_public.json" || { echo "ERROR: 30k data is not prepared: $DATA_DIR" >&2; exit 1; }
python -m src.model.model_manifest --model-dir "$MODEL_DIR" --verify
python -m src.data.audit_prepared_data --data-dir "$DATA_DIR"
if [[ -e "$PIPELINE_DIR" ]]; then
    echo "ERROR: Refuse to overwrite pipeline directory: $PIPELINE_DIR" >&2
    exit 1
fi
if [[ -n "$(git -C "$SOPPO_ROOT" status --porcelain)" ]]; then
    echo "ERROR: Server SOPPO checkout must be clean before submission" >&2
    exit 1
fi
GIT_COMMIT="$(git -C "$SOPPO_ROOT" rev-parse HEAD)"
mkdir -p "$PIPELINE_DIR/logs" "$PIPELINE_DIR/hardware"

echo "Slurm routing: auxiliary=$CPU_PARTITION/$AUX_GRES, smoke=$SMOKE_PARTITION/$GPU2_GRES, formal=$GPU_PARTITION/$GPU2_GRES"

declare -a SUBMITTED_JOBS=()
submission_failed() {
    local status=$?
    trap - ERR INT TERM
    (( status != 0 )) || status=1
    echo "ERROR: DAG submission stopped; rolling back jobs submitted by this invocation." >&2
    if (( ${#SUBMITTED_JOBS[@]} > 0 )); then
        printf '  cancelling %s\n' "${SUBMITTED_JOBS[@]}" >&2
        scancel "${SUBMITTED_JOBS[@]}" 2>/dev/null || true
        echo "Pipeline directory retained for failure inspection: $PIPELINE_DIR" >&2
    else
        rmdir "$PIPELINE_DIR/logs" "$PIPELINE_DIR/hardware" "$PIPELINE_DIR" 2>/dev/null || true
    fi
    exit "$status"
}
trap submission_failed ERR INT TERM

submit() {
    local stage="${1:?stage required}"
    shift
    local raw
    if ! raw="$(sbatch --parsable "$@")"; then
        echo "ERROR: sbatch rejected stage: $stage" >&2
        return 1
    fi
    LAST_JOB_ID="${raw%%;*}"
    [[ "$LAST_JOB_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid sbatch job id: $raw" >&2; return 1; }
    SUBMITTED_JOBS+=("$LAST_JOB_ID")
    echo "Submitted $stage as Slurm job $LAST_JOB_ID" >&2
}

COMMON_EXPORT="ALL,RUN_CONTEXT=cluster,EXPERIMENT_ID=$EXPERIMENT_ID,SOPPO_DATA_DIR=$DATA_DIR,SOPPO_MODEL_DIR=$MODEL_DIR"
submit tests -J soppo-tests -p "$CPU_PARTITION" -N 1 -c 8 --gres="$AUX_GRES" -t 01:00:00 \
    -o "$PIPELINE_DIR/logs/tests-%j.out" --export="$COMMON_EXPORT" "$SCRIPT_DIR/01_server_tests.sh"
TESTS="$LAST_JOB_ID"
submit smoke -J soppo-smoke -p "$SMOKE_PARTITION" -N 1 -c 32 --gres="$GPU2_GRES" -t 00:50:00 \
    -d "afterok:$TESTS" -o "$PIPELINE_DIR/logs/smoke-%j.out" \
    --export="$COMMON_EXPORT,SOPPO_NPROC_PER_NODE=2,SOPPO_REQUIRE_A800=0" "$SCRIPT_DIR/03_smoke.sh"
SMOKE="$LAST_JOB_ID"
submit reference_cache -J soppo-ref -p "$GPU_PARTITION" -N 1 -c 32 --gres="$GPU2_GRES" -t 24:00:00 \
    -d "afterok:$SMOKE" -o "$PIPELINE_DIR/logs/reference-%j.out" \
    --export="$COMMON_EXPORT,SOPPO_NPROC_PER_NODE=2,SOPPO_REQUIRE_A800=1" "$SCRIPT_DIR/02_finalize_inputs.sh"
REFERENCE="$LAST_JOB_ID"
submit dpo_headroom_runs -J soppo-dpo -p "$GPU_PARTITION" -N 1 -c 32 --gres="$GPU2_GRES" -t 2-00:00:00 \
    --array="0-1%$ARRAY_LIMIT" -d "afterok:$REFERENCE" -o "$PIPELINE_DIR/logs/dpo-%A_%a.out" \
    --export="$COMMON_EXPORT,SOPPO_NPROC_PER_NODE=2,SOPPO_REQUIRE_A800=1" "$SCRIPT_DIR/03_preexperiment.sh"
PRE="$LAST_JOB_ID"
submit headroom_gate -J soppo-headroom -p "$CPU_PARTITION" -N 1 -c 4 --gres="$AUX_GRES" -t 01:00:00 \
    -d "afterok:$PRE" -o "$PIPELINE_DIR/logs/headroom-%j.out" \
    --export="$COMMON_EXPORT" "$SCRIPT_DIR/03_select_preexperiment.sh"
PRESELECT="$LAST_JOB_ID"
submit pe_static_runs -J soppo-static -p "$GPU_PARTITION" -N 1 -c 32 --gres="$GPU2_GRES" -t 3-00:00:00 \
    --array="0-3%$ARRAY_LIMIT" -d "afterok:$PRESELECT" -o "$PIPELINE_DIR/logs/static-%A_%a.out" \
    --export="$COMMON_EXPORT,SOPPO_NPROC_PER_NODE=2,SOPPO_REQUIRE_A800=1" "$SCRIPT_DIR/04_lambda_search.sh"
LAMBDA="$LAST_JOB_ID"
submit static_select -J soppo-static-select -p "$CPU_PARTITION" -N 1 -c 4 --gres="$AUX_GRES" -t 01:00:00 \
    -d "afterok:$LAMBDA" -o "$PIPELINE_DIR/logs/static-select-%j.out" \
    --export="$COMMON_EXPORT" "$SCRIPT_DIR/04_select_lambda.sh"
LAMBDA_SELECT="$LAST_JOB_ID"
submit dynamic_runs -J soppo-dynamic -p "$GPU_PARTITION" -N 1 -c 32 --gres="$GPU2_GRES" -t 3-00:00:00 \
    --array="0-1%$ARRAY_LIMIT" -d "afterok:$LAMBDA_SELECT" -o "$PIPELINE_DIR/logs/dynamic-%A_%a.out" \
    --export="$COMMON_EXPORT,SOPPO_NPROC_PER_NODE=2,SOPPO_REQUIRE_A800=1" "$SCRIPT_DIR/05_run_main.sh"
MAIN="$LAST_JOB_ID"
submit c_epsilon_prepare -J soppo-ceprep -p "$CPU_PARTITION" -N 1 -c 4 --gres="$AUX_GRES" -t 01:00:00 \
    -d "afterok:$MAIN" -o "$PIPELINE_DIR/logs/ceprep-%j.out" \
    --export="$COMMON_EXPORT" "$SCRIPT_DIR/06_prepare_c_epsilon.sh"
CE_PREP="$LAST_JOB_ID"
submit c_epsilon_raw -J soppo-ce -p "$GPU_PARTITION" -N 1 -c 16 --gres="$GPU1_GRES" -t 2-00:00:00 \
    --array="0-8%$ARRAY_LIMIT" -d "afterok:$CE_PREP" -o "$PIPELINE_DIR/logs/ce-%A_%a.out" \
    --export="$COMMON_EXPORT,SOPPO_NPROC_PER_NODE=1,SOPPO_REQUIRE_A800=1" "$SCRIPT_DIR/06_c_epsilon.sh"
CE_RAW="$LAST_JOB_ID"
submit c_epsilon_derive -J soppo-cederive -p "$CPU_PARTITION" -N 1 -c 8 --gres="$AUX_GRES" -t 02:00:00 \
    -d "afterok:$CE_RAW" -o "$PIPELINE_DIR/logs/ce-derive-%j.out" \
    --export="$COMMON_EXPORT" "$SCRIPT_DIR/06_derive_c_epsilon.sh"
CE_DERIVE="$LAST_JOB_ID"
submit evaluation -J soppo-eval -p "$GPU_PARTITION" -N 1 -c 16 --gres="$GPU1_GRES" -t 24:00:00 \
    --array="0-7%$ARRAY_LIMIT" -d "afterok:$CE_DERIVE" -o "$PIPELINE_DIR/logs/eval-%A_%a.out" \
    --export="$COMMON_EXPORT,SOPPO_NPROC_PER_NODE=1,SOPPO_REQUIRE_A800=1" "$SCRIPT_DIR/07_evaluate.sh"
EVAL="$LAST_JOB_ID"
submit aggregate -J soppo-aggregate -p "$CPU_PARTITION" -N 1 -c 8 --gres="$AUX_GRES" -t 02:00:00 \
    -d "afterok:$EVAL" -o "$PIPELINE_DIR/logs/aggregate-%j.out" \
    --export="$COMMON_EXPORT" "$SCRIPT_DIR/08_aggregate.sh"
AGG="$LAST_JOB_ID"

python - "$REGISTRY" "$EXPERIMENT_ID" "$GIT_COMMIT" \
    "tests=$TESTS" "smoke=$SMOKE" "reference_cache=$REFERENCE" \
    "dpo_headroom_runs=$PRE" "headroom_gate=$PRESELECT" "pe_static_runs=$LAMBDA" \
    "static_select=$LAMBDA_SELECT" "dynamic_runs=$MAIN" "c_epsilon_prepare=$CE_PREP" \
    "c_epsilon_raw=$CE_RAW" "c_epsilon_derive=$CE_DERIVE" "evaluation=$EVAL" "aggregate=$AGG" <<'PY'
import json
import os
import sys
path, experiment, commit, *pairs = sys.argv[1:]
jobs = dict(pair.split("=", 1) for pair in pairs)
payload = {
    "schema_version": 1,
    "cycle_id": "cycle-20260818-01",
    "experiment_id": experiment,
    "experiment_design": "v0.6-sspo-aligned-30k",
    "git_commit": commit,
    "submission_status": "submitted",
    "jobs": jobs,
}
temporary = path + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

trap - ERR INT TERM
echo "Complete Slurm DAG submitted."
echo "Registry: $REGISTRY"
echo "Read-only status: bash $SCRIPT_DIR/status_pipeline.sh"
