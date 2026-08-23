#!/bin/bash
# Reuse completed tests/smoke/reference gates and submit a new DAG from DPO onward.
set -Eeuo pipefail

if [[ "${RUN_CONTEXT:-}" != "cluster" ]]; then
    echo "ERROR: RUN_CONTEXT=cluster is required" >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init

REUSE_REGISTRY=""
FORMAL_GPU_COUNT="${SOPPO_FORMAL_GPU_COUNT:-${SOPPO_GPU2_COUNT:-2}}"
while (( $# > 0 )); do
    case "$1" in
        --reuse-registry)
            REUSE_REGISTRY="${2:?--reuse-registry requires a path}"
            shift 2
            ;;
        --formal-gpus)
            FORMAL_GPU_COUNT="${2:?--formal-gpus requires 1, 2, or 4}"
            shift 2
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done
[[ -n "$REUSE_REGISTRY" ]] || { echo "ERROR: --reuse-registry is required" >&2; exit 2; }
test -f "$REUSE_REGISTRY" || { echo "ERROR: reused registry not found: $REUSE_REGISTRY" >&2; exit 1; }

DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
MODEL_DIR="${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}"
GIT_REPO_ROOT="${SOPPO_GIT_REPO_ROOT:-$SOPPO_ROOT}"
SOURCE_COMMITISH="${SOPPO_SOURCE_COMMIT:-HEAD}"
REFERENCE_ROOT="${SOPPO_REFERENCE_CACHE:-$CACHE_ROOT/soppo/reference/qwen3-4b-mvp-v0.5-30k}"
PIPELINE_DIR="$RUN_ROOT/$EXPERIMENT_ID/pipeline"
REGISTRY="$PIPELINE_DIR/task_registry.json"
SOURCE_ROOT="$PIPELINE_DIR/source/SOPPO"
SOURCE_MANIFEST="$PIPELINE_DIR/source_manifest.json"
GPU_PARTITION="${SOPPO_GPU_PARTITION:-gpu}"
CPU_PARTITION="${SOPPO_CPU_PARTITION:-$GPU_PARTITION}"
GPU1_COUNT="${SOPPO_GPU1_COUNT:-1}"
AUX_GPU_COUNT="${SOPPO_AUX_GPU_COUNT:-$GPU1_COUNT}"
ARRAY_LIMIT="${SOPPO_ARRAY_PARALLELISM:-4}"
EXCLUDE_NODES="${SOPPO_EXCLUDE_NODES-gn005,gn014,gn021}"

case "$FORMAL_GPU_COUNT" in
    1|2|4) ;;
    *) echo "ERROR: --formal-gpus/SOPPO_FORMAL_GPU_COUNT must be 1, 2, or 4" >&2; exit 1 ;;
esac
[[ "$GPU1_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid SOPPO_GPU1_COUNT" >&2; exit 1; }
[[ "$AUX_GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid SOPPO_AUX_GPU_COUNT" >&2; exit 1; }
[[ "$ARRAY_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid SOPPO_ARRAY_PARALLELISM" >&2; exit 1; }

command -v sbatch >/dev/null || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
command -v scancel >/dev/null || { echo "ERROR: scancel is unavailable" >&2; exit 1; }
command -v sacct >/dev/null || { echo "ERROR: sacct is unavailable" >&2; exit 1; }
command -v git >/dev/null || { echo "ERROR: git is unavailable" >&2; exit 1; }
command -v tar >/dev/null || { echo "ERROR: tar is unavailable" >&2; exit 1; }
test -f "$MODEL_DIR/model_manifest.json" || { echo "ERROR: Qwen3 manifest is missing" >&2; exit 1; }
test -f "$DATA_DIR/manifest_public.json" || { echo "ERROR: 30k data manifest is missing" >&2; exit 1; }
test -f "$DATA_DIR/oracle_train.private.jsonl" || { echo "ERROR: oracle training file is missing" >&2; exit 1; }
test -f "$REFERENCE_ROOT/COMPLETE" || { echo "ERROR: reusable reference cache is incomplete" >&2; exit 1; }
test -d "$REFERENCE_ROOT/targets" || { echo "ERROR: reference cache targets are missing" >&2; exit 1; }
[[ ! -e "$PIPELINE_DIR" ]] || { echo "ERROR: Refuse to overwrite pipeline directory: $PIPELINE_DIR" >&2; exit 1; }
for arm in dpo10 dpo100; do
    [[ ! -e "$RUN_ROOT/$EXPERIMENT_ID/main/$arm" ]] || {
        echo "ERROR: archive partial run before recovery: $RUN_ROOT/$EXPERIMENT_ID/main/$arm" >&2
        exit 1
    }
done
git -C "$GIT_REPO_ROOT" cat-file -e "$SOURCE_COMMITISH^{commit}" 2>/dev/null || {
    echo "ERROR: Invalid SOPPO source commit: $SOURCE_COMMITISH" >&2
    exit 1
}
if [[ -n "$(git -C "$GIT_REPO_ROOT" status --porcelain)" ]]; then
    echo "ERROR: Server SOPPO checkout must be clean before submission" >&2
    exit 1
fi
GIT_COMMIT="$(git -C "$GIT_REPO_ROOT" rev-parse "$SOURCE_COMMITISH^{commit}")"

mapfile -t REUSED < <(python - "$REUSE_REGISTRY" "$EXPERIMENT_ID" <<'PY'
import json
import re
import sys
from pathlib import Path

path, expected_experiment = sys.argv[1:]
registry = json.loads(Path(path).read_text(encoding="utf-8"))
if registry.get("experiment_id") != expected_experiment:
    raise SystemExit("reused registry has the wrong experiment_id")
commit = str(registry.get("git_commit", ""))
if not re.fullmatch(r"[0-9a-f]{40}", commit):
    raise SystemExit("reused registry has an invalid git_commit")
jobs = registry.get("jobs", {})
formal_gpus = registry.get("slurm", {}).get("gpu_counts", {}).get(
    "smoke_and_formal_training"
)
if formal_gpus not in (1, 2, 4):
    raise SystemExit("reused registry has no valid 1/2/4 formal GPU profile")
required = ("tests", "smoke", "reference_cache")
values = []
for stage in required:
    job_id = str(jobs.get(stage, ""))
    if not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise SystemExit(f"reused registry is missing a valid {stage} job")
    values.append(job_id)
print(commit)
print(*values, sep="\n")
print(formal_gpus)
PY
)
(( ${#REUSED[@]} == 5 )) || { echo "ERROR: could not parse reused gates" >&2; exit 1; }
REUSED_COMMIT="${REUSED[0]}"
TESTS="${REUSED[1]}"
SMOKE="${REUSED[2]}"
REFERENCE="${REUSED[3]}"
REUSED_FORMAL_GPU_COUNT="${REUSED[4]}"
if [[ "$REUSED_FORMAL_GPU_COUNT" != "$FORMAL_GPU_COUNT" ]]; then
    echo "ERROR: reused smoke profile used $REUSED_FORMAL_GPU_COUNT GPUs, requested $FORMAL_GPU_COUNT" >&2
    echo "Run a full submit_all.sh when changing the 1/2/4-GPU profile." >&2
    exit 1
fi

git -C "$GIT_REPO_ROOT" cat-file -e "$REUSED_COMMIT^{commit}" 2>/dev/null || {
    echo "ERROR: reused commit is unavailable locally: $REUSED_COMMIT" >&2
    exit 1
}
git -C "$GIT_REPO_ROOT" merge-base --is-ancestor "$REUSED_COMMIT" "$GIT_COMMIT" || {
    echo "ERROR: current checkout is not a descendant of reused commit: $REUSED_COMMIT" >&2
    exit 1
}
mapfile -t CHANGED_PATHS < <(
    git -C "$GIT_REPO_ROOT" diff --name-only "$REUSED_COMMIT".."$GIT_COMMIT"
)
for path in "${CHANGED_PATHS[@]}"; do
    case "$path" in
        *.md|code/scripts/cluster/cancel_pipeline.sh|code/scripts/cluster/submit_from_dpo.sh) ;;
        *)
            echo "ERROR: runtime path changed since the reused gates: $path" >&2
            exit 1
            ;;
    esac
done

job_state() {
    sacct -n -X -j "$1" --format=State --parsable2 \
        | awk -F'|' 'NF && $1 != "" {gsub(/[[:space:]]+/, "", $1); print $1; exit}'
}
for record in "tests=$TESTS" "smoke=$SMOKE" "reference_cache=$REFERENCE"; do
    stage="${record%%=*}"
    job_id="${record##*=}"
    state="$(job_state "$job_id")"
    state="${state%+}"
    [[ "$state" == "COMPLETED" ]] || {
        echo "ERROR: cannot reuse $stage job $job_id with state '${state:-UNKNOWN}'" >&2
        exit 1
    }
done

mkdir -p "$PIPELINE_DIR/logs" "$PIPELINE_DIR/hardware"
mkdir -p "$SOURCE_ROOT"
git -C "$GIT_REPO_ROOT" archive "$GIT_COMMIT" | tar -x -C "$SOURCE_ROOT"
SOURCE_SCRIPT_DIR="$SOURCE_ROOT/code/scripts/cluster"
SOURCE_MANIFEST_SHA256="$(
    python "$SOURCE_SCRIPT_DIR/source_snapshot.py" create \
        --root "$SOURCE_ROOT" --manifest "$SOURCE_MANIFEST" --commit "$GIT_COMMIT"
)"
[[ "$SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: Failed to create immutable source manifest" >&2
    exit 1
}
export PYTHONPATH="$SOURCE_ROOT/code:${PYTHONPATH:-}"
python -m src.model.model_manifest --model-dir "$MODEL_DIR" --verify
python -m src.data.audit_prepared_data --data-dir "$DATA_DIR"

echo "Reusing completed gates: tests=$TESTS, smoke=$SMOKE, reference_cache=$REFERENCE"
echo "Slurm routing: auxiliary=$CPU_PARTITION/${AUX_GPU_COUNT}GPU, formal=$GPU_PARTITION/${FORMAL_GPU_COUNT}GPU"
echo "Slurm node exclusions: ${EXCLUDE_NODES:-none}"
echo "Immutable source: $SOURCE_ROOT @ ${GIT_COMMIT:0:12}"

declare -a NODE_ARGS=()
if [[ -n "$EXCLUDE_NODES" ]]; then
    NODE_ARGS+=(--exclude="$EXCLUDE_NODES")
fi
declare -a SUBMITTED_JOBS=()
submission_failed() {
    local status=$?
    trap - ERR INT TERM
    (( status != 0 )) || status=1
    echo "ERROR: recovery DAG submission stopped; rolling back this invocation." >&2
    if (( ${#SUBMITTED_JOBS[@]} > 0 )); then
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
    if ! raw="$(sbatch --parsable "${NODE_ARGS[@]}" "$@")"; then
        echo "ERROR: sbatch rejected stage: $stage" >&2
        return 1
    fi
    LAST_JOB_ID="${raw%%;*}"
    [[ "$LAST_JOB_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid sbatch job id: $raw" >&2; return 1; }
    SUBMITTED_JOBS+=("$LAST_JOB_ID")
    echo "Submitted $stage as Slurm job $LAST_JOB_ID" >&2
}

COMMON_EXPORT="ALL,RUN_CONTEXT=cluster,EXPERIMENT_ID=$EXPERIMENT_ID,SOPPO_CLUSTER_SCRIPT_DIR=$SOURCE_SCRIPT_DIR,SOPPO_SERVER_BASE=$SERVER_BASE,SOPPO_DATA_DIR=$DATA_DIR,SOPPO_MODEL_DIR=$MODEL_DIR,SOPPO_EXPECTED_GIT_COMMIT=$GIT_COMMIT,SOPPO_SOURCE_MANIFEST=$SOURCE_MANIFEST,SOPPO_SOURCE_MANIFEST_SHA256=$SOURCE_MANIFEST_SHA256"
submit dpo_headroom_runs -J soppo-dpo -p "$GPU_PARTITION" -N 1 -c 32 -G "$FORMAL_GPU_COUNT" -t 2-00:00:00 \
    --array="0-1%$ARRAY_LIMIT" -o "$PIPELINE_DIR/logs/dpo-%A_%a.out" \
    --export="$COMMON_EXPORT,SOPPO_NPROC_PER_NODE=$FORMAL_GPU_COUNT,SOPPO_REQUIRE_A800=1" "$SOURCE_SCRIPT_DIR/03_preexperiment.sh"
PRE="$LAST_JOB_ID"
submit headroom_gate -J soppo-headroom -p "$CPU_PARTITION" -N 1 -c 4 -G "$AUX_GPU_COUNT" -t 01:00:00 \
    -d "afterok:$PRE" -o "$PIPELINE_DIR/logs/headroom-%j.out" \
    --export="$COMMON_EXPORT" "$SOURCE_SCRIPT_DIR/03_select_preexperiment.sh"
PRESELECT="$LAST_JOB_ID"
submit pe_static_runs -J soppo-static -p "$GPU_PARTITION" -N 1 -c 32 -G "$FORMAL_GPU_COUNT" -t 3-00:00:00 \
    --array="0-3%$ARRAY_LIMIT" -d "afterok:$PRESELECT" -o "$PIPELINE_DIR/logs/static-%A_%a.out" \
    --export="$COMMON_EXPORT,SOPPO_NPROC_PER_NODE=$FORMAL_GPU_COUNT,SOPPO_REQUIRE_A800=1" "$SOURCE_SCRIPT_DIR/04_lambda_search.sh"
LAMBDA="$LAST_JOB_ID"
submit static_select -J soppo-static-select -p "$CPU_PARTITION" -N 1 -c 4 -G "$AUX_GPU_COUNT" -t 01:00:00 \
    -d "afterok:$LAMBDA" -o "$PIPELINE_DIR/logs/static-select-%j.out" \
    --export="$COMMON_EXPORT" "$SOURCE_SCRIPT_DIR/04_select_lambda.sh"
LAMBDA_SELECT="$LAST_JOB_ID"
submit dynamic_runs -J soppo-dynamic -p "$GPU_PARTITION" -N 1 -c 32 -G "$FORMAL_GPU_COUNT" -t 3-00:00:00 \
    --array="0-1%$ARRAY_LIMIT" -d "afterok:$LAMBDA_SELECT" -o "$PIPELINE_DIR/logs/dynamic-%A_%a.out" \
    --export="$COMMON_EXPORT,SOPPO_NPROC_PER_NODE=$FORMAL_GPU_COUNT,SOPPO_REQUIRE_A800=1" "$SOURCE_SCRIPT_DIR/05_run_main.sh"
MAIN="$LAST_JOB_ID"
submit c_epsilon_prepare -J soppo-ceprep -p "$CPU_PARTITION" -N 1 -c 4 -G "$AUX_GPU_COUNT" -t 01:00:00 \
    -d "afterok:$MAIN" -o "$PIPELINE_DIR/logs/ceprep-%j.out" \
    --export="$COMMON_EXPORT" "$SOURCE_SCRIPT_DIR/06_prepare_c_epsilon.sh"
CE_PREP="$LAST_JOB_ID"
submit c_epsilon_raw -J soppo-ce -p "$GPU_PARTITION" -N 1 -c 16 -G "$GPU1_COUNT" -t 2-00:00:00 \
    --array="0-8%$ARRAY_LIMIT" -d "afterok:$CE_PREP" -o "$PIPELINE_DIR/logs/ce-%A_%a.out" \
    --export="$COMMON_EXPORT,SOPPO_NPROC_PER_NODE=1,SOPPO_REQUIRE_A800=1" "$SOURCE_SCRIPT_DIR/06_c_epsilon.sh"
CE_RAW="$LAST_JOB_ID"
submit c_epsilon_derive -J soppo-cederive -p "$CPU_PARTITION" -N 1 -c 8 -G "$AUX_GPU_COUNT" -t 02:00:00 \
    -d "afterok:$CE_RAW" -o "$PIPELINE_DIR/logs/ce-derive-%j.out" \
    --export="$COMMON_EXPORT" "$SOURCE_SCRIPT_DIR/06_derive_c_epsilon.sh"
CE_DERIVE="$LAST_JOB_ID"
submit evaluation -J soppo-eval -p "$GPU_PARTITION" -N 1 -c 16 -G "$GPU1_COUNT" -t 24:00:00 \
    --array="0-7%$ARRAY_LIMIT" -d "afterok:$CE_DERIVE" -o "$PIPELINE_DIR/logs/eval-%A_%a.out" \
    --export="$COMMON_EXPORT,SOPPO_NPROC_PER_NODE=1,SOPPO_REQUIRE_A800=1" "$SOURCE_SCRIPT_DIR/07_evaluate.sh"
EVAL="$LAST_JOB_ID"
submit aggregate -J soppo-aggregate -p "$CPU_PARTITION" -N 1 -c 8 -G "$AUX_GPU_COUNT" -t 02:00:00 \
    -d "afterok:$EVAL" -o "$PIPELINE_DIR/logs/aggregate-%j.out" \
    --export="$COMMON_EXPORT" "$SOURCE_SCRIPT_DIR/08_aggregate.sh"
AGG="$LAST_JOB_ID"

python - "$REGISTRY" "$EXPERIMENT_ID" "$GIT_COMMIT" "$EXCLUDE_NODES" \
    "$AUX_GPU_COUNT" "$FORMAL_GPU_COUNT" "$GPU1_COUNT" "$REUSE_REGISTRY" "$REUSED_COMMIT" \
    "$SOURCE_ROOT" "$SOURCE_MANIFEST" "$SOURCE_MANIFEST_SHA256" \
    "tests=$TESTS" "smoke=$SMOKE" "reference_cache=$REFERENCE" \
    "dpo_headroom_runs=$PRE" "headroom_gate=$PRESELECT" "pe_static_runs=$LAMBDA" \
    "static_select=$LAMBDA_SELECT" "dynamic_runs=$MAIN" "c_epsilon_prepare=$CE_PREP" \
    "c_epsilon_raw=$CE_RAW" "c_epsilon_derive=$CE_DERIVE" "evaluation=$EVAL" "aggregate=$AGG" <<'PY'
import json
import os
import sys

(
    path, experiment, commit, excluded, aux_gpus, formal_gpus, post_gpus,
    reused_registry, reused_commit, source_root, source_manifest,
    source_manifest_sha256, *pairs
) = sys.argv[1:]
jobs = dict(pair.split("=", 1) for pair in pairs)
payload = {
    "schema_version": 1,
    "cycle_id": "cycle-20260818-01",
    "experiment_id": experiment,
    "experiment_design": "v0.6-sspo-aligned-30k",
    "git_commit": commit,
    "source_snapshot": {
        "root": source_root,
        "manifest": source_manifest,
        "manifest_sha256": source_manifest_sha256,
    },
    "submission_status": "resumed_from_dpo",
    "recovery": {
        "reason": "restart_from_dpo_with_reused_gates",
        "reused_registry": os.path.abspath(reused_registry),
        "reused_git_commit": reused_commit,
        "reused_completed_stages": ["tests", "smoke", "reference_cache"],
        "restart_stage": "dpo_headroom_runs",
    },
    "slurm": {
        "node_exclusions": [name for name in excluded.split(",") if name],
        "gpu_counts": {
            "auxiliary": int(aux_gpus),
            "smoke_and_formal_training": int(formal_gpus),
            "single_gpu_postprocessing": int(post_gpus),
        },
    },
    "jobs": jobs,
}
temporary = path + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

trap - ERR INT TERM
echo "Recovery Slurm DAG submitted from DPO."
echo "Registry: $REGISTRY"
echo "Read-only status: SOPPO_SERVER_BASE=$SERVER_BASE bash $SOURCE_SCRIPT_DIR/status_pipeline.sh $REGISTRY"
