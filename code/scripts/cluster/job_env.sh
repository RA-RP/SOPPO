#!/bin/bash
# Shared fail-closed runtime for Slurm workers.

soppo_job_init() {
    if [[ "${RUN_CONTEXT:-}" != "cluster" ]]; then
        echo "ERROR: RUN_CONTEXT=cluster is required" >&2
        return 1
    fi
    local script_dir
    script_dir="${SOPPO_CLUSTER_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
    if [[ ! -f "$script_dir/server_paths.sh" || ! -f "$script_dir/runtime_env.sh" ]]; then
        echo "ERROR: Invalid SOPPO cluster script directory: $script_dir" >&2
        return 1
    fi
    source "$script_dir/server_paths.sh"
    source "$script_dir/runtime_env.sh"
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        if [[ ! "${SOPPO_EXPECTED_GIT_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
            echo "ERROR: Slurm worker is missing SOPPO_EXPECTED_GIT_COMMIT" >&2
            return 1
        fi
        if [[ -n "${SOPPO_SOURCE_MANIFEST:-}" || -n "${SOPPO_SOURCE_MANIFEST_SHA256:-}" ]]; then
            if [[ ! "${SOPPO_SOURCE_MANIFEST_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
                echo "ERROR: Slurm worker has an invalid source-manifest checksum" >&2
                return 1
            fi
            if [[ ! -f "$script_dir/source_snapshot.py" ]]; then
                echo "ERROR: Snapshot verifier is missing: $script_dir/source_snapshot.py" >&2
                return 1
            fi
            "$ENV_ROOT/youc/bin/python" "$script_dir/source_snapshot.py" verify \
                --root "$SOPPO_ROOT" \
                --manifest "$SOPPO_SOURCE_MANIFEST" \
                --manifest-sha256 "$SOPPO_SOURCE_MANIFEST_SHA256" \
                --commit "$SOPPO_EXPECTED_GIT_COMMIT" || return 1
        else
            local actual_commit
            actual_commit="$(git -C "$SOPPO_ROOT" rev-parse HEAD)"
            if [[ "$actual_commit" != "$SOPPO_EXPECTED_GIT_COMMIT" ]]; then
                echo "ERROR: Server checkout changed after DAG submission" >&2
                echo "  expected: $SOPPO_EXPECTED_GIT_COMMIT" >&2
                echo "  actual:   $actual_commit" >&2
                return 1
            fi
            if [[ -n "$(git -C "$SOPPO_ROOT" status --porcelain)" ]]; then
                echo "ERROR: Server checkout became dirty after DAG submission" >&2
                return 1
            fi
        fi
    fi
    soppo_activate_env "$ENV_ROOT/youc"
    export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
    export HF_HOME="$CACHE_ROOT/huggingface"
    export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
    unset TRANSFORMERS_CACHE
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

soppo_hardware_gate() {
    local output_file="${1:?hardware output path required}"
    local required_count="${SOPPO_NPROC_PER_NODE:-2}"
    mkdir -p "$(dirname "$output_file")"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader > "$output_file"
    local visible_count
    visible_count="$(wc -l < "$output_file" | tr -d ' ')"
    if (( visible_count != required_count )); then
        echo "ERROR: requested $required_count visible GPUs, got $visible_count" >&2
        return 1
    fi
    if [[ "${SOPPO_REQUIRE_A800:-0}" == "1" ]]; then
        if [[ "$(grep -Eic 'A800' "$output_file")" != "$required_count" ]]; then
            echo "ERROR: formal job requires $required_count A800 GPUs" >&2
            return 1
        fi
    fi
}

soppo_torchrun() {
    torchrun --standalone --nnodes=1 --nproc_per_node="${SOPPO_NPROC_PER_NODE:-2}" "$@"
}
