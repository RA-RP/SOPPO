#!/bin/bash
# Environment activation for a dedicated server without site-specific modules.

soppo_find_conda() {
    if [[ -n "${SOPPO_CONDA_EXE:-}" && -x "$SOPPO_CONDA_EXE" ]]; then
        printf '%s\n' "$SOPPO_CONDA_EXE"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi
    return 1
}

soppo_activate_env() {
    local env_dir="${1:?environment directory is required}"

    if [[ -f "$env_dir/conda-meta/history" ]]; then
        local conda_exe
        if ! conda_exe="$(soppo_find_conda)"; then
            echo "ERROR: Conda environment exists but conda is unavailable: $env_dir" >&2
            echo "Set SOPPO_CONDA_EXE to the conda executable and retry." >&2
            return 1
        fi
        eval "$("$conda_exe" shell.bash hook)"
        conda activate "$env_dir"
    elif [[ -f "$env_dir/bin/activate" ]]; then
        source "$env_dir/bin/activate"
    else
        echo "ERROR: Python environment not found or incomplete: $env_dir" >&2
        echo "Run 00_server_setup.sh successfully before this stage." >&2
        return 1
    fi

    if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
        echo "ERROR: SOPPO requires Python 3.10: $(python --version 2>&1)" >&2
        return 1
    fi
}

soppo_set_cache_env() {
    export PIP_CACHE_DIR="$CACHE_ROOT/pip"
    export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-180}"
    export PIP_RETRIES="${PIP_RETRIES:-10}"
    export CONDA_PKGS_DIRS="$CACHE_ROOT/conda/pkgs"
    export HF_HOME="$CACHE_ROOT/huggingface"
    export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
    export MODELSCOPE_CACHE="$CACHE_ROOT/modelscope"
    unset TRANSFORMERS_CACHE
}
