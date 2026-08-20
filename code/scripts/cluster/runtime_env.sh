#!/bin/bash
# Shared Miniforge/Conda activation for SOPPO cluster scripts.
# This file is sourced by stage scripts; it does not execute a stage itself.

SOPPO_MODULES_INIT="/home-ssd/Soft/modules/bashrc"
SOPPO_MINIFORGE_MODULE="miniforge3/25.11.0-0"
SOPPO_CONDA_VERSION="25.11.0"
SOPPO_PROXY_MODULE="proxy/proxy"

soppo_init_modules() {
    if [[ ! -f "$SOPPO_MODULES_INIT" ]]; then
        echo "ERROR: Module initialization file not found: $SOPPO_MODULES_INIT" >&2
        return 1
    fi

    source "$SOPPO_MODULES_INIT"
    if ! type module >/dev/null 2>&1; then
        echo "ERROR: The cluster module command is unavailable" >&2
        return 1
    fi
}

soppo_load_miniforge() {
    if command -v conda >/dev/null 2>&1 \
        && [[ "$(conda --version 2>/dev/null)" == "conda $SOPPO_CONDA_VERSION" ]]; then
        return 0
    fi

    soppo_init_modules || return 1

    echo "Loading server module: $SOPPO_MINIFORGE_MODULE"
    if ! module load "$SOPPO_MINIFORGE_MODULE"; then
        echo "ERROR: Failed to load $SOPPO_MINIFORGE_MODULE" >&2
        return 1
    fi

    if ! command -v conda >/dev/null 2>&1 \
        || [[ "$(conda --version 2>/dev/null)" != "conda $SOPPO_CONDA_VERSION" ]]; then
        echo "ERROR: conda $SOPPO_CONDA_VERSION is unavailable after loading $SOPPO_MINIFORGE_MODULE" >&2
        return 1
    fi
}

soppo_load_proxy() {
    soppo_init_modules || return 1

    echo "Loading server module: $SOPPO_PROXY_MODULE"
    if ! module load "$SOPPO_PROXY_MODULE"; then
        echo "ERROR: Failed to load $SOPPO_PROXY_MODULE" >&2
        return 1
    fi
}

soppo_activate_env() {
    local env_dir="${1:?environment directory is required}"

    soppo_load_miniforge || return 1

    if [[ ! -f "$env_dir/conda-meta/history" ]]; then
        echo "ERROR: Conda environment not found or incomplete: $env_dir" >&2
        echo "Run 00_server_setup.sh successfully before this stage." >&2
        return 1
    fi

    eval "$(conda shell.bash hook)"
    conda activate "$env_dir"
}
