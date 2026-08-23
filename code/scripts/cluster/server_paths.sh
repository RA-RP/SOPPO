#!/bin/bash
# Shared server path contract. Source this file from cluster scripts only.

CLUSTER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$CLUSTER_SCRIPT_DIR/../.." && pwd)"
SOPPO_ROOT="$(cd "$CODE_ROOT/.." && pwd)"
if [[ -n "${SOPPO_SERVER_BASE:-}" ]]; then
    [[ "$SOPPO_SERVER_BASE" = /* ]] || {
        echo "ERROR: SOPPO_SERVER_BASE must be absolute" >&2
        return 1 2>/dev/null || exit 1
    }
    SERVER_BASE="$(cd "$SOPPO_SERVER_BASE" && pwd)"
else
    SERVER_BASE="$(cd "$SOPPO_ROOT/.." && pwd)"
fi

ICLR_ROOT="$SERVER_BASE/ICLR"
ENV_ROOT="$SERVER_BASE/envs"
CACHE_ROOT="$SERVER_BASE/cache"
DATA_ROOT="$SERVER_BASE/data"
MODEL_ROOT="$SERVER_BASE/models"
RUN_ROOT="$SERVER_BASE/runs"
EXPORT_ROOT="$SERVER_BASE/exports"
PLATFORM_LOG_ROOT="$SERVER_BASE/platform_logs"
OBSERVE_ROOT="$CODE_ROOT/observe/LLM-output-density"

EXPERIMENT_ID="${EXPERIMENT_ID:-exp-20260819-01-mvp}"

export SERVER_BASE ICLR_ROOT SOPPO_ROOT CODE_ROOT OBSERVE_ROOT
export ENV_ROOT CACHE_ROOT DATA_ROOT MODEL_ROOT RUN_ROOT EXPORT_ROOT PLATFORM_LOG_ROOT
export EXPERIMENT_ID
