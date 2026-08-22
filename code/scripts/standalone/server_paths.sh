#!/bin/bash
# Shared path contract for the dedicated, non-Slurm server.

STANDALONE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$STANDALONE_SCRIPT_DIR/../.." && pwd)"
SOPPO_ROOT="$(cd "$CODE_ROOT/.." && pwd)"
DETECTED_SERVER_BASE="$(cd "$SOPPO_ROOT/.." && pwd)"

if [[ -n "${SERVER_BASE:-}" ]]; then
    if [[ ! -d "$SERVER_BASE" ]]; then
        echo "ERROR: SERVER_BASE does not exist: $SERVER_BASE" >&2
        return 1 2>/dev/null || exit 1
    fi
    REQUESTED_SERVER_BASE="$(cd "$SERVER_BASE" && pwd)"
    if [[ "$REQUESTED_SERVER_BASE" != "$DETECTED_SERVER_BASE" ]]; then
        echo "ERROR: SERVER_BASE does not match the SOPPO checkout location" >&2
        echo "  requested: $REQUESTED_SERVER_BASE" >&2
        echo "  detected:  $DETECTED_SERVER_BASE" >&2
        return 1 2>/dev/null || exit 1
    fi
fi
SERVER_BASE="$DETECTED_SERVER_BASE"

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
export EXPERIMENT_ID STANDALONE_SCRIPT_DIR
