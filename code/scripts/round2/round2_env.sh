#!/usr/bin/env bash
# Shared round2 paths and resource contract. First-round paths remain untouched.

set -euo pipefail

ROUND2_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$ROUND2_SCRIPT_DIR/../.." && pwd)"
SOPPO_ROOT="$(cd "$CODE_ROOT/.." && pwd)"
SERVER_BASE="$(cd "$SOPPO_ROOT/.." && pwd)"

if [[ -n "${SOPPO_SERVER_BASE:-}" ]]; then
    REQUESTED_BASE="$(cd "$SOPPO_SERVER_BASE" && pwd)"
    if [[ "$REQUESTED_BASE" != "$SERVER_BASE" ]]; then
        echo "ERROR: SOPPO_SERVER_BASE does not match this checkout" >&2
        echo "  requested: $REQUESTED_BASE" >&2
        echo "  detected:  $SERVER_BASE" >&2
        exit 1
    fi
fi

ENV_ROOT="${ENV_ROOT:-$SERVER_BASE/envs}"
DATA_ROOT="${DATA_ROOT:-$SERVER_BASE/data}"
MODEL_ROOT="${MODEL_ROOT:-$SERVER_BASE/models}"
RUN_ROOT="${RUN_ROOT:-$SERVER_BASE/runs}"
EXPORT_ROOT="${EXPORT_ROOT:-$SERVER_BASE/exports}"
CACHE_ROOT="${CACHE_ROOT:-$SERVER_BASE/cache}"

ROUND2_EXPERIMENT_ID="${ROUND2_EXPERIMENT_ID:-exp-20260823-01-round2-megatron}"
ROUND2_RUN_ROOT="$RUN_ROOT/$ROUND2_EXPERIMENT_ID"
ROUND2_CONFIG_DIR="$CODE_ROOT/configs/round2"
ROUND2_PYTHON="${ROUND2_PYTHON:-$ENV_ROOT/youc/bin/python}"

export ROUND2_SCRIPT_DIR CODE_ROOT SOPPO_ROOT SERVER_BASE ENV_ROOT DATA_ROOT MODEL_ROOT
export RUN_ROOT EXPORT_ROOT CACHE_ROOT ROUND2_EXPERIMENT_ID ROUND2_RUN_ROOT
export ROUND2_CONFIG_DIR ROUND2_PYTHON
