#!/bin/bash
# Read-only status for the standalone sequential pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/server_paths.sh"
REGISTRY="$RUN_ROOT/$EXPERIMENT_ID/pipeline/task_registry.json"
DRIVER_LOG="$RUN_ROOT/$EXPERIMENT_ID/standalone-driver.log"
PYTHON_STATUS="$ENV_ROOT/youc/bin/python"
if (( $# > 1 )) || [[ "${1:-}" != "" && "${1:-}" != "--no-tail" ]]; then
    echo "Usage: bash status_pipeline.sh [--no-tail]" >&2
    exit 2
fi

if [[ ! -f "$REGISTRY" ]]; then
    echo "Registry not created yet: $REGISTRY"
    if [[ -f "$DRIVER_LOG" ]]; then
        tail -n 80 "$DRIVER_LOG"
    fi
    exit 1
fi
test -x "$PYTHON_STATUS" || { echo "ERROR: Environment Python is missing" >&2; exit 1; }
"$PYTHON_STATUS" "$SCRIPT_DIR/pipeline_state.py" show --path "$REGISTRY"
PID="$("$PYTHON_STATUS" "$SCRIPT_DIR/pipeline_state.py" get --path "$REGISTRY" --field pid)"
if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
    echo
    echo "Process: alive"
else
    echo
    echo "Process: not running"
fi

if [[ "${1:-}" != "--no-tail" && -f "$DRIVER_LOG" ]]; then
    echo
    echo "Last 40 driver-log lines:"
    tail -n 40 "$DRIVER_LOG"
fi
