#!/bin/bash
# Stop only the process group recorded by this standalone experiment registry.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/server_paths.sh"
REGISTRY="$RUN_ROOT/$EXPERIMENT_ID/pipeline/task_registry.json"
PYTHON_STATUS="$ENV_ROOT/youc/bin/python"
EXECUTE=0
if [[ "${1:-}" == "--execute" ]]; then
    EXECUTE=1
elif (( $# > 0 )); then
    echo "Usage: bash stop_pipeline.sh [--execute]" >&2
    exit 2
fi

test -f "$REGISTRY" || { echo "ERROR: Registry not found: $REGISTRY" >&2; exit 1; }
test -x "$PYTHON_STATUS" || { echo "ERROR: Environment Python is missing" >&2; exit 1; }
PID="$("$PYTHON_STATUS" "$SCRIPT_DIR/pipeline_state.py" get --path "$REGISTRY" --field pid)"
PGID="$("$PYTHON_STATUS" "$SCRIPT_DIR/pipeline_state.py" get --path "$REGISTRY" --field process_group_id)"
[[ "$PID" =~ ^[0-9]+$ && "$PGID" =~ ^[0-9]+$ ]] || {
    echo "ERROR: Invalid PID/PGID in registry" >&2
    exit 1
}
if [[ "$PID" != "$PGID" ]]; then
    echo "ERROR: Registry describes a foreground run, not a dedicated process group" >&2
    echo "Stop it from the terminal where run_pipeline.sh is active." >&2
    exit 1
fi
if ! kill -0 "$PID" 2>/dev/null; then
    echo "Pipeline process is not running: $PID"
    exit 0
fi
ACTUAL_PGID="$(ps -o pgid= -p "$PID" | tr -d ' ')"
COMMAND_LINE="$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null || true)"
if [[ "$ACTUAL_PGID" != "$PGID" || "$COMMAND_LINE" != *"run_pipeline.sh"* ]]; then
    echo "ERROR: Refuse to signal a process that does not match the registry" >&2
    exit 1
fi

echo "Would send TERM to standalone pipeline process group: $PGID"
echo "Command: $COMMAND_LINE"
if (( EXECUTE == 0 )); then
    echo "Dry run only. Re-run with --execute to continue."
    exit 0
fi
printf 'Type stop-%s to confirm: ' "$EXPERIMENT_ID"
read -r CONFIRMATION
if [[ "$CONFIRMATION" != "stop-$EXPERIMENT_ID" ]]; then
    echo "Confirmation mismatch; nothing stopped." >&2
    exit 1
fi
kill -TERM -- "-$PGID"
echo "TERM sent only to process group $PGID. Check status before archiving the attempt."
