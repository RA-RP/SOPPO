#!/usr/bin/env bash
# Stop only the process group recorded for this round2 experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

PID_FILE="$ROUND2_RUN_ROOT/controller.pid"
STARTTIME_FILE="$ROUND2_RUN_ROOT/controller.starttime"
[[ -f "$PID_FILE" ]] || { echo "No round2 controller PID file: $PID_FILE"; exit 0; }
controller_pid="$(tr -d '[:space:]' < "$PID_FILE")"
[[ "$controller_pid" =~ ^[0-9]+$ ]] || {
    echo "ERROR: Invalid controller PID record: $controller_pid" >&2
    exit 1
}
if ! kill -0 "$controller_pid" 2>/dev/null; then
    echo "Controller PID $controller_pid is no longer running; no signal sent."
    exit 0
fi
[[ -f "$STARTTIME_FILE" ]] || {
    echo "ERROR: Missing controller process fingerprint: $STARTTIME_FILE" >&2
    exit 1
}
recorded_starttime="$(tr -d '[:space:]' < "$STARTTIME_FILE")"
actual_starttime="$(awk '{print $22}' "/proc/$controller_pid/stat")"
actual_pgid="$(ps -o pgid= -p "$controller_pid" | tr -d '[:space:]')"
actual_command="$(ps -o args= -p "$controller_pid")"
if [[ "$recorded_starttime" != "$actual_starttime" \
    || "$actual_pgid" != "$controller_pid" \
    || "$actual_command" != *"$SCRIPT_DIR/run_all.sh"* ]]; then
    echo "ERROR: PID fingerprint does not match the recorded Round2 controller" >&2
    echo "No signal sent." >&2
    exit 1
fi

echo "Target round2 process group: $controller_pid"
ps -o pid,ppid,pgid,state,etime,command -g "$controller_pid"
if [[ "${1:-}" != "--execute" ]]; then
    echo "Preview only. Re-run with --execute to send TERM to this process group."
    exit 0
fi
kill -TERM -- "-$controller_pid"
echo "TERM sent only to round2 process group $controller_pid"
