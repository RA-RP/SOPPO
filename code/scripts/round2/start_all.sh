#!/usr/bin/env bash
# Start the complete round2 controller in its own session so SSH may disconnect.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

PID_FILE="$ROUND2_RUN_ROOT/controller.pid"
STARTTIME_FILE="$ROUND2_RUN_ROOT/controller.starttime"
LOG_FILE="$ROUND2_RUN_ROOT/controller.log"
STATUS_FILE="$ROUND2_RUN_ROOT/controller.json"
if [[ -f "$PID_FILE" ]]; then
    existing_pid="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "ERROR: Round2 controller is already running with PID $existing_pid" >&2
        exit 1
    fi
    echo "ERROR: Stale controller PID file exists; preserve and inspect it: $PID_FILE" >&2
    exit 1
fi
[[ ! -e "$STATUS_FILE" && ! -e "$LOG_FILE" ]] || {
    echo "ERROR: Refuse to reuse round2 experiment: $ROUND2_RUN_ROOT" >&2
    exit 1
}

mkdir -p "$ROUND2_RUN_ROOT"
nohup setsid bash "$SCRIPT_DIR/run_all.sh" > "$LOG_FILE" 2>&1 < /dev/null &
controller_pid=$!
printf '%s\n' "$controller_pid" > "$PID_FILE"
controller_pgid=""
for _ in $(seq 1 20); do
    controller_pgid="$(ps -o pgid= -p "$controller_pid" 2>/dev/null | tr -d '[:space:]')"
    [[ -n "$controller_pgid" ]] && break
    sleep 0.1
done
if [[ "$controller_pgid" != "$controller_pid" ]]; then
    echo "ERROR: Round2 controller did not obtain an isolated process group" >&2
    kill "$controller_pid" 2>/dev/null || true
    exit 1
fi
awk '{print $22}' "/proc/$controller_pid/stat" > "$STARTTIME_FILE"

echo "Round2 controller started: PID=$controller_pid"
echo "Status: bash $SCRIPT_DIR/status_all.sh"
echo "Log:    $LOG_FILE"
