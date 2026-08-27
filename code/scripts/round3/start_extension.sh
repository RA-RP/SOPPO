#!/usr/bin/env bash
# Start the reviewed extension in one isolated background process group.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
round3_require_baseline_experiment_id
CONTROL_DIR="$ROUND3_RUN_ROOT/control"
PID_FILE="$CONTROL_DIR/controller.pid"
PGID_FILE="$CONTROL_DIR/controller.pgid"
STARTTIME_FILE="$CONTROL_DIR/controller.starttime"
[[ ! -e "$PID_FILE" && ! -e "$PGID_FILE" && ! -e "$STARTTIME_FILE" && ! -e "$ROUND3_RUN_ROOT/controller.json" ]] || {
    echo "ERROR: extension controller evidence already exists; refuse duplicate launch" >&2
    exit 1
}
mkdir -p "$CONTROL_DIR" "$ROUND3_LOG_ROOT"
setsid nohup bash "$SCRIPT_DIR/run_extension.sh" \
    > "$ROUND3_LOG_ROOT/controller.log" 2>&1 < /dev/null &
pid=$!
pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
[[ "$pgid" == "$pid" ]] || { echo "ERROR: extension controller lacks isolated process group" >&2; exit 1; }
starttime="$(awk '{print $22}' "/proc/$pid/stat")"
printf '%s\n' "$pid" > "$PID_FILE"
printf '%s\n' "$pgid" > "$PGID_FILE"
printf '%s\n' "$starttime" > "$STARTTIME_FILE"
echo "Round3 DPO-reward extension controller started."

