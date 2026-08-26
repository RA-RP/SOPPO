#!/usr/bin/env bash
# Preview by default; --execute targets only the recorded Round3 controller group.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
PID_FILE="$ROUND3_RUN_ROOT/control/controller.pid"
PGID_FILE="$ROUND3_RUN_ROOT/control/controller.pgid"
STARTTIME_FILE="$ROUND3_RUN_ROOT/control/controller.starttime"
[[ -f "$PID_FILE" && -f "$PGID_FILE" && -f "$STARTTIME_FILE" ]] || { echo "No complete recorded Round3 controller identity."; exit 0; }
pid="$(<"$PID_FILE")"
pgid="$(<"$PGID_FILE")"
starttime="$(<"$STARTTIME_FILE")"
echo "Recorded Round3 controller PID=$pid PGID=$pgid"
if [[ "${1:-}" != "--execute" ]]; then
    echo "Preview only. Re-run with --execute only after explicit stop authorization."
    exit 0
fi
[[ "$pid" =~ ^[0-9]+$ && "$pgid" == "$pid" ]] || { echo "ERROR: invalid recorded controller identity" >&2; exit 1; }
[[ "$starttime" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid recorded controller start time" >&2; exit 1; }
[[ -r "/proc/$pid/stat" ]] || { echo "ERROR: recorded Round3 controller no longer exists" >&2; exit 1; }
[[ "$(awk '{print $22}' "/proc/$pid/stat")" == "$starttime" ]] || { echo "ERROR: recorded controller PID was reused" >&2; exit 1; }
actual="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
[[ "$actual" == "$pgid" ]] || { echo "ERROR: recorded controller identity is stale" >&2; exit 1; }
kill -TERM -- "-$pgid"
echo "TERM sent only to the recorded Round3 controller group. No checkpoint was deleted."
