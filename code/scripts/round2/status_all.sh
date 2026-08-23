#!/usr/bin/env bash
# Read-only summary for the complete round2 controller and both methods.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

echo "Experiment: $ROUND2_EXPERIMENT_ID"
if [[ -f "$ROUND2_RUN_ROOT/controller.pid" ]]; then
    controller_pid="$(tr -d '[:space:]' < "$ROUND2_RUN_ROOT/controller.pid")"
    recorded_starttime=""
    actual_starttime=""
    [[ -f "$ROUND2_RUN_ROOT/controller.starttime" ]] && \
        recorded_starttime="$(tr -d '[:space:]' < "$ROUND2_RUN_ROOT/controller.starttime")"
    [[ -r "/proc/$controller_pid/stat" ]] && \
        actual_starttime="$(awk '{print $22}' "/proc/$controller_pid/stat")"
    if [[ "$controller_pid" =~ ^[0-9]+$ ]] \
        && kill -0 "$controller_pid" 2>/dev/null \
        && [[ -n "$recorded_starttime" && "$actual_starttime" == "$recorded_starttime" ]]; then
        echo "Controller process: RUNNING (PID=$controller_pid)"
    else
        echo "Controller process: NOT RUNNING (recorded PID=$controller_pid)"
    fi
fi
if [[ -f "$ROUND2_RUN_ROOT/controller.json" ]]; then
    cat "$ROUND2_RUN_ROOT/controller.json"
else
    echo "Controller status is not available yet."
fi

for config_name in \
    soppo_pe_sft_rollout_exp.yaml \
    soppo_pe_rollout_only_exp.yaml; do
    method_name="${config_name%.yaml}"
    method_status="$ROUND2_RUN_ROOT/$method_name/controller_status.json"
    if [[ -f "$method_status" ]]; then
        echo
        echo "## $method_name"
        cat "$method_status"
    fi
done

if [[ -f "$ROUND2_RUN_ROOT/controller.log" ]]; then
    echo
    echo "## controller.log (last 60 lines)"
    tail -n 60 "$ROUND2_RUN_ROOT/controller.log"
fi
