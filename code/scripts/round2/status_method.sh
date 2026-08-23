#!/usr/bin/env bash
# Read-only status for one round2 TP/vLLM method.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

CONFIG_NAME="${1:-soppo_pe_sft_rollout_exp.yaml}"
METHOD_NAME="${CONFIG_NAME%.yaml}"
RESOLVED="${2:-$ROUND2_RUN_ROOT/$METHOD_NAME/config.resolved.yaml}"
RUN_DIR="$(cd "$(dirname "$RESOLVED")" && pwd)"
for status in "$RUN_DIR/controller_status.json" "$RUN_DIR/state.json"; do
    if [[ -f "$status" ]]; then
        echo "## $status"
        cat "$status"
        echo
    fi
done
if [[ ! -f "$RUN_DIR/controller_status.json" && ! -f "$RUN_DIR/state.json" ]]; then
    echo "Round2 status is not available: $RUN_DIR" >&2
    exit 1
fi
for log in "$RUN_DIR/logs/tp_train.log" "$RUN_DIR/logs/vllm_worker.log"; do
    if [[ -f "$log" ]]; then
        echo "## $log"
        tail -n 40 "$log"
        echo
    fi
done
