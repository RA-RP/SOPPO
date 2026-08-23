#!/usr/bin/env bash
# Read-only status for a round2 method or experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

CONFIG_NAME="${1:-soppo_pe_sft_rollout_exp.yaml}"
RUN_DIR="$ROUND2_RUN_ROOT/$CONFIG_NAME"
STATUS_FILE="$RUN_DIR/status.json"
if [[ ! -f "$STATUS_FILE" ]]; then
    echo "Round2 status is not available: $STATUS_FILE" >&2
    exit 1
fi

cat "$STATUS_FILE"
for log in "$RUN_DIR/logs/megatron.log" "$RUN_DIR/logs/rollout.log"; do
    if [[ -f "$log" ]]; then
        echo
        echo "## $log"
        tail -n 40 "$log"
    fi
done
