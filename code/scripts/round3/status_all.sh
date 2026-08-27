#!/usr/bin/env bash
# Read-only Round3 controller/method/GPU/disk status. Never sends signals.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
echo "Round3 experiment: $SOPPO_ROUND3_EXPERIMENT_ID"
if [[ -f "$ROUND3_RUN_ROOT/controller.json" ]]; then
    "$ROUND3_TRAIN_PYTHON" -m json.tool "$ROUND3_RUN_ROOT/controller.json"
else
    echo "controller.json: absent"
fi
if [[ -f "$ROUND3_RUN_ROOT/baseline_link.json" ]]; then
    echo "[immutable baseline link]"
    "$ROUND3_TRAIN_PYTHON" -m json.tool "$ROUND3_RUN_ROOT/baseline_link.json"
fi
for method in \
    dpo_1k \
    sspo_code_loss_stratified_ultrachat_2df9e9a \
    dpo_8k \
    dpo_pe_sft_rollout \
    dpo_pe_rollout_only \
    dpo_pe_dpo_reward_sft_rollout \
    dpo_pe_dpo_reward_rollout_only; do
    echo "[$method]"
    for path in \
        "$ROUND3_RUN_ROOT/control/formal/$method/status.json" \
        "$ROUND3_RUN_ROOT/formal/$method/state.json" \
        "$ROUND3_RUN_ROOT/formal/$method/best.json"; do
        if [[ -f "$path" ]]; then
            "$ROUND3_TRAIN_PYTHON" -m json.tool "$path"
        else
            echo "absent: $path"
        fi
    done
    metrics="$ROUND3_RUN_ROOT/formal/$method/logs/metrics.jsonl"
    [[ -f "$metrics" ]] && tail -n 3 "$metrics" || true
done
echo "automatic checkpoint pruner: disabled by resolved contract"
nvidia-smi
df -h "$SERVER_BASE"
