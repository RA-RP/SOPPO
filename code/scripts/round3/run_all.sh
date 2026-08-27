#!/usr/bin/env bash
# Foreground formal controller. Requires prior reviewed code and successful strong smoke.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
CONTROLLER="$ROUND3_RUN_ROOT/controller.json"
[[ ! -e "$CONTROLLER" ]] || { echo "ERROR: refuse to reuse Round3 formal controller" >&2; exit 1; }
mkdir -p "$ROUND3_RUN_ROOT" "$ROUND3_LOG_ROOT"

update_controller() {
    local state="$1"
    local stage="$2"
    local code="${3:-0}"
    "$ROUND3_TRAIN_PYTHON" - "$CONTROLLER" "$state" "$stage" "$code" \
        "$SOPPO_ROUND3_EXPERIMENT_ID" "$(git -C "$SOPPO_ROOT" rev-parse HEAD)" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "state": sys.argv[2], "stage": sys.argv[3], "exit_code": int(sys.argv[4]),
    "experiment_id": sys.argv[5], "git_commit": sys.argv[6],
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "automatic_checkpoint_pruner": False,
}
partial = path.with_suffix(".json.partial")
partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
partial.replace(path)
PY
}
failed_stage=initialization
trap 'code=$?; update_controller failed "$failed_stage" "$code" || true; exit "$code"' ERR
trap 'update_controller stopped "$failed_stage" 143 || true; exit 143' INT TERM

failed_stage=resolve_formal
update_controller running "$failed_stage"
bash "$SCRIPT_DIR/01_resolve_all.sh" formal

failed_stage=formal_storage_gate
update_controller running "$failed_stage"
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
"$ROUND3_TRAIN_PYTHON" -m src.round3.storage_gate \
    --config "$ROUND3_RUN_ROOT/resolved/formal/dpo_1k.yaml" \
    --output "$ROUND3_RUN_ROOT/formal_storage_gate.json"

static_methods=(
    dpo_1k
    sspo_code_loss_stratified_ultrachat_2df9e9a
    dpo_8k
)
static_pids=()
failed_stage=formal_static_parallel
update_controller running "$failed_stage"
for method in "${static_methods[@]}"; do
    bash "$SCRIPT_DIR/run_method.sh" "$method" formal &
    static_pids+=("$!")
done
static_failed=0
for index in "${!static_pids[@]}"; do
    if ! wait "${static_pids[$index]}"; then
        echo "ERROR: concurrent Round3 formal method failed: ${static_methods[$index]}" >&2
        static_failed=1
    fi
done
(( static_failed == 0 ))

for method in \
    dpo_pe_sft_rollout \
    dpo_pe_rollout_only \
    dpo_pe_dpo_reward_sft_rollout \
    dpo_pe_dpo_reward_rollout_only; do
    failed_stage="formal_$method"
    update_controller running "$failed_stage"
    bash "$SCRIPT_DIR/run_method.sh" "$method" formal
done

failed_stage=selected_checkpoint_final_test
update_controller running "$failed_stage"
bash "$SCRIPT_DIR/04_evaluate.sh"

failed_stage=sample_free_aggregate
update_controller running "$failed_stage"
bash "$SCRIPT_DIR/05_aggregate.sh"

trap - ERR INT TERM
update_controller completed all_methods 0
