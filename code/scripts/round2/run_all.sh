#!/usr/bin/env bash
# Foreground controller: resolve, test, smoke, train, evaluate, and aggregate.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

STATUS_FILE="$ROUND2_RUN_ROOT/controller.json"
[[ ! -e "$STATUS_FILE" ]] || {
    echo "ERROR: Refuse to reuse round2 experiment: $ROUND2_RUN_ROOT" >&2
    exit 1
}
mkdir -p "$ROUND2_RUN_ROOT"
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
GIT_COMMIT="$(git -C "$SOPPO_ROOT" rev-parse HEAD)"

update_status() {
    local state="$1"
    local stage="$2"
    local exit_code="${3:-0}"
    "$ROUND2_TRAIN_PYTHON" - "$STATUS_FILE" "$state" "$stage" "$exit_code" \
        "$GIT_COMMIT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "state": sys.argv[2],
    "stage": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "experiment_id": Path(path).parent.name,
    "git_commit": sys.argv[5],
}
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
}

failed_stage="initialization"
handle_failure() {
    local exit_code=$?
    update_status "failed" "$failed_stage" "$exit_code" || true
    trap - ERR INT TERM
    exit "$exit_code"
}
handle_signal() {
    update_status "stopped" "$failed_stage" 143 || true
    trap - ERR INT TERM
    exit 143
}
trap handle_failure ERR
trap handle_signal INT TERM

failed_stage="resolve_configs"
update_status "running" "$failed_stage"
bash "$SCRIPT_DIR/01_resolve_all.sh"

failed_stage="server_tests"
update_status "running" "$failed_stage"
bash "$SCRIPT_DIR/00_server_tests.sh"

failed_stage="strong_smoke"
update_status "running" "$failed_stage"
bash "$SCRIPT_DIR/02_strong_smoke.sh"

for config_name in \
    soppo_pe_sft_rollout_exp.yaml \
    soppo_pe_rollout_only_exp.yaml; do
    failed_stage="formal_${config_name%.yaml}"
    update_status "running" "$failed_stage"
    bash "$SCRIPT_DIR/run_method.sh" "$config_name"
done

failed_stage="independent_evaluation"
update_status "running" "$failed_stage"
bash "$SCRIPT_DIR/03_evaluate.sh"

failed_stage="aggregate_export"
update_status "running" "$failed_stage"
bash "$SCRIPT_DIR/04_aggregate.sh"

trap - ERR INT TERM
update_status "completed" "all_methods" 0
echo "Complete round2 sequence finished: $ROUND2_RUN_ROOT"
