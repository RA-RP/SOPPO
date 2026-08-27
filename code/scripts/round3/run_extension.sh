#!/usr/bin/env bash
# Foreground two-method extension controller; legacy formal must already be terminal.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
round3_require_baseline_experiment_id
CONTROLLER="$ROUND3_RUN_ROOT/controller.json"
[[ ! -e "$CONTROLLER" ]] || { echo "ERROR: refuse to reuse extension controller" >&2; exit 1; }
mkdir -p "$ROUND3_RUN_ROOT" "$ROUND3_LOG_ROOT"

update_controller() {
    local state="$1" stage="$2" code="${3:-0}"
    "$ROUND3_TRAIN_PYTHON" - "$CONTROLLER" "$state" "$stage" "$code" \
        "$SOPPO_ROUND3_EXPERIMENT_ID" "$SOPPO_ROUND3_BASELINE_EXPERIMENT_ID" \
        "$(git -C "$SOPPO_ROOT" rev-parse HEAD)" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "state": sys.argv[2], "stage": sys.argv[3], "exit_code": int(sys.argv[4]),
    "experiment_id": sys.argv[5], "baseline_experiment_id": sys.argv[6],
    "git_commit": sys.argv[7], "updated_at": datetime.now(timezone.utc).isoformat(),
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

[[ -f "$ROUND3_RUN_ROOT/baseline_link.json" ]] || {
    echo "ERROR: missing immutable extension baseline link" >&2
    exit 1
}
failed_stage=extension_readiness
update_controller running "$failed_stage"
"$ROUND3_TRAIN_PYTHON" - "$ROUND3_RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
methods = (
    "dpo_pe_dpo_reward_sft_rollout",
    "dpo_pe_dpo_reward_rollout_only",
)
projection = json.loads((root / "storage_projection.json").read_text(encoding="utf-8"))
if (
    projection.get("schema_version") != "round3.storage_projection.v2"
    or projection.get("projected_methods") != list(methods)
    or int(projection.get("projected_peak_bytes", 0)) <= 0
    or projection.get("automatic_deletion") is not False
):
    raise SystemExit("extension storage projection is missing or malformed")
for method in methods:
    status = json.loads(
        (root / "control" / "strong_smoke" / method / "status.json").read_text(
            encoding="utf-8"
        )
    )
    smoke = json.loads(
        (root / "strong_smoke" / method / "smoke_complete.json").read_text(
            encoding="utf-8"
        )
    )
    verified = json.loads(
        (
            root
            / "control"
            / "strong_smoke"
            / method
            / "checkpoint_verified.json"
        ).read_text(encoding="utf-8")
    )
    if (
        status.get("state") != "completed"
        or status.get("stage") != "training"
        or status.get("exit_code") != 0
        or smoke.get("status") != "succeeded"
        or smoke.get("method_id") != method
        or smoke.get("optimizer_steps") != 1
        or verified.get("status") != "verified"
        or verified.get("method_id") != method
        or verified.get("global_step") != 1
    ):
        raise SystemExit(f"extension strong-smoke evidence is incomplete: {method}")
PY
failed_stage=resolve_extension_formal
update_controller running "$failed_stage"
bash "$SCRIPT_DIR/01_resolve_all.sh" formal extension

failed_stage=extension_storage_gate
update_controller running "$failed_stage"
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
"$ROUND3_TRAIN_PYTHON" -m src.round3.storage_gate \
    --config "$ROUND3_RUN_ROOT/resolved/formal/dpo_pe_dpo_reward_sft_rollout.yaml" \
    --output "$ROUND3_RUN_ROOT/formal_storage_gate.json"

for method in dpo_pe_dpo_reward_sft_rollout dpo_pe_dpo_reward_rollout_only; do
    failed_stage="formal_$method"
    update_controller running "$failed_stage"
    bash "$SCRIPT_DIR/run_method.sh" "$method" formal
done

failed_stage=selected_checkpoint_final_test
update_controller running "$failed_stage"
bash "$SCRIPT_DIR/04_evaluate_extension.sh"

failed_stage=cross_run_sample_free_aggregate
update_controller running "$failed_stage"
bash "$SCRIPT_DIR/05_aggregate_extension.sh"

trap - ERR INT TERM
update_controller completed all_extension_methods 0
