#!/usr/bin/env bash
# Run one production-path optimizer step for each approved round2 method.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

SMOKE_ROOT="$ROUND2_RUN_ROOT/strong_smoke"
COMPLETE_FILE="$SMOKE_ROOT/complete.json"
[[ ! -e "$SMOKE_ROOT" ]] || {
    echo "ERROR: Refuse to overwrite strong smoke directory: $SMOKE_ROOT" >&2
    exit 1
}

for config_name in \
    soppo_pe_sft_rollout_exp.yaml \
    soppo_pe_rollout_only_exp.yaml; do
    method_name="${config_name%.yaml}"
    smoke_run_dir="$SMOKE_ROOT/$method_name"
    SOPPO_ROUND2_RUN_DIR="$smoke_run_dir" \
    SOPPO_ROUND2_MAX_STEPS=1 \
    SOPPO_ROUND2_SMOKE_MODE=true \
    SOPPO_ROUND2_EVAL_MAX_SAMPLES=8 \
    SOPPO_ROUND2_MIN_NEW_TOKENS=512 \
    SOPPO_ROUND2_IGNORE_EOS=true \
    SOPPO_ROUND2_SMOKE_OBJECTIVE_STEP=1 \
        bash "$SCRIPT_DIR/01_resolve_config.sh" "$config_name"
    bash "$SCRIPT_DIR/run_method.sh" \
        "$config_name" "$smoke_run_dir/config.resolved.yaml"
done

export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
GIT_COMMIT="$(git -C "$SOPPO_ROOT" rev-parse HEAD)"
"$ROUND2_TRAIN_PYTHON" - "$COMPLETE_FILE" "$SMOKE_ROOT" "$GIT_COMMIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
root = Path(sys.argv[2])
methods = (
    "soppo_pe_sft_rollout_exp",
    "soppo_pe_rollout_only_exp",
)
evidence = {}
for method in methods:
    run_dir = root / method
    complete = json.loads((run_dir / "complete.json").read_text())
    if complete.get("status") != "succeeded" or int(complete.get("steps", 0)) != 1:
        raise RuntimeError(f"Strong smoke method did not complete exactly one step: {method}")
    config_path = run_dir / "config.resolved.yaml"
    metric_lines = [
        json.loads(line)
        for line in (run_dir / "logs" / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    training_metric = next(row for row in metric_lines if "loss_pe" in row)
    if float(training_metric["aux_weight"]) <= 0:
        raise RuntimeError(f"Strong smoke did not exercise PE gradient weight: {method}")
    if int(training_metric["dynamic_pairs"]) != 56:
        raise RuntimeError(f"Strong smoke dynamic population changed: {method}")
    if int(training_metric["rollout_statistics"]["min_generated_tokens"]) != 512:
        raise RuntimeError(f"Strong smoke did not generate 512-token candidates: {method}")
    evidence[method] = {
        "complete": str(run_dir / "complete.json"),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "tp_evidence": str(run_dir / "tp_evidence.json"),
        "objective_scheduler_step": training_metric["objective_scheduler_step"],
        "aux_weight": training_metric["aux_weight"],
    }
payload = {
    "status": "succeeded",
    "git_commit": sys.argv[3],
    "methods": evidence,
    "contract": "tp2_online_rollout_exact_8_plus_56_max_length_v1",
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "Round2 strong smoke passed: $COMPLETE_FILE"
