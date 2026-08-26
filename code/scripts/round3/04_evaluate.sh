#!/usr/bin/env bash
# Verify common selection and evaluate only five selected checkpoints plus base.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
EVAL_ROOT="$ROUND3_RUN_ROOT/evaluations"
LOG_DIR="$ROUND3_LOG_ROOT/evaluation"
mkdir -p "$LOG_DIR"
base_config="$ROUND3_RUN_ROOT/resolved/formal/dpo_1k.yaml"
CUDA_VISIBLE_DEVICES=0,1,2 "$ROUND3_TRAIN_PYTHON" -m src.round3.preflight \
    --config "$base_config" --repo-root "$SOPPO_ROOT" \
    --output "$ROUND3_RUN_ROOT/control/evaluation_preflight.json" \
    --global-storage-evidence "$ROUND3_RUN_ROOT/formal_storage_gate.json"

for method in \
    dpo_1k \
    sspo_code_loss_stratified_ultrachat_2df9e9a \
    dpo_8k \
    dpo_pe_sft_rollout \
    dpo_pe_rollout_only; do
    run_dir="$ROUND3_RUN_ROOT/formal/$method"
    "$ROUND3_TRAIN_PYTHON" -m src.round3.selection --run-dir "$run_dir" \
        > "$LOG_DIR/${method}.selection.log" 2>&1
done

CUDA_VISIBLE_DEVICES=0 "$ROUND3_TRAIN_PYTHON" -m src.round3.final_evaluate \
    --config "$base_config" --frozen-base \
    --output-dir "$EVAL_ROOT/frozen_base" \
    --batch-size "${SOPPO_ROUND3_EVAL_BATCH_SIZE:-1}" \
    > "$LOG_DIR/frozen_base.log" 2>&1

for method in \
    dpo_1k \
    sspo_code_loss_stratified_ultrachat_2df9e9a \
    dpo_8k \
    dpo_pe_sft_rollout \
    dpo_pe_rollout_only; do
    config="$ROUND3_RUN_ROOT/resolved/formal/$method.yaml"
    best="$ROUND3_RUN_ROOT/formal/$method/best.json"
    checkpoint="$($ROUND3_TRAIN_PYTHON - "$best" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"])
PY
)"
    CUDA_VISIBLE_DEVICES=0 "$ROUND3_TRAIN_PYTHON" -m src.round3.final_evaluate \
        --config "$config" --checkpoint "$checkpoint" --best-json "$best" \
        --output-dir "$EVAL_ROOT/$method" \
        --batch-size "${SOPPO_ROUND3_EVAL_BATCH_SIZE:-1}" \
        > "$LOG_DIR/${method}.log" 2>&1
done
