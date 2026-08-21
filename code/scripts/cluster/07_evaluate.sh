#!/bin/bash
# Independent test evaluation for the eight preregistered final adapters.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init

INDEX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[[ "$INDEX" =~ ^[0-7]$ ]] || { echo "ERROR: evaluation array index must be 0..7" >&2; exit 2; }
METHODS=(
    dpo10
    dpo100
    sspo_hard_exp
    soppo_pe_exp
    soppo_pe_static_lambda_0.1
    soppo_pe_static_lambda_0.3
    soppo_pe_static_lambda_0.5
    soppo_pe_static_lambda_1.0
)
METHOD="${METHODS[$INDEX]}"
DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
MODEL_DIR="${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}"
MANIFEST="$MODEL_DIR/model_manifest.json"
CACHE_DIR="${SOPPO_REFERENCE_CACHE:-$CACHE_ROOT/soppo/reference/qwen3-4b-mvp-v0.5-30k}/targets"
RUN_DIR="$RUN_ROOT/$EXPERIMENT_ID/main/$METHOD"
OUTPUT="$RUN_ROOT/$EXPERIMENT_ID/evaluation/$METHOD"
soppo_hardware_gate "$RUN_ROOT/$EXPERIMENT_ID/pipeline/hardware/evaluation_${INDEX}.csv"

STEP="$(python - "$RUN_DIR/best.json" <<'PY'
import json
import sys
print(int(json.load(open(sys.argv[1], encoding="utf-8"))["step"]))
PY
)"
CHECKPOINT="$RUN_DIR/checkpoints/$(printf 'step_%06d' "$STEP")"
test -f "$CHECKPOINT/adapter_config.json" || {
    echo "ERROR: best LoRA adapter missing: $CHECKPOINT" >&2
    exit 1
}

python -m src.evaluation.evaluator \
    --checkpoint "$CHECKPOINT" \
    --base-model "$MODEL_DIR" \
    --model-manifest "$MANIFEST" \
    --test-inputs "$DATA_DIR/test_inputs.jsonl" \
    --private-labels "$DATA_DIR/private_labels/test_labels.jsonl" \
    --reference-cache "$CACHE_DIR" \
    --output "$OUTPUT" --max-length 2048 --batch-size 4 --dtype bfloat16
