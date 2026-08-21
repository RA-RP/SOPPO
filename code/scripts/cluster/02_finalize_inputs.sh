#!/bin/bash
# Materialize the DPO-100 oracle file and one immutable reference cache bundle.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init

DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
MODEL_DIR="${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}"
MODEL_MANIFEST="$MODEL_DIR/model_manifest.json"
CACHE_DIR="${SOPPO_REFERENCE_CACHE:-$CACHE_ROOT/soppo/reference/qwen3-4b-mvp-v0.5-30k}"
soppo_hardware_gate "$RUN_ROOT/$EXPERIMENT_ID/pipeline/hardware/reference_cache.csv"
python -m src.model.model_manifest --model-dir "$MODEL_DIR" --verify

ORACLE="$DATA_DIR/oracle_train.private.jsonl"
if [[ ! -f "$ORACLE" ]]; then
    python -m src.data.materialize_oracle \
        --labeled "$DATA_DIR/labeled_train.jsonl" \
        --unlabeled "$DATA_DIR/unlabeled_train.jsonl" \
        --private-labels "$DATA_DIR/private_labels/unlabeled_labels.jsonl" \
        --output "$ORACLE"
else
    python -m src.data.materialize_oracle \
        --labeled "$DATA_DIR/labeled_train.jsonl" \
        --unlabeled "$DATA_DIR/unlabeled_train.jsonl" \
        --private-labels "$DATA_DIR/private_labels/unlabeled_labels.jsonl" \
        --output "$ORACLE" --verify
fi
if [[ -f "$CACHE_DIR/COMPLETE" ]]; then
    echo "Reference cache bundle already complete: $CACHE_DIR"
    exit 0
fi
if [[ -e "$CACHE_DIR" ]]; then
    echo "ERROR: Incomplete reference cache directory exists: $CACHE_DIR" >&2
    exit 1
fi

PARTIAL="$CACHE_DIR.partial.$$"
mkdir -p "$PARTIAL/targets"
python -m src.training.cache_tools combine \
    --input "$DATA_DIR/labeled_train.jsonl" \
    --input "$DATA_DIR/labeled_val.jsonl" \
    --input "$DATA_DIR/unlabeled_train.jsonl" \
    --input "$DATA_DIR/test_inputs.jsonl" \
    --output "$PARTIAL/cache_source.jsonl"
soppo_torchrun -m src.training.reference_cache \
    --model "$MODEL_DIR" --model-manifest "$MODEL_MANIFEST" \
    --input "$PARTIAL/cache_source.jsonl" --output "$PARTIAL/combined.ref.jsonl" \
    --max-length 2048 --batch-size 4 --dtype bfloat16
python -m src.training.cache_tools split \
    --combined-cache "$PARTIAL/combined.ref.jsonl" \
    --target "$DATA_DIR/labeled_train.jsonl" --target "$DATA_DIR/labeled_val.jsonl" \
    --target "$DATA_DIR/unlabeled_train.jsonl" --target "$DATA_DIR/test_inputs.jsonl" \
    --target "$ORACLE" --output-dir "$PARTIAL/targets" \
    --model-manifest "$MODEL_MANIFEST" --max-length 2048
touch "$PARTIAL/COMPLETE"
mv "$PARTIAL" "$CACHE_DIR"
echo "Reference cache bundle complete: $CACHE_DIR/targets"
