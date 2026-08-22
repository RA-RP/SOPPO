#!/bin/bash
# Download and freeze Qwen3-4B on the dedicated server.
set -euo pipefail

if [[ "${RUN_CONTEXT:-}" != "standalone" ]]; then
    echo "ERROR: RUN_CONTEXT=standalone is required" >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/server_paths.sh"
source "$SCRIPT_DIR/runtime_env.sh"
soppo_activate_env "$ENV_ROOT/youc"
soppo_set_cache_env
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

MODEL_ID="Qwen/Qwen3-4B"
MODEL_DIR="${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}"
MANIFEST="$MODEL_DIR/model_manifest.json"
if [[ -f "$MANIFEST" ]]; then
    python -m src.model.model_manifest --model-dir "$MODEL_DIR" --verify
    echo "Model is already complete and immutable: $MODEL_DIR"
    exit 0
fi
if [[ -e "$MODEL_DIR" ]]; then
    echo "ERROR: Incomplete model directory exists: $MODEL_DIR" >&2
    exit 1
fi

mkdir -p "$MODEL_ROOT"
PARTIAL_DIR="$MODEL_ROOT/.Qwen3-4B.partial.$$"
trap 'if [[ -d "$PARTIAL_DIR" ]]; then echo "Partial download retained for inspection: $PARTIAL_DIR" >&2; fi' EXIT
modelscope download --model "$MODEL_ID" --local_dir "$PARTIAL_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m src.model.model_manifest --model-dir "$PARTIAL_DIR" --write
mv "$PARTIAL_DIR" "$MODEL_DIR"
trap - EXIT
python -m src.model.model_manifest --model-dir "$MODEL_DIR" --verify
echo "Qwen3 model frozen at: $MODEL_DIR"
