#!/bin/bash
# Run on gn001 after 00_server_setup.sh. No GPU and no direct compute-node SSH.
set -euo pipefail

if [[ "${RUN_CONTEXT:-}" != "cluster" ]]; then
    echo "ERROR: RUN_CONTEXT=cluster is required" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/server_paths.sh"
source "$SCRIPT_DIR/runtime_env.sh"
soppo_activate_env "$ENV_ROOT/youc"
soppo_load_proxy
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
export MODELSCOPE_CACHE="$CACHE_ROOT/modelscope"

MODEL_ID="Qwen/Qwen3-4B"
MODEL_DIR="$MODEL_ROOT/Qwen3-4B"
MANIFEST="$MODEL_DIR/model_manifest.json"

if [[ -f "$MANIFEST" ]]; then
    python -m src.model.model_manifest --model-dir "$MODEL_DIR" --verify
    echo "Model is already complete and immutable: $MODEL_DIR"
    exit 0
fi
if [[ -e "$MODEL_DIR" ]]; then
    echo "ERROR: Incomplete model directory exists: $MODEL_DIR" >&2
    echo "Move it aside after manual inspection; this script refuses to overwrite it." >&2
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
