#!/usr/bin/env bash
# Download and freeze Qwen/Qwen3-1.7B; resolved revision must be supplied, never guessed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
[[ -x "$ROUND3_TRAIN_PYTHON" ]] || { echo "ERROR: Round3 train environment is missing" >&2; exit 1; }
MODEL_REVISION="$(round3_resolved_source_sha model)"
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"

if [[ -f "$ROUND3_MODEL_DIR/model_manifest.json" ]]; then
    "$ROUND3_TRAIN_PYTHON" -m src.round3.model_manifest --model-dir "$ROUND3_MODEL_DIR" --verify
    recorded="$($ROUND3_TRAIN_PYTHON - "$ROUND3_MODEL_DIR/model_manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["resolved_revision"])
PY
)"
    [[ "$recorded" == "$MODEL_REVISION" ]] || {
        echo "ERROR: existing Qwen3-1.7B manifest revision differs from requested revision" >&2
        exit 1
    }
    exit 0
fi
[[ ! -e "$ROUND3_MODEL_DIR" ]] || { echo "ERROR: incomplete model directory exists: $ROUND3_MODEL_DIR" >&2; exit 1; }
mkdir -p "$MODEL_ROOT"
PARTIAL="$MODEL_ROOT/.Qwen3-1.7B.partial.$$"
trap 'echo "Partial model download retained for inspection: $PARTIAL" >&2' EXIT
"$ROUND3_TRAIN_ENV/bin/modelscope" download \
    --model "Qwen/Qwen3-1.7B" \
    --revision "$MODEL_REVISION" \
    --local_dir "$PARTIAL"
"$ROUND3_TRAIN_PYTHON" -m src.round3.model_manifest \
    --model-dir "$PARTIAL" --resolved-revision "$MODEL_REVISION" --write
mv "$PARTIAL" "$ROUND3_MODEL_DIR"
trap - EXIT
"$ROUND3_TRAIN_PYTHON" -m src.round3.model_manifest --model-dir "$ROUND3_MODEL_DIR" --verify
