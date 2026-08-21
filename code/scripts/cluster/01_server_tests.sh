#!/bin/bash
# CPU/static-runtime test gate. Executed on the server by Slurm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init

OUTPUT_DIR="${1:-$RUN_ROOT/$EXPERIMENT_ID/pipeline/server_tests}"
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "ERROR: Refuse to overwrite server test output: $OUTPUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

test -d "$CODE_ROOT/tests" || { echo "ERROR: tests/ is mandatory" >&2; exit 1; }
pytest "$CODE_ROOT/tests" -v --tb=short | tee "$OUTPUT_DIR/pytest.txt"

python - <<'PY' > "$OUTPUT_DIR/environment_summary.json"
import json
import platform
import torch
import transformers
import datasets
import accelerate
import peft
import modelscope

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "accelerate": accelerate.__version__,
    "peft": peft.__version__,
    "modelscope": modelscope.__version__,
}, indent=2, sort_keys=True))
PY

python -m src.model.model_manifest \
    --model-dir "${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}" --verify
python -m src.data.audit_prepared_data \
    --data-dir "${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}" \
    --output "$OUTPUT_DIR/data_audit.json"
touch "$OUTPUT_DIR/TESTS_COMPLETE"
echo "Server test gate passed: $OUTPUT_DIR"
