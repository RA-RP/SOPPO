#!/bin/bash
# Stage 1: Data Preparation
# Server-only script - DO NOT run locally

set -euo pipefail

echo "=== Stage 1: Data Preparation ==="
echo "Cycle: cycle-20260818-01"
echo "Experiment design: v0.5 30k MVP (10% labeled / 80% unlabeled / 10% test)"
echo "Date: $(date)"

# Check if running on server
if [[ "${RUN_CONTEXT:-}" != "cluster" ]]; then
    echo "ERROR: This script must run on the server (RUN_CONTEXT=cluster)"
    exit 1
fi

# Parse arguments from the shared server path contract.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/server_paths.sh"
ENV_DIR=${1:-"$ENV_ROOT/youc"}
CODE_DIR=${2:-"$CODE_ROOT"}
DATA_DIR=${3:-"$DATA_ROOT/ultrafeedback/mvp-v0.5-30k"}

echo "Environment: $ENV_DIR"
echo "Code: $CODE_DIR"
echo "Data output: $DATA_DIR"

# Activate environment
source "$SCRIPT_DIR/runtime_env.sh"
soppo_activate_env "$ENV_DIR"
soppo_load_proxy

export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Add code to PYTHONPATH
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

# Check if data already exists
if [[ -f "$DATA_DIR/manifest_public.json" ]]; then
    python - "$DATA_DIR/manifest_public.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "labeled_train": 2700,
    "labeled_val": 300,
    "unlabeled_train": 24000,
    "test": 3000,
}
actual = {key: int(manifest.get(key, -1)) for key in expected}
if actual != expected:
    raise SystemExit(f"Existing data manifest violates the frozen 30k split: {actual}")
if manifest.get("split_ratios") != {
    "labeled_total": 0.10,
    "labeled_train": 0.09,
    "labeled_val": 0.01,
    "unlabeled_train": 0.80,
    "test": 0.10,
}:
    raise SystemExit("Existing data manifest has missing or incorrect split ratios")
PY
    python -m src.data.audit_prepared_data --data-dir "$DATA_DIR"
    echo "Verified existing immutable 30k dataset: $DATA_DIR"
    exit 0
fi

# Create data directory
mkdir -p "$DATA_DIR"

echo ""
echo "Downloading and processing UltraFeedback dataset..."
echo "This may take 20-60 minutes depending on network speed"
echo "Hugging Face endpoint: $HF_ENDPOINT"

# Run data preparation
python "$CODE_DIR/src/data/prepare_ultrafeedback.py" \
    --output_dir "$DATA_DIR" \
    --total_samples 30000 \
    --labeled_train_samples 2700 \
    --labeled_val_samples 300 \
    --unlabeled_samples 24000 \
    --test_samples 3000 \
    --seed 42 \
    --dataset_name "openbmb/UltraFeedback" \
    --max_seq_len 2048

PREP_EXIT_CODE=$?

if [[ $PREP_EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Data preparation failed with exit code $PREP_EXIT_CODE"
    exit $PREP_EXIT_CODE
fi

# Verify data files exist
echo ""
echo "Verifying data files..."

REQUIRED_FILES=(
    "labeled_train.jsonl"
    "labeled_val.jsonl"
    "unlabeled_train.jsonl"
    "test_inputs.jsonl"
    "private_labels/unlabeled_labels.jsonl"
    "private_labels/test_labels.jsonl"
    "manifest_public.json"
    "manifest_private.json"
)

ALL_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$DATA_DIR/$file" ]]; then
        echo "ERROR: Missing file: $file"
        ALL_EXIST=false
    else
        echo "✓ $file"
    fi
done

if [[ "$ALL_EXIST" != "true" ]]; then
    echo "ERROR: Some data files are missing"
    exit 1
fi

# Run audit checks
echo ""
echo "Running data audit checks..."

python -c "
import json
import sys

# Load manifests
with open('$DATA_DIR/manifest_public.json') as f:
    manifest = json.load(f)

print('=== Data Summary ===')
print(f\"Labeled train: {manifest['labeled_train']}\")
print(f\"Labeled val: {manifest['labeled_val']}\")
print(f\"Unlabeled: {manifest['unlabeled_train']}\")
print(f\"Test: {manifest['test']}\")

ratios = manifest['split_ratios']
assert ratios == {
    'labeled_total': 0.10,
    'labeled_train': 0.09,
    'labeled_val': 0.01,
    'unlabeled_train': 0.80,
    'test': 0.10,
}, ratios
assert manifest['labeled_train'] == 2700
assert manifest['labeled_val'] == 300
assert manifest['unlabeled_train'] == 24000
assert manifest['test'] == 3000
print('Split contract: 10% labeled (9% train + 1% validation), 80% unlabeled, 10% test')

print('\\n=== Position Randomization ===')
for split in ['unlabeled', 'test']:
    ratio = manifest['position_randomization_ratio'][split]
    print(f\"{split}: {ratio:.3f} (expected ~0.50)\")
    if abs(ratio - 0.5) > 0.05:
        print(f\"  WARNING: Deviation from 0.5 exceeds threshold\")

print('\\n=== Truncation Ratios ===')
for split in ['labeled', 'unlabeled', 'test']:
    ratio = manifest['truncation_ratio'][split]
    print(f\"{split}: {ratio:.3f} (expected <0.05)\")
    if ratio > 0.05:
        print(f\"  WARNING: Truncation ratio exceeds threshold\")
"

# Run cross-split leakage check
echo ""
echo "Checking for cross-split leakage..."

python -c "
import sys
sys.path.insert(0, '$CODE_DIR')

from src.data.data_utils import check_cross_split_leakage

report = check_cross_split_leakage(
    '$DATA_DIR/labeled_train.jsonl',
    '$DATA_DIR/unlabeled_train.jsonl',
    '$DATA_DIR/test_inputs.jsonl'
)

print('=== Cross-split Leakage Check ===')
print(f\"Labeled-Unlabeled overlap: {report['labeled_unlabeled_overlap']}\")
print(f\"Labeled-Test overlap: {report['labeled_test_overlap']}\")
print(f\"Unlabeled-Test overlap: {report['unlabeled_test_overlap']}\")

if report['no_leakage']:
    print('✓ No cross-split leakage detected')
else:
    print('ERROR: Cross-split leakage detected!')
    sys.exit(1)
"

AUDIT_EXIT_CODE=$?

if [[ $AUDIT_EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Data audit failed"
    exit $AUDIT_EXIT_CODE
fi

python -m src.data.audit_prepared_data --data-dir "$DATA_DIR"

echo ""
echo "=== Stage 1 Complete ==="
echo "Data prepared at: $DATA_DIR"
echo "Manifest (can be returned to local): $DATA_DIR/manifest_public.json"
echo ""
echo "Next: download Qwen3 if needed, then use submit_all.sh after code handoff authorization"
