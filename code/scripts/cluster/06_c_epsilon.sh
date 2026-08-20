#!/bin/bash
# Stage 5: C_epsilon Observation
# Server-only script - DO NOT run locally

set -euo pipefail

echo "=== Stage 5: C_epsilon Observation ==="
echo "Cycle: cycle-20260818-01"
echo "Experiment: v0.3 MVP"
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
DATA_DIR=${3:-"$DATA_ROOT/ultrafeedback/mvp-v0.3"}
EXPERIMENT_DIR=${4:-"$RUN_ROOT/$EXPERIMENT_ID"}
GETSLICE_DIR=${5:-"$OBSERVE_ROOT/GetSlice"}

echo "Environment: $ENV_DIR"
echo "Code: $CODE_DIR"
echo "Data: $DATA_DIR"
echo "Experiment: $EXPERIMENT_DIR"
echo "GetSlice: $GETSLICE_DIR"

# Activate environment
source "$SCRIPT_DIR/runtime_env.sh"
soppo_activate_env "$ENV_DIR"
export PYTHONPATH="$CODE_DIR:$GETSLICE_DIR:${PYTHONPATH:-}"

echo ""
echo "=== C_epsilon Observation Configuration ==="
echo "Checkpoints: 10 per method (every 20 steps)"
echo "  - Front: ckpt_1, ckpt_2, ckpt_3"
echo "  - Back: ckpt_5, ckpt_7, ckpt_9, ..."
echo "Probe setup:"
echo "  - S (Source): labeled_train.jsonl (900 samples)"
echo "  - X (Target): unlabeled_train.jsonl (8k samples)"
echo "  - s_nsamples: 128 (= batch_size)"
echo "  - x_nsamples: 128 (= batch_size)"
echo "Epsilon threshold: 0.05 (headline)"
echo "Module aggregation: Equal-5 non-QK (v, o, gate, up, down)"
echo "Target layer: 14 (middle layer)"
echo ""

# Find all methods and checkpoints
METHODS=(
    "dpo10_fixed"
    "pseudo_target_fixed"
    "pseudo_target_linear_warmup"
    "pseudo_target_exp_warmup"
    "dpo_pe_fixed"
    "dpo_pe_linear_warmup"
    "dpo_pe_exp_warmup"
    "dpo100_fixed"
)

echo "Methods to observe:"
for method in "${METHODS[@]}"; do
    echo "  - $method"
done

# Prepare data for GetSlice
echo ""
echo "Preparing data for GetSlice..."

# S-side: labeled_train.jsonl
# GetSlice expects format: {"question": "...", "answer": "..."}
python -c "
import jsonlines
import sys

labeled_path = '$DATA_DIR/labeled_train.jsonl'
output_path = '$EXPERIMENT_DIR/c_epsilon_data/s_labeled.jsonl'

import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with jsonlines.open(labeled_path) as reader:
    with jsonlines.open(output_path, 'w') as writer:
        for obj in reader:
            # Use prompt as question, response_a as answer
            writer.write({
                'question': obj['prompt'],
                'answer': obj['response_a']
            })

print(f'S-side data prepared: {output_path}')
"

# X-side: unlabeled_train.jsonl
# GetSlice expects format: {"output": {"text": "..."}}
python -c "
import jsonlines

unlabeled_path = '$DATA_DIR/unlabeled_train.jsonl'
output_path = '$EXPERIMENT_DIR/c_epsilon_data/x_unlabeled.jsonl'

with jsonlines.open(unlabeled_path) as reader:
    with jsonlines.open(output_path, 'w') as writer:
        for obj in reader:
            # Use response_a (could also use response_b)
            writer.write({
                'output': {
                    'text': obj['response_a']
                }
            })

print(f'X-side data prepared: {output_path}')
"

echo ""
echo "=== C_epsilon Observation Plan ==="

cat > "$EXPERIMENT_DIR/c_epsilon_plan.txt" <<EOF
C_epsilon Observation Plan

For each method × checkpoint:
1. Run GetSlice with config:
   - model: [checkpoint_path]
   - save_path: [method]/seed-42/diagnostics/c_epsilon/[step]
   - mode: split_whitened_svd
   - target_layer: 14
   - s_nsamples: 128
   - s_jsonl_path: c_epsilon_data/s_labeled.jsonl
   - x_nsamples: 128
   - x_jsonl_path: c_epsilon_data/x_unlabeled.jsonl

2. Output per checkpoint:
   - sMat_S.json (S-side singular values)
   - xMat_X_S.json (X-side singular values)
   - profile_s.pt (S-side profiling)
   - profile_x.pt (X-side profiling)

3. Post-processing:
   - Run cycle09_relative_functional_contraction.py
   - Run cycle09_equal5_non_qk.py
   - Generate c_epsilon_trajectory.csv per method

4. Cross-method comparison:
   - Compare c_epsilon across methods
   - Generate c_epsilon_comparison.png

Total GetSlice runs: 8 methods × 10 checkpoints = 80 runs
Expected duration: ~10-20 min per run
Total: ~13-27 GPU hours

After C_epsilon observation:
- Delete intermediate checkpoints (keep only best)
- Saves ~90% storage
EOF

cat "$EXPERIMENT_DIR/c_epsilon_plan.txt"

echo ""
echo "=== Stage 5 Planning Complete ==="
echo "C_epsilon plan saved to: $EXPERIMENT_DIR/c_epsilon_plan.txt"
echo ""
echo "Implementation needed:"
echo "1. Generate GetSlice config for each method × checkpoint"
echo "2. Run GetSlice (can parallelize)"
echo "3. Run post-processing scripts"
echo "4. Generate comparison visualizations"
echo ""
echo "After C_epsilon observation:"
echo "1. Proceed to Stage 6 (test evaluation)"
echo "2. Clean up intermediate checkpoints"
