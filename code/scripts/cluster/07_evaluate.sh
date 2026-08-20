#!/bin/bash
# Stage 6: Test Evaluation
# Server-only script - DO NOT run locally

set -euo pipefail

echo "=== Stage 6: Test Evaluation ==="
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

echo "Environment: $ENV_DIR"
echo "Code: $CODE_DIR"
echo "Data: $DATA_DIR"
echo "Experiment: $EXPERIMENT_DIR"

# Activate environment
source "$SCRIPT_DIR/runtime_env.sh"
soppo_activate_env "$ENV_DIR"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

echo ""
echo "=== Test Evaluation Configuration ==="
echo "Test set: 1k samples (test_inputs.jsonl)"
echo "Private labels: private_labels/test_labels.jsonl"
echo "Evaluation: Best checkpoint per method"
echo "Metrics: Accuracy, Brier, Calibration, Confidence distribution"
echo ""

# Methods to evaluate
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

echo "Methods to evaluate:"
for method in "${METHODS[@]}"; do
    echo "  - $method"
done

# Placeholder: actual evaluation would happen here
# For each method:
# 1. Load best checkpoint
# 2. Load test_inputs.jsonl (without labels)
# 3. Compute predictions
# 4. Load private_labels/test_labels.jsonl
# 5. Compute metrics
# 6. Save predictions (server only)
# 7. Save metrics (can be returned)

echo ""
echo "Creating test evaluation script..."

cat > "$EXPERIMENT_DIR/evaluate_test.py" <<'EOF'
"""
Test evaluation script for all methods.
SERVER ONLY - DO NOT run locally.
"""

import sys
import json
from pathlib import Path

# This is a placeholder
# Actual implementation would:
# 1. Load each method's best checkpoint
# 2. Run evaluation on test set with private labels
# 3. Compute and save metrics

print("Test evaluation script created")
print("Implementation needed:")
print("1. Load policy model and reference model")
print("2. Load test dataloader (test_inputs.jsonl)")
print("3. Load private labels (test_labels.jsonl)")
print("4. Compute predictions and metrics")
print("5. Save results")
EOF

python "$EXPERIMENT_DIR/evaluate_test.py"

echo ""
echo "=== Test Evaluation Plan ==="

cat > "$EXPERIMENT_DIR/test_evaluation_plan.txt" <<EOF
Test Evaluation Plan

For each method:
1. Load best checkpoint (from checkpoints/step_best/)
2. Load test inputs (test_inputs.jsonl) - no labels visible
3. Compute predictions: p_i = σ(r_θ(x, y_a) - r_θ(x, y_b))
4. Load private labels (private_labels/test_labels.jsonl) - evaluation only
5. Compute metrics:
   - Accuracy: 1[(p_i > 0.5) = z_i]
   - Brier: (p_i - z_i)²
   - Calibration (ECE)
   - Confidence distribution

Output per method:
- predictions/ (SERVER ONLY)
  - test_predictions.jsonl (sample_id, prediction, label)
- test_metrics.json (can be returned)
- calibration.json (can be returned)

Aggregate results:
- Compare all methods on test set
- Generate comparison tables and figures
- Check hypothesis validity:
  - H1: DPO+PE > DPO-10% + 2%?
  - H2: DPO+PE > Pseudo-target + 2%?
  - H3: Mechanism validated? (Acc + Brier + c_epsilon)

Expected Duration: ~30 min per method
Total: ~4 hours
EOF

cat "$EXPERIMENT_DIR/test_evaluation_plan.txt"

echo ""
echo "=== Stage 6 Planning Complete ==="
echo "Test evaluation plan saved to: $EXPERIMENT_DIR/test_evaluation_plan.txt"
echo ""
echo "After test evaluation:"
echo "1. Proceed to Stage 7 (aggregation and reporting)"
echo "2. Generate final comparison report"
