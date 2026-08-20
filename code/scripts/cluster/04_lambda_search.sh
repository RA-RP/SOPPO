#!/bin/bash
# Stage 3: Lambda Search for DPO+PE and Pseudo-target
# Server-only script - DO NOT run locally

set -euo pipefail

echo "=== Stage 3: Lambda Search ==="
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
MODEL_PATH=${4:-"$MODEL_ROOT/Qwen3-4B"}
OUTPUT_DIR=${5:-"$RUN_ROOT/$EXPERIMENT_ID/lambda_search"}

echo "Environment: $ENV_DIR"
echo "Code: $CODE_DIR"
echo "Data: $DATA_DIR"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"

# Activate environment
source "$ENV_DIR/bin/activate"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "=== Lambda Search Configuration ==="
echo "Search space: {0.1, 0.3, 0.5, 1.0}"
echo "Methods: DPO+PE, Pseudo-target"
echo "Scheduler: Fixed (for search)"
echo "Seed: 42"
echo "Evaluation: Validation set (100 samples)"
echo ""

# Note: Placeholder for actual lambda search
# Would involve training both methods with each lambda value
# and selecting lambda* based on validation accuracy

cat > "$OUTPUT_DIR/lambda_search_plan.txt" <<EOF
Lambda Search Plan for cycle-20260818-01

Goal: Find optimal lambda_star for DPO+PE and Pseudo-target

Search Space:
- lambda ∈ {0.1, 0.3, 0.5, 1.0}
- Methods: DPO+PE, Pseudo-target
- Seed: 42
- Scheduler: Fixed (constant lambda)

Training:
- Data: 900 labeled train + 8k unlabeled
- Evaluation: 100 labeled validation
- Use locked hyperparameters from Stage 2 (epsilon, beta, lr)

Selection Criteria:
- lambda_star = argmax_lambda validation_accuracy
- Both methods use the same lambda_star (for fair comparison)

Expected:
- 4 lambda values × 2 methods = 8 training runs
- ~2-4 GPU hours per run
- Total: 16-32 GPU hours

Output:
- lambda_validation_curve.png
- lambda_search_report.md
- Selected lambda_star recorded in configs/mvp/base.yaml
EOF

cat "$OUTPUT_DIR/lambda_search_plan.txt"

echo ""
echo "=== Stage 3 Planning Complete ==="
echo "Lambda search plan saved to: $OUTPUT_DIR/lambda_search_plan.txt"
echo ""
echo "After lambda search completes:"
echo "1. Update configs/mvp/dpo_pe.yaml with lambda_star"
echo "2. Update configs/mvp/pseudo_target.yaml with same lambda_star"
echo "3. Proceed to Stage 4 (main experiment with 3 schedulers)"
