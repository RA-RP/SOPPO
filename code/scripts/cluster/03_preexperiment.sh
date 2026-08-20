#!/bin/bash
# Stage 2: Pre-experiment (numerical stability, headroom, hyperparameter search)
# Server-only script - DO NOT run locally

set -euo pipefail

echo "=== Stage 2: Pre-experiment ==="
echo "Cycle: cycle-20260818-01"
echo "Experiment: v0.3 MVP"
echo "Date: $(date)"

# Check if running on server
if [[ "${RUN_CONTEXT:-}" != "cluster" ]]; then
    echo "ERROR: This script must run on the server (RUN_CONTEXT=cluster)"
    exit 1
fi

# Parse arguments
ENV_DIR=${1:-"/nfs4/ICLR/envs/mvp-v0.1"}
CODE_DIR=${2:-"/nfs4/ICLR/code/cycle-20260818-01"}
DATA_DIR=${3:-"/nfs4/ICLR/data/ultrafeedback/mvp-v0.3"}
MODEL_PATH=${4:-"/jiangwenhao/Qwen/3_4B"}
OUTPUT_DIR=${5:-"/nfs4/ICLR/work/cycle-20260818-01/preexperiment"}

echo "Environment: $ENV_DIR"
echo "Code: $CODE_DIR"
echo "Data: $DATA_DIR"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"

# Activate environment
source "$ENV_DIR/bin/activate"
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "=== Pre-experiment Goals ==="
echo "1. Numerical stability: Test epsilon values (1e-5, 1e-6, 1e-8)"
echo "2. Headroom verification: DPO-100% vs SFT baseline (need ≥5% gap)"
echo "3. Hyperparameter selection: beta (0.1, 0.5), lr (5e-7, 1e-6)"
echo "4. Memory validation: batch_size=128 on 2×A800"
echo ""

# Note: This is a placeholder script
# The actual pre-experiment would involve:
# 1. Training DPO+PE with different epsilon values
# 2. Training SFT baseline on labeled data
# 3. Training DPO-100% with different (beta, lr) combinations
# 4. Evaluating on validation set (100 samples)
# 5. Selecting best hyperparameters

cat > "$OUTPUT_DIR/preexperiment_plan.txt" <<EOF
Pre-experiment Plan for cycle-20260818-01

Stage 2 Tasks:
1. Numerical Stability Test
   - Run DPO+PE with epsilon ∈ {1e-5, 1e-6, 1e-8}
   - Monitor for NaN/Inf, denominator degeneration
   - Select stable epsilon for main experiment

2. Headroom Verification
   - Train SFT baseline on 1k labeled (chosen responses)
   - Train DPO-100% on full 9k labeled
   - Evaluate both on validation set
   - Verify: DPO-100% Acc > SFT Acc + 5%
   - If gap < 5%: consider switching to smaller model or harder subset

3. Hyperparameter Search
   - Test beta ∈ {0.1, 0.5}
   - Test lr ∈ {5e-7, 1e-6}
   - 4 combinations total for DPO-100%
   - Evaluate on validation set (100 samples)
   - Select (beta*, lr*) with highest validation Acc

4. Memory Validation
   - Confirm batch_size=128 (per_device=8, grad_accum=8, devices=2)
   - If OOM: enable gradient checkpointing or reduce batch size
   - Record memory profile

Expected Duration: 1-2 days
Expected GPU Hours: 8-12 hours
EOF

cat "$OUTPUT_DIR/preexperiment_plan.txt"

echo ""
echo "=== Stage 2 Planning Complete ==="
echo "Pre-experiment plan saved to: $OUTPUT_DIR/preexperiment_plan.txt"
echo ""
echo "IMPORTANT: This stage requires actual training runs"
echo "Implement training scripts for:"
echo "  - Numerical stability test (DPO+PE with different epsilon)"
echo "  - SFT baseline"
echo "  - DPO-100% with hyperparameter grid search"
echo ""
echo "After completing pre-experiment:"
echo "1. Lock hyperparameters (epsilon, beta, lr) in configs/mvp/base.yaml"
echo "2. Record headroom gap in pre_experiment_report.md"
echo "3. Proceed to Stage 3 (lambda search) and Stage 4 (main experiment)"
