#!/bin/bash
# Stage 4: Main Experiment (MVP single seed)
# Server-only script - DO NOT run locally

set -euo pipefail

echo "=== Stage 4: Main Experiment (MVP) ==="
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
OUTPUT_DIR=${5:-"/nfs4/ICLR/runs/exp-mvp-seed42"}

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
echo "=== MVP Experiment Configuration ==="
echo "Seed: 42 (single seed)"
echo "Methods: DPO-10%, Pseudo-target, DPO+PE, DPO-100%"
echo "Schedulers (for Pseudo-target and DPO+PE): Fixed, Linear-Warmup, Exp-Warmup"
echo "Total runs: 8"
echo "  - DPO-10%: 1 run"
echo "  - Pseudo-target: 3 runs (3 schedulers)"
echo "  - DPO+PE: 3 runs (3 schedulers)"
echo "  - DPO-100%: 1 run"
echo ""

# Method configurations
METHODS=(
    "dpo10:fixed:0.0"
    "pseudo_target:fixed:LAMBDA_STAR"
    "pseudo_target:linear_warmup:LAMBDA_STAR"
    "pseudo_target:exp_warmup:LAMBDA_STAR"
    "dpo_pe:fixed:LAMBDA_STAR"
    "dpo_pe:linear_warmup:LAMBDA_STAR"
    "dpo_pe:exp_warmup:LAMBDA_STAR"
    "dpo100:fixed:0.0"
)

echo "Training configurations:"
for method_config in "${METHODS[@]}"; do
    IFS=':' read -r method scheduler lambda <<< "$method_config"
    echo "  - Method: $method, Scheduler: $scheduler, Lambda: $lambda"
done

echo ""
echo "Note: LAMBDA_STAR should be replaced with value from Stage 3"
echo ""

# Placeholder: actual training would happen here
# Each method would be an independent job

cat > "$OUTPUT_DIR/main_experiment_plan.txt" <<EOF
Main Experiment Plan for cycle-20260818-01 (MVP)

Configuration:
- Seed: 42 (single seed for MVP)
- Locked hyperparameters from Stage 2:
  - epsilon: [from preexperiment]
  - beta: [from preexperiment]
  - lr: [from preexperiment]
- Lambda: [from lambda search]

Training Runs (8 total):
1. DPO-10% (baseline)
   - Data: 900 labeled train only
   - Scheduler: N/A
   - Output: $OUTPUT_DIR/dpo10_fixed/seed-42/

2-4. Pseudo-target (3 schedulers)
   - Data: 900 labeled + 8k unlabeled
   - Schedulers: fixed, linear_warmup, exp_warmup
   - Output: $OUTPUT_DIR/pseudo_target_[scheduler]/seed-42/

5-7. DPO+PE (3 schedulers)
   - Data: 900 labeled + 8k unlabeled
   - Schedulers: fixed, linear_warmup, exp_warmup
   - Output: $OUTPUT_DIR/dpo_pe_[scheduler]/seed-42/

8. DPO-100% (oracle)
   - Data: all 9k labeled (unlabeled labels revealed)
   - Scheduler: N/A
   - Output: $OUTPUT_DIR/dpo100_fixed/seed-42/

Each run produces:
- config.yaml (frozen configuration)
- checkpoints/ (every 20 steps, ~10 total)
- logs/metrics.jsonl (step-by-step metrics)
- diagnostics/ (responsibility quality, encoding estimates)

Checkpoints:
- Keep all during C_epsilon observation (Stage 5)
- Delete intermediate checkpoints after C_epsilon complete
- Retain only best checkpoint per method

Evaluation:
- Validation: every 20 steps on 100 labeled val
- Test: only after training complete, on 1k test

Expected Duration:
- Training: ~2-4 hours per run
- Total: ~16-32 GPU hours
- Wall time: 1-2 days (with parallelization)
EOF

cat "$OUTPUT_DIR/main_experiment_plan.txt"

echo ""
echo "=== Stage 4 Planning Complete ==="
echo "Main experiment plan saved to: $OUTPUT_DIR/main_experiment_plan.txt"
echo ""
echo "Implementation needed:"
echo "1. Training script that takes method, scheduler, seed as arguments"
echo "2. Checkpoint saving every 20 steps"
echo "3. Validation evaluation every 20 steps"
echo "4. Diagnostic logging (responsibility, encoding estimates)"
echo ""
echo "After training completes:"
echo "1. Proceed to Stage 5 (C_epsilon observation)"
echo "2. Then Stage 6 (test evaluation)"
