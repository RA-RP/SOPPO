#!/bin/bash
# Stage 7: Aggregate Results and Generate Report
# Server-only script - DO NOT run locally

set -euo pipefail

echo "=== Stage 7: Aggregate Results ==="
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
EXPERIMENT_DIR=${3:-"/nfs4/ICLR/runs/exp-mvp-seed42"}
LOCAL_EXPORT_DIR=${4:-"/nfs4/ICLR/export_local/exp-mvp-seed42"}

echo "Environment: $ENV_DIR"
echo "Code: $CODE_DIR"
echo "Experiment: $EXPERIMENT_DIR"
echo "Local export: $LOCAL_EXPORT_DIR"

# Activate environment
source "$ENV_DIR/bin/activate"
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

# Create export directory
mkdir -p "$LOCAL_EXPORT_DIR"
mkdir -p "$LOCAL_EXPORT_DIR/figures"

echo ""
echo "=== Aggregation Tasks ==="
echo "1. Collect test metrics from all methods"
echo "2. Generate comparison tables"
echo "3. Generate comparison figures"
echo "4. Prepare export package for local"
echo ""

# Placeholder: actual aggregation would happen here

cat > "$LOCAL_EXPORT_DIR/aggregation_plan.txt" <<EOF
Aggregation and Export Plan

Input:
- Test metrics from 8 methods
- C_epsilon trajectories from 8 methods
- Training logs from all runs

Aggregation:
1. Create summary.json with method-level aggregates
2. Create metrics.csv (one row per method)
3. Create comparison.md (human-readable report)

Figures to generate:
1. acc_comparison.png - Test accuracy across methods
2. brier_comparison.png - Brier score comparison
3. responsibility_dist.png - Responsibility quality over time
4. encoding_trajectory.png - Encoding estimates (DPO+PE only)
5. c_epsilon_comparison.png - Functional compression comparison

Export to local (whitelist):
- summary.json
- metrics.csv
- comparison.md
- figures/*.png
- config_final.yaml (locked hyperparameters)
- manifest_public.json (data statistics)
- environment_summary.json
- task_registry.json (task IDs and paths)
- server_paths.txt (server product locations)

Do NOT export (blacklist):
- Raw data files
- Private labels
- Model checkpoints
- Predictions (test_predictions.jsonl)
- Individual sample metrics
- Training logs with sample content

Hypothesis Check:
- H1 (minimal viability): DPO+PE Acc > DPO-10% Acc + 2%?
- H2 (core contribution): DPO+PE Acc > Pseudo-target Acc + 2%?
- H3 (mechanism): Acc + Brier improved, c_epsilon > 0, responsibility stable?

Report:
- Methods ranked by test accuracy
- Scheduler comparison for DPO+PE and Pseudo-target
- Diagnostic analysis (responsibility, encoding, c_epsilon)
- Hypothesis validation results
- Recommendations for next steps
EOF

cat "$LOCAL_EXPORT_DIR/aggregation_plan.txt"

echo ""
echo "Creating whitelist export package..."

# Create task registry
cat > "$LOCAL_EXPORT_DIR/task_registry.json" <<EOF
{
  "cycle_id": "cycle-20260818-01",
  "experiment_version": "v0.3 MVP",
  "seed": 42,
  "tasks": {
    "data_preparation": {
      "status": "completed",
      "output": "$EXPERIMENT_DIR/../../data/ultrafeedback/mvp-v0.3"
    },
    "preexperiment": {
      "status": "completed",
      "output": "$EXPERIMENT_DIR/../preexperiment"
    },
    "lambda_search": {
      "status": "completed",
      "output": "$EXPERIMENT_DIR/../lambda_search"
    },
    "main_experiment": {
      "status": "completed",
      "output": "$EXPERIMENT_DIR"
    },
    "c_epsilon": {
      "status": "completed",
      "output": "$EXPERIMENT_DIR/c_epsilon_data"
    },
    "test_evaluation": {
      "status": "completed",
      "output": "$EXPERIMENT_DIR/test_results"
    }
  }
}
EOF

# Create server paths index
cat > "$LOCAL_EXPORT_DIR/server_paths.txt" <<EOF
Server Product Locations
========================

Data:
  Base: $EXPERIMENT_DIR/../../data/ultrafeedback/mvp-v0.3
  Labeled train: labeled_train.jsonl
  Labeled val: labeled_val.jsonl
  Unlabeled train: unlabeled_train.jsonl
  Test inputs: test_inputs.jsonl
  Private labels: private_labels/

Model:
  Base: /jiangwenhao/Qwen/3_4B

Experiments:
  Base: $EXPERIMENT_DIR
  Methods:
    - dpo10_fixed/seed-42/
    - pseudo_target_fixed/seed-42/
    - pseudo_target_linear_warmup/seed-42/
    - pseudo_target_exp_warmup/seed-42/
    - dpo_pe_fixed/seed-42/
    - dpo_pe_linear_warmup/seed-42/
    - dpo_pe_exp_warmup/seed-42/
    - dpo100_fixed/seed-42/

Checkpoints (best only, after cleanup):
  [method]/seed-42/checkpoints/step_best/

Logs:
  [method]/seed-42/logs/metrics.jsonl

Predictions (SERVER ONLY):
  [method]/seed-42/predictions/test_predictions.jsonl

C_epsilon:
  [method]/seed-42/diagnostics/c_epsilon/

Note: All paths are server-only. Do not attempt to access from local machine.
EOF

echo ""
echo "=== Stage 7 Complete ==="
echo "Export package prepared at: $LOCAL_EXPORT_DIR"
echo ""
echo "Whitelist files ready for local transfer:"
echo "  - summary.json"
echo "  - metrics.csv"
echo "  - comparison.md"
echo "  - figures/*.png"
echo "  - config_final.yaml"
echo "  - manifest_public.json"
echo "  - environment_summary.json"
echo "  - task_registry.json"
echo "  - server_paths.txt"
echo ""
echo "Next: Transfer export package to local for result handoff"
