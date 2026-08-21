#!/bin/bash
# CPU aggregation and local-transfer whitelist creation.
set -euo pipefail
SCRIPT_DIR="${SOPPO_CLUSTER_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init

EXPERIMENT_DIR="$RUN_ROOT/$EXPERIMENT_ID"
EXPORT_DIR="$EXPORT_ROOT/$EXPERIMENT_ID"
DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
TEST_SUMMARY="$EXPERIMENT_DIR/pipeline/server_tests/environment_summary.json"
DATA_AUDIT="$EXPERIMENT_DIR/pipeline/server_tests/data_audit.json"
REGISTRY="$EXPERIMENT_DIR/pipeline/task_registry.json"
FINAL_CONFIG="$EXPERIMENT_DIR/lambda_search/selection/config_final.yaml"
CEPSILON="$EXPERIMENT_DIR/c_epsilon/aggregate/c_epsilon_trajectory.csv"
test -f "$EXPERIMENT_DIR/c_epsilon/aggregate/complete.json"

python -m src.evaluation.aggregate \
    --experiment "$EXPERIMENT_DIR" --export "$EXPORT_DIR" \
    --data-manifest "$DATA_DIR/manifest_public.json" \
    --environment-summary "$TEST_SUMMARY" \
    --data-audit "$DATA_AUDIT" \
    --task-registry "$REGISTRY" --final-config "$FINAL_CONFIG" \
    --c-epsilon "$CEPSILON"
cat > "$EXPORT_DIR/server_paths.txt" <<EOF
experiment=$EXPERIMENT_DIR
data=$DATA_DIR
model=$MODEL_ROOT/Qwen3-4B
reference_cache=${SOPPO_REFERENCE_CACHE:-$CACHE_ROOT/soppo/reference/qwen3-4b-mvp-v0.5-30k}
raw_c_epsilon=$EXPERIMENT_DIR/c_epsilon/raw
private_predictions=$EXPERIMENT_DIR/evaluation
EOF
test -s "$EXPORT_DIR/c_epsilon_trajectory.csv"
test -s "$EXPORT_DIR/server_paths.txt"
printf 'succeeded\n' > "$EXPORT_DIR/EXPORT_COMPLETE"
echo "Whitelisted aggregate ready: $EXPORT_DIR"
