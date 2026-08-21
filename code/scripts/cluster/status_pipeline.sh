#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/server_paths.sh"
source "$SCRIPT_DIR/runtime_env.sh"
soppo_activate_env "$ENV_ROOT/youc"
REGISTRY="${1:-$RUN_ROOT/$EXPERIMENT_ID/pipeline/task_registry.json}"
test -f "$REGISTRY" || { echo "ERROR: registry not found: $REGISTRY" >&2; exit 1; }
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
python -m src.training.pipeline_status --registry "$REGISTRY"
