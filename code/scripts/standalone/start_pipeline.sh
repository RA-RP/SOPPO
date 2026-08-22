#!/bin/bash
# Launch the standalone pipeline in a detached process group.
set -euo pipefail

if [[ "${RUN_CONTEXT:-}" != "standalone" ]]; then
    echo "ERROR: RUN_CONTEXT=standalone is required" >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/server_paths.sh"
MODEL_DIR="${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}"
DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"

command -v nohup >/dev/null || { echo "ERROR: nohup is unavailable" >&2; exit 1; }
command -v setsid >/dev/null || { echo "ERROR: setsid is unavailable" >&2; exit 1; }
if [[ ! -f "$ENV_ROOT/youc/conda-meta/history" && ! -f "$ENV_ROOT/youc/bin/activate" ]]; then
    echo "ERROR: Environment is missing; run 00_server_setup.sh first" >&2
    exit 1
fi
test -f "$MODEL_DIR/model_manifest.json" || {
    echo "ERROR: Model is missing; run 02_download_model.sh first" >&2
    exit 1
}
test -f "$DATA_DIR/manifest_public.json" || {
    echo "ERROR: Data is missing; run 02_prepare_data.sh first" >&2
    exit 1
}
if [[ -n "$(git -C "$SOPPO_ROOT" status --porcelain)" ]]; then
    echo "ERROR: Server SOPPO checkout must be clean before pipeline start" >&2
    exit 1
fi

EXPERIMENT_DIR="$RUN_ROOT/$EXPERIMENT_ID"
PIPELINE_DIR="$EXPERIMENT_DIR/pipeline"
DRIVER_LOG="$EXPERIMENT_DIR/standalone-driver.log"
if [[ -e "$PIPELINE_DIR" || -e "$DRIVER_LOG" ]]; then
    echo "ERROR: Refuse to overwrite an existing standalone attempt" >&2
    echo "  pipeline: $PIPELINE_DIR" >&2
    echo "  driver:   $DRIVER_LOG" >&2
    exit 1
fi
mkdir -p "$EXPERIMENT_DIR"

export SOPPO_EXPECTED_GIT_COMMIT="$(git -C "$SOPPO_ROOT" rev-parse HEAD)"
nohup setsid bash "$SCRIPT_DIR/run_pipeline.sh" > "$DRIVER_LOG" 2>&1 < /dev/null &
PIPELINE_PID=$!
sleep 1
if ! kill -0 "$PIPELINE_PID" 2>/dev/null; then
    echo "ERROR: Pipeline process exited during launch" >&2
    tail -n 80 "$DRIVER_LOG" >&2 || true
    exit 1
fi

echo "Standalone pipeline started."
echo "PID: $PIPELINE_PID"
echo "Driver log: $DRIVER_LOG"
echo "Status: bash $SCRIPT_DIR/status_pipeline.sh"
