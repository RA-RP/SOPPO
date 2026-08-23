#!/usr/bin/env bash
# Run one online round2 method: persistent vLLM on GPU 2 plus TP=2 training on GPU 0-1.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

CONFIG_NAME="${1:-soppo_pe_sft_rollout_exp.yaml}"
METHOD_NAME="${CONFIG_NAME%.yaml}"
RESOLVED="${2:-$ROUND2_RUN_ROOT/$METHOD_NAME/config.resolved.yaml}"
RUN_DIR="$(cd "$(dirname "$RESOLVED")" && pwd)"
LOG_DIR="$RUN_DIR/logs"
STATUS_FILE="$RUN_DIR/controller_status.json"
ROLLOUT_PID_FILE="$RUN_DIR/rollout.pid"
[[ -f "$RESOLVED" ]] || {
    echo "ERROR: resolved config is missing; run 01_resolve_config.sh first" >&2
    exit 1
}
[[ ! -e "$STATUS_FILE" && ! -e "$ROLLOUT_PID_FILE" ]] || {
    echo "ERROR: Refuse to reuse round2 method attempt: $RUN_DIR" >&2
    exit 1
}

bash "$SCRIPT_DIR/00_preflight.sh" "$CONFIG_NAME" "$RESOLVED"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
TRAIN_GPU_IDS="$(round2_resolved_value "$RESOLVED" tensor_parallel.gpu_ids)"
ROLLOUT_GPU_IDS="$(round2_resolved_value "$RESOLVED" rollout.gpu_ids)"
GIT_COMMIT="$(round2_resolved_value "$RESOLVED" provenance.git_commit)"

update_status() {
    local state="$1"
    local failed_stage="${2:-}"
    local exit_code="${3:-0}"
    "$ROUND2_TRAIN_PYTHON" - "$STATUS_FILE" "$state" "$failed_stage" "$exit_code" \
        "$CONFIG_NAME" "$TRAIN_GPU_IDS" "$ROLLOUT_GPU_IDS" "$GIT_COMMIT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "state": sys.argv[2],
    "failed_stage": sys.argv[3] or None,
    "exit_code": int(sys.argv[4]),
    "method_config": sys.argv[5],
    "backend": "transformers-native-tp",
    "training_gpu_ids": sys.argv[6],
    "rollout_gpu_ids": sys.argv[7],
    "git_commit": sys.argv[8],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_suffix(path.suffix + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
}

ROLLOUT_PID=""
cleanup_worker() {
    if [[ -n "$ROLLOUT_PID" ]]; then
        if [[ -d "$RUN_DIR/rollouts" ]]; then
            : > "$RUN_DIR/rollouts/STOP"
        fi
        if kill -0 "$ROLLOUT_PID" 2>/dev/null; then
            kill "$ROLLOUT_PID" 2>/dev/null || true
        fi
        wait "$ROLLOUT_PID" 2>/dev/null || true
    fi
}
handle_signal() {
    cleanup_worker
    update_status "stopped" "signal" 143 || true
    trap - EXIT INT TERM
    exit 143
}
trap cleanup_worker EXIT
trap handle_signal INT TERM

update_status "starting_rollout"
CUDA_VISIBLE_DEVICES="$ROLLOUT_GPU_IDS" \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
    "$ROUND2_ROLLOUT_PYTHON" -m src.round2.run_rollout \
    --config "$RESOLVED" > "$LOG_DIR/vllm_worker.log" 2>&1 &
ROLLOUT_PID=$!
printf '%s\n' "$ROLLOUT_PID" > "$ROLLOUT_PID_FILE"

READY_FILE="$RUN_DIR/rollouts/worker.ready.json"
for _ in $(seq 1 600); do
    [[ -f "$READY_FILE" ]] && break
    if ! kill -0 "$ROLLOUT_PID" 2>/dev/null; then
        update_status "failed" "rollout_startup" 1
        echo "ERROR: vLLM worker exited before readiness" >&2
        tail -n 100 "$LOG_DIR/vllm_worker.log" >&2 || true
        exit 1
    fi
    sleep 1
done
if [[ ! -f "$READY_FILE" ]]; then
    update_status "failed" "rollout_startup_timeout" 1
    echo "ERROR: vLLM worker did not become ready within 10 minutes" >&2
    exit 1
fi

update_status "training"
set +e
CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS" \
    "$ROUND2_TRAIN_PYTHON" -m src.round2.run_tp \
    --config "$RESOLVED" > "$LOG_DIR/tp_train.log" 2>&1
TRAIN_STATUS=$?
set -e
if (( TRAIN_STATUS != 0 )); then
    update_status "failed" "tp_training" "$TRAIN_STATUS"
    exit "$TRAIN_STATUS"
fi

: > "$RUN_DIR/rollouts/STOP"
set +e
wait "$ROLLOUT_PID"
ROLLOUT_STATUS=$?
set -e
ROLLOUT_PID=""
if (( ROLLOUT_STATUS != 0 )); then
    update_status "failed" "rollout_worker" "$ROLLOUT_STATUS"
    exit "$ROLLOUT_STATUS"
fi
rm -f "$ROLLOUT_PID_FILE"
update_status "completed"
trap - EXIT INT TERM
echo "Round2 method completed: $CONFIG_NAME"
