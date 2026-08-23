#!/usr/bin/env bash
# Validate a resolved TP=2 + one-GPU vLLM round2 method before any GPU process starts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

CONFIG_NAME="${1:-soppo_pe_sft_rollout_exp.yaml}"
METHOD_NAME="${CONFIG_NAME%.yaml}"
RESOLVED="${2:-$ROUND2_RUN_ROOT/$METHOD_NAME/config.resolved.yaml}"
RUN_DIR="$(cd "$(dirname "$RESOLVED")" && pwd)"
[[ -f "$RESOLVED" ]] || {
    echo "ERROR: resolved config is missing; run 01_resolve_config.sh first" >&2
    exit 1
}
[[ -x "$ROUND2_TRAIN_PYTHON" ]] || { echo "ERROR: missing $ROUND2_TRAIN_PYTHON" >&2; exit 1; }
[[ -x "$ROUND2_ROLLOUT_PYTHON" ]] || { echo "ERROR: missing $ROUND2_ROLLOUT_PYTHON" >&2; exit 1; }
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"

if [[ -n "$(git -C "$SOPPO_ROOT" status --porcelain)" ]]; then
    echo "ERROR: formal round2 requires a clean SOPPO checkout" >&2
    exit 1
fi
EXPECTED_GIT_COMMIT="$(round2_resolved_value "$RESOLVED" provenance.git_commit)"
ACTUAL_GIT_COMMIT="$(git -C "$SOPPO_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_GIT_COMMIT" != "$EXPECTED_GIT_COMMIT" ]]; then
    echo "ERROR: server checkout differs from the resolved round2 commit" >&2
    echo "  resolved: $EXPECTED_GIT_COMMIT" >&2
    echo "  checkout: $ACTUAL_GIT_COMMIT" >&2
    exit 1
fi
for marker in \
    "$RUN_DIR/controller_status.json" \
    "$RUN_DIR/state.json" \
    "$RUN_DIR/tp_launch.resolved.json" \
    "$RUN_DIR/preflight" \
    "$RUN_DIR/rollouts/worker.ready.json"; do
    if [[ -e "$marker" ]]; then
        echo "ERROR: Refuse to reuse an existing round2 attempt: $marker" >&2
        exit 1
    fi
done

mapfile -t GPU_PROCESSES < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF && $1 != "N/A" {print $1}'
)
if (( ${#GPU_PROCESSES[@]} > 0 )); then
    echo "ERROR: round2 requires all three exclusive 4090 GPUs to be idle" >&2
    printf '  active GPU PID: %s\n' "${GPU_PROCESSES[@]}" >&2
    exit 1
fi

TRAIN_GPU_IDS="$(round2_resolved_value "$RESOLVED" tensor_parallel.gpu_ids)"
ROLLOUT_GPU_IDS="$(round2_resolved_value "$RESOLVED" rollout.gpu_ids)"
mkdir -p "$RUN_DIR/preflight"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS" \
    "$ROUND2_TRAIN_PYTHON" -m src.round2.preflight \
    --config "$RESOLVED" --role training \
    > "$RUN_DIR/preflight/training.json"
CUDA_VISIBLE_DEVICES="$ROLLOUT_GPU_IDS" \
    "$ROUND2_ROLLOUT_PYTHON" -m src.round2.preflight \
    --config "$RESOLVED" --role rollout \
    > "$RUN_DIR/preflight/rollout.json"

echo "Round2 preflight passed: $METHOD_NAME"
echo "Training GPUs: $TRAIN_GPU_IDS (TP=2, DP=1)"
echo "Rollout GPU:  $ROLLOUT_GPU_IDS (vLLM TP=1)"
echo "Evidence:     $RUN_DIR/preflight"
