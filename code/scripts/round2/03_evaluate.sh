#!/usr/bin/env bash
# Evaluate both validation-selected round2 adapters on the private-label test split.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"

mapfile -t GPU_PROCESSES < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF && $1 != "N/A" {print $1}'
)
if (( ${#GPU_PROCESSES[@]} > 0 )); then
    echo "ERROR: Round2 evaluation requires the training/rollout processes to exit" >&2
    printf '  active GPU PID: %s\n' "${GPU_PROCESSES[@]}" >&2
    exit 1
fi

for config_name in \
    soppo_pe_sft_rollout_exp.yaml \
    soppo_pe_rollout_only_exp.yaml; do
    method_name="${config_name%.yaml}"
    resolved="$ROUND2_RUN_ROOT/$method_name/config.resolved.yaml"
    [[ -f "$resolved" ]] || { echo "ERROR: missing $resolved" >&2; exit 1; }
    evaluation_log="$ROUND2_RUN_ROOT/$method_name/logs/evaluation.log"
    [[ ! -e "$evaluation_log" ]] || {
        echo "ERROR: Refuse to overwrite evaluation log: $evaluation_log" >&2
        exit 1
    }
    evaluation_gpu="$(round2_resolved_value "$resolved" evaluation.gpu_id)"
    CUDA_VISIBLE_DEVICES="$evaluation_gpu" \
        "$ROUND2_TRAIN_PYTHON" -m src.round2.evaluate \
        --config "$resolved" \
        > "$evaluation_log" 2>&1
    echo "Evaluated round2 method: $method_name"
done
