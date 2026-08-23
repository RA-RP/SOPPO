#!/usr/bin/env bash
# Print the internal TP/vLLM commands without writing launch records or starting GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

CONFIG_NAME="${1:-soppo_pe_sft_rollout_exp.yaml}"
METHOD_NAME="${CONFIG_NAME%.yaml}"
RESOLVED="${2:-$ROUND2_RUN_ROOT/$METHOD_NAME/config.resolved.yaml}"
[[ -f "$RESOLVED" ]] || {
    echo "ERROR: resolved config is missing; run 01_resolve_config.sh first" >&2
    exit 1
}
[[ -x "$ROUND2_TRAIN_PYTHON" ]] || { echo "ERROR: missing $ROUND2_TRAIN_PYTHON" >&2; exit 1; }
[[ -x "$ROUND2_ROLLOUT_PYTHON" ]] || { echo "ERROR: missing $ROUND2_ROLLOUT_PYTHON" >&2; exit 1; }
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"

"$ROUND2_TRAIN_PYTHON" -m src.round2.run_tp --config "$RESOLVED" --dry-run
"$ROUND2_ROLLOUT_PYTHON" - "$RESOLVED" "$ROUND2_ROLLOUT_PYTHON" <<'PY'
import shlex
import sys
from src.round2.config import load_round2_config, validate_round2_config
from src.round2.rollout_backend import build_rollout_command, launch_spec_from_config

config = load_round2_config(sys.argv[1])
validate_round2_config(config)
spec = launch_spec_from_config(config, sys.argv[1], sys.argv[2])
print("CUDA_VISIBLE_DEVICES=" + spec.gpu_ids)
print(" ".join(shlex.quote(value) for value in build_rollout_command(spec)))
PY
