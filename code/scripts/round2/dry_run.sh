#!/usr/bin/env bash
# Dry-run round2 execution without starting Megatron or vLLM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

CONFIG_NAME="${1:-soppo_pe_sft_rollout_exp.yaml}"
CONFIG_PATH="$ROUND2_CONFIG_DIR/$CONFIG_NAME"
RUN_DIR="$ROUND2_RUN_ROOT/$CONFIG_NAME"
RESOLVED="$RUN_DIR/config.resolved.yaml"

[[ -f "$RESOLVED" ]] || {
    echo "ERROR: resolved config is missing; run 01_resolve_config.sh first" >&2
    exit 1
}
[[ -x "$ROUND2_PYTHON" ]] || { echo "ERROR: Python missing: $ROUND2_PYTHON" >&2; exit 1; }
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"

"$ROUND2_PYTHON" -m src.round2.run_megatron --config "$RESOLVED" --dry-run
"$ROUND2_PYTHON" -m src.round2.run_rollout --config "$RESOLVED" --dry-run
