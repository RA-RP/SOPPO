#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-formal}"
for method in \
    dpo_1k \
    sspo_code_loss_stratified_ultrachat_2df9e9a \
    dpo_8k \
    dpo_pe_sft_rollout \
    dpo_pe_rollout_only; do
    bash "$SCRIPT_DIR/01_resolve_config.sh" "$method" "$MODE"
done

