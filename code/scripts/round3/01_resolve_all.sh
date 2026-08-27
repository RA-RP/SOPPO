#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-formal}"
PROFILE="${2:-all}"
case "$PROFILE" in
    all)
        methods=(
            dpo_1k
            sspo_code_loss_stratified_ultrachat_2df9e9a
            dpo_8k
            dpo_pe_sft_rollout
            dpo_pe_rollout_only
            dpo_pe_dpo_reward_sft_rollout
            dpo_pe_dpo_reward_rollout_only
        )
        ;;
    legacy)
        methods=(
            dpo_1k
            sspo_code_loss_stratified_ultrachat_2df9e9a
            dpo_8k
            dpo_pe_sft_rollout
            dpo_pe_rollout_only
        )
        ;;
    extension)
        methods=(
            dpo_pe_dpo_reward_sft_rollout
            dpo_pe_dpo_reward_rollout_only
        )
        ;;
    *) echo "ERROR: profile must be all, legacy, or extension" >&2; exit 1 ;;
esac
for method in "${methods[@]}"; do
    bash "$SCRIPT_DIR/01_resolve_config.sh" "$method" "$MODE" "$PROFILE"
done
