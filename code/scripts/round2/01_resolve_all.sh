#!/usr/bin/env bash
# Freeze both formal round2 method configs under one experiment root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

for config_name in \
    soppo_pe_sft_rollout_exp.yaml \
    soppo_pe_rollout_only_exp.yaml; do
    bash "$SCRIPT_DIR/01_resolve_config.sh" "$config_name"
done

echo "Both formal round2 configs are resolved under: $ROUND2_RUN_ROOT"
