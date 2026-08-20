#!/usr/bin/env bash
# Cycle 08 — re-apply local verl source patches (idempotent).
#
# The verl checkout at $VERL_ROOT is an editable install on autodl-tmp; if it is
# re-synced/re-copied (as happened on server switches) these patches are lost and
# must be re-applied before running OPD. Run this after any verl re-copy.
#
# Patch 1: distillation loss strategy dispatch missing "fsdp2".
#   verl/trainer/distillation/losses.py compute_topk_loss() only matched
#   `case "fsdp" | "veomni"`, so strategy=fsdp2 raised "Unsupported strategy".
#   fsdp2's loss computation is identical to fsdp (per the code's own comment),
#   and single-GPU student training REQUIRES fsdp2 (fsdp1 hits
#   "offload_to_cpu=True and NO_SHARD is not supported" in collect_lora_params).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERL_ROOT=${VERL_ROOT:-/root/autodl-tmp/verl}
F="$VERL_ROOT/verl/trainer/distillation/losses.py"

if grep -q 'case "fsdp" | "fsdp2" | "veomni":' "$F"; then
    echo "[patch1] already applied"
else
    sed -i 's/        case "fsdp" | "veomni":/        case "fsdp" | "fsdp2" | "veomni":/' "$F"
    grep -q 'case "fsdp" | "fsdp2" | "veomni":' "$F" && echo "[patch1] applied" || { echo "[patch1] FAILED"; exit 1; }
fi

# Audit-only: preserve response token ids, finish reason, and prompt id in
# trainer.rollout_data_dir. This patch does not mutate training data or loss.
python3 "$SCRIPT_DIR/patch_cycle09_verl_rollout_dump.py" --verl-root "$VERL_ROOT"
python3 -m py_compile \
    "$VERL_ROOT/verl/trainer/ppo/ray_trainer.py" \
    "$VERL_ROOT/verl/trainer/ppo/v1/trainer_base.py"
