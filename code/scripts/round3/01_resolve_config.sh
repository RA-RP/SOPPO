#!/usr/bin/env bash
# Resolve one immutable Round3 method config outside the Git checkout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
METHOD="${1:?usage: 01_resolve_config.sh METHOD [strong_smoke|formal]}"
MODE="${2:-formal}"
[[ "$MODE" == "strong_smoke" || "$MODE" == "formal" ]] || { echo "ERROR: invalid Round3 mode: $MODE" >&2; exit 1; }
SOURCE_CONFIG="$ROUND3_CONFIG_DIR/$METHOD.yaml"
[[ -f "$SOURCE_CONFIG" ]] || { echo "ERROR: unknown Round3 method config: $SOURCE_CONFIG" >&2; exit 1; }
[[ -x "$ROUND3_TRAIN_PYTHON" ]] || { echo "ERROR: Round3 train environment is missing" >&2; exit 1; }
[[ -z "$(git -C "$SOPPO_ROOT" status --porcelain)" ]] || {
    echo "ERROR: Round3 configs can only resolve from a clean, reviewed checkout" >&2
    exit 1
}
GIT_COMMIT="$(git -C "$SOPPO_ROOT" rev-parse HEAD)"
UF_REVISION="$(round3_resolved_source_sha ultrafeedback)"
UC_REVISION="$(round3_resolved_source_sha ultrachat)"
MODEL_REVISION="$(round3_resolved_source_sha model)"
case "$METHOD" in
    dpo_1k) TRAIN_GPU=0 ;;
    sspo_code_loss_stratified_ultrachat_2df9e9a) TRAIN_GPU=1 ;;
    dpo_8k) TRAIN_GPU=2 ;;
    dpo_pe_sft_rollout|dpo_pe_rollout_only) TRAIN_GPU=0 ;;
    *) echo "ERROR: no Round3 GPU assignment for $METHOD" >&2; exit 1 ;;
esac

RUN_DIR="$ROUND3_RUN_ROOT/$MODE/$METHOD"
RESOLVED="$ROUND3_RUN_ROOT/resolved/$MODE/$METHOD.yaml"
[[ ! -e "$RESOLVED" ]] || { echo "ERROR: refuse to overwrite resolved config: $RESOLVED" >&2; exit 1; }
OVERRIDES=(
    --override "provenance.git_commit=$GIT_COMMIT"
    --override "provenance.experiment_id=$SOPPO_ROUND3_EXPERIMENT_ID"
    --override "execution.mode=$MODE"
    --override "model.name_or_path=$ROUND3_MODEL_DIR"
    --override "model.manifest_path=$ROUND3_MODEL_DIR/model_manifest.json"
    --override "model.resolved_revision=$MODEL_REVISION"
    --override "data.ultrafeedback_revision=$UF_REVISION"
    --override "data.ultrachat_revision=$UC_REVISION"
    --override "data.data_dir=$ROUND3_DATA_DIR"
    --override "data.reference_cache_dir=$ROUND3_REFERENCE_DIR"
    --override "training.train_gpu=$TRAIN_GPU"
    --override "training.physical_pair_subbatch=${SOPPO_ROUND3_PHYSICAL_PAIR_SUBBATCH:-1}"
    --override "rollout.artifact_dir=$RUN_DIR/rollouts"
    --override "output.run_dir=$RUN_DIR"
)
if [[ "$MODE" == "strong_smoke" ]]; then
    OVERRIDES+=(--override "execution.smoke_max_steps=1")
else
    PROJECTED="${SOPPO_ROUND3_PROJECTED_PEAK_BYTES:-}"
    if [[ -z "$PROJECTED" && -f "$ROUND3_RUN_ROOT/storage_projection.json" ]]; then
        PROJECTED="$($ROUND3_TRAIN_PYTHON - "$ROUND3_RUN_ROOT/storage_projection.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["projected_peak_bytes"])
PY
)"
    fi
    [[ "$PROJECTED" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: formal resolve requires SOPPO_ROUND3_PROJECTED_PEAK_BYTES from strong smoke" >&2
        exit 1
    }
    OVERRIDES+=(--override "storage.projected_peak_bytes=$PROJECTED")
fi
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
"$ROUND3_TRAIN_PYTHON" -m src.round3.validate_config \
    --config "$SOURCE_CONFIG" "${OVERRIDES[@]}" >/dev/null
mkdir -p "$(dirname "$RESOLVED")"
"$ROUND3_TRAIN_PYTHON" -m src.round3.validate_config \
    --config "$SOURCE_CONFIG" "${OVERRIDES[@]}" --write-resolved "$RESOLVED"
echo "$RESOLVED"
