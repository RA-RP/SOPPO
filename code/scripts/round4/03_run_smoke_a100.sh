#!/usr/bin/env bash
# Server task: run the fail-closed Round4 two-step smoke on exactly two A100 GPUs.
set -euo pipefail
umask 077

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

A100_BASE="${ROUND4_A100_BASE:-/root}"
REPO_ROOT="${ROUND4_REPO_ROOT:-$A100_BASE/SOPPO}"
ENV_ROOT="${ROUND4_ENV_ROOT:?set ROUND4_ENV_ROOT to the exact-commit Python 3.12 environment}"
EXPECTED_COMMIT="${ROUND4_EXPECTED_COMMIT:?set ROUND4_EXPECTED_COMMIT to the approved full Git SHA}"
MODEL_PATH="${ROUND4_MODEL_PATH:-$A100_BASE/models/Qwen3-1.7B-b9352fbb}"
RAW_ROOT="${ROUND4_RAW_DATA_ROOT:-$A100_BASE/data/round4-v1/raw}"
PREPARED_DIR="${ROUND4_PREPARED_DATA_DIR:-$A100_BASE/data/round4-v1/prepared}"
RUN_ID="${ROUND4_SMOKE_RUN_ID:-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${ROUND4_SMOKE_RUN_ROOT:-$A100_BASE/runs/round4/smoke/$RUN_ID}"
EXPORT_ROOT="${ROUND4_SMOKE_EXPORT_ROOT:-$A100_BASE/exports/round4/smoke/$RUN_ID}"
SMOKE_DATA_DIR="${ROUND4_SMOKE_DATA_DIR:-$A100_BASE/data/round4-v1/smoke/$RUN_ID}"
LOG_ROOT="${ROUND4_SMOKE_LOG_ROOT:-$A100_BASE/platform_logs/round4/$EXPECTED_COMMIT/smoke/$RUN_ID}"
HF_HOME="${ROUND4_HF_HOME:-$A100_BASE/cache/huggingface-round4}"
RUN_JUDGE="${ROUND4_RUN_JUDGE:-1}"
ALPACA_OUTPUTS="${ROUND4_SMOKE_ALPACA_OUTPUTS:-2}"

[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "ROUND4_EXPECTED_COMMIT must be a full lowercase Git SHA"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || fail "ROUND4_SMOKE_RUN_ID contains unsupported characters"
[[ "$RUN_JUDGE" == "1" ]] || fail "full-chain smoke requires ROUND4_RUN_JUDGE=1"
[[ "$ALPACA_OUTPUTS" =~ ^[1-9][0-9]*$ ]] || fail "ROUND4_SMOKE_ALPACA_OUTPUTS must be positive"
[[ -n "${OPENAI_API_KEY:-}" ]] || fail "OPENAI_API_KEY is absent; refusing to start a partial full-chain smoke"

for path in "$A100_BASE" "$REPO_ROOT" "$ENV_ROOT" "$MODEL_PATH" "$RAW_ROOT"; do
    [[ "$path" == /* && -e "$path" ]] || fail "required absolute path is missing: $path"
done
CANON_BASE="$(realpath -e "$A100_BASE")"
CANON_REPO="$(realpath -e "$REPO_ROOT")"
for path in "$CANON_REPO" "$(realpath -e "$ENV_ROOT")" "$(realpath -e "$MODEL_PATH")" "$(realpath -e "$RAW_ROOT")"; do
    case "$path" in
        "$CANON_BASE"|"$CANON_BASE"/*) ;;
        *) fail "Round4 path escapes A100 base: $path" ;;
    esac
done
for path in "$PREPARED_DIR" "$RUN_ROOT" "$EXPORT_ROOT" "$SMOKE_DATA_DIR" "$LOG_ROOT" "$HF_HOME"; do
    [[ "$path" == /* ]] || fail "Round4 writable path must be absolute: $path"
    resolved="$(realpath -m "$path")"
    case "$resolved" in
        "$CANON_BASE"/*) ;;
        *) fail "Round4 writable path escapes A100 base: $path" ;;
    esac
done
[[ -d "$CANON_REPO/.git" ]] || fail "ROUND4_REPO_ROOT is not the SOPPO Git checkout"
[[ "$(git -C "$CANON_REPO" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "checkout is not at expected commit"
[[ -z "$(git -C "$CANON_REPO" status --porcelain)" ]] || fail "checkout must be clean"
[[ "$(cat "$ENV_ROOT/ROUND4_CODE_COMMIT")" == "$EXPECTED_COMMIT" ]] || fail "Python environment is not bound to expected commit"

PYTHON="$ENV_ROOT/bin/python"
LF_CLI="$ENV_ROOT/bin/llamafactory-cli"
ALPACA_CLI="$ENV_ROOT/bin/alpaca_eval"
for executable in "$PYTHON" "$LF_CLI" "$ALPACA_CLI"; do
    [[ -x "$executable" ]] || fail "required environment executable is missing: $executable"
done

export PATH="$ENV_ROOT/bin:/root/miniconda3/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

"$PYTHON" - <<'PY'
import sys
import torch

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected Python 3.12, got {sys.version}")
if torch.__version__ != "2.5.1+cu124" or torch.version.cuda != "12.4":
    raise SystemExit(f"unexpected torch runtime: torch={torch.__version__}, cuda={torch.version.cuda}")
if torch.cuda.device_count() != 2:
    raise SystemExit(f"expected exactly two visible GPUs, got {torch.cuda.device_count()}")
for index in range(2):
    name = torch.cuda.get_device_name(index)
    if "A100" not in name:
        raise SystemExit(f"GPU {index} is not an A100: {name}")
print("Round4 A100 preflight passed")
PY

mkdir -p "$LOG_ROOT" "$HF_HOME"
[[ ! -e "$RUN_ROOT" ]] || fail "run root already exists: $RUN_ROOT"
[[ ! -e "$EXPORT_ROOT" ]] || fail "export root already exists: $EXPORT_ROOT"
[[ ! -e "$SMOKE_DATA_DIR" ]] || fail "smoke data directory already exists: $SMOKE_DATA_DIR"

FIND_ASSET="$CANON_REPO/code/scripts/round4/03_find_frozen_asset.py"
ULTRAFEEDBACK="$($PYTHON "$FIND_ASSET" --search-root "$RAW_ROOT" --repo-id HuggingFaceH4/ultrafeedback_binarized)"
ULTRACHAT="$($PYTHON "$FIND_ASSET" --search-root "$RAW_ROOT" --repo-id HuggingFaceH4/ultrachat_200k)"
ALPACA_ASSET="$($PYTHON "$FIND_ASSET" --search-root "$RAW_ROOT" --repo-id tatsu-lab/alpaca_eval --required-file alpaca_eval.json)"
"$PYTHON" "$CANON_REPO/code/scripts/round4/03_materialize_hf_cache.py" \
    --asset-dir "$ALPACA_ASSET" \
    --hf-home "$HF_HOME" \
    --expected-repo-id tatsu-lab/alpaca_eval \
    2>&1 | tee "$LOG_ROOT/alpaca-hub-cache.log"

read_revision() {
    "$PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads((Path(sys.argv[1]) / "ROUND4_ASSET_MANIFEST.json").read_text(encoding="utf-8"))
revision = manifest.get("resolved_revision")
if not isinstance(revision, str) or len(revision) != 40:
    raise SystemExit("invalid asset revision")
print(revision)
PY
}

ULTRAFEEDBACK_REVISION="$(read_revision "$ULTRAFEEDBACK")"
ULTRACHAT_REVISION="$(read_revision "$ULTRACHAT")"
ALPACA_REVISION="$(read_revision "$ALPACA_ASSET")"

if [[ ! -e "$PREPARED_DIR" ]]; then
    PREPARED_PARTIAL="${PREPARED_DIR}.partial-${EXPECTED_COMMIT:0:12}"
    [[ ! -e "$PREPARED_PARTIAL" ]] || fail "auditable partial preprocessing directory already exists: $PREPARED_PARTIAL"
    mkdir -p "$(dirname "$PREPARED_DIR")"
    "$PYTHON" "$CANON_REPO/SSPO/preprocessing_data/preprocessing_ultrachat.py" \
        --train_num_ratio 1 \
        --fb 0.1 \
        --ch 0.1 \
        --ultrafeedback_source "$ULTRAFEEDBACK" \
        --ultrachat_source "$ULTRACHAT" \
        --ultrafeedback_revision "$ULTRAFEEDBACK_REVISION" \
        --ultrachat_revision "$ULTRACHAT_REVISION" \
        --output_dir "$PREPARED_PARTIAL" \
        2>&1 | tee "$LOG_ROOT/preprocessing.log"
    mv "$PREPARED_PARTIAL" "$PREPARED_DIR"
fi

"$PYTHON" "$CANON_REPO/code/scripts/round4/03_verify_prepared.py" \
    --prepared-dir "$PREPARED_DIR" \
    --ultrafeedback-source "$ULTRAFEEDBACK" \
    --ultrachat-source "$ULTRACHAT" \
    2>&1 | tee "$LOG_ROOT/prepared-verify.log"

"$PYTHON" "$CANON_REPO/code/scripts/round4/03_prepare_smoke.py" \
    --prepared-data-dir "$PREPARED_DIR" \
    --smoke-data-dir "$SMOKE_DATA_DIR" \
    --run-root "$RUN_ROOT" \
    --export-root "$EXPORT_ROOT" \
    --model-path "$MODEL_PATH" \
    --model-revision b9352fbb8ce704292730cf54b3b1dceb2a808738 \
    --code-commit "$EXPECTED_COMMIT" \
    --run-id "$RUN_ID" \
    2>&1 | tee "$LOG_ROOT/smoke-plan.log"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$CANON_REPO/SSPO/preprocessing_data/generate_staticpe_candidates.py" \
    --input_file "$SMOKE_DATA_DIR/staticpe_input.json" \
    --output_file "$SMOKE_DATA_DIR/staticpe_train.json" \
    --dataset_info "$SMOKE_DATA_DIR/dataset_info.json" \
    --dataset_name round4_smoke_staticpe_train \
    --model_name_or_path "$MODEL_PATH" \
    --model_revision b9352fbb8ce704292730cf54b3b1dceb2a808738 \
    --cache_dir "$HF_HOME" \
    --batch_size 4 \
    --max_prompt_length 1024 \
    --max_new_tokens 64 \
    --seed 42 \
    --dtype bfloat16 \
    --fail_on_drop \
    2>&1 | tee "$LOG_ROOT/staticpe-candidates.log"

"$PYTHON" - "$SMOKE_DATA_DIR/staticpe_train.json.manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
counts = manifest.get("counts", {})
expected = {"input_total": 128, "input_labeled": 64, "input_unlabeled": 64, "output_total": 128, "output_unlabeled": 64}
for key, value in expected.items():
    if counts.get(key) != value:
        raise SystemExit(f"StaticPE candidate population mismatch: {key}={counts.get(key)!r}, expected {value}")
if counts.get("dropped_empty_generation") or counts.get("dropped_exact_duplicate"):
    raise SystemExit("StaticPE smoke candidate generation dropped rows")
print("StaticPE smoke candidate population passed")
PY

export FORCE_TORCHRUN=1
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=2
export MASTER_ADDR=127.0.0.1

for method_port in dpo:29541 sspo:29542 staticpe:29543; do
    METHOD="${method_port%%:*}"
    export MASTER_PORT="${method_port##*:}"
    TRAIN_CONFIG="$RUN_ROOT/configs/${METHOD}_train.json"
    EXPORT_CONFIG="$RUN_ROOT/configs/${METHOD}_export.json"
    METHOD_EXPORT="$EXPORT_ROOT/$METHOD"
    mkdir -p "$METHOD_EXPORT"

    "$LF_CLI" train "$TRAIN_CONFIG" 2>&1 | tee "$LOG_ROOT/${METHOD}-train.log"
    CUDA_VISIBLE_DEVICES=0 "$LF_CLI" export "$EXPORT_CONFIG" 2>&1 | tee "$LOG_ROOT/${METHOD}-export.log"

    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$CANON_REPO/SSPO/examples/staticpe/generate_alpacaeval_outputs.py" \
        --model_name_or_path "$METHOD_EXPORT/merged" \
        --model_revision b9352fbb8ce704292730cf54b3b1dceb2a808738 \
        --dataset_file "$ALPACA_ASSET/alpaca_eval.json" \
        --dataset_revision "$ALPACA_REVISION" \
        --max_samples "$ALPACA_OUTPUTS" \
        --output_file "$METHOD_EXPORT/alpacaeval_smoke_outputs.json" \
        --generator_name "round4-${METHOD}-${RUN_ID}" \
        --cache_dir "$HF_HOME" \
        --batch_size 2 \
        --max_prompt_length 1024 \
        --max_new_tokens 64 \
        --dtype bfloat16 \
        2>&1 | tee "$LOG_ROOT/${METHOD}-generation.log"

    "$ALPACA_CLI" \
        --model_outputs "$METHOD_EXPORT/alpacaeval_smoke_outputs.json" \
        --reference_outputs "$ALPACA_ASSET/alpaca_eval_gpt4_baseline.json" \
        --annotators_config weighted_alpaca_eval_gpt4_turbo \
        --output_path "$METHOD_EXPORT/alpacaeval_judge" \
        --max_instances 1 \
        2>&1 | tee "$LOG_ROOT/${METHOD}-judge.log"
done

"$PYTHON" "$CANON_REPO/code/scripts/round4/03_validate_smoke.py" \
    --run-root "$RUN_ROOT" \
    --export-root "$EXPORT_ROOT" \
    --run-id "$RUN_ID" \
    --code-commit "$EXPECTED_COMMIT" \
    --expected-alpaca-outputs "$ALPACA_OUTPUTS" \
    2>&1 | tee "$LOG_ROOT/smoke-validation.log"

printf 'Round4 full-chain smoke PASS: %s\n' "$RUN_ID"
