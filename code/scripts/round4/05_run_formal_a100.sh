#!/usr/bin/env bash
# Server task: prepare or execute one recoverable Round4 formal arm on A100-2.
set -euo pipefail
umask 077

ACTION="${1:?usage: 05_run_formal_a100.sh prepare|base|dpo|sspo|staticpe|frozenpe RUN_ID}"
RUN_ID="${2:?missing RUN_ID}"
case "$ACTION" in prepare|base|dpo|sspo|staticpe|frozenpe) ;; *) echo "unknown action: $ACTION" >&2; exit 2 ;; esac

BASE="${ROUND4_A100_BASE:-/root}"
REPO="$BASE/SOPPO"
ENV_DIR="${ROUND4_A100_ENV:-$BASE/envs/round4-py312-92259df7fa61}"
MODEL="$BASE/models/Qwen3-1.7B-b9352fbb"
PREPARED="$BASE/data/round4-v2/prepared"
DATA_VIEW="$BASE/data/round4-v2/formal/$RUN_ID"
RUN_ROOT="$BASE/runs/round4/formal/$RUN_ID"
EXPORT_ROOT="$BASE/exports/round4/formal/$RUN_ID"
CODE_COMMIT="${ROUND4_CODE_COMMIT:-$(git -C "$REPO" rev-parse HEAD)}"
LOG_ROOT="$BASE/platform_logs/round4/$CODE_COMMIT/formal/$RUN_ID"
PYTHON="$ENV_DIR/bin/python"
LFACTORY="$ENV_DIR/bin/llamafactory-cli"

[[ -x "$PYTHON" && -x "$LFACTORY" ]] || { echo "Round4 environment is unavailable" >&2; exit 1; }
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$CODE_COMMIT" ]] || { echo "A100 checkout does not match ROUND4_CODE_COMMIT" >&2; exit 1; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "A100 worktree must be clean" >&2; exit 1; }
[[ -d "$MODEL" && -d "$PREPARED" ]] || { echo "model or prepared data is unavailable" >&2; exit 1; }

export PATH="$ENV_DIR/bin:/root/miniconda3/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO/SSPO/src_sspo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$BASE/cache/huggingface-round4"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$LOG_ROOT"

if [[ "$ACTION" == "prepare" ]]; then
    "$PYTHON" "$REPO/code/scripts/round4/05_prepare_formal.py" \
        --prepared-data-dir "$PREPARED" \
        --formal-data-dir "$DATA_VIEW" \
        --run-root "$RUN_ROOT" \
        --export-root "$EXPORT_ROOT" \
        --model-path "$MODEL" \
        --code-commit "$CODE_COMMIT" \
        --run-id "$RUN_ID" \
        2>&1 | tee "$LOG_ROOT/prepare.log"
    exit 0
fi

[[ -f "$RUN_ROOT/FORMAL_PLAN.json" && -f "$DATA_VIEW/dataset_info.json" ]] || {
    echo "formal plan is absent; run prepare first" >&2; exit 1;
}
ALPACA_ASSET="$($PYTHON "$REPO/code/scripts/round4/03_find_frozen_asset.py" \
    --search-root "$BASE/data/round4-v1/raw" \
    --repo-id tatsu-lab/alpaca_eval \
    --required-file alpaca_eval.json)"
REFERENCE_OUTPUTS="$ALPACA_ASSET/alpaca_eval_gpt4_baseline.json"
[[ -f "$REFERENCE_OUTPUTS" ]] || { echo "AlpacaEval reference outputs are unavailable" >&2; exit 1; }

generate_outputs() {
    local method="$1"
    local model_path="$2"
    local output="$EXPORT_ROOT/$method/alpacaeval_outputs.json"
    [[ ! -e "$output" ]] || { echo "refusing to overwrite generated outputs: $output" >&2; return 1; }
    mkdir -p "$(dirname "$output")"
    CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" "$REPO/SSPO/examples/evaluation/generate_alpacaeval_outputs.py" \
        --model_name_or_path "$model_path" \
        --model_revision b9352fbb8ce704292730cf54b3b1dceb2a808738 \
        --dataset_file "$ALPACA_ASSET/alpaca_eval.json" \
        --dataset_revision 2edc6fad8be6b14ea7230aabfd08188da6b8b814 \
        --max_samples 805 \
        --output_file "$output" \
        --generator_name "round4-${method}-${RUN_ID}" \
        --cache_dir "$HF_HOME" \
        --batch_size 8 \
        --max_prompt_length 1024 \
        --max_new_tokens 1024 \
        --dtype bfloat16 \
        2>&1 | tee "$LOG_ROOT/${method}-generation.log"
    "$PYTHON" "$REPO/code/scripts/round4/04_make_judge_request.py" \
        --export-root "$EXPORT_ROOT" \
        --reference-outputs "$REFERENCE_OUTPUTS" \
        --run-id "$RUN_ID" \
        --code-commit "$CODE_COMMIT" \
        --methods "$method" \
        --outputs-filename alpacaeval_outputs.json \
        --output-file "$EXPORT_ROOT/JUDGE_REQUEST_${method}.json" \
        2>&1 | tee "$LOG_ROOT/${method}-judge-request.log"
}

if [[ "$ACTION" == "base" ]]; then
    generate_outputs base "$MODEL"
    exit 0
fi

if [[ "$ACTION" == "frozenpe" ]]; then
    CANDIDATES="$DATA_VIEW/ultra_combined_fb0.1_ch0.1_frozenpe.json"
    [[ ! -e "$CANDIDATES" ]] || { echo "FrozenPE candidates already exist; use the recorded run instead of overwriting" >&2; exit 1; }
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO/SSPO/preprocessing_data/generate_frozenpe_candidates.py" \
        --input_file "$DATA_VIEW/ultra_combined_fb0.1_ch0.1.json" \
        --output_file "$CANDIDATES" \
        --dataset_info "$DATA_VIEW/dataset_info.json" \
        --dataset_name ultra_combined_fb0.1_ch0.1_frozenpe \
        --model_name_or_path "$MODEL" \
        --model_revision b9352fbb8ce704292730cf54b3b1dceb2a808738 \
        --cache_dir "$HF_HOME" \
        --batch_size 4 \
        --max_prompt_length 1024 \
        --max_new_tokens 1024 \
        --seed 42 \
        --dtype bfloat16 \
        --fail_on_drop \
        2>&1 | tee "$LOG_ROOT/frozenpe-candidates.log"
fi

export CUDA_VISIBLE_DEVICES=0,1
export FORCE_TORCHRUN=1
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=2
export MASTER_ADDR=127.0.0.1
case "$ACTION" in dpo) export MASTER_PORT=29641 ;; sspo) export MASTER_PORT=29642 ;; staticpe) export MASTER_PORT=29643 ;; frozenpe) export MASTER_PORT=29644 ;; esac
"$LFACTORY" train "$RUN_ROOT/configs/${ACTION}_train.json" 2>&1 | tee "$LOG_ROOT/${ACTION}-train.log"
CUDA_VISIBLE_DEVICES=0 "$LFACTORY" export "$RUN_ROOT/configs/${ACTION}_export.json" 2>&1 | tee "$LOG_ROOT/${ACTION}-export.log"
generate_outputs "$ACTION" "$EXPORT_ROOT/$ACTION/merged"
