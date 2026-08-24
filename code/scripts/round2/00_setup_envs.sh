#!/usr/bin/env bash
# Create two isolated environments because Transformers TP-LoRA and vLLM have different pins.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"
source "$SCRIPT_DIR/../standalone/runtime_env.sh"

TRAIN_REQUIREMENTS="$CODE_ROOT/requirements-round2-train.txt"
ROLLOUT_REQUIREMENTS="$CODE_ROOT/requirements-round2-rollout.txt"
[[ -f "$TRAIN_REQUIREMENTS" && -f "$ROLLOUT_REQUIREMENTS" ]] || {
    echo "ERROR: round2 requirement files are missing" >&2
    exit 1
}
CONDA_EXE="${SOPPO_CONDA_EXE:-}"
if [[ -z "$CONDA_EXE" ]]; then
    CONDA_EXE="$(soppo_find_conda || true)"
fi
[[ -n "$CONDA_EXE" && -x "$CONDA_EXE" ]] || {
    echo "ERROR: conda is required; set SOPPO_CONDA_EXE to its absolute path" >&2
    exit 1
}

mkdir -p "$ENV_ROOT" "$CACHE_ROOT/pip" "$CACHE_ROOT/conda/pkgs"
soppo_set_cache_env

create_env() {
    local env_dir="$1"
    local requirements="$2"
    if [[ ! -x "$env_dir/bin/python" ]]; then
        "$CONDA_EXE" create --prefix "$env_dir" python=3.11 pip -y
    fi
    "$CONDA_EXE" run --prefix "$env_dir" python -m pip install --upgrade pip
    "$CONDA_EXE" run --prefix "$env_dir" python -m pip install -r "$requirements"
    "$CONDA_EXE" run --prefix "$env_dir" python -m pip check
}

create_env "$ENV_ROOT/round2-train" "$TRAIN_REQUIREMENTS"
create_env "$ENV_ROOT/round2-rollout" "$ROLLOUT_REQUIREMENTS"

"$ENV_ROOT/round2-train/bin/python" - <<'PY'
import datasets
import peft
import torch
import transformers
import tqdm

print(
    "round2-train:",
    torch.__version__,
    transformers.__version__,
    peft.__version__,
    datasets.__version__,
    tqdm.__version__,
)
PY
"$ENV_ROOT/round2-rollout/bin/python" - <<'PY'
import torch
import transformers
import vllm

print("round2-rollout:", torch.__version__, transformers.__version__, vllm.__version__)
PY

echo "Round2 environments ready:"
echo "  $ENV_ROOT/round2-train"
echo "  $ENV_ROOT/round2-rollout"
