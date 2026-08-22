#!/bin/bash
# Create the frozen Python 3.10 environment on the dedicated server.
set -euo pipefail

if [[ "${RUN_CONTEXT:-}" != "standalone" ]]; then
    echo "ERROR: RUN_CONTEXT=standalone is required" >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/server_paths.sh"
source "$SCRIPT_DIR/runtime_env.sh"

test -f "$CODE_ROOT/requirements.lock.txt" || {
    echo "ERROR: requirements.lock.txt not found: $CODE_ROOT/requirements.lock.txt" >&2
    exit 1
}
test -d "$SOPPO_ROOT/.git" || {
    echo "ERROR: SOPPO must be the only Git repository root: $SOPPO_ROOT" >&2
    exit 1
}
test -d "$ICLR_ROOT" || {
    echo "ERROR: Static ICLR sibling directory is missing: $ICLR_ROOT" >&2
    exit 1
}
test ! -e "$ICLR_ROOT/.git" || {
    echo "ERROR: ICLR must remain static and non-Git: $ICLR_ROOT/.git" >&2
    exit 1
}
test -d "$OBSERVE_ROOT" || {
    echo "ERROR: Required static observation source is missing: $OBSERVE_ROOT" >&2
    exit 1
}
if find "$OBSERVE_ROOT" -type d -name .git -print -quit | grep -q .; then
    echo "ERROR: Nested Git repository found under $OBSERVE_ROOT" >&2
    exit 1
fi

echo "=== Standalone Environment Setup ==="
echo "Base: $SERVER_BASE"
echo "SOPPO (only Git repo): $SOPPO_ROOT"
echo "ICLR (static, no Git): $ICLR_ROOT"

mkdir -p "$ENV_ROOT" "$DATA_ROOT" "$MODEL_ROOT" "$RUN_ROOT" "$EXPORT_ROOT" \
    "$PLATFORM_LOG_ROOT" "$CACHE_ROOT/pip" "$CACHE_ROOT/conda/pkgs" \
    "$CACHE_ROOT/huggingface" "$CACHE_ROOT/modelscope"
soppo_set_cache_env

ENV_DIR="$ENV_ROOT/youc"
if [[ -e "$ENV_DIR" && ! -f "$ENV_DIR/conda-meta/history" && ! -f "$ENV_DIR/bin/activate" ]]; then
    echo "ERROR: Existing environment directory is incomplete: $ENV_DIR" >&2
    exit 1
fi

if [[ ! -e "$ENV_DIR" ]]; then
    if CONDA_EXE="$(soppo_find_conda)"; then
        echo "Creating Conda environment with Python 3.10: $ENV_DIR"
        "$CONDA_EXE" create --prefix "$ENV_DIR" python=3.10 pip -y
    else
        PYTHON_BOOTSTRAP="${SOPPO_PYTHON_BIN:-$(command -v python3.10 || true)}"
        if [[ -z "$PYTHON_BOOTSTRAP" ]]; then
            echo "ERROR: Neither conda nor python3.10 is available." >&2
            echo "Install one of them, or set SOPPO_CONDA_EXE / SOPPO_PYTHON_BIN." >&2
            exit 1
        fi
        if [[ "$("$PYTHON_BOOTSTRAP" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
            echo "ERROR: SOPPO_PYTHON_BIN must point to Python 3.10" >&2
            exit 1
        fi
        echo "Creating Python 3.10 venv: $ENV_DIR"
        "$PYTHON_BOOTSTRAP" -m venv "$ENV_DIR"
    fi
fi

soppo_activate_env "$ENV_DIR"
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "$CODE_ROOT/requirements.lock.txt"

python - <<'PY'
import accelerate
import datasets
import modelscope
import peft
import torch
import transformers
from transformers import Qwen3ForCausalLM

print(f"Python environment verified")
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}; visible GPUs: {torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(f"  GPU {index}: {properties.name}, {properties.total_memory // (1024**2)} MiB")
print(f"Transformers: {transformers.__version__}")
print(f"Datasets: {datasets.__version__}")
print(f"Accelerate: {accelerate.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"ModelScope: {modelscope.__version__}")
PY

cat > "$SERVER_BASE/project_config.json" <<EOF
{
  "cycle_id": "cycle-20260818-01",
  "execution_platform": "standalone",
  "base_dir": "$SERVER_BASE",
  "iclr_dir": "$ICLR_ROOT",
  "soppo_dir": "$SOPPO_ROOT",
  "code_dir": "$CODE_ROOT",
  "env_dir": "$ENV_DIR",
  "data_dir": "$DATA_ROOT",
  "models_dir": "$MODEL_ROOT",
  "runs_dir": "$RUN_ROOT",
  "exports_dir": "$EXPORT_ROOT",
  "cache_dir": "$CACHE_ROOT",
  "created_at": "$(date -Iseconds)"
}
EOF

cat > "$SERVER_BASE/activate_env.sh" <<'ACTIVATE_EOF'
#!/bin/bash
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_HELPER="$BASE_DIR/SOPPO/code/scripts/standalone/runtime_env.sh"
PATH_HELPER="$BASE_DIR/SOPPO/code/scripts/standalone/server_paths.sh"
if [[ ! -f "$RUNTIME_HELPER" || ! -f "$PATH_HELPER" ]]; then
    echo "ERROR: Standalone runtime helpers are missing under $BASE_DIR/SOPPO" >&2
    return 1 2>/dev/null || exit 1
fi
source "$PATH_HELPER"
source "$RUNTIME_HELPER"
soppo_activate_env "$ENV_ROOT/youc" || return 1 2>/dev/null || exit 1
soppo_set_cache_env
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
echo "Environment activated: $ENV_ROOT/youc"
echo "Python: $(command -v python)"
ACTIVATE_EOF
chmod +x "$SERVER_BASE/activate_env.sh"

echo "Standalone environment ready: $ENV_DIR"
echo "Activation: source $SERVER_BASE/activate_env.sh"
