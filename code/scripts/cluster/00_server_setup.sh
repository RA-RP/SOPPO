#!/bin/bash
# Stage -1: Environment Setup
# 在编译节点（gn001）运行；脚本会加载服务器的 Python 3.10 module
# Purpose: Create directory structure, install dependencies

set -euo pipefail

if [[ "${RUN_CONTEXT:-}" != "cluster" ]]; then
    echo "ERROR: This script must run on the server (RUN_CONTEXT=cluster)"
    exit 1
fi

echo "=== Stage -1: Environment Setup ==="
echo "Cycle: cycle-20260818-01"
echo "Experiment: v0.3 MVP"
echo "Date: $(date)"

# ===================================================
# 基础路径配置（自动检测）
# ===================================================

# 从脚本位置自动检测路径。
# 服务器目标: <SERVER_BASE>/SOPPO/code/scripts/cluster/00_server_setup.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/server_paths.sh"
CODE_DIR="$CODE_ROOT"
SOPPO_DIR="$SOPPO_ROOT"
ICLR_DIR="$ICLR_ROOT"
BASE_DIR="$SERVER_BASE"

echo ""
echo "Auto-detected paths:"
echo "  Script: $SCRIPT_DIR"
echo "  Code: $CODE_DIR"
echo "  SOPPO (only Git repo): $SOPPO_DIR"
echo "  ICLR (static, no Git): $ICLR_DIR"
echo "  Base: $BASE_DIR"

# 验证目录结构
if [[ ! -f "$CODE_DIR/requirements.lock.txt" ]]; then
    echo ""
    echo "ERROR: requirements.lock.txt not found at $CODE_DIR"
    echo "Please ensure you are in the correct directory"
    exit 1
fi

if [[ ! -d "$SOPPO_DIR/.git" ]]; then
    echo "ERROR: SOPPO must be the Git repository root: $SOPPO_DIR/.git"
    exit 1
fi

if [[ -e "$ICLR_DIR/.git" ]]; then
    echo "ERROR: ICLR must be a static non-Git directory: $ICLR_DIR/.git"
    exit 1
fi

if find "$OBSERVE_ROOT" -type d -name .git -print -quit | grep -q .; then
    echo "ERROR: Nested Git repository found under $OBSERVE_ROOT"
    exit 1
fi

echo "✓ Directory structure verified"

# ===================================================
# Python 运行时预检（任何安装动作之前）
# ===================================================

BOOTSTRAP_PYTHON=""
for candidate in python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 \
        && "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)' 2>/dev/null; then
        BOOTSTRAP_PYTHON="$(command -v "$candidate")"
        break
    fi
done

if [[ -z "$BOOTSTRAP_PYTHON" && -f /home-ssd/Soft/modules/bashrc ]]; then
    source /home-ssd/Soft/modules/bashrc
    if type module >/dev/null 2>&1; then
        echo "Loading server Python module: python/3.10.4"
        module load python/3.10.4
    fi

    for candidate in python3.10 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 \
            && "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)' 2>/dev/null; then
            BOOTSTRAP_PYTHON="$(command -v "$candidate")"
            break
        fi
    done
fi

if [[ -z "$BOOTSTRAP_PYTHON" ]]; then
    echo ""
    echo "ERROR: Python 3.10 is required to create $ENV_ROOT/youc."
    echo "The current shell does not provide a compatible Python runtime."
    if command -v python3 >/dev/null 2>&1; then
        echo "Detected python3: $(command -v python3) ($(python3 --version 2>&1))"
    fi
    if command -v python >/dev/null 2>&1; then
        echo "Detected python: $(command -v python) ($(python --version 2>&1))"
    fi
    echo "Load the server module with: module load python/3.10.4"
    echo "Then rerun this script; do not use the system Python 3.6.8 runtime."
    exit 2
fi

echo "Python bootstrap: $BOOTSTRAP_PYTHON ($($BOOTSTRAP_PYTHON --version 2>&1))"

# ===================================================
# 创建目录结构
# ===================================================

echo ""
echo "Creating directory structure..."

mkdir -p "$ENV_ROOT"
mkdir -p "$DATA_ROOT"
mkdir -p "$MODEL_ROOT"
mkdir -p "$RUN_ROOT"
mkdir -p "$EXPORT_ROOT"
mkdir -p "$PLATFORM_LOG_ROOT"
mkdir -p "$CACHE_ROOT/pip"
mkdir -p "$CACHE_ROOT/huggingface"
mkdir -p "$CACHE_ROOT/modelscope"

echo "✓ Directory structure created"

# ===================================================
# 创建 Python 虚拟环境
# ===================================================

echo ""
echo "=== Creating Python Virtual Environment: $ENV_ROOT/youc ==="

ENV_DIR="$ENV_ROOT/youc"

# 拒绝覆盖无法识别的已有目录；正常重跑则复用已创建的 venv。
if [[ -e "$ENV_DIR" && ! -x "$ENV_DIR/bin/python" ]]; then
    echo "ERROR: Existing environment directory is incomplete or incompatible: $ENV_DIR"
    echo "Move it aside after manual inspection, then rerun this script."
    exit 1
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo "Creating Python venv: $ENV_DIR"
    "$BOOTSTRAP_PYTHON" -m venv "$ENV_DIR"
    echo "✓ Python venv created"
else
    echo "Using existing Python venv: $ENV_DIR"
fi

source "$ENV_DIR/bin/activate"

echo "Environment activated: $ENV_DIR"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# ===================================================
# 设置缓存路径
# ===================================================

echo ""
echo "Setting cache directories..."

export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export MODELSCOPE_CACHE="$CACHE_ROOT/modelscope"

echo "✓ Cache paths configured"

# ===================================================
# 加载网络代理（必须）
# ===================================================

echo ""
echo "Loading network proxy..."

source /home-ssd/Soft/modules/bashrc
module load proxy/proxy

echo "✓ Network proxy loaded"

# ===================================================
# 安装依赖
# ===================================================

echo ""
echo "=== Installing Dependencies ==="

# 升级 pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# 安装 PyTorch
echo "Installing PyTorch 2.4.0 with CUDA 12.1..."
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
if [[ -f "$CODE_DIR/requirements.lock.txt" ]]; then
    echo "Installing packages from requirements.lock.txt..."
    python -m pip install -r "$CODE_DIR/requirements.lock.txt"
else
    echo "Installing core packages..."
    python -m pip install 'transformers>=4.40.0'
    python -m pip install 'datasets>=2.18.0'
    python -m pip install 'accelerate>=0.28.0'
    python -m pip install numpy scipy tqdm
    python -m pip install jsonlines pyyaml
fi

echo "✓ Dependencies installed"

# ===================================================
# 验证安装
# ===================================================

echo ""
echo "=== Verifying Installations ==="

python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'✓ Datasets: {datasets.__version__}')"
python -c "import accelerate; print(f'✓ Accelerate: {accelerate.__version__}')"

echo ""
echo "Checking CUDA..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# ===================================================
# 保存配置
# ===================================================

echo ""
echo "Saving configuration..."

PROJECT_CONFIG="$BASE_DIR/project_config.json"

cat > "$PROJECT_CONFIG" <<EOF
{
  "cycle_id": "cycle-20260818-01",
  "base_dir": "$BASE_DIR",
  "iclr_dir": "$ICLR_DIR",
  "soppo_dir": "$SOPPO_DIR",
  "code_dir": "$CODE_DIR",
  "env_dir": "$ENV_DIR",
  "data_dir": "$DATA_ROOT",
  "models_dir": "$MODEL_ROOT",
  "runs_dir": "$RUN_ROOT",
  "platform_logs_dir": "$PLATFORM_LOG_ROOT",
  "export_dir": "$EXPORT_ROOT",
  "cache_dir": "$CACHE_ROOT",
  "created_at": "$(date -Iseconds)"
}
EOF

echo "✓ Configuration saved to: $PROJECT_CONFIG"

# 保存环境激活脚本
ACTIVATE_SCRIPT="$BASE_DIR/activate_env.sh"

cat > "$ACTIVATE_SCRIPT" <<'ACTIVATE_EOF'
#!/bin/bash
# Activate youc environment with all settings

# 自动检测 BASE_DIR（从脚本位置）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"

# Restore the Python runtime used to create the venv.
if [[ -f /home-ssd/Soft/modules/bashrc ]]; then
    source /home-ssd/Soft/modules/bashrc
    if type module >/dev/null 2>&1; then
        module load python/3.10.4 >/dev/null 2>&1 || {
            echo "ERROR: Failed to load python/3.10.4 module" >&2
            return 1 2>/dev/null || exit 1
        }
    fi
fi

# Activate the path-based Python venv
ENV_DIR="$BASE_DIR/envs/youc"
if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
    echo "ERROR: Python environment not found: $ENV_DIR" >&2
    return 1 2>/dev/null || exit 1
fi
source "$ENV_DIR/bin/activate"

# Set cache paths
export PIP_CACHE_DIR="$BASE_DIR/cache/pip"
export HF_HOME="$BASE_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$BASE_DIR/cache/huggingface/transformers"
export MODELSCOPE_CACHE="$BASE_DIR/cache/modelscope"

# Set PYTHONPATH
export PYTHONPATH="$BASE_DIR/SOPPO/code:${PYTHONPATH:-}"

# Load proxy (if on compile node)
if command -v module &> /dev/null; then
    source /home-ssd/Soft/modules/bashrc 2>/dev/null || true
    module load proxy/proxy 2>/dev/null || true
fi

echo "Environment activated: $BASE_DIR/envs/youc"
echo "Python: $(which python)"
ACTIVATE_EOF

chmod +x "$ACTIVATE_SCRIPT"

echo "✓ Activation script created: $ACTIVATE_SCRIPT"

# ===================================================
# 完成
# ===================================================

echo ""
echo "=== Stage -1 Complete ==="
echo ""
echo "Summary:"
echo "  Environment: $ENV_DIR"
echo "  Base path: $BASE_DIR"
echo "  ICLR (static, no Git): $ICLR_DIR"
echo "  SOPPO (only Git repo): $SOPPO_DIR"
echo "  Code: $CODE_DIR"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "To activate environment in future sessions:"
echo "  source $ACTIVATE_SCRIPT"
echo ""
echo "Next: Run 01_server_tests.sh (on compile node)"
