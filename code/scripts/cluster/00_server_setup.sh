#!/bin/bash
# Stage -1: Environment Setup
# 在编译节点上运行（gn001）
# Purpose: Create directory structure, install dependencies

set -euo pipefail

echo "=== Stage -1: Environment Setup ==="
echo "Cycle: cycle-20260818-01"
echo "Experiment: v0.3 MVP"
echo "Date: $(date)"

# ===================================================
# 基础路径配置
# ===================================================

BASE_DIR="/home-ssd/Users/nsgm_jiangwh/youchang"
ICLR_DIR="$BASE_DIR/ICLR"
CODE_DIR="$ICLR_DIR/work/code"

echo "Base directory: $BASE_DIR"
echo "ICLR directory: $ICLR_DIR"
echo "Code directory: $CODE_DIR"

# 验证 ICLR 目录存在
if [[ ! -d "$ICLR_DIR" ]]; then
    echo "ERROR: ICLR directory not found at $ICLR_DIR"
    echo "Please upload ICLR folder first"
    exit 1
fi

# ===================================================
# 创建目录结构
# ===================================================

echo ""
echo "Creating directory structure..."

mkdir -p "$BASE_DIR/envs"
mkdir -p "$BASE_DIR/data"
mkdir -p "$BASE_DIR/models"
mkdir -p "$BASE_DIR/exp"
mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/export_local"
mkdir -p "$BASE_DIR/cache/pip"
mkdir -p "$BASE_DIR/cache/huggingface"
mkdir -p "$BASE_DIR/cache/modelscope"

echo "✓ Directory structure created"

# ===================================================
# 创建 Conda 环境
# ===================================================

echo ""
echo "=== Creating Conda Environment: youc ==="

ENV_NAME="youc"

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists"
    read -p "Remove and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n "$ENV_NAME" -y
        echo "Removed existing environment"
    else
        echo "Using existing environment"
    fi
fi

# 创建环境
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.10 -y
    echo "✓ Conda environment created"
fi

# 激活环境
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Environment activated: $ENV_NAME"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# ===================================================
# 设置缓存路径
# ===================================================

echo ""
echo "Setting cache directories..."

export PIP_CACHE_DIR="$BASE_DIR/cache/pip"
export HF_HOME="$BASE_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$BASE_DIR/cache/huggingface/transformers"
export MODELSCOPE_CACHE="$BASE_DIR/cache/modelscope"

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
pip install --upgrade pip

# 安装 PyTorch
echo "Installing PyTorch 2.4.0 with CUDA 12.1..."
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
if [[ -f "$CODE_DIR/requirements.lock.txt" ]]; then
    echo "Installing packages from requirements.lock.txt..."
    pip install -r "$CODE_DIR/requirements.lock.txt"
else
    echo "Installing core packages..."
    pip install transformers>=4.40.0
    pip install datasets>=2.18.0
    pip install accelerate>=0.28.0
    pip install numpy scipy tqdm
    pip install jsonlines pyyaml
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
  "code_dir": "$CODE_DIR",
  "env_name": "$ENV_NAME",
  "data_dir": "$BASE_DIR/data",
  "models_dir": "$BASE_DIR/models",
  "exp_dir": "$BASE_DIR/exp",
  "logs_dir": "$BASE_DIR/logs",
  "export_dir": "$BASE_DIR/export_local",
  "cache_dir": "$BASE_DIR/cache",
  "created_at": "$(date -Iseconds)"
}
EOF

echo "✓ Configuration saved to: $PROJECT_CONFIG"

# 保存环境激活脚本
ACTIVATE_SCRIPT="$BASE_DIR/activate_env.sh"

cat > "$ACTIVATE_SCRIPT" <<'EOF'
#!/bin/bash
# Activate youc environment with all settings

BASE_DIR="/home-ssd/Users/nsgm_jiangwh/youchang"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate youc

# Set cache paths
export PIP_CACHE_DIR="$BASE_DIR/cache/pip"
export HF_HOME="$BASE_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$BASE_DIR/cache/huggingface/transformers"
export MODELSCOPE_CACHE="$BASE_DIR/cache/modelscope"

# Set PYTHONPATH
export PYTHONPATH="$BASE_DIR/ICLR/work/code:$PYTHONPATH"

# Load proxy (if on compile node)
if command -v module &> /dev/null; then
    source /home-ssd/Soft/modules/bashrc 2>/dev/null || true
    module load proxy/proxy 2>/dev/null || true
fi

echo "Environment activated: youc"
echo "Python: $(which python)"
EOF

chmod +x "$ACTIVATE_SCRIPT"

echo "✓ Activation script created: $ACTIVATE_SCRIPT"

# ===================================================
# 完成
# ===================================================

echo ""
echo "=== Stage -1 Complete ==="
echo ""
echo "Summary:"
echo "  Environment: youc"
echo "  Base path: $BASE_DIR"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "To activate environment in future sessions:"
echo "  source $ACTIVATE_SCRIPT"
echo ""
echo "Next: Run 01_server_tests.sh (on compile node)"
