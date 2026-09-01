# Round4 Python 3.12 / CUDA 12.4离线环境

本目录中的脚本都是服务器任务，代码交接和服务器执行授权前不得运行。

## 环境结论

- FusionOne复用既有`cuda12.4-cudnn-devel-ubuntu22.04-py312-ssh`镜像。
- Round4运行时固定为CPython 3.12.x、PyTorch `2.5.1+cu124`。
- `SSPO/pyproject.toml`的`requires-python = ">=3.10"`是包的最低版本声明，包含Python 3.12，不要求运行时退回3.10。
- `SSPO/requirements.txt`的固定版本已做Python 3.12静态PyPI元数据检查；116个直接pin均未声明排斥Python 3.12，160条直接pin之间的适用依赖约束未发现冲突。该结论不是resolver/import/GPU测试，后者必须在服务器smoke完成。
- `fire==0.7.0`和`jieba==0.42.1`只有source distribution，因此在4090-3的Python 3.12 builder中提前构成wheel；A100只执行离线wheel安装。
- 本地`llamafactory` wheel从exact commit的临时Git archive构建，不在checkout内留下`build/`或`*.egg-info`；构建结束再次硬检查仓库clean。

静态审计日期为2026-09-01：逐个读取PyPI官方JSON元数据中的`requires_python`、distribution filenames和`requires_dist`，按CPython3.12/Linux x86_64环境marker检查直接pin。它没有下载或导入项目依赖，也不能替代服务器端完整resolver。

## 服务器执行顺序

1. 在clean exact-commit 4090-3 checkout上运行`00_build_offline_wheelhouse.sh`。
2. 运行`02_stage_hf_assets_4090.sh`。它从wheelhouse创建一个很小的Python 3.12下载环境，冻结并下载指定Qwen模型以及UltraFeedback/UltraChat数据仓库，逐文件生成SHA-256 manifest；该下载环境不是训练环境。
3. 将整个带`SHA256SUMS`的package目录、模型目录和两个数据目录续传到A100仓库外目录。
4. 在A100目标端先校验SHA，再运行`01_install_a100_env.sh`创建全新venv。
5. 安装脚本会检查核心imports、`pip check`、torch CUDA版本和可见GPU数；默认要求2张GPU。

示例只使用占位符：

```bash
export ROUND4_SERVER_BASE='<SERVER_BASE>'
export ROUND4_REPO_ROOT='<SERVER_BASE>/SOPPO'
export ROUND4_BUILDER_PYTHON='<SERVER_BASE>/envs/round4-wheel-builder-py312/bin/python'
bash '<SERVER_BASE>/SOPPO/code/scripts/round4/00_build_offline_wheelhouse.sh'
```

```bash
export ROUND4_SERVER_BASE='<SERVER_BASE>'
export ROUND4_REPO_ROOT='<SERVER_BASE>/SOPPO'
export ROUND4_BUILDER_PYTHON='<SERVER_BASE>/envs/round4-wheel-builder-py312/bin/python'
export ROUND4_PACKAGE_ROOT='<SERVER_BASE>/exports/round4-offline-package/<EXACT_COMMIT>'
bash '<SERVER_BASE>/SOPPO/code/scripts/round4/02_stage_hf_assets_4090.sh'
```

```bash
export ROUND4_PACKAGE_ROOT='<A100_BASE>/exports/round4-offline-package/<EXACT_COMMIT>'
export ROUND4_A100_BASE='<A100_BASE>'
export ROUND4_ENV_ROOT='<A100_BASE>/envs/round4-py312-cu124'
export ROUND4_PYTHON_BIN='/usr/bin/python3.12'
export ROUND4_EXPECTED_GPUS=2
bash '<A100_BASE>/SOPPO/code/scripts/round4/01_install_a100_env.sh'
```

脚本拒绝覆盖已有A100 venv；若安装中断，应先审计并明确删除该精确环境目录，再重新执行，不能使用宽泛递归清理命令。4090资产下载支持对来源一致的partial目录续传；已有完整资产每次都会重新核对逐文件SHA，来源不一致或存在未登记文件时会fail closed。
