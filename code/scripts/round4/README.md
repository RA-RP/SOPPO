# Round4 Python 3.12 / CUDA 12.4离线环境

本目录中的脚本都是服务器任务，代码交接和服务器执行授权前不得运行。Round4 smoke固定使用两张A100、恰好2个optimizer step，并且fail closed：API key、资产、commit、环境或任一产物不满足合同时，不启动或不继续后续步骤。

## 环境结论

- FusionOne复用既有`cuda12.4-cudnn-devel-ubuntu22.04-py312-ssh`镜像。
- Round4运行时固定为CPython 3.12.x、PyTorch `2.5.1+cu124`。
- `SSPO/pyproject.toml`的`requires-python = ">=3.10"`是包的最低版本声明，包含Python 3.12，不要求运行时退回3.10。
- `SSPO/requirements.txt`的固定版本已做Python 3.12静态PyPI元数据检查；116个直接pin均未声明排斥Python 3.12，160条直接pin之间的适用依赖约束未发现冲突。resolver、核心imports和GPU门禁已在旧exact commit环境通过；新代码commit仍须重新构建、安装并复核。
- `fire==0.7.0`和`jieba==0.42.1`只有source distribution，因此在4090-3的Python 3.12 builder中提前构成wheel；A100只执行离线wheel安装。
- 本地`llamafactory` wheel从exact commit的临时Git archive构建，不在checkout内留下`build/`或`*.egg-info`；构建结束再次硬检查仓库clean。

静态审计日期为2026-09-01：逐个读取PyPI官方JSON元数据中的`requires_python`、distribution filenames和`requires_dist`，按CPython3.12/Linux x86_64环境marker检查直接pin。它没有下载或导入项目依赖，也不能替代服务器端完整resolver。

## 服务器执行顺序

1. 在clean exact-commit 4090-3 checkout上运行`00_build_offline_wheelhouse.sh`。
2. 运行`02_stage_hf_assets_4090.sh`。它从wheelhouse创建一个很小的Python 3.12下载环境，冻结并下载指定Qwen模型、UltraFeedback、UltraChat和AlpacaEval数据仓库，逐文件生成SHA-256 manifest；该下载环境不是训练环境。三份既有训练资产使用已冻结exact revision，不重新解析`main`。
3. 将整个带`SHA256SUMS`的package目录、新代码checkout和新增/变化资产续传到A100仓库外目录；目标端重算全部SHA。
4. 在A100目标端先校验SHA，再运行`01_install_a100_env.sh`创建全新venv。
5. 安装脚本会检查核心imports、`pip check`、torch CUDA版本和可见GPU数；默认要求2张GPU。
6. 运行`03_run_smoke_a100.sh`。它完成冻结数据预处理、fixture、StaticPE固定候选、三方法训练/eval、adapter、merge、新进程生成、每方法一个judge样本和aggregate-only验收。

## 2-step smoke合同

| 方法 | fixture | per-device train | GPU | GA | effective batch | optimizer step |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DPO | 32条共享labeled子集 | 1 | 2 | 8 | 16 | 2 |
| SSPO | 64 labeled + 64 unlabeled | 4 | 2 | 8 | 64 | 2 |
| StaticPE | 同一64+64，unlabeled预生成固定B | 4 | 2 | 8 | 64 | 2 |

- 三方法`per_device_eval_batch_size=4`、`eval_steps=1`，在step 1和2各触发一次eval；共同eval view为8条labeled rows。
- smoke对SSPO/StaticPE把two-stream最小值临时固定为每个per-device physical batch `2 labeled + 2 unlabeled`，只为确定性覆盖两个分支，不修改formal配置中的总体比例口径。
- StaticPE候选生成若出现空回答或与原回答完全重复，整个smoke失败，不静默缩小population。
- merge后由新Python进程从merged目录重载；每方法在冻结AlpacaEval顺序的前2条生成，judge只取1条，因此每方法恰好一次judge请求。该结果只验证链路，不是论文指标。
- `alpaca-eval==0.6.2`内部仍会通过`hf_hub_download`读取LC辅助CSV。`03_materialize_hf_cache.py`把已冻结本地AlpacaEval资产用硬链接映射到离线Hub cache，并显式向CLI传入本地GPT-4 baseline；A100无需访问Hugging Face网络。
- `03_validate_smoke.py`要求每方法`global_step==2`、至少两次eval、分项loss、adapter、merged safetensors、固定数量输出、judge annotation和leaderboard全部存在；只写无样本级`SMOKE_SUMMARY.json`。

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
export ROUND4_EXPECTED_COMMIT='<EXACT_COMMIT>'
bash '<A100_BASE>/SOPPO/code/scripts/round4/01_install_a100_env.sh'
```

完整smoke入口（只允许在A100服务器执行）：

```bash
export ROUND4_A100_BASE='<A100_BASE>'
export ROUND4_REPO_ROOT='<A100_BASE>/SOPPO'
export ROUND4_ENV_ROOT='<A100_BASE>/envs/round4-py312-<SHORT_COMMIT>'
export ROUND4_EXPECTED_COMMIT='<EXACT_COMMIT>'
export ROUND4_MODEL_PATH='<A100_BASE>/models/Qwen3-1.7B-b9352fbb'
export ROUND4_SMOKE_RUN_ID='<UNIQUE_RUN_ID>'
export OPENAI_API_KEY='<FROM_SERVER_SECRET_STORE>'
bash '<A100_BASE>/SOPPO/code/scripts/round4/03_run_smoke_a100.sh'
```

脚本不会打印API key；若key不存在，会在数据处理和训练之前终止。`ROUND4_RUN_JUDGE`必须为`1`，不能把跳过judge的部分链路冒充完整smoke。

脚本拒绝覆盖已有A100 venv；环境会写入`ROUND4_CODE_COMMIT`并核对离线包`BUILD_INFO`，smoke拒绝环境、checkout和期望commit三者不一致。若安装中断，应先审计并明确删除该精确环境目录，再重新执行，不能使用宽泛递归清理命令。4090资产下载支持对来源一致的partial目录续传；已有完整资产每次都会重新核对逐文件SHA，来源不一致或存在未登记文件时会fail closed。
