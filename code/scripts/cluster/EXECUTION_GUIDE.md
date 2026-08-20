# SOPPO 服务器执行指南

> **状态：LOCKED。** 本文件只定义服务器目录与命令接口，不构成上传或执行授权。只有实验设计获明确批准、代码交接完成，并且用户明确确认“当前代码版本可以提交服务器”后，才能执行下列服务器命令。

## 0. 先说明命令中的三个容易混淆之处

本服务器当前记录的实际根目录是：

```text
/home-ssd/Users/nsgm_jiangwh/youchang
```

文档中的 `<SERVER_BASE>` 只是这个绝对路径的占位符，**不能把尖括号原样输入 shell**。为了减少重复输入，登录服务器后先定义一次：

```bash
export SERVER_BASE=/home-ssd/Users/nsgm_jiangwh/youchang
```

原命令中的三行分别表示：

```bash
source "$SERVER_BASE/activate_env.sh"                 # 激活 Python 环境并设置缓存、PYTHONPATH
cd "$SERVER_BASE/SOPPO/code/scripts/cluster"         # 进入八个阶段脚本所在目录
export RUN_CONTEXT=cluster                            # 声明当前是服务器会话；脚本用它阻止本地误执行
```

`activate_env.sh` **不在 Git 仓库中**。它由服务器首次成功运行 `00_server_setup.sh` 后生成，实际位置是：

```text
/home-ssd/Users/nsgm_jiangwh/youchang/activate_env.sh
```

因此，尚未运行完 `00_server_setup.sh` 时看不到它是正常的。可在服务器检查：

```bash
export SERVER_BASE=/home-ssd/Users/nsgm_jiangwh/youchang
ls -l "$SERVER_BASE/activate_env.sh"
```

若返回 `No such file or directory`，不要继续执行 `01`—`08`；先按第 3.3 节完成一次性环境准备。

## 1. 唯一目录合同

`<SERVER_BASE>` 表示 `/home-ssd/Users/nsgm_jiangwh/youchang`。目标结构如下：

```text
<SERVER_BASE>/
├── ICLR/                              # 静态资料镜像，无 .git
├── SOPPO/                             # 唯一 Git 仓库
│   ├── .git/
│   ├── code/
│   │   ├── src/
│   │   ├── configs/
│   │   ├── scripts/cluster/
│   │   └── observe/LLM-output-density/  # 普通目录，无嵌套 .git
│   ├── human_read/
│   └── exp/                       # 仅小型摘要和远程索引
├── envs/youc/                         # Git 外
├── cache/{pip,conda/pkgs,huggingface,modelscope}/
├── data/
├── models/
├── runs/<experiment_id>/             # 重量级实验产物
├── exports/<experiment_id>/          # 可回传白名单
├── platform_logs/
├── project_config.json
└── activate_env.sh
```

三个 Git 不变量：

```text
必须存在：<SERVER_BASE>/SOPPO/.git
必须不存在：<SERVER_BASE>/ICLR/.git
必须不存在：<SERVER_BASE>/SOPPO/code/observe/LLM-output-density/.git
```

`ICLR/` 与 `SOPPO/` 必须平级；服务器上不存在 `<SERVER_BASE>/ICLR/SOPPO/`。

## 2. 路径变量

集群脚本通过 `server_paths.sh` 从自身位置自动推导路径，用户通常只需设置 `SERVER_BASE` 方便导航，无需逐个设置以下变量：

```bash
export SERVER_BASE=/home-ssd/Users/nsgm_jiangwh/youchang
ICLR_ROOT="$SERVER_BASE/ICLR"
SOPPO_ROOT="$SERVER_BASE/SOPPO"
CODE_ROOT="$SOPPO_ROOT/code"
OBSERVE_ROOT="$CODE_ROOT/observe/LLM-output-density"

ENV_ROOT="$SERVER_BASE/envs"
CACHE_ROOT="$SERVER_BASE/cache"
DATA_ROOT="$SERVER_BASE/data"
MODEL_ROOT="$SERVER_BASE/models"
RUN_ROOT="$SERVER_BASE/runs"
EXPORT_ROOT="$SERVER_BASE/exports"
PLATFORM_LOG_ROOT="$SERVER_BASE/platform_logs"
```

除非运行手册明确记录覆盖值，其他脚本不得自行硬编码 `/nfs4/ICLR`、本地 `/Users/...` 或旧的 `ICLR/work` 路径。

## 3. 服务器准备流程

以下命令只能在 `SERVER_EXECUTION` 获授权后运行。

### 3.1 准备静态 ICLR 镜像

通过获准的文件传输方式把本地静态资料同步到 `<SERVER_BASE>/ICLR/`，同步时排除：

```text
.git/
SOPPO/
.DS_Store
```

`ICLR/` 不使用 `git clone`，也不得在服务器上执行 `git init`。

### 3.2 克隆唯一仓库

```bash
export SERVER_BASE=/home-ssd/Users/nsgm_jiangwh/youchang
cd "$SERVER_BASE"
git clone --branch master https://github.com/RA-RP/SOPPO.git SOPPO
```

不得把克隆目标写成 `ICLR` 或 `ICLR/SOPPO`。

### 3.3 环境准备

`gn001` 的系统默认 Python 3.6.8 不能用于本项目，`python/3.10.4` module 又因缺少 `libffi` 依赖而不可用。服务器已确认可正常加载 `miniforge3/25.11.0-0`；更新到包含本修正的代码版本后，在 `gn001` 执行：

```bash
source /home-ssd/Soft/modules/bashrc
module load miniforge3/25.11.0-0
conda --version

export SERVER_BASE=/home-ssd/Users/nsgm_jiangwh/youchang
cd "$SERVER_BASE/SOPPO/code/scripts/cluster"
export RUN_CONTEXT=cluster
bash 00_server_setup.sh
```

`conda --version` 应显示 `conda 25.11.0`。Miniforge 自身显示 Python 3.12.12 是正常的；`00_server_setup.sh` 会用它在独立路径环境中安装 Python 3.10。修正后的脚本也会在 Conda 未加载时自动加载该 Miniforge module。

该脚本创建 Git 外的 `cache/`、`data/`、`models/`、`runs/`、`exports/` 和 `platform_logs/`，创建固定 Python 3.10 的 Conda 路径环境 `$SERVER_BASE/envs/youc`，并拒绝错误的 Git 布局。它还会在最后创建 `$SERVER_BASE/activate_env.sh`；`00`—`08` 均通过仓库内的 `runtime_env.sh` 使用同一套 Miniforge 激活逻辑。

完成后立即确认：

```bash
ls -l "$SERVER_BASE/activate_env.sh"
"$SERVER_BASE/envs/youc/bin/python" --version
```

如果脚本仍尝试加载 `python/3.10.4`，说明服务器上的 `SOPPO` 仍是旧代码；先停止，不要继续 `01`—`08`，并将服务器仓库更新到包含本指南、新版 `00_server_setup.sh` 和 `runtime_env.sh` 的提交。

若 PyTorch 的 NVIDIA CUDA wheel 下载出现 `ReadTimeoutError`，不需要删除 `$SERVER_BASE/envs/youc`。脚本默认使用 180 秒超时和 10 次重试；更新脚本后直接重跑即可。也可以在本次服务器会话临时覆盖：

```bash
export PIP_DEFAULT_TIMEOUT=180
export PIP_RETRIES=10
bash 00_server_setup.sh
```

重跑会复用已经创建的 Conda 环境以及 `$SERVER_BASE/cache/pip` 中成功下载的文件。如果同一 NVIDIA 地址在加长超时后仍失败，应停止并改用 PyTorch 官方 Conda 渠道方案，而不是删除环境或反复从头安装。

若 `02_prepare_data.sh` 在 Hugging Face Hub 报 `ProxyError`，说明当前服务器会话没有正确加载集群代理。新版 `02` 会自行加载 `proxy/proxy` 并固定 `HF_HOME`/`HF_DATASETS_CACHE`；更新代码后可直接重跑，不需要删除已创建的数据目录。旧版脚本的本次会话临时修复是：

```bash
source /home-ssd/Soft/modules/bashrc
module load proxy/proxy
bash 02_prepare_data.sh
```

## 4. 分阶段命令接口

### 4.1 每次重新登录服务器后：只初始化一次会话

下面这段可以直接复制。它不会启动实验，只会激活已经创建的环境、进入脚本目录并设置服务器安全标志：

```bash
export SERVER_BASE=/home-ssd/Users/nsgm_jiangwh/youchang
source "$SERVER_BASE/activate_env.sh"
cd "$SERVER_BASE/SOPPO/code/scripts/cluster"
export RUN_CONTEXT=cluster

echo "SERVER_BASE=$SERVER_BASE"
echo "工作目录=$(pwd)"
echo "RUN_CONTEXT=$RUN_CONTEXT"
```

预期最后三项分别显示正确的绝对根目录、以 `/SOPPO/code/scripts/cluster` 结尾的工作目录，以及 `RUN_CONTEXT=cluster`。

### 4.2 一次只执行一个阶段

不要一次性粘贴 `01`—`08` 的全部命令。每个阶段成功后都应暂停，回传允许的聚合状态并等待下一阶段确认，再执行下一行。

| 阶段 | 作用 | 本阶段命令 |
|---|---|---|
| 01 | 服务器测试 | `bash 01_server_tests.sh` |
| 02 | 数据准备 | `bash 02_prepare_data.sh` |
| 03 | 预实验与数值稳定性检查 | `bash 03_preexperiment.sh` |
| 04 | λ 搜索 | `bash 04_lambda_search.sh` |
| 05 | MVP 主实验 | `bash 05_run_main.sh` |
| 06 | `C_ε` 观测 | `bash 06_c_epsilon.sh` |
| 07 | 测试集评价 | `bash 07_evaluate.sh` |
| 08 | 聚合结果与生成报告 | `bash 08_aggregate.sh` |

例如，首次获准执行时只运行：

```bash
bash 01_server_tests.sh
```

这些入口当前仍含待服务器验证或占位实现；目录修正不等于数值、训练或评价逻辑已经通过验证。

## 5. 产物边界

服务器保留、不得回传：

- 原始或处理后数据、私有标签；
- 模型、adapter、checkpoint、optimizer state；
- token cache、embedding、逐样本预测和可能包含样本文本的原始日志；
- 包目录、虚拟环境和依赖缓存。

允许回传到本地 `SOPPO/exp/<experiment_id>/`：

- 聚合 Markdown；
- 无样本级内容的汇总 JSON/CSV；
- 汇总图表；
- 最终配置、公开 manifest、环境摘要和校验值；
- 任务状态、失败摘要及远程路径索引。

## 6. 静态核验

服务器执行前至少核验：

```bash
export SERVER_BASE=/home-ssd/Users/nsgm_jiangwh/youchang
test -d "$SERVER_BASE/SOPPO/.git"
test ! -e "$SERVER_BASE/ICLR/.git"
test ! -e "$SERVER_BASE/SOPPO/code/observe/LLM-output-density/.git"
bash -n "$SERVER_BASE"/SOPPO/code/scripts/cluster/*.sh
```

上述检查只验证目录与 shell 语法；项目测试、环境检查、数据处理和模型操作仍必须作为已授权服务器任务执行。
