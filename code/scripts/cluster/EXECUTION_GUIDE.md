# SOPPO 服务器执行指南

> **状态：LOCKED。** 本文件只定义服务器目录与命令接口，不构成上传或执行授权。只有实验设计获明确批准、代码交接完成，并且用户明确确认“当前代码版本可以提交服务器”后，才能执行下列服务器命令。

## 1. 唯一目录合同

服务器私有配置中的 `<SERVER_BASE>` 是用户项目根目录。目标结构如下：

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
├── cache/{pip,huggingface,modelscope}/
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

集群脚本通过 `server_paths.sh` 从自身位置推导路径：

```bash
SERVER_BASE=<SERVER_BASE>
ICLR_ROOT=$SERVER_BASE/ICLR
SOPPO_ROOT=$SERVER_BASE/SOPPO
CODE_ROOT=$SOPPO_ROOT/code
OBSERVE_ROOT=$CODE_ROOT/observe/LLM-output-density

ENV_ROOT=$SERVER_BASE/envs
CACHE_ROOT=$SERVER_BASE/cache
DATA_ROOT=$SERVER_BASE/data
MODEL_ROOT=$SERVER_BASE/models
RUN_ROOT=$SERVER_BASE/runs
EXPORT_ROOT=$SERVER_BASE/exports
PLATFORM_LOG_ROOT=$SERVER_BASE/platform_logs
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
cd <SERVER_BASE>
git clone <SOPPO_REMOTE> SOPPO
```

不得把克隆目标写成 `ICLR` 或 `ICLR/SOPPO`。

### 3.3 环境准备

在服务器允许安装依赖的节点执行：

```bash
cd <SERVER_BASE>/SOPPO/code/scripts/cluster
RUN_CONTEXT=cluster bash 00_server_setup.sh
```

该脚本应创建 Git 外的 `envs/`、`cache/`、`data/`、`models/`、`runs/`、`exports/` 和 `platform_logs/`，并拒绝错误的 Git 布局。

## 4. 分阶段命令接口

每个阶段完成后都应暂停，回传允许的聚合状态，并等待下一阶段确认。

```bash
source <SERVER_BASE>/activate_env.sh
cd <SERVER_BASE>/SOPPO/code/scripts/cluster
export RUN_CONTEXT=cluster

bash 01_server_tests.sh
bash 02_prepare_data.sh
bash 03_preexperiment.sh
bash 04_lambda_search.sh
bash 05_run_main.sh
bash 06_c_epsilon.sh
bash 07_evaluate.sh
bash 08_aggregate.sh
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
test -d <SERVER_BASE>/SOPPO/.git
test ! -e <SERVER_BASE>/ICLR/.git
test ! -e <SERVER_BASE>/SOPPO/code/observe/LLM-output-density/.git
bash -n <SERVER_BASE>/SOPPO/code/scripts/cluster/*.sh
```

上述检查只验证目录与 shell 语法；项目测试、环境检查、数据处理和模型操作仍必须作为已授权服务器任务执行。
