# exp-20260901-01-round4-server-prep：Round4服务器准备与smoke门禁

## 元信息

- Experiment ID：`exp-20260901-01-round4-server-prep`
- Cycle ID：`cycle-20260901-01`
- 状态：运行中
- 对应实验设计：`../../human_read/exp/current_experiment.md` `round4-exp-v1.0`
- 对应理论：`../../human_read/theory/current_theory.md` `r4-theory-v1.0`
- 理论批准版本与日期：`r4-theory-v1.0`，2026-09-01用户明确通过
- 实验设计批准版本与日期：`round4-exp-v1.0`，2026-09-01用户明确通过
- 代码版本/静态校验值：`round4-code-v1.0.3`，exact code commit `2854c10b6da56f650c91f45d9a685b98cdc02375`
- 代码交接状态与日期：已确认，2026-09-01
- 服务器执行授权与日期：已明确授权，2026-09-01
- 执行位置：4090-3联网准备面与2×A100容器执行面；本地未运行项目任务
- 服务器运行手册：`../../../machine/CURRENT_STATE.md`、`../../../machine/4090-3/README.md`、`../../../machine/A800-8/README.md`

## 要回答的问题

- 目标：建立三方法共用的不可变代码、离线依赖、模型、数据和2×A100运行基线，并完成formal前的全链smoke。
- 当前步骤不产生论文指标；成功标准是环境与manifest闭环、测试/smoke合同全部通过。
- formal训练未授权，不能由本记录的准备成功自动启动。

## 已完成的服务器事实

- 4090与A100 checkout：exact clean code commit `2854c10b6da56f650c91f45d9a685b98cdc02375`。
- 离线依赖：CPython3.12/Linux x86_64，torch `2.5.1+cu124`，137个wheel，package约3.2GiB；`SHA256SUMS`在4090和A100分别复核通过。首次A100安装暴露AlpacaEval 0.6.2依赖已移除`pkg_resources`的新版setuptools；正式脚本固定并强制检查`setuptools==78.1.0`后重建成功。
- 模型：`Qwen/Qwen3-1.7B@b9352fbb8ce704292730cf54b3b1dceb2a808738`。
- 数据：`HuggingFaceH4/ultrafeedback_binarized@3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`；`HuggingFaceH4/ultrachat_200k@8049631c405ae6576f93f445c6b8166f76f5505a`。
- 资产总计：3份资产、30个payload文件、5.917GiB；逐文件SHA、manifest SHA和总索引SHA均独立复核通过。
- 复用旧缓存时，只有与上述官方revision的LFS SHA256逐文件一致的内容才被硬链接；不一致内容未复用。
- A100容器：2×NVIDIA A100-PCIE-40GB；Python3.12.3、CUDA12.4工具链可用。镜像默认shell未设置Python/CUDA/driver库路径，显式补齐`PATH`和`LD_LIBRARY_PATH`后两卡查询通过。
- A100存储：数据盘与文件存储未挂载；用户明确选择直接使用系统盘`/root`作为base，不创建`/root/round4`。代码、环境、缓存、数据、模型、运行、导出和平台日志均在`/root`下保持平级，Git元数据仅位于`/root/SOPPO/.git/`。
- 4090→A100：已创建独立、受限、严格主机指纹校验的传输认证；直连`rsync`完成，约94MiB仓库、3.41GB离线包、4.08GB模型和2.27GB数据均已传入。A100逐项复核137个wheel及3份资产30个payload文件，总payload字节`6353474410`全部通过。
- A100环境：`/root/envs/round4-py312`从离线包全新创建；`pip check`、核心imports、Python3.12.3、torch2.5.1+cu124、CUDA12.4、2 GPU、AlpacaEval0.6.2及setuptools78.1.0门禁通过。

## 远程证据索引

| 产物 | 服务器位置 | 本地保留 | 状态 |
| --- | --- | --- | --- |
| wheelhouse与`SHA256SUMS` | 4090仓库外`exports/round4-offline-package/<exact_commit>/` | 否 | 已验证 |
| wheelhouse构建日志 | 4090仓库外`platform_logs/round4/<exact_commit>/wheelhouse-build.log` | 否 | 已完成 |
| 资产总索引与SHA | 4090仓库外`exports/round4-assets/<exact_commit>/` | 仅本记录中的聚合值 | 已验证 |
| 模型与manifest | 4090仓库外`models/Qwen3-1.7B-b9352fbb/` | 否 | 已验证 |
| 两数据集与manifest | 4090仓库外`data/round4-v1/raw/` | 否 | 已验证 |
| 资产下载日志 | 4090仓库外`platform_logs/round4/<exact_commit>/asset-stage.log` | 否 | 已完成 |
| 传输与A100逐文件复核日志 | 双端仓库外`platform_logs/round4/<exact_commit>/` | 否 | 已完成 |
| A100离线环境与freeze | A100仓库外`envs/round4-py312/` | 否 | 已验证 |
| A100环境安装日志 | A100仓库外`platform_logs/round4/<exact_commit>/env-install.log` | 否 | 已完成 |

## 当前未完成与阻塞

- A100 CPU合同测试、2-step smoke、eval、merge/reload、Alpaca小样本和judge smoke尚未开始。
- A100数据盘/文件存储仍未挂载；本轮已按用户选择使用系统盘，容器重置/删除前必须把允许回传的摘要与远程证据索引交接完毕。

## 本地回传边界

- 已回传白名单：本README中的commit、revision、数量、体积、硬件和状态摘要。
- 服务器保留黑名单：wheel、环境、模型、数据、逐样本内容、原始日志和后续checkpoint。
- 本地仅记录远程路径模板与校验状态，不保存内部地址、认证材料或样本。
- 确认本地未运行环境、测试、数据、模型、评价或GPU任务：是。

## 结果交接

- 当前仍为`SERVER_EXECUTION`，不是结果交接。
- 只有全部smoke完成或形成完整失败摘要后，才进入`RESULT_HANDOFF`。
