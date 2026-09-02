# exp-20260901-01-round4-server-prep：Round4服务器准备与smoke门禁

## 元信息

- Experiment ID：`exp-20260901-01-round4-server-prep`
- Cycle ID：`cycle-20260901-01`
- 状态：`af6dac4`服务器执行在预处理门禁失败后终止；已返回`CODE_IMPLEMENTATION`修复
- 对应实验设计：`../../human_read/exp/current_experiment.md` `round4-exp-v1.0`
- 对应理论：`../../human_read/theory/current_theory.md` `r4-theory-v1.0`
- 理论批准版本与日期：`r4-theory-v1.0`，2026-09-01用户明确通过
- 实验设计批准版本与日期：`round4-exp-v1.0`，2026-09-01用户明确通过
- 已执行代码版本：`round4-code-v1.1.0`，exact code commit `af6dac49044978d76aeca4d5fcb0d11856a1c104`
- 当前修复候选：`round4-code-v1.1.1`，exact code commit `6b010b89d1c62aaa8a42af65d06b53d301b1aee8`
- 代码交接状态与日期：`af6dac4`曾于2026-09-02获准执行；`6b010b8`为失败后纯实现修复，当前**PENDING USER CONFIRMATION**
- 服务器执行授权与日期：新commit尚未获重新上传/执行确认；smoke与formal均重新锁定
- 执行位置：4090-3联网准备面与2×A100容器执行面；本地未运行项目任务
- 服务器运行手册：`../../../machine/CURRENT_STATE.md`、`../../../machine/4090-3/README.md`、`../../../machine/A800-8/README.md`

## 要回答的问题

- 目标：建立三方法共用的不可变代码、离线依赖、模型、数据和2×A100运行基线，并完成formal前的全链smoke。
- 当前步骤不产生论文指标；成功标准是环境与manifest闭环、测试/smoke合同全部通过。
- formal训练已获明确授权，但不能由环境准备成功自动启动；必须先通过全部smoke门禁，并按DPO→SSPO→StaticPE顺序执行。

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

以上已完成的环境事实只绑定旧exact commit `2854c10b…`。2026-09-02审计发现缺口后形成`round4-code-v1.1.0` / `af6dac4`，用户已明确批准提交并执行该exact版本的smoke；新wheelhouse、资产、环境与smoke结果须独立记录，不能把旧门禁外推为新版本已验证。

## `af6dac4`新增执行事实与终止点

- 4090-3 clean checkout绑定`af6dac49044978d76aeca4d5fcb0d11856a1c104`；复用136个不变依赖wheel，只重建项目wheel，最终137个wheel与package清单全量SHA通过。
- 新增冻结`alpaca_eval@2edc6fad8be6b14ea7230aabfd08188da6b8b814`；连同既有模型、UltraFeedback和UltraChat，A100端共4项资产、42个payload文件、`6429762565`字节逐文件SHA通过。
- A100通过离线Git bundle导入并clean detached checkout到`af6dac4`。新环境`/root/envs/round4-py312-af6dac490449`绑定该完整commit，并通过Python3.12.3、torch2.5.1+cu124、CUDA12.4、2×A100、核心imports与`pip check`。
- 旧`/root/envs/round4-py312`保留但没有`ROUND4_CODE_COMMIT`标记，本轮不会使用；两个环境均约6.6GiB，当前磁盘余量充足，未执行删除。
- 全量预处理成功生成3个JSON和`round4-preprocessing-v1` manifest，随后严格校验失败并停止。聚合审计显示：UltraFeedback训练抽样6113条中8条为空或仅有单侧回答，共同eval 2000条中3条缺少rejected；UltraChat抽样20786条中3条回答为空。StaticPE候选、DPO/SSPO/StaticPE训练、merge、生成和judge均未开始。
- 失败数据与原始日志仅留服务器；本地没有回传任何样本行。当前代码候选`6b010b8`在确定性抽样后过滤无效行，把manifest升级为`v2`并记录输入/丢弃原因/有效行数，同时修复资产索引`.sha256`携带源机绝对路径的问题。

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
| `af6dac4`复用wheelhouse构建日志 | 4090仓库外`platform_logs/round4/af6dac49044978d76aeca4d5fcb0d11856a1c104/wheelhouse-build-reused.log` | 否 | 已完成并通过 |
| 四资产索引与下载日志 | 4090仓库外`exports/round4-assets/af6dac49044978d76aeca4d5fcb0d11856a1c104/`、同commit平台日志 | 仅聚合数量/字节 | 已完成并通过 |
| A100 exact-commit环境 | A100仓库外`envs/round4-py312-af6dac490449/` | 否 | 已完成并通过 |
| 失败prepared数据 | A100仓库外`data/round4-v1/prepared/` | 否 | 保留，`v1`严格校验失败，不得用于训练 |
| 前处理与失败校验日志 | A100仓库外`platform_logs/round4/af6dac49044978d76aeca4d5fcb0d11856a1c104/preparation/{preprocessing,prepared-verify}.log` | 仅本记录聚合摘要 | 已保留 |

## 当前未完成与阻塞

- `6afebd3`于2026-09-02获用户明确批准并完成两次A100 smoke尝试。第一次在首个训练入口前因包内绝对导入无法解析停止；第二次以临时`PYTHONPATH`验证该路径后，预处理、fixture及FrozenPE候选构造均通过，但DPO在参数初始化前因`accelerate==1.0.1`不满足`transformers==4.51.3`对`data_seed`的`accelerate>=1.1.0`要求而停止。两次均未执行优化step、merge、生成或judge。
- 这是纯实现/依赖锁定缺陷，已返回`CODE_IMPLEMENTATION`形成`round4-code-v2.0.1`候选：入口改为包内相对导入，依赖固定为`accelerate==1.1.0`。它须重新获得用户对新exact commit的代码交接确认，之后独立重建wheelhouse/环境并从四臂full-chain smoke重新开始。
- A100和4090的API judge安全配置仍须在full-chain smoke的4090阶段独立核验；密钥只能由用户直接在服务器安全配置，不能写入仓库、日志或聊天。
- A100数据盘/文件存储仍未挂载；本轮已按用户选择使用系统盘，容器重置/删除前必须把允许回传的摘要与远程证据索引交接完毕。

## 本地回传边界

- 已回传白名单：本README中的commit、revision、数量、体积、硬件和状态摘要。
- 服务器保留黑名单：wheel、环境、模型、数据、逐样本内容、原始日志和后续checkpoint。
- 本地仅记录远程路径模板与校验状态，不保存内部地址、认证材料或样本。
- 确认本地未运行环境、测试、数据、模型、评价或GPU任务：是。

## 结果交接

- 当前因纯实现缺陷返回`CODE_IMPLEMENTATION`，不是结果交接。
- 新代码完成交接后才能重进服务器执行；若用户选择不修复重跑，则以本节失败事实形成终止摘要并进入`RESULT_HANDOFF`。
