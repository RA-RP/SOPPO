# SOPPO Round4 代码状态总览

## 0. 当前状态与授权边界

- Cycle：`cycle-20260901-01` / Round4
- 当前唯一活动阶段：`SERVER_EXECUTION`
- 当前理论：`../human_read/theory/current_theory.md` `r4-theory-v1.0`，2026-09-01用户明确通过
- 当前实验：`../human_read/exp/current_experiment.md` `round4-exp-v1.0`，2026-09-01用户明确通过
- Round4代码版本：`round4-code-v1.0.3`；服务器执行累计修复runtime requirements未过滤`-e .`、本地wheel构建污染checkout及AlpacaEval 0.6.2与新版setuptools不兼容三个纯构建脚本缺陷，研究语义未改变
- 代码交接：2026-09-01用户明确要求完成4090-3任务，已批准提交当前代码并执行4090-3离线依赖、数据和模型准备

本文件承担Round4实现映射与最终代码交接。Round3旧formal已行政结项，不改写为Round4入口；4090-3旧runs/envs已按用户指令行政清空。

## 1. Round4目标通路

```text
冻结模型/数据/模板
  ├─ DPO: UltraFeedback labeled only ───────────────┐
  ├─ SSPO: labeled + UltraChat unlabeled ──────────┤
  └─ StaticPE: labeled + fixed two-candidate stream ┤
                                                     ↓
             1 epoch train → train-time eval → adapter
             → LoRA merge → offline reload/generation
             → AlpacaEval 2.0 generation/judge/LC aggregate
```

正式目标是三条独立、可复现的通路，不允许一个method ID根据缺失字段静默切换目标函数。

## 2. 已有 legacy 候选

候选代码位于仓库内 `SSPO/`：

| 能力 | 位置 | 状态 |
| --- | --- | --- |
| DPO/SSPO/StaticPE trainer | `SSPO/src_sspo/llamafactory/train/dpo/trainer.py` | StaticPE保留DPO+PE；SSPO labeled已改为未归一化DPO-base，reward norm只服务unlabeled threshold；待服务器数值/梯度验收 |
| two-stream sampler | `SSPO/src_sspo/llamafactory/data/sampler.py` | SSPO与StaticPE共用无放回比例保持sampler；分布式/累积语义待测 |
| StaticPE固定B候选生成 | `SSPO/preprocessing_data/generate_staticpe_candidates.py` | 已编写；数据、模型与manifest未在服务器验证 |
| Qwen3模板 | `SSPO/src_sspo/llamafactory/data/template.py` | 已加入Qwen3候选；须与模型native template对照验收 |
| 三方法YAML生成 | `SSPO/examples/train/make_yaml.py` | Qwen3-1.7B Round4分支固定DPO/SSPO/StaticPE口径 |
| 三方法formal YAML | `SSPO/examples/train/qwen3-1.7b-it/{dpo,sspo,staticpe}/` | DPO effective16，SSPO/StaticPE effective64；共同labeled eval view |
| Alpaca输出/judge入口 | `SSPO/examples/staticpe/` | 固定`alpaca_eval==0.6.2`；尚待整理为三方法统一入口 |
| 本地包与依赖 | `SSPO/pyproject.toml`、`SSPO/requirements.txt` | Python 3.12静态元数据检查通过；无服务器resolver/import证据 |
| 4090离线wheelhouse | `code/scripts/round4/00_build_offline_wheelhouse.sh` | 已在exact code commit `2854c10…`生成并复核137-wheel、3.2GiB package；固定`setuptools==78.1.0`兼容AlpacaEval 0.6.2；构建后checkout clean |
| A100离线venv | `code/scripts/round4/01_install_a100_env.sh` | `/root/envs/round4-py312`已通过pip check、核心imports、Python3.12.3、torch2.5.1+cu124、CUDA12.4与2 GPU门禁 |
| 4090模型/数据冻结 | `code/scripts/round4/02_stage_hf_assets_4090.sh`、`02_stage_hf_assets.py` | 已完成3份不可变revision、30个payload文件、5.917GiB逐文件复核；总索引SHA通过 |

Round4实现已获批开展；离线resolver/wheelhouse、冻结资产、A100安装/import和两卡可见性已通过服务器验收，数值、DDP和gradient accumulation仍待smoke验收。

## 3. 已冻结的配置口径

用户于2026-09-01选择DPO全局有效batch16。正式设计为：

| 方法 | per-device train | GPU | GA | effective batch | per-device eval |
| --- | ---: | ---: | ---: | ---: | ---: |
| DPO | 1 | 2 | 8 | 16 | 4 |
| SSPO | 4 | 2 | 8 | 64 | 4 |
| StaticPE | 4 | 2 | 8 | 64 | 4 |

三方法均为1 epoch。StaticPE `lambda=0.1`，联合目标为 `(L_DPO+0.1L_PE)/1.1`。模型ID统一为 `Qwen/Qwen3-1.7B`。

三方法formal YAML已生成；服务器预处理仍须解析实际数据行数、来源revision与SHA，并据此确认`sspo_gamma_min`的resolved值。

## 4. StaticPE实现风险

当前trainer对labeled部分计算DPO-base；对unlabeled固定候选计算DPO-base reward概率 `p` 和条件编码 `c_1/c_2`，并分别记录DPO、PE和joint loss。

正式口径固定为`physical-microbatch-PE`：在两卡physical microbatch的unlabeled子集上同步形成一组`c_1/c_2`，然后累计8次梯度；不跨gradient accumulation形成单一population。

还需测试：denominator梯度、单类/空类分支、p接近0/1、non-finite、DDP gather、两流比例、resume和多卡指标去重。

## 5. 服务器执行与验收清单

代码交接后按以下清单生成执行证据；其中任何失败均不得静默绕过：

1. 冻结三方法共享的UltraFeedback labeled view与SSPO/StaticPE共享UltraChat view，并生成无样本级manifest；
2. 预生成StaticPE候选B，绑定base revision、template、generation config、seed和输出SHA；
3. 解析并核验已提交的label-only DPO配置：每设备1、2卡、GA8、effective16；
4. 数值核验SSPO labeled DPO-base、threshold/norm/prior、eval不污染running state与分项日志；
5. 按`physical-microbatch-PE`语义数值核验StaticPE loss和two-stream sampler；
6. 统一三方法train/eval/export/merge/reload入口；
7. 统一AlpacaEval 2.0生成、完整805条formal、少量smoke子集和LC聚合；
8. 编写三方法2-step smoke orchestrator，并确定性覆盖labeled/unlabeled分支；
9. 增加loss/gradient、DDP、gradient accumulation、checkpoint/resume、merge一致性和输出schema测试；
10. 复核已新增的CPython3.12/CUDA12.4离线wheelhouse与A100既有镜像内venv安装脚本，并补齐执行manifest。

## 6. 训练日志合同

正式实现至少输出：

- DPO：`loss/dpo`、chosen/rejected reward、margin、accuracy；
- SSPO：`loss/labeled`、`loss/unlabeled`、`loss/joint`、混合权重、threshold/norm状态；
- StaticPE：`loss/dpo`、`loss/pe`、`loss/joint`、p与c向量摘要；
- 所有方法：optimizer step、lr、grad norm、train/eval loss、labeled/unlabeled counts、tokens、吞吐和显存峰值。

日志标量必须在detach后汇总，只用于观测；不能参与反向传播或改变多卡梯度。历史SSPO分项日志能否直接复用，需要在新入口上确认，不能仅依据字段名推断。

## 7. 服务器验证顺序

代码交接获批后的执行顺序：

1. 用户审阅完整diff并明确授权；
2. commit/push不可变版本；
3. 4090-3拉取exact clean commit，使用Python3.12 builder生成torch2.5.1+cu124及全部依赖的离线wheelhouse与SHA清单；
4. 在4090-3下载冻结revision数据和模型，分别生成manifest/SHA；
5. FusionOne复用既有`cuda12.4-cudnn-devel-ubuntu22.04-py312-ssh`镜像并创建/占用2张A100；
6. 经SSH续传wheelhouse、数据和模型，目标端逐文件复核SHA，在镜像内新建Round4 py312 venv并做核心import/CUDA检查；
7. 在这2张A100上运行服务器单元/合同测试；
8. 三方法顺序完成2 optimizer-step smoke、eval、adapter保存；
9. merge并在新进程离线重载，运行固定小样本生成；
10. Alpaca小子集格式测试与一次judge API smoke；
11. 重新申请formal执行授权后，三方法继续共用这2张A100顺序训练并运行完整AlpacaEval。

4090-3 SSH落点没有Docker daemon，因此Round4不在该落点构建镜像；复用FusionOne既有镜像。旧runs/envs已按用户明确指令行政清空，当前仍须在wheelhouse与数据/模型下载前实时核验约69GiB可用空间是否足够。4090-3不承担Round4训练smoke。

## 8. 本地静态边界

本地只允许编辑源码、YAML、shell和Markdown以及纯文本静态检查；不安装/import项目依赖，不运行数据、模型、测试、训练、评价或聚合。模型、数据、checkpoint、逐样本输出和原始日志全部留在服务器。

任何账号、密码、token、内部地址或API key都不得写进源码、配置、文档、日志或镜像层。运维命令只使用占位符。

当前代码交接状态：**APPROVED，当前累计纯构建修复版`round4-code-v1.0.3`，2026-09-01；用户要求继续当前服务器任务，允许提交不改变研究语义的纯实现修复并继续`SERVER_EXECUTION`。**

当前服务器边界：4090-3的exact checkout、wheelhouse、冻结数据/模型和manifest/SHA任务已完成，不执行训练。4090→A100专用认证、传输、A100资产复核和离线环境安装已经完成；下一步允许执行合同测试与smoke，formal训练仍须按获批实验合同另行授权。执行证据见`../exp/exp-20260901-01-round4-server-prep/README.md`。
