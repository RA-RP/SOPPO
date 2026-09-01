# SOPPO Round4 代码状态总览

## 0. 当前状态与授权边界

- Cycle：`cycle-20260901-01` / Round4
- 当前唯一活动阶段：`THEORY_DISCUSSION`
- 当前理论：`../human_read/theory/current_theory.md` `r4-theory-v0.2`，尚未获批
- 当前实验：`../human_read/exp/current_experiment.md` `round4-exp-draft-v0.2`，锁定草案
- Round4代码状态：只有此前形成的 legacy StaticPE 候选；尚未进入本轮 `CODE_IMPLEMENTATION`
- commit/push、服务器部署、测试、smoke、镜像、训练与评价：全部未授权

本文件只盘点已有候选和正式实现缺口。Round3旧formal已行政结项；其代码与远程产物保持只读，不改写为Round4入口。

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

候选代码位于仓库内 `SSPO/SSPO/`：

| 能力 | 位置 | 状态 |
| --- | --- | --- |
| DPO/SSPO/StaticPE trainer候选 | `SSPO/SSPO/src_sspo/llamafactory/train/dpo/trainer.py` | 已有本地修改；仅静态阅读，未做服务器数值/梯度验收 |
| two-stream sampler | `SSPO/SSPO/src_sspo/llamafactory/data/sampler.py` | 保持总体 labeled/unlabeled 数量；分布式/累积语义待测 |
| StaticPE固定B候选生成 | `SSPO/SSPO/preprocessing_data/generate_staticpe_candidates.py` | 已编写；数据、模型与manifest未在服务器验证 |
| Qwen3模板 | `SSPO/SSPO/src_sspo/llamafactory/data/template.py` | 已加入Qwen3候选；须与模型native template对照验收 |
| YAML生成 | `SSPO/SSPO/examples/train/make_yaml.py` | StaticPE已有2卡、每设备4、GA8、epoch1候选 |
| StaticPE YAML | `SSPO/SSPO/examples/train/qwen3-1.7b-it/staticpe/` | 名义全局有效batch64 |
| Alpaca输出/judge入口 | `SSPO/SSPO/examples/staticpe/` | 原型固定`alpaca_eval==0.6.2`；尚非三方法统一入口 |
| 本地包与依赖 | `SSPO/SSPO/pyproject.toml`、`requirements.txt` | 仅静态检查；无服务器安装证据 |

这些文件的存在不等于Round4实现已获批，也不表示公式、DDP或gradient accumulation正确。

## 3. 已冻结的候选配置口径

用户于2026-09-01选择DPO全局有效batch16。当前设计候选为：

| 方法 | per-device train | GPU | GA | effective batch | per-device eval |
| --- | ---: | ---: | ---: | ---: | ---: |
| DPO | 1 | 2 | 8 | 16 | 4 |
| SSPO | 4 | 2 | 8 | 64 | 4 |
| StaticPE | 4 | 2 | 8 | 64 | 4 |

三方法均为1 epoch。StaticPE `lambda=0.1`，联合目标候选为 `(L_DPO+0.1L_PE)/1.1`。模型ID统一为 `Qwen/Qwen3-1.7B`。

正式DPO YAML、正式SSPO method ID与三方法共享的数据manifest尚未生成；只有StaticPE候选YAML不能构成完整Round4配置。

## 4. StaticPE实现风险

当前trainer对labeled部分计算DPO-base；对unlabeled固定候选计算DPO-base reward概率 `p` 和条件编码 `c_1/c_2`，并分别记录DPO、PE和joint loss。

当前PE统计发生在两卡physical microbatch的unlabeled子集，然后累计8次梯度。它不是在全局有效64行上收集完整population后只形成一次 `c_1/c_2`。因此代码阶段前必须由理论/实验明确选择：

- 接受并命名为 `microbatch-PE`；或
- 实现跨gradient accumulation的 `optimizer-population-PE`，并保证只做一个精确optimizer update。

还需测试：denominator梯度、单类/空类分支、p接近0/1、non-finite、DDP gather、两流比例、resume和多卡指标去重。

## 5. 正式实现缺口

进入 `CODE_IMPLEMENTATION` 后至少需要：

1. 冻结三方法共享的UltraFeedback labeled view与SSPO/StaticPE共享UltraChat view，并生成无样本级manifest；
2. 预生成StaticPE候选B，绑定base revision、template、generation config、seed和输出SHA；
3. 创建label-only DPO配置：每设备1、2卡、GA8、effective16；
4. 冻结SSPO的base loss、method ID、threshold/norm/prior与分项日志；
5. 根据获批population语义修订StaticPE loss和two-stream sampler；
6. 统一三方法train/eval/export/merge/reload入口；
7. 统一AlpacaEval 2.0生成、完整805条formal、少量smoke子集和LC聚合；
8. 编写三方法2-step smoke orchestrator，并确定性覆盖labeled/unlabeled分支；
9. 增加loss/gradient、DDP、gradient accumulation、checkpoint/resume、merge一致性和输出schema测试；
10. 编写无凭据容器定义、dependency lock、非`latest` tag/digest记录和FusionOne去敏runbook。

## 6. 训练日志合同

正式实现至少输出：

- DPO：`loss/dpo`、chosen/rejected reward、margin、accuracy；
- SSPO：`loss/labeled`、`loss/unlabeled`、`loss/joint`、混合权重、threshold/norm状态；
- StaticPE：`loss/dpo`、`loss/pe`、`loss/joint`、p与c向量摘要；
- 所有方法：optimizer step、lr、grad norm、train/eval loss、labeled/unlabeled counts、tokens、吞吐和显存峰值。

日志标量必须在detach后汇总，只用于观测；不能参与反向传播或改变多卡梯度。历史SSPO分项日志能否直接复用，需要在新入口上确认，不能仅依据字段名推断。

## 7. 服务器验证顺序

代码交接获批后的候选顺序：

1. 用户审阅完整diff并明确授权；
2. commit/push不可变版本；
3. 4090-3拉取exact clean commit，实时核验Docker/BuildKit、scratch与网络；
4. 在4090-3构建无凭据镜像，配置不自动训练的安全启动入口，使用非`latest` tag并记录digest；
5. 在4090-3下载冻结revision数据，生成manifest/SHA，并通过SSH传到A100仓库外数据目录；
6. 在FusionOne先创建/占用2张A100，核验GPU、显存、拓扑、挂载、依赖与目标端数据SHA；
7. 在这2张A100上运行服务器单元/合同测试；
8. 三方法顺序完成2 optimizer-step smoke、eval、adapter保存；
9. merge并在新进程离线重载，运行固定小样本生成；
10. Alpaca小子集格式测试与一次judge API smoke；
11. 重新申请formal执行授权后，三方法继续共用这2张A100顺序训练并运行完整AlpacaEval。

4090-3历史快照显示当前SSH落点没有可用Docker daemon且`/data`空间紧张。必须实时复核；若状态未变，用户指定的4090-3镜像步骤存在真实前置阻塞，需先补充构建权限和独立scratch。不能删除旧实验为镜像腾空间。4090-3不再承担Round4训练smoke。

## 8. 本地静态边界

本地只允许编辑源码、YAML、shell和Markdown以及纯文本静态检查；不安装/import项目依赖，不运行数据、模型、测试、训练、评价或聚合。模型、数据、checkpoint、逐样本输出和原始日志全部留在服务器。

任何账号、密码、token、内部地址或API key都不得写进源码、配置、文档、日志或镜像层。运维命令只使用占位符。

当前代码交接状态：**未开始Round4正式实现，不可提交或执行。**
