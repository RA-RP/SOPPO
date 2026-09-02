# SOPPO Round4 代码状态总览

## 0. 当前状态与授权边界

- Cycle：`cycle-20260901-01` / Round4
- 当前唯一活动阶段：`CODE_IMPLEMENTATION`
- 当前理论：`../human_read/theory/current_theory.md` `r4-theory-v1.0`，2026-09-01用户明确通过
- 当前实验：`../human_read/exp/current_experiment.md` `round4-exp-v1.0`，2026-09-01用户明确通过
- 已执行代码：`round4-code-v1.1.0` / exact code commit `af6dac49044978d76aeca4d5fcb0d11856a1c104`；依赖、四项固定资产和A100环境通过，但预处理严格校验在训练前失败
- Round4代码候选：`round4-code-v1.1.1` / exact code commit `6b010b89d1c62aaa8a42af65d06b53d301b1aee8`；过滤固定源数据中的空/单边回答并记录审计，同时使资产索引SHA文件可跨主机校验
- 当前代码交接：**PENDING USER CONFIRMATION**；`v1.1.1`尚未获准上传或执行，既有formal授权仍受新代码交接和完整smoke门禁约束

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
| DPO/SSPO/StaticPE trainer | `SSPO/src_sspo/llamafactory/train/dpo/trainer.py` | StaticPE保留DPO+PE；SSPO labeled使用未归一化DPO-base；普通DPO未定义`policy_unlabeled_logits`指标引用已移除并新增`dpo/loss`；待服务器数值/梯度验收 |
| two-stream sampler | `SSPO/src_sspo/llamafactory/data/sampler.py` | SSPO与StaticPE共用无放回比例保持sampler；分布式/累积语义待测 |
| StaticPE固定B候选生成 | `SSPO/preprocessing_data/generate_staticpe_candidates.py` | 本地模型不再传无意义Hub revision；smoke启用`--fail_on_drop`，空/重复候选直接失败；待服务器验证 |
| Qwen3模板 | `SSPO/src_sspo/llamafactory/data/template.py` | 已加入Qwen3候选；须与模型native template对照验收 |
| 三方法YAML生成 | `SSPO/examples/train/make_yaml.py` | Qwen3-1.7B Round4分支固定DPO/SSPO/StaticPE口径 |
| 三方法formal YAML | `SSPO/examples/train/qwen3-1.7b-it/{dpo,sspo,staticpe}/` | DPO effective16，SSPO/StaticPE effective64；共同labeled eval view |
| Alpaca输出入口 | `SSPO/examples/staticpe/generate_alpacaeval_outputs.py` | 已泛化到三方法；支持冻结本地JSON、固定前N条、merged模型本地重载和输出SHA manifest；待服务器验证 |
| Round4全链smoke | `code/scripts/round4/03_run_smoke_a100.sh`及同目录`03_*`辅助脚本 | 已实现exact commit/env/asset门禁、预处理manifest、固定fixture、2-step训练/eval、merge/reload、每方法1条judge与aggregate-only验收；待服务器验证 |
| Round4数据预处理与审计 | `SSPO/preprocessing_data/preprocessing_ultrachat.py`、`code/scripts/round4/03_verify_prepared.py` | `v1.1.1`在确定性0.1抽样后过滤空/单边回答，manifest升级为`round4-preprocessing-v2`并记录输入、丢弃原因和有效行数；待服务器复跑 |
| 本地包与依赖 | `SSPO/pyproject.toml`、`SSPO/requirements.txt` | Python 3.12静态元数据检查通过；无服务器resolver/import证据 |
| 4090离线wheelhouse | `code/scripts/round4/00_build_offline_wheelhouse.sh` | `af6dac4`已复用136个依赖wheel、重建项目wheel并通过137-wheel全量SHA；`v1.1.1`仍须重建唯一项目wheel并重新绑定commit |
| A100离线venv | `code/scripts/round4/01_install_a100_env.sh` | 新环境`/root/envs/round4-py312-af6dac490449`已通过pip check、核心imports、Python3.12.3、torch2.5.1+cu124、CUDA12.4与2 GPU门禁；旧`/root/envs/round4-py312`不含commit标记且本轮不使用 |
| 4090模型/数据冻结 | `code/scripts/round4/02_stage_hf_assets_4090.sh`、`02_stage_hf_assets.py` | `af6dac4`已固定并传输Qwen、UltraFeedback、UltraChat、AlpacaEval四资产，A100端42个payload逐文件SHA通过；`v1.1.1`把索引`.sha256`从源机绝对路径改为相对文件名 |

`af6dac4`执行已完成wheelhouse、四资产和A100 exact-commit环境门禁。随后全量预处理写盘成功，但`03_verify_prepared.py`发现抽样数据含不满足ranking合同的记录：UltraFeedback训练抽样8条无效、共同eval 3条无效、UltraChat抽样3条空回答；因此在StaticPE候选生成和任何训练前停止。`v1.1.1`保持随机抽样结果不变，只在抽样后过滤这些行并把原因计入manifest；服务器复跑前仍需新代码交接确认。

### 2.1 smoke数据与执行映射

- 全量预处理从已冻结UltraFeedback/UltraChat本地snapshot生成3个JSON和`ROUND4_PREPROCESS_MANIFEST.json`；`v2` manifest再次验证来源revision、过滤策略、输入/丢弃/有效行数、互斥行类型和SHA。
- fixture固定为DPO 32条labeled、SSPO/StaticPE共享64条labeled+64条unlabeled、共同eval 8条labeled；DPO 32条是共享64条的子集。
- smoke不改formal batch：DPO `1×2×GA8=16`，SSPO/StaticPE `4×2×GA8=64`；`max_steps=2`，step 1/2各eval一次。仅将mixed smoke sampler的每设备物理batch固定为`2 labeled+2 unlabeled`以覆盖分支。
- StaticPE先从冻结初始化模型为64条unlabeled各生成一次固定B；任一空结果或exact duplicate都失败，不能缩小population后继续。
- 每方法独立保存adapter、CPU合并为BF16 safetensors，并由新进程从merged目录加载，生成冻结AlpacaEval顺序前2条。judge使用`weighted_alpaca_eval_gpt4_turbo`和本地GPT-4 baseline，`max_instances=1`，即每方法1条API smoke。
- `alpaca-eval==0.6.2`内部LC辅助文件通过已冻结Alpaca资产映射到本地Hub cache；训练、生成和LC文件读取均不访问Hugging Face网络。OpenAI key只从服务器环境读取，缺失时在训练前终止。

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
7. **已实现、待服务器验收**：统一AlpacaEval 2.0本地生成、smoke子集、judge和LC离线辅助资产；完整805条formal已授权但必须等待smoke通过；
8. **已实现、待服务器验收**：三方法2-step smoke orchestrator与labeled/unlabeled确定性fixture；
9. **本次smoke内已覆盖部分**：分项loss、DDP、gradient accumulation、adapter、merge/reload和输出schema；独立数值梯度、checkpoint/resume合同测试仍待后续补齐；
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
11. smoke全部通过后，按已记录授权让三方法继续共用这2张A100顺序训练并运行完整AlpacaEval。

4090-3 SSH落点没有Docker daemon，因此Round4不在该落点构建镜像；复用FusionOne既有镜像。旧runs/envs已按用户明确指令行政清空，当前仍须在wheelhouse与数据/模型下载前实时核验约69GiB可用空间是否足够。4090-3不承担Round4训练smoke。

## 8. 本地静态边界

本地只允许编辑源码、YAML、shell和Markdown以及纯文本静态检查；不安装/import项目依赖，不运行数据、模型、测试、训练、评价或聚合。模型、数据、checkpoint、逐样本输出和原始日志全部留在服务器。

任何账号、密码、token、内部地址或API key都不得写进源码、配置、文档、日志或镜像层。运维命令只使用占位符。

静态复核：`v1.1.1`于2026-09-02对修改的Bash入口执行`bash -n`、对两个修改的Python文件执行标准库AST parse，并执行`git diff --check`与敏感信息模式检查；均通过。按本地控制面规则未导入项目、未运行数据/模型/test/smoke。服务器仍须复跑全量预处理、`v2` manifest验收和完整smoke。

当前代码交接状态：**PENDING USER CONFIRMATION，`round4-code-v1.1.1` / `6b010b89d1c62aaa8a42af65d06b53d301b1aee8`。**

当前服务器边界：`af6dac4`执行已在预处理验收失败处停止；不得上传或运行`6b010b8`。用户明确确认新exact commit后，才可增量重建项目wheel、创建新commit环境、保留并隔离失败prepared目录后重新预处理，再从三方法full-chain smoke开始；只有smoke全部通过才可按DPO→SSPO→StaticPE顺序执行formal。证据索引见`../exp/exp-20260901-01-round4-server-prep/README.md`。
