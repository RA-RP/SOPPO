# SOPPO v0.6 SSPO-aligned 30k MVP：代码总览与交接

## 0. 状态

- Cycle：`cycle-20260818-01`
- Experiment：`exp-20260819-01-mvp`
- 设计依据：`../human_read/exp/current_experiment.md` v0.6，用户已于 2026-08-21 明确要求开始编码
- 当前阶段：`CODE_IMPLEMENTATION`
- 代码交接：实现与本地静态复核已完成，等待用户明确确认当前代码版本可以提交服务器
- 服务器执行：`LOCKED`；本文不构成上传或 Slurm 授权

本轮本地只编辑纯文本源码、配置和说明。没有在本地安装/import 项目依赖，没有运行 pytest、数据、模型、训练、评价或 GPU 任务。运行正确性必须由获批后的服务器 tests/strong smoke 证明。

## 1. 实现范围

实现 Qwen3-4B、30k UltraFeedback、seed42 的八条最终 LoRA 轨迹：

```text
DPO-10
DPO-100
SSPO-hard-exp
SOPPO-PE-exp
SOPPO-PE-static lambda = 0.1 / 0.3 / 0.5 / 1.0
```

旧 SFT、hard-static、Pseudo-target、DPO-style PE、linear/exp-warmup lambda 和全参 FSDP 路径已删除。保留数据隔离、Qwen manifest、DPO reference cache、独立 test、L18 `C_epsilon` 和 held-first Slurm DAG。

## 2. 关键实现

### 2.1 配置合同

`src/config.py` 递归加载 YAML、应用 dotted override，并 fail-closed 验证：

- 方法名只能是五类配置接口；static lambda 只能取四个预注册值；
- formal 模型必须 bf16/2048，LoRA 必须 r8/alpha16/dropout0/all projections；
- formal global batch 必须64，DPO为4×8×2，joint为全局8/56；
- DPO 固定1 epoch/lr1e-6/beta.1；SSPO/PE固定2 epochs/lr1e-5/beta10/margin2；
- AdamW/wd0/cosine/warmup.1/clip1/seed42；
- paper gamma、30k counts/ratios、reference cache与 label isolation 约束。

smoke 明确设置 `training.smoke_mode=true`，只缩小长度、batch和step，不改变损失定义。

### 2.2 LoRA 与 checkpoint

`src/model/model_utils.py`：

- 每次从冻结绝对路径、manifest 校验后的 Qwen3 base 离线加载；
- `peft.LoraConfig` 注入 q/k/v/o/gate/up/down；base 参数冻结；
- 启动时逐名验证只有预注册 target 的 `lora_A/lora_B` 可训练；adapter 重载还核对 rank/alpha/dropout/targets、base 与 manifest；
- 两 rank 使用 DDP，不再做 full-state FSDP gather；
- checkpoint 写 `adapter_model.safetensors`、`adapter_config.json`、tokenizer、`run_config.yaml` 和 `checkpoint_meta.json`；
- evaluator 可重新加载 adapter；GetSlice loader 识别 adapter 并在内存中 `safe_merge`。

adapter 不含 optimizer/scheduler state，`--init-checkpoint` 可继续微调参数，但不是 bit-exact resume；hard threshold EMA 也会重新初始化。

### 2.3 损失

- `src/model/dpo_loss.py`：response-only token-sum logp、response-token mean logp、任意 A/B label DPO。
- `src/model/sspo_loss.py`：SimPO labeled loss、margin-free PE pair probability、paper gamma、normalized fixed lambda、Gaussian KDE Bayes threshold、mean/std/threshold EMA 和 single-response hard logistic risk。
- KDE 使用论文的 200 grid；论文未给 bandwidth，代码冻结 per-class Scott rule并记录 bandwidth/risk/threshold。
- `src/model/pe_loss.py`：dense PE，加跨 rank完整 optimizer population 的精确一阶系数和第二次前向 surrogate。

SSPO-hard 对 unlabeled pair 的 A/B 独立打 hard label；PE 对 A/B 形成一个 direction-unknown pair probability。两者都看不到 hidden label。

### 2.4 Trainer 与 batch

`src/training/trainer.py` 提供统一 CLI。DPO 使用 reference cache与标准 8 microstep accumulation。joint 路径的每 rank pattern 为：

```text
unlabeled microbatch: 3,4,3,4,3,4,3,4  -> 28 pairs/rank
labeled microsteps:   0,2,4,6, each 1   -> 4 pairs/rank
two ranks                              -> 56 U + 8 L
```

hard 第一遍收集全局 labeled winning/losing 与全部 response reward并更新 KDE/EMA，第二遍回传。PE 第一遍只收集56个全局 pair probability并求 exact coefficient，第二遍回传。DDP `no_sync` 只延迟通信到该 optimizer step 的最后一次 backward，不改变归一化。

validation：DPO 用 reference delta；SSPO/PE 用 margin-free SimPO mean-logp delta。best 依次按更高 accuracy、更低 Brier；每次 metrics 都保留 score type、loss weights、global batch与 hard/PE诊断。

### 2.5 选择、评价与观测

- `src/training/selectors.py`：共同 raw mean-logp score 下 DPO-10 对训练前显式禁用 adapter 的冻结 base 的 .05 headroom gate，并核对前后 score type/样本数；四条 static lambda validation-only selector。DPO-100只作为 oracle。
- `src/evaluation/evaluator.py`：独立 adapter 加载；只有 DPO 读取 reference cache，只有此入口读取 test private labels。
- `src/evaluation/aggregate.py`：要求八个评价完整，报告 dynamic controlled 与 validation-selected static 差值，不在 test 上挑最好方法。
- `src/evaluation/c_epsilon.py`：Qwen L18，epsilon `{.01,.025,.05,.10}`，module-first equal5/equal7。
- `src/data/audit_prepared_data.py`：提交前重验30k行数、SHA-256、跨 split ID、公开隐藏标签和私有标签精确连接；审计摘要进入回传白名单。
- `observe/.../GetSlice/utils/model_utils.py`：相对原工具唯一的当前项目兼容改动是 adapter-aware offline load + in-memory safe merge。

## 3. 冻结超参

| 项 | DPO-10 / DPO-100 | SSPO-hard / SOPPO-PE |
| --- | ---: | ---: |
| epoch | 1 | 2 |
| lr | 1e-6 | 1e-5 |
| loss beta | DPO 0.1 | SimPO 10 |
| margin | — | 2 |
| global batch | 64 | 64 = 8 L + 56 U |
| max seq len | 2048 | 2048 |
| LoRA | r8/alpha16/dropout0 | 同左 |
| optimizer | AdamW, wd0 | 同左 |
| scheduler | cosine, warmup .1 | 同左 |
| epochs | 1 | 2 |

SSPO/PE exp：`gamma0=1`、`gamma_min=2700/26700`、`decay=.01`。PE static使用 normalized lambda `{.1,.3,.5,1.0}`。hard：prior.5、EMA.95、KDE grid200、Scott bandwidth。PE：epsilon1e-8、L1、denominator不detach。

## 4. 服务器入口与不重复的八条训练

在 `gn001` 单独完成环境、模型和数据：

1. `00_server_setup.sh`：补装锁定的 `peft==0.15.2` 并验证 Qwen3/PEFT；
2. `02_download_model.sh`：若冻结 Qwen3 已存在则只校验；
3. `02_prepare_data.sh`：若 30k 数据已完成则复用；
4. 获得新的服务器执行授权后运行 `submit_all.sh`；提交器自行激活环境，并在任何 `sbatch` 前重验模型 manifest、30k 数据和 clean Git checkout。

完整 DAG：

```text
tests -> strong smoke -> oracle/reference
      -> stage03 [DPO-10, DPO-100] -> DPO-10 vs frozen-base headroom
      -> stage04 [4 static PE] -> static selection
      -> stage05 [hard-exp, PE-exp]
      -> C_epsilon [base + 8]
      -> evaluation [8] -> aggregate/export
```

stage03/04/05 合计正好八条 final trajectories，都写在 `runs/<experiment>/main/`，不存在预实验后重复训练。

## 5. checkpoint、产物与空间

- DPO：每20 step及 final；DPO-10预计约42 steps，保留20/40/final。
- SSPO/PE：每40 step及 final；全部为小型 LoRA adapter，显著小于旧全模型 checkpoint估算。
- `C_epsilon` 每轨迹最多选10点；合并只发生在当前进程内。
- server 保存 data/model/cache/adapters/raw logs/private predictions/C_epsilon raw。
- 本地只回传 summary JSON/CSV/Markdown、聚合图表、配置、manifest、环境摘要、registry 与路径索引。

## 6. strong smoke

`03_smoke.sh` 在 `gpu_test` 请求2 GPU、50分钟：

- 真实 Qwen3 offline/manifest 与 response mask；
- reference cache；
- LoRA base-frozen、两 rank DDP；
- DPO-10、DPO-100、hard-exp、PE-exp、PE-static各一步；
- hard KDE/EMA、exact-global PE、finite gradient；
- PE adapter保存后重载再训练一步。

正式任务另有2×A800 runtime gate。smoke通过只表示工程接口闭环，不代表30k训练一定不会出现后期数值或wall-time问题。

## 7. 静态复核与服务器待验证

本地复核范围：所有 cluster shell `bash -n`、`git diff --check`、旧接口/方法/路径静态搜索、八方法名称和配置入口交叉核对。依赖 import、pytest、Qwen3/PEFT兼容、DDP、显存、数值、adapter round-trip、GetSlice与DAG均必须在服务器验证。

已知风险：

- 2048长度和 hard/PE两次前向会增加 wall time；实际耗时以 strong smoke和首个formal日志校准。
- `gpu_test` 的Tesla具体型号未知；50分钟是partition硬上限内的强验收预算。
- SSPO论文未给KDE bandwidth，Scott rule是明确记录的复现决定。
- v0.6将已有pair拆成两个SSPO unpaired response，是数据形态适配，不等同于论文使用UltraChat single-response corpus。
- 单种子不能支持显著性结论；`C_epsilon`不是因果证据。

上述静态复核已完成且无报错；这只表示代码已具备请求执行授权的条件。只有用户新的明确确认才能进入 `SERVER_EXECUTION`。
