# OPD SFT H1 附带实验目录

本目录是 **H1 最小化实验** 的 TRL-first OPD-like 附带工程（sidecar）。
它独立于主仓库中的 `Train`、`Eval`、`MyFunc/GetSlice` 管线，用于在最小规模下验证
"TRL on-policy distillation-like LoRA 训练 → held-out eval loss 选 best → 全量评估 → matched-gain 配对 → GetSlice 几何交叉矩阵" 的完整闭环。

> **当前主入口是 `scripts/run_opd_minimal_closure_v2.py`（v2 严谨版）。**
> 真实运行结果与分析见 [`RESULTS_03_v2.md`](RESULTS_03_v2.md)。
> v2 相对 v1（`run_opd_minimal_closure.py`）修复了 5 个方法论缺陷（统一母池 / eval loss 选 best / DataSize 语义 / 正确 S-X 探针 / S×model 交叉矩阵）。

---

## 目录结构

```
opd_sft_h1/
├── README.md                          # 本文件
├── .gitignore                         # 忽略 runs/ 和 __pycache__/
├── minimal_trl_opd_experiment_03.md   # 详细实验计划（第三版最小闭环设计）
│
├── configs/                           # 实验配置文件
│   ├── trl_first_minimal.yaml         #   TRL OPD-like 主训练配置
│   ├── eval_targets_minimal.yaml      #   评估目标配置
│   └── experiment0_getslice_smoke.yaml#   GetSlice 冒烟测试配置
│
├── opd_sft_h1/                        # 核心 Python 包
│   ├── __init__.py
│   ├── paths.py                       #   路径常量与目录管理
│   ├── registry.py                    #   checkpoint / run 注册表读写
│   ├── trl_runner.py                  #   TRL 训练入口（OPD-like distillation）
│   ├── run_builder.py                 #   训练 run 构建与配置解析
│   ├── eval_ingest.py                 #   评估结果解析与摄入
│   ├── geometry_reader.py             #   GetSlice 几何数据读取
│   ├── geometry_metrics.py            #   谱间隙、有效秩、谱漂移等几何指标计算
│   ├── ood_metrics.py                 #   OOD-lite 指标计算（MMLU/WinoGrande/TruthfulQA drop）
│   ├── match_math500.py               #   OPD-SFT checkpoint 匹配逻辑
│   └── table_builder.py               #   结果汇总表生成
│
├── scripts/                           # 可执行脚本
│   ├── run_opd_minimal_closure_v2.py      #   ★ 主入口：v2 严谨版全闭环（smoke / 中等规模）
│   ├── run_opd_minimal_closure.py         #   v1 闭环（已被 v2 取代，保留供参考/工具函数复用）
│   ├── run_trl_distill_smoke.py           #   单步 TRL 蒸馏冒烟测试
│   ├── run_experiment0_getslice_smoke.py  #   GetSlice 冒烟测试
│   └── summarize_minimal_results.py       #   最小实验结果汇总
│
├── RESULTS_03_v2.md                   # v2 真实实验结果说明与分析
│
├── registry/                          # schema 定义
│   ├── run.schema.json                #   run 记录 JSON Schema
│   ├── checkpoint.schema.json         #   checkpoint 记录 JSON Schema
│   └── run_registry.example.jsonl     #   注册表示例
│
└── tests/                             # CPU 可运行的单元测试
    ├── conftest.py                    #   pytest / unittest fixtures
    ├── test_geometry_reader.py        #   几何读取测试
    ├── test_match_math500.py          #   匹配逻辑测试
    ├── test_ood_metrics.py            #   OOD 指标测试
    ├── test_registry.py               #   注册表测试
    └── fixtures/                      #   测试数据
        ├── tiny_eval_trajectory.csv
        ├── tiny_smat.json
        └── tiny_xmat.json
```

---

## 当前最小化计划（v2 真实规模）

训练方式为 **LoRA**（r=16, alpha=32, dropout=0.05，7 个 target module），base/student = Qwen3-1.7B，teacher = Qwen3-4B。

```text
统一母池:        NuminaMath-1___5，seed=42 一次性采样后切分为
                 train_prompts / train_sft / heldout_eval / probe_prompts（互不重叠）
cold-start theta0: 512 prompts，OPD-like(teacher 在线蒸馏)，grad_accum=4 → 128 步
OPD (lmbda=1.0):  从 theta0 出发，200 步 × grad_accum 4 = 消耗 800 prompt
SFT 对照:         从 theta0 出发，数据量扫描 [256, 512, 1024, 2048]，lr=1e-5，3 epoch
held-out eval:    64 条；OPD 算 JSD / SFT 算 CE，load_best_model_at_end
序列长度:         4096（GetSlice probe seqlen=512）
```

`cold-start / OPD` 都走 `opd_sft_h1/trl_runner.py` 的 OPD-like 蒸馏（teacher 在线提供 rollout/distillation 信号，不是已有 completion 的普通 SFT）。
`SFT 对照` 走闭包内独立的 `train_sft_control`（用母池里的 problem+solution 做监督，通过**增加数据量**扫出不同 GSM8K_gain 点，而非增 epoch）。

### 运行顺序（`run_opd_minimal_closure_v2.py`）

```text
1. 构建统一 prompt 母池并切分（train / heldout / probe 互不重叠）。
2. cold-start theta0（OPD-like，512 prompt）+ held-out eval loss 选 best。
3. OPD lmbda=1.0（从 theta0，800 prompt）+ held-out JSD eval 选 best。
4. SFT 对照扫数据量 256/512/1024/2048（从 theta0）+ held-out CE eval 选 best。
5. 6 模型统一全量评估（lm_eval vLLM，eval_limit=200）。
6. 选择 + 匹配：OPD 按 GSM8K 选 best；SFT 按 GSM8K_gain 最接近 OPD 配对。
7. 生成 S/X 探针：X=theta0 rollout 冻结共用；S 按模型区分(teacher/student rollout/训练数据)。
8. S×model 交叉矩阵 GetSlice（layer_14）→ 几何表 → 图表 → summary。
```

### checkpoint 选择策略（两层）

**第 1 层（训练轨迹内，防过拟合）**：`eval_strategy=steps` + `load_best_model_at_end=True` +
`metric_for_best_model=eval_loss`（OPD/cold-start 用 held-out JSD，SFT 用 held-out CE），
取轨迹内 eval_loss 最低的 checkpoint，`save_total_limit=2`。

**第 2 层（跨模型，构建主结论）**：OPD 端按 `max(GSM8K)` 选 best run；
SFT 端在数据量扫描点里选 `GSM8K_gain` 最接近 OPD 的那个做配对。

> 注：第 2 层的 OPD 选择目前按 GSM8K 绝对值（在 `eval_limit=200` 下噪声较大）；
> 本次只有 1 个 OPD run 故未暴露，多 OPD run 时需改更稳健策略。

### 评估策略

```text
训练中选择器: held-out eval loss（OPD=JSD / SFT=CE）
全量评估:     GSM8K、MATH500、MMLU、WinoGrande、TruthfulQA-MC1、TruthfulQA-MC2（eval_limit=200）
匹配指标:     GSM8K_gain
有效匹配阈值: 0.02（0-1 标度）
```

### 不纳入主表的指标

```text
AIME24: 禁用（高难度数学题目过于稀疏）
IFEval: 暂缓
HumanEval: 禁用（涉及不安全代码执行）
```

---

## 主要命令

运行完整闭环（v2 主入口）。脚本顶部 `main()` 内的 `cfg` 切换 smoke / 中等规模：

```bash
# 需在 density 环境 + 设置 TRL_EXPERIMENTAL_SILENCE=1
python experiments/opd_sft_h1/scripts/run_opd_minimal_closure_v2.py
# 可选 --shutdown-on-exit：完成或失败后尝试关机
```

- **smoke**（正确性验证）：`n_cold=8, n_opd_steps=2, sft_sizes=[8,16], eval_limit=8`
- **中等规模**（出科学结论）：`n_cold=512, n_opd_steps=200, sft_sizes=[256,512,1024,2048], eval_limit=200`

单步 TRL 蒸馏冒烟测试：

```bash
python experiments/opd_sft_h1/scripts/run_trl_distill_smoke.py \
  --config experiments/opd_sft_h1/configs/trl_first_minimal.yaml
```

CPU 安全单元测试：

```bash
python -m unittest discover experiments/opd_sft_h1/tests
```

---

## 判断标准

### 工程成功

满足以下条件即视为第一阶段工程成功：

1. `theta_0` SFT(Teacher-Rollout) 冷启动成功输出 checkpoint。
2. `theta_0` 成功完成全量基线评估和 GetSlice。
3. TRL OPD-like run 成功输出 checkpoint，并能用 GSM8K 选择器选出 `best_by_gsm8k`。
4. continued SFT 对照成功输出 checkpoint，并能选出 `closest_to_opd_gsm8k`。
5. 选中的 OPD/SFT checkpoint 都能进入同一评估解析器。
6. 选中的 OPD/SFT checkpoint 在删除前都完成了 GetSlice。
7. 匹配器能输出配对表。

### 初步支持假设 K1

仅当同时满足：

```text
TRL-OPD-like 的 OOD_lite_penalty_p2 更低
+ 几何漂移/间隙变化不差于 SFT
+ match_status = valid_match
```

才能写成"初步支持"。

> **v2 中等规模结果：三条件全部满足（见 [`RESULTS_03_v2.md`](RESULTS_03_v2.md)）。**
> matched 对 OPD(800)↔SFT-256（gap=0.010, valid_match）下：OOD_penalty_p2 delta=−0.0082，
> 谱漂移 OPD 0.0497 < SFT 0.0544，o_proj 子空间扰动 OPD 小一个量级 → **初步支持 K1**。

### 不支持或需调整

以下情况不支持直接推进 K1：

- TRL-OPD-like 无法达到可观察的 GSM8K gain。
- SFT 对照无法接近 OPD 的 GSM8K gain，`unmatched_nearest` 过多。
- SFT 完全匹配或优于 TRL-OPD-like。
- OOD-lite 更好但几何无差异，只能写成 weak mechanism。
- 几何更好但 OOD-lite 不好，不能支持 K1。

---

## 下一阶段决策

| 结果 | 下一步 |
|---|---|
| TRL 闭环跑通且有信号（**当前状态**） | 扩大 eval 样本量降噪 / 换 theta0 未饱和的更难分布，论证"提升+更稳"而非仅"维持+更稳" |
| TRL 闭环跑通但信号弱 | 调整 `lmbda`、教师模型、prompt 来源、学习率，重复小实验 |
| TRL 工程受限 | 切换到 verl 标准 OPD sidecar |
| 需要高并行 | 等前两步稳定后再做 verl async / fully async |

### 已知局限（v2）

- GSM8K 未真正"上升"：最高仅与 theta0 持平，差异在 `eval_limit=200` 噪声内 → 当前结论是"维持性能时更稳"而非"提升性能且更稳"。
- 第 2 层跨模型 OPD 选择按 GSM8K 绝对值，对噪声敏感（本次单 OPD run 未暴露）。
- theta0 已接近该分布上限，后续训练提升空间小。
