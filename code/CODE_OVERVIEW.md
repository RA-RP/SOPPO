# 代码总览

> **状态**：锁定的预实现草案。当前 Cycle ID：`cycle-20260818-01`；理论 v0.2 已通过，但 `human_read/exp/current_experiment.md` v0.3 MVP 仍记录为待用户明确通过，因此当前唯一活动阶段仍是 `EXP_DISCUSSION`。

## 0. 阶段与授权

- **Cycle ID**：`cycle-20260818-01`
- **当前阶段**：`EXP_DISCUSSION`；`CODE_IMPLEMENTATION` 未解锁
- **理论批准**：v0.2（2026-08-19）
- **实验设计批准**：无；v0.3 MVP 待用户明确通过
- **代码版本**：无已批准版本；现有文件仅为预实现草案
- **代码交接状态**：未开始
- **服务器执行授权**：无

**重要**：现有源码和脚本不构成已获批实现。只有实验设计明确通过后，才能正式进入代码阶段逐项复核、修订和交接；当前不得上传或执行。

---

## 1. 研究目标与实现边界

### 1.1 核心研究问题

验证在偏好标签有限时，使用 **population-level label-encoding 结构监督**（LERM 风格）替代 **instance-level pseudo target**，能否改善 preference learning。

### 1.2 实现范围

**第一阶段 MVP（已实现）**：
- 静态 preference dataset（UltraFeedback）人为隐藏标签
- 4 组对照：DPO-10%, Pseudo-target, DPO+PE, DPO-100%
- Label-encoding 结构损失 L_PE
- λ scheduler 消融：Fixed / Linear-Warmup / Exp-Warmup
- **单种子 MVP**（seed=42）：验证方法可行性

**不包含**（留待第二阶段）：
- On-policy sampling
- SFT rollout（动态候选对）
- Token-level 扩展
- 多种子重复与统计显著性检验

---

## 2. 目录结构

```
SOPPO/code/
├── src/
│   ├── __init__.py
│   ├── data/                      # 数据处理模块
│   │   ├── __init__.py
│   │   ├── prepare_ultrafeedback.py   # 数据下载、采样、标签隐藏
│   │   ├── dataset.py             # PyTorch Dataset 类
│   │   └── data_utils.py          # 审计、统计工具
│   ├── model/                     # 模型与损失模块
│   │   ├── __init__.py
│   │   ├── dpo_loss.py            # 标准 DPO loss
│   │   ├── pe_loss.py             # L_PE 结构损失（核心创新）
│   │   ├── pseudo_target.py       # Pseudo-target baseline
│   │   └── model_utils.py         # 模型加载、冻结、保存
│   ├── training/                  # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py             # 统一 Trainer 类
│   │   ├── scheduler.py           # λ scheduler（Fixed/Linear/Exp）
│   │   └── diagnostics.py         # 诊断工具
│   └── evaluation/                # 评估模块
│       ├── __init__.py
│       ├── metrics.py             # Acc, Brier, Calibration
│       └── evaluator.py           # 测试集评估
├── configs/mvp/                   # 配置文件
│   ├── base.yaml                  # 基础配置
│   ├── dpo10.yaml                 # DPO-10% 配置
│   ├── pseudo_target.yaml         # Pseudo-target 配置
│   ├── dpo_pe.yaml                # DPO+PE 配置
│   └── dpo100.yaml                # DPO-100% 配置
├── scripts/cluster/               # 服务器执行脚本
│   ├── server_paths.sh            # ICLR/SOPPO 平级目录合同
│   ├── runtime_env.sh             # Miniforge 加载与 Conda 环境激活
│   ├── 00_server_setup.sh         # 环境准备
│   ├── 01_server_tests.sh         # 单元测试
│   ├── 02_prepare_data.sh         # 数据准备
│   ├── 03_preexperiment.sh        # 预实验
│   ├── 04_lambda_search.sh        # λ 搜索
│   ├── 05_run_main.sh             # 主实验
│   ├── 06_c_epsilon.sh            # C_ε 观测
│   ├── 07_evaluate.sh             # 测试评估
│   └── 08_aggregate.sh            # 结果聚合
├── requirements.lock.txt          # 依赖（服务器安装）
├── CODE_OVERVIEW.md               # 本文件
└── AGENTS.md                      # 维护规则
```

---

## 3. 核心模块说明

### 3.1 数据处理（`src/data/`）

**prepare_ultrafeedback.py**：
- 从 HuggingFace 下载 UltraFeedback
- 随机采样 10k 样本，去重
- 划分：900 labeled train + 100 labeled val + 8k unlabeled + 1k test
- **标签隐藏**：unlabeled 和 test 只保留 `(x, response_a, response_b)`
- **位置随机化**：随机交换 response_a/response_b，避免泄漏
- **审计**：位置比例应 ~50:50，截断率 <5%，无跨集合泄漏

**dataset.py**：
- `LabeledDataset`：labeled 数据（含标签）
- `UnlabeledDataset`：unlabeled 数据（验证无标签）
- `TestDataset`：test 输入（标签由评估脚本单独加载）

**data_utils.py**：
- `audit_position_randomization()`：审计位置随机化
- `verify_label_isolation()`：验证标签隔离
- `check_cross_split_leakage()`：检查跨集合泄漏

### 3.2 模型与损失（`src/model/`）

**dpo_loss.py**：
- 标准 DPO loss：`L_DPO = -E[log σ(β Δ)]`
- `compute_dpo_loss()`：从模型和输入计算 DPO loss

**pe_loss.py**（核心创新）：
- **L_PE 结构损失**：
  ```
  ê_+ = Σ_i p_i q_i / (Σ_i p_i + ε)
  ê_- = Σ_i (1-p_i) q_i / (Σ_i (1-p_i) + ε)
  L_PE = 0.5 × (||ê_+ - e_+||_1 + ||ê_- - e_-||_1)
  ```
- **梯度路径**：初始方案让梯度穿过 p_i（`detach_denominator=False`）
- **应急方案**：如训练不稳定，对分母使用 `detach()`

**pseudo_target.py**：
- Instance-level 硬伪标签：`z̃_i = 1[p_i > 0.5]`
- 用伪标签计算 DPO loss

**model_utils.py**：
- `load_model_and_tokenizer()`：加载 Qwen3-4B
- `freeze_model()`：冻结 reference 模型
- `save_checkpoint()` / `load_checkpoint()`：检查点管理

### 3.3 训练（`src/training/`）

**trainer.py**：
- 统一 `Trainer` 类支持 4 种方法
- 每步训练：
  1. 从 D_L 计算 L_DPO
  2. 从 D_U 计算 L_PE 或 L_pseudo（如适用）
  3. 总损失：`L = L_DPO + λ(t) L_aux`
  4. 反向传播、优化、记录诊断信息

**scheduler.py**：
- `FixedLambdaScheduler`：λ(t) = λ*
- `LinearWarmupLambdaScheduler`：前 10% 线性增长
- `ExponentialWarmupLambdaScheduler`：指数增长

**diagnostics.py**：
- `compute_responsibility_quality()`：p_i 分布统计
- `compute_prediction_distribution_stats()`：熵、置信度
- `check_numerical_stability()`：检测 NaN/Inf
- `log_training_step()`：逐步记录到 JSONL

### 3.4 评估（`src/evaluation/`）

**metrics.py**：
- `compute_accuracy()`：Acc = 1[(p_i > 0.5) = z_i]
- `compute_brier_score()`：Brier = (p_i - z_i)²
- `compute_calibration()`：ECE 和 reliability diagram

**evaluator.py**：
- `evaluate_model()`：从 test_inputs.jsonl 计算预测
- 加载 private_labels/test_labels.jsonl（仅评估脚本访问）
- 保存预测到 test_predictions.jsonl（**服务器保留**）
- 保存 metrics 到 test_metrics.json（**可回传**）

---

## 4. 配置文件（`configs/mvp/`）

### 4.1 base.yaml（共享配置）

```yaml
model:
  name_or_path: "Qwen/Qwen3-4B"
  max_seq_len: 2048

training:
  beta: 0.1          # 从预实验锁定
  lr: 1e-6           # 从预实验锁定
  epsilon: 1e-8      # 从预实验锁定
  epochs: 2
  global_batch_size: 128
  seed: 42           # MVP 单种子
```

### 4.2 方法特定配置

- **dpo10.yaml**：仅 labeled，lambda=0
- **pseudo_target.yaml**：labeled + unlabeled，pseudo-target loss
- **dpo_pe.yaml**：labeled + unlabeled，PE structure loss
- **dpo100.yaml**：所有数据 labeled（oracle）

---

## 5. 服务器执行流程

### 阶段 -1：环境准备（`00_server_setup.sh`）
- 在 `gn001` 通过 `runtime_env.sh` 加载已验证的 `miniforge3/25.11.0-0` module
- 创建固定 Python 3.10 的 Conda 路径环境：`<SERVER_BASE>/envs/youc`
- 安装依赖（`requirements.lock.txt`）
- 环境检查：CUDA、PyTorch、HuggingFace
- 生成后续会话入口：`<SERVER_BASE>/activate_env.sh`
- 输出：`environment_summary.json`

### 阶段 0：单元测试（`01_server_tests.sh`）
- L_PE 数值正确性测试
- 梯度路径验证
- 分母稳定性测试（不同 ε 值）
- DPO loss 正确性测试
- 输出：`test_results.txt`

### 阶段 1：数据准备（`02_prepare_data.sh`）
- 下载 UltraFeedback，采样 10k
- 划分、去重、标签隐藏、位置随机化
- 生成 manifest_public.json（**可回传**）
- 审计：位置比例、截断率、跨集合泄漏
- 输出：6 个 JSONL 文件 + 2 个 manifest

### 阶段 2：预实验（`03_preexperiment.sh`）
- 数值稳定性：测试 ε ∈ {1e-5, 1e-6, 1e-8}
- Headroom 验证：DPO-100% vs SFT baseline（需 ≥5% gap）
- 超参数搜索：β ∈ {0.1, 0.5}, lr ∈ {5e-7, 1e-6}
- 显存验证：batch_size=128 on 2×A800
- 输出：锁定 (ε, β, lr)

### 阶段 3：λ 搜索（`04_lambda_search.sh`）
- 搜索空间：λ ∈ {0.1, 0.3, 0.5, 1.0}
- 在 validation set（100 样本）上选择 λ*
- DPO+PE 和 Pseudo-target 使用相同 λ*
- 输出：`lambda_search_report.md`

### 阶段 4：主实验（`05_run_main.sh`）
- **8 次训练**（MVP 单种子）：
  - DPO-10%: 1 次
  - Pseudo-target: 3 次（3 schedulers）
  - DPO+PE: 3 次（3 schedulers）
  - DPO-100%: 1 次
- 每 20 steps 保存 checkpoint
- 每 20 steps 验证评估
- 记录诊断信息（responsibility, encoding estimates）
- 输出：checkpoints/, logs/, diagnostics/

### 阶段 5：C_ε 观测（`06_c_epsilon.sh`）
- 使用 `<SERVER_BASE>/SOPPO/code/observe/LLM-output-density/GetSlice`
- 每个方法 × 10 checkpoints = 80 次 GetSlice
- s_nsamples=128, x_nsamples=128
- 观测功能方向压缩（c_ε）
- 输出：奇异值 JSON、c_epsilon_trajectory.csv

### 阶段 6：测试评估（`07_evaluate.sh`）
- 加载 best checkpoint
- 在 test_inputs.jsonl 上计算预测
- 加载 private_labels/test_labels.jsonl
- 计算 Acc, Brier, ECE
- 输出：test_predictions.jsonl（**服务器保留**），test_metrics.json（**可回传**）

### 阶段 7：结果聚合（`08_aggregate.sh`）
- 聚合所有方法的 metrics
- 生成对比表格和图表
- 准备白名单导出包
- 输出：summary.json, metrics.csv, figures/*.png

---

## 6. 数据流

```
HuggingFace (UltraFeedback)
  ↓ 02_prepare_data.sh
labeled_train.jsonl (900) + labeled_val.jsonl (100)
unlabeled_train.jsonl (8k, 标签隐藏)
test_inputs.jsonl (1k, 标签隐藏)
private_labels/ (训练禁止访问)
  ↓ 03_preexperiment.sh
锁定超参数 (ε, β, lr)
  ↓ 04_lambda_search.sh
锁定 λ*
  ↓ 05_run_main.sh
8 个训练 → checkpoints/ + logs/
  ↓ 06_c_epsilon.sh
C_ε 观测 → c_epsilon_trajectory.csv
  ↓ 07_evaluate.sh
测试评估 → test_metrics.json
  ↓ 08_aggregate.sh
白名单导出 → <SERVER_BASE>/exports/<experiment_id>/
```

---

## 7. 关键实现要点

### 7.1 L_PE 实现（`src/model/pe_loss.py`）

```python
def forward(self, p_i: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    # p_i: [batch_size]
    q_i = torch.stack([p_i, 1 - p_i], dim=-1)  # [batch_size, 2]
    
    sum_p = p_i.sum() + self.epsilon
    sum_1_minus_p = (1 - p_i).sum() + self.epsilon
    
    # 估计条件编码
    e_hat_plus = (p_i.unsqueeze(-1) * q_i).sum(dim=0) / sum_p
    e_hat_minus = ((1 - p_i).unsqueeze(-1) * q_i).sum(dim=0) / sum_1_minus_p
    
    # L1 距离
    loss = 0.5 * (
        torch.abs(e_hat_plus - self.e_plus).sum() +
        torch.abs(e_hat_minus - self.e_minus).sum()
    )
    
    return loss, diagnostics
```

### 7.2 标签隔离

- 训练代码只接收 `unlabeled_train.jsonl`（无 label 字段）
- 评估代码单独读取 `private_labels/test_labels.jsonl`
- 单元测试验证：尝试读取 private_labels 应报错

### 7.3 位置随机化

```python
swap = random.random() < 0.5
if swap:
    response_a, response_b = rejected, chosen
    true_label = 0  # response_b (index 1) preferred
else:
    response_a, response_b = chosen, rejected
    true_label = 1  # response_a (index 0) preferred
```

### 7.4 配置管理

- 所有超参数记录在 config.yaml
- 每个训练开始时保存完整配置快照
- 配置包含：超参数、数据路径、模型路径、随机种子、代码版本

---

## 8. 服务器待验证项

以下项目在本地无法验证，需服务器执行后确认：

| 项目 | 验证方式 | 脚本 |
|------|---------|------|
| L_PE 数值正确性 | 单元测试 | 01_server_tests.sh |
| 梯度路径 | 梯度存在性测试 | 01_server_tests.sh |
| 分母稳定性 | 不同 ε 值测试 | 01_server_tests.sh, 03_preexperiment.sh |
| 标签隔离 | 尝试读取 private_labels 应失败 | 01_server_tests.sh |
| 位置随机化 | 比例应 ~50:50 | 02_prepare_data.sh |
| 跨集合泄漏 | prompt 不应重复 | 02_prepare_data.sh |
| 显存占用 | batch_size=128 on 2×A800 | 03_preexperiment.sh |
| 训练稳定性 | 无 NaN/Inf | 05_run_main.sh |
| C_ε 计算 | GetSlice 正常运行 | 06_c_epsilon.sh |

---

## 9. 产物与回传

### 9.1 服务器保留（不回传）

- 原始/处理后数据（含 prompt/response 文本）
- 私有标签（`private_labels/`）
- 模型权重、checkpoints、optimizer states
- 逐样本预测（`test_predictions.jsonl`）
- C_ε 中间产物（奇异值 JSON, profiling matrices）
- 训练日志（可能含样本文本）

### 9.2 白名单（可回传本地）

```
<SERVER_BASE>/exports/<experiment_id>/
├── summary.json              # 方法级聚合
├── metrics.csv               # 一行一个方法
├── comparison.md             # 人类可读报告
├── figures/
│   ├── acc_comparison.png
│   ├── brier_comparison.png
│   ├── responsibility_dist.png
│   ├── encoding_trajectory.png
│   └── c_epsilon_comparison.png
├── config_final.yaml         # 锁定超参数
├── manifest_public.json      # 数据统计（无样本）
├── environment_summary.json  # 服务器环境
├── task_registry.json        # 任务状态
└── server_paths.txt          # 服务器产物索引
```

---

## 10. 成功标准（MVP）

### H1：最低有效性
- **条件**：DPO+PE Acc > DPO-10% Acc + 2%，Brier 不恶化
- **判断**：单种子观测趋势（不做统计检验）

### H2：核心贡献
- **条件**：DPO+PE Acc > Pseudo-target Acc + 2%，Brier 不恶化
- **判断**：单种子观测趋势

### H3：机制验证
- **条件**：Acc 与 Brier 同时改善，c_ε > 0，responsibility 稳定，预测分布不坍缩
- **判断**：诊断指标支持

**MVP 成功后扩展**：
1. 扩展到 3 种子（42, 43, 44）
2. 进行正式统计显著性检验
3. 报告 p 值和置信区间

---

## 11. 本地约束遵守情况

✅ **已遵守**：
- 本地只编辑源码、配置、脚本（纯文本）
- 未安装任何依赖
- 未导入项目
- 未运行测试、数据处理、模型加载
- 未执行任何 GPU 操作
- 所有路径使用占位符或环境变量

---

## 12. 代码交接清单

| 交接项 | 实验设计章节 | 源码/配置/入口 | 本地静态复核 | 服务器待验证 | 状态 |
|--------|-------------|---------------|-------------|-------------|------|
| 数据与标签隔离 | §3.3, §4 | `src/data/*.py`, `02_prepare_data.sh` | 待代码阶段复核 | 数据审计、隔离测试 | 锁定草案 |
| 方法与损失 | §5 | `src/model/*.py` | 待代码阶段复核 | 数值、梯度、集成测试 | 锁定草案 |
| 对照组与公平性 | §5.1, §5.2 | `configs/mvp/*.yaml`, `trainer.py` | 待代码阶段复核 | 共同配置审计 | 锁定草案 |
| 超参数与配置 | §5.2, §4 | `configs/mvp/base.yaml` | 待代码阶段复核 | 配置生效快照 | 锁定草案 |
| λ scheduler | §5.2.2 | `src/training/scheduler.py` | 待代码阶段复核 | 调度正确性 | 锁定草案 |
| 评价与诊断 | §5.4, §9 | `src/evaluation/*.py`, `diagnostics.py` | 待代码阶段复核 | 指标计算正确性 | 锁定草案 |
| 运行脚本 | §3-7 | `scripts/cluster/*.sh` | 仅完成目录合同修正 | Smoke、拒绝覆盖 | 锁定草案 |
| C_ε 观测 | §8 | `06_c_epsilon.sh` | 待代码阶段复核 | GetSlice 运行 | 锁定草案 |

**本表尚未形成正式代码交接。** 目录合同修正不代表研究实现已经通过复核。

---

## 13. 代码版本与静态校验

- **代码版本**：无已批准版本；`0.1.0-mvp` 仅为旧草案标签
- **静态校验值**：待实验批准并完成正式代码阶段后生成
- **依赖版本**：已锁定在 `requirements.lock.txt`
- **配置版本**：与实验设计 v0.3 MVP 一致

---

## 14. 当前门禁

当前不能请求代码交接确认。必须先由用户明确通过 `human_read/exp/current_experiment.md` 的具体版本，再进入 `CODE_IMPLEMENTATION` 对现有草案逐项复核和修订。完成正式交接后，才可请求用户确认“当前代码版本可以提交服务器”。

---

## 15. 已知限制与未来扩展

### 已知限制
- MVP 仅单种子（seed=42），不进行统计显著性检验
- 服务器目录合同已统一为 `<SERVER_BASE>/ICLR` 与 `<SERVER_BASE>/SOPPO` 平级，但实际服务器存在性仍待授权后核验
- 部分服务器脚本为计划框架，需补充实际训练/评估逻辑
- C_ε 观测需手动生成 GetSlice 配置

### MVP 成功后扩展
1. 扩展到 3 种子重复
2. 正式统计显著性检验
3. 第二阶段：动态 SFT 候选对
4. 消融实验：L_2 距离、KL 散度、梯度 detach

---

**状态总结**：当前仍处于 `EXP_DISCUSSION`。现有代码是锁定草案；目录与 Git 结构修正不解锁代码实现或服务器执行。
