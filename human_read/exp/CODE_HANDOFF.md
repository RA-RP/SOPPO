# Exp → Code 交接文档

**生成时间**: 2026-08-19  
**Cycle ID**: `cycle-20260818-01`  
**实验设计版本**: v0.3 MVP（待用户明确通过）
**目标角色**: Code agent  
**当前阶段**: `EXP_DISCUSSION`；本文档是锁定草案

---

## 1. 门禁状态

✅ **理论批准**: `../theory/current_theory.md` v0.2（已通过，2026-08-19）  
⏳ **实验设计批准**: `current_experiment.md` v0.3 MVP 待用户明确通过
🔒 **CODE_IMPLEMENTATION**: 未解锁
🔒 **服务器执行授权**: 无

---

## 2. 你的任务

### 2.1 核心目标

实验设计 v0.3 MVP 明确通过后，才能把它转化为可在服务器执行的代码、配置和脚本。当前只能保留和阅读本锁定草案，不得据此实现或上传。

### 2.2 交付物

1. **代码实现**：`../../code/` 目录下的所有源码、配置、脚本
2. **代码说明文档**：`../../code/CODE_OVERVIEW.md`
3. **任务命令清单**：每个阶段的服务器执行命令（网页粘贴格式）

### 2.3 关键约束

**本地硬约束**（绝对禁止）：
- ❌ 不得在本地安装任何依赖、创建虚拟环境
- ❌ 不得在本地导入项目、运行测试
- ❌ 不得在本地下载或处理数据、模型
- ❌ 不得在本地执行任何 GPU 操作

**允许的本地操作**：
- ✅ 编辑源码、配置、脚本（纯文本）
- ✅ 静态文本检查（不导入项目）
- ✅ 编写 `CODE_OVERVIEW.md` 说明文档

**服务器使用方式**：
- 用户通过**网页界面**提交任务（不使用 SFTP）
- 所有任务编写为 `.sh` 或 `.py` 脚本
- 提供完整命令行供用户在网页粘贴执行
- 结果存储在指定文件夹（路径在命令行中指定）

### 2.4 完成标准

门禁解锁并完成正式代码实现后，需要：
1. 在 `CODE_OVERVIEW.md` 中记录完整的模块、接口、数据流
2. 提供每个阶段的服务器执行命令
3. 明确标注"待服务器验证"的项（如数值稳定性、梯度正确性）
4. **向用户明确确认**："当前代码版本可以提交服务器"
5. 只有用户明确确认后，才能解锁 `SERVER_EXECUTION` 阶段

---

## 3. 实验设计 v0.3 MVP 核心要点

### 3.1 MVP 定位

**这是一个单种子的最小可行实验**，目标是快速验证方法可行性。

**不包含**：
- 多种子重复（MVP 只用 seed=42）
- 统计显著性检验（观测趋势即可）
- On-policy / SFT rollout（第二阶段才引入）

**MVP 成功后**：
- 扩展到 3 种子重复
- 进行正式统计检验

### 3.2 实验规模

| 项目 | 配置 |
|------|------|
| 训练次数 | 8 次（单种子） |
| 对照组 | DPO-10%, Pseudo-target, DPO+PE, DPO-100% |
| λ scheduler | Fixed / Linear-Warmup / Exp-Warmup |
| 随机种子 | **42**（仅此一个） |
| GPU 时间 | ~53-103 GPU·小时 |
| 存储 | 峰值 ~650GB，最终 ~70GB |
| 端到端时间 | ~6-9 天（含阶段间讨论） |

### 3.3 阶段执行模式

**重要**：实验采用**阶段间暂停反馈**模式。

**执行流程**：
```
阶段 -1 完成 → 报告结果 → 用户讨论与确认 → 进入阶段 0
阶段 0 完成 → 报告结果 → 用户讨论与确认 → 进入阶段 1
阶段 1+2 完成 → 报告结果 → 用户讨论与确认 → 进入阶段 3
阶段 3 完成 → 报告结果 → 用户讨论与确认 → 结果分析
```

**你的代码设计需要支持**：
- 每个阶段独立运行
- 阶段间可暂停
- 用户可在阶段间多次讨论和调整

---

## 4. 数据与模型

### 4.1 数据集

**来源**：HuggingFace `openbmb/UltraFeedback`

**规模**：10k 子集，随机采样

**划分**：
- labeled_train: 900 样本
- labeled_val: 100 样本（从 1k labeled 划出）
- unlabeled_train: 8k 样本（标签隐藏）
- test: 1k 样本（标签隐藏）

**预处理要求**（阶段 -1）：
1. 随机采样 10k 样本
2. 按 prompt 去重
3. 确保同一 prompt 的回复对不跨集合
4. **标签隐藏**：
   - unlabeled 和 test 只保留 `(x, response_a, response_b)`
   - 真实标签存入 `private_labels/`（训练代码不可访问）
   - 随机交换 response_a/response_b 位置
5. **统计审计**：
   - 检查位置分布（应 ~50:50）
   - 检查长度、格式与真实标签的相关性
   - 输出 `manifest_public.json`（可回传）和 `manifest_private.json`（服务器保留）

**数据产物**（服务器路径，代码阶段确定）：
```
<DATA_PATH>/ultrafeedback/<revision>/
├── labeled_train.jsonl        # 900 样本
├── labeled_val.jsonl          # 100 样本
├── unlabeled_train.jsonl      # 8k 样本，位置随机
├── test_inputs.jsonl          # 1k 样本，位置随机
├── private_labels/
│   ├── unlabeled_labels.jsonl # 8k 真实标签（训练禁止访问）
│   └── test_labels.jsonl      # 1k 真实标签（仅评估访问）
├── manifest_private.json      # 样本级元信息（禁止回传）
└── manifest_public.json       # 汇总统计（可回传）
```

### 4.2 模型

**模型**：Qwen3-4B（Instruct 版本）

**下载方式**：`modelscope download --model Qwen/Qwen3-4B`

**服务器路径合同**：`<SERVER_BASE>/models/Qwen3-4B`（实际存在性待授权后核验）

**序列长度**：max_seq_len = 2048

**Reference 模型**：与训练模型共享权重，训练时冻结

---

## 5. 实验阶段详细说明

### 5.1 阶段 -1：环境准备

**目标**：
1. 服务器环境检查与依赖安装
2. 数据集下载与预处理
3. 模型下载与加载测试
4. 代码单元测试

**服务器环境**（待穿透完成后确认）：
- 集群地址：`172.16.11.78:32206`
- 账号：`jiangwenhaoT`
- 镜像：`pytorch2.4.0-cu121-zhaoyuhangc:v2.0`
- 资源：A800 × 16（任务独立调度）

**核心依赖**：
- PyTorch 2.4.0, CUDA 12.1
- transformers >= 4.40.0
- datasets >= 2.18.0
- accelerate >= 0.28.0

**单元测试**（服务器运行）：
- L_PE 数值/梯度正确性
- 分母稳定性（Σp_i 和 Σ(1-p_i) 不为零）
- 梯度路径（是否穿过 responsibility p_i）
- 标签隔离（训练代码无法读取 `private_labels/`）
- 回复交换（位置随机化）
- 配置合并（4 组方法共享 backbone）
- 输出拒绝覆盖（已存在实验目录时报错）

**验收**：测试全部通过后才进入阶段 0

**阶段间暂停**：完成后报告结果，用户确认再继续

---

### 5.2 阶段 0：预实验

**目标**：
1. **数值稳定性测试**：测试 L_PE 中的数值稳定项 ε ∈ {1e-5, 1e-6, 1e-8}
2. **Headroom 验证**：确认 DPO-100% vs SFT baseline 有 ≥5% Acc 提升
3. **β 选择**：在 {0.1, 0.5} 中选择更优值
4. **lr 选择**：在 {5e-7, 1e-6} 中选择更优值
5. **显存验证**：确认 batch_size=128 在 2×A800 上可行

**实验设置**：
- 数据：完整 1k labeled + 8k unlabeled
- 种子：seed=42（单种子）
- 对照组：
  1. 数值稳定性测试：DPO+PE 测试 3 个 ε 值
  2. SFT baseline：在 labeled 1k 上用 chosen 回复做 SFT
  3. DPO-100%：在完整 9k 样本上训练 DPO（4 组超参数：β×lr）

**训练配置**：
- Epochs: 2
- Batch size: global=128, per_device=8, grad_accum=8, devices=2
- Optimizer: AdamW (β1=0.9, β2=0.999, weight_decay=0.01)
- Warmup: 10% steps
- 评估频率: 每 20 steps

**成功标准**：
- 数值稳定性：选择无 NaN/Inf、分母不退化的 ε 值
- Headroom：DPO-100% Acc > SFT Acc + 5%（否则换模型或数据）
- 超参数：在 validation set（100 样本）上选择最优 (β, lr)
- 显存：确认不 OOM（否则启用 gradient checkpointing）

**产物**：
- 预实验报告：`pre_experiment_report.md`
- 锁定超参数：ε, β, lr
- 显存 profile：`memory_profile.json`

**阶段间暂停**：完成后报告结果，用户确认再继续

---

### 5.3 阶段 1+2：主实验与 λ 搜索

#### 5.3.1 四组对照方法

**1. DPO-10%**（有限标签下界）：
- 数据：仅 D_L（900 labeled + 100 validation）
- 损失：标准 DPO loss
- 种子：42

**2. Pseudo-target**（逐样本伪目标）：
- 数据：D_L + D_U
- 训练流程（每个训练步）：
  1. 从 D_L 采样 batch_L，计算 L_DPO(D_L)
  2. 从 D_U 采样 batch_U
  3. 用当前 θ 计算 p_i = σ(Δ_i)
  4. 生成硬伪标签：z̃_i = 1[p_i > 0.5]
  5. 根据 z̃_i 确定 (y_w, y_l)
  6. 用伪标签对计算 L_DPO_pseudo(D_U)
  7. 反向传播：L = L_DPO(D_L) + λ L_DPO_pseudo(D_U)
- λ scheduler：测试 3 种（Fixed / Linear-Warmup / Exp-Warmup）
- 种子：42

**3. DPO+PE**（本方法：群体结构监督）：
- 数据：D_L + D_U
- 训练流程（每个训练步）：
  1. 从 D_L 采样 batch_L，计算 L_DPO(D_L)
  2. 从 D_U 采样 batch_U
  3. 用当前 θ 计算 p_i = σ(Δ_i)
  4. 估计条件编码：
     - ê_+ = Σ_i p_i q_i / (Σ_i p_i + ε)
     - ê_- = Σ_i (1-p_i) q_i / (Σ_i (1-p_i) + ε)
     - 其中 q_i = [p_i, 1-p_i]ᵀ，ε 为数值稳定项（从预实验选出）
  5. 计算结构损失：
     - L_PE = 0.5 × (||ê_+ - e_+||₁ + ||ê_- - e_-||₁)
     - 理想编码：e_+ = [1,0]ᵀ, e_- = [0,1]ᵀ
  6. 反向传播：L = L_DPO(D_L) + λ(t) L_PE(D_U)
- 梯度路径：初始方案让梯度穿过 p_i（不使用 stop_gradient）
- λ scheduler：测试 3 种（Fixed / Linear-Warmup / Exp-Warmup）
- 种子：42

**4. DPO-100%**（oracle 上界）：
- 数据：恢复 D_U 真实标签，使用完整 9k labeled
- 损失：标准 DPO loss
- 种子：42

**公平性约束**：
- 所有方法共享：相同初始化、reference 模型、优化器、lr、batch size、训练步数、数据划分、评估协议
- 唯一差异：训练数据使用范围、损失函数

#### 5.3.2 训练配置

**固定超参数**（从阶段 0 锁定）：
- ε（数值稳定项）：从 {1e-5, 1e-6, 1e-8} 选出
- β：从 {0.1, 0.5} 选出
- lr：从 {5e-7, 1e-6} 选出
- Epochs：2-3
- Batch size：global=128, per_device=8, grad_accum=8, devices=2
- Optimizer：AdamW (β1=0.9, β2=0.999, weight_decay=0.01)
- Warmup：10% steps
- Max seq len：2048
- **随机种子**：**42**（MVP 单种子）

**λ 搜索**：
- 搜索空间：{0.1, 0.3, 0.5, 1.0}
- 搜索策略：
  1. 对每个 λ，在 validation set（100 样本）上训练（Fixed-λ，seed=42）
  2. 选择 validation Acc 最高的 λ*
  3. 用 λ* 测试 3 种 scheduler
- Pseudo-target 的 λ：与 DPO+PE 使用相同值

**λ Scheduler**（3 种）：

1. **Fixed-λ**：
   ```python
   lambda_t = lambda_star  # constant
   ```

2. **Linear-Warmup**：
   ```python
   t_warmup = 0.1 * total_steps
   lambda_t = lambda_star * min(1.0, t / t_warmup)
   ```
   - 前 10% steps：λ 从 0 线性增长到 λ*
   - 后 90% steps：λ = λ*

3. **Exponential-Warmup**：
   ```python
   lambda_t = lambda_star * (1 - exp(-5 * t / total_steps))
   ```
   - 更平滑的增长曲线

**实验组合（MVP 单种子）**：
- DPO+PE: 3 schedulers = 3 次训练
- Pseudo-target: 3 schedulers = 3 次训练
- DPO-10%: 1 次训练
- DPO-100%: 1 次训练
- **总计**：8 次训练

**每个任务的产物结构**：
```
<EXP_PATH>/<METHOD>_<SCHEDULER>/seed-42/
├── config.yaml              # 冻结配置
├── checkpoints/             # 10 个 checkpoints（每 20 steps）
│   ├── step_020/
│   ├── step_040/
│   ...
│   └── step_best/          # validation Acc 最高
├── logs/
│   ├── training.log        # 训练日志
│   └── metrics.jsonl       # 逐步指标
├── predictions/
│   └── test_predictions.jsonl  # 服务器保留
└── diagnostics/
    ├── responsibility_dist.json    # p_i 分布
    ├── encoding_estimates.json     # ê_+, ê_-
    └── c_epsilon/                  # C_ε 观测
```

#### 5.3.3 评估与监控

**评估频率**：
- Validation：每 20 steps（共约 10 个 checkpoints）
- Test：仅在训练结束后，使用 best checkpoint

**核心指标**：
- Preference Accuracy：Acc = (1/|D_test|) × Σ_i 1[(p_i > 0.5) = z_i]
- Brier Score：Brier = (1/|D_test|) × Σ_i (p_i - z_i)²

**诊断信息**（每步记录）：
1. Responsibility 质量：Σ_i p_i, Σ_i (1-p_i) 的均值/方差，p_i 分布直方图
2. 编码估计（仅 DPO+PE）：ê_+, ê_- 的值，||ê_+ - e_+||₁, ||ê_- - e_-||₁
3. 伪标签统计（仅 Pseudo-target）：伪标签比例，翻转率
4. Loss 分解：L_DPO(D_L), L_PE(D_U) 或 L_DPO_pseudo(D_U), Total loss
5. 梯度统计：梯度范数，是否出现 NaN/Inf

**Checkpoint 保存策略**：
- 训练期间：每 20 steps 保存（共约 10 个）
- C_ε 观测期间：保留所有 checkpoints
- C_ε 观测完成后：**只保留 best checkpoint**（节省 ~90% 存储）

**阶段间暂停**：完成后报告结果，用户确认再继续

---

### 5.4 阶段 3：C_ε 相对功能压缩观测

**目标**：观测不同方法训练后的模型在偏好表示上的功能方向压缩

**C_ε 定义**：
```
r_ε(A) = min k such that Σ_{i>k} σ_i² / Σ_i σ_i² ≤ ε
其中 A_{D,t,m} = W_{t,m} S_{D,t,m}

c_{ε,D,t,m} = (r_{ε,D,0,m} - r_{ε,D,t,m}) / r_{ε,D,0,m}
```

**解释**：
- c_ε > 0：相对 base 压缩，需要的功能方向变少
- c_ε = 0：不变
- c_ε < 0：膨胀，功能方向变多

**观测策略（MVP 单种子）**：

**Checkpoint 选择**：
- Base：初始模型（Qwen3-4B Instruct）
- 训练中：
  - 前 3 个连续：ckpt_1, ckpt_2, ckpt_3
  - 后 7 个间隔一个：ckpt_5, ckpt_7, ckpt_9, ...（共 10 个）

**Probe 配置**：
- S（Source）：D_L（labeled training，900 样本）
- X（Target）：X_S = D_U（unlabeled set，8k 样本，静态数据）
- **第二阶段才引入 X_O**（on-policy rollout）

**ε 阈值**：
- Headline：ε = 0.05（保留 95% 能量）
- 稳健性检查：{0.01, 0.025, 0.05, 0.10}

**Module 聚合**：
- Equal-5 non-QK（主口径）：v, o, gate, up, down
- Equal-7（辅助）：q, k, v, o, gate, up, down

**层选择**：
- Headline layer：待确认（通常中间层，如 layer_14）
- 全层：如果计算资源允许

**GetSlice 配置示例**：
```json
{
  "model": "<MODEL_PATH>/checkpoints/step_020",
  "save_path": "<EXP_PATH>/diagnostics/c_epsilon/step_020",
  "mode": "split_whitened_svd",
  "DEV": "cuda",
  "model_seq_len": 2048,
  "seed": 3,
  "target_layer": 14,
  "s_nsamples": 128,
  "s_jsonl_path": "<DATA_PATH>/labeled_train.jsonl",
  "x_nsamples": 128,
  "x_jsonl_path": "<DATA_PATH>/unlabeled_for_x.jsonl",
  "cleanup_intermediate": false,
  "save_profile_s_path": "profile_s.pt",
  "save_profile_x_path": "profile_x.pt"
}
```

**重要**：s_nsamples 和 x_nsamples 设置为 **128**（与 batch_size 一致）

**实现路径**：
1. 使用 `<SERVER_BASE>/SOPPO/code/observe/LLM-output-density/GetSlice/slice.py`
2. 准备 S 侧和 X 侧 JSONL 数据
3. 为每个 method × checkpoint 运行 GetSlice
4. 后处理：使用 `cycle09_relative_functional_contraction.py` 和 `cycle09_equal5_non_qk.py`

**产物**（服务器保留）：
```
<EXP_PATH>/<METHOD>/seed-42/diagnostics/c_epsilon/
├── step_020/
│   ├── sMat_S.json
│   ├── xMat_X_S.json
│   ├── profile_s.pt
│   ├── profile_x.pt
│   └── c_epsilon_summary.csv
├── step_040/
...
└── c_epsilon_trajectory.csv  # 聚合所有 checkpoints
```

**可回传**：
- `c_epsilon_trajectory.csv`（方法级聚合）
- `c_epsilon_comparison.png`（方法间对比图）

**阶段间暂停**：完成后报告结果，用户确认再继续

---

## 6. 评估指标与成功标准（MVP）

### 6.1 核心指标

**Preference Accuracy**：
```
Acc = (1 / |D_test|) × Σ_i 1[(p_i > 0.5) = z_i]
```

**Brier Score**：
```
Brier = (1 / |D_test|) × Σ_i (p_i - z_i)²
```

### 6.2 成功标准（MVP 单种子）

**H1：最低有效性**
- 条件：DPO+PE Acc > DPO-10% Acc + 2%，Brier 不恶化
- 失败：提升 ≤2% 或更差 → 1-2 周诊断 → 撤退

**H2：核心贡献**
- 条件：DPO+PE Acc > Pseudo-target Acc + 2%，Brier 不恶化
- 失败：无法支撑"优于 instance-level target"论点

**H3：机制验证**
- 条件：Acc 与 Brier 同时改善，c_ε > 0，responsibility 质量稳定，预测分布不坍缩
- 失败：记为"机制未确认"

**MVP 判断**：
- 单种子（seed=42）观测趋势，**不做统计显著性检验**
- Acc 提升 >2% 视为有效信号
- 同时检查 Brier、responsibility 质量、c_ε 等诊断指标

**MVP 成功后扩展**：
1. 扩展到 3 种子（42, 43, 44）
2. 进行正式统计显著性检验（paired t-test, bootstrap）
3. 报告 p 值和置信区间

---

## 7. 服务器产物与本地回传

### 7.1 服务器产物（仅保留）

**数据**：
- 完整 UltraFeedback 处理后数据（含 private_labels）

**Checkpoints**（C_ε 观测完成后）：
- 每个训练只保留 best checkpoint（~8GB × 8 = ~64GB）
- 删除所有中间 checkpoints

**逐样本预测**：
- `test_predictions.jsonl`（服务器保留，不回传）

**C_ε 中间产物**：
- 奇异值 JSON、profiling matrices（服务器保留）

### 7.2 本地回传（白名单）

**聚合指标**：
```
<LOCAL_PATH>/export_local/
├── summary.json              # 方法级聚合
├── metrics.csv               # 一行对应一个方法
├── comparison.md             # 人类可读对比
└── figures/
    ├── acc_comparison.png
    ├── brier_comparison.png
    ├── responsibility_dist.png
    ├── encoding_trajectory.png
    └── c_epsilon_comparison.png
```

**配置快照**：
- `config_final.yaml`（锁定超参数）
- `manifest_public.json`（数据统计，无样本内容）
- `environment_summary.json`（服务器环境摘要）

**任务状态**：
- `task_registry.json`（任务 ID、状态、路径索引）
- `server_paths.txt`（服务器产物路径清单）

### 7.3 禁止回传（黑名单）

- 原始或处理后数据（含 prompt/response 文本）
- 私有标签（`private_labels/`）
- 模型权重、checkpoints、adapters、optimizer states
- Tokenized cache、embeddings
- 逐样本预测或逐样本指标
- 包目录、虚拟环境、wheel、依赖 cache
- 可能包含样本文本的原始日志

---

## 8. 代码实现要点

### 8.1 核心模块

你需要实现以下模块（建议结构）：

**数据处理**（`src/data/`）：
- `prepare_ultrafeedback.py`：下载、采样、划分、标签隐藏
- `dataset.py`：PyTorch Dataset 类（labeled / unlabeled / test）
- `data_utils.py`：位置随机化、审计统计

**模型与损失**（`src/model/`）：
- `dpo_loss.py`：标准 DPO loss
- `pe_loss.py`：L_PE 结构损失（关键实现）
- `pseudo_target.py`：Pseudo-target baseline
- `model_utils.py`：模型加载、reference 冻结

**训练**（`src/training/`）：
- `trainer.py`：统一训练循环
- `scheduler.py`：λ scheduler（Fixed / Linear-Warmup / Exp-Warmup）
- `diagnostics.py`：责任质量、编码估计、伪标签统计

**评估**（`src/evaluation/`）：
- `metrics.py`：Acc, Brier 计算
- `evaluator.py`：测试集评估
- `c_epsilon.py`：C_ε 观测封装

**服务器脚本**（`scripts/cluster/`）：
- `00_server_setup.sh`：环境创建与预检
- `01_server_tests.sh`：单元/集成测试
- `02_prepare_data.sh`：数据下载与预处理
- `03_preexperiment.sh`：预实验（数值稳定性 + headroom + 超参数）
- `04_lambda_search.sh`：λ 搜索
- `05_run_main.sh`：主实验训练
- `06_c_epsilon.sh`：C_ε 观测
- `07_evaluate.sh`：独立评估
- `08_aggregate.sh`：结果聚合

**配置**（`configs/mvp/`）：
- `base.yaml`：共享基础配置
- `dpo10.yaml`：DPO-10% 配置
- `pseudo_target.yaml`：Pseudo-target 配置
- `dpo_pe.yaml`：DPO+PE 配置
- `dpo100.yaml`：DPO-100% 配置

### 8.2 L_PE 实现要点

**数值稳定性**：
```python
def compute_l_pe(p_i, epsilon):
    """
    p_i: [batch_size] tensor, 偏好概率
    epsilon: 数值稳定项（从预实验选出）
    """
    q_i = torch.stack([p_i, 1 - p_i], dim=-1)  # [batch_size, 2]
    
    # 计算 responsibility 加权和
    sum_p = p_i.sum() + epsilon
    sum_1_minus_p = (1 - p_i).sum() + epsilon
    
    # 估计条件编码
    e_hat_plus = (p_i.unsqueeze(-1) * q_i).sum(dim=0) / sum_p
    e_hat_minus = ((1 - p_i).unsqueeze(-1) * q_i).sum(dim=0) / sum_1_minus_p
    
    # 理想编码
    e_plus = torch.tensor([1.0, 0.0], device=p_i.device)
    e_minus = torch.tensor([0.0, 1.0], device=p_i.device)
    
    # L1 距离
    loss_pe = 0.5 * (
        torch.abs(e_hat_plus - e_plus).sum() +
        torch.abs(e_hat_minus - e_minus).sum()
    )
    
    return loss_pe, {
        'e_hat_plus': e_hat_plus.detach().cpu().numpy(),
        'e_hat_minus': e_hat_minus.detach().cpu().numpy(),
        'sum_p': sum_p.item(),
        'sum_1_minus_p': sum_1_minus_p.item()
    }
```

**梯度路径**：
- 初始方案：让梯度穿过 p_i（不使用 `detach()`）
- 应急方案：如果训练不稳定，对分母使用 `detach()`

### 8.3 标签隔离

**关键约束**：训练代码**绝对不能**读取 `private_labels/`

**实现方式**：
1. 训练脚本只接收 `unlabeled_train.jsonl` 和 `test_inputs.jsonl`
2. 评估脚本单独运行，读取 `private_labels/` 并计算指标
3. 单元测试验证：训练入口尝试读取 `private_labels/` 应报错

### 8.4 配置管理

**所有超参数必须记录在配置文件**：
- 从预实验锁定的：ε, β, lr
- 固定的：epochs, batch_size, optimizer, warmup
- 方法特定的：λ, scheduler

**配置冻结**：
- 每个训练任务开始时，保存完整配置快照到输出目录
- 配置包含：所有超参数、数据路径、模型路径、随机种子、代码版本/校验值

---

## 9. 代码实现检查清单

### 9.1 实现阶段检查

- [ ] 所有模块编写完成（data / model / training / evaluation）
- [ ] 服务器脚本编写完成（00-08）
- [ ] 配置文件与实验设计一致
- [ ] L_PE 实现正确（待服务器测试验证）
- [ ] 标签隔离机制实现
- [ ] 回复位置随机化实现
- [ ] 输出路径包含 experiment_id/method/scheduler/seed
- [ ] 默认拒绝覆盖已存在目录
- [ ] `CODE_OVERVIEW.md` 完整记录模块、接口、数据流

### 9.2 本地约束检查

- [ ] 未在本地安装任何依赖
- [ ] 未在本地导入项目
- [ ] 未在本地运行测试、数据处理、模型加载
- [ ] 未在本地执行任何 GPU 操作
- [ ] 所有路径使用占位符或从环境变量/配置读取

### 9.3 服务器脚本检查

- [ ] 每个脚本开头检查 `RUN_CONTEXT=cluster`
- [ ] 使用 `set -euo pipefail`
- [ ] 路径从环境变量或显式参数获取
- [ ] 保存最终配置、源码校验值、环境摘要
- [ ] 拒绝覆盖已完成的输出目录
- [ ] 提供完整命令行供用户在网页粘贴

---

## 10. CODE_OVERVIEW.md 必需内容

你需要在 `../../code/CODE_OVERVIEW.md` 中记录：

### 10.1 模块结构
- 目录树
- 每个模块的职责
- 模块间依赖关系

### 10.2 接口说明
- 数据格式（JSONL schema）
- 模型输入输出
- 损失函数接口
- 训练/评估接口

### 10.3 数据流
- 数据预处理 → 训练 → 评估 → 聚合
- 每个阶段的输入输出

### 10.4 服务器任务清单
- 每个阶段的脚本名称
- 完整命令行（用户在网页粘贴）
- 预期输入输出
- 验收条件

### 10.5 配置说明
- 所有超参数及其来源（预实验锁定 vs 固定）
- 配置文件层次（base + method-specific）
- 如何修改配置

### 10.6 待服务器验证项
- L_PE 数值/梯度正确性
- 分母稳定性
- 标签隔离
- 其他单元测试项

### 10.7 代码版本
- 静态校验值（如 git commit hash）
- 或明确标注"无版本控制"

---

## 11. 完成标准与交接

### 11.1 代码实现完成标准

- [ ] 所有源码、配置、脚本已编写
- [ ] `CODE_OVERVIEW.md` 已完整记录
- [ ] 每个阶段的服务器执行命令已明确
- [ ] 本地约束已遵守（未安装/导入/运行）
- [ ] 待服务器验证项已明确标注

### 11.2 用户确认

**代码实现完成后，你需要**：
1. 向用户展示 `CODE_OVERVIEW.md`
2. 逐项说明实现要点
3. **明确请求用户确认**："当前代码版本可以提交服务器"

**只有用户明确确认后**，才能解锁 `SERVER_EXECUTION` 阶段。

### 11.3 服务器执行门禁

```
✅ 理论通过
⏳ 实验设计待明确通过 → 才能解锁 CODE_IMPLEMENTATION
🔒 代码实现与交接未开始
🔒 SERVER_EXECUTION 无授权
```

---

## 12. 关键文件路径

**理论**：
- `../theory/current_theory.md` v0.2

**实验设计**：
- `current_experiment.md` v0.3 MVP（本文件的依据）

**代码目录**：
- `../../code/`（你的工作目录）
- `../../code/CODE_OVERVIEW.md`（你需要创建）

**服务器要求**：
- `../../../machine/AGENTS.md`（服务器使用规则）
- `../../../machine/A800_standard_MVP_runbook.md`（运行手册模板）

**C_ε 观测工具**：
- `/Users/rarp/Desktop/ICLR/SOPPO/code/observe/LLM-output-density/GetSlice/`
- `/Users/rarp/Desktop/ICLR/SOPPO/code/observe/AGENTS.md`（C_ε 定义与使用）

---

## 13. 常见问题

**Q1：服务器具体路径未知怎么办？**
A：使用占位符（如 `<DATA_PATH>`, `<MODEL_PATH>`, `<EXP_PATH>`）或环境变量，在 `CODE_OVERVIEW.md` 中说明"待服务器穿透完成后确定"。

**Q2：L_PE 实现不确定是否正确？**
A：在代码中实现，标注"待服务器单元测试验证"，单元测试会验证数值和梯度正确性。

**Q3：本地无法测试代码？**
A：这是预期的。你只编写代码，不运行。所有测试在服务器完成。

**Q4：阶段间如何暂停？**
A：每个阶段是独立脚本，用户手动决定何时运行下一个阶段的脚本。在 `CODE_OVERVIEW.md` 中说明"每个阶段完成后，用户查看结果并决定是否继续"。

**Q5：如何处理 OOM？**
A：在代码中实现应急方案（gradient checkpointing, 调整 batch size），在 `CODE_OVERVIEW.md` 中说明触发条件和调整步骤。

---

## 14. 门禁解锁后的步骤

以下步骤当前锁定，只有实验设计 v0.3 MVP 获用户明确通过后才能开始：

1. **阅读已批准的完整实验设计**：`current_experiment.md` v0.3 MVP
2. **设计代码架构**：模块划分、接口设计
3. **编写代码**（本地纯文本编辑，不运行）：
   - 数据处理模块
   - 模型与损失模块（重点：L_PE 实现）
   - 训练与评估模块
   - 服务器脚本（00-08）
   - 配置文件
4. **编写 CODE_OVERVIEW.md**：完整记录实现细节
5. **向用户确认**："当前代码版本可以提交服务器"

**重要**：当前仍在实验讨论阶段；遵守本地硬约束，不安装、不导入、不运行，也不把预写文件称为已批准实现。
