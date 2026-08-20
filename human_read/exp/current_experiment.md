# 当前实验设计：有限偏好监督下的 Label-Encoding 结构监督验证

## 版本信息

- **Cycle ID**：`cycle-20260818-01`
- **实验版本**：**v0.3 MVP**（真正的最小可行实验）
- **当前阶段**：`EXP_DISCUSSION`
- **状态**：**待用户最终确认与通过**
- **理论依据**：`../theory/current_theory.md` v0.2（已通过，2026-08-19）
- **实验设计确认日期**：待确认
- **下一阶段**：`CODE_IMPLEMENTATION`（当前锁定；只在本设计明确通过后解锁）
- **代码交接状态**：未开始
- **服务器执行授权**：无（需实验设计通过 → 完成代码实现与说明 → 用户明确确认代码可提交服务器）
- **最近更新**：2026-08-19（v0.3：简化为真正的 MVP）
  - **单种子实验**（seed=42），不做 3 种子重复
  - C_ε 的 s_nsamples 和 x_nsamples 与 batch_size 一致（128）
  - **阶段间可暂停反馈**：每个阶段完成后可与用户讨论再继续
  - 调整服务器使用方式（网页命令行模式）
  - 增加数值稳定性预实验
- **证据等级**：实验设计阶段，尚无执行结果
- **代码阶段入口**：`../../code/CODE_OVERVIEW.md`
- **服务器运行手册**：`../../../machine/A800_standard_MVP_runbook.md`

---

## 1. 核心研究问题

基于理论 v0.2，本实验验证：

> **在偏好标签有限时，使用 population-level label-encoding 结构监督（LERM 风格）替代 instance-level pseudo target，能否改善 preference learning？**

### 1.1 关键对比

```
传统方法：z̃_i = 1[p_i > 0.5]  →  instance-level pseudo target
我们的方法：L_PE = ||ê_+ - e_+||₁ + ||ê_- - e_-||₁  →  population-level structural supervision
```

### 1.2 可证伪假设

| 编号 | 假设 | 可观察预测 | MVP 判断标准 |
| --- | --- | --- | --- |
| **H1** | 少量真实偏好对 + 结构监督 > 仅少量真实偏好对 | DPO+PE 的 Acc 高于 DPO-10%，Brier 不恶化 | Acc 提升 >2%，单种子观测趋势 |
| **H2** | 结构监督 > 逐样本伪目标 | DPO+PE 优于 Pseudo-target | Acc 提升 >2%，单种子观测趋势 |
| **H3** | 收益来自结构组织，非仅置信度变化 | Acc 与 Brier 同时改善，c_ε 显示功能压缩 | 诊断指标支持结构学习假设 |

---

## 2. 实验范围与边界

### 2.1 第一阶段 MVP 边界

本实验专注**第一阶段：静态无标签偏好对 MVP**，验证结构监督本身的有效性。

**包含**：
- 静态 preference dataset（UltraFeedback），人为隐藏大部分标签
- 4 组对照：DPO-10%, Pseudo-target, DPO+PE, DPO-100%
- Label-encoding 结构损失 L_PE
- 固定模型、固定数据、固定超参数预注册
- **λ scheduler 消融**：Fixed / Linear-Warmup / Exp-Warmup
- **单种子 MVP**（seed=42）：验证方法可行性，不做统计显著性检验

**不包含**（留待第二阶段或后续扩展）：
- On-policy sampling
- SFT rollout（动态候选对生成）
- Token-level 扩展
- AlpacaEval 等生成质量评估
- 多种子重复与统计显著性检验（MVP 后扩展）

### 2.2 成功标准（MVP 单种子）

**最低有效性**：H1 成立（DPO+PE > DPO-10%）
**核心贡献**：H2 成立（DPO+PE > Pseudo-target）
**诊断验证**：H3 成立（不仅是置信度效应）

**MVP 判断准则**：
- 单种子（seed=42）观测趋势，不做统计显著性检验
- Acc 提升 >2% 视为有效信号（实质差异）
- 同时检查 Brier、responsibility 质量、c_ε 等诊断指标
- **MVP 成功后**：扩展到 3 种子重复，进行正式统计检验

**失败撤退**：
- 触发条件：H1 不成立（DPO+PE ≤ DPO-10% 或提升 <2%）
- 时间盒：1-2 周诊断分析
- 撤退决策：不进入第二阶段，承认静态结构监督无效

---

## 3. 阶段 -1：环境准备与配置

### 3.0 阶段执行模式

**重要**：本实验采用**阶段间暂停反馈**模式：
- 每个阶段完成后，向用户报告结果
- 用户审核、讨论、提出调整
- 用户明确确认后，才进入下一阶段
- **示例**：阶段 -1 完成后，用户可能需要多次讨论环境配置，确认无误后才进入阶段 0

**执行流程**：
```
阶段 -1 完成 → 报告结果 → 用户讨论与确认 → 进入阶段 0
阶段 0 完成 → 报告结果 → 用户讨论与确认 → 进入阶段 1
...以此类推
```

### 3.1 目标

在预实验前完成：
1. 服务器环境检查与依赖安装
2. 数据集下载与验证
3. 模型下载与加载测试
4. 代码单元测试

### 3.2 服务器使用方式

- **集群地址**：`172.16.11.78:32206`（需 VPN/内网穿透）
- **账号**：`jiangwenhaoT`
- **访问方式**：网页界面（**不使用 SFTP**）
- **任务提交**：通过网页复制粘贴命令行
- **镜像**：`pytorch2.4.0-cu121-zhaoyuhangc:v2.0`
- **资源**：A800 × 16（任务独立调度，不强制单任务 16 卡）

**代码阶段准则**：
1. 所有任务编写为 `.sh` 或 `.py` 脚本
2. 每个任务提供完整的命令行（用户在网页粘贴执行）
3. 结果存储在指定文件夹（路径在命令行中指定）
4. 不依赖 SFTP 上传流程

### 3.3 服务器环境与路径

**重要说明**：
- 服务器具体路径、文件夹结构、环境配置将在 `CODE_IMPLEMENTATION` 阶段详细确定
- 实际服务器环境信息将记录在 `/Users/rarp/Desktop/ICLR/machine/` 中
- 本阶段只列出逻辑存储需求，不预设具体路径

**逻辑存储需求**：
| 内容 | 用途 | 说明 |
| --- | --- | --- |
| 源码 | 脚本与配置文件 | 通过网页上传或直接在服务器编辑 |
| 环境 | Python 虚拟环境与依赖 | 服务器创建，不回传本地 |
| 缓存 | HuggingFace/ModelScope 下载缓存 | 避免重复下载 |
| 数据 | UltraFeedback 原始与处理后数据 | 包含 private_labels |
| 模型 | Qwen3-4B checkpoint | `<SERVER_BASE>/models/Qwen3-4B` |
| 实验产物 | Logs, checkpoints, metrics | 按 experiment_id/method/seed 组织 |

**路径状态**：目录合同已统一为 `<SERVER_BASE>/ICLR` 与 `<SERVER_BASE>/SOPPO` 平级；实际目录、模型与挂载是否存在，待获得服务器执行授权后核验并记录。

### 3.4 数据集准备

#### 数据源
- **数据集**：HuggingFace `openbmb/UltraFeedback`
- **规模**：10k 子集，随机采样
- **划分**：1k (labeled) : 8k (unlabeled) : 1k (test)
- **Validation**：从 labeled 1k 中再划出 100 样本（10%）

#### 预处理要求
1. **随机采样**：从完整 UltraFeedback 随机采样 10k 样本
2. **去重**：按 prompt 或等价内容去重
3. **划分策略**：确保同一 prompt 的不同回复对不跨集合
4. **标签隐藏**：
   - 对 D_U（8k unlabeled）和 D_test（1k test），只保留 `(x, response_a, response_b)`
   - 真实标签存入 `private_labels/`，训练代码不可访问
   - 随机交换 response_a/response_b 位置，避免 chosen/rejected 字段泄漏
5. **统计审计**（验证标签隐藏）：
   - 检查 response_a/response_b 位置分布（应接近 50:50）
   - 检查长度、格式等浅层特征与真实标签的相关性
   - 输出审计报告：`manifest_public.json`（可回传）和 `manifest_private.json`（服务器保留）

#### 数据产物合同

服务器端生成并冻结：
```
<SERVER_BASE>/data/ultrafeedback/<revision>/
├── labeled_train.jsonl        # 900 样本
├── labeled_val.jsonl          # 100 样本（从 1k labeled 划出）
├── unlabeled_train.jsonl      # 8k 样本，只含 (x, y_a, y_b)，位置随机
├── test_inputs.jsonl          # 1k 样本，只含 (x, y_a, y_b)，位置随机
├── private_labels/
│   ├── unlabeled_labels.jsonl # 8k 真实标签（训练代码禁止访问）
│   └── test_labels.jsonl      # 1k 真实标签（仅评估代码访问）
├── manifest_private.json      # 样本级元信息（禁止回传）
└── manifest_public.json       # 汇总统计与审计结果（可回传）
```

**审计要求**：
- 集合互斥性检查（prompt 不跨集合）
- 位置随机化验证
- 长度/格式分布与标签相关性检查
- SHA-256 校验值
- 失败时停止下游任务

### 3.5 模型准备

#### 模型规格
- **模型**：Qwen3-4B（Instruct 版本）
- **参数量**：~4B
- **下载方式**：`modelscope download --model Qwen/Qwen3-4B`
- **服务器路径合同**：`<SERVER_BASE>/models/Qwen3-4B`（实际存在性待授权后核验）
- **Reference 模型**：与训练模型共享权重，训练时冻结

#### 序列长度配置
- **max_seq_len**：2048
- **预处理**：截断超长样本
- **统计**：在数据审计中记录被截断样本比例（预期 <5%）

### 3.6 依赖与环境

#### 核心依赖
- PyTorch：2.4.0
- CUDA：12.1
- transformers：>=4.40.0
- datasets：>=2.18.0
- accelerate：>=0.28.0
- wandb（可选）：实验跟踪

#### 环境创建
服务器任务：`scripts/cluster/00_server_setup.sh`
- 创建路径环境：`<SERVER_BASE>/envs/youc`
- 安装锁定依赖：`requirements.lock.txt`
- 检查：代码/挂载/磁盘/网络/驱动/PyTorch-CUDA/HuggingFace 访问
- 输出：`server_preflight.json`（环境摘要）

#### 代码单元测试
服务器任务：`scripts/cluster/01_server_tests.sh`
- L_PE 数值/梯度正确性
- 分母稳定性（Σp_i 和 Σ(1-p_i) 不为零）
- 梯度路径（是否穿过 responsibility p_i）
- 标签隔离（训练代码无法读取 `private_labels/`）
- 回复交换（位置随机化）
- 配置合并（5 组方法共享 backbone）
- 输出拒绝覆盖（已存在实验目录时报错）

**验收**：测试全部通过后才进入预实验。

---

## 4. 阶段 0：预实验

### 4.1 目标

1. **数值稳定性测试**：测试 L_PE 中的数值稳定项 ε ∈ {1e-5, 1e-6, 1e-8}，选择最优值
2. **Headroom 验证**：确认 DPO-100% vs SFT baseline 有 ≥5% Acc 提升空间
3. **β 选择**：在 {0.1, 0.5} 中选择更优值
4. **学习率选择**：在 {5e-7, 1e-6} 中选择更优值
5. **显存验证**：确认 batch_size=128 在 2×A800 上可行

### 4.2 实验设置

#### 数据
- 使用完整 1k labeled + 8k unlabeled
- 只跑 1 个随机种子（seed=42）

#### 对照组
1. **数值稳定性测试**：
   - 在 DPO+PE 上测试 ε ∈ {1e-5, 1e-6, 1e-8}
   - 观测训练稳定性、是否出现 NaN/Inf、分母退化
   - 选择最稳定的 ε 值用于后续实验
   
2. **SFT baseline**：在 labeled 1k 上用 chosen 回复做监督微调

3. **DPO-100%**：在完整 9k 样本（恢复 unlabeled 的真实标签）上训练 DPO

#### 超参数候选
- **ε（数值稳定项）**：{1e-5, 1e-6, 1e-8}
- **β**：{0.1, 0.5}
- **lr**：{5e-7, 1e-6}
- 交叉测试（数值稳定性确定后）：2×2=4 组 DPO-100% 配置

#### 训练配置
- Epochs：2
- Batch size：global=128, per_device=8, grad_accum=8, devices=2
- Optimizer：AdamW (β1=0.9, β2=0.999, weight_decay=0.01)
- Warmup：10% steps
- 评估频率：每 20 steps

### 4.3 成功标准

**数值稳定性**：
- 选择训练过程中无 NaN/Inf、分母不退化的 ε 值
- 锁定用于主实验

**Headroom**：
- DPO-100%（最优配置）的 Acc > SFT baseline Acc + 5%
- 如果 headroom < 5%：换 Qwen2.5-1.5B 或选择更难的 UltraFeedback 子集

**超参数选择**：
- 在 validation set（100 样本）上选择 Acc 最高的 (β, lr) 组合
- 锁定用于主实验

**显存验证**：
- 确认 global_batch=128 在 2×A800 上不 OOM
- 如果 OOM：启用 gradient checkpointing 或调整为 per_device=4, grad_accum=16

### 4.4 产物

- 预实验报告：`pre_experiment_report.md`
- 锁定超参数：ε, β, lr
- 显存 profile：`memory_profile.json`
- Headroom 验证：DPO-100% vs SFT baseline Acc 差值

---

## 5. 阶段 1：主实验

### 5.1 实验设置

#### 4 组对照方法

**1. DPO-10%**（有限标签下界）
- 数据：仅使用 D_L（900 labeled training + 100 validation）
- 损失：标准 DPO loss
- 目的：建立有限标签 baseline

**2. Pseudo-target**（逐样本伪目标）
- 数据：D_L（900 labeled）+ D_U（8k unlabeled）
- 训练流程（每个训练步）：
  1. 从 D_L 采样 batch_L，计算 L_DPO(D_L)
  2. 从 D_U 采样 batch_U
  3. 用当前 θ 计算 p_i = σ(Δ_i)
  4. 生成硬伪标签：z̃_i = 1[p_i > 0.5]
  5. 根据 z̃_i 确定 (y_w, y_l)：
     - 如果 z̃_i = 1，则 y_w = y_1, y_l = y_2
     - 如果 z̃_i = 0，则 y_w = y_2, y_l = y_1
  6. 用伪标签对计算 L_DPO_pseudo(D_U)
  7. 反向传播：L = L_DPO(D_L) + λ L_DPO_pseudo(D_U)
- 实现参考：SSPO 风格（硬伪标签，每步实时生成）
- 目的：instance-level pseudo target baseline

**3. DPO+PE**（本方法：群体结构监督）
- 数据：D_L（900 labeled）+ D_U（8k unlabeled）
- 训练流程（每个训练步）：
  1. 从 D_L 采样 batch_L，计算 L_DPO(D_L)
  2. 从 D_U 采样 batch_U
  3. 用当前 θ 计算 p_i = σ(Δ_i)
  4. 估计条件编码：
     - ê_+ = Σ_i p_i q_i / Σ_i p_i
     - ê_- = Σ_i (1-p_i) q_i / Σ_i (1-p_i)
     - 其中 q_i = [p_i, 1-p_i]ᵀ
  5. 计算结构损失：
     - L_PE = 0.5 × (||ê_+ - e_+||₁ + ||ê_- - e_-||₁)
     - 理想编码：e_+ = [1,0]ᵀ, e_- = [0,1]ᵀ
  6. 反向传播：L = L_DPO(D_L) + λ L_PE(D_U)
- 梯度路径：初始方案让梯度穿过 p_i（不使用 stop_gradient）
  - 应急方案：如果训练不稳定，对分母 Σp_i 使用 detach
- 目的：population-level structural supervision

**4. DPO-100%**（oracle 上界）
- 数据：恢复 D_U 的真实标签，使用完整 9k labeled 样本
- 损失：标准 DPO loss
- 目的：oracle upper bound（非竞争 baseline，用于判断结构方法与完整标签的差距）

#### 公平性约束

所有方法必须共享：
- 相同初始化（Qwen3-4B Instruct checkpoint）
- 相同 reference 模型（冻结的 Qwen3-4B Instruct）
- 相同优化器配置（AdamW, β1, β2, weight_decay）
- 相同学习率、batch size、训练步数
- 相同数据划分（D_L, D_U, D_test 的样本）
- 相同评估协议（validation 频率、指标计算）

唯一差异：
- 训练数据使用范围（DPO-10% 仅用 D_L，其他用 D_L + D_U）
- 损失函数（DPO vs DPO+Pseudo vs DPO+PE）

### 5.2 训练配置

#### 固定超参数
- **ε（数值稳定项）**：从预实验选出（1e-5, 1e-6 或 1e-8）
- **β**：从预实验选出（0.1 或 0.5）
- **lr**：从预实验选出（5e-7 或 1e-6）
- **Epochs**：2-3（根据预实验收敛情况确定）
- **Batch size**：global=128, per_device=8, grad_accum=8, devices=2
- **Optimizer**：AdamW (β1=0.9, β2=0.999, weight_decay=0.01)
- **Warmup**：10% steps
- **Max seq len**：2048
- **随机种子**：**42**（MVP 单种子）

#### λ（PE loss weight）配置
- **搜索空间**：{0.1, 0.3, 0.5, 1.0}
- **搜索策略**：
  1. 对每个 λ，在 validation set（100 样本）上训练（Fixed-λ，seed=42）
  2. 选择 validation Acc 最高的 λ*
  3. 用 λ* 测试 3 种 scheduler（Fixed, Linear-Warmup, Exp-Warmup）
- **Pseudo-target 的 λ**：与 DPO+PE 使用相同值（公平对比）

#### λ Scheduler（MVP 主实验）
对 DPO+PE 和 Pseudo-target，测试 3 种 scheduler（seed=42）：

1. **Fixed-λ**：
   ```
   λ(t) = λ*  (constant)
   ```

2. **Linear-Warmup**：
   ```
   λ(t) = λ* × min(1, t / t_warmup)
   其中 t_warmup = 10% × total_steps
   ```
   - 前 10% steps：λ 从 0 线性增长到 λ*
   - 后 90% steps：λ = λ*

3. **Exponential-Warmup**：
   ```
   λ(t) = λ* × (1 - exp(-5 × t / total_steps))
   ```

**实验组合（MVP 单种子）**：
- 2 方法（DPO+PE, Pseudo-target）× 3 schedulers = 6 次训练
- DPO-10% 和 DPO-100%（不涉及 λ）= 2 次训练
- **总计**：8 次训练（MVP 主实验）

### 5.3 训练流程（MVP 单种子）

每个方法 × 每个 scheduler = 独立的 1-GPU 任务：
- DPO+PE 和 Pseudo-target：各 3 schedulers = 6 次训练
- DPO-10% 和 DPO-100%：各 1 次 = 2 次训练
- **总计**：8 个独立任务
- 资源：每任务 1×A800
- 调度：由平台按可用资源并行或排队

每个任务的产物：
```
<服务器实验路径>/<METHOD>_<SCHEDULER>/seed-42/
├── config.yaml              # 冻结配置
├── checkpoints/             # 10 个 checkpoints（每 20 steps）
│   ├── step_020/
│   ├── step_040/
│   ...
│   └── step_best/          # validation Acc 最高的 checkpoint
├── logs/
│   ├── training.log        # 训练日志
│   └── metrics.jsonl       # 逐步指标（loss, Acc, Brier, 诊断信息）
├── predictions/
│   └── test_predictions.jsonl  # 测试集逐样本预测（服务器保留）
└── diagnostics/
    ├── responsibility_dist.json    # p_i 分布统计
    ├── encoding_estimates.json     # ê_+, ê_- 轨迹
    └── c_epsilon/                  # C_ε 观测结果（见阶段 3）
```

### 5.4 评估与监控

#### 评估频率
- **Validation**：每 20 steps（共约 10 个 checkpoints，根据总步数调整）
- **Test**：仅在训练结束后，使用 best checkpoint

#### 核心指标

**Preference Accuracy**：
```
Acc = (1 / |D_test|) × Σ_i 1[(p_i > 0.5) = z_i]
```

**Brier Score**：
```
Brier = (1 / |D_test|) × Σ_i (p_i - z_i)²
```

#### 诊断信息（每个训练步记录）

1. **Responsibility 质量**：
   - Σ_i p_i 和 Σ_i (1-p_i) 的均值与方差
   - p_i 分布的直方图（bins=[0-0.1, 0.1-0.2, ..., 0.9-1.0]）

2. **编码估计**（仅 DPO+PE）：
   - ê_+ 和 ê_- 的值（每步）
   - ||ê_+ - e_+||₁ 和 ||ê_- - e_-||₁

3. **伪标签统计**（仅 Pseudo-target）：
   - 每步的伪标签比例（正类 vs 负类）
   - 伪标签翻转率（与上一步相比）

4. **Loss 分解**：
   - L_DPO(D_L)
   - L_PE(D_U) 或 L_DPO_pseudo(D_U)
   - Total loss

5. **梯度统计**：
   - 梯度范数（各层）
   - 是否出现 NaN/Inf

#### Checkpoint 保存策略
- **训练期间**：每 20 steps 保存 checkpoint（共约 10 个）
- **C_ε 观测期间**：保留所有 checkpoints（用于观测）
- **C_ε 观测完成后**：
  - **只保留**：best checkpoint（validation Acc 最高）
  - **删除**：所有中间 checkpoints（节省 ~90% 存储空间）
- **Early stopping**：不使用（观测完整训练轨迹）

---

## 6. 阶段 2：λ 超参数搜索

### 6.1 目标

为 DPO+PE 和 Pseudo-target 选择最优 λ。

### 6.2 搜索空间

λ ∈ {0.1, 0.3, 0.5, 1.0}

### 6.3 搜索策略

#### 数据划分
- Training：900 labeled (D_L) + 8k unlabeled (D_U)
- Validation：100 labeled（从原 1k labeled 划出）
- Test：1k（不参与 λ 选择，只用于最终评估）

#### 流程
1. 对每个 λ 候选值：
   - 训练 DPO+PE 和 Pseudo-target（各 1 个种子，seed=42）
   - 在 validation set 上评估 Acc
2. 选择 validation Acc 最高的 λ*
3. 用 λ* 在主实验（阶段 1）中训练 3 个种子

#### 避免过拟合
- λ 选择**仅基于 validation set**
- Test set 在 λ 选择过程中保持隔离
- 不允许根据 test set 结果调整 λ

### 6.4 产物

- λ 搜索报告：`lambda_search_report.md`
- 选出的 λ*：记录在 `config_final.yaml`
- Validation 曲线：`lambda_validation_curve.png`

---

## 7. 阶段 3：C_ε 相对功能压缩观测

### 8.1 目标

观测不同方法训练后的模型，在偏好表示上的功能方向压缩。

### 8.2 C_ε 定义

基于 `observe/LLM-output-density` 的 activation-aware whitening：

**功能秩**：
```
r_ε(A) = min k such that Σ_{i>k} σ_i² / Σ_i σ_i² ≤ ε
其中 A_{D,t,m} = W_{t,m} S_{D,t,m}
```

**相对压缩**：
```
c_{ε,D,t,m} = (r_{ε,D,0,m} - r_{ε,D,t,m}) / r_{ε,D,0,m}
```

**解释**：
- c_ε > 0：相对 base 压缩，需要的功能方向变少
- c_ε = 0：不变
- c_ε < 0：膨胀，功能方向变多

### 8.3 观测策略（第一阶段 MVP）

#### Checkpoint 选择
- **Base**：初始模型（Qwen3-4B Instruct，checkpoint_0）
- **训练中**：
  - 前 3 个连续：ckpt_1, ckpt_2, ckpt_3
  - 后 7 个间隔一个：ckpt_5, ckpt_7, ckpt_9, ...（共 10 个 ckpt）

#### Probe 配置
- **S（Source）**：D_L（labeled training set，900 样本）
- **X（Target）**：仅 **X_S** = D_U（unlabeled set，8k 样本，静态数据）
- **第二阶段才引入 X_O**（on-policy rollout）

#### ε 阈值
- **Headline**：ε = 0.05（保留 95% 能量）
- **稳健性检查**：{0.01, 0.025, 0.05, 0.10}

#### Module 聚合
- **Equal-5 non-QK**（主口径）：v, o, gate, up, down
- **Equal-7**（辅助）：q, k, v, o, gate, up, down

#### 层选择
- **Headline layer**：待确认（通常选择中间层，如 layer_14 for 28-layer model）
- **全层**：如果计算资源允许

### 8.4 实现路径

#### 数据准备
1. **S 侧 JSONL**（labeled training）：
```json
{"question": "...", "answer": "..."}
```

2. **X 侧 JSONL**（unlabeled set）：
```json
{"output": {"text": "response_a or response_b ..."}}
```

#### GetSlice 运行
```bash
cd <SERVER_BASE>/SOPPO/code/observe/LLM-output-density/GetSlice
python slice.py --config config_<method>_<seed>_<ckpt>.json
```

**配置示例**：
```json
{
  "model": "<服务器模型路径>/checkpoints/step_020",
  "save_path": "<服务器实验路径>/diagnostics/c_epsilon/step_020",
  "mode": "split_whitened_svd",
  "DEV": "cuda",
  "model_seq_len": 2048,
  "seed": 3,
  "target_layer": 14,
  "s_nsamples": 128,
  "s_jsonl_path": "<服务器数据路径>/labeled_train.jsonl",
  "s_jsonl_file": "labeled_train.jsonl",
  "x_nsamples": 128,
  "x_jsonl_path": "<服务器数据路径>/unlabeled_for_x.jsonl",
  "cleanup_intermediate": false,
  "save_profile_s_path": "profile_s.pt",
  "save_profile_x_path": "profile_x.pt",
  "save_s_json_path": "sMat_S.json",
  "save_x_json_path": "xMat_X_S.json"
}
```

**注意**：
- **s_nsamples 和 x_nsamples 设置为 128**（与 batch_size 一致）
- 具体路径在代码阶段确定
```

#### 后处理
使用 `observe/LLM-output-density/experiments/opd_sft_h1/scripts/`:
```bash
python cycle09_relative_functional_contraction.py --phase derive
python cycle09_equal5_non_qk.py
```

输出：
```
relative_functional_contraction_module_audit.csv
relative_functional_contraction_all_cells.csv
relative_functional_contraction_equal5.csv
```

### 8.5 对比分析

#### 方法间对比
- **DPO-10%** vs **DPO+PE** vs **Pseudo-target** vs **DPO-100%**
- 假设：DPO+PE 应表现出更强的功能压缩（c_ε 更高）

#### 训练轨迹
- 观测 c_ε 随训练步数的变化
- 检查是否存在过度压缩（c_ε 过高）或膨胀（c_ε < 0）

#### 与性能关联
- c_ε 与 Acc/Brier 的相关性
- 验证 H3：收益来自结构组织，非仅置信度变化

### 8.6 产物

```
<SERVER_BASE>/runs/exp-20260819-01-mvp/<METHOD>/seed-<SEED>/diagnostics/c_epsilon/
├── step_020/
│   ├── sMat_S.json
│   ├── xMat_X_S.json
│   ├── profile_s.pt
│   ├── profile_x.pt
│   └── c_epsilon_summary.csv
├── step_040/
...
└── c_epsilon_trajectory.csv       # 聚合所有 checkpoints
```

**可回传**：
- `c_epsilon_trajectory.csv`（方法/种子级聚合）
- `c_epsilon_comparison.png`（方法间对比图）

**服务器保留**：
- 原始奇异值 JSON
- Profiling matrices `.pt`
- Module 级详细表

---

## 9. 评估指标与成功标准

### 9.1 核心指标

#### Preference Accuracy
```
Acc = (1 / |D_test|) × Σ_i 1[(p_i > 0.5) = z_i]
```
- **主要成功标准**
- 3 个种子的均值与标准差
- Paired bootstrap 置信区间（95%）

#### Brier Score
```
Brier = (1 / |D_test|) × Σ_i (p_i - z_i)²
```
- **校准质量**
- 确保 Acc 提升不是以牺牲校准为代价

### 9.2 诊断指标

#### Responsibility 质量
- Σ_i p_i 和 Σ_i (1-p_i) 的分布
- 检测退化（所有 p_i 接近 0.5 或 1）

#### 预测分布
- p_i 的直方图（检测坍缩）
- 熵：H = -Σ_i [p_i log p_i + (1-p_i) log(1-p_i)]

#### 编码质量（仅 DPO+PE）
- ||ê_+ - e_+||₁ 和 ||ê_- - e_-||₁
- 训练轨迹：是否单调下降

#### C_ε 相对功能压缩
- c_ε (equal-5 non-QK)
- 方法间对比：DPO+PE 是否表现出更强压缩

### 9.3 成功标准（MVP 单种子）

#### H1：最低有效性
**条件**：
- DPO+PE 的 Acc > DPO-10% 的 Acc + 2%
- Brier 不恶化（差异 ≤1% 或改善）
- 单种子观测趋势，不做统计显著性检验

**失败**：
- DPO+PE 与 DPO-10% 无实质差异（提升 ≤2%）或更差
- 触发 1-2 周诊断分析时间盒
- 撤退：不进入第二阶段

#### H2：核心贡献
**条件**：
- DPO+PE 的 Acc > Pseudo-target 的 Acc + 2%
- Brier 不恶化
- 单种子观测趋势

**失败**：
- 只能说明结构正则可能有帮助
- 不能支撑"优于 instance-level target"的核心论点

#### H3：机制验证
**条件**：
- Acc 与 Brier 同时改善
- c_ε 显示功能压缩（c_ε > 0）
- Responsibility 质量稳定（不退化）
- 预测分布不坍缩（熵不过低）

**失败**：
- 仅置信度下降（熵降低但 Acc 无提升）
- 准确率或校准恶化
- 记为"机制未确认"

### 9.4 MVP 后扩展

**如果 MVP 成功**（H1 和 H2 成立）：
1. 扩展到 3 个随机种子（42, 43, 44）
2. 进行正式统计显著性检验（paired t-test, bootstrap）
3. 报告统计显著性与实质差异

---

## 10. 资源预算与风险

### 10.1 计算资源

#### GPU 时间估算
**单次训练**（1 方法 × 1 种子）：
- 数据量：9k 样本
- Batch size：128
- Epochs：2-3
- Steps per epoch：~70（9k / 128）
- 总步数：140-210 steps
- 每 step 时间：~30-60s（包括前向、反向、优化）
- **单次训练时间**：~2-4 小时（1×A800）

**主实验**（含 λ scheduler 消融，MVP 单种子）：
- 2 方法（DPO+PE, Pseudo-target）× 3 schedulers = 6 次训练
- DPO-10% 和 DPO-100%：2 次训练
- 总计：8 次训练
- 总 GPU 时间：16-32 GPU·小时（1×A800）
- 并行调度：如果 8 张 A800 可用，~2-4 小时完成

**预实验**：
- 数值稳定性测试（3 个 ε）：~3 小时
- SFT baseline：~1 小时
- DPO-100%（4 组超参数）：~4-8 小时
- 总计：~8-12 GPU·小时

**λ 搜索**：
- 4 个 λ × 2 方法 × 1 种子 = 8 次训练
- 总计：~16-32 GPU·小时

**C_ε 观测**（MVP 单种子）：
- 8 次训练 × 10 ckpts = 80 次 GetSlice 运行
- 每次 ~10-20 分钟（**s_nsamples=128**）
- 总计：~13-27 GPU·小时

**总预算**（MVP，含所有阶段）：
- **保守估计**：~53-103 GPU·小时（A800）
- **并行调度**：如果 8 张 A800 可用，1 天内完成

### 10.2 存储需求（MVP 单种子）

#### Checkpoints
- 单个 checkpoint（Qwen3-4B）：~8GB
- **训练期间**：每个训练 10 ckpts × 8GB = 80GB
- **C_ε 观测完成后**：每个训练仅保留 best ckpt = 8GB
- 8 次训练（MVP 主实验）：
  - 训练期间峰值：~640GB
  - **最终保留**：8 × 8GB = **64GB**（节省 ~90% 存储空间）

#### 数据
- UltraFeedback 原始：~2GB
- 处理后（10k 子集）：~200MB
- C_ε 中间产物（奇异值 JSON + profiles）：~3GB（s_nsamples=128）

#### 日志与指标
- 每个训练：~100MB
- 8 次训练：~800MB

**总存储（MVP）**：
- **峰值**（训练+观测期间）：~650GB
- **最终保留**：~70GB（删除中间 checkpoints 后）

### 10.3 时间预算（MVP 单种子）

#### 环境准备（阶段 -1）
- 数据下载与预处理：~2-4 小时
- 模型下载：~1-2 小时（取决于网络）
- 环境配置与测试：~2-4 小时
- **总计**：~1 天
- **阶段间暂停**：完成后报告，用户确认再继续

#### 预实验（阶段 0）
- 数值稳定性测试 + 训练：~8-12 GPU·小时
- 分析与超参数锁定：~2-4 小时
- **总计**：~1 天
- **阶段间暂停**：完成后报告，用户确认再继续

#### 主实验（阶段 1 + 2，含 λ scheduler，MVP 单种子）
- λ 搜索：~16-32 GPU·小时
- 主实验训练（8 次）：~16-32 GPU·小时
- 并行调度：~1-2 天
- **总计**：~2-3 天
- **阶段间暂停**：完成后报告，用户确认再继续

#### C_ε 观测（阶段 3，MVP 单种子）
- GetSlice 运行（s_nsamples=128）：~13-27 GPU·小时
- 后处理与分析：~4-8 小时
- 并行调度：~1 天
- **总计**：~1 天
- **阶段间暂停**：完成后报告，用户确认再继续

#### 结果分析与报告
- 指标聚合：~2-4 小时
- 可视化：~2-4 小时
- 撰写报告：~1 天
- **总计**：~1-2 天

**总时间（MVP 端到端）**：~6-9 天（含阶段间讨论时间）

**MVP 成功后扩展到 3 种子**：
- 额外训练时间：~2-3 天
- 统计分析与报告：~1 天
- **扩展总时间**：~9-13 天

### 10.4 风险与应对

#### 风险 1：OOM（显存不足）
**触发条件**：batch_size=128, seq_len=2048 超出 2×A800 显存
**应对**：
1. 启用 gradient checkpointing
2. 降低 per_device_batch=4, grad_accum=16
3. 如仍失败，降至 global_batch=64
4. 记录偏差并在报告中说明

#### 风险 2：训练不稳定（L_PE 导致）
**触发条件**：
- Loss 出现 NaN/Inf
- Σp_i 或 Σ(1-p_i) 接近 0（分母退化）
- 梯度爆炸

**应对**：
1. 对分母 Σp_i 使用 detach（应急方案）
2. 添加数值稳定项：ε = 1e-8
3. 降低 λ（减弱结构损失）
4. 参考 DAPO/GRPO 的分层采样策略
5. 记录调整并在报告中说明

#### 风险 3：Headroom 不足
**触发条件**：DPO-100% vs SFT baseline Acc 差值 < 5%
**应对**：
1. 换 Qwen2.5-1.5B（更小模型，更难拟合）
2. 选择更难的 UltraFeedback 子集（高歧义样本）
3. 增加训练 epochs
4. 记录调整并重新运行预实验

#### 风险 4：H1 失败（结构监督无效）
**触发条件**：DPO+PE 与 DPO-10% 无显著差异或更差（3 种子一致）
**应对**：
1. 进入 1-2 周诊断分析时间盒
2. 检查：
   - L_PE 是否正确实现（梯度、数值）
   - Responsibility 质量（是否退化）
   - 编码估计轨迹（是否收敛）
   - λ 是否过小或过大
3. 尝试调整：
   - 不同 λ 范围
   - 不同结构距离（L_2 vs L_1）
   - 梯度路径（detach vs no detach）
4. 如无改善：撤退，承认静态结构监督在偏好学习中无效
5. 不进入第二阶段

#### 风险 5：服务器资源竞争
**触发条件**：A800 资源不足，任务长时间排队
**应对**：
1. 拆分任务为更小粒度（单方法单种子）
2. 优先运行关键路径（预实验 → 主实验）
3. C_ε 观测和消融可延后
4. 与用户沟通调整时间预算

---

## 11. 产物与交付

### 11.1 服务器产物（仅保留）

**数据**：
```
<SERVER_BASE>/data/ultrafeedback/<revision>/
├── labeled_train.jsonl
├── labeled_val.jsonl
├── unlabeled_train.jsonl
├── test_inputs.jsonl
└── private_labels/
```

**Checkpoints**：
```
<SERVER_BASE>/runs/exp-20260819-01-mvp/<METHOD>/seed-<SEED>/checkpoints/
├── step_best/       # 保留
└── step_last/       # 保留
# 其他中间 ckpts 在 C_ε 观测后可删除
```

**逐样本预测**：
```
<SERVER_BASE>/runs/exp-20260819-01-mvp/<METHOD>/seed-<SEED>/predictions/
└── test_predictions.jsonl  # 服务器保留，不回传
```

**C_ε 中间产物**：
```
<SERVER_BASE>/runs/exp-20260819-01-mvp/<METHOD>/seed-<SEED>/diagnostics/c_epsilon/
├── sMat_*.json
├── xMat_*.json
└── profile_*.pt
```

### 11.2 本地回传（白名单）

**聚合指标**：
```
/Users/rarp/Desktop/ICLR/SOPPO/exp/exp-20260819-01-mvp/export_local/
├── summary.json              # 方法/种子级聚合
├── metrics.csv               # 一行对应方法/种子
├── comparison.md             # 人类可读对比报告
└── figures/
    ├── acc_comparison.png
    ├── brier_comparison.png
    ├── responsibility_dist.png
    ├── encoding_trajectory.png
    └── c_epsilon_comparison.png
```

**配置快照**：
```
├── config_final.yaml         # 锁定的超参数
├── manifest_public.json      # 数据统计（无样本内容）
└── environment_summary.json  # 服务器环境摘要
```

**任务状态**：
```
├── task_registry.json        # 任务 ID、状态、路径索引
└── server_paths.txt          # 服务器产物路径清单
```

### 11.3 禁止回传（黑名单）

- 原始或处理后数据（含 prompt/response 文本）
- 私有标签（`private_labels/`）
- 模型权重、checkpoints、adapters、optimizer states
- Tokenized cache、embeddings
- 逐样本预测或逐样本指标
- 包目录、虚拟环境、wheel、依赖 cache
- 可能包含样本文本的原始日志

### 11.4 结果报告

**主报告**：`/Users/rarp/Desktop/ICLR/SOPPO/human_read/result/current_result.md`
- 实验设计回顾
- 核心发现（H1, H2, H3 是否成立）
- 方法对比（Acc, Brier, c_ε）
- 训练轨迹分析
- 诊断信息解读
- 失败模式分析（如果有）
- 对第二阶段的建议

**技术附录**：
- 超参数搜索结果
- 统计检验详情
- C_ε 完整表格
- 显存 profile
- 失败案例分析

---

## 12. 门禁与执行授权

### 12.1 前置条件

完整路径包含以下四个顺序门禁，不能把实验批准、代码交接和服务器授权合并：

1. ✅ **理论批准**：`../theory/current_theory.md` v0.2 已获用户明确通过（2026-08-19）
2. ⏳ **实验设计批准**：本文档需用户逐项审核并明确通过
3. 🔒 **代码实现完成**：在 `../../code/` 将获批设计转化为源码、配置和服务器脚本，并完整更新 `../../code/CODE_OVERVIEW.md`
4. 🔒 **代码交接与服务器授权**：用户明确确认当前代码版本可以提交服务器；此前禁止 SFTP 上传、创建实验目录或启动任务

本实验设计获得通过时，只能从 `EXP_DISCUSSION` 进入 `CODE_IMPLEMENTATION`，不会自动进入服务器执行。

### 12.2 授权失效条件

以下情况会导致执行授权立即失效：
- 理论 v0.2 发生实质修改
- 本实验设计发生实质修改
- 已确认的代码版本发生修改，或服务器验证发现需要返回代码阶段修复
- 用户明确要求暂停或返回讨论阶段

### 12.3 执行顺序

```
理论明确通过（✅）
  → 实验设计逐项讨论并明确通过（⏳）
    → CODE_IMPLEMENTATION：在 SOPPO/code 编写源码/配置/服务器脚本并更新 CODE_OVERVIEW（🔒）
      → 用户明确确认代码交接完成、可以提交服务器（🔒）
        → SERVER_EXECUTION：按获批方式同步已确认的 SOPPO 代码
          → 服务器测试、数据、smoke、训练、评价与聚合
            → RESULT_HANDOFF
```

**重要**：
- 用户沉默、未反对、继续提问或查看本文档**均不构成授权**
- 用户明确说“通过实验设计”只解锁代码阶段
- 只有用户在代码交接时明确确认“当前代码版本可以提交服务器”后，才能上传与执行

---

## 13. 附录：实验检查清单

### 13.1 设计阶段检查

- [ ] 理论 v0.2 已通过
- [ ] 研究问题与假设明确
- [ ] 4 组对照方法定义清晰
- [ ] 公平性约束列举完整
- [ ] 超参数预注册（β, lr, λ, epochs, batch_size）
- [ ] 成功标准明确（H1, H2, H3）
- [ ] 失败撤退条件明确
- [ ] 资源预算合理
- [ ] 风险应对策略完备

### 13.2 `CODE_IMPLEMENTATION` 代码交接检查（本地不运行）

- [ ] 获批实验的每项规范均映射到明确的源码、配置、入口或输出
- [ ] L_PE、DPO、对照方法、数据隔离、回复位置随机化与观测接口均已写入代码或明确标为未完成
- [ ] 配置文件与获批实验设计一致，输出路径包含 experiment_id/method/seed 且默认拒绝覆盖
- [ ] 服务器环境、测试、数据准备、smoke、训练、评价和聚合脚本均已准备
- [ ] `../../code/CODE_OVERVIEW.md` 已记录模块、超参、数据流、代码版本/静态校验值和服务器待验证项
- [ ] 本地未安装、导入或运行项目，未执行单元测试、数据/模型任务或 GPU 工作
- [ ] 用户已明确确认当前代码版本可以提交服务器

### 13.3 `SERVER_EXECUTION` 验证与执行检查

- [ ] 服务器环境与项目导入检查通过
- [ ] 服务器单元/集成/数值/梯度测试通过，L_PE 实现正确
- [ ] 服务器标签隔离测试通过（训练入口无法读取 `private_labels/`）
- [ ] 服务器回复位置随机化与输出拒绝覆盖测试通过
- [ ] 数据统计审计通过（位置/长度/格式分布）
- [ ] 预实验 headroom ≥5%
- [ ] β 和 lr 从预实验锁定
- [ ] λ 从 validation set 选出
- [ ] 3 个种子独立运行
- [ ] 每个任务产生完整产物（config, logs, ckpts, predictions）
- [ ] 无 OOM、NaN、Inf 等训练失败
- [ ] C_ε 观测覆盖所有方法与种子

### 13.4 分析阶段检查

- [ ] 统计检验正确（paired t-test, bootstrap）
- [ ] 实质差异阈值应用（≤1% 视为无实质差异）
- [ ] H1, H2, H3 判断有据
- [ ] 诊断信息完整（responsibility, encoding, c_ε）
- [ ] 失败模式分析（如果有）
- [ ] 只回传白名单产物
- [ ] 服务器路径索引记录完整

### 13.5 报告阶段检查

- [ ] 核心发现清晰
- [ ] 方法对比表格与图表完整
- [ ] 统计显著性与实质差异都报告
- [ ] 诊断信息解读合理
- [ ] 对第二阶段的建议明确
- [ ] 记录所有偏离预注册的调整
- [ ] 承认局限性

---

## 14. 变更日志

### v0.1（2026-08-19）
- 初始实验设计
- 基于理论 v0.2（已通过）
- 定义 4 组对照、训练配置、评估指标
- 引入 C_ε 相对功能压缩观测
- 明确第一阶段 MVP 边界（不含 on-policy/rollout）
- λ scheduler 作为可选消融
- 3 个随机种子重复
- 待用户审核与通过

### v0.2（2026-08-19）
- **λ scheduler 消融移至第一阶段主实验**（用户反馈 #1）
- **调整服务器使用方式**（用户反馈 #2）：网页命令行模式
- **优化 C_ε 观测配置**（用户反馈 #3）：s_nsamples=32
- **增加数值稳定性预实验**（用户反馈 #4）：ε ∈ {1e-5, 1e-6, 1e-8}
- 仍为 3 种子设计
- 待用户最终确认

### v0.3 MVP（2026-08-19）
- **简化为真正的 MVP**（用户反馈 #5）
  - **单种子实验**（seed=42），不做 3 种子重复
  - MVP 判断准则：Acc 提升 >2% 视为有效信号
  - 不做统计显著性检验（观测趋势）
  - 实验组合：8 次训练（vs v0.2 的 24 次）
- **C_ε 配置调整**（用户反馈 #6）
  - s_nsamples 和 x_nsamples 与 batch_size 一致（128）
- **阶段间暂停反馈**（用户反馈 #7）
  - 每个阶段完成后报告结果
  - 用户讨论、确认后再进入下一阶段
  - 支持阶段内多轮讨论
- **资源优化**：
  - 总 GPU 时间：~53-103 GPU·小时（vs v0.2 的 ~82-170）
  - 最终存储：~70GB（vs v0.2 的 ~200GB）
  - 端到端时间：~6-9 天（含阶段间讨论）
- **MVP 成功后扩展路径**：
  - 扩展到 3 种子重复
  - 进行正式统计显著性检验
- 待用户最终确认与通过

### 流程状态附注（2026-08-19，不改变实验设计版本）

- 在 `EXP_DISCUSSION` 与 `SERVER_EXECUTION` 之间新增独立的 `CODE_IMPLEMENTATION`。
- 实验设计通过只解锁代码实现；代码交接经用户明确确认后才形成服务器执行授权。
- 本附注只调整阶段职责和门禁，不表示实验设计 v0.3 已通过。

---

**状态**：`EXP_DISCUSSION` - 待用户最终确认并明确通过实验设计 v0.3 MVP
