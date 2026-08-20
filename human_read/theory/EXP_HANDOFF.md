# Theory → Exp 交接文档

**生成时间**: 2026-08-19  
**Cycle ID**: `cycle-20260818-01`  
**理论版本**: v0.2（已通过）  
**目标角色**: exp agent

---

## 项目背景

你好！我是 theory agent。我们刚刚完成了本 cycle 的理论讨论阶段，用户已明确确认理论 v0.2 通过。现在进入 **`EXP_DISCUSSION`** 阶段，由你负责将理论转化为可执行的实验设计。

---

## 一句话研究目标

**在偏好标签有限时，用 population-level label-encoding 结构监督（LERM 风格）替代 instance-level pseudo target，验证这种更弱但更可靠的监督能否改善 preference learning。**

---

## 核心理论（必读）

### 问题定义
- 给定少量真实偏好对 $D_L = \{(x_i, y_i^w, y_i^l)\}$（已知谁更好）
- 给定大量无标签回复对 $D_U = \{(x_j, y_{j1}, y_{j2})\}$（不知道谁更好）
- 目标：用 $D_L$ 锚定方向，用 $D_U$ 提供结构监督

### 核心机制
对无标签对 $(x, y_1, y_2)$，定义偏好概率：
$$p = \sigma[\beta(\log\pi_\theta(y_1|x) - \log\pi_{\text{ref}}(y_1|x)) - \beta(\log\pi_\theta(y_2|x) - \log\pi_{\text{ref}}(y_2|x))]$$

预测编码：$\mathbf{q} = [p, 1-p]^T$

估计两类偏好方向的条件编码：
$$\widehat{\mathbf{e}}_+ = \frac{\sum_i p_i \mathbf{q}_i}{\sum_i p_i}, \quad \widehat{\mathbf{e}}_- = \frac{\sum_i (1-p_i) \mathbf{q}_i}{\sum_i (1-p_i)}$$

理想编码：$\mathbf{e}_+ = [1,0]^T, \mathbf{e}_- = [0,1]^T$

结构损失：
$$L_{\text{PE}} = \frac{1}{2}(\|\widehat{\mathbf{e}}_+ - \mathbf{e}_+\|_1 + \|\widehat{\mathbf{e}}_- - \mathbf{e}_-\|_1)$$

总目标：
$$\boxed{L = L_{\text{DPO}}(D_L) + \lambda L_{\text{PE}}(D_U)}$$

### 与伪目标方法的差异
- **Pseudo-target**：$\widetilde{z}_i = \mathbb{1}[p_i > 0.5]$，逐样本生成硬标签
- **我们的方法**：不给单个样本生成目标，只约束群体编码结构接近理想编码

---

## MVP 实验设计约束（已确认）

### 数据
- **数据集**：UltraFeedback
- **规模**：10k 子集，划分为 1k (labeled) : 8k (unlabeled) : 1k (test)
- **预处理**：
  - 随机交换回复位置（避免 chosen/rejected 字段泄漏）
  - 确保同一 prompt 的回复对不跨集合
  - 需要验证标签隐藏有效性（无监督 baseline）

### 模型
- **主模型**：Qwen2.5-4B-Instruct
- **预实验**：用 1k 样本验证 DPO-100% 的 Acc 显著高于 SFT baseline
  - 如果 headroom < 5%，换成 Qwen2.5-1.5B 或更难数据子集

### Baseline 设置（4 组对比）
1. **DPO-10%**：仅用 $D_L$（有限标签下界）
2. **Pseudo-target**：
   - 硬标签 $\widetilde{z}_i = \mathbb{1}[p_i > 0.5]$
   - 每步实时计算 $p_i$ 并生成伪标签（与 DPO+PE 对齐）
   - 用标准 DPO loss 训练
3. **DPO+PE**：本方法
4. **DPO-100%**：恢复 $D_U$ 真实标签（oracle upper bound，非竞争 baseline）

### 超参数
- **固定**：$\beta = 0.1$（或 0.5，从文献选择）
- **搜索**：$\lambda \in [0.1, 0.3, 0.5, 1.0]$，在 validation set（从 $D_L$ 划出 10%）上选择
- **可选消融**：$\lambda$ scheduler（Fixed / Warmup / Decay）
  - 主实验用 Fixed-$\lambda$
  - 如果时间允许，补充 Warmup（从小到大）和 Decay（从大到小）
- **共享**：所有方法使用相同 lr, batch size, epochs

### 评估
- **指标**：Preference Acc（测试集方向预测准确率）、Brier score
- **重复**：3 个随机种子
- **显著性**：p < 0.05
- **报告**：Acc 和 Brier 分别报告，差异 ≤1% 视为无实质差异

### 成功标准
- **最低有效性**：DPO+PE 显著优于 DPO-10%（说明无标签数据有用）
- **核心贡献**：DPO+PE 显著优于 Pseudo-target（说明结构监督优于伪目标）

### 失败撤退
- 如果 DPO+PE 与 DPO-10% 无显著差异或更差（3 种子一致）
- 1-2 周诊断分析时间盒
- 无改善则撤退，不进入第二阶段

---

## 技术实现要点

### 梯度路径
- **初始方案**：让梯度穿过 responsibility $p_i$（不使用 stop_gradient）
- **应急**：如果训练不稳定，对分母 $\sum_i p_i$ 使用 detach

### 数值稳定性
- **Batch size**：128-256
- **监控**：$\sum_i p_i$ 和 $\sum_i (1-p_i)$ 的分布
- **借鉴**：如果出现退化（所有 $p_i$ 接近 0.5 或 1），考虑 DAPO/GRPO 的分层采样

### 训练流程
每个训练步：
1. 从 $D_L$ 采样 batch，计算 $L_{\text{DPO}}$
2. 从 $D_U$ 采样 batch，用**当前权重** $\theta$ 实时计算 $p_i$
3. 估计 $\widehat{\mathbf{e}}_+, \widehat{\mathbf{e}}_-$，计算 $L_{\text{PE}}$
4. 反向传播 $L = L_{\text{DPO}} + \lambda L_{\text{PE}}$

**重要**：Pseudo-target baseline 必须对齐此流程（每步实时生成伪标签，不固定）。

---

## 你的任务

1. **阅读完整理论**：`/Users/rarp/Desktop/ICLR/work/human_read/theory/current_theory.md`
2. **设计预实验**：Headroom 验证（1k 样本 DPO-100% vs SFT baseline）
3. **设计主实验**：
   - 数据处理流程（隐藏标签、随机交换、划分、验证）
   - 4 组 baseline 的完整训练配置
   - $\lambda$ 搜索策略（validation split、选择准则）
   - 可选的 scheduler 消融设计
4. **预注册**：在 `current_experiment.md` 中明确记录所有超参数、随机种子、评估指标和成功标准
5. **与用户讨论**：逐项确认实验设计，直到用户明确通过

---

## 门禁约束

- 在用户**明确通过**实验设计前，**不得**进入 `CODE_IMPLEMENTATION`
- 实验设计通过后，只能在 `/Users/rarp/Desktop/ICLR/work/code/` 编写源码、配置和服务器脚本，并维护 `CODE_OVERVIEW.md`；此时仍不得上传服务器或启动任务
- 只有代码交接完成且用户明确确认当前代码版本可以提交服务器后，才能进入 `SERVER_EXECUTION`
- 实验设计必须记录在 `/Users/rarp/Desktop/ICLR/work/human_read/exp/current_experiment.md`
- 参考 `/Users/rarp/Desktop/ICLR/work/human_read/exp/AGENTS.md` 了解 exp 阶段的维护规则

---

## 关键文件路径

- 理论完整版：`../theory/current_theory.md`
- 理论 QA 记录：`../theory/theory_qa_current.md` 和 `theory_qa_archive.md`
- 实验设计文件：`current_experiment.md`（你需要创建/更新）
- 下一阶段代码说明：`../../code/CODE_OVERVIEW.md`
- 全局规则：`/Users/rarp/Desktop/ICLR/AGENTS.md`
- Exp 阶段规则：`AGENTS.md`（与你同目录）

---

## 待澄清问题（供参考）

1. UltraFeedback 的具体版本和过滤策略
2. Qwen2.5-4B-Instruct 的具体 checkpoint
3. 训练总步数、warmup、scheduler、优化器选择
4. Validation set 的使用时机（每个 epoch？固定间隔？）
5. Pseudo-target 的具体实现（同一训练循环？分阶段？）
6. 如何记录和监控训练动态（$p_i$ 分布、编码估计、loss 曲线）

---

祝实验设计顺利！如果有任何理论相关的问题，可以随时参考 `current_theory.md` 或询问用户。
