# Theory QA - 历史精简记录

本文件记录每轮 theory 讨论完成后的核心结论。每完成一轮三阶段 QA 后更新。

旧`theory_qa_current.md`的完整357行Round1/Round2问答快照固定于Git commit `d338eb5bedef16d83a42790c3faa97f8f404315b`。2026-08-25起，current QA只呈现Round3仍有效的决定与未决项；该结构整理不改写旧回答。

---

## `cycle-20260818-01` 理论 v0.2

**状态**: 已完成，用户已确认通过  
**开始日期**: 2026-08-19  
**完成日期**: 2026-08-19  
**QA 轮次**: 3 轮（Theory→User, User→Theory, Theory→User 最终确认）

### 核心结论

#### 研究核心
- **核心命题**：用 population-level label-encoding 结构监督替代 instance-level pseudo target
- **MVP 目标**：在静态无标签偏好对上验证 DPO+PE 优于 DPO-10% 和 Pseudo-target
- **第二阶段**：仅在 MVP 成功后，扩展到动态 SFT 候选对

#### 关键决策
1. **数据**：UltraFeedback 10k 子集（1k:8k:1k），随机交换位置防泄漏
2. **模型**：Qwen2.5-4B-Instruct，需预实验验证 headroom ≥5%
3. **Baseline**：保留 Pseudo-target（硬标签，实时采样）以支撑核心论点
4. **超参数**：固定 $\beta=0.1$，搜索 $\lambda \in [0.1,0.3,0.5,1.0]$，可选 scheduler 消融
5. **成功标准**：3 种子，p<0.05，Acc/Brier 分别报告
6. **撤退**：1-2 周诊断时间盒，无改善则撤退

#### 技术细节
- **梯度路径**：初始不用 stop_gradient，监控稳定性
- **数值稳定**：batch size 128-256，必要时借鉴 DAPO/GRPO 分层采样
- **训练流程**：每步实时计算 $p_i$（不固定）
- **$C_{\gamma}$**：第二优先级，暂不纳入 MVP

#### 未决问题
- Headroom 预实验结果可能影响模型/数据选择
- $\lambda$ scheduler 是否有效（可选消融）
- Pseudo-target 的具体实现细节（epoch/lr/scheduler 对齐）

---

## 历史 Cycle 记录

_（未来 cycle 的 theory QA 精简结论将追加在此）_
