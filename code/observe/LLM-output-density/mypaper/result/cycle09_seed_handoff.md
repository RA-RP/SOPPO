# Cycle 09 Seed & Cross-Conversation Handoff

```yaml
artifact_type: next_cycle_seed_handoff
authored_by: Result Analysis conversation (2026-07-07)
status: ⚠️ 待同步 — Result 只写 result/。下列各段由对应对话读取并内化到自己目录：
  - Related Work 对话 → related_work/(文献 + 定位)
  - Theory 对话      → theory/(理论重构 + 创新点)
  - Experiment Design → exp/ + code/current_code_brief.md(cycle09 设计)
  - Next Cycle Seed   → next_cycle/(open_questions / intent)
source_of_truth: result/incremental_log.md(cycle08 节)、current_picture.md、claims_allowed.md；
  memory: cycle08-opd-geometry-lit-scoop
```

---

## 0. Cycle 08 结论一句话

OPD ≫ SFT 能力(MATH500 +0.10~0.24,统一 cap 16384;OPD 0.848 vs SFT 0.752);OPD 保 MMLU-Pro、
SFT 侵蚀;**唯一干净区分两臂的是激活 effective_rank(SFT step_20 中层 rank-bump,OPD 无)**。
权重-主方向侧(OverlapLift 撤回、ρ bf16+LoRA 混淆)**判不了 on/off-principal,本轮不下结论**。

---

## 1. → Related Work 对话(SCOOP 风险,最高优先)

**"OPD 在权重空间 off-principal"已被抢发(2026-06,全参)。先读前两篇全文核重叠:**
- **arXiv 2606.13657** Dense Supervision, Sparse Updates: On the Sparsity and Geometry of On-Policy Distillation(Guo Yu…Han-Jia Ye)——OPD 更新满秩但谱集中、落主子空间之外、偏 |W0|≈0;定位 relaxed off-principal(SFT-on → OPD-relaxed-off → RLVR-tight-off);subspace locking。
- **arXiv 2606.07082** On the Geometry of On-Policy Distillation(Zhennan Shen, Yanshu Li, Qingyu Yin…)——明说 parameter space 非 activation space;"OPD 自成一套更新几何"。
- 旁支:**2512.23165**(Evaluating PEFT for RLVR:PiSSA/MiLoRA spectral collapse)、**2603.02224**(LoRA 遗忘定律 F=α(1−cos²θ_min)+β)、**2604.08844**(LoRA 权重谱编码训练目标,跨方法要重标定)、**2511.08567**(TPNT 母论文)、LoRA Without Regret(Thinking Machines)。

**读全文时重点核三件事**(决定护城河):他们的 OPD 是否同定义 on-policy 蒸馏;有没有做 **on-policy vs reward-density 消融**;有没有碰 **激活空间 / OOD-preservation**。若都没碰后两者 → 我们的创新点还在。

**入 source_matrix**:以上文献 + 我们的立场(权重侧 off-principal 已被占,差异化在激活/压缩/OOD/测量学)。

---

## 2. → Theory 对话(理论重构 + 定创新点)

**被实测冲击的旧假设**:我们曾用 TPNT Gate I 推"OPD 锚外部 teacher → 该像 SFT on-principal"。全参
文献实测 OPD=off-principal(像 relaxed RL)→ **"on-policy 不是驱动"被削弱**:OPD 与 RL 同 on-policy
且同 off-principal,SFT(off-policy)才 on-principal → **on-policy 可能才是推向 off 的变量,reward 密度
只调 tight/relaxed**。

**候选新创新点(护城河,按可辩护度排)**:
1. **观测空间**:激活白化/压缩空间(effective_rank)是区分 OPD/SFT、且跟踪 OOD 的唯一干净观测量;
   权重-主方向空间(即便做对的 ρ)在此问题上不 discriminating。**机制理由**:白化尺度不变,抹掉杀死
   OverlapLift/ρ 的 bf16 幅度混淆。
2. **机制**:"破坏压缩→能力注入"——ID dip ↔ 压缩破坏(ER 上凸),**破坏幅度 ↔ OOD 损伤**(SFT 大破坏
   +4.56→OOD 侵蚀;OPD 小 +1.3→保住),ID-dip 深度 ↔ 更新速度/相干性。
3. **测量学**:LoRA+bf16 让权重-主方向指标系统性失效(OverlapLift=magnitude/bf16、ρ 偏 on),需全参/fp32。

定创新点前必须先完成 §1 的三点核对。

---

## 3. → Experiment Design / Code 对话(cycle09 设计)

**成本阶梯(从省到贵,①能解决就不上③)**:
- ① **fp32-master 重训**(去 bf16 confound,LoRA 秩不变;≈重训 + 2× ckpt 存储)。测:去 bf16 后 ρ 是否分开?
- ② **8-bit AdamW 全参**(优化器态 32→8G,训练态~40G,可能保 rollout+update colocate 单卡)。
- ③ **标准全参**(最干净;≈1.5–2× GPU-hours/臂;布局见下)。

**GPU 布局(全参,你已 2×96G)**:全参 student 训练态 ~64–80G 占满一张卡 → **rollout 与 update 必须分卡**:
卡0=全参训练/更新;卡1=student vLLM rollout + teacher 8B。新增每步 student 权重同步(~8G bf16)。GPU 数不增,
每步时间 ~1.2–1.5×。(8-bit 优化器可能让其重新 colocate。)

**cycle09 三个实验**:
1. **干净判 OPD on/off-principal**:①/②/③ 之一取得干净权重 → ρ + p_on/p_off(见 `cycle08_rho_metric_spec.md`)
   + 每臂自身尺度 random-null(现有 null 只 OPD 尺度,SFT 侧不干净,需补)。
2. **on-policy vs reward-density 消融**:设计一个能把二者拆开的臂(如 off-policy 蒸馏 / 稀疏 vs 稠密目标),
   看 off-principal 由哪个驱动。
3. **压缩→OOD 因果**:在 **OOD 域输入(MMLU-Pro/通用文本)的激活**上测压缩轨迹,证 SFT 重压缩渐进挤占 OOD
   方向、且 L18 破坏幅度**预测** MMLU-Pro 的 Δ;多 seed。

---

## 4. → Next Cycle Seed 对话(open_questions 增量)

新增 open questions:
- OPD 权重 on/off-principal 真值(全参/fp32 判);LoRA+bf16 读数作废。
- 驱动 off-principal 的是 on-policy 还是 reward-density?(消融)
- 压缩破坏 → OOD 损伤 是否因果?(OOD-域激活 + 预测 + 多 seed)
- 相对 2606.13657/07082,本项目创新点确切边界?

resolved(移入 history):cycle08 B08/C08/D08(OPD≫SFT 能力、OPD 保 OOD、激活几何区分);A08 OverlapLift 撤回。

---

**Result 侧到此收尾。以上为跨对话种子,各对话读取后内化到自己目录并按 readme 追加 evolution 条目。**
