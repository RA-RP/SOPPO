# Cycle 08 补测指南：尺度不变的方向对齐度 ρ(+ 方向随机 null)

```yaml
artifact_type: metric_spec_for_coder
status: ⚠️ draft spec (Result 对话起草，交 coder 执行；结论回填后 Result 再读)
scope: 纯几何,零训练/零推理/零 eval——只在已有权重上做线性代数
goal: 在 LoRA 尺度下,判 OPD/SFT 的权重更新方向是 on- / off-principal 还是与随机不可区分
motivation: OverlapLift/主夹角在 LoRA 下被 bf16-幅度阈值主导(连 SFT 都从 super-random 翻成 sub-random),判不了方向。ρ 是比值→尺度不变,直接测方向。
```

## 0. 一句话

对每个权重矩阵,量**更新 ΔW 的能量有多少落在基座 W0 的 top-k 主奇异子空间里**,并和"**同谱但方向随机**"的 null 比。ρ 是比值,`‖ΔW‖` 在分子分母同时出现被约掉 → 与更新幅度无关,只看方向。

## 1. 指标定义(全部 fp32,别用 bf16)

设 `W0 ∈ R^{m×n}`(m=out,n=in),微调后 `W+ = W0 + ΔW`,`ΔW = W+ − W0`。
对 W0 做经济 SVD:`W0 = U S Vᵀ`,取 `U_k = U[:, :k]`、`V_k = V[:, :k]`。

三个能量占比(∈[0,1],**报 ρ² = 能量分数**,更可加、更好读):

```
ρ²_U  = ‖ U_kᵀ · ΔW ‖_F²        / ‖ΔW‖_F²     # 左/输出方向落在 top-k 左子空间的能量占比
ρ²_V  = ‖ ΔW · V_k ‖_F²         / ‖ΔW‖_F²     # 右/输入方向落在 top-k 右子空间的能量占比
ρ²_UV = ‖ U_kᵀ · ΔW · V_k ‖_F²  / ‖ΔW‖_F²     # 两侧都在(最严格的 on-principal 能量)
```

高效算法(别显式构造 m×m 投影矩阵):
```
A  = U_k.T @ dW            # (k×n)   -> rho2_U  = (A**2).sum() / dW_sq
B  = dW @ V_k              # (m×k)   -> rho2_V  = (B**2).sum() / dW_sq
C  = A @ V_k               # (k×k)   -> rho2_UV = (C**2).sum() / dW_sq
dW_sq = (dW**2).sum()
```

`ρ²_U`(左)和 `ρ²_V`(右)分别对标 TPNT 的 U/V 主夹角,是主指标;`ρ²_UV` 辅助。

## 2. 方向随机 null(关键——ρ 单独没意义,必须和它比)

**首选 null:同谱、随机方向(spectrum-matched random rotation)**——直接对标 TPNT 的 rotate/permute 干预:
```
对真实 dW 做经济 SVD:  dW = P Σ Qᵀ   (P: m×r, Q: n×r, r = dW 的数值秩)
重复 n_draws≥20 次:
    抽 Haar 随机正交 P'(m×r)、Q'(n×r)   # QR(randn(m,r)) 取 Q 因子;Q' 同理
    dW_rand = P' Σ Q'ᵀ                  # 奇异值不变,方向随机
    用同一 U_k,V_k 算 dW_rand 的 ρ²_U/ρ²_V/ρ²_UV
报 null 的 mean ± std
```
> 这个 null 保留了 ΔW 自己的奇异值谱,只随机化方向,所以"real vs null"的差**纯是方向信号**。因 ρ² 尺度不变,不需要 scale-match,只需 rank/谱-match。
>
> 备选(更简单但略松):`dW_rand = randn(m,r) @ randn(r,n)`,r 取真实更新的秩。

**先测真实更新的秩 r**:对每个 (arm, layer, module) 的 dW 看数值秩(奇异值 > 1e-6·σ_max 的个数)。LoRA→应 ≈ adapter rank(如 32);若 OPD 是全参合并出来的、秩很高,如实记录并让 null 的 r 跟着改。

## 3. 扫 k(别只报单个 k)

`k ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}`(上限 min(m,n))。出"ρ² 随 k 的累积曲线"。特别关注 k 在真实更新秩 r 附近的行为。

## 4. 跑哪些 / 输入

- **臂**:`opd`、`sft`,外加 **null**(如上,每臂各自的谱)。
- **step**:全 grid `[0,5,10,20,40,80,160,320,480,624]`(step_0 的 ΔW=0,跳过或标 NA)。
- **layer × module**:沿用现有 `{9,18,27} × 7 modules`。
- **输入文件**:base 的 `W0` + 各 checkpoint 的 `W+`(或 LoRA 的 B、A 直接给 dW)。**复用 principalEvidence 已导出的 `.npy`**;若已清,从 checkpoint 重新 dump(仍零训练)。SVD(W0) 这一步 principalEvidence 已经在做,可直接复用 U_k/V_k。

## 5. 判读规则(每臂各自 vs 自己的 null,在每个 k)

| 情形 | 结论 |
|---|---|
| `ρ²_real > null_mean + 2·std` | **on-principal**:更新能量比随机更集中在主子空间(像 TPNT 的 SFT) |
| `ρ²_real < null_mean − 2·std` | **off-principal**:更新比随机更避开主子空间(像 TPNT 的 RL) |
| `|ρ²_real − null_mean| < 2·std` | **与随机不可区分**:LoRA 秩承载不了这个方向信号 → 需全参(cycle09) |

关键对比两组:①每臂 vs 自己的 null(判 on/off/无);②OPD vs SFT(判两者方向是否真不同)。

## 6. 接口文件 schema(coder 回填成这三个文件,Result 之后照此读)

落在 `local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01/geometry/`:

**(a) `rho_trajectory.csv`** — 真实更新
```
arm, step, layer, module, k, rho2_U, rho2_V, rho2_UV, dW_fro, dW_rank
```

**(b) `rho_null.csv`** — 随机方向 null(每臂各自谱;与 step 无关可省 step,或按 step 存)
```
arm, step, layer, module, k, rho2_U_mean, rho2_U_std, rho2_V_mean, rho2_V_std, rho2_UV_mean, rho2_UV_std, n_draws
```

**(c) `rho_summary.md`** — 人读小结:在 k∈{32,128} 各出一张
```
step | OPD ρ²_U | SFT ρ²_U | null ρ²_U(mean±std) | OPD 判定 | SFT 判定
```
每臂末尾一行 verdict:on / off / indistinguishable（2σ 口径,mean over layer×module）。
并附一句:真实更新的数值秩 r(OPD / SFT 各是多少)。

## 7. 参考实现骨架(加到 principalEvidence.py 的 `_analyse_one_weight` 或独立 `rho_probe.py`)

```python
# 已有: W0 (fp32), Wp (fp32), U0k=U[:, :k], V0k = Vh0k.T  (principalEvidence 已算)
dW = (Wp - W0).float()
dW_sq = float((dW**2).sum())
def rho2(dW, Uk, Vk):
    A = Uk.T @ dW; C = A @ Vk; Bm = dW @ Vk
    return (A.pow(2).sum()/dW_sq).item(), (Bm.pow(2).sum()/dW_sq).item(), (C.pow(2).sum()/dW_sq).item()
# real:
rU, rV, rUV = rho2(dW, U0k, V0k)
# null (spectrum-matched random rotation):
P, S, Qh = torch.linalg.svd(dW, full_matrices=False)
r = int((S > 1e-6*S[0]).sum()); Pr, Sr, Qr = P[:, :r], S[:r], Qh[:r, :]   # dW ≈ Pr diag(Sr) Qr
draws=[]
for _ in range(n_draws):
    Pp = torch.linalg.qr(torch.randn(dW.shape[0], r))[0]
    Qp = torch.linalg.qr(torch.randn(dW.shape[1], r))[0]
    dW_rand = (Pp * Sr) @ Qp.T
    draws.append(rho2(dW_rand, U0k, V0k))
# 对每个 k 重复(把 U0k/V0k 换成对应 k 的切片)
```

## 8. 常见坑

- **必须 fp32**:ΔW = Wp − W0 在 bf16 下会丢精度、正好踩回我们要躲开的那个坑。
- **W0 的 SVD 用 full_matrices=False**(经济),否则 U 是 m×m 巨阵。
- **ρ² 尺度不变自检**:把某个 dW 乘 10,ρ² 应不变——coder 顺手 assert 一下,确认实现对。
- **step_0**:ΔW=0,ρ 无定义,标 NA。
- **null 用同一 U_k,V_k**:null 只换 ΔW 的方向,principal 子空间(来自 W0)不动。

---

**交付**:coder 跑完把 (a)(b)(c) 三个文件回填到上面的路径;我(Result 对话)按 §6 schema + §5 判读规则读,写进 result/ 并更新 cycle08 的 A08(OverlapLift 撤回 → ρ 结论)。这份 spec 属 theory/code-brief 范畴,由 Result 起草标 ⚠️,正式内化仍走 Experiment Design/Code 对话。
