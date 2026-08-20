# OPD (Cycle 08) 训练框架逐层精读

> **这份文档怎么用。** 本文档是我们一起精读 Cycle 08「OPD + verl」训练框架的工作底稿，
> 目标是用大约一天时间，**自顶向下**从当前代码入口一路读到 GPU 层结构。
> 它随讨论**动态维护**：每深入一层就把该层从「待深入」补全为正文，讨论中确认/推翻的结论
> 记到最后的「讨论区」。所以任何时刻这份文档就是我们当前理解的快照。
>
> 约定：中文叙述、英文标识符（保持与 `Readme-dataSplit.md` / `Readme-experiments.md` 一致）。
> 代码坐标一律用**可点击跳转链接**（在 VSCode 预览里点一下即跳到对应行）：仓库文件相对本文档路径，
> verl 源码在 `/root/autodl-tmp/verl`（本机可读、可无限深入，用 `file://` 绝对链接）。
> 本文档只读代码、不改代码；`mypaper/` 不动。

---

## 0. 分几层，以及为什么这么分

我的建议是分成 **7 层（L0–L6）**，构成一条从「实验入口」到「GPU 硬件」的下降阶梯。
每一层是一个清晰的「高度」：有独立的关注点、独立的文件集、以及一个它专门回答的问题。
一天的节奏大致是「一层一站」，边读边在这份文档里补全。

| 层 | 名称 | 这一层回答的问题 | 关键文件 | 状态 |
|---|---|---|---|---|
| **L0** | 鸟瞰：系统在做什么 | 我到底在看一套什么东西？整体调用链长什么样？ | [run_cycle08.py](experiments/opd_sft_h1/scripts/run_cycle08.py)、[watchdog_cycle08.sh](experiments/opd_sft_h1/scripts/watchdog_cycle08.sh)、[run_cycle08_opd.sh](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh)（黑盒） | ✅ 已完成 |
| **L1** | 编排层 | 一次 run 如何在**不碰 GPU** 的前提下自我驱动、断点续跑、控盘、切环境？ | [watchdog_cycle08.sh](experiments/opd_sft_h1/scripts/watchdog_cycle08.sh)、[run_cycle08.py](experiments/opd_sft_h1/scripts/run_cycle08.py)、[cycle08_ckpt_pruner.py](experiments/opd_sft_h1/scripts/cycle08_ckpt_pruner.py)、[cycle08_data_prep.py](experiments/opd_sft_h1/scripts/cycle08_data_prep.py) | ✅ 已完成 |
| **L2** | 训练启动层（交接契约） | 我们的代码用什么「契约」把训练任务交给 verl？每个配置组为什么这么设？ | [run_cycle08_opd.sh](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh)、[cycle08_null_reward.py](experiments/opd_sft_h1/scripts/cycle08_null_reward.py)、[apply_verl_patches.sh](experiments/opd_sft_h1/scripts/apply_verl_patches.sh) | ✅ 已完成 |
| **L3** | verl 训练器层 | verl 收到配置后怎么起 Ray、怎么组织训练循环（rollout→teacher→loss→update）？ | [main_ppo.py](file:///root/autodl-tmp/verl/verl/trainer/main_ppo.py)、[verl/trainer/ppo/](file:///root/autodl-tmp/verl/verl/trainer/ppo) | ⬜ 待深入 |
| **L4** | worker / 并行层 | 谁在哪张卡上跑？student 训练 + rollout 混合引擎、teacher 独立池怎么协作？ | verl workers、[distillation/fsdp/](file:///root/autodl-tmp/verl/verl/trainer/distillation/fsdp) | ⬜ 待深入 |
| **L5** | 蒸馏损失层（科学内核） | `forward_kl_topk` 到底算什么？梯度从哪来？为什么是「监督 OPD」而非策略梯度？ | [losses.py](file:///root/autodl-tmp/verl/verl/trainer/distillation/losses.py) | ⬜ 待深入 |
| **L6** | GPU / 系统层（金属层） | 2×96G 上显存怎么排布？OOM 风险在哪？FSDP2 / vLLM KV / checkpoint I/O 怎么权衡？ | [run_cycle08_opd.sh](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh)（显存/OOM 注释）、verl FSDP workers、[cycle08_convert_ckpt.py](experiments/opd_sft_h1/scripts/cycle08_convert_ckpt.py) | ⬜ 待深入 |

**为什么这样切，而不是更多/更少：**

- 每层对应一次**真实的抽象跳变**，而不是任意切段：
  L1→L2 是「Python 编排」跳到「Hydra CLI 契约」；
  L2→L3 是「我们的仓库」跳到「verl 的仓库」；
  L3→L4 是「单进程逻辑循环」跳到「多 Ray actor / 多 GPU」；
  L4→L5 是「系统/工程」跳到「数学/梯度」；
  L5→L6 是「算法」跳到「显存与硬件」。
  每次跳变都值得单独停一站。
- 合并会藏掉重点：把 L5（损失数学）并进 L3 会淹没「为什么 `use_policy_gradient=False`」这个
  全实验最关键的科学判断；把 L6（显存）并进 L4 会淹没那些 OOM 注释里沉淀的工程细节。
- 再细分则过碎：例如把 vLLM rollout 和 FSDP 训练拆成两层，会割裂「混合引擎在同一张卡上
  sleep/wake 切换」这个必须整体理解的机制。

**推荐节奏**：从 L1 开始逐层下降；每层我先把正文写进本文档，你读完提问，我们在「讨论区」
定稿后再进下一层。哪一层你已经清楚也可以跳过或压缩。**下一步默认从 L1 开始**，你也可以直接
点名跳到某层（例如你现在最关心 L5 损失或 L6 显存，可以先去那儿）。

---

## L0 — 鸟瞰：这套系统到底在做什么 ✅

### L0.1 一句话

Cycle 08 用 **verl** 训练**一条 OPD（on-policy distillation，在线策略蒸馏）轨迹**：
学生 `Qwen3-4B-Base` 在线自采样、由教师 `Qwen3-8B` 提供 top-k 逐 token 分布做监督目标，
沿着**与 Cycle 07 SFT 轨迹逐步对齐**的检查点网格训练、评测、几何分析，
从而在完全可比的协议下回答假设 H1：**OPD 相比同规模 SFT，是否有不同的「先降后升」形态与几何轨迹**。

一句话点出三个「它是什么 / 它不是什么」，避免一开始就走偏：

- 它是 **监督式蒸馏**（学生 rollout 的 token 上，最小化对教师 top-k 分布的前向 KL），
  **不是** RL：虽然跑在 verl 的 PPO/GRPO 框架里，但 `use_task_rewards=False` + 常数 0 奖励，
  奖励/优势那套机器基本空转（细节留到 L2/L5）。
- 它是 **on-policy**：每步的训练序列由学生**当前**权重现采（`ppo_epochs=1`，rollout 不复用），
  这正是「OPD」里 on-policy 的含义 —— 与 SFT 喂固定数据集答案形成对照。
- 它对比的是 **轨迹**，不是单点：重点在 checkpoint 网格 `[0,5,10,…,480,624]` 上
  与 Cycle 07 SFT 逐步对齐，看形状（dip / recovery）与几何漂移，而非只看终点分数。

### L0.2 实验目标：OPD (c08) vs SFT (c07)

| 维度 | Cycle 07（对照） | Cycle 08（本框架） |
|---|---|---|
| 学生 | Qwen3-4B-Base | Qwen3-4B-Base（相同） |
| 训练信号 | 数据集答案的交叉熵（SFT） | 教师 Qwen3-8B top-k 分布的前向 KL（OPD） |
| 框架 | TRL SFTTrainer | verl（`main_ppo` + distillation 扩展） |
| 数据 | seed=42 的 5000 prompt 样本 | **同一份** `opd_prompts_5k.parquet`（复用 c07） |
| 有效 batch / epoch | eff_batch 16、2 epoch | `train_batch=ppo_mini=16`、`total_epochs=2` |
| 评测终点 | step 624 | step 624（verl 过滤到 4999 行 + drop_last，凑到 624 步） |
| 探针 / 几何 | S/X 探针 + GetSlice + PE | **复用 c07 逐字节相同的探针**，保证几何可比 |

设计上处处对齐 c07，就是为了让「OPD 唯一改变了训练目标（数据 CE → 教师 KL）」这一条尽量干净。
对齐的具体参数（步数换算、LoRA、长度、loss 归一化）在 L2 展开。

### L0.3 五阶段流水线

单一 Python 入口 `run_cycle08.py` 串行编排五个阶段，全程 **skip-if-exists**、可 `--start-from-phase` 续跑：

```text
Phase 2   train      跑 verl OPD（shell out 到 verl env）→ checkpoints/global_step_<N>/actor      [≥2 GPU]
   │                  伴随 cycle08_ckpt_pruner.py 在线删非网格 ckpt（省磁盘）
Phase 2b  convert     verl 分片 checkpoint → 合并 HF 模型（verl env 子进程），删原始分片省磁盘
Phase 3   eval        复用 c07 runner_think（MATH500/Numina/AIME24）+ lm_eval（GPQA-D/MMLU-Pro）  [2 GPU 并行]
Phase 4   geometry    复用 c07 GetSlice + export_weights + principalEvidence，同一批 S/X 探针        [2 GPU 并行]
Phase 5   aggregate   RESULTS_08.md + trajectory_scores.csv + opd_vs_sft_comparison.md（读 c07 轨迹） [CPU]
```

本框架**训练**部分是 Phase 2 / 2b（L1–L6 的主线）；Phase 3–5 基本是**复用 Cycle 07** 的评测/几何/汇总
（`run_cycle08.py` 直接 `from scripts import run_cycle07 as c07` 调它的叶子函数），本次精读会略过，只在需要时点到。

### L0.4 两套 conda 环境（关键，容易踩坑）

整条流水线跨两个 Python 环境，靠子进程切换：

| 环境 | 解释器 | 负责 |
|---|---|---|
| `density` | `/root/miniconda3/envs/density/bin/python` | 编排（`run_cycle08.py`）、Phase 3 eval、Phase 4 geometry、Phase 5 aggregate |
| `verl` | `/root/autodl-tmp/envs/verl/bin/python` | Phase 2 训练（`run_cycle08_opd.sh`）、Phase 2b 转换（`cycle08_convert_ckpt.py`） |

`watchdog` 与 `run_cycle08.py` 本体跑在 `density`；到了 Phase 2，`phase2_train()` 把 `PATH` 前置为
verl env 再 `bash run_cycle08_opd.sh`；Phase 2b 用 `VERL_PY` 显式调 verl 解释器。
verl（torch2.9 / vllm0.12 / verl0.9dev）与 density（评测/几何栈）依赖不兼容，所以必须隔离。

### L0.5 自顶向下调用链（本文档的主轴）

这条链就是「从入口到 GPU」的完整竖切，也是 L1–L6 逐层放大的对象：

```text
watchdog_cycle08.sh                         [density python; setsid nohup 脱离终端]
  └─ run_cycle08.py  main()                 [编排；skip-if-exists；--start-from-phase]   ← L1
       │
       ├─ Phase 2  phase2_train()
       │    ├─ cycle08_ckpt_pruner.py        [并行 Popen；在线删非网格 checkpoint]
       │    └─ bash run_cycle08_opd.sh       [切到 verl env；构造 Hydra 覆盖参数]        ← L2
       │         └─ python -m verl.trainer.main_ppo  <7 组 Hydra overrides>
       │              ├─ ray.init(runtime_env=…)                                        ← L3
       │              └─ TaskRunner.remote().run(config)          [Ray driver]
       │                   └─ (Ray)PPOTrainer.fit()   主循环 × 624 步
       │                        每步：
       │                        (a) student vLLM rollout 现采序列 ───────────► GPU0 学生卡  ← L4
       │                        (b) teacher 8B vLLM 算 top-k logprob ────────► GPU1 教师卡  ← L4
       │                        (c) forward_kl_topk 蒸馏损失（losses.py）                ← L5
       │                        (d) FSDP2 actor 反向 + 优化器更新 ────────────► GPU0 学生卡  ← L6
       │                        (e) 每 save_freq=5 存 global_step_<N>/actor（分片）
       │
       ├─ Phase 2b phase2b_convert()         [verl python 子进程]
       │    └─ cycle08_convert_ckpt.py       分片 checkpoint → 合并 HF 模型
       ├─ Phase 3  phase3_eval()             [density；ProcessPool，每卡一个 checkpoint]
       │    └─ 复用 c07 runner_think + lm_eval
       ├─ Phase 4  phase4_geometry()         [density；ProcessPool]
       │    └─ GetSlice + export_weights + principalEvidence
       └─ Phase 5  aggregate()               [density；CPU]
            └─ RESULTS_08.md + opd_vs_sft_comparison.md

  （pipeline 完成或崩溃后，watchdog 倒计时 GRACE 秒 → shutdown 关机，停止 GPU 计费；
    touch ABORT_SHUTDOWN 可取消关机）
```

> 说明：`main_ppo` 内部的确切类名与调用（`TaskRunner` 的版本、`fit()` 循环里蒸馏分支的落点）
> 在 L0 先当**功能黑盒**处理 —— 已确认的是入口 `verl.trainer.main_ppo` → `run_ppo()` → `ray.init()`
> → `TaskRunner.remote().run(config)`（见 `verl/trainer/main_ppo.py`）。精确路径留到 L3 打开源码核实，
> 避免在鸟瞰层臆测 verl 内部。

### L0.6 关键坐标（一页速查）

```text
# 模型
student  Qwen3-4B-Base   /root/autodl-tmp/model/Qwen/Qwen3-4B-Base
teacher  Qwen3-8B        /root/autodl-tmp/model/Qwen/Qwen3-8B

# 数据（复用 cycle07 的 seed=42 5000 样本；verl 过滤到 4999 行）
train    /root/autodl-tmp/cycle08_opd_trajectory/data/opd_prompts_5k.parquet
smoke    /root/autodl-tmp/cycle08_opd_trajectory/data/opd_prompts_smoke16.parquet

# 硬件 / 拓扑
2×96G；student 1 卡（FSDP2 训练 + vLLM rollout 混合引擎），teacher 8B 独占 1 卡（vLLM 推理）
NGPUS_PER_NODE=1（student 池）＋ TEACHER_WORLD_SIZE=1（teacher 池），两池不相交

# 训练关键超参（详见 L2）
loss=forward_kl_topk  topk=32  use_policy_gradient=False  use_task_rewards=False
LoRA r=32 alpha=64 all-linear；fsdp2；train_batch=ppo_mini=16；ppo_epochs=1；total_epochs=2
total_training_steps=624；max_prompt=1024；max_response=10240；lr=5e-5；loss_agg=token-mean

# checkpoint 网格（评测/几何在这些步上做，与 c07 对齐）
[0, 5, 10, 20, 40, 80, 160, 320, 480, 624]      save_freq=5

# 产物
run root   /root/autodl-tmp/cycle08_opd_trajectory
copyback   mypaper/local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01
           （这是 pipeline 自己的回拷目标；本文档不写它）
```

---

## L1 — 编排层：run 如何自我驱动 ✅

> 本层文件都在 `experiments/opd_sft_h1/scripts/`，下文 `文件:行` 均相对此目录（verl 源码除外）。
> 本层是整条流水线的**控制流**：决定「什么时候跑哪一步、在哪个环境跑、崩了怎么续、盘满怎么办」，
> 它本身几乎不碰 GPU —— 真正的 GPU 计算从 L2（verl 启动）之后才开始。
>
> 一句话看清 L1 的三个角色：
> `watchdog_cycle08.sh`（**监工**：脱离终端、跑完/崩溃都关机省钱）
> → `run_cycle08.py`（**调度**：五相位串行、断点续跑、双环境切换）
> → `cycle08_ckpt_pruner.py` + `cycle08_data_prep.py`（**两个辅助进程/前置步**：磁盘守护、数据物化）。

### L1.1 watchdog_cycle08.sh —— 最外层监工（跑完就关机）

它不做任何计算，只干一件事：**把整条 pipeline 包起来跑，结束后自动关机停止 GPU 计费**。这对
按小时计费的 autodl 很关键 —— 25 小时的 run 你不可能守着，跑完必须自动停。

- **脱离终端**：脚本头注释 [watchdog_cycle08.sh:13](experiments/opd_sft_h1/scripts/watchdog_cycle08.sh#L13) 说明它「Detached via `setsid nohup` at launch」——
  即在**启动命令**处用 `setsid nohup` 脱离，关掉 vscode/ssh 不会杀掉它（脱离动作在启动命令，不在脚本内部）。
- **变量区** [watchdog_cycle08.sh:20-27](experiments/opd_sft_h1/scripts/watchdog_cycle08.sh#L20-L27)：`RUN_ROOT`（run 根目录）、`PY=/root/miniconda3/envs/density/bin/python`
  （[:22](experiments/opd_sft_h1/scripts/watchdog_cycle08.sh#L22)，注意监工本体用 **density** 解释器）、日志与状态文件、`GRACE=90`（[:27](experiments/opd_sft_h1/scripts/watchdog_cycle08.sh#L27)，关机前的宽限秒数）。
- **跑 pipeline 并捕获退出码** [watchdog_cycle08.sh:40-41](experiments/opd_sft_h1/scripts/watchdog_cycle08.sh#L40-L41)：
  `"$PY" run_cycle08.py >> "$RUN_LOG" 2>&1` 然后 `RC=$?`。注意这里**没有** `--start-from-phase`，
  即默认从 Phase 2 起，靠 skip-if-exists（见 L1.3）跳过已完成的部分 —— 这就是「无脑重启即续跑」。
- **写状态文件** [watchdog_cycle08.sh:43-54](experiments/opd_sft_h1/scripts/watchdog_cycle08.sh#L43-L54)：无论成功（`RC==0`）还是失败都写 `RUN_STATUS.txt`，记录退出码、
  最后 checkpoint、日志路径，方便重启后排查。
- **关机 + 逃生阀** [watchdog_cycle08.sh:58-67](experiments/opd_sft_h1/scripts/watchdog_cycle08.sh#L58-L67)：倒计时 `GRACE` 秒，其间若发现 `ABORT_SHUTDOWN` 文件则
  `exit 0` 不关机（[:59-62](experiments/opd_sft_h1/scripts/watchdog_cycle08.sh#L59-L62)）；否则 `shutdown`（[:66-67](experiments/opd_sft_h1/scripts/watchdog_cycle08.sh#L66-L67)）。开跑前先清掉上一轮的 stale abort 标志（[:31](experiments/opd_sft_h1/scripts/watchdog_cycle08.sh#L31)）。
  > 实操：`touch /root/autodl-tmp/cycle08_opd_trajectory/ABORT_SHUTDOWN` 就能在你醒来后保住机器。

### L1.2 run_cycle08.py 骨架：main() 的相位门与「接力棒」

[run_cycle08.py:501-534](experiments/opd_sft_h1/scripts/run_cycle08.py#L501-L534) 的 `main()` 是真正的调度中枢。它把五个相位用**门控 if** 串起来：

```text
main() 相位门（run_cycle08.py:516-533）
  argparse: --start-from-phase ∈ {2,3,4,5}, 默认 2        (:502-508)
  if start <= 2:  phase2_train() → write train_info.json → phase2b_convert()   (:516-521)
  else:           train_info = json.load(train_info.json)                       (:522-523)
  if start <= 3:  phase3_eval(train_info)                                        (:525-527)
  if start <= 4:  phase4_geometry(train_info)                                    (:528-530)
  always:         aggregate(train_info)                                          (:532-533)
```

两个设计点必须看到具体代码：

- **`train_info.json` 是相位间的「接力棒」**：Phase 2 训练时算出网格/步数/save_freq 等元信息，`main()`
  在 [run_cycle08.py:519](experiments/opd_sft_h1/scripts/run_cycle08.py#L519) 用 `write_json(info_path, train_info)` 落盘；续跑时（`--start-from-phase 3/4/5`）
  在 [run_cycle08.py:523](experiments/opd_sft_h1/scripts/run_cycle08.py#L523) 直接 `json.loads` 读回，**不重新训练**。所以你可以只重跑评测/几何而不碰训练。
- **大量复用 Cycle 07**：[run_cycle08.py:48](experiments/opd_sft_h1/scripts/run_cycle08.py#L48) 的 `from scripts import run_cycle07 as c07`，再在 [:51-57](experiments/opd_sft_h1/scripts/run_cycle08.py#L51-L57)
  把 c07 的叶子函数（`ensure_dir`/`write_json`/`step_label`/`binom_se`、`BASE_MODEL`）别名过来。这就是为什么
  Phase 3/4/5 能「逐字节复用 c07 协议」—— 它们调的就是同一批函数（L0.3 所说）。
- **关键坐标常量**集中在 [run_cycle08.py:58-78](experiments/opd_sft_h1/scripts/run_cycle08.py#L58-L78)：`VERL_PY`（[:58](experiments/opd_sft_h1/scripts/run_cycle08.py#L58)，verl 解释器）、`LAUNCH_SH`（[:59](experiments/opd_sft_h1/scripts/run_cycle08.py#L59)，
  即 L2 的 `run_cycle08_opd.sh`）、`CONVERT_PY`（[:60](experiments/opd_sft_h1/scripts/run_cycle08.py#L60)）、`CHECKPOINT_STEPS=[5,10,20,40,80,160,320,480]`（[:75](experiments/opd_sft_h1/scripts/run_cycle08.py#L75)）、
  `FINAL_STEP=624`（[:76](experiments/opd_sft_h1/scripts/run_cycle08.py#L76)）、`SAVE_FREQ=5`（[:77](experiments/opd_sft_h1/scripts/run_cycle08.py#L77)）、`TOTAL_EPOCHS=2`（[:78](experiments/opd_sft_h1/scripts/run_cycle08.py#L78)）。

### L1.3 skip-if-exists：每个相位都能幂等重入

这是 watchdog「无脑重启」和 `--start-from-phase` 能work的底层保证：**每个相位在做任何一个 checkpoint 前，
先检查产物是否已存在，存在就跳过**。四处检查各不相同，都值得看一眼：

| 相位 | 跳过条件（存在即 skip） | 代码 |
|---|---|---|
| Phase 2 train | 最终 checkpoint `global_step_624/actor` 目录已存在 | [run_cycle08.py:112-114](experiments/opd_sft_h1/scripts/run_cycle08.py#L112-L114) |
| Phase 2b convert | 合并目录里 `config.json` + 任一 `*.safetensors` 都在 | [run_cycle08.py:175-177](experiments/opd_sft_h1/scripts/run_cycle08.py#L175-L177) |
| Phase 3 eval | 该 step 所有生成任务 json 都在 **且** lm_eval 都完成 | [run_cycle08.py:244-248](experiments/opd_sft_h1/scripts/run_cycle08.py#L244-L248) |
| Phase 4 geometry | 每层 GetSlice 的 `sMat/xMat json` 在 **且** 导出的 `.npy` 数量够 | [run_cycle08.py:323-333](experiments/opd_sft_h1/scripts/run_cycle08.py#L323-L333) |

配合 verl 自身的 `trainer.resume_mode=auto`（L2 会讲），训练进程被杀后重启会从最近 checkpoint 接着跑，
而 `run_cycle08.py` 这一层则跳过已完成的整个相位/checkpoint。两层续跑叠加，才有「崩了直接重启」的鲁棒性。

### L1.4 双环境切换的具体实现（density ↔ verl）

L0.4 说了「为什么」要两个 env，这里看「怎么切」——**靠子进程 + 环境变量**，不是 `conda activate`：

- **本体在 density**：`main()` 及 Phase 3/4 的 worker 用 `sys.executable`（就是启动 `run_cycle08.py` 的
  density 解释器）。例如 pruner 用 `sys.executable` 起（[run_cycle08.py:142](experiments/opd_sft_h1/scripts/run_cycle08.py#L142)）。
- **切到 verl（训练）**：`phase2_train()` 在 [run_cycle08.py:120-121](experiments/opd_sft_h1/scripts/run_cycle08.py#L120-L121) 先 `env = dict(os.environ)`，再把
  verl 的 bin 目录**前置**进 `PATH`；然后 [:122-130](experiments/opd_sft_h1/scripts/run_cycle08.py#L122-L130) 把训练超参塞进**环境变量**
  （`TRAIN_PARQUET`/`CKPT_DIR`/`SAVE_FREQ`/`TOTAL_EPOCHS`/`TOTAL_TRAINING_STEPS`/`USE_POLICY_GRADIENT`）；
  最后 [run_cycle08.py:146](experiments/opd_sft_h1/scripts/run_cycle08.py#L146) `subprocess.run(["bash", LAUNCH_SH], env=env, ...)`。
  > 这就是 density→verl 的**传参通道**：不是命令行参数，而是环境变量，由 `run_cycle08_opd.sh` 里的
  > `${TRAIN_PARQUET:-默认}` 这类写法接收（L2 详解）。这样 orchestrator 能覆盖 shell 脚本的默认值。
- **切到 verl（转换）**：`phase2b_convert()` 在 [run_cycle08.py:183](experiments/opd_sft_h1/scripts/run_cycle08.py#L183) 直接用 `VERL_PY`（verl 解释器绝对路径）
  跑 `cycle08_convert_ckpt.py` —— 因为 verl 的 `model_merger` 需要 torch2.9 + verl。

一句话：**整条编排在 density 里跑，只有「训练」和「转换」这两步 shell out 到 verl**，其余（数据、评测、
几何、汇总）都在 density。

### L1.5 磁盘控制：在线 pruner + 原始分片回收（双保险）

这是 L1 里最容易被忽略、但对 624 步长跑生死攸关的一环。

**为什么需要**（[cycle08_ckpt_pruner.py:4-9](experiments/opd_sft_h1/scripts/cycle08_ckpt_pruner.py#L4-L9) docstring）：verl 每个 checkpoint 存的是 **FULL model
state dict**（4B 模型 ~16 GB fp32，**不是**那个小小的 LoRA adapter）。`save_freq=5` 跑 624 步 ≈ 125 个
checkpoint × 16 GB ≈ **2 TB**，远超磁盘。而我们只需要网格上的 ~9 个（`{5,10,…,480,624}`）。

**pruner 怎么删**（[cycle08_ckpt_pruner.py:45-57](experiments/opd_sft_h1/scripts/cycle08_ckpt_pruner.py#L45-L57)）：
- 每 20 秒（[:40](experiments/opd_sft_h1/scripts/cycle08_ckpt_pruner.py#L40) 的 `--poll`）扫一遍 `global_step_*` 目录（`_steps()`，[:27-33](experiments/opd_sft_h1/scripts/cycle08_ckpt_pruner.py#L27-L33)）；
- `cur_max = steps[-1]`（当前最大步）**永远不删**（[:48](experiments/opd_sft_h1/scripts/cycle08_ckpt_pruner.py#L48)，它可能正在写）；
- 删除既不是 `cur_max`、又不在 grid 里的目录（[:49-54](experiments/opd_sft_h1/scripts/cycle08_ckpt_pruner.py#L49-L54)），每删一个回收 ~16 GB；
- **绝不 crash**：整个循环包在 `try/except` 里（[:55-56](experiments/opd_sft_h1/scripts/cycle08_ckpt_pruner.py#L55-L56)，注释「a pruner must never crash the run」）。
- 「completed」的判据（[:11-12](experiments/opd_sft_h1/scripts/cycle08_ckpt_pruner.py#L11-L12) docstring）：有更高的 step 目录出现 = verl 已经 move on = 该目录写完、可安全删。

**谁启动/收尾**：`phase2_train()` 在 [run_cycle08.py:141-143](experiments/opd_sft_h1/scripts/run_cycle08.py#L141-L143) 用 `subprocess.Popen` 起 pruner（density 进程，
与训练**并行**盯着 ckpt 目录），并在 [:147-148](experiments/opd_sft_h1/scripts/run_cycle08.py#L147-L148) 的 `finally` 里 `pruner.terminate()`（训练正常结束或抛异常都收掉）。

**第二道保险**：即便 pruner 漏了，`phase2b_convert()` 在把 raw 分片转成合并 HF 模型后，会再删原始分片
（[run_cycle08.py:189-192](experiments/opd_sft_h1/scripts/run_cycle08.py#L189-L192)）并清掉所有非网格 raw checkpoint（[:194-201](experiments/opd_sft_h1/scripts/run_cycle08.py#L194-L201)）。

### L1.6 数据准备：cycle08_data_prep.py（一次性前置，钉死可比性）

这一步不在 `run_cycle08.py` 里自动调用 —— `phase2_train()` 只在 [run_cycle08.py:116-118](experiments/opd_sft_h1/scripts/run_cycle08.py#L116-L118) 检查 parquet
是否存在，不存在就**报错让你先手动跑** `cycle08_data_prep.py`。它是整个 OPD-vs-SFT 可比性的地基：

- **OPD 只要 prompts**（[cycle08_data_prep.py:4-6](experiments/opd_sft_h1/scripts/cycle08_data_prep.py#L4-L6) docstring）：因为 on-policy —— student 自己采样、teacher
  打分，所以训练数据是「纯 prompt」，不像 c07 SFT 需要 question+response 对。
- **复现 c07 完全相同的样本**：[cycle08_data_prep.py:40-41](experiments/opd_sft_h1/scripts/cycle08_data_prep.py#L40-L41) 用 `df.sample(n=5000, random_state=42)`，
  与 c07 同种子同规模。
- **硬校验防漂移**（[cycle08_data_prep.py:44-52](experiments/opd_sft_h1/scripts/cycle08_data_prep.py#L44-L52)）：抽样的第 0 条 question 必须和 c07 的
  `probe_rows.jsonl[0]` 前 80 字符一致，否则 `raise SystemExit`（[:52](experiments/opd_sft_h1/scripts/cycle08_data_prep.py#L52)）拒绝写出 —— 宁可报错也不写出
  「看起来对齐、其实错位」的 prompt。这是保证 OPD 与 SFT 逐 prompt 可比的保险丝。
- **verl parquet schema**（[cycle08_data_prep.py:58-64](experiments/opd_sft_h1/scripts/cycle08_data_prep.py#L58-L64)）：`data_source / prompt(messages) / ability /
  reward_model{style,ground_truth} / extra_info`。注意 `ground_truth` 存了但 **OPD 忽略**
  （[:57](experiments/opd_sft_h1/scripts/cycle08_data_prep.py#L57) 注释：`use_task_rewards=False`）—— 它只是为了填满 verl 期望的 schema。
- 产物：`opd_prompts_5k.parquet` + `data_meta_opd.json`（[:66-74](experiments/opd_sft_h1/scripts/cycle08_data_prep.py#L66-L74)）。

### L1.7 GPU 工作的「调度」（编排视角，非计算本身）

Phase 3/4 确实用 GPU，但**如何把工作铺到两张卡上**属于编排，值得在 L1 点一下（真正的评测/几何计算复用 c07，本次略过）：

- 用 `ProcessPoolExecutor` **每卡一个 checkpoint 并行**：eval 在 [run_cycle08.py:261-263](experiments/opd_sft_h1/scripts/run_cycle08.py#L261-L263)，geometry 在 [:344-347](experiments/opd_sft_h1/scripts/run_cycle08.py#L344-L347)；
- 每个 worker 进程**钉一张卡**：`_eval_one_checkpoint` 在 [run_cycle08.py:214](experiments/opd_sft_h1/scripts/run_cycle08.py#L214) 设 `CUDA_VISIBLE_DEVICES`，
  `_geometry_one_checkpoint` 在 [:295](experiments/opd_sft_h1/scripts/run_cycle08.py#L295) 同理；用 `i % workers` 轮流分卡。
- 每模型 `tensor_parallel_size=1`（[:213](experiments/opd_sft_h1/scripts/run_cycle08.py#L213) 注释），保证结果与 c07 单模型一致，只是同时飞 2 个 checkpoint、
  把两张 96G 都用上，墙钟时间约减半。

### L1 小结：这一层的心智模型

```text
一次 run 的生命周期（控制流视角）：
  [手动一次] cycle08_data_prep.py  →  opd_prompts_5k.parquet（钉死与 c07 对齐 + 硬校验）
  [启动]     setsid nohup watchdog_cycle08.sh
               └─ density python run_cycle08.py            （无 --start-from-phase = 从 Phase2 起）
                    P2  切 verl env 跑训练  ‖  并行 pruner 守盘   →  ~9 个网格 raw checkpoint
                    P2b 切 verl env 转换                          →  合并 HF 模型（删 raw）
                    P3  density 双卡并行评测（复用 c07）
                    P4  density 双卡并行几何（复用 c07）
                    P5  density CPU 汇总 → RESULTS_08.md / 对比表
               └─ 结束/崩溃 → 写 RUN_STATUS.txt → GRACE 倒计时 → shutdown（除非 touch ABORT）
  [任意时刻可续] run_cycle08.py --start-from-phase N   （靠 train_info.json 接力 + skip-if-exists 幂等）
```

**关键文件**：`scripts/watchdog_cycle08.sh`、`scripts/run_cycle08.py`、`scripts/cycle08_ckpt_pruner.py`、
`scripts/cycle08_data_prep.py`。

**下一层（L2）预告**：[run_cycle08.py:146](experiments/opd_sft_h1/scripts/run_cycle08.py#L146) 那句 `bash run_cycle08_opd.sh` 就是从「我们的编排」跨进
「verl 的世界」的门。L2 会拆开这个 shell 脚本构造的 7 组 Hydra 覆盖参数，看我们到底给 verl 递了一份什么「契约」。

---

## L2 — 训练启动层：交给 verl 的契约 ✅

> 本层主角是 [run_cycle08_opd.sh](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh)。它不训练，只做一件事——把 L1 通过环境变量传进来的参数，
> 翻译成 verl 能吃的一长串 **Hydra 覆盖参数**，然后 `python3 -m verl.trainer.main_ppo` 把控制权交出去。
> 换句话说，这一层是「我们的仓库」与「verl 的仓库」之间的**接口契约**：读懂它 = 读懂我们到底让 verl 做了什么。
> 另两个小文件：[cycle08_null_reward.py](experiments/opd_sft_h1/scripts/cycle08_null_reward.py)（让 verl 奖励回路空转不崩）、[apply_verl_patches.sh](experiments/opd_sft_h1/scripts/apply_verl_patches.sh)（给 verl 打的 fsdp2 补丁）。

### L2.1 入口一行：拼数组 → main_ppo

脚本尾部 [run_cycle08_opd.sh:184-192](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L184-L192) 是整层的落点：

```bash
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${ROLLOUT[@]}" \
    "${TRAINER[@]}" "${REWARD[@]}" "${DISTILL[@]}" "$@"
```

7 个 bash 数组（`DATA`/`MODEL`/`ACTOR`/`ROLLOUT`/`TRAINER`/`REWARD`/`DISTILL`）各自是一组 `key=value` 形式的
Hydra 覆盖项，展开后成为 `main_ppo` 的命令行参数，逐条覆盖 verl 默认配置树里的对应字段；`"$@"` 透传 L1 之外临时追加的参数。
**这就是「契约」的物理形态**：一份被覆盖后的 verl 配置。

- [run_cycle08_opd.sh:31](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L31) `set -xeuo pipefail`：`-x` 打印每条命令（日志里能看到最终展开的完整参数）、`-e` 出错即停、
  `-u` 未定义变量报错、`pipefail` 管道任一段失败即失败。
- [run_cycle08_opd.sh:32](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L32) `SCRIPT_DIR` 取绝对路径——因为 Hydra 运行时会改工作目录，后面引用 null_reward 脚本必须用绝对路径。
- [run_cycle08_opd.sh:35-36](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L35-L36)：若调用方传入了 `expandable_segments:True` 的 CUDA 分配器配置就**丢掉它**——vLLM V1 的显存池不兼容这个 allocator 旋钮（L6 显存主题的前哨）。

### L2.2 传参通道：环境变量 + `${VAR:-默认}`

脚本里每个可调参数都写成 `X=${X:-默认值}`：L1 的 `phase2_train()`（见 L1.4）通过**环境变量**注入的值优先，否则用脚本内默认。例如：

- 模型/数据路径 [run_cycle08_opd.sh:39-43](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L39-L43)：`STUDENT_MODEL`、`TEACHER_MODEL`、`TRAIN_PARQUET`（L1 注入）、`CKPT_DIR`（L1 注入）；
  `VAL_PARQUET` 默认指向同一份（[:42](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L42)，反正 `val_before_train=False` 用不到）。

所以「L1 用环境变量、L2 用 `${VAR:-默认}` 接收」是这两层唯一的耦合方式——**没有共享 Python 状态**，干净地跨了 density→verl 的进程/环境边界。

### L2.3 全场最重要：与 Cycle 07 SFT 的逐项对齐

这份契约里几乎每个数值都不是「verl 默认」，而是**为了让 OPD 轨迹和 c07 SFT 轨迹逐步可比**而反推的。
脚本头 [run_cycle08_opd.sh:6-16](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L6-L16) 的注释就是这份对齐说明书，逐条落到参数：

| 对齐意图 | c07 SFT | c08 OPD 参数 | 代码 |
|---|---|---|---|
| 同一批 prompt | seed=42 5000 样本 | `TRAIN_PARQUET`=opd_prompts_5k | [:41](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L41) |
| 每次更新消费的 prompt 数 | eff_batch 16 | `train_batch_size=ppo_mini_batch_size=16` | [:63-64](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L63-L64) |
| rollout 不复用（on-policy 纯度） | —（SFT 无此概念） | `ppo_epochs=1` | [:65](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L65) |
| 数据集扫几遍 | epoch=2 | `total_epochs=2` | [:66](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L66) |
| 评测终点步数 | step 624 | `total_training_steps=624` | [:67](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L67) |
| 序列长度归一化 | token 级 | `loss_agg_mode=token-mean` | [:68](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L68) |
| 学习率 | 5e-5 | `actor_lr=5e-5` | [:69](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L69) |
| 序列上限 | max_len 10240 | `max_response_length=10240` | [:73](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L73) |
| LoRA | r32/a64 all-linear | `lora_rank/alpha/target_modules` | [:77-79](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L77-L79) |

两个**极易混淆**的点，注释 [run_cycle08_opd.sh:10-13](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L10-L13) 特意点破：

- `ppo_epochs=1` **不是**「数据集只过一遍」——它是 PPO 语义里「同一批 rollout 复用几次做梯度」。设成 1 = 每批 rollout 只用一次就丢，
  保证**纯 on-policy**（学生每步都用当前权重现采）。「数据集过几遍」由 `total_epochs=2` 管。
- **624 步从哪来**：verl 过滤掉超长 prompt（`filter_overlong_prompts`）后 5000→4999 行，每步吃 16、`drop_last` →
  `floor(4999/16)=312` 步/epoch，×2 epoch = **624** 步，正好对上 c07 评测的最终点（交叉印证见 [run_cycle08.py:73-76](experiments/opd_sft_h1/scripts/run_cycle08.py#L73-L76)）。

### L2.4 七组覆盖参数逐组读

按数组顺序过一遍，只标注**非默认/有讲究**的项。

**DATA** [run_cycle08_opd.sh:92-104](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L92-L104) — 数据与优势估计
- `algorithm.adv_estimator=grpo`（[:93](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L93)）+ `use_kl_in_reward=False`（[:94](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L94)）：名义上是 GRPO，但见 L2.5——奖励恒 0，优势那套**空转**，
  留着只因为跑在 PPO/GRPO 主循环里。
- `filter_overlong_prompts=True` + `truncation='error'`（[:100-101](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L100-L101)）：超长 prompt 直接过滤（5000→4999），而非静默截断；
  `shuffle=False`（[:102](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L102)）保证与 c07 相同的样本顺序。

**MODEL** [run_cycle08_opd.sh:106-114](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L106-L114) — 学生模型与 LoRA
- `use_remove_padding=True`（[:108](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L108)）+ `attn_implementation=flash_attention_2`（[:109](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L109)）+
  `enable_gradient_checkpointing=True`（[:110](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L110)）：三个都是 L6 显存/吞吐旋钮，先记「变长打包 + flash-attn + 重算省显存」。
- LoRA `r=32 / alpha=64 / target_modules=all-linear`（[:111-113](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L111-L113)）：与 c07 完全一致。

**ACTOR** [run_cycle08_opd.sh:116-127](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L116-L127) — 学生训练器
- `strategy=fsdp2`（[:117](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L117)）：单卡训练**必须** fsdp2（fsdp1 单卡会踩 collect_lora_params 的坑，见 L2.7）。
- `use_kl_loss=False`（[:122](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L122)）：**关掉对参考策略的 KL**——只朝教师分布正则，不朝一个 ref policy（契约里根本没有 reference policy）。
- `use_dynamic_bsz=True` + `ppo_max_token_len_per_gpu=16384`（[:123-124](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L123-L124)）：按 token 数动态组 micro-batch，L6 详述。
- `param_offload=False` / `optimizer_offload=False`（[:125-126](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L125-L126)）：不 offload 到 CPU（96G 放得下 4B）。

**ROLLOUT** [run_cycle08_opd.sh:129-141](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L129-L141) — 学生采样引擎（vLLM）
- `name=vllm` + `tensor_model_parallel_size=1`（[:130-131](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L130-L131)）：单卡 vLLM。
- `gpu_memory_utilization=0.6`（[:132](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L132)）：**故意压低**——留余量给训练瞬时峰值和 vLLM sleep/wake（L6 重点）。
- `n=1`（[:133](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L133)）：每 prompt 采 1 条（OPD 不需要 GRPO 的多样本组）。
- `temperature=0.6 / top_p=0.9`（[:134-135](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L134-L135)）：学生采样温度。
- `load_format=safetensors` + `layered_summon=True`（[:137-138](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L137-L138)）：训练侧权重同步到 vLLM 的方式，L4/L6 讲。

**TRAINER** [run_cycle08_opd.sh:143-157](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L143-L157) — 驱动与 checkpoint
- `n_gpus_per_node=NGPUS_PER_NODE`（[:149](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L149)，=1）：**学生池**卡数（见 L2.6）。
- `val_before_train=False`（[:151](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L151)）+ `test_freq=100000`（[:83](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L83)）：全程不在训练里做 eval——评测是 L1 的独立 Phase 3。
- `resume_mode=auto`（[:152](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L152)）：进程被杀重启后从最近 checkpoint 续跑（配合 L1.3 的 skip-if-exists = 双层续跑）。
- `save_freq=5`（[:82](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L82)）：每 5 步存一个 checkpoint，抓早期网格步 {5,10,20,...}（配合 L1.5 的 pruner 只留网格）。

**REWARD** [run_cycle08_opd.sh:159-165](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L159-L165) — 见 L2.5。

**DISTILL** [run_cycle08_opd.sh:167-182](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L167-L182) — 蒸馏与教师（本 cycle 的灵魂）
- `distillation.enabled=True`（[:168](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L168)）：打开 verl 的蒸馏扩展。
- `n_gpus_per_node=TEACHER_WORLD_SIZE`（[:169](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L169)，=1）：**教师池**卡数，与学生池不相交（L2.6/L4）。
- teacher 模型 [:171-175](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L171-L175)：路径、`tensor_model_parallel_size=1`、`inference.name=vllm`、`gpu_memory_utilization=0.80`、`max_model_len`。教师用 vLLM 独立推理。
- `loss_mode=forward_kl_topk`（[:176](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L176)）+ `topk=32`（[:177](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L177)）：教师加权的 top-32 前向 KL。**这就是 OPD 的损失**，数学留 L5。
- `use_task_rewards=False`（[:178](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L178)）：**忽略任务奖励**——纯蒸馏，与 REWARD 组的空转互为印证。
- `use_policy_gradient=False`（[:179](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L179)）：**监督式**（直接反传 KL），不是策略梯度。全实验最关键的开关，L5 展开为什么。
- `loss_max_clamp=10.0` / `log_prob_min_clamp=-10.0`（[:180-181](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L180-L181)）：数值稳定的钳制，L5/L6 提。

### L2.5 cycle08_null_reward.py：让奖励回路空转

[cycle08_null_reward.py](experiments/opd_sft_h1/scripts/cycle08_null_reward.py) 全文只有一个函数 [cycle08_null_reward.py:11-12](experiments/opd_sft_h1/scripts/cycle08_null_reward.py#L11-L12)：

```python
def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    return 0.0
```

为什么需要它（docstring [cycle08_null_reward.py:1-8](experiments/opd_sft_h1/scripts/cycle08_null_reward.py#L1-L8)）：verl 的 V1 RewardLoopWorker **总会运行**，按 `data_source` 分派任务奖励函数；
我们的 `data_source="Math-CoT-20k"` 未注册，会直接 `NotImplementedError` 崩掉。既然 OPD 是纯蒸馏（`use_task_rewards=False`，奖励值根本不用），
就塞一个恒返回 0 的打分器让回路空转而非崩溃。`**kwargs` 是为了容忍不同 verl 版本的签名差异。

> 串起来看：REWARD 组把 `custom_reward_function.path` 指到这个文件（[run_cycle08_opd.sh:163-164](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L163-L164)），
> 于是「GRPO 优势 + 奖励回路」结构上还在、数值上恒 0——这正是 L0.1 说的「跑在 RL 框架里，但不是 RL」。

### L2.6 GPU 池拓扑：学生 1 卡 + 教师 1 卡（不相交）

契约里最容易读错的是 GPU 数。注释 [run_cycle08_opd.sh:45-48](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L45-L48) 说清楚：verl 把**学生池**（`trainer.n_gpus_per_node`）和
**教师池**（`distillation.n_gpus_per_node`）**分别**求和、两池不相交。

- `NGPUS_PER_NODE=1`（[:50](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L50)）：学生 4B 在 1 张 96G 上**不分片**训练 + 同卡 vLLM rollout（混合引擎，L4）。
- `TEACHER_WORLD_SIZE=1`（[:51](experiments/opd_sft_h1/scripts/run_cycle08_opd.sh#L51)）：教师 8B 独占**另一张** 96G 做 vLLM 推理。
- 总卡数 = 1 + 1 = **2**（不是 2+2，也不是 1）。注释明确警告：单卡 colocate（学生+教师挤一张卡）是本 cycle 头号风险，当前用「各占一卡」规避（L4/L6 详述）。

### L2.7 apply_verl_patches.sh：单卡必须的 fsdp2 补丁

[apply_verl_patches.sh](experiments/opd_sft_h1/scripts/apply_verl_patches.sh) 幂等地给 verl 源码打补丁（verl 是 autodl-tmp 上的可编辑安装，重拷后补丁会丢）。当前只有 patch1
（[apply_verl_patches.sh:7-23](experiments/opd_sft_h1/scripts/apply_verl_patches.sh#L7-L23)）：

- 问题：verl 的 [losses.py](file:///root/autodl-tmp/verl/verl/trainer/distillation/losses.py) 里 `compute_topk_loss()` 的策略分派只匹配了 `case "fsdp" | "veomni"`，strategy=`fsdp2` 会抛「Unsupported strategy」。
- 补丁：把那行改成 `case "fsdp" | "fsdp2" | "veomni"`（[apply_verl_patches.sh:21](experiments/opd_sft_h1/scripts/apply_verl_patches.sh#L21) 的 sed），因为 fsdp2 的 loss 计算与 fsdp 相同（verl 自己注释也这么说）。
- 为什么非 fsdp2 不可：单卡用 fsdp1 会在 collect_lora_params 撞上「offload_to_cpu=True and NO_SHARD is not supported」——与 ACTOR 组 `strategy=fsdp2`（L2.4）是同一件事的两面。

> 实操提醒：这个补丁**不在 pipeline 里自动跑**。任何一次 verl 被重新同步/拷贝后，都要先手动 `bash apply_verl_patches.sh` 再训练，
> 否则一进蒸馏 loss 就报 Unsupported strategy。

### L2 小结 + L3 预告

```text
L1 环境变量  ──►  run_cycle08_opd.sh 的 ${VAR:-默认}  ──►  7 组 Hydra 覆盖
                                                              │
   （对齐 c07 的数值：16 / 1 / 2 / 624 / token-mean / LoRA r32）
   （空转的 RL 外壳：grpo + null_reward + use_task_rewards=False）
   （真正的信号：forward_kl_topk + topk32 + use_policy_gradient=False）
   （拓扑：学生 1 卡混合引擎 + 教师 1 卡，两池不相交）
                                                              ▼
                              python -m verl.trainer.main_ppo  ← 控制权交给 verl（L3）
```

契约递出去之后，`main_ppo` 拿这份配置做了什么？**L3** 打开 verl 源码，看它怎么 `ray.init`、怎么按这两个池建 worker group、
`fit()` 循环怎么把「学生 rollout → 教师 top-k → forward_kl_topk → FSDP2 更新」串起来。

---

## L3 — verl 训练器层：main_ppo → TaskRunner → fit ⬜ 待深入

**这一层回答**：verl 收到配置后如何起 Ray、构造 resource pool、组织训练主循环
（generate → teacher logprob → loss → actor update → save）？蒸馏分支落在循环哪一步？

**范围**：`verl/trainer/main_ppo.py` 的 `@hydra.main`/`run_ppo`/`TaskRunner`；PPO trainer 的 `fit()`；
distillation 在 V1（`AgentLoopManager` / `TransferQueue`）路径里的接入点。

**关键文件**：`/root/autodl-tmp/verl/verl/trainer/main_ppo.py`、`/root/autodl-tmp/verl/verl/trainer/ppo/*`。

_（正文待补全，需打开 verl 源码核实确切调用。）_

---

## L4 — worker / 并行层：角色与混合引擎 ⬜ 待深入

**这一层回答**：谁在哪张卡上跑？student 的「FSDP2 训练 + vLLM rollout」混合引擎如何在同一张卡上
sleep/wake 与同步权重？teacher 独立池如何算 top-k logprob 并回传？两个不相交 GPU 池怎么分配？

**范围**：`ActorRolloutRefWorker`（student）、teacher worker group、reward loop worker；
`layered_summon` 权重召唤；vLLM sleep 模式；student/teacher 池 = `trainer.n_gpus_per_node` +
`distillation.n_gpus_per_node`（不相交求和）。单卡 colocate 是本 cycle 的头号风险，此层说明它如何被规避。

**关键文件**：verl workers、`/root/autodl-tmp/verl/verl/trainer/distillation/fsdp/*`、`run_cycle08_opd.sh` 拓扑块。

_（正文待补全。）_

---

## L5 — 蒸馏损失层：科学内核 ⬜ 待深入

**这一层回答**：`forward_kl_topk` 具体算什么？
`L = Σ_v teacher_p(v)·(log teacher_p(v) − log student_p(v))`（在教师 top-k 上）的梯度为什么依赖教师概率、
因而是「真正的蒸馏信号」？为什么 `use_policy_gradient=False` 是最干净的「监督 OPD ≈ GKD」，
而 verl 会拒绝 `use_policy_gradient=False + loss_mode=k1`？`loss_max_clamp`/`log_prob_min_clamp` 与 topk=32 的作用？

**范围**：`verl/trainer/distillation/losses.py` 的 `compute_topk_loss()` 及各 `loss_mode`；
监督路径 vs 策略梯度路径（advantages = −loss.detach()）的分野。

**关键文件**：`/root/autodl-tmp/verl/verl/trainer/distillation/losses.py`、`run_cycle08_opd.sh` 的 DISTILL 块。

_（正文待补全。）_

---

## L6 — GPU / 系统层：金属层 ⬜ 待深入

**这一层回答**：2×96G 上显存怎么排布？为什么 `rollout_gpu_memory_utilization=0.6`、
`ppo_max_token_len_per_gpu=16384`、gradient checkpointing、`use_dynamic_bsz`、`use_remove_padding`、
`flash_attention_2` 这些旋钮是这么拧的？vLLM sleep→wake 的 KV 重分配 OOM、长序列训练瞬时峰值、
checkpoint I/O（每个 ~16GB、`layered_summon`、LoRA 参数收集）各自的失败模式与权衡是什么？

**范围**：`run_cycle08_opd.sh` 里所有显存/OOM 注释所沉淀的工程决策；verl FSDP2 worker 的分片与 offload；
`cycle08_convert_ckpt.py` 的分片合并 I/O。

**关键文件**：`run_cycle08_opd.sh`（显存/OOM 注释）、verl FSDP workers、`scripts/cycle08_convert_ckpt.py`。

_（正文待补全。）_

---

## 讨论区 / 待澄清

> 逐层深入时，把确认的结论、推翻的猜测、你补充的背景记在这里；每层定稿后我再进下一层。

- **(L0, 我的假设，待你确认)** 本次精读主线是**训练**（L1–L6 的 Phase 2/2b + verl 内部）；
  Phase 3–5（eval/geometry/aggregate）因复用 Cycle 07 而略过，只在交叉处点到。若你想把评测/几何也纳入，
  告诉我，我再加一条支线。
- **(L0)** verl 内部确切调用（`TaskRunner` 版本、`fit()` 中蒸馏分支落点）在 L0 按黑盒处理，L3 打开源码核实。

## 变更记录

- 初版：建立 L0–L6 分层阶梯 + 写完 L0（鸟瞰、五阶段、双环境、自顶向下调用链、关键坐标）；L1–L6 立为待深入桩。
- L1 定稿：编排层全文（watchdog 监工/自动关机、main() 相位门与 train_info.json 接力、skip-if-exists 四处幂等、
  density↔verl 子进程+环境变量切换、在线 pruner+分片回收双保险、data_prep 硬校验、Phase3/4 双卡调度、L1 生命周期小结）；
  全程标注 `文件:行`。
- 全文代码引用改为**可点击跳转链接**（仓库文件相对路径 + `#L` 行锚点；verl 源码用 `file://` 绝对链接）。
- L2 定稿：训练启动契约层全文（main_ppo 入口一行 + 7 组数组、环境变量↔`${VAR:-默认}` 传参通道、与 c07 逐项对齐表
  + `ppo_epochs=1` vs `total_epochs=2` 与 624 步推导、七组覆盖参数逐组读、null_reward 空转奖励回路、
  学生/教师不相交 GPU 池拓扑、fsdp2 补丁与「非 fsdp2 不可」、L2 小结+L3 预告）；全程可点击链接。
