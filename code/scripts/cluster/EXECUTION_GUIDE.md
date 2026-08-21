# SOPPO v0.6 SSPO-aligned 30k：服务器执行指南

> 当前状态：`AUTHORIZED`。实现基线 `e047ce7` 已完成代码交接，用户于 2026-08-21 明确要求挂载任务；提交时仍须使用包含该基线及授权记录的 clean、commit-locked checkout。

## 1. 路径只记这一条

```bash
export SERVER_BASE=/home-ssd/Users/nsgm_jiangwh/youchang
```

服务器结构必须为：

```text
<SERVER_BASE>/
├── SOPPO/                 # 唯一 Git repo
├── ICLR/                  # 与 SOPPO 平级，静态且无 Git
├── envs/youc/
├── cache/
├── data/
├── models/
├── runs/
└── exports/
```

不存在 `<SERVER_BASE>/ICLR/SOPPO`；也不要给 `ICLR` 或 `LLM-output-density` 建 Git。

## 2. 每次登录后的初始化

直接复制：

```bash
export SERVER_BASE=/home-ssd/Users/nsgm_jiangwh/youchang
export RUN_CONTEXT=cluster
cd "$SERVER_BASE/SOPPO/code/scripts/cluster"
```

环境建立以后，可再执行：

```bash
source "$SERVER_BASE/activate_env.sh"
```

`activate_env.sh` 不在 Git 中，由 `00_server_setup.sh` 首次成功运行后生成。看不到它时先运行第 4 节的环境准备，不要手工创建。

## 3. 先更新唯一仓库

```bash
export SERVER_BASE=/home-ssd/Users/nsgm_jiangwh/youchang
cd "$SERVER_BASE/SOPPO"
git pull --ff-only origin master
git status --short
git rev-parse --short HEAD
```

`git status --short` 应无输出。`submit_all.sh` 会拒绝 dirty checkout，以保证 task registry 能固定唯一 commit。

## 4. 首次准备的三个前置步骤（此前已在 gn001 完成）

这些步骤不请求 GPU，也不需要 SSH `gn006`。`gn001` 是本轮首次建环境、下载和数据准备时使用的编译入口，不是 `submit_all.sh` 的硬编码提交主机；环境、模型和数据均位于共享存储，已经成功完成后无需为了提交 DAG 再进入 `gn001`。

### 4.1 环境

```bash
export SERVER_BASE=/home-ssd/Users/nsgm_jiangwh/youchang
export RUN_CONTEXT=cluster
cd "$SERVER_BASE/SOPPO/code/scripts/cluster"
bash 00_server_setup.sh
```

脚本会加载已验证的 `miniforge3/25.11.0-0`，复用或创建 Python 3.10 的 `$SERVER_BASE/envs/youc`，并安装锁定依赖（包括新增的 `peft==0.15.2`）。不要再加载损坏的 `python/3.10.4` module；即使旧环境已经存在，也要重跑一次本步骤以补齐 PEFT。

如果安装因网络暂时超时，直接重跑 `00_server_setup.sh`；它会复用 env 与 cache，不要删除整个环境。

### 4.2 下载并冻结 Qwen3

```bash
source "$SERVER_BASE/activate_env.sh"
cd "$SERVER_BASE/SOPPO/code/scripts/cluster"
export RUN_CONTEXT=cluster
bash 02_download_model.sh
```

成功标志：

```bash
test -f "$SERVER_BASE/models/Qwen3-4B/model_manifest.json" && echo MODEL_OK
```

脚本通过 ModelScope 下载 `Qwen/Qwen3-4B`，先写临时目录，校验 Qwen3/36 层/safetensors/逐文件 SHA-256 后再原子冻结。

### 4.3 准备新的 30k 数据

```bash
source "$SERVER_BASE/activate_env.sh"
cd "$SERVER_BASE/SOPPO/code/scripts/cluster"
export RUN_CONTEXT=cluster
bash 02_prepare_data.sh
```

默认 endpoint 为已在本服务器实测可达的 `https://hf-mirror.com`。新输出是：

```text
<SERVER_BASE>/data/ultrafeedback/mvp-v0.5-30k
```

旧 `mvp-v0.3` 不会被覆盖。成功 manifest 必须是：

```text
labeled train  = 2,700  (9%)
labeled val    =   300  (1%)
unlabeled      = 24,000 (80%)
test           = 3,000  (10%)
labeled total  = 3,000  (10%)
```

脚本会自行验证这些数量/比例、逐文件 SHA-256、公开文件隐藏标签、私有标签 ID 精确连接、A/B 随机化和跨 split 泄漏；已有 30k 目录也必须通过同一审计才会复用。

## 5. 提交前检查存储与资源

本轮 checkpoint 已改为 LoRA adapter，空间需求远小于旧版每点保存4B全模型的1–2TB估算，但仍要区分全盘余量与个人 quota：

```bash
df -h /home-ssd
quota -s 2>/dev/null || true
command -v lfs >/dev/null 2>&1 && lfs quota -u "$USER" /home-ssd 2>/dev/null || true
```

如果集群不提供 quota 命令，只能联系管理员确认用户/项目配额；不要仅依据 `df` 推断个人可写空间。

查看当前测试/正式节点：

```bash
sinfo -N -o '%N|%P|%T|%G|%C'
```

无需等待节点显示 idle 才提交；Slurm 会排队。脚本请求卡而不硬编码节点，正式 job 获得 GPU 后还会检查实际型号。

## 6. 从可用 Slurm 提交节点一条命令提交全部剩余任务

提交节点不固定为 `gn001`。只要当前登录节点同时满足以下条件即可运行 `submit_all.sh`：能访问 `<SERVER_BASE>` 共享存储；`sbatch`/`scancel` 可用并能连接 Slurm controller；能激活共享的 `envs/youc`；服务器 SOPPO checkout clean。当前 `mn006` 已满足共享存储、Git、`sbatch` 和 `squeue` 条件，可作为 `gn001` 故障期间的提交入口。训练仍由 Slurm 分配到 GPU 节点，不在 `mn006` 本机执行。

本轮服务器执行授权已于 2026-08-21 明确记录。完成失败 DAG 归档并更新到 clean checkout 后执行：

```bash
export SERVER_BASE=/home-ssd/Users/nsgm_jiangwh/youchang
source "$SERVER_BASE/activate_env.sh"
cd "$SERVER_BASE/SOPPO/code/scripts/cluster"
export RUN_CONTEXT=cluster
bash submit_all.sh
```

不要再逐个手工运行 `01`、`03`、`04`……。实测当前账户只有 `nsgm_jiangwh|gpu|normal` 关联，集群拒绝普通用户的 `sbatch --hold` 和 typed `--gres=gpu:tesla:N`，但已实测接受 `sbatch -G 1` 与 `sbatch -G 2`。因此提交器将所有阶段路由到 `gpu` partition，并统一使用 `-G N`：辅助阶段申请1张卡，smoke与正式训练申请2张卡；所有 job 直接按 `afterok` 依赖提交。若中途某个 `sbatch` 被拒绝，提交器会自动取消本次已经提交的 job。完整任务图为：

```text
CPU tests
  → gpu strong smoke (2×A800, bf16/2048 training-split length gate, 90m)
  → formal reference/oracle (2×A800)
  → DPO-10 + DPO-100 final array
  → DPO-10 vs frozen-base headroom gate
  → four normalized fixed-lambda PE final runs → validation selection
  → SSPO-hard-exp + SOPPO-PE-exp final runs
  → C_ε prepare/base+8 adapters/derive
  → independent evaluation of all 8 adapters
  → aggregate/export
```

所有依赖为 `afterok`；任何上游失败都会阻止下游，不要求 SSH 会话持续在线。现在没有双卡资源不影响提交，2-GPU job 会以 `PD (Resources/Priority)` 等待。默认最多同时运行 4 个 array 单元，仍取决于集群调度和用户额度。

提交器默认加 `--exclude=gn005,gn021`：`gn005` 已在本轮出现 DDP/NCCL timeout且处于 draining，`gn021` 延续此前已验证的人工排除。若管理员确认节点已恢复，可显式设置 `export SOPPO_EXCLUDE_NODES=` 清空；也可提供逗号分隔的新名单。

`submit_all.sh` 会自行激活锁定环境，并在提交任何 job 前重验 Qwen3 manifest 与 30k 数据审计；它也拒绝覆盖已存在的 pipeline 目录。如果尚未提交任何 job 就失败，脚本会清理由本次创建的空目录；若部分 job 已提交后失败，则自动取消这些 job并保留 pipeline 目录供检查，不要直接删除。

Slurm 实际执行的是 `/var/spool/slurmd/.../slurm_script` 副本。提交器会通过 `SOPPO_CLUSTER_SCRIPT_DIR` 给所有 worker 传递仓库中的真实 cluster 目录；不得删除这一 export，否则 worker 会在 spool 目录下错误寻找 `job_env.sh`。

每个 worker 还会核对 `submit_all.sh` 提交时记录的完整 Git commit 和 clean checkout。DAG 排队或运行期间不要再次 `git pull` 或修改服务器仓库；否则尚未启动的 worker 会明确失败，防止同一 pipeline 混用不同代码版本。PyTorch allocator 使用 `expandable_segments:True`，并只使用 `HF_HOME`，不再触发 `TRANSFORMERS_CACHE` 的 v5 deprecation warning。

## 7. 日常查看状态

```bash
export SERVER_BASE=/home-ssd/Users/nsgm_jiangwh/youchang
cd "$SERVER_BASE/SOPPO/code/scripts/cluster"
bash status_pipeline.sh
```

也可以看自己的队列：

```bash
squeue -u "$USER" -o '%.18i %.12P %.24j %.2t %.10M %.30R'
```

`PD (Priority)` 表示已排队、等待优先级和资源，不是报错。计算节点由 Slurm 分配，不需要也通常不允许直接 SSH。

`status_pipeline.sh` 会把 array job 按 `RUNNING:n/PENDING:n` 汇总，同时列出各 array task 的节点或等待原因。

stage03/04/05 合计正好八条 final trajectories，不会把预实验重新训练一遍。headroom 比较同一个 margin-free mean-logp score 下、显式禁用 adapter 的训练前 Qwen3 与 DPO-10，并核对前后 score type/validation 样本数；DPO-100 只作为 oracle。

## 8. 失败后保留证据再重提

失败 DAG 的 pipeline 目录和 partial run 不得删除。先取消仍在 `PENDING (Dependency)` 的下游 job，再把旧目录移动到：

```text
<SERVER_BASE>/runs/<experiment>/failed_attempts/<attempt-name>/
```

至少保留旧 `pipeline/`（registry、hardware、logs）以及已创建的 `main/<arm>/`。完整 reference cache 位于 `<SERVER_BASE>/cache/`，与某次 pipeline 分离，校验通过后可直接复用。归档完成、服务器 checkout 更新到新 clean commit 后，重新运行 `submit_all.sh`；新的 full-length strong smoke 必须先通过，才会解锁 formal DPO。

## 9. checkpoint 与重启策略

- DPO-10/DPO-100：每 20 step 加最终点；DPO-10 预计保留 step20、step40 和 final。
- SSPO-hard/所有 PE：每 40 step 加最终点。
- `save_total_limit=null`，代码不自动删除。
- checkpoint 是 PEFT LoRA adapter，不再为每个点复制完整 Qwen3 base；`C_ε` 在内存中合并。
- adapter 不含 optimizer/scheduler state；参数可以重载继续微调，但不保证 optimizer 或 SSPO threshold EMA 的精确断点续跑。

当前 `--init-checkpoint` 只是 LoRA warm start，不是正式实验可接受的 exact resume：global step、optimizer、cosine scheduler、数据位置、RNG 和 SSPO KDE/EMA state 都会重置。因此本轮失败 arm 必须从头重跑，不能把 partial checkpoint 当作无缝续训。

## 10. 完成与回传

最终成功标志：

```bash
test -f "$SERVER_BASE/exports/exp-20260819-01-mvp/EXPORT_COMPLETE" && echo PIPELINE_OK
```

只回传该 export 目录中的白名单聚合文件，其中包含环境摘要、数据审计和聚合后的 `C_ε` 轨迹。不要回传数据、私有标签、模型/checkpoint、reference cache、原始训练日志、C_ε raw matrices 或逐样本 predictions。
