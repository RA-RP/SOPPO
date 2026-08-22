# SOPPO v0.6：独占服务器执行指南

本目录面向无 Slurm、由当前用户独占的服务器。训练算法、配置、数据契约和评价代码与旧服务器完全相同；这里只替换服务器路径、环境激活、GPU 路由和长任务编排。旧 `scripts/cluster/` 保留，不要删除，也不要在新服务器运行其中的 `submit_all.sh`。

## 1. 唯一路径与目录结构

每次登录先执行：

```bash
export SERVER_BASE=/data/youchang/youchang
export RUN_CONTEXT=standalone
cd "$SERVER_BASE/SOPPO/code/scripts/standalone"
```

服务器目录必须为：

```text
/data/youchang/youchang/
├── SOPPO/                 # 唯一 Git repo
├── ICLR/                  # 静态资料，与 SOPPO 平级且无 .git
├── envs/youc/
├── cache/
├── data/
├── models/
├── runs/
├── exports/
└── platform_logs/
```

`server_paths.sh` 会从 `SOPPO` 的实际位置反推基目录；若导出的 `SERVER_BASE` 与实际位置不一致，脚本会直接拒绝执行。不要创建 `/data/youchang/youchang/ICLR/SOPPO`，也不要给 `ICLR` 或 `LLM-output-density` 增加 `.git`。

## 2. 迁移前先处理旧服务器任务

新旧服务器不要同时运行同一个实验。旧集群若仍有当前 SOPPO DAG 排队，应在旧服务器用旧 registry 精确取消：

```bash
cd <OLD_SERVER_BASE>/SOPPO/code/scripts/cluster
export RUN_CONTEXT=cluster
bash cancel_pipeline.sh
bash cancel_pipeline.sh --execute
```

先看 dry-run 列出的 job ID，确认全部属于当前 SOPPO registry，再输入脚本要求的确认词。不要运行 `scancel -u "$USER"`，旧账户上还有同事任务。

## 3. 放置代码和静态资料

把唯一 Git 仓库放在：

```text
/data/youchang/youchang/SOPPO
```

把本地 `ICLR` 的静态参考资料放到平级目录：

```text
/data/youchang/youchang/ICLR
```

在新服务器确认：

```bash
git -C "$SERVER_BASE/SOPPO" status --short
git -C "$SERVER_BASE/SOPPO" rev-parse --short HEAD
test -d "$SERVER_BASE/SOPPO/.git" && echo SOPPO_GIT_OK
test ! -e "$SERVER_BASE/ICLR/.git" && echo ICLR_STATIC_OK
find "$SERVER_BASE/SOPPO/code/observe/LLM-output-density" -type d -name .git -print
```

最后一条必须没有输出。开始长链前 `git status --short` 也必须没有输出。

## 4. 首次建立环境

```bash
export SERVER_BASE=/data/youchang/youchang
export RUN_CONTEXT=standalone
cd "$SERVER_BASE/SOPPO/code/scripts/standalone"
bash 00_server_setup.sh
source "$SERVER_BASE/activate_env.sh"
```

脚本优先使用服务器已有的 `conda` 创建路径环境；若没有 conda，则使用 `python3.10 -m venv`。两者都没有时，先安装其中之一，或显式设置：

```bash
export SOPPO_CONDA_EXE=/absolute/path/to/conda
# 或
export SOPPO_PYTHON_BIN=/absolute/path/to/python3.10
```

依赖仍锁定为 Python 3.10、PyTorch 2.4.0/CUDA 12.1 和 `requirements.lock.txt` 中的版本。新服务器不加载旧集群的 Miniforge/proxy module。

## 5. 模型和数据

直接在新服务器重新获取并验证：

```bash
cd "$SERVER_BASE/SOPPO/code/scripts/standalone"
bash 02_download_model.sh
bash 02_prepare_data.sh
```

也可以把旧服务器中已经完整成功的以下目录复制到新服务器相同的相对位置：

```text
models/Qwen3-4B/
data/ultrafeedback/mvp-v0.5-30k/
cache/soppo/reference/qwen3-4b-mvp-v0.5-30k/   # 可选，必须包含 COMPLETE
```

复制后仍分别运行 `02_download_model.sh` 和 `02_prepare_data.sh`；发现完整 manifest 时它们只做校验，不会重新下载。不要复制 `.partial.*` 目录。完整 reference cache 可跨服务器复用，因为其门禁依据模型 manifest 与数据文件 SHA-256，而不是旧绝对路径。

若 Hugging Face 镜像不可用，可在数据准备前改为可达 endpoint：

```bash
export HF_ENDPOINT=https://huggingface.co
```

## 6. GPU 契约

默认设置是：

```text
正式训练：物理 GPU 0,1；torchrun 两进程
单卡后处理：物理 GPU 0
最低显存：每卡 79000 MiB
```

先查看本机：

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
```

如果希望使用其他两张卡，在启动前设置，例如：

```bash
export SOPPO_TRAIN_GPU_IDS=2,3
export SOPPO_POST_GPU_ID=2
```

当前实验的 global batch 与 DDP 契约固定为两张卡，因此 `SOPPO_TRAIN_GPU_IDS` 必须正好包含两个不同编号。脚本不再限定型号必须叫 A800，但默认仍要求约 80GB 显存，以保护已经通过的 `backward_subbatch_size_per_device=2` 设置。不要仅为了绕过门禁而降低 `SOPPO_MIN_GPU_MEMORY_MIB`；若新卡显存不足，应先重新讨论 batch/显存配置。

## 7. 一条命令启动完整长链

确认环境、模型、数据和 Git 状态后执行：

```bash
export SERVER_BASE=/data/youchang/youchang
export RUN_CONTEXT=standalone
source "$SERVER_BASE/activate_env.sh"
cd "$SERVER_BASE/SOPPO/code/scripts/standalone"
bash start_pipeline.sh
```

`start_pipeline.sh` 用独立进程组和 `nohup` 在后台启动，因此 SSH 断开后仍可继续。它不会调用 `sbatch`，也不存在排队或依赖等待。服务器同一时间只运行一个需要两卡的训练 arm；完整顺序为：

```text
GPU/输入预检
→ tests
→ strong smoke
→ reference cache
→ DPO-10 → DPO-100 → headroom gate
→ 4 个 fixed-lambda PE → validation selection
→ SSPO-hard-exp → SOPPO-PE-exp
→ C_epsilon prepare → base+8 methods → derive
→ 8 个独立评价
→ aggregate/export
```

每一步成功后才会进入下一步；失败立即停止。运行期间不要 `git pull`、切换 commit 或修改服务器仓库，后续阶段会检查启动时锁定的完整 commit 与 clean worktree。

如需在前台运行以便直接观察，也可以执行：

```bash
bash run_pipeline.sh
```

前台和后台入口都拒绝覆盖任何已有 pipeline、训练输出或 export。

## 8. 查看状态和日志

```bash
cd "$SERVER_BASE/SOPPO/code/scripts/standalone"
bash status_pipeline.sh
```

该命令显示当前阶段、每个阶段的 `pending/running/completed/failed` 状态、退出码，并附带 driver log 的最后 40 行。只看状态表、不打印日志尾部：

```bash
bash status_pipeline.sh --no-tail
```

完整总日志与分阶段日志分别位于：

```text
<SERVER_BASE>/runs/exp-20260819-01-mvp/standalone-driver.log
<SERVER_BASE>/runs/exp-20260819-01-mvp/pipeline/logs/<stage>.log
```

例如：

```bash
tail -f "$SERVER_BASE/runs/exp-20260819-01-mvp/standalone-driver.log"
tail -n 200 "$SERVER_BASE/runs/exp-20260819-01-mvp/pipeline/logs/dpo_0.log"
```

## 9. 只停止本实验

先 dry-run：

```bash
bash stop_pipeline.sh
```

确认 PID、进程组和命令行都正确后：

```bash
bash stop_pipeline.sh --execute
```

脚本只向当前 registry 记录且再次核验过的独立进程组发送 `TERM`，不会按用户名批量杀进程。

## 10. 失败、重跑与完成标志

当前 checkpoint 仍不是 optimizer/scheduler/RNG 的 exact resume。某个正式 arm 失败后，不要把 partial checkpoint 当作无缝续训；先保留完整失败证据，再从干净实验目录重跑。reference cache 位于 experiment 之外，完整时可继续复用。

建议把整个失败实验目录移动到独立归档位置，而不是删除：

```text
<SERVER_BASE>/runs/failed_attempts/exp-20260819-01-mvp-<timestamp>/
```

最终成功标志：

```bash
test -f "$SERVER_BASE/exports/exp-20260819-01-mvp/EXPORT_COMPLETE" && echo PIPELINE_OK
```

只回传 export 目录中的白名单聚合文件；不要回传数据、私有标签、模型/checkpoint、reference cache、原始日志、raw matrices 或逐样本 predictions。
