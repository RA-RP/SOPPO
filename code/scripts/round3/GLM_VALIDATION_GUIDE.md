# Round3 GLM测试与部署指南

本指南规定GLM在3×RTX 4090服务器上的角色：GLM只执行命令、部署已经由用户审阅的精确版本、运行测试并回传证据；Round3的研究设计、源码、配置、shell和Markdown只由Codex在本地修改。GLM发现问题后不得自行修代码。

当前合同为`r3-theory-v1.0`、`round3-exp-v1.5`和`round3-code-candidate-v0.5`，唯一活动阶段为`SERVER_EXECUTION`。用户已授权Codex负责部署与持续测试；GLM仅在Codex另行派发具体命令任务时执行，不得从本指南自行启动或修改任何内容。

## 1. 角色和绝对边界

GLM可以：

- 运行只读状态、Git、进程、GPU和磁盘查询；
- 在用户明确给出exact commit并授权部署后，fetch并切换到该commit；
- 使用仓库已有脚本运行环境核验、合同测试、数据准备、preflight和strong smoke；
- 把退出码、无样本聚合、manifest摘要、日志路径和SHA-256回传给Codex。

GLM不可以：

- 手工创建、编辑或格式化任何`SOPPO/`内文件；机械切换到用户指定的exact commit是唯一部署例外，仍禁止`sed -i`、重定向覆盖源码、`apply_patch`以及临时修补；
- `git commit`、`git push`、merge、rebase，或自行选择commit/ref/experiment ID；
- 未经逐项授权执行`git reset`、`git restore`、`git clean`、`git stash`或覆盖dirty checkout；
- 安装/升级/降级依赖、重建已有环境或通过改requirements解决测试错误；
- 删除、覆盖或移动data、model、cache、run、checkpoint、日志和失败证据；失败遗留的空`dual_source_v1`也必须保留；
- 停止Round2/Round3或其他进程；不得使用`pkill`、`killall`或按用户批量杀进程；
- 修改实验数量、batch、长度、显存门禁、测试断言或重试参数来绕过失败；
- 打印或回传canonical JSONL样本、private labels、逐样本预测、rollout request/response或原始数据文本；
- 运行`start_all.sh`或任何formal训练，除非之后收到单独、明确的formal授权。

任何命令非零退出、checkout dirty、路径已存在但不完整、GPU被占用、磁盘风险、数量/SHA不符或不确定状态，都应立即停止当前阶段并向Codex报告。不要边测试边修复。

## 2. 已知事实和仍需填写的值

当前已有的服务器只读事实：

- 本地方案B实现当前位于commit `8eedca51067162562cade72408930c8b2321ffda`，但该commit尚不能由GLM自行认定为`AUTHORIZED_COMMIT`；
- Round2 experiment：`exp-20260824-05-round2-tp2`，已在第一方法step590停止；第二方法未启动，checkpoint保留；
- Round3准备attempt：`round3-20260826-01`，只有`source_revisions.json`，尚无v2 data、reference cache或strong smoke证据；
- model revision：`4855588ea1a12789f2e965e5f52a9e4a24c94b2a`；
- UltraFeedback revision：`3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`；
- UltraChat revision：`8049631c405ae6576f93f445c6b8166f76f5505a`；
- `dual_source_v1`是失败遗留空目录，不能删除；方案B使用`dual_source_v2`；
- 服务器checkout曾存在GLM留下的`code/src/round3/data.py` dirty diff。它不是候选实现，未获明确处理授权前必须停在只读阶段。

部署前必须由用户或Codex明确提供：

```bash
export AUTHORIZED_COMMIT='<用户审阅后提供的40字符commit SHA>'
export SOPPO_ROUND3_EXPERIMENT_ID='round3-20260826-01'
```

不得执行含`<...>`占位符的命令。`AUTHORIZED_COMMIT`缺失时只能做第3节。

## 3. 阶段A：只读现场核验

这一阶段始终可以先做；不改变checkout、不创建实验产物、不发送信号。登录已配置的服务器别名后执行：

```bash
ssh 4090-3
```

```bash
set -euo pipefail
export SERVER_BASE=/data/youchang/youchang
export SOPPO_REPO="$SERVER_BASE/SOPPO"
export ROUND2_EXPERIMENT_ID=exp-20260824-05-round2-tp2
export SOPPO_ROUND3_EXPERIMENT_ID=round3-20260826-01

date -Is
git -C "$SOPPO_REPO" rev-parse HEAD
git -C "$SOPPO_REPO" status --short
git -C "$SOPPO_REPO" diff --stat
git -C "$SOPPO_REPO" diff --check
git -C "$SOPPO_REPO" diff -- code/src/round3/data.py | sha256sum

bash "$SOPPO_REPO/code/scripts/round2/status_all.sh"

for state_file in \
  "$SERVER_BASE/runs/$ROUND2_EXPERIMENT_ID/soppo_pe_sft_rollout_exp/state.json" \
  "$SERVER_BASE/runs/$ROUND2_EXPERIMENT_ID/soppo_pe_rollout_only_exp/state.json"; do
  if [[ -f "$state_file" ]]; then
    echo "STATE $state_file"
    sed -n '1,220p' "$state_file"
  else
    echo "ABSENT $state_file"
  fi
done

for metrics_file in \
  "$SERVER_BASE/runs/$ROUND2_EXPERIMENT_ID/soppo_pe_sft_rollout_exp/logs/metrics.jsonl" \
  "$SERVER_BASE/runs/$ROUND2_EXPERIMENT_ID/soppo_pe_rollout_only_exp/logs/metrics.jsonl"; do
  if [[ -f "$metrics_file" ]]; then
    echo "METRICS_TAIL $metrics_file"
    tail -n 3 "$metrics_file"
  else
    echo "ABSENT $metrics_file"
  fi
done

shopt -s nullglob
for pid_file in "$SERVER_BASE/runs/$ROUND2_EXPERIMENT_ID"/checkpoint_pruner*.pid; do
  pruner_pid="$(tr -d '[:space:]' < "$pid_file")"
  echo "PRUNER_PID_FILE $pid_file PID=$pruner_pid"
  ps -o pid,ppid,state,etime,args -p "$pruner_pid" || true
done

ls -ld \
  "$SERVER_BASE/envs/round3-train" \
  "$SERVER_BASE/envs/round3-rollout" \
  "$SERVER_BASE/models/Qwen3-1.7B" \
  "$SERVER_BASE/data/round3/dual_source_v1" \
  "$SERVER_BASE/data/round3/dual_source_v2" \
  "$SERVER_BASE/cache/round3/reference_qwen3_1.7b_dual_source_v2" \
  "$SERVER_BASE/runs/$SOPPO_ROUND3_EXPERIMENT_ID" 2>&1 || true

if [[ -f "$SERVER_BASE/runs/$SOPPO_ROUND3_EXPERIMENT_ID/source_revisions.json" ]]; then
  "$SERVER_BASE/envs/round3-train/bin/python" -m json.tool \
    "$SERVER_BASE/runs/$SOPPO_ROUND3_EXPERIMENT_ID/source_revisions.json"
fi

nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits
df -h "$SERVER_BASE"
du -sh \
  "$SERVER_BASE/runs/$ROUND2_EXPERIMENT_ID" \
  "$SERVER_BASE/runs/$SOPPO_ROUND3_EXPERIMENT_ID" \
  "$SERVER_BASE/data/round3" \
  "$SERVER_BASE/cache/round3" \
  "$SERVER_BASE/models/Qwen3-1.7B" 2>&1 || true
```

阶段A完成后先回传结果并停下。特别是只要`git status --short`非空，就不得进入部署。GLM不得自行丢弃目前已知的server-side `data.py`补丁。

## 4. 阶段B：部署用户审阅的精确commit

只有收到“允许GLM把服务器切换到`AUTHORIZED_COMMIT`”的明确指令后才能执行。本节不授权处理dirty文件；若阶段A仍dirty，继续停止并等待Codex或用户给出专门处理方案。

```bash
set -euo pipefail
export SERVER_BASE=/data/youchang/youchang
export SOPPO_REPO="$SERVER_BASE/SOPPO"
: "${AUTHORIZED_COMMIT:?AUTHORIZED_COMMIT is required}"
[[ "$AUTHORIZED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git -C "$SOPPO_REPO" status --porcelain)" ]] || {
  echo 'STOP: server checkout is dirty; do not stash/reset/restore/clean' >&2
  exit 1
}

git -C "$SOPPO_REPO" fetch origin
git -C "$SOPPO_REPO" cat-file -e "$AUTHORIZED_COMMIT^{commit}"
git -C "$SOPPO_REPO" switch --detach "$AUTHORIZED_COMMIT"

test "$(git -C "$SOPPO_REPO" rev-parse HEAD)" = "$AUTHORIZED_COMMIT"
test -z "$(git -C "$SOPPO_REPO" status --porcelain)"
git -C "$SOPPO_REPO" show --no-patch --format=fuller HEAD
```

部署后再次执行`nvidia-smi`、`df -h "$SERVER_BASE"`与Round2 `status_all.sh`。若Round2状态和阶段A不一致，停止，不运行测试。

## 5. 阶段C：环境和CPU合同测试

本阶段需要用户明确授权“运行Round3 server tests”。复用现有环境，不运行`00_setup_envs.sh`。若环境缺失或`pip check`失败，原样报告，不安装任何包。

```bash
set -euo pipefail
export SERVER_BASE=/data/youchang/youchang
export SOPPO_REPO="$SERVER_BASE/SOPPO"
export SOPPO_ROUND3_EXPERIMENT_ID=round3-20260826-01
export AUTHORIZED_COMMIT='<已经获批的40字符commit SHA>'
export EVIDENCE_DIR="$SERVER_BASE/platform_logs/$SOPPO_ROUND3_EXPERIMENT_ID/glm_validation/${AUTHORIZED_COMMIT:0:12}"

test "$(git -C "$SOPPO_REPO" rev-parse HEAD)" = "$AUTHORIZED_COMMIT"
test -z "$(git -C "$SOPPO_REPO" status --porcelain)"
test -x "$SERVER_BASE/envs/round3-train/bin/python"
test -x "$SERVER_BASE/envs/round3-rollout/bin/python"
mkdir -p "$EVIDENCE_DIR"

for script_file in $(rg --files "$SOPPO_REPO/code/scripts/round3" -g '*.sh'); do
  bash -n "$script_file"
done

"$SERVER_BASE/envs/round3-train/bin/python" --version
"$SERVER_BASE/envs/round3-rollout/bin/python" --version
"$SERVER_BASE/envs/round3-train/bin/python" -m pip check
"$SERVER_BASE/envs/round3-rollout/bin/python" -m pip check

bash "$SOPPO_REPO/code/scripts/round3/02_server_tests.sh" \
  2>&1 | tee "$EVIDENCE_DIR/02_server_tests.log"
test "${PIPESTATUS[0]}" -eq 0
sha256sum "$EVIDENCE_DIR/02_server_tests.log"
```

预期是6项合同测试通过，而不是旧候选的5项。GLM只报告实际结果；数字不同即停止，不自行改测试。

回传：exact commit、两个Python版本、两次`pip check`结果、pytest摘要、日志绝对路径和SHA-256、测试后的`git status --short`。

## 6. 阶段D：冻结数据v2构造与审计

本阶段需要单独授权“运行`00_prepare_data.sh`”。它会读取冻结数据并写仓库外产物，不能与CPU测试授权混为一谈。

先核对现状：

```bash
set -euo pipefail
export SERVER_BASE=/data/youchang/youchang
export SOPPO_REPO="$SERVER_BASE/SOPPO"
export SOPPO_ROUND3_EXPERIMENT_ID=round3-20260826-01
export AUTHORIZED_COMMIT='<已经获批的40字符commit SHA>'
export EVIDENCE_DIR="$SERVER_BASE/platform_logs/$SOPPO_ROUND3_EXPERIMENT_ID/glm_validation/${AUTHORIZED_COMMIT:0:12}"
source "$SOPPO_REPO/code/scripts/round3/round3_env.sh"

test "$(git -C "$SOPPO_REPO" rev-parse HEAD)" = "$AUTHORIZED_COMMIT"
test -z "$(git -C "$SOPPO_REPO" status --porcelain)"
test -f "$ROUND3_RUN_ROOT/source_revisions.json"
test -f "$ROUND3_MODEL_DIR/model_manifest.json"
mkdir -p "$EVIDENCE_DIR"

if [[ -e "$ROUND3_DATA_DIR" ]]; then
  echo "STOP: v2 data path already exists; do not delete or overwrite: $ROUND3_DATA_DIR" >&2
  ls -la "$ROUND3_DATA_DIR"
  exit 1
fi

bash "$SOPPO_REPO/code/scripts/round3/00_download_model.sh"
df -h "$SERVER_BASE"
```

`00_download_model.sh`在模型已存在时只验证manifest和revision；若它尝试新下载或报告revision不一致，停止。确认后运行：

```bash
bash "$SOPPO_REPO/code/scripts/round3/00_prepare_data.sh" \
  2>&1 | tee "$EVIDENCE_DIR/00_prepare_data.log"
test "${PIPESTATUS[0]}" -eq 0
sha256sum "$EVIDENCE_DIR/00_prepare_data.log"
```

不要展示任何JSONL内容。只运行聚合检查：

```bash
"$ROUND3_TRAIN_PYTHON" - "$ROUND3_DATA_DIR/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
safe = {
    "schema_version": manifest.get("schema_version"),
    "source_audit": manifest.get("source_audit"),
    "exclusions": manifest.get("exclusions"),
    "files": {
        name: {"rows": value.get("rows"), "bytes": value.get("bytes"), "sha256": value.get("sha256")}
        for name, value in manifest.get("files", {}).items()
    },
}
print(json.dumps(safe, indent=2, sort_keys=True))
PY

wc -l \
  "$ROUND3_DATA_DIR/paired_train_8k.jsonl" \
  "$ROUND3_DATA_DIR/paired_train_1k.jsonl" \
  "$ROUND3_DATA_DIR/unpaired_train_7k.jsonl" \
  "$ROUND3_DATA_DIR/validation_1k.jsonl" \
  "$ROUND3_DATA_DIR/test.public.jsonl" \
  "$ROUND3_DATA_DIR/test.private_labels.jsonl" \
  "$ROUND3_DATA_DIR/source_manifest.jsonl" \
  "$ROUND3_DATA_DIR/malformed_source_rows.jsonl"

df -h "$SERVER_BASE"
git -C "$SOPPO_REPO" status --short
```

必须看到：

- schema `round3.data_manifest.v2`；
- views为8,000/1,000/7,000/1,000/997/997；
- `source_manifest.jsonl`为17,997行；
- `malformed_source_rows.jsonl`为12,197行；
- `test_prefs`为2,000 source、3 malformed、1,997 valid；
- test文件名不含`1k`；
- Git checkout仍clean。

任一不符都停止。不得从train补3条，不得把test动态改回1,000。

## 7. 阶段E：五方法strong smoke

本阶段使用GPU并创建reference cache、resolved configs、五个one-step run和storage projection，必须再获得明确的“运行Round3 strong smoke”授权。执行前要求三张GPU没有compute PID，并记录磁盘；不停止占用者，只等待或报告。

注意：`03_strong_smoke.sh`不是可覆盖入口。只要`resolved/strong_smoke`、reference v2、方法status或smoke目录存在但不完整，先报告给Codex；不得删除后重跑。

```bash
set -euo pipefail
export SERVER_BASE=/data/youchang/youchang
export SOPPO_REPO="$SERVER_BASE/SOPPO"
export SOPPO_ROUND3_EXPERIMENT_ID=round3-20260826-01
export AUTHORIZED_COMMIT='<已经获批的40字符commit SHA>'
export SOPPO_ROUND3_PHYSICAL_PAIR_SUBBATCH=1
export EVIDENCE_DIR="$SERVER_BASE/platform_logs/$SOPPO_ROUND3_EXPERIMENT_ID/glm_validation/${AUTHORIZED_COMMIT:0:12}"
source "$SOPPO_REPO/code/scripts/round3/round3_env.sh"

test "$(git -C "$SOPPO_REPO" rev-parse HEAD)" = "$AUTHORIZED_COMMIT"
test -z "$(git -C "$SOPPO_REPO" status --porcelain)"
test -f "$SERVER_BASE/data/round3/dual_source_v2/manifest.json"
mkdir -p "$EVIDENCE_DIR"

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits | tee "$EVIDENCE_DIR/pre_smoke_compute_processes.txt"
test ! -s "$EVIDENCE_DIR/pre_smoke_compute_processes.txt"
nvidia-smi | tee "$EVIDENCE_DIR/pre_smoke_nvidia_smi.txt"
df -h "$SERVER_BASE" | tee "$EVIDENCE_DIR/pre_smoke_df.txt"

bash "$SOPPO_REPO/code/scripts/round3/03_strong_smoke.sh" \
  2>&1 | tee "$EVIDENCE_DIR/03_strong_smoke.log"
test "${PIPESTATUS[0]}" -eq 0

sha256sum \
  "$EVIDENCE_DIR/03_strong_smoke.log" \
  "$ROUND3_RUN_ROOT/storage_projection.json"
nvidia-smi
df -h "$SERVER_BASE"
git -C "$SOPPO_REPO" status --short
```

完成后只回传以下无样本证据：

- 五个`control/strong_smoke/<method>/status.json`；
- 五个`checkpoint_verified.json`的状态、路径和SHA；
- `storage_projection.json`全文；
- 两个动态方法replica ready/ACK数量、adapter SHA一致性和日志路径；不要回传request/response正文；
- 每个方法训练日志最后80行、退出码、finite loss/gradient和峰值显存摘要；
- strong-smoke日志、preflight JSON和环境freeze的路径与SHA；
- smoke后GPU、`df`和clean Git证据。

如果任一方法失败，后续方法是否已被脚本阻断也要说明。不得单独重跑失败方法、改subbatch、清理status或继续formal。

## 8. 明确禁止在本作业单中执行formal

以下命令仅供识别，当前禁止运行：

```text
code/scripts/round3/start_all.sh
code/scripts/round3/run_all.sh
code/scripts/round3/run_method.sh <method> formal
code/scripts/round3/04_evaluate.sh
code/scripts/round3/05_aggregate.sh
```

Codex收到strong-smoke证据后负责判断是修代码、返回实验设计，还是向用户申请formal授权。GLM不能自行做这个决策。

## 9. GLM回传模板

每次只回传本轮实际执行的阶段：

```text
阶段：A / B / C / D / E
开始与结束时间：
AUTHORIZED_COMMIT：
experiment ID：
执行的命令：
每条命令退出码：
Git HEAD / status：
Round2保护状态：
GPU / df摘要：
通过项：
失败项及首次错误：
证据文件绝对路径与SHA-256：
是否产生新目录或文件：
明确未执行的后续阶段：
```

不要只回复“完成”或“测试通过”。不要把计划、推测或预期值写成实测结果。
