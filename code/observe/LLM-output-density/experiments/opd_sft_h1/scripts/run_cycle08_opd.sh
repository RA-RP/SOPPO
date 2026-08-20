#!/usr/bin/env bash
# Cycle 08 — OPD (on-policy distillation) trajectory via verl, to compare against
# the Cycle 07 SFT trajectory. Student = Qwen3-4B-Base, Teacher = Qwen3-8B.
# vLLM rollout + FSDP training + top-k teacher logprobs. Run inside the `verl` env.
#
# ALIGNMENT WITH CYCLE 07 SFT (so the OPD-vs-SFT trajectory is comparable):
#   - same seed=42 5000-prompt sample (data/opd_prompts_5k.parquet)
#   - LoRA r=32 alpha=64 on all-linear (cycle07: r32/a64, q,k,v,o,gate,up,down)
#   - train_batch=ppo_mini=16  -> one grad update per 16 prompts (cycle07 eff_batch=16)
#   - ppo_epochs=1             -> rollouts NOT reused (on-policy purity; NOT the dataset epoch)
#   - total_epochs=2           -> sweep the prompt sample twice = cycle07's epoch=2
#         => verl filters to 4999 rows and drop_last gives 624 updates, matching cycle07's
#            evaluated final checkpoint step_624
#   - loss_agg_mode=token-mean -> same seqlen normalization as cycle07
#   - max_response ~ cycle07 max_len 10240; lr 5e-5 = cycle07 SFT lr
#   - reference-policy KL OFF (regularize toward teacher only, not a ref policy)
#
# LOSS / GRADIENT MODE (decided: supervised OPD, user-confirmed):
#   use_policy_gradient=False + loss_mode=forward_kl_topk  ==  GKD supervised distillation:
#     L = sum_v teacher_p(v) * (log teacher_p(v) - log student_p(v))  over the teacher top-k.
#     Its gradient wrt student weights depends on the teacher probs -> a real distillation
#     signal, and the cleanest analog to SFT (only the target changes: data-CE -> teacher-KL).
#   NB verl REJECTS use_policy_gradient=False + loss_mode=k1 (k1's gradient drops the teacher
#     term, so direct backprop is not distillation). The single-sample estimators {k1,k2,...}
#     are for the policy-gradient path (use_policy_gradient=True, advantages=-loss.detach()).
#   forward_kl_topk pairs with topk=32 (dense, low-variance target — good for the small batch).
#
# HARDWARE: 2x96G. Teacher (8B vLLM) gets TEACHER_WORLD_SIZE cards; student rollout+train
#   gets the rest. The verl teacher/student GPU-pool allocation on a 2-card node is the
#   #1 item to verify in the large-GPU smoke (Stage 3); single-card colocation is the risk.
set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"   # absolute (Hydra changes cwd at runtime)
# vLLM V1's memory pool rejects expandable_segments. Drop only that incompatible
# allocator knob if inherited from the caller; leave any other allocator config intact.
case "${PYTORCH_CUDA_ALLOC_CONF:-}" in *expandable_segments:True*) unset PYTORCH_CUDA_ALLOC_CONF ;; esac
case "${PYTORCH_ALLOC_CONF:-}" in *expandable_segments:True*) unset PYTORCH_ALLOC_CONF ;; esac

# ---- paths ----
STUDENT_MODEL=${STUDENT_MODEL:-/root/autodl-tmp/model/Qwen/Qwen3-4B-Base}
TEACHER_MODEL=${TEACHER_MODEL:-/root/autodl-tmp/model/Qwen/Qwen3-8B}
TRAIN_PARQUET=${TRAIN_PARQUET:-/root/autodl-tmp/cycle08_opd_trajectory/data/opd_prompts_5k.parquet}
VAL_PARQUET=${VAL_PARQUET:-$TRAIN_PARQUET}   # val unused (val_before_train=False, test_freq big)
CKPT_DIR=${CKPT_DIR:-/root/autodl-tmp/cycle08_opd_trajectory/checkpoints}

# ---- topology (2x96G default: student 1 card + teacher 1 dedicated card = 2 total) ----
# verl sums the student pool (trainer.n_gpus_per_node) and the teacher pool
# (distillation.n_gpus_per_node); they are disjoint. 4B student trains on one 96G card
# unsharded; 8B teacher infers on the other. So NGPUS_PER_NODE=1 (NOT 2) + TEACHER_WORLD_SIZE=1.
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
rollout_tp=${ROLLOUT_TP:-1}
teacher_tp=${TEACHER_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}   # 0.6 fixes vLLM sleep wake_up OOM: wake re-allocs gpu_mem*96=58G KV; only 16 seqs so KV over-provisioned at 0.75 -> lower ask, big margin for training transient on long batches. ~0 throughput cost.
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.80}

# ---- OPD loss (supervised GKD: forward top-k KL; see LOSS / GRADIENT MODE note above) ----
distillation_loss_mode=${DISTILLATION_LOSS_MODE:-forward_kl_topk}  # teacher-weighted top-k forward KL
use_policy_gradient=${USE_POLICY_GRADIENT:-False}                  # supervised (direct backprop of the KL)
distillation_topk=${DISTILLATION_TOPK:-32}                        # dense target, small-batch stability

# ---- batch / step alignment with SFT ----
train_batch_size=${TRAIN_BATCH_SIZE:-16}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
ppo_epochs=${PPO_EPOCHS:-1}
total_epochs=${TOTAL_EPOCHS:-2}
total_training_steps=${TOTAL_TRAINING_STEPS:-624}
loss_agg_mode=${LOSS_AGG_MODE:-token-mean}
actor_lr=${ACTOR_LR:-5e-5}

# ---- lengths (≈ cycle07 max_len 10240) ----
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-10240}   # ≈cycle07 total cap 10240 (prompt~111 tok, so response≈total; ~12% truncation matches cycle07)
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}   # lowered from 24576: caps training micro-batch memory so more is physically free when vLLM rollout wakes (mild ~10-20% train slowdown, reliability for 25h run)

# ---- LoRA (= cycle07) ----
lora_rank=${LORA_RANK:-32}
lora_alpha=${LORA_ALPHA:-64}
target_modules=${TARGET_MODULES:-all-linear}

# ---- checkpointing: grid-save to mirror cycle07 step grid ----
save_freq=${SAVE_FREQ:-5}      # dense early saves; trajectory needs step_005,010,020,...
test_freq=${TEST_FREQ:-100000} # no in-loop eval; eval is a separate stage (reuse cycle07 runner)

project_name=${PROJECT_NAME:-cycle08_opd_vs_sft}
experiment_name=${EXPERIMENT_NAME:-qwen3_4b_base_opd_from_qwen3_8b}

train_files="['$TRAIN_PARQUET']"
val_files="['$VAL_PARQUET']"
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
    data.prompt_key=prompt
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING:-True}
    +actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL:-flash_attention_2}
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora_rank=${lora_rank}
    actor_rollout_ref.model.lora_alpha=${lora_alpha}
    actor_rollout_ref.model.target_modules=${target_modules}
)

ACTOR=(
    actor_rollout_ref.actor.strategy=${FSDP_STRATEGY:-fsdp2}
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs}
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.temperature=0.6
    actor_rollout_ref.rollout.top_p=0.9
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.default_local_dir=${CKPT_DIR}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.resume_mode=auto
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
)

REWARD=(
    # Pure distillation: verl's reward loop still runs but its value is unused
    # (use_task_rewards=False). A constant-0 custom scorer avoids the
    # "Reward function not implemented for data_source" crash on our Math-CoT-20k source.
    reward.custom_reward_function.path="$SCRIPT_DIR/cycle08_null_reward.py"
    reward.custom_reward_function.name=compute_score
)

DISTILL=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "${DISTILL[@]}" \
    "$@"
