#!/usr/bin/env bash
# Cycle09 Q1 Stage A/B: Qwen alpha=.5 strict self/external support mixture.
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
case "${PYTORCH_CUDA_ALLOC_CONF:-}" in *expandable_segments:True*) unset PYTORCH_CUDA_ALLOC_CONF ;; esac
case "${PYTORCH_ALLOC_CONF:-}" in *expandable_segments:True*) unset PYTORCH_ALLOC_CONF ;; esac

MODE=${MODE:-formal}
Q1_STAGE=${Q1_STAGE:-stage_a}
ROOT=${ROOT:-/root/autodl-tmp/cycle09_block3/qwen_alpha05}
STUDENT_MODEL=${STUDENT_MODEL:-/root/autodl-tmp/model/Qwen/Qwen3-4B-Base}
TEACHER_MODEL=${TEACHER_MODEL:-/root/autodl-tmp/model/Qwen/Qwen3-8B}
if [[ "$MODE" == "smoke" ]]; then
    if [[ "$Q1_STAGE" != "stage_a" ]]; then
        echo "Q1 smoke only supports Q1_STAGE=stage_a" >&2
        exit 64
    fi
    TRAIN_PARQUET=${TRAIN_PARQUET:-$ROOT/data/qwen_alpha05_schedule_smoke32.parquet}
    CKPT_DIR=${CKPT_DIR:-$ROOT/smoke/checkpoints}
    ROLLOUT_DIR=${ROLLOUT_DIR:-$ROOT/smoke/rollouts}
    TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
    SAVE_FREQ=${SAVE_FREQ:-1}
    PROJECT_NAME=${PROJECT_NAME:-cycle09_q1_alpha05_smoke}
elif [[ "$Q1_STAGE" == "stage_a" ]]; then
    TRAIN_PARQUET=${TRAIN_PARQUET:-$ROOT/data/qwen_alpha05_schedule_624.parquet}
    CKPT_DIR=${CKPT_DIR:-$ROOT/checkpoints}
    ROLLOUT_DIR=${ROLLOUT_DIR:-$ROOT/rollouts}
    TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-160}
    SAVE_FREQ=${SAVE_FREQ:-5}
    PROJECT_NAME=${PROJECT_NAME:-cycle09_q1_alpha05_stage_a}
    if (( TOTAL_TRAINING_STEPS > 160 )); then
        echo "Q1 Stage A hard stop forbids TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS (>160)" >&2
        exit 64
    fi
elif [[ "$Q1_STAGE" == "stage_b" ]]; then
    TRAIN_PARQUET=${TRAIN_PARQUET:-$ROOT/data/qwen_alpha05_schedule_320.parquet}
    CKPT_DIR=${CKPT_DIR:-$ROOT/checkpoints}
    ROLLOUT_DIR=${ROLLOUT_DIR:-$ROOT/rollouts}
    TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-320}
    SAVE_FREQ=${SAVE_FREQ:-5}
    PROJECT_NAME=${PROJECT_NAME:-cycle09_q1_alpha05_stage_b}
    if (( TOTAL_TRAINING_STEPS != 320 )); then
        echo "Q1 Stage B must end exactly at TOTAL_TRAINING_STEPS=320" >&2
        exit 64
    fi
else
    echo "unknown Q1_STAGE=$Q1_STAGE (expected stage_a or stage_b)" >&2
    exit 64
fi
if [[ ! -f "$TRAIN_PARQUET" ]]; then
    echo "missing prepared Q1 schedule: $TRAIN_PARQUET" >&2
    exit 66
fi

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.60}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.80}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-10240}
MAX_NUM_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))

mkdir -p "$CKPT_DIR" "$ROLLOUT_DIR"
train_files="['$TRAIN_PARQUET']"

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$train_files"
    data.train_batch_size=16
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=False
    data.prompt_key=prompt
    data.custom_cls.path="$SCRIPT_DIR/cycle09_q1_mixture_runtime.py"
    data.custom_cls.name=Q1MixtureDataset
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora_rank=32
    actor_rollout_ref.model.lora_alpha=64
    actor_rollout_ref.model.target_modules=all-linear
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=5e-5
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.seed=42
    actor_rollout_ref.rollout.temperature=0.6
    actor_rollout_ref.rollout.top_p=0.9
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.max_model_len=${MAX_NUM_TOKENS}
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$SCRIPT_DIR/cycle09_q1_agent_loops.yaml"
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=qwen3_4b_alpha05_support_mixture
    trainer.default_local_dir="$CKPT_DIR"
    trainer.rollout_data_dir="$ROLLOUT_DIR"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.resume_mode=auto
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=100000
    trainer.total_epochs=1
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
)

REWARD=(
    reward.custom_reward_function.path="$SCRIPT_DIR/cycle08_null_reward.py"
    reward.custom_reward_function.name=compute_score
)

DISTILL=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
    distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_NUM_TOKENS}
    distillation.distillation_loss.loss_mode=forward_kl_topk_source_equal
    distillation.distillation_loss.topk=32
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=False
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
