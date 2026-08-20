#!/usr/bin/env bash
# H5 frozenSelf0-KD: fixed step0-self support, fixed teacher RAW top-32 targets.
set -euo pipefail
if [[ $# -ne 2 || "$1" != "--contract" ]]; then echo "usage: $0 --contract CONTRACT.json" >&2; exit 2; fi
CONTRACT="$2"; SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
ROOT=${FROZEN_SELF_ROOT:-/root/autodl-tmp/cycle09_stage3_followup/H5_frozen_self}
TRAIN_PARQUET="$ROOT/data/frozen_self_schedule.parquet"; STORE="$ROOT/frozen_store"; CKPT="$ROOT/checkpoints"; ROLLOUT="$ROOT/rollouts"
MODEL_PATH=${FROZEN_SELF_MODEL_PATH:-/root/autodl-tmp/cycle09_block3/llama_opd/model/student_runtime}
TRAIN_STEPS=${FROZEN_SELF_TRAIN_STEPS:-320}; TRAIN_EPOCHS=${FROZEN_SELF_TRAIN_EPOCHS:-2}; TRAIN_BATCH=${FROZEN_SELF_TRAIN_BATCH:-16}
SAVE_FREQ=${FROZEN_SELF_SAVE_FREQ:-5}; RESUME_MODE=${FROZEN_SELF_RESUME_MODE:-auto}; EXPERIMENT_NAME=${FROZEN_SELF_EXPERIMENT_NAME:-frozenSelf0_KD}
for p in "$CONTRACT" "$TRAIN_PARQUET" "$STORE/prompt_ids.npy" "$STORE/top32_ids.npy" "$MODEL_PATH/tokenizer_config.json"; do [[ -f "$p" ]] || { echo "missing frozen-self prerequisite: $p" >&2; exit 66; }; done
mkdir -p "$CKPT" "$ROLLOUT" "$ROOT/logs"
RETENTION_ENABLED=${FROZEN_SELF_ENABLE_RETENTION_PRUNER:-1}
if [[ "$RETENTION_ENABLED" == "1" ]]; then
  /root/autodl-tmp/envs/verl/bin/python "$SCRIPT_DIR/cycle09_h5_checkpoint_retention.py" \
    --root "$ROOT" --parent-pid "$$" --poll-seconds 15 \
    > "$ROOT/logs/checkpoint_retention.log" 2>&1 &
fi
# `student_runtime` shares the immutable step-0 base weights and adds only the
# Llama tokenizer chat template required by Verl's dataset length filter.
exec /root/autodl-tmp/envs/verl/bin/python -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
 data.train_files="['$TRAIN_PARQUET']" data.val_files="['$TRAIN_PARQUET']" data.train_batch_size=$TRAIN_BATCH data.max_prompt_length=1024 data.max_response_length=10240 data.filter_overlong_prompts=True data.truncation=error data.shuffle=False data.prompt_key=prompt data.custom_cls.path="$SCRIPT_DIR/cycle09_frozen_self_runtime.py" data.custom_cls.name=FrozenSelfDataset \
 actor_rollout_ref.model.path="$MODEL_PATH" actor_rollout_ref.model.use_remove_padding=True +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.model.lora_rank=32 actor_rollout_ref.model.lora_alpha=64 actor_rollout_ref.model.target_modules=all-linear \
 actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.optim.lr=5e-5 actor_rollout_ref.actor.ppo_mini_batch_size=16 actor_rollout_ref.actor.ppo_epochs=1 actor_rollout_ref.actor.loss_agg_mode=token-mean actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.60 actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.seed=42 actor_rollout_ref.rollout.temperature=0.6 actor_rollout_ref.rollout.top_p=0.9 actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.max_model_len=11265 actor_rollout_ref.rollout.load_format=safetensors actor_rollout_ref.rollout.layered_summon=True actor_rollout_ref.rollout.calculate_log_probs=False actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 actor_rollout_ref.rollout.agent.agent_loop_config_path="$SCRIPT_DIR/cycle09_frozen_self_agent_loops.yaml" \
 trainer.use_v1=True trainer.balance_batch=True trainer.logger='["console"]' trainer.project_name=cycle09_stage3 trainer.experiment_name="$EXPERIMENT_NAME" trainer.default_local_dir="$CKPT" trainer.rollout_data_dir="$ROLLOUT" trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.val_before_train=False trainer.resume_mode=$RESUME_MODE trainer.save_freq=$SAVE_FREQ trainer.test_freq=100000 trainer.total_epochs=$TRAIN_EPOCHS trainer.total_training_steps=$TRAIN_STEPS \
 reward.custom_reward_function.path="$SCRIPT_DIR/cycle08_null_reward.py" reward.custom_reward_function.name=compute_score \
 distillation.enabled=True +distillation.precomputed_teacher_tensors=True distillation.n_gpus_per_node=1 distillation.nnodes=1 distillation.teacher_models.teacher_model.model_path=/root/autodl-tmp/model/Meta/modelscope/Meta-Llama-3.1-8B-Instruct distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 distillation.teacher_models.teacher_model.inference.name=vllm distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.80 distillation.teacher_models.teacher_model.inference.max_model_len=11265 distillation.distillation_loss.loss_mode=forward_kl_topk distillation.distillation_loss.topk=32 distillation.distillation_loss.use_task_rewards=False distillation.distillation_loss.use_policy_gradient=False distillation.distillation_loss.loss_max_clamp=10.0 distillation.distillation_loss.log_prob_min_clamp=-10.0
