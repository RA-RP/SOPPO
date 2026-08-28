#!/bin/bash

"""
Train SSRM.
Run this script to train SSRM with iterations.

This code is created based on the official code of LLaMA-Factory and SSRM.
(https://github.com/hiyouga/LLaMA-Factory)
(https://github.com/RLHFlow/RLHF-Reward-Modeling)

(He, Yifei, et al. 
Semi-supervised reward modeling via iterative self-training.
arXiv preprint arXiv:2409.06903 (2024).)

"""


# Set PYTHONPATH
export PYTHONPATH="./src/llamafactory/src:$PYTHONPATH" # for the others

# Use these if required
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export FORCE_TORCHRUN=1
export DISABLE_VERSION_CHECK=1

# Set GPU devices (modify GPU numbers as needed)
export CUDA_VISIBLE_DEVICES="NUMBERS_OF_GPU_DEVICES"

export WANDB_PROJECT="YOUR_WANDB_PROJECT"
export WANDB_NAME="YOUR_WANDB_NAME"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"

# 1. generate responses

torchrun --nproc_per_node=1 \
        generate_responses.py \
        --input_file ./data/ultra_combined_fb0.1_ch0.0.json \
        --output_file ./examples/SSRM/data/generated.json \
        --batch_size 32 \
        --base_model_path "BASE_MODEL_PATH" \ 
        --adapter_model_path "ADAPTER_MODEL_PATH_IF_EXISTS"


# T=1
echo "Starting T=1 iteration..."

# 2. pseudo label (T=1)
torchrun --nproc_per_node=1 \
        pseudo_label.py \
        --dataset_path ./examples/SSRM/data/generated.json \
        --ultrachat_ratio 0.1 \
        --max_len 1024 \
        --iteration 1 \
        --base_model_path "base_model_path_of_the_reward_model" \
        --model_path "path_of_the_trained_adapter_of_the_reward_model" 

# 3. conf threshold (0.8)

python conf_threshold.py --dataset_path ./examples/SSRM/data/pseudo_labeled_data_0.1.json \
        --ultrachat_ratio 0.1 \
        --conf_threshold 0.8 \
        --iteration 1

# # 4. merge json
python merge_json.py  --output_path ./data/ultra_combined_fb0.1_ch0.0_ssrm_phi-2_t1.json \
        --input_path ./data/ultra_combined_fb0.1_ch0.0.json \
        --filtered_data_path ./examples/SSRM/data/filtered_data_0.1_conf0.8.json       



# 5. train reward model
# Set PYTHONPATH
export PYTHONPATH="./src/llamafactory/src:$PYTHONPATH"

llamafactory-cli train ./examples/SSRM/train-ssrm-phi-2.yaml
## end of T=1
## Paste the similar commands for T=2, 3, ...
