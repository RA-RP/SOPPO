#!/bin/bash

# """
# Train SSPO with llama3.
# Paste the commands to this file.

# This code is created based on the official code of LLaMA-Factory and the alignment handbook.
# (https://github.com/hiyouga/LLaMA-Factory)
# (https://github.com/huggingface/alignment-handbook)

# (Zheng, Y., Zhang, R., Zhang, J., Ye, Y., & Luo, Z. (2024). 
# Llamafactory: Unified efficient fine-tuning of 100+ language models. 
# arXiv preprint arXiv:2403.13372.)

# """

# Set PYTHONPATH
export PYTHONPATH="./src_sspo:$PYTHONPATH" # for SSPO
# export PYTHONPATH="./src/llamafactory/src:$PYTHONPATH" # for the others

# Use these if required
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export FORCE_TORCHRUN=1
export DISABLE_VERSION_CHECK=1

# Set GPU devices (modify GPU numbers as needed)
# export CUDA_VISIBLE_DEVICES="NUMBERS_OF_GPU_DEVICES"

export WANDB_PROJECT="YOUR_WANDB_PROJECT"
export WANDB_NAME="YOUR_WANDB_NAME"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"

# Run training : llamafactory-cli train "yaml file path"
# Copy and paste the commands from running the make_yaml.py file to this file.
# You can run multiple commands at once.

# llamafactory-cli train "YOUR_YAML_FILE_PATH"

llamafactory-cli train YOUR_YAML_FILE_PATH