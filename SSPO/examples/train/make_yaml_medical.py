"""
Make yaml file for training DPO, ORPO, SimPO, KTO and SSPO.

base SFT model : 
mistral : https://huggingface.co/dmis-lab/meerkat-7b-v1.0
llama-3 : https://huggingface.co/TsinghuaC3I/Llama-3-8B-UltraMedical

or, use any other SFT model.

This code is created based on the official code of LLaMA-Factory and the alignment handbook.
(https://github.com/hiyouga/LLaMA-Factory)
(https://github.com/huggingface/alignment-handbook)

(Zheng, Y., Zhang, R., Zhang, J., Ye, Y., & Luo, Z. (2024). 
Llamafactory: Unified efficient fine-tuning of 100+ language models. 
arXiv preprint arXiv:2403.13372.)

"""

import yaml
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--peft", type=str, default="lora", help="full or lora or q-lora")
parser.add_argument("--method", type=str, default="sspo", help="sft, dpo, orpo, simpo, kto, or sspo")
parser.add_argument("--model_path", type=str, default="TsinghuaC3I/Llama-3-8B-UltraMedical", help="TsinghuaC3I/Llama-3-8B-UltraMedical or dmis-lab/meerkat-7b-v1.0")
args = parser.parse_args()

peft = args.peft
method = args.method
model_path = args.model_path

if peft == "full":
    finetuning_type = "full"
else:
    finetuning_type = "lora"

if method == "sft":
    stage = "sft"
else:
    stage = "dpo"

# Determine cache directory based on model_path
def get_cache_dir(model_path):
    model_name = model_path.lower()
    if "llama" in model_name and "3" in model_name:
        return "./cache/llama3-8b-it-medical"
    elif "meerkat" in model_name:
        return "./cache/mistral-7b-it-medical"
    else:
        # Fallback to model name
        return f"./cache/{model_path.split('/')[-1]}"

# Determine trust_remote_code setting based on model_path
def get_trust_remote_code(model_path):
    if model_path == "lole25/phi-2-sft-ultrachat-full":
        return False
    else:
        return True

# Determine backbone name based on model_path
def get_backbone_name(model_path):
    model_name = model_path.lower()
    if "llama" in model_name and "3" in model_name:
        return "llama3-8b-it-medical"
    elif "meerkat" in model_name:
        return "mistral-7b-it-medical"
    else:
        # Fallback to original logic
        return model_path.split('/')[-1]

base_config = {
    "model_name_or_path": model_path,
    "trust_remote_code": get_trust_remote_code(model_path),
    "stage": stage,
    "do_train": True,
    "finetuning_type": finetuning_type,
    "template": "default", # please change this to preferred template. Refer to https://github.com/hiyouga/LLaMA-Factory
    "cutoff_len": 1024,
    "max_samples": 10000000,
    "overwrite_cache": True,
    "preprocessing_num_workers": 12,
    "max_grad_norm": 1.0,
    "logging_steps": 20, #5
    "save_steps": 300, #30
    "plot_loss": True,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "ddp_timeout": 180000000,
    "val_size": 0.1,
    "per_device_eval_batch_size": 1,
    "eval_strategy": "steps",
    "eval_steps": 100, #10
    "cache_dir": get_cache_dir(model_path),
}

# hyperparameters
datasets = ["ultramed_fb0.1_ch0.05"]
fb_ratio = 0.1
ch_ratio = 0.05
learning_rates = [5e-6]
num_train_epochs = [1]
lora_ranks = [8]

sspo_gamma_decays = [0.001] #, 0.05, 0.005, 0.001]
sspo_priors = [0.1, 0.3, 0.7, 0.9, 0.01, 0.09]
sspo_gamma_mins = [round(1093/(1093+20479), 4)] # n_L / (n_L + n_U) # 10935, 20479
sspo_gamma_0s = [1.0]
sspo_bases = ["simpo"]  # Add sspo_base options

# gpu 2개 기준 -> train batch size 총 64
per_device_train_batch_sizes = [4]
per_device_eval_batch_sizes = [4]
gradient_accumulation_steps = [8]
pref_betas = [10.0]
cutoff_lens = [1024]
simpo_gammas = [2.0]


# generate yaml files
combinations = []
for dataset in datasets:
    for lr in learning_rates:
        for tb in per_device_train_batch_sizes:
            for eb in per_device_eval_batch_sizes:
                for ga in gradient_accumulation_steps:
                    for epochs in num_train_epochs:
                        for rank in lora_ranks:
                            for sspo_gamma_decay in sspo_gamma_decays:
                                for sspo_gamma_0 in sspo_gamma_0s:
                                    for sspo_prior in sspo_priors:
                                        for sspo_gamma_min in sspo_gamma_mins:
                                            for sspo_base in sspo_bases:
                                                for cutoff_len in cutoff_lens:
                                                    for beta in pref_betas:
                                                        for simpo_gamma in simpo_gammas:
                                                            combinations.append((dataset, lr, tb, eb, ga, epochs, rank, sspo_gamma_decay, sspo_gamma_0, sspo_prior, sspo_gamma_min, sspo_base, cutoff_len, beta, simpo_gamma))

print(f"We have {len(combinations)} combinations. Copy and paste the following command to the train-sspo-sweep.sh file.")
print("======================")

# Create directory structure: train/{backbone_name}/{method_name}/{fb_ratio}_ch{ch_ratio}/
backbone_name = get_backbone_name(model_path)
yaml_dir = f"./examples/train/{backbone_name}/{method}/fb{fb_ratio}_ch{ch_ratio}/"
os.makedirs(yaml_dir, exist_ok=True)

for (dataset, lr, tb, eb, ga, epochs, rank, sspo_gamma_decay, sspo_gamma_0, sspo_prior, sspo_gamma_min, sspo_base, cutoff_len, beta, simpo_gamma) in combinations:
    config = base_config.copy()
    
    if peft == "q-lora":
        config.update({
            "quantization_bit": 4,
            "quantization_method": "bitsandbytes",
            "dataset": dataset,
            "learning_rate": lr,
            "num_train_epochs": epochs,
            "lora_rank": rank,
            "lora_target": "all",
            "per_device_train_batch_size": tb,
            "per_device_eval_batch_size": eb,
            "gradient_accumulation_steps": ga,
            "cutoff_len": cutoff_len,
            "output_dir": f"./saves_{model_path.split('/')[-1]}/fb{fb_ratio}_ch{ch_ratio}/{peft}_{model_path.split('/')[-1]}_{method}_lr{lr}_rank{rank}_beta{beta}_margins{simpo_gamma}_prior{sspo_prior}_gamma_decay{sspo_gamma_decay}_gamma_init{sspo_gamma_0}_gamma_min{sspo_gamma_min}_cutoff{cutoff_len}_ep{epochs}_tb{tb}_eb{eb}_ga{ga}"
        })

        if method != "sft":
            config.update({
                "pref_beta": beta,
                "pref_loss": method,
                "simpo_gamma": simpo_gamma,
            })
        
        if method == "sspo":
            config.update({
                "sspo_gamma_decay": sspo_gamma_decay,
                "sspo_gamma_0": sspo_gamma_0,
                "sspo_gamma_min": sspo_gamma_min,
                "sspo_prior": sspo_prior,
                "simpo_gamma": simpo_gamma,
                "sspo_base": sspo_base,
                "sspo_min_labeled_per_batch": 2,  # Add minimum labeled data per batch
            })

        filename = f"fb{fb_ratio}_ch{ch_ratio}_{peft}_{model_path.split('/')[-1]}_{method}_lr{lr}_rank{rank}_beta{beta}_margins{simpo_gamma}_prior{sspo_prior}_gamma_decay{sspo_gamma_decay}_gamma_init{sspo_gamma_0}_gamma_min{sspo_gamma_min}_base{sspo_base}_cutoff{cutoff_len}_ep{epochs}_tb{tb}_eb{eb}_ga{ga}.yaml"

    elif peft == "lora":
        config.update({
            "dataset": dataset,
            "learning_rate": lr,
            "num_train_epochs": epochs,
            "lora_rank": rank,
            "lora_target": "all",
            "per_device_train_batch_size": tb,
            "per_device_eval_batch_size": eb,
            "gradient_accumulation_steps": ga,
            "cutoff_len": cutoff_len,
            "output_dir": f"./saves_{model_path.split('/')[-1]}/fb{fb_ratio}_ch{ch_ratio}/{peft}_{model_path.split('/')[-1]}_{method}_lr{lr}_rank{rank}_beta{beta}_margins{simpo_gamma}_prior{sspo_prior}_gamma_decay{sspo_gamma_decay}_gamma_init{sspo_gamma_0}_gamma_min{sspo_gamma_min}_cutoff{cutoff_len}_ep{epochs}_tb{tb}_eb{eb}_ga{ga}"
        })

        if method != "sft":
            if method == "dpo":
                config.update({
                    "pref_loss": "sigmoid",
                    "pref_beta": beta,
                    "simpo_gamma": simpo_gamma,
                })
            else:
                config.update({
                    "pref_beta": beta,
                    "pref_loss": method,
                    "simpo_gamma": simpo_gamma,
                })
        
        if method == "sspo":
            config.update({
                "sspo_gamma_decay": sspo_gamma_decay,
                "sspo_gamma_0": sspo_gamma_0,
                "sspo_gamma_min": sspo_gamma_min,
                "sspo_prior": sspo_prior,
                "simpo_gamma": simpo_gamma,
                "sspo_base": sspo_base,
                "sspo_min_labeled_per_batch": 2,  # Add minimum labeled data per batch
            })

        filename = f"fb{fb_ratio}_ch{ch_ratio}_{peft}_{model_path.split('/')[-1]}_{method}_lr{lr}_rank{rank}_beta{beta}_margins{simpo_gamma}_prior{sspo_prior}_gamma_decay{sspo_gamma_decay}_gamma_init{sspo_gamma_0}_gamma_min{sspo_gamma_min}_base{sspo_base}_cutoff{cutoff_len}_ep{epochs}_tb{tb}_eb{eb}_ga{ga}.yaml"

    elif peft == "full":
        config.update({
            "dataset": dataset,
            "learning_rate": lr,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": tb,
            "per_device_eval_batch_size": eb,
            "gradient_accumulation_steps": ga,
            "cutoff_len": cutoff_len,
            "output_dir": f"./saves_{model_path.split('/')[-1]}/fb{fb_ratio}_ch{ch_ratio}/{peft}_{model_path.split('/')[-1]}_{method}_lr{lr}_rank{rank}_beta{beta}_margins{simpo_gamma}_prior{sspo_prior}_gamma_decay{sspo_gamma_decay}_gamma_init{sspo_gamma_0}_gamma_min{sspo_gamma_min}_cutoff{cutoff_len}_ep{epochs}_tb{tb}_eb{eb}_ga{ga}"
        })

        if method != "sft":
            config.update({
                "pref_beta": beta,
                "pref_loss": method,
                "simpo_gamma": simpo_gamma,
            })
        
        if method == "sspo":
            config.update({
                "sspo_gamma_decay": sspo_gamma_decay,
                "sspo_gamma_0": sspo_gamma_0,
                "sspo_gamma_min": sspo_gamma_min,
                "sspo_prior": sspo_prior,
                "simpo_gamma": simpo_gamma,
                "sspo_base": sspo_base,
                "sspo_min_labeled_per_batch": 2,  # Add minimum labeled data per batch
            })

        filename = f"fb{fb_ratio}_ch{ch_ratio}_{peft}_{model_path.split('/')[-1]}_{method}_lr{lr}_beta{beta}_margins{simpo_gamma}_prior{sspo_prior}_gamma_decay{sspo_gamma_decay}_gamma_init{sspo_gamma_0}_gamma_min{sspo_gamma_min}_base{sspo_base}_cutoff{cutoff_len}_ep{epochs}_tb{tb}_eb{eb}_ga{ga}.yaml"

    filepath = os.path.join(yaml_dir, filename)
    
    print("llamafactory-cli train "+filepath)
    
    try:
        with open(filepath, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        print(f"Error occurred while creating file: {e}")

