"""
Make yaml file for training DPO, ORPO, SimPO, KTO and SSPO.

base SFT model : 
phi-2 : https://huggingface.co/lole25/phi-2-sft-ultrachat-full (set "trust_remote_code" to False in the yaml file.)
mistral : https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
llama-3 : https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

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
parser.add_argument("--method", type=str, default="sspo", help="sft, dpo, orpo, simpo, kto, sspo, or staticpe")
parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-1.7B", help="meta-llama/Meta-Llama-3-8B-Instruct or lole25/phi-2-sft-ultrachat-full or mistralai/Mistral-7B-Instruct-v0.2 or Qwen/Qwen3-1.7B or Qwen/Qwen3-4B")
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
    if "phi" in model_name:
        return "./cache/phi-2"
    elif "llama" in model_name and "3" in model_name:
        return "./cache/llama3-8b-it"
    elif "mistral" in model_name:
        return "./cache/mistral-7b-it"
    elif "qwen" in model_name and "1.7b" in model_name:
        return "./cache/qwen3-1.7b-it"
    elif "qwen" in model_name and "4b" in model_name:
        return "./cache/qwen3-4b-it"
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
    if "phi" in model_name:
        return "phi-2"
    elif "llama" in model_name and "3" in model_name:
        return "llama3-8b-it"
    elif "mistral" in model_name:
        return "mistral-7b-it"
    elif "qwen" in model_name and "1.7b" in model_name:
        return "qwen3-1.7b-it"
    elif "qwen" in model_name and "4b" in model_name:
        return "qwen3-4b-it"
    else:
        # Fallback to original logic
        return model_path.split('/')[-1]

def get_template_name(model_path):
    return "qwen3" if "qwen3" in model_path.lower() else "default"

base_config = {
    "model_name_or_path": model_path,
    "trust_remote_code": get_trust_remote_code(model_path),
    "stage": stage,
    "do_train": True,
    "finetuning_type": finetuning_type,
    "template": get_template_name(model_path),
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

if model_path == "Qwen/Qwen3-1.7B":
    # Keep training and one-time candidate generation on the same initialization.
    base_config["model_revision"] = "b9352fbb8ce704292730cf54b3b1dceb2a808738"


if method == "staticpe":
    staticpe_beta = 0.1
    staticpe_lambda = 0.1
    staticpe_epochs = 1
    staticpe_train_batch_size = 4
    staticpe_eval_batch_size = 4
    staticpe_gradient_accumulation = 8
    staticpe_rank = 8
    staticpe_cutoff_len = 1024
    staticpe_dataset = "ultra_combined_fb0.1_ch0.1_staticpe"

    config = base_config.copy()
    config.update({
        "dataset": staticpe_dataset,
        "learning_rate": 1e-5,
        "num_train_epochs": staticpe_epochs,
        "per_device_train_batch_size": staticpe_train_batch_size,
        "per_device_eval_batch_size": staticpe_eval_batch_size,
        # Gradients are accumulated over conventional distributed micro-batches.
        "gradient_accumulation_steps": staticpe_gradient_accumulation,
        "cutoff_len": staticpe_cutoff_len,
        "pref_loss": "staticpe",
        "pref_beta": staticpe_beta,
        "staticpe_lambda": staticpe_lambda,
        "staticpe_epsilon": 1e-8,
        # The source labeled ratio is below 1/4, so preserving it requires a
        # small number of four-row micro-batches with no labeled row.
        "staticpe_min_labeled_per_batch": 0,
        "staticpe_min_unlabeled_per_batch": 2,
        "bf16": True,
        "fp16": False,
    })
    if peft in ["lora", "q-lora"]:
        config.update({
            "lora_rank": staticpe_rank,
            "lora_target": "all",
        })
    if peft == "q-lora":
        config.update({
            "quantization_bit": 4,
            "quantization_method": "bitsandbytes",
        })
    if peft not in ["full", "lora", "q-lora"]:
        raise ValueError("StaticPE supports --peft full, lora, or q-lora.")

    model_name = model_path.split("/")[-1]
    run_name = (
        f"{peft}_{model_name}_staticpe_lr1e-05"
        f"_beta{staticpe_beta}_lambda{staticpe_lambda}_cutoff{staticpe_cutoff_len}"
        f"_ep{staticpe_epochs}_tb{staticpe_train_batch_size}_eb{staticpe_eval_batch_size}"
        f"_ga{staticpe_gradient_accumulation}"
    )
    if peft != "full":
        run_name += f"_rank{staticpe_rank}"
    config["output_dir"] = f"./saves_{model_name}/fb0.1_ch0.1/{run_name}"

    yaml_dir = f"./examples/train/{get_backbone_name(model_path)}/staticpe/fb0.1_ch0.1/"
    os.makedirs(yaml_dir, exist_ok=True)
    filepath = os.path.join(yaml_dir, f"{run_name}.yaml")
    with open(filepath, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)
    print("We have 1 StaticPE configuration.")
    print("llamafactory-cli train " + filepath)
    raise SystemExit(0)

# hyperparameters
datasets = ["ultra_combined_fb0.1_ch0.1"]
fb_ratio = 0.1
ch_ratio = 0.1
learning_rates = [1e-5]
num_train_epochs = [2]
lora_ranks = [8]

sspo_gamma_decays = [0.001] #, 0.05, 0.005, 0.001]
sspo_priors = [0.5]
sspo_gamma_mins = [round(6113/(6113+20785), 4)] # n_L / (n_L + n_U) # 6113, 20785
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
