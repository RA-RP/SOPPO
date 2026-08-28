"""
Train SSRM.
2) Pseudo-label the responses.

This code is created based on the official code of LLaMA-Factory and SSRM.
(https://github.com/hiyouga/LLaMA-Factory)
(https://github.com/RLHFlow/RLHF-Reward-Modeling)

(He, Yifei, et al. "Semi-supervised reward modeling via iterative self-training." 
arXiv preprint arXiv:2409.06903 (2024).)

"""



from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm.auto import trange, tqdm
import argparse
import torch.distributed as dist
import json
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--ultrachat_ratio", type=float, default=0.001)
parser.add_argument("--max_len", type=int, default=1024)
parser.add_argument("--base_model_path", type=str, default=None)
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--iteration", type=int, default=1)
args = parser.parse_args()

# specify where to save the pseudo-labeled dataset
ultrachat_ratio = args.ultrachat_ratio
dataset_path = args.dataset_path
max_len = args.max_len

def setup():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        gpu = 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    return rank, world_size, gpu

def cleanup():
    dist.destroy_process_group()

class SlicPairPMPipeline:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.tokenizer_data_format = AutoTokenizer.from_pretrained(
            args.base_model_path, use_fast=True
        )
        
        self.tokenizer_data_format.chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}Human: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}{% endfor %}"""

        self.prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"
        token_id_A = self.tokenizer.encode("A", add_special_tokens=False)
        token_id_B = self.tokenizer.encode("B", add_special_tokens=False)
        assert len(token_id_A) == 1 and len(token_id_B) == 1
        self.token_id_A = token_id_A[0]
        self.token_id_B = token_id_B[0]
        self.temperature = 0.7
        self.device = device
        self.batch_size = 8  # Set batch size

    def generate_responses(self, instructions):
        # Generate responses in batches
        responses = []
        for instruction in instructions:
            messages = [{"role": "user", "content": instruction}]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Create attention mask
            attention_mask = inputs.ne(self.tokenizer.pad_token_id).long()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            responses.append(response)
        return responses

    def process_batch(self, batch):
        # Handle batch data in dictionary format
        if isinstance(batch, dict):
            instructions = batch["instruction"]
            unlabeled_responses = batch["unlabeled"]
            response_list = batch["response"]
        # Handle batch data in list format
        else:
            instructions = [example["instruction"] for example in batch]
            unlabeled_responses = [example["unlabeled"] for example in batch]
            response_list = [example["response"] for example in batch]
        
        batch_probs = []
        batch_preferences = []
        
        for instruction, unlabeled, response in zip(instructions, unlabeled_responses, response_list):
            # Handle empty responses
            if (unlabeled is None or unlabeled.strip() == "") and (response is None or response.strip() == ""):
                # If both are empty, ignore (or set to 0.5)
                batch_probs.append(0.5)
                batch_preferences.append(0.5)
                continue
            elif unlabeled is None or unlabeled.strip() == "":
                # If unlabeled is empty, response is chosen
                batch_probs.append(0.0)
                batch_preferences.append(0.0)
                continue
            elif response is None or response.strip() == "":
                # If response is empty, unlabeled is chosen
                batch_probs.append(1.0)
                batch_preferences.append(1.0)
                continue

            chosen = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": unlabeled}
            ]
            
            rejected = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response}
            ]
            
            context = self.tokenizer_data_format.apply_chat_template(chosen[:-1], tokenize=False)
            responses = [chosen[-1]["content"], rejected[-1]["content"]]
            probs_chosen = []

            for chosen_position in [0, 1]:
                response_A = responses[chosen_position]
                response_B = responses[1 - chosen_position]
                prompt = self.prompt_template.format(context=context, response_A=response_A, response_B=response_B)
                message = [
                    {"role": "user", "content": prompt},
                ]

                input_ids = self.tokenizer.encode(
                    self.tokenizer.apply_chat_template(message, tokenize=False).replace(self.tokenizer.bos_token, ""),
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(self.device)

                with torch.no_grad():
                    output = self.model(input_ids)
                logit_A = output.logits[0, -1, self.token_id_A].item()
                logit_B = output.logits[0, -1, self.token_id_B].item()
                Z = np.exp(logit_A / self.temperature) + np.exp(logit_B / self.temperature)
                logit_chosen = [logit_A, logit_B][chosen_position]
                prob_chosen = np.exp(logit_chosen / self.temperature) / Z
                probs_chosen.append(prob_chosen)
            
            prob_chosen_A = np.mean(probs_chosen)
            batch_probs.append(prob_chosen_A)
            preference = 0.5 if prob_chosen_A == 0.5 else float(prob_chosen_A > 0.5)
            batch_preferences.append(preference)
        
        return batch_preferences, batch_probs

    def __call__(self, candidates_A, candidates_B, **kwargs):
        probs_choose_A = []
        preferences = []
        
        # Process in batches
        total_examples = len(candidates_A)
        total_batches = (total_examples + self.batch_size - 1) // self.batch_size
        
        # Show progress with tqdm
        with tqdm(total=total_batches, desc="Pseudo-labeling") as pbar:
            for i in range(0, total_examples, self.batch_size):
                # Get batch data
                batch = {
                    "instruction": candidates_A["instruction"][i:i + self.batch_size],
                    "unlabeled": candidates_A["unlabeled"][i:i + self.batch_size],
                    "response": candidates_A["response"][i:i + self.batch_size]
                }
                
                batch_preferences, batch_probs = self.process_batch(batch)
                preferences.extend(batch_preferences)
                probs_choose_A.extend(batch_probs)
                
                # Update progress
                pbar.update(1)
                pbar.set_postfix({
                    "processed": f"{i + len(batch['instruction'])}/{total_examples}",
                    "current_batch": len(batch['instruction'])
                })
        
        return preferences, probs_choose_A

def get_token_count(example):
    instruction = example["instruction"]
    unlabeled = example["unlabeled"]
    
    # Convert instruction and unlabeled to conversation format
    chosen = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": unlabeled}
    ]
    
    # Set rejected as empty response
    rejected = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": ""}
    ]
    
    try:
        context = pipeline.tokenizer_data_format.apply_chat_template(chosen[:-1], tokenize=False)
        responses = [chosen[-1]["content"], rejected[-1]["content"]]
        
        chosen_position = 0
        response_A = responses[chosen_position]
        response_B = responses[1 - chosen_position]
        prompt = pipeline.prompt_template.format(context=context, response_A=response_A, response_B=response_B)
        message = [
            {"role": "user", "content": prompt},
        ]

        input_ids = pipeline.tokenizer.encode(
            pipeline.tokenizer.apply_chat_template(message, tokenize=False).replace(pipeline.tokenizer.bos_token, ""),
            add_special_tokens=False,
        )
        return {"length": len(input_ids)}
    except Exception as e:
        print(f"Error processing example: {e}")
        return {"length": 0}

if __name__ == "__main__":
    rank, world_size, gpu = setup()
    device = torch.device(f"cuda:{gpu}")
    
    base_model_path = args.base_model_path
    model_path = args.model_path

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=None
    ).to(device)

    # Load trained LoRA weights
    model.load_adapter(model_path)
    model.eval()

    print(f'Loaded base model from {base_model_path} and adapter from {model_path} to gpu {device}')
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)

    # Configure tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # Set chat template
    llama3_template = """{% for message in messages %}{% if message['role'] == 'user' %}Human: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}{% endfor %}"""
    tokenizer.chat_template = llama3_template

    pipeline = SlicPairPMPipeline(model, tokenizer, device=device)

    # Load dataset
    ds = load_dataset('json', data_files=dataset_path, split="train")
    
    # Distribute data evenly
    total_size = len(ds)
    per_rank_size = total_size // world_size
    start_idx = rank * per_rank_size
    end_idx = start_idx + per_rank_size if rank != world_size - 1 else total_size
    
    ds_subset = ds.select(range(start_idx, end_idx))

    print(f"GPU {rank} is processing {len(ds_subset)} examples (indices {start_idx} to {end_idx})")
    
    ds_subset = ds_subset.map(get_token_count)
    lengths = np.array(ds_subset['length'])
    print(f'Data shard of {len(ds_subset)} examples, max length: {lengths.max()}, mean length: {lengths.mean()}')
    ds_subset = ds_subset.filter(lambda x: x['length'] <= max_len)
    print(f'Filtered examples of token length > {max_len}, remaining {len(ds_subset)} examples')

    print("Data preprocessing done. Start pseudo-labeling.")
    # Get pseudo-labels
    preferences, chosen_probs = pipeline(candidates_A=ds_subset, candidates_B=ds_subset)
    chosen_probs = np.array(chosen_probs)
    
    # Modify result save path
    folder = f'./LLaMA_Factory/examples/SSRM/{args.base_model_path.split("/")[-1]}-pred-probs_T{args.iteration}'
    os.makedirs(folder, exist_ok=True)
    ds_subset = ds_subset.add_column('chosen_prob', chosen_probs)
    
    # Collect results from each GPU
    all_data = []
    for i in range(len(ds_subset)):
        example = {
            "instruction": ds_subset[i]["instruction"],
            "unlabeled": ds_subset[i]["unlabeled"],
            "response": ds_subset[i]["response"],
            "chosen_prob": float(chosen_probs[i])
        }
        all_data.append(example)
    
    # Gather data from all GPUs
    gathered_data = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_data, all_data)
    
    # Save all results in one file from rank 0
    if rank == 0:
        final_data = []
        for data in gathered_data:
            final_data.extend(data)
    
        # Save as JSON file
        output_file = os.path.join(folder, f'pseudo_labeled_data_{ultrachat_ratio}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved final dataset with {len(final_data)} examples to {output_file}")
    
    cleanup()
