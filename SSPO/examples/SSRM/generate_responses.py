"""
Train SSRM.
1) Generate responses from the base model.

This code is created based on the official code of LLaMA-Factory and SSRM.
(https://github.com/hiyouga/LLaMA-Factory)
(https://github.com/RLHFlow/RLHF-Reward-Modeling)

(He, Yifei, et al. "Semi-supervised reward modeling via iterative self-training." 
arXiv preprint arXiv:2409.06903 (2024).)

"""



import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import argparse
from typing import List, Dict
import torch.distributed as dist
from torch.utils.data import Dataset
import logging
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--base_model_path", type=str, required=True)
parser.add_argument("--adapter_model_path", type=str, default=None)
args = parser.parse_args()


class InstructionDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def setup_model_and_tokenizer():
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        if args.adapter_model_path is not None:
            model = PeftModel.from_pretrained(model, args.adapter_model_path)

        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        
        phi2_template = """{% for message in messages %}{% if message['role'] == 'user' %}Human: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}{% endfor %}"""
        tokenizer.chat_template = phi2_template
        
        return model, tokenizer, device
    except Exception as e:
        logger.error(f"Error setting up model and tokenizer: {str(e)}")
        raise

def generate_responses_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    instructions: List[str],
    batch_size: int = 8,
) -> List[str]:
    """Generate responses for multiple instructions in batches."""
    try:
        responses = []
        
        for i in tqdm(range(0, len(instructions), batch_size), desc="Generating responses"):
            batch_instructions = instructions[i:i + batch_size]
            batch_messages = [[{"role": "user", "content": inst}] for inst in batch_instructions]
            
            inputs = tokenizer.apply_chat_template(
                batch_messages,
                return_tensors="pt",
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)
            
            attention_mask = inputs.ne(tokenizer.pad_token_id).long()
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            for j in range(len(batch_instructions)):
                response = tokenizer.decode(
                    outputs[j][inputs.shape[1]:],
                    skip_special_tokens=True
                )
                responses.append(response)
        
        return responses
    except Exception as e:
        logger.error(f"Error generating responses: {str(e)}")
        raise

def gather_results(local_results: List[Dict], world_size: int, device: str) -> List[Dict]:
    """Collect results using tensor-based communication."""
    try:
        # Convert results to tensor
        max_length = max(len(json.dumps(result, ensure_ascii=False).encode('utf-8')) for result in local_results)
        padded_results = []
        
        for result in local_results:
            try:
                result_str = json.dumps(result, ensure_ascii=False)
                result_bytes = result_str.encode('utf-8')
                padded_bytes = result_bytes.ljust(max_length, b'\0')
                padded_results.append([int(b) for b in padded_bytes])
            except Exception as e:
                logger.error(f"Error serializing result: {str(e)}")
                continue
        
        if not padded_results:
            logger.error("No processable results available.")
            return []
        
        # Create tensor
        local_tensor = torch.tensor(padded_results, dtype=torch.int32, device=device)
        gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        
        # Gather results
        dist.all_gather(gathered_tensors, local_tensor)
        
        # Convert tensors back to results
        all_results = []
        for tensor in gathered_tensors:
            for row in tensor:
                try:
                    # Extract non-zero bytes
                    bytes_data = bytes([b for b in row.cpu().numpy() if b != 0])
                    if not bytes_data:
                        continue
                        
                    # Try UTF-8 decoding
                    try:
                        decoded_str = bytes_data.decode('utf-8')
                    except UnicodeDecodeError:
                        # Try alternative encoding if UTF-8 fails
                        decoded_str = bytes_data.decode('utf-8', errors='replace')
                    
                    result = json.loads(decoded_str)
                    all_results.append(result)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.error(f"Error converting data: {str(e)}")
                    continue
        
        return all_results
    except Exception as e:
        logger.error(f"Error gathering results: {str(e)}")
        raise

def main():
    try:        
        # Create output directory
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        
        # Initialize distributed training
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Assign GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
        
        # Set NCCL environment variables
        os.environ["NCCL_DEBUG"] = "WARN"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        
        # Initialize distributed training
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=local_rank,
        )
        
        logger.info(f"Process {local_rank}/{world_size} started (device: {device})")
        
        # Load input file
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Distribute data to each process
        chunk_size = len(data) // world_size
        start_idx = local_rank * chunk_size
        end_idx = start_idx + chunk_size if local_rank < world_size - 1 else len(data)
        local_data = data[start_idx:end_idx]
        
        logger.info(f"Process {local_rank}: processing {len(local_data)} examples")
        
        # Setup model and tokenizer
        model, tokenizer, device = setup_model_and_tokenizer()
        
        # Process local data
        instructions = [item["instruction"] for item in local_data]
        responses = generate_responses_batch(model, tokenizer, device, instructions, args.batch_size)
        
        # Collect results
        local_results = []
        for item, response in zip(local_data, responses):
            result = {
                "instruction": item["instruction"],
                "unlabeled": item["unlabeled"],
                "response": response
            }
            local_results.append(result)
        
        # Gather results from all processes
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, local_results)
        
        # Save results in main process
        if local_rank == 0:
            all_results = []
            for results in gathered_results:
                all_results.extend(results)
            
            # Save output file
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Generated responses saved to {args.output_file}")
        
        # Clean up distributed training
        dist.destroy_process_group()
        
    except Exception as e:
        logger.error(f"Error during program execution: {str(e)}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise

if __name__ == "__main__":
    main() 