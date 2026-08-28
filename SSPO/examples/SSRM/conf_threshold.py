"""
Train SSRM.
3) Filter the data based on the confidence threshold.

This code is created based on the official code of LLaMA-Factory and SSRM.
(https://github.com/hiyouga/LLaMA-Factory)
(https://github.com/RLHFlow/RLHF-Reward-Modeling)

(He, Yifei, et al. "Semi-supervised reward modeling via iterative self-training." 
arXiv preprint arXiv:2409.06903 (2024).)

"""


import numpy as np
from transformers import AutoTokenizer
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--base_model_path", type=str, default=None)
parser.add_argument("--ultrachat_ratio", type=float, default=0.001)
parser.add_argument("--conf_threshold", type=float, default=0.8)
parser.add_argument("--iteration", type=int, default=1)
args = parser.parse_args()

HOME = os.path.expanduser("~")
folder = f'./LLaMA_Factory/examples/SSRM/{args.base_model_path.split("/")[-1]}-pred-probs_T{args.iteration}'

# Load JSON file directly
json_path = args.dataset_path
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

def filter_example(example):
    # Check if all required fields exist
    if not all(key in example for key in ['instruction', 'unlabeled', 'response']):
        return False
    
    # Check if content is not empty
    if not all(len(str(example[key])) > 0 for key in ['instruction', 'unlabeled', 'response']):
        return False
    
    return True

def process_example(example):
    # Randomly select position for A or B
    chosen_position = np.random.randint(2)
    label = ['A', 'B'][chosen_position]
    
    # Create context
    context = example['instruction']
    
    # Set responses
    responses = [example['unlabeled'], example['response']]
    response_A = responses[chosen_position]
    response_B = responses[1-chosen_position]
    
    # Apply prompt template
    my_prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"
    prompt = my_prompt_template.format(context=context, response_A=response_A, response_B=response_B)
    
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": label}
    ]
    
    return {
        "messages": messages,
        "chosen_prob": example['chosen_prob'],
        "instruction": example['instruction'],
        "unlabeled": example['unlabeled'],
        "response": example['response'],
        "chosen_position": chosen_position
    }

# Use tokenizer
tokenizer_path = args.base_model_path
tokenizer_plain = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer_plain.chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}Human: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}{% endfor %}"""

# Filter and process data
filtered_data = [example for example in data if filter_example(example)]
processed_data = [process_example(example) for example in filtered_data]

# Calculate confidence
chosen_probs = np.array([example['chosen_prob'] for example in processed_data])
confidence = np.maximum(chosen_probs, 1-chosen_probs)

# Filter based on confidence
conf_threshold = args.conf_threshold
mask = np.argwhere(confidence > conf_threshold).flatten()
filtered_indices = mask.tolist()

print(f'Confidence threshold: {conf_threshold}')
print(f'Total data: {len(processed_data)}')
print(f'Filtered data: {len(filtered_indices)}')
print(f'Percentage selected: {len(filtered_indices) / len(processed_data) * 100:.2f}%')

# Save filtered data
filtered_data = []
for i in filtered_indices:
    example = processed_data[i]
    chosen_position = example['chosen_position']
    responses = [example['unlabeled'], example['response']]
    
    final_example = {
        "instruction": example['instruction'],
        "chosen": responses[chosen_position],  # Winner response
        "rejected": responses[1-chosen_position],  # Loser response
        "win": "ultrachat" if chosen_position == 0 else "baseline",  # A is ultrachat, B is baseline
        "chosen_prob": example['chosen_prob']
    }
    filtered_data.append(final_example)

output_path = os.path.join(folder, f'filtered_data_{args.ultrachat_ratio}_conf{conf_threshold}.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f'Filtered data saved to: {output_path}')