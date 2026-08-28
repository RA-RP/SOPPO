"""
Train SSRM.
4) Merge the data.

This code is created based on the official code of LLaMA-Factory and SSRM.
(https://github.com/hiyouga/LLaMA-Factory)
(https://github.com/RLHFlow/RLHF-Reward-Modeling)

(He, Yifei, et al. "Semi-supervised reward modeling via iterative self-training." 
arXiv preprint arXiv:2409.06903 (2024).)

"""

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--filtered_data_path", type=str, default=None)
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
filtered_data_path = args.filtered_data_path


def merge_json_files():
    # Read first file
    with open(input_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    
    # Read second file
    with open(filtered_data_path, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)
    
    # Extract required fields from filtered_data_path
    if isinstance(data2, list):
        filtered_data2 = []
        for item in data2:
            filtered_item = {
                "instruction": item.get("instruction", ""),
                "chosen": item.get("chosen", ""),
                "rejected": item.get("rejected", ""),
                "unlabeled": ""
            }
            filtered_data2.append(filtered_item)
        data2 = filtered_data2
    
    # Merge data
    # If both files are lists
    if isinstance(data1, list) and isinstance(data2, list):
        merged_data = data1 + data2
    # If both files are dictionaries
    elif isinstance(data1, dict) and isinstance(data2, dict):
        merged_data = {**data1, **data2}
    else:
        raise ValueError("JSON file formats do not match. Both must be either lists or dictionaries.")
    
    # Save merged data to new file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"Files successfully merged. Saved to: {output_path}")

if __name__ == "__main__":
    merge_json_files()