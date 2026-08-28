"""
Preprocess ultramedpref & ultramed dataset
- Remove labels(chosen or rejected) of some pairs and randomly allocate them
- Convert to JSON format
- Save the dataset

This code is created based on the official code of LLaMA-Factory and the alignment handbook.
(https://github.com/hiyouga/LLaMA-Factory)
(https://github.com/huggingface/alignment-handbook)

(Zheng, Y., Zhang, R., Zhang, J., Ye, Y., & Luo, Z. (2024). 
Llamafactory: Unified efficient fine-tuning of 100+ language models. 
arXiv preprint arXiv:2403.13372.)

"""
import argparse
import json
import logging
import os
import random
from typing import List, Optional
import numpy as np
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

args = argparse.ArgumentParser()
args.add_argument("--train_num_ratio", type=float, default=1, help="The ratio of the training dataset to the original dataset. max is 1.")
args.add_argument("--fb", type=float, default=0.1, help="The ratio of remaining the training dataset to the original feedback dataset. max is 1.")
args.add_argument("--ch", type=float, default=0.1, help="The ratio of remaining the training dataset to the original unpaired (SFT) dataset. max is 1.")
args = args.parse_args()

train_num_ratio = args.train_num_ratio
ultramedpref_keep_ratio = args.fb
ultramed_keep_ratio = args.ch

#@Alignment Handbook utils
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

def get_datasets(
    data_config: dict,
    splits: List[str] = ["train", "test"],
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """

    if type(data_config) is dict:
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(dataset_mixer, splits=splits, shuffle=shuffle)
    return raw_datasets


def mix_datasets(dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for ds, frac in dataset_mixer.items():
        fracs.append(frac)
        for split in splits:
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, split=split)
            except DatasetGenerationError:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(ds, split))

            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
        )

    return raw_datasets




## 1. shuffle chosen and rejected
def shuffle_chosen_rejected(dataset, remove_ratio):
    """
    Randomly swaps chosen and rejected responses for a portion (remove_ratio) of the dataset
    (To capture the lower performance at unlabeled data setting)
    
    Args:
        dataset: Original dataset
        remove_ratio (float): Ratio of data to modify (0.0 ~ 1.0)
    
    Returns:
        Modified dataset
        
    """
    formatted_data = []
    total_samples = len(dataset)
    num_samples_to_modify = int(total_samples * remove_ratio)
    
    # randomly select indices to modify
    indices_to_modify = set(random.sample(range(total_samples), num_samples_to_modify))
    
    for idx, example in tqdm(enumerate(dataset), desc="Processing data"):
        formatted_example = {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }
        
        # if selected index, swap chosen and rejected with 50% probability
        if idx in indices_to_modify and random.random() < 0.5:
            formatted_example["chosen"], formatted_example["rejected"] = (
                formatted_example["rejected"],
                formatted_example["chosen"]
            )
            
        formatted_data.append(formatted_example)
    
    return formatted_data


## 2. keep partial data
def keep_partial_data(dataset, keep_ratio):
    """
    Function to randomly keep only keep_ratio portion of the dataset
    
    Args:
        dataset: Original dataset
        keep_ratio (float): Ratio of data to keep (0.0 ~ 1.0)
    
    Returns:
        List containing only the selected data
    """
    total_samples = len(dataset)
    num_samples_to_keep = int(total_samples * keep_ratio)
    
    # Randomly select indices to keep
    indices_to_keep = set(random.sample(range(total_samples), num_samples_to_keep))
    
    # Keep only data with selected indices
    kept_data = [
        example for idx, example in enumerate(dataset)
        if idx in indices_to_keep
    ]
    
    return kept_data


## 3. create PNU data

def create_pnu_data(dataset, remove_ratio):
    """
    Creates unlabeled data by randomly selecting either chosen or rejected responses for a portion of the dataset.
    
    Args:
        dataset: Original dataset containing 'prompt', 'chosen', and 'rejected' columns
        remove_ratio (float): Ratio of data to convert to unlabeled (0.0 ~ 1.0)
    
    Returns:
        Modified dataset with new 'unlabeled' column where:
        - remove_ratio portion of data has 'unlabeled' filled and 'chosen'/'rejected' empty
        - remaining data has 'unlabeled' empty but original 'chosen'/'rejected' preserved
    """

    formatted_data = []
    total_samples = len(dataset)
    num_samples_to_modify = int(total_samples * remove_ratio)
    
    # Randomly select indices to modify
    indices_to_modify = set(random.sample(range(total_samples), num_samples_to_modify))
    
    for idx, example in tqdm(enumerate(dataset), desc="Creating unlabeled data"):
        if idx in indices_to_modify:
            # For selected indices, randomly choose either 'chosen' or 'rejected' as unlabeled
            is_chosen = random.random() < 0.5
            formatted_example = {
                "prompt": example["prompt"],
                "chosen": "",  # Empty string for chosen
                "rejected": "",  # Empty string for rejected
                "unlabeled": example["chosen"] if is_chosen else example["rejected"]
            }
        else:
            # For unselected indices, keep original chosen/rejected and empty unlabeled
            formatted_example = {
                "prompt": example["prompt"],
                "chosen": example["chosen"],
                "rejected": example["rejected"],
                "unlabeled": ""  # Empty string for unlabeled
            }
        
        formatted_data.append(formatted_example)
    
    return formatted_data


## 4. convert to json format
def convert_to_json_format(dataset):
    """
    Convert dataset to JSON format.
    
    Args:
        dataset: Original dataset
    
    Returns:
        List of formatted data
    """
    formatted_data = []
    for example in tqdm(dataset, desc="Converting to JSON format"):
        formatted_example = {
            "instruction": example["prompt"],
            "chosen": example["chosen"][1]["content"] if isinstance(example["chosen"], list) else example["chosen"],
            "rejected": example["rejected"][1]["content"] if isinstance(example["rejected"], list) else example["rejected"],
            "unlabeled": example["unlabeled"][1]["content"] if isinstance(example["unlabeled"], list) else example["unlabeled"]
        }
        
        formatted_data.append(formatted_example)
    return formatted_data



######## load dataset and preprocess #########

raw_ultramedpref = get_datasets(
    {"TsinghuaC3I/UltraMedical-Preference" : train_num_ratio},
    splits = ["train"],
)

# Load ultramed dataset
ultramed_data = load_dataset("TsinghuaC3I/UltraMedical", split=["train"])

kept_ultramedpref = keep_partial_data(raw_ultramedpref['train'], keep_ratio=ultramedpref_keep_ratio)

kept_ultramed = keep_partial_data(ultramed_data[0], keep_ratio=ultramed_keep_ratio)

# Create new dataset by combining ultramedpref and ultramed
def create_combined_dataset(ultramedpref, ultramed):
    """
    Combine ultramedpref and ultramed datasets.
    
    Args:
        ultramedpref: Dataset with 'chosen' and 'rejected' columns.
        ultramed: Dataset with 'prompt' and 'unlabeled' columns.
    
    Returns:
        Combined dataset.
    """
    combined_data = []

    # Add ultramedpref data
    for uf_example in tqdm(ultramedpref, desc="Adding ultramedpref data"):
        combined_example = {
            "prompt": uf_example["prompt"],
            "chosen": uf_example["chosen"],
            "rejected": uf_example["rejected"],
            "unlabeled": ""  # Empty string for unlabeled
        }
        combined_data.append(combined_example)

    # Add ultramed data
    for uc_example in tqdm(ultramed, desc="Adding ultramedical data"):
        # conversations 배열에서 첫 번째는 human, 두 번째는 gpt
        conversations = uc_example["conversations"]
        if len(conversations) >= 2:
            human_message = conversations[0]["value"]  # 첫 번째 value (human)
            gpt_message = conversations[1]["value"]    # 두 번째 value (gpt)
            
            combined_example = {
                "prompt": human_message,
                "chosen": "",  # Empty string for chosen
                "rejected": "",  # Empty string for rejected
                "unlabeled": gpt_message
            }
            combined_data.append(combined_example)

    # Shuffle the combined dataset
    random.shuffle(combined_data)

    return combined_data

combined_dataset = create_combined_dataset(kept_ultramedpref, kept_ultramed) 

# Convert combined dataset to JSON format
conversation_format = convert_to_json_format(combined_dataset)
dataset_name = f"ultramed_fb{ultramedpref_keep_ratio}_ch{ultramed_keep_ratio}"
conversation_json_filename = f"./data/{dataset_name}.json"
with open(conversation_json_filename, 'w', encoding='utf-8') as f:
    json.dump(conversation_format, f, ensure_ascii=False, indent=2)

logger.info(f"Saved combined dataset: '{conversation_json_filename}'")

# Update dataset_info.json
dataset_info_path = "./data/dataset_info.json"

try:
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
except FileNotFoundError:
    dataset_info = {}

dataset_info[dataset_name] = {
    "file_name": conversation_json_filename,
    "ranking": True,
    "columns": {
        "prompt": "instruction",
        "chosen": "chosen",
        "rejected": "rejected",
        "unlabeled": "unlabeled"
    }
}

with open(dataset_info_path, 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, ensure_ascii=False, indent=2)

logger.info(f"Dataset info updated in '{dataset_info_path}'")





