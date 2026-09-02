"""
Preprocess ultrafeedback dataset
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
import hashlib
import json
import logging
import os
import random
from datetime import datetime, timezone
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
args.add_argument("--ultrafeedback_source", default="HuggingFaceH4/ultrafeedback_binarized")
args.add_argument("--ultrachat_source", default="HuggingFaceH4/ultrachat_200k")
args.add_argument("--ultrafeedback_revision", default=None)
args.add_argument("--ultrachat_revision", default=None)
args.add_argument("--output_dir", default="./data")
args = args.parse_args()

train_num_ratio = args.train_num_ratio
ultrafeedback_keep_ratio = args.fb
ultrachat_keep_ratio = args.ch
output_dir = os.path.abspath(args.output_dir)
os.makedirs(output_dir, exist_ok=True)
manifest_path = os.path.join(output_dir, "ROUND4_PREPROCESS_MANIFEST.json")
if os.path.exists(manifest_path):
    raise FileExistsError(f"Refusing to overwrite existing preprocessing manifest: {manifest_path}")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

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


def filter_invalid_preference_rows(dataset_name, payload):
    """Keep only rows accepted by the Round4 ranking-data contract.

    Frozen upstream snapshots contain a small number of empty or one-sided
    responses.  Sampling remains deterministic and ratio-based; invalid rows
    are removed only after sampling and their counts are recorded in the
    preprocessing manifest instead of being passed to DPO-family losses.
    """
    valid_rows = []
    dropped_reasons = {
        "malformed_fields": 0,
        "empty_instruction": 0,
        "empty_response": 0,
        "incomplete_labeled_pair": 0,
        "non_exclusive_response": 0,
    }

    for row in payload:
        if not isinstance(row, dict):
            dropped_reasons["malformed_fields"] += 1
            continue

        fields = [row.get(key) for key in ("instruction", "chosen", "rejected", "unlabeled")]
        if not all(isinstance(value, str) for value in fields):
            dropped_reasons["malformed_fields"] += 1
            continue
        instruction, chosen, rejected, unlabeled = (value.strip() for value in fields)
        if not instruction:
            dropped_reasons["empty_instruction"] += 1
            continue

        has_chosen = bool(chosen)
        has_rejected = bool(rejected)
        has_unlabeled = bool(unlabeled)
        labeled = has_chosen and has_rejected and not has_unlabeled
        unlabeled_only = has_unlabeled and not has_chosen and not has_rejected
        if labeled or unlabeled_only:
            valid_rows.append(row)
        elif not has_chosen and not has_rejected and not has_unlabeled:
            dropped_reasons["empty_response"] += 1
        elif has_chosen != has_rejected and not has_unlabeled:
            dropped_reasons["incomplete_labeled_pair"] += 1
        else:
            dropped_reasons["non_exclusive_response"] += 1

    dropped_reasons = {key: value for key, value in dropped_reasons.items() if value}
    dropped_rows = len(payload) - len(valid_rows)
    if not valid_rows:
        raise ValueError(f"No valid Round4 rows remain after filtering {dataset_name}.")
    if dropped_rows:
        logger.warning(
            "Filtered %d invalid rows from %s: %s",
            dropped_rows,
            dataset_name,
            dropped_reasons,
        )
    return valid_rows, {
        "input_rows": len(payload),
        "dropped_invalid_rows": dropped_rows,
        "dropped_reasons": dropped_reasons,
    }



######## load dataset and preprocess #########

def load_frozen_split(source, split, revision):
    kwargs = {"split": split}
    if revision and not os.path.isdir(source):
        kwargs["revision"] = revision
    return load_dataset(source, **kwargs)


raw_ultrafeedback_train = load_frozen_split(
    args.ultrafeedback_source,
    "train_prefs",
    args.ultrafeedback_revision,
).shuffle(seed=42)
raw_ultrafeedback_eval = load_frozen_split(
    args.ultrafeedback_source,
    "test_prefs",
    args.ultrafeedback_revision,
)
if not 0 <= train_num_ratio <= 1:
    raise ValueError("--train_num_ratio must be between 0 and 1.")
raw_ultrafeedback_train = raw_ultrafeedback_train.select(
    range(int(train_num_ratio * len(raw_ultrafeedback_train)))
)

# Round4 uses only the UltraChat train_sft split for the unlabeled stream.
ultrachat_train = load_frozen_split(
    args.ultrachat_source,
    "train_sft",
    args.ultrachat_revision,
).shuffle(seed=42)

kept_ultrafeedback = keep_partial_data(raw_ultrafeedback_train, keep_ratio=ultrafeedback_keep_ratio)

kept_ultrachat = keep_partial_data(ultrachat_train, keep_ratio=ultrachat_keep_ratio)

# Create new dataset by combining ultrafeedback and ultrachat
def create_combined_dataset(ultrafeedback, ultrachat, shuffle=True):
    """
    Combine ultrafeedback and ultrachat datasets.
    
    Args:
        ultrafeedback: Dataset with 'chosen' and 'rejected' columns.
        ultrachat: Dataset with 'prompt' and 'unlabeled' columns.
    
    Returns:
        Combined dataset.
    """
    combined_data = []

    # Add ultrafeedback data
    for uf_example in tqdm(ultrafeedback, desc="Adding ultrafeedback data"):
        combined_example = {
            "prompt": uf_example["prompt"],
            "chosen": uf_example["chosen"],
            "rejected": uf_example["rejected"],
            "unlabeled": ""  # Empty string for unlabeled
        }
        combined_data.append(combined_example)

    # Add ultrachat data
    for uc_example in tqdm(ultrachat, desc="Adding ultrachat data"):
        combined_example = {
            "prompt": uc_example["prompt"],
            "chosen": "",  # Empty string for chosen
            "rejected": "",  # Empty string for rejected
            "unlabeled": uc_example["messages"]
        }
        combined_data.append(combined_example)

    # Shuffle the combined dataset
    if shuffle:
        random.shuffle(combined_data)

    return combined_data

combined_dataset = create_combined_dataset(kept_ultrafeedback, kept_ultrachat)
dpo_dataset = create_combined_dataset(kept_ultrafeedback, [], shuffle=False)
eval_dataset = create_combined_dataset(raw_ultrafeedback_eval, [], shuffle=False)

raw_dataset_payloads = {
    f"ultra_combined_fb{ultrafeedback_keep_ratio}_ch{ultrachat_keep_ratio}": convert_to_json_format(combined_dataset),
    f"ultrafeedback_fb{ultrafeedback_keep_ratio}_dpo": convert_to_json_format(dpo_dataset),
    "ultrafeedback_round4_eval": convert_to_json_format(eval_dataset),
}
dataset_payloads = {}
filter_audit = {}
for dataset_name, payload in raw_dataset_payloads.items():
    dataset_payloads[dataset_name], filter_audit[dataset_name] = filter_invalid_preference_rows(dataset_name, payload)

for dataset_name, payload in dataset_payloads.items():
    output_path = os.path.join(output_dir, f"{dataset_name}.json")
    if os.path.exists(output_path):
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_path}")
    temporary_path = output_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
    os.replace(temporary_path, output_path)
    logger.info(f"Saved Round4 dataset: '{output_path}'")

# Update dataset_info.json in the same dataset directory.
dataset_info_path = os.path.join(output_dir, "dataset_info.json")

try:
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
except FileNotFoundError:
    dataset_info = {}

for dataset_name in dataset_payloads:
    dataset_info[dataset_name] = {
        "file_name": f"{dataset_name}.json",
        "ranking": True,
        "columns": {
            "prompt": "instruction",
            "chosen": "chosen",
            "rejected": "rejected",
            "unlabeled": "unlabeled"
        }
    }

temporary_dataset_info = dataset_info_path + ".tmp"
with open(temporary_dataset_info, 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, ensure_ascii=False, indent=2)
os.replace(temporary_dataset_info, dataset_info_path)

logger.info(f"Dataset info updated in '{dataset_info_path}'")

manifest = {
    "schema": "round4-preprocessing-v2",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "seed": 42,
    "filter_policy": "exclusive_nonempty_preference_or_unlabeled_v1",
    "ratios": {
        "train_num_ratio": train_num_ratio,
        "ultrafeedback": ultrafeedback_keep_ratio,
        "ultrachat": ultrachat_keep_ratio,
    },
    "sources": {
        "ultrafeedback": {
            "source": os.path.abspath(args.ultrafeedback_source),
            "revision": args.ultrafeedback_revision,
        },
        "ultrachat": {
            "source": os.path.abspath(args.ultrachat_source),
            "revision": args.ultrachat_revision,
        },
    },
    "outputs": {},
}
for dataset_name, payload in dataset_payloads.items():
    file_name = f"{dataset_name}.json"
    file_path = os.path.join(output_dir, file_name)
    labeled_rows = sum(bool(row["chosen"].strip() and row["rejected"].strip()) for row in payload)
    unlabeled_rows = sum(bool(row["unlabeled"].strip()) for row in payload)
    manifest["outputs"][dataset_name] = {
        "file_name": file_name,
        "sha256": sha256_file(file_path),
        **filter_audit[dataset_name],
        "rows": len(payload),
        "labeled_rows": labeled_rows,
        "unlabeled_rows": unlabeled_rows,
    }
manifest["dataset_info_sha256"] = sha256_file(dataset_info_path)
temporary_manifest = manifest_path + ".tmp"
with open(temporary_manifest, "w", encoding="utf-8") as file:
    json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
os.replace(temporary_manifest, manifest_path)
logger.info(f"Round4 preprocessing manifest saved in '{manifest_path}'")


