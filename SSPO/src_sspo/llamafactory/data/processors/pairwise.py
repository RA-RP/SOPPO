# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template


logger = logging.get_logger(__name__)


def _encode_pairwise_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
) -> Dict[str, List[int]]:
    # Check data type - distinguish between chosen/rejected pairs and unlabeled data
    has_chosen_rejected = response[0]["content"] != "" and response[1]["content"] != ""
    has_unlabeled = len(response) > 2 and response[2]["content"] != ""
    
    result = {}
    
    # Process chosen/rejected pairs if they exist
    if has_chosen_rejected:
        chosen_messages = template.mm_plugin.process_messages(prompt + [response[0]], images, videos, processor)
        rejected_messages = template.mm_plugin.process_messages(prompt + [response[1]], images, videos, processor)
        prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, chosen_messages, system, tools)
        _, rejected_ids = template.encode_oneturn(tokenizer, rejected_messages, system, tools)
        
        if template.efficient_eos:
            chosen_ids += [tokenizer.eos_token_id]
            rejected_ids += [tokenizer.eos_token_id]
            
        prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, images, videos, tokenizer, processor)
        
        # Consider the response is more important
        source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), cutoff_len)
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]
        
        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
        
        result["chosen_input_ids"] = chosen_input_ids
        result["chosen_labels"] = chosen_labels
        result["rejected_input_ids"] = rejected_input_ids
        result["rejected_labels"] = rejected_labels
    else:
        # Set empty lists if chosen/rejected are empty
        result["chosen_input_ids"] = []
        result["chosen_labels"] = []
        result["rejected_input_ids"] = []
        result["rejected_labels"] = []
    
    # Process unlabeled data if it exists
    if has_unlabeled:
        unlabeled_messages = template.mm_plugin.process_messages(prompt + [response[2]], images, videos, processor)
        prompt_ids, unlabeled_ids = template.encode_oneturn(tokenizer, unlabeled_messages, system, tools)
        
        if template.efficient_eos:
            unlabeled_ids += [tokenizer.eos_token_id]
            
        prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, images, videos, tokenizer, processor)
        
        source_len, target_len = infer_seqlen(len(prompt_ids), len(unlabeled_ids), cutoff_len)
        prompt_ids = prompt_ids[:source_len]
        unlabeled_ids = unlabeled_ids[:target_len]
        
        unlabeled_input_ids = prompt_ids + unlabeled_ids
        unlabeled_labels = [IGNORE_INDEX] * source_len + unlabeled_ids
        
        result["unlabeled_input_ids"] = unlabeled_input_ids
        result["unlabeled_labels"] = unlabeled_labels
    else:
        # Set empty lists if unlabeled is empty
        result["unlabeled_input_ids"] = []
        result["unlabeled_labels"] = []
    
    return result


def preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # Initialize counters for tracking data statistics
    total_examples = len(examples["_prompt"])
    valid_examples = 0
    labeled_examples = 0
    unlabeled_examples = 0
    
    # Initialize dictionary for storing results
    model_inputs = defaultdict(list)
    
    # Separate and collect labeled and unlabeled data
    labeled_data = []
    unlabeled_data = []
    
    for i in range(len(examples["_prompt"])):
        # Basic validation - check prompt structure
        if len(examples["_prompt"][i]) % 2 != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i])
            )
            continue
        
        # Validate responses and check types
        responses = examples["_response"][i]
        if len(responses) < 2:
            logger.warning_rank0(
                "Dropped example with insufficient responses: {}".format(examples["_response"][i])
            )
            continue
        
        # Check data type - whether there are chosen/rejected pairs or only unlabeled data
        has_chosen_rejected = responses[0]["content"] != "" and responses[1]["content"] != ""
        has_unlabeled = len(responses) > 2 and responses[2]["content"] != ""
        
        if not (has_chosen_rejected or has_unlabeled):
            logger.warning_rank0(
                "Dropped example with no valid responses: {}".format(responses)
            )
            continue
        
        valid_examples += 1
        
        # Encode the example
        encoded_example = _encode_pairwise_example(
            prompt=examples["_prompt"][i],
            response=responses,
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
        )
        
        # Classify data by type
        if has_chosen_rejected:
            labeled_examples += 1
            labeled_data.append({
                "chosen_input_ids": encoded_example["chosen_input_ids"],
                "chosen_labels": encoded_example["chosen_labels"],
                "rejected_input_ids": encoded_example["rejected_input_ids"],
                "rejected_labels": encoded_example["rejected_labels"],
                "images": examples["_images"][i] or [],
                "videos": examples["_videos"][i] or []
            })
        
        if has_unlabeled:
            unlabeled_examples += 1
            unlabeled_data.append({
                "unlabeled_input_ids": encoded_example["unlabeled_input_ids"],
                "unlabeled_labels": encoded_example["unlabeled_labels"],
                "images": examples["_images"][i] or [],
                "videos": examples["_videos"][i] or []
            })
    
    # Convert data back to original format
    for item in labeled_data:
        model_inputs["chosen_input_ids"].append(item["chosen_input_ids"])
        model_inputs["chosen_labels"].append(item["chosen_labels"])
        model_inputs["rejected_input_ids"].append(item["rejected_input_ids"])
        model_inputs["rejected_labels"].append(item["rejected_labels"])
        model_inputs["chosen_attention_mask"].append([1] * len(item["chosen_input_ids"]))
        model_inputs["rejected_attention_mask"].append([1] * len(item["rejected_input_ids"]))
        model_inputs["data_types"].append("labeled")
        model_inputs["images"].append(item["images"])
        model_inputs["videos"].append(item["videos"])
        # Set empty values for unlabeled fields
        model_inputs["unlabeled_input_ids"].append([])
        model_inputs["unlabeled_labels"].append([])
        model_inputs["unlabeled_attention_mask"].append([])
    
    for item in unlabeled_data:
        model_inputs["unlabeled_input_ids"].append(item["unlabeled_input_ids"])
        model_inputs["unlabeled_labels"].append(item["unlabeled_labels"])
        model_inputs["unlabeled_attention_mask"].append([1] * len(item["unlabeled_input_ids"]))
        model_inputs["data_types"].append("unlabeled")
        model_inputs["images"].append(item["images"])
        model_inputs["videos"].append(item["videos"])
        # Set empty values for chosen/rejected fields
        model_inputs["chosen_input_ids"].append([])
        model_inputs["chosen_labels"].append([])
        model_inputs["rejected_input_ids"].append([])
        model_inputs["rejected_labels"].append([])
        model_inputs["chosen_attention_mask"].append([])
        model_inputs["rejected_attention_mask"].append([])
    
    return model_inputs


def print_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
    valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
    valid_unlabeled_labels = list(filter(lambda x: x != IGNORE_INDEX, example["unlabeled_labels"]))
    print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
    print("chosen_inputs:\n{}".format(tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False)))
    print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
    print(f"chosen_labels:\n{tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
    print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
    print("rejected_inputs:\n{}".format(tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)))
    print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
    print(f"rejected_labels:\n{tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")
    print("unlabeled_input_ids:\n{}".format(example["unlabeled_input_ids"]))
    print("unlabeled_inputs:\n{}".format(tokenizer.decode(example["unlabeled_input_ids"], skip_special_tokens=False)))
    print("unlabeled_label_ids:\n{}".format(example["unlabeled_labels"]))
    print(f"unlabeled_labels:\n{tokenizer.decode(valid_unlabeled_labels, skip_special_tokens=False)}")