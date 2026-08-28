# Copyright 2024 OpenAccess AI Collective and the LlamaFactory team.
#
# This code is inspired by the OpenAccess AI Collective's axolotl library.
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence, List

import torch
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq

from ..extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER
from ..extras.packages import is_pillow_available
from ..extras import logging

logger = logging.get_logger(__name__)


if is_pillow_available():
    from PIL import Image


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from .template import Template


def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    """
    Expands the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    while handles packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    Example:
        Input: [[1, 1, 2, 2, 2, 0]]
        Output: [
            [
                [
                    [o, x, x, x, x, x],
                    [o, o, x, x, x, x],
                    [x, x, o, x, x, x],
                    [x, x, o, o, x, x],
                    [x, x, o, o, o, x],
                    [x, x, x, x, x, x],
                ]
            ]
        ]
        where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    bsz, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
    padding_mask = torch.where(expanded_mask != 0, 1, 0)
    attention_mask_4d = torch.eq(expanded_mask, expanded_mask.transpose(-1, -2)).int() * padding_mask
    attention_mask_4d *= torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long))
    attention_mask_4d = torch.where(attention_mask_4d != 0, torch.tensor(0, dtype=dtype), min_dtype)
    return attention_mask_4d


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """
    Data collator that supports VLMs.

    Features should contain input_ids, attention_mask, labels, and optionally contain images and videos.
    """

    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        batch_images, batch_videos, batch_imglens, batch_vidlens, batch_input_ids = [], [], [], [], []
        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_input_ids.append(feature["input_ids"])

        if (
            self.processor is not None and sum(batch_imglens) == 0 and sum(batch_vidlens) == 0
        ):  # avoid process hanging in zero3/fsdp case
            fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
            fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
            fake_messages = self.template.mm_plugin.process_messages(fake_messages, fake_images, [], self.processor)
            fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                fake_input_ids, None, fake_images, [], self.tokenizer, self.processor
            )
            if self.tokenizer.padding_side == "right":
                features[0]["input_ids"] = features[0]["input_ids"] + fake_input_ids
                features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_input_ids)
                features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_input_ids)
            else:
                features[0]["input_ids"] = fake_input_ids + features[0]["input_ids"]
                features[0]["attention_mask"] = [0] * len(fake_input_ids) + features[0]["attention_mask"]
                features[0]["labels"] = [IGNORE_INDEX] * len(fake_input_ids) + features[0]["labels"]

            batch_images = fake_images
            batch_imglens[0] = 1
            batch_input_ids[0] = features[0]["input_ids"]

        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            batch_images, batch_videos, batch_imglens, batch_vidlens, batch_input_ids, self.processor
        )
        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        features: Dict[str, "torch.Tensor"] = super().__call__(features)

        if self.model is not None and hasattr(self.model, "get_rope_index"):  # for qwen2vl mrope
            rope_index_kwargs = {
                "input_ids": features["input_ids"],
                "image_grid_thw": mm_inputs.get("image_grid_thw"),
                "video_grid_thw": mm_inputs.get("video_grid_thw"),
                "attention_mask": features["attention_mask"],
            }
            if "second_per_grid_ts" in mm_inputs:
                rope_index_kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")

            features["position_ids"], features["rope_deltas"] = self.model.get_rope_index(**rope_index_kwargs)

        if "cross_attention_mask" in mm_inputs:  # for mllama inputs when pad_to_multiple_of is enabled
            cross_attention_mask = mm_inputs.pop("cross_attention_mask")
            seq_len = features["input_ids"].size(1)
            orig_len = cross_attention_mask.size(1)
            mm_inputs["cross_attention_mask"] = F.pad(cross_attention_mask, (0, 0, 0, 0, 0, seq_len - orig_len))

        features.update(mm_inputs)
        if isinstance(features.get("pixel_values"), list):  # for pixtral inputs
            features = features.data  # use default_collate() instead of BatchEncoding.to()

        if "image_bound" in features:  # for minicpmv inputs
            bsz, seq_length = features["input_ids"].shape
            features["position_ids"] = torch.arange(seq_length).long().repeat(bsz, 1)
            return {"data": features, "input_ids": features["input_ids"], "labels": features["labels"]}

        return features


@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    """
    Data collator for 4d attention mask.
    """

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)

        for key, value in features.items():  # cast data dtype for paligemma
            if torch.is_tensor(value) and torch.is_floating_point(value):
                features[key] = value.to(self.compute_dtype)

        return features


@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    """
    Data collator for pairwise data with SSPO support.
    """
    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        """
        Prepare batch for SSPO: Organize chosen, rejected, and unlabeled data in appropriate ratios.
        """
        
        # Separate chosen/rejected/unlabeled data from original dataset
        chosen_features = []
        rejected_features = []
        unlabeled_features = []
    
        for feature in features:
            # Check data type using data_types field
            data_type = feature.get("data_types", "")
            
            if data_type == "labeled" or data_type == "both":
                chosen_feature = {
                    "input_ids": feature["chosen_input_ids"],
                    "attention_mask": feature["chosen_attention_mask"],
                    "labels": feature["chosen_labels"],
                    "images": feature.get("images", []),
                    "videos": feature.get("videos", [])
                }
                chosen_features.append(chosen_feature)
                
                rejected_feature = {
                    "input_ids": feature["rejected_input_ids"],
                    "attention_mask": feature["rejected_attention_mask"],
                    "labels": feature["rejected_labels"],
                    "images": feature.get("images", []),
                    "videos": feature.get("videos", [])
                }
                rejected_features.append(rejected_feature)
            
            if data_type == "unlabeled" or data_type == "both":
                unlabeled_feature = {
                    "input_ids": feature["unlabeled_input_ids"],
                    "attention_mask": feature["unlabeled_attention_mask"],
                    "labels": feature["unlabeled_labels"],
                    "images": feature.get("images", []),
                    "videos": feature.get("videos", [])
                }
                unlabeled_features.append(unlabeled_feature)
        
        # Data validation
        if len(chosen_features) != len(rejected_features):
            # chosen and rejected should always be paired, but use minimum if different
            min_len = min(len(chosen_features), len(rejected_features))
            chosen_features = chosen_features[:min_len]
            rejected_features = rejected_features[:min_len]
            logger.warning(f"Chosen and rejected features mismatch. Using minimum: {min_len}")
        
        # Combine data (order: chosen, rejected, unlabeled)
        concatenated_features = []
        concatenated_features.extend(chosen_features)
        concatenated_features.extend(rejected_features)
        concatenated_features.extend(unlabeled_features)
        
        # Call base padding logic
        batch = super().__call__(concatenated_features)
        
        # Store metadata as separate attributes
        num_chosen = len(chosen_features)
        num_rejected = len(rejected_features)
        num_unlabeled = len(unlabeled_features)
        
        # Add each quantity as individual tensors
        batch["num_chosen"] = torch.tensor([num_chosen], dtype=torch.int32)
        batch["num_rejected"] = torch.tensor([num_rejected], dtype=torch.int32)
        batch["num_unlabeled"] = torch.tensor([num_unlabeled], dtype=torch.int32)

        return batch


@dataclass
class KTODataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    """
    Data collator for KTO data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        target_features = []
        kl_features = []
        kto_tags = []
        for feature in features:
            target_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"],
                "images": feature["images"],
                "videos": feature["videos"],
            }
            kl_feature = {
                "input_ids": feature["kl_input_ids"],
                "attention_mask": feature["kl_attention_mask"],
                "labels": feature["kl_labels"],
                "images": feature["images"],
                "videos": feature["videos"],
            }
            target_features.append(target_feature)
            kl_features.append(kl_feature)
            kto_tags.append(feature["kto_tags"])

        batch = super().__call__(target_features)
        kl_batch = super().__call__(kl_features)
        batch["kl_input_ids"] = kl_batch["input_ids"]
        batch["kl_attention_mask"] = kl_batch["attention_mask"]
        batch["kl_labels"] = kl_batch["labels"]
        if "token_type_ids" in kl_batch:
            batch["kl_token_type_ids"] = kl_batch["token_type_ids"]

        batch["kto_tags"] = torch.tensor(kto_tags)
        return batch
