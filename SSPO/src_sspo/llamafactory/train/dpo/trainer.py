"""
Train SSPO.

This code is created based on the official code of LLaMA-Factory and the alignment handbook.
(https://github.com/hiyouga/LLaMA-Factory)
(https://github.com/huggingface/alignment-handbook)

(Zheng, Y., Zhang, R., Zhang, J., Ye, Y., & Luo, Z. (2024). 
Llamafactory: Unified efficient fine-tuning of 100+ language models. 
arXiv preprint arXiv:2403.13372.)

"""

# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
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


import warnings
import math
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ...extras import logging
from ...data.sampler import StaticPETwoStreamSampler
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps, nested_detach

logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        #! EDIT : remove processing_class for SSPO
        # if is_transformers_version_greater_than("4.46"):
        #     kwargs["processing_class"] = kwargs.pop("tokenizer")

        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        #! EDIT : SSPO hyperparams
        self.sspo_gamma_min = finetuning_args.sspo_gamma_min
        self.sspo_gamma_0 = finetuning_args.sspo_gamma_0
        self.sspo_gamma_decay = finetuning_args.sspo_gamma_decay
        self.sspo_prior = finetuning_args.sspo_prior

        # StaticPE hyperparameters
        self.staticpe_lambda = finetuning_args.staticpe_lambda
        self.staticpe_epsilon = finetuning_args.staticpe_epsilon
        self.staticpe_min_labeled_per_batch = finetuning_args.staticpe_min_labeled_per_batch
        self.staticpe_min_unlabeled_per_batch = finetuning_args.staticpe_min_unlabeled_per_batch
        
        #! EDIT : Add moving average params for reward normalization
        self.reward_norm_momentum = 0.95  # moving average momentum value
        self.running_mean = None  # moving average value
        self.running_var = None  # moving variance value
        self.reward_clip_range = 5.0  # reward clipping range
        
        Trainer.__init__(self, model=model, **kwargs)
        # super().__init__(model=model, ref_model=ref_model, **kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.loss_type == "staticpe":
            return StaticPETwoStreamSampler(
                data_types=self.train_dataset["data_types"],
                batch_size=self.args.train_batch_size,
                seed=self.args.seed,
                min_labeled_per_batch=self.staticpe_min_labeled_per_batch,
                min_unlabeled_per_batch=self.staticpe_min_unlabeled_per_batch,
            )

        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def _get_eval_sampler(self, eval_dataset) -> Optional["torch.utils.data.Sampler"]:
        if self.loss_type == "staticpe":
            return StaticPETwoStreamSampler(
                data_types=eval_dataset["data_types"],
                batch_size=self.args.eval_batch_size,
                seed=self.args.seed,
                min_labeled_per_batch=self.staticpe_min_labeled_per_batch,
                min_unlabeled_per_batch=self.staticpe_min_unlabeled_per_batch,
            )

        return super()._get_eval_sampler(eval_dataset)

    @override
    def get_batch_samples(self, epoch_iterator, num_batches):
        r"""
        Replaces the method of KTO Trainer with the one of the standard Trainer.
        """
        return Trainer.get_batch_samples(self, epoch_iterator, num_batches)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    def _gather_variable_tensor(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Differentiably gather a variable-length 1-D tensor from all ranks."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return tensor

        world_size = torch.distributed.get_world_size()
        local_size = torch.tensor([tensor.numel()], device=tensor.device, dtype=torch.long)
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_sizes, local_size)
        sizes = [int(size.item()) for size in gathered_sizes]
        if sum(sizes) == 0:
            return tensor

        max_size = max(sizes)
        padded = F.pad(tensor.reshape(-1), (0, max_size - tensor.numel()))
        from torch.distributed.nn.functional import all_gather

        gathered = all_gather(padded)
        return torch.cat([rank_tensor[:rank_size] for rank_tensor, rank_size in zip(gathered, sizes)], dim=0)

    def staticpe_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        policy_unlabeled_logps: "torch.Tensor",
        policy_unlabeled_b_logps: "torch.Tensor",
        reference_chosen_logps: "torch.Tensor",
        reference_rejected_logps: "torch.Tensor",
        reference_unlabeled_logps: "torch.Tensor",
        reference_unlabeled_b_logps: "torch.Tensor",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        """Compute labeled DPO plus StaticPE over the global candidate population."""
        device = policy_unlabeled_logps.device
        zero = (
            policy_chosen_logps.sum()
            + policy_rejected_logps.sum()
            + policy_unlabeled_logps.sum()
            + policy_unlabeled_b_logps.sum()
        ) * 0.0

        dpo_logits = self.beta * (
            (policy_chosen_logps - policy_rejected_logps)
            - (reference_chosen_logps - reference_rejected_logps)
        )
        global_dpo_logits = self._gather_variable_tensor(dpo_logits)
        dpo_loss = -F.logsigmoid(global_dpo_logits).mean() if global_dpo_logits.numel() > 0 else zero

        reward_a = self.beta * (policy_unlabeled_logps - reference_unlabeled_logps)
        reward_b = self.beta * (policy_unlabeled_b_logps - reference_unlabeled_b_logps)
        probabilities = torch.sigmoid(reward_a - reward_b)
        global_probabilities = self._gather_variable_tensor(probabilities)

        if global_probabilities.numel() > 0:
            q = torch.stack([global_probabilities, 1.0 - global_probabilities], dim=-1)
            weights_1 = global_probabilities
            weights_2 = 1.0 - global_probabilities
            c_1 = (weights_1.unsqueeze(-1) * q).sum(dim=0) / (weights_1.sum() + self.staticpe_epsilon)
            c_2 = (weights_2.unsqueeze(-1) * q).sum(dim=0) / (weights_2.sum() + self.staticpe_epsilon)
            target_1 = torch.tensor([1.0, 0.0], device=device, dtype=q.dtype)
            target_2 = torch.tensor([0.0, 1.0], device=device, dtype=q.dtype)
            pe_loss = 0.5 * (torch.abs(c_1 - target_1).sum() + torch.abs(c_2 - target_2).sum())
        else:
            c_1 = torch.zeros(2, device=device)
            c_2 = torch.zeros(2, device=device)
            pe_loss = zero

        # Keep the approved component weights fixed even when a physical
        # micro-batch contains no labeled row. Gradient accumulation averages
        # these distributed micro-batch objectives at the optimizer step.
        normalizer = 1.0 + self.staticpe_lambda
        loss = (dpo_loss + self.staticpe_lambda * pe_loss) / normalizer

        statistics = {
            "dpo_loss": dpo_loss,
            "pe_loss": pe_loss,
            "weighted_dpo_loss": dpo_loss / normalizer,
            "weighted_pe_loss": self.staticpe_lambda * pe_loss / normalizer,
            "probability_mean": global_probabilities.mean() if global_probabilities.numel() > 0 else zero,
            "c1_first": c_1[0],
            "c1_second": c_1[1],
            "c2_first": c_2[0],
            "c2_second": c_2[1],
            "reward_a_mean": reward_a.mean() if reward_a.numel() > 0 else zero,
            "reward_b_mean": reward_b.mean() if reward_b.numel() > 0 else zero,
        }
        return loss, statistics
    
    #! EDIT : add unlabeled data for SSPO training
    def sspo_loss(
            self, 
            policy_chosen_logps: "torch.Tensor", 
            policy_rejected_logps: "torch.Tensor", 
            policy_unlabeled_logps: "torch.Tensor",
            reference_chosen_logps: Optional["torch.Tensor"] = None,
            reference_rejected_logps: Optional["torch.Tensor"] = None,
            reference_unlabeled_logps: Optional["torch.Tensor"] = None
        ) -> "torch.Tensor":

        device = self.accelerator.device
        t = self.state.global_step
        current_gamma = max(self.sspo_gamma_min, self.sspo_gamma_0 * math.exp(-self.sspo_gamma_decay * t))
        
        def normalize_rewards(logps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            """Apply Z-score normalization with moving average and clipping to log probabilities"""
            if logps.numel() == 0:
                return logps
            
            batch_mean = logps.mean().detach()
            batch_var = logps.var(unbiased=False).detach() + eps
            
            if self.running_mean is None or self.running_var is None:
                self.running_mean = batch_mean.clone()
                self.running_var = batch_var.clone()
            else:
                a = self.reward_norm_momentum
                self.running_mean = a * self.running_mean + (1 - a) * batch_mean
                self.running_var = a * self.running_var + (1 - a) * batch_var
            
            std = torch.sqrt(self.running_var)
            normalized_logps = (logps - self.running_mean) / std
            normalized_logps = torch.clamp(normalized_logps, -self.reward_clip_range, self.reward_clip_range)
            
            # logger.info(f"Reward stats - batch_mean: {batch_mean:.4f}, running_mean: {self.running_mean:.4f}, std: {std:.4f}")
            
            return normalized_logps
        
        normalized_policy_chosen_logps = normalize_rewards(policy_chosen_logps)
        normalized_policy_rejected_logps = normalize_rewards(policy_rejected_logps)
        normalized_policy_unlabeled_logps = normalize_rewards(policy_unlabeled_logps)
        
        if reference_chosen_logps is not None and reference_rejected_logps is not None:
            normalized_reference_chosen_logps = normalize_rewards(reference_chosen_logps)
            normalized_reference_rejected_logps = normalize_rewards(reference_rejected_logps)
            
            if reference_unlabeled_logps is not None:
                normalized_reference_unlabeled_logps = normalize_rewards(reference_unlabeled_logps)
            else:
                normalized_reference_unlabeled_logps = None
        else:
            normalized_reference_chosen_logps = None
            normalized_reference_rejected_logps = None
            normalized_reference_unlabeled_logps = None
        
        if normalized_reference_chosen_logps is not None and normalized_reference_rejected_logps is not None:
            policy_chosen_logps_adjusted = normalized_policy_chosen_logps - normalized_reference_chosen_logps
            policy_rejected_logps_adjusted = normalized_policy_rejected_logps - normalized_reference_rejected_logps
            
            logits = self.beta * (policy_chosen_logps_adjusted - policy_rejected_logps_adjusted)
            pn_loss = -F.logsigmoid(logits)
            
            if normalized_reference_unlabeled_logps is not None:
                policy_unlabeled_logps_adjusted = normalized_policy_unlabeled_logps - normalized_reference_unlabeled_logps
                
                if normalized_policy_chosen_logps.numel() > 0:
                    threshold = torch.min(policy_chosen_logps_adjusted)
                else:
                    threshold = policy_unlabeled_logps_adjusted.mean()
                
                diff = self.beta * (policy_unlabeled_logps_adjusted - threshold) ##if batch without labeled sample let loss=0.5
                log_sigmoid_diff = F.logsigmoid(diff)
                
                u_loss_greater = self.sspo_prior * (-log_sigmoid_diff)
                u_loss_less_equal = (1 - self.sspo_prior) * (-F.logsigmoid(-diff))
                
                u_losses_tensor = torch.where(diff > 0, u_loss_greater, u_loss_less_equal)
                final_u_loss = u_losses_tensor.mean()
            else:
                final_u_loss = torch.tensor(0.0, device=device, requires_grad=True)
                u_losses_tensor = torch.tensor([], device=device)
        else:
            pn_loss = self.simpo_loss(policy_chosen_logps, policy_rejected_logps) if policy_chosen_logps.numel() > 0 or policy_rejected_logps.numel() > 0 else torch.tensor(0.0, device=device, requires_grad=True)
            threshold = torch.min(normalized_policy_chosen_logps) if normalized_policy_chosen_logps.numel() > 0 else normalized_policy_unlabeled_logps.mean()
            
            diff = self.beta * (normalized_policy_unlabeled_logps - threshold)
            log_sigmoid_diff = F.logsigmoid(diff)
            
            u_loss_greater = self.sspo_prior * (-log_sigmoid_diff)
            u_loss_less_equal = (1 - self.sspo_prior) * (-F.logsigmoid(-diff))
            
            u_losses_tensor = torch.where(diff > 0, u_loss_greater, u_loss_less_equal)
            final_u_loss = u_losses_tensor.mean()
        
        pn_loss_mean = pn_loss.mean() if pn_loss.numel() > 0 else torch.tensor(0.0, device=device, requires_grad=True)
        sspo_loss = current_gamma * pn_loss_mean + (1-current_gamma) * final_u_loss
        
        if reference_chosen_logps is not None and reference_rejected_logps is not None:
            orpo_loss = -F.logsigmoid(self.beta * (policy_chosen_logps_adjusted - policy_rejected_logps_adjusted))
        else:
            log_odds = (policy_chosen_logps - policy_rejected_logps) - (
                torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps))
            ) if policy_chosen_logps.numel() > 0 or policy_rejected_logps.numel() > 0 else torch.tensor(0.0, device=device, requires_grad=False)
            sft_loss = -policy_chosen_logps if policy_chosen_logps.numel() > 0 else torch.tensor(0.0, device=device, requires_grad=False)
            odds_ratio_loss = -F.logsigmoid(log_odds) if log_odds.numel() > 0 else torch.tensor(0.0, device=device, requires_grad=False)
            orpo_loss = sft_loss + self.beta * odds_ratio_loss
        
        # logger.info(f"Loss in GPU {torch.cuda.current_device()} = pn_loss: {pn_loss_mean:.4f}, final_u_loss: {final_u_loss:.4f}, sspo_loss: {sspo_loss:.4f}, current_gamma: {current_gamma:.4f}")
        return sspo_loss, pn_loss, orpo_loss, u_losses_tensor

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        policy_unlabeled_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
        reference_unlabeled_logps: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Computes loss for preference learning.
        """
        device = self.accelerator.device
        
        if self.loss_type == "sspo":
            losses, simpo_losses, orpo_losses, unlabeled_losses = self.sspo_loss(
                policy_chosen_logps, 
                policy_rejected_logps, 
                policy_unlabeled_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                reference_unlabeled_logps
            )
            chosen_rewards = self.beta * policy_chosen_logps.to(device)
            rejected_rewards = self.beta * policy_rejected_logps.to(device)
            unlabeled_rewards = self.beta * policy_unlabeled_logps.to(device)
            
            return losses, simpo_losses, orpo_losses, unlabeled_losses, chosen_rewards, rejected_rewards, unlabeled_rewards
        elif not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                return losses, self.beta * chosen_rewards, self.beta * rejected_rewards
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            
            return losses, chosen_rewards, rejected_rewards


    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", ...]:
        """
        Data distribution logic for SSPO training
        """
            
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)
        
        num_chosen = batch['num_chosen'].item() if 'num_chosen' in batch else 0
        num_rejected = batch['num_rejected'].item() if 'num_rejected' in batch else 0
        num_unlabeled = batch['num_unlabeled'].item() if 'num_unlabeled' in batch else 0
        num_unlabeled_b = batch['num_unlabeled_b'].item() if 'num_unlabeled_b' in batch else 0

        count_keys = {"num_chosen", "num_rejected", "num_unlabeled", "num_unlabeled_b"}
        model_batch = {key: value for key, value in batch.items() if key not in count_keys}
        all_logits: "torch.Tensor" = model(**model_batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        
        if self.loss_type in ["ipo", "orpo", "simpo", "sspo"]:
            all_logps = all_logps / valid_length
        
        device = all_logps.device
        split_sizes = [num_chosen, num_rejected, num_unlabeled]
        if self.loss_type == "staticpe":
            if num_unlabeled != num_unlabeled_b:
                raise ValueError(
                    "StaticPE requires the same number of unlabeled-A and unlabeled-B candidates, "
                    f"but got {num_unlabeled} and {num_unlabeled_b}."
                )
            split_sizes.append(num_unlabeled_b)
        
        logps_list = torch.split(all_logps, split_sizes)
        logits_list = torch.split(all_logits, split_sizes)
        length_list = torch.split(valid_length, split_sizes)
        
        chosen_logps = logps_list[0].to(device) if num_chosen > 0 else torch.tensor([], device=device)
        rejected_logps = logps_list[1].to(device) if num_rejected > 0 else torch.tensor([], device=device)
        unlabeled_logps = logps_list[2].to(device) if num_unlabeled > 0 else torch.tensor([], device=device)
        unlabeled_b_logps = (
            logps_list[3].to(device) if self.loss_type == "staticpe" and num_unlabeled_b > 0
            else torch.tensor([], device=device)
        )

        chosen_logits = logits_list[0].to(device) if num_chosen > 0 else torch.tensor([], device=device)
        rejected_logits = logits_list[1].to(device) if num_rejected > 0 else torch.tensor([], device=device)
        unlabeled_logits = logits_list[2].to(device) if num_unlabeled > 0 else torch.tensor([], device=device)
        unlabeled_b_logits = (
            logits_list[3].to(device) if self.loss_type == "staticpe" and num_unlabeled_b > 0
            else torch.tensor([], device=device)
        )

        chosen_length = length_list[0].to(device) if num_chosen > 0 else torch.tensor([], device=device)

        if self.loss_type == "staticpe":
            return (
                chosen_logps,
                rejected_logps,
                unlabeled_logps,
                unlabeled_b_logps,
                chosen_logits,
                rejected_logits,
                unlabeled_logits,
                unlabeled_b_logits,
            )
        elif self.loss_type == "sspo":
            return (
                chosen_logps, rejected_logps, unlabeled_logps,
                chosen_logits, rejected_logits, unlabeled_logits,
                chosen_logps / chosen_length if num_chosen > 0 else torch.tensor([], device=device),
                rejected_logps / length_list[1].to(device) if num_rejected > 0 else torch.tensor([], device=device),
                unlabeled_logps / length_list[2].to(device) if num_unlabeled > 0 else torch.tensor([], device=device)
            )
        elif self.loss_type in ["ipo", "orpo", "simpo"]:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps
        else:
            return (
                chosen_logps, rejected_logps, chosen_logits, rejected_logits,
                chosen_logps / chosen_length if num_chosen > 0 else torch.tensor([], device=device)
            )

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[
        Optional["torch.Tensor"],
        Optional["torch.Tensor"],
        Optional["torch.Tensor"],
        Optional["torch.Tensor"],
    ]:
        """
        Computes log probabilities of the reference model.
        """
        if not self.finetuning_args.use_ref_model:
            return None, None, None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            if self.loss_type == "staticpe":
                reference_outputs = self.concatenated_forward(ref_model, batch)
                return tuple(output.to(model.device) for output in reference_outputs[:4])
            elif self.loss_type == "sspo":
                reference_outputs = self.concatenated_forward(ref_model, batch)
                if len(reference_outputs) >= 9:
                    reference_chosen_logps, reference_rejected_logps, reference_unlabeled_logps = reference_outputs[:3]
                    return (reference_chosen_logps.to(model.device), 
                            reference_rejected_logps.to(model.device), 
                            reference_unlabeled_logps.to(model.device),
                            None)
                else:
                    reference_chosen_logps, reference_rejected_logps = reference_outputs[:2]
                    return reference_chosen_logps.to(model.device), reference_rejected_logps.to(model.device), None, None
            else:
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(ref_model, batch)[:2]
                return reference_chosen_logps.to(model.device), reference_rejected_logps.to(model.device), None, None

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        """
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""
        device = next(model.parameters()).device

        if self.loss_type == "staticpe":
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_unlabeled_logps,
                policy_unlabeled_b_logps,
                policy_chosen_logits,
                policy_rejected_logits,
                policy_unlabeled_logits,
                policy_unlabeled_b_logits,
            ) = self.concatenated_forward(model, batch)

            (
                reference_chosen_logps,
                reference_rejected_logps,
                reference_unlabeled_logps,
                reference_unlabeled_b_logps,
            ) = self.compute_reference_log_probs(model, batch)
            if any(
                logps is None
                for logps in [
                    reference_chosen_logps,
                    reference_rejected_logps,
                    reference_unlabeled_logps,
                    reference_unlabeled_b_logps,
                ]
            ):
                raise RuntimeError("StaticPE requires a frozen reference model for both labeled and unlabeled rows.")

            losses, statistics = self.staticpe_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                policy_unlabeled_logps,
                policy_unlabeled_b_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                reference_unlabeled_logps,
                reference_unlabeled_b_logps,
            )

            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
            metrics[f"{prefix}staticpe/loss_dpo"] = statistics["dpo_loss"].item()
            metrics[f"{prefix}staticpe/loss_pe"] = statistics["pe_loss"].item()
            metrics[f"{prefix}staticpe/loss_dpo_weighted"] = statistics["weighted_dpo_loss"].item()
            metrics[f"{prefix}staticpe/loss_pe_weighted"] = statistics["weighted_pe_loss"].item()
            metrics[f"{prefix}staticpe/loss_total"] = losses.item()
            metrics[f"{prefix}staticpe/p_mean"] = statistics["probability_mean"].item()
            metrics[f"{prefix}staticpe/c1_first"] = statistics["c1_first"].item()
            metrics[f"{prefix}staticpe/c1_second"] = statistics["c1_second"].item()
            metrics[f"{prefix}staticpe/c2_first"] = statistics["c2_first"].item()
            metrics[f"{prefix}staticpe/c2_second"] = statistics["c2_second"].item()
            metrics[f"{prefix}staticpe/reward_a"] = statistics["reward_a_mean"].item()
            metrics[f"{prefix}staticpe/reward_b"] = statistics["reward_b_mean"].item()
            metrics[f"{prefix}staticpe/lambda"] = self.staticpe_lambda
            metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item() if chosen_rewards.numel() > 0 else 0.0
            metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item() if rejected_rewards.numel() > 0 else 0.0
            metrics[f"{prefix}rewards/accuracies"] = (
                (chosen_rewards > rejected_rewards).float().mean().item()
                if chosen_rewards.numel() > 0 else 0.5
            )
            metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item() if policy_chosen_logps.numel() > 0 else 0.0
            metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item() if policy_rejected_logps.numel() > 0 else 0.0
            metrics[f"{prefix}logps/unlabeled_a"] = policy_unlabeled_logps.mean().item() if policy_unlabeled_logps.numel() > 0 else 0.0
            metrics[f"{prefix}logps/unlabeled_b"] = policy_unlabeled_b_logps.mean().item() if policy_unlabeled_b_logps.numel() > 0 else 0.0
            metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item() if policy_chosen_logits.numel() > 0 else 0.0
            metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item() if policy_rejected_logits.numel() > 0 else 0.0
            metrics[f"{prefix}logits/unlabeled_a"] = policy_unlabeled_logits.mean().item() if policy_unlabeled_logits.numel() > 0 else 0.0
            metrics[f"{prefix}logits/unlabeled_b"] = policy_unlabeled_b_logits.mean().item() if policy_unlabeled_b_logits.numel() > 0 else 0.0
            return losses, metrics

        elif self.loss_type == "sspo":
            has_data = (
                batch['num_chosen'].item() > 0 or batch['num_rejected'].item() > 0
            )
            
            # if not has_data:
            #     logger.warning(f"Empty labeled data detected in GPU {torch.cuda.current_device()}, skipping...")
                
            (policy_chosen_logps, policy_rejected_logps, policy_unlabeled_logps,
            policy_chosen_logits, policy_rejected_logits, policy_unlabeled_logits,
            policy_chosen_logps_avg, policy_rejected_logps_avg, policy_unlabeled_logps_avg,
            ) = self.concatenated_forward(model, batch)

            # logger.info(f"Using {'DPO' if self.finetuning_args.use_ref_model else 'SimPO'} based SSPO in GPU {torch.cuda.current_device()}")
            
            reference_chosen_logps, reference_rejected_logps, reference_unlabeled_logps, _ = self.compute_reference_log_probs(model, batch)

            losses, simpo_losses, orpo_losses, unlabeled_losses, chosen_rewards, rejected_rewards, unlabeled_rewards = self.compute_preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                policy_unlabeled_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                reference_unlabeled_logps
            )

            sft_loss = -policy_unlabeled_logps_avg
            if self.ftx_gamma > 1e-6:
                losses = losses + self.ftx_gamma * sft_loss

            metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item() if chosen_rewards.numel() > 0 else 0.0
            metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item() if rejected_rewards.numel() > 0 else 0.0
            metrics[f"{prefix}rewards/unlabeled"] = unlabeled_rewards.mean().item() if unlabeled_rewards.numel() > 0 else 0.0
            metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item() if chosen_rewards.numel() > 0 and rejected_rewards.numel() > 0 else 0.5
            metrics[f"{prefix}rewards/avg_margins"] = (chosen_rewards - rejected_rewards).mean().item() if chosen_rewards.numel() > 0 and rejected_rewards.numel() > 0 else 0.0
            metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item() if policy_chosen_logps.numel() > 0 else 0.0
            metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item() if policy_rejected_logps.numel() > 0 else 0.0      
            metrics[f"{prefix}logps/unlabeled"] = policy_unlabeled_logps.mean().item() if policy_unlabeled_logps.numel() > 0 else 0.0
            metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item() if policy_chosen_logits.numel() > 0 else 0.0
            metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item() if policy_rejected_logits.numel() > 0 else 0.0   
            metrics[f"{prefix}logits/unlabeled"] = policy_unlabeled_logits.mean().item() if policy_unlabeled_logits.numel() > 0 else 0.0
            metrics[f"{prefix}orpo_loss"] = orpo_losses.mean().item() if chosen_rewards.numel() > 0 and rejected_rewards.numel() > 0 else 0.0
            metrics[f"{prefix}simpo_loss"] = simpo_losses.mean().item() if chosen_rewards.numel() > 0 and rejected_rewards.numel() > 0 else 0.0
            metrics[f"{prefix}unlabeled_loss"] = unlabeled_losses.mean().item() if unlabeled_rewards.numel() > 0 else 0.0
            # Stable, base-agnostic names. Keep the legacy aliases above so
            # existing SSPO dashboards remain readable.
            metrics[f"{prefix}sspo/loss_labeled"] = simpo_losses.mean().item() if simpo_losses.numel() > 0 else 0.0
            metrics[f"{prefix}sspo/loss_unlabeled"] = unlabeled_losses.mean().item() if unlabeled_losses.numel() > 0 else 0.0
            metrics[f"{prefix}sspo/loss_total"] = losses.mean().item()

            t = self.state.global_step
            current_gamma = max(self.sspo_gamma_min, self.sspo_gamma_0 * math.exp(-self.sspo_gamma_decay * t))
            
            metrics[f"{prefix}sspo/gamma"] = current_gamma

            if self.finetuning_args.use_ref_model:
                metrics[f"{prefix}dpo/policy_chosen_logps"] = policy_chosen_logps.mean().item() if policy_chosen_logps.numel() > 0 else 0.0
                metrics[f"{prefix}dpo/policy_rejected_logps"] = policy_rejected_logps.mean().item() if policy_rejected_logps.numel() > 0 else 0.0
                metrics[f"{prefix}dpo/reference_chosen_logps"] = reference_chosen_logps.mean().item() if reference_chosen_logps is not None and reference_chosen_logps.numel() > 0 else 0.0
                metrics[f"{prefix}dpo/reference_rejected_logps"] = reference_rejected_logps.mean().item() if reference_rejected_logps is not None and reference_rejected_logps.numel() > 0 else 0.0

            return losses, metrics

        else:
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
                policy_chosen_logps_avg,
            ) = self.concatenated_forward(model, batch)

            reference_chosen_logps, reference_rejected_logps, _, _ = self.compute_reference_log_probs(model, batch)
            losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                torch.tensor([], device=device),
                reference_chosen_logps,
                reference_rejected_logps,
            )
            sft_loss = -policy_chosen_logps_avg
            if self.ftx_gamma > 1e-6:
                losses += self.ftx_gamma * sft_loss

            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
            metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
            metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
            metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
            metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
            metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
            metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
            metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
            metrics[f"{prefix}logits/unlabeled"] = policy_unlabeled_logits.mean().item()
            if self.loss_type == "orpo":
                metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
                metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()

            return losses, metrics

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        r"""
        Subclass and override to accept extra kwargs.
        """
        return super().compute_loss(model, inputs, return_outputs)

    @override
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        r"""
        Log `logs` on the various objects watching training, including stored metrics.
        """
        train_eval = "train" if "loss" in logs else "eval"
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 10:
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        for key, metric in zip(key_list, metric_list):
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs, *args, **kwargs)
