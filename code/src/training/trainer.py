"""
Unified trainer for all methods: DPO-10%, Pseudo-target, DPO+PE, DPO-100%
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import yaml

from ..model import compute_dpo_loss, compute_pe_loss, compute_pseudo_target_loss
from ..model.model_utils import save_checkpoint
from .scheduler import LambdaScheduler, create_lambda_scheduler
from .diagnostics import (
    DiagnosticsTracker,
    compute_responsibility_quality,
    compute_prediction_distribution_stats,
    check_numerical_stability,
    log_training_step
)


class Trainer:
    """
    Unified trainer for preference learning methods.

    Supports:
    - DPO-10%: Only labeled data with DPO loss
    - Pseudo-target: Labeled + unlabeled with instance-level pseudo labels
    - DPO+PE: Labeled + unlabeled with population-level structure supervision
    - DPO-100%: All data labeled (oracle upper bound)
    """

    def __init__(
        self,
        method: str,
        policy_model: nn.Module,
        reference_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional = None,
        lambda_scheduler: Optional[LambdaScheduler] = None,
        config: Optional[Dict] = None,
        output_dir: str = "./output",
        device: str = "cuda"
    ):
        """
        Args:
            method: Training method ('dpo10', 'pseudo_target', 'dpo_pe', 'dpo100')
            policy_model: Policy model to train
            reference_model: Frozen reference model
            optimizer: Optimizer
            lr_scheduler: Optional learning rate scheduler
            lambda_scheduler: Optional lambda scheduler (for pseudo_target and dpo_pe)
            config: Training configuration
            output_dir: Output directory for checkpoints and logs
            device: Training device
        """
        self.method = method
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lambda_scheduler = lambda_scheduler
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.device = device

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "diagnostics").mkdir(exist_ok=True)

        # Save config
        with open(self.output_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f)

        # Initialize diagnostics
        self.diagnostics = DiagnosticsTracker()
        self.log_file = self.output_dir / "logs" / "metrics.jsonl"

        # Training state
        self.global_step = 0
        self.best_val_metric = float('-inf')

        print(f"Trainer initialized for method: {method}")
        print(f"Output directory: {output_dir}")

    def train_step(
        self,
        batch_labeled: Dict,
        batch_unlabeled: Optional[Dict] = None
    ) -> Dict:
        """
        Execute one training step.

        Args:
            batch_labeled: Batch from labeled dataset
            batch_unlabeled: Optional batch from unlabeled dataset

        Returns:
            info: Dictionary with loss and diagnostic information
        """
        self.policy_model.train()
        self.optimizer.zero_grad()

        # Extract labeled data
        input_ids_chosen = batch_labeled['input_ids_a'].to(self.device)
        attention_mask_chosen = batch_labeled['attention_mask_a'].to(self.device)
        input_ids_rejected = batch_labeled['input_ids_b'].to(self.device)
        attention_mask_rejected = batch_labeled['attention_mask_b'].to(self.device)
        labels = batch_labeled['labels'].to(self.device)

        # Swap chosen/rejected based on labels
        # label=1 means response_a is preferred, label=0 means response_b is preferred
        mask = (labels == 0)
        if mask.any():
            input_ids_chosen[mask], input_ids_rejected[mask] = \
                input_ids_rejected[mask].clone(), input_ids_chosen[mask].clone()
            attention_mask_chosen[mask], attention_mask_rejected[mask] = \
                attention_mask_rejected[mask].clone(), attention_mask_chosen[mask].clone()

        # Compute DPO loss on labeled data
        beta = self.config.get('beta', 0.1)
        loss_dpo, info_dpo = compute_dpo_loss(
            self.policy_model,
            self.reference_model,
            input_ids_chosen,
            attention_mask_chosen,
            input_ids_rejected,
            attention_mask_rejected,
            beta=beta
        )

        total_loss = loss_dpo
        info = {
            'loss_dpo': info_dpo['loss'],
            'dpo_accuracy': info_dpo['accuracy'],
            'dpo_probs_mean': info_dpo['probs_mean']
        }

        # Add auxiliary loss for methods using unlabeled data
        lambda_t = 0.0
        if self.method in ['pseudo_target', 'dpo_pe'] and batch_unlabeled is not None:
            # Get current lambda
            if self.lambda_scheduler is not None:
                lambda_t = self.lambda_scheduler.get_lambda(self.global_step)
            else:
                lambda_t = self.config.get('lambda_pe', 0.5)

            # Extract unlabeled data
            input_ids_a = batch_unlabeled['input_ids_a'].to(self.device)
            attention_mask_a = batch_unlabeled['attention_mask_a'].to(self.device)
            input_ids_b = batch_unlabeled['input_ids_b'].to(self.device)
            attention_mask_b = batch_unlabeled['attention_mask_b'].to(self.device)

            if self.method == 'pseudo_target':
                # Compute pseudo-target loss
                loss_aux, info_aux = compute_pseudo_target_loss(
                    self.policy_model,
                    self.reference_model,
                    input_ids_a,
                    attention_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    beta=beta,
                    threshold=self.config.get('pseudo_threshold', 0.5)
                )
                info.update({
                    'loss_pseudo': info_aux['loss'],
                    'pseudo_label_ratio': info_aux['pseudo_label_ratio'],
                    'unlabeled_p_i_mean': info_aux['p_i_mean']
                })

            elif self.method == 'dpo_pe':
                # Compute PE structure loss
                loss_aux, info_aux = compute_pe_loss(
                    self.policy_model,
                    self.reference_model,
                    input_ids_a,
                    attention_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    beta=beta,
                    epsilon=self.config.get('epsilon', 1e-8),
                    distance=self.config.get('pe_distance', 'l1'),
                    detach_denominator=self.config.get('detach_denominator', False)
                )
                info.update({
                    'loss_pe': info_aux['loss'],
                    'e_hat_plus_0': info_aux['e_hat_plus_0'],
                    'e_hat_plus_1': info_aux['e_hat_plus_1'],
                    'e_hat_minus_0': info_aux['e_hat_minus_0'],
                    'e_hat_minus_1': info_aux['e_hat_minus_1'],
                    'dist_plus': info_aux['dist_plus'],
                    'dist_minus': info_aux['dist_minus'],
                    'sum_p': info_aux['sum_p'],
                    'sum_1_minus_p': info_aux['sum_1_minus_p'],
                    'unlabeled_p_i_mean': info_aux['p_i_mean'],
                    'unlabeled_p_i_std': info_aux['p_i_std']
                })

            total_loss = loss_dpo + lambda_t * loss_aux
            info['loss_aux'] = info_aux['loss']
            info['lambda'] = lambda_t

        # Backward and optimize
        total_loss.backward()

        # Gradient clipping
        if self.config.get('max_grad_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config['max_grad_norm']
            )

        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Update diagnostics
        info['loss_total'] = total_loss.item()
        info['learning_rate'] = self.optimizer.param_groups[0]['lr']
        info['step'] = self.global_step

        self.diagnostics.update(info)

        # Log to file
        log_training_step(
            step=self.global_step,
            total_loss=total_loss.item(),
            dpo_loss=loss_dpo.item(),
            aux_loss=info.get('loss_aux', 0.0),
            lambda_t=lambda_t,
            learning_rate=info['learning_rate'],
            diagnostics=info,
            log_file=str(self.log_file)
        )

        self.global_step += 1

        return info

    def save_checkpoint_at_step(self, step: int, tag: str = ""):
        """Save checkpoint at current step."""
        if tag:
            checkpoint_dir = self.output_dir / "checkpoints" / f"step_{step:06d}_{tag}"
        else:
            checkpoint_dir = self.output_dir / "checkpoints" / f"step_{step:06d}"

        save_checkpoint(
            model=self.policy_model,
            tokenizer=None,  # Save separately if needed
            output_dir=str(checkpoint_dir),
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            step=step,
            config=self.config
        )

        print(f"Checkpoint saved at step {step}")

    def evaluate(self, dataloader, split: str = "val") -> Dict:
        """
        Evaluate model on validation or test set.

        Args:
            dataloader: DataLoader for evaluation
            split: Split name ('val' or 'test')

        Returns:
            metrics: Evaluation metrics
        """
        self.policy_model.eval()

        total_correct = 0
        total_samples = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
                input_ids_a = batch['input_ids_a'].to(self.device)
                attention_mask_a = batch['attention_mask_a'].to(self.device)
                input_ids_b = batch['input_ids_b'].to(self.device)
                attention_mask_b = batch['attention_mask_b'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Compute preference probabilities
                from ..model.dpo_loss import compute_sequence_logprob

                policy_a_outputs = self.policy_model(
                    input_ids=input_ids_a,
                    attention_mask=attention_mask_a,
                    return_dict=True
                )
                policy_b_outputs = self.policy_model(
                    input_ids=input_ids_b,
                    attention_mask=attention_mask_b,
                    return_dict=True
                )
                reference_a_outputs = self.reference_model(
                    input_ids=input_ids_a,
                    attention_mask=attention_mask_a,
                    return_dict=True
                )
                reference_b_outputs = self.reference_model(
                    input_ids=input_ids_b,
                    attention_mask=attention_mask_b,
                    return_dict=True
                )

                policy_a_logps = compute_sequence_logprob(
                    policy_a_outputs.logits, input_ids_a, attention_mask_a
                )
                policy_b_logps = compute_sequence_logprob(
                    policy_b_outputs.logits, input_ids_b, attention_mask_b
                )
                reference_a_logps = compute_sequence_logprob(
                    reference_a_outputs.logits, input_ids_a, attention_mask_a
                )
                reference_b_logps = compute_sequence_logprob(
                    reference_b_outputs.logits, input_ids_b, attention_mask_b
                )

                beta = self.config.get('beta', 0.1)
                reward_a = beta * (policy_a_logps - reference_a_logps)
                reward_b = beta * (policy_b_logps - reference_b_logps)
                delta = reward_a - reward_b
                probs = torch.sigmoid(delta)

                # Compute accuracy
                preds = (probs > 0.5).long()
                correct = (preds == labels).sum().item()

                total_correct += correct
                total_samples += len(labels)

                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        # Compute metrics
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # Compute Brier score
        all_probs = torch.tensor(all_probs)
        all_labels = torch.tensor(all_labels, dtype=torch.float)
        brier = ((all_probs - all_labels) ** 2).mean().item()

        metrics = {
            f'{split}_accuracy': accuracy,
            f'{split}_brier': brier,
            f'{split}_samples': total_samples
        }

        print(f"{split.upper()} - Accuracy: {accuracy:.4f}, Brier: {brier:.4f}")

        return metrics
