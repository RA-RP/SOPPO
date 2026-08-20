#!/usr/bin/env python3
"""Idempotently add Q1 frozen-teacher routing and source-equal loss to VERL."""

from __future__ import annotations

import argparse
from pathlib import Path


AGENT_MARKER = "# cycle09-q1: honor frozen per-sample teacher tensors"
ENGINE_MARKER = "# cycle09-q1: freeze source-specific global denominators before micro-batching"
LOSS_MARKER = "# cycle09-q1: strict source-equal token means"


def replace_once(path: Path, old: str, new: str, marker: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one patch anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def patch_agent_loop(path: Path) -> bool:
    old = '''        """Compute teacher logprobs for single sample."""
        if self.distillation_enabled and not validate:
'''
    new = f'''        """Compute teacher logprobs for single sample."""
        {AGENT_MARKER}
        frozen_ids = output.extra_fields.get("teacher_ids")
        frozen_logprobs = output.extra_fields.get("teacher_logprobs")
        if frozen_ids is not None or frozen_logprobs is not None:
            if frozen_ids is None or frozen_logprobs is None:
                raise RuntimeError("precomputed teacher_ids/logprobs must be supplied together")
            if frozen_ids.shape != frozen_logprobs.shape:
                raise RuntimeError(
                    f"precomputed teacher shape mismatch: {{frozen_ids.shape}} != {{frozen_logprobs.shape}}"
                )
            expected = len(prompt_ids) + len(response_ids)
            if frozen_ids.ndim != 2 or frozen_ids.shape[0] != expected:
                raise RuntimeError(
                    f"precomputed teacher rows={{frozen_ids.shape}} expected sequence length {{expected}}"
                )
            return
        if self.distillation_enabled and not validate:
'''
    return replace_once(path, old, new, AGENT_MARKER)


def patch_fsdp_engine(path: Path) -> bool:
    old = '''        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
'''
    new = f'''        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        {ENGINE_MARKER}
        if "support_source_id" in data.keys():
            source_ids = data["support_source_id"].reshape(-1).to(get_device_id(), dtype=torch.int64)
            if source_ids.numel() != data.shape[0] or not torch.all((source_ids == 0) | (source_ids == 1)):
                raise RuntimeError(f"invalid Q1 support_source_id tensor: {{source_ids}}")
            response_mask = data["response_mask"]
            if response_mask.is_nested:
                sample_tokens = torch.stack(
                    [row.to(torch.int64).sum() for row in response_mask.unbind()]
                ).to(get_device_id())
            else:
                sample_tokens = response_mask.to(get_device_id(), dtype=torch.int64).sum(dim=-1)
            source_samples = torch.stack([(source_ids == value).sum() for value in (0, 1)])
            source_tokens = torch.stack(
                [sample_tokens[source_ids == value].sum() for value in (0, 1)]
            )
            torch.distributed.all_reduce(
                source_samples, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
            )
            torch.distributed.all_reduce(
                source_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
            )
            if source_samples.tolist() != [8, 8]:
                raise RuntimeError(
                    f"Q1 requires exactly 8 external and 8 self samples per update; got {{source_samples.tolist()}}"
                )
            if torch.any(source_tokens <= 0):
                raise RuntimeError(f"Q1 source has no valid response tokens: {{source_tokens.tolist()}}")
            tu.assign_non_tensor(
                data,
                q1_external_sample_count=int(source_samples[0].item()),
                q1_self_sample_count=int(source_samples[1].item()),
                q1_external_token_count=int(source_tokens[0].item()),
                q1_self_token_count=int(source_tokens[1].item()),
            )

        micro_batches, indices = prepare_micro_batches(
'''
    return replace_once(path, old, new, ENGINE_MARKER)


def patch_losses(path: Path) -> list[str]:
    changed: list[str] = []
    text = path.read_text(encoding="utf-8")
    if "from verl.utils import tensordict_utils as tu" not in text:
        anchor = "from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty\n"
        if text.count(anchor) != 1:
            raise RuntimeError(f"{path}: missing tensordict import anchor")
        text = text.replace(anchor, anchor + "from verl.utils import tensordict_utils as tu\n", 1)
        changed.append("loss_import")

    decorator = (
        '@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], '
        "use_topk=True))  # type: ignore[arg-type]"
    )
    replacement = (
        '@register_distillation_loss(DistillationLossSettings('
        'names=["forward_kl_topk", "forward_kl_topk_source_equal"], use_topk=True'
        "))  # type: ignore[arg-type]"
    )
    if "forward_kl_topk_source_equal" not in text:
        if text.count(decorator) != 1:
            raise RuntimeError(f"{path}: missing forward-KL registration anchor")
        text = text.replace(decorator, replacement, 1)
        changed.append("loss_registration")

    if LOSS_MARKER not in text:
        helper_anchor = "\ndef distillation_loss(\n"
        if text.count(helper_anchor) != 1:
            raise RuntimeError(f"{path}: missing distillation_loss anchor")
        helper = f'''

def _q1_source_equal_loss(
    distillation_losses: torch.Tensor,
    response_mask: torch.Tensor,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Metric]]:
    """Aggregate .5*mean_token(self)+.5*mean_token(external) across dynamic micro-batches."""

    {LOSS_MARKER}
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)
    source_ids = data["support_source_id"].reshape(-1).to(
        device=distillation_losses.device, dtype=torch.int64
    )
    if source_ids.shape[0] != distillation_losses.shape[0]:
        raise RuntimeError("Q1 source labels are not aligned with the loss batch")
    dp_size = int(tu.get_non_tensor_data(data, "dp_size", 1))
    global_external_tokens = int(tu.get_non_tensor_data(data, "q1_external_token_count", 0))
    global_self_tokens = int(tu.get_non_tensor_data(data, "q1_self_token_count", 0))
    global_external_samples = int(tu.get_non_tensor_data(data, "q1_external_sample_count", 0))
    global_self_samples = int(tu.get_non_tensor_data(data, "q1_self_sample_count", 0))
    if (
        global_external_tokens <= 0
        or global_self_tokens <= 0
        or global_external_samples != 8
        or global_self_samples != 8
    ):
        raise RuntimeError(
            "Q1 global source denominators missing or invalid: "
            f"external=(samples={{global_external_samples}},tokens={{global_external_tokens}}), "
            f"self=(samples={{global_self_samples}},tokens={{global_self_tokens}})"
        )

    valid = response_mask.to(torch.bool)
    external_mask = valid & source_ids.eq(0).unsqueeze(-1)
    self_mask = valid & source_ids.eq(1).unsqueeze(-1)
    external_sum = distillation_losses.masked_select(external_mask).sum()
    self_sum = distillation_losses.masked_select(self_mask).sum()
    external_tokens = external_mask.sum()
    self_tokens = self_mask.sum()
    external_samples = source_ids.eq(0).sum()
    self_samples = source_ids.eq(1).sum()

    external_mean_contribution = external_sum / global_external_tokens * dp_size
    self_mean_contribution = self_sum / global_self_tokens * dp_size
    loss = 0.5 * external_mean_contribution + 0.5 * self_mean_contribution
    global_tokens = global_external_tokens + global_self_tokens
    global_samples = global_external_samples + global_self_samples
    metrics = {{
        "distillation/source_external_loss": Metric(
            AggregationType.SUM, external_mean_contribution
        ),
        "distillation/source_self_loss": Metric(
            AggregationType.SUM, self_mean_contribution
        ),
        "distillation/source_external_kl": Metric(
            AggregationType.SUM, external_mean_contribution
        ),
        "distillation/source_self_kl": Metric(
            AggregationType.SUM, self_mean_contribution
        ),
        "distillation/source_external_token_share": Metric(
            AggregationType.SUM, external_tokens / global_tokens * dp_size
        ),
        "distillation/source_self_token_share": Metric(
            AggregationType.SUM, self_tokens / global_tokens * dp_size
        ),
        "distillation/source_external_sample_share": Metric(
            AggregationType.SUM, external_samples / global_samples * dp_size
        ),
        "distillation/source_self_sample_share": Metric(
            AggregationType.SUM, self_samples / global_samples * dp_size
        ),
        "distillation/source_external_length": Metric(
            AggregationType.SUM, external_tokens / global_external_samples * dp_size
        ),
        "distillation/source_self_length": Metric(
            AggregationType.SUM, self_tokens / global_self_samples * dp_size
        ),
    }}
    return loss, metrics
'''
        text = text.replace(helper_anchor, helper + helper_anchor, 1)
        changed.append("loss_helper")

        old_branch = '''    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )
'''
        new_branch = '''    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if loss_config.loss_mode == "forward_kl_topk_source_equal":
            distillation_loss, source_metrics = _q1_source_equal_loss(
                distillation_losses=distillation_losses,
                response_mask=response_mask,
                data=data,
            )
            distillation_metrics.update(source_metrics)
        else:
            if response_mask.is_nested:
                response_mask = response_mask.to_padded_tensor(False)
            distillation_loss = agg_loss(
                loss_mat=distillation_losses,
                loss_mask=response_mask,
                loss_agg_mode=loss_agg_mode,
                **config.global_batch_info,
            )
'''
        if text.count(old_branch) != 1:
            raise RuntimeError(f"{path}: missing direct distillation aggregation anchor")
        text = text.replace(old_branch, new_branch, 1)
        changed.append("loss_branch")

    path.write_text(text, encoding="utf-8")
    return changed


def patch(verl_root: Path) -> dict[str, object]:
    files = {
        "agent_loop": verl_root / "verl/experimental/agent_loop/agent_loop.py",
        "fsdp_engine": verl_root / "verl/workers/engine/fsdp/transformer_impl.py",
        "losses": verl_root / "verl/trainer/distillation/losses.py",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    changed = {
        "agent_loop": patch_agent_loop(files["agent_loop"]),
        "fsdp_engine": patch_fsdp_engine(files["fsdp_engine"]),
        "losses": patch_losses(files["losses"]),
    }
    for path, marker in (
        (files["agent_loop"], AGENT_MARKER),
        (files["fsdp_engine"], ENGINE_MARKER),
        (files["losses"], LOSS_MARKER),
    ):
        if marker not in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"patch verification failed: {path}")
    return {"verl_root": str(verl_root), "changed": changed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-root", type=Path, default=Path("/root/autodl-tmp/verl"))
    args = parser.parse_args()
    print(patch(args.verl_root))


if __name__ == "__main__":
    main()
