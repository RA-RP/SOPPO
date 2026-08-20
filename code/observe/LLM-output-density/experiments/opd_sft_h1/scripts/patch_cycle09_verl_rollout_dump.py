#!/usr/bin/env python3
"""Idempotently add loss-neutral Cycle09 audit fields to verl rollout dumps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MARKER = "CYCLE09_BLOCK3_ROLLOUT_AUDIT_BEGIN"
ANCHOR = '''            if "request_id" in batch.non_tensor_batch:\n'''
LEGACY_ANCHOR = '''        if "request_id" in batch.non_tensor_batch:\n'''
INSERT = "\n".join(
    (
        "        # CYCLE09_BLOCK3_ROLLOUT_AUDIT_BEGIN: audit-only, no training mutation.",
        '        responses = batch.batch["responses"]',
        '        response_mask = batch.batch.get("response_mask", None)',
        "        if response_mask is None:",
        "            response_length = responses.shape[-1]",
        '            response_mask = batch.batch["attention_mask"][:, -response_length:]',
        "        response_ids_cpu = responses.detach().cpu().tolist()",
        "        response_mask_cpu = response_mask.detach().cpu().bool().tolist()",
        "        response_token_ids = [",
        "            [int(token) for token, keep in zip(tokens, mask, strict=True) if keep]",
        "            for tokens, mask in zip(response_ids_cpu, response_mask_cpu, strict=True)",
        "        ]",
        "        response_lengths = [len(tokens) for tokens in response_token_ids]",
        "        eos_token_id = self.tokenizer.eos_token_id",
        "        max_response_length = int(self.config.data.max_response_length)",
        "        finish_reasons = [",
        '            "length"',
        "            if length >= max_response_length",
        '            else "eos"',
        "            if tokens and eos_token_id is not None and tokens[-1] == eos_token_id",
        '            else "stop"',
        "            for tokens, length in zip(response_token_ids, response_lengths, strict=True)",
        "        ]",
        '        reward_extra_infos_to_dump["response_token_length"] = response_lengths',
        '        reward_extra_infos_to_dump["finish_reason"] = finish_reasons',
        '        extra_infos = batch.non_tensor_batch.get("extra_info", None)',
        "        if extra_infos is not None:",
        '            extra_infos = extra_infos.tolist() if hasattr(extra_infos, "tolist") else list(extra_infos)',
        "            prompt_ids = [",
        '                int(item.get("prompt_id", item.get("index")))',
        '                if isinstance(item, dict) and item.get("prompt_id", item.get("index")) is not None',
        "                else None",
        "                for item in extra_infos",
        "            ]",
        "            if len(prompt_ids) == len(outputs):",
        '                reward_extra_infos_to_dump["prompt_id"] = prompt_ids',
        "        # CYCLE09_BLOCK3_ROLLOUT_AUDIT_END",
        "",
        "",
    )
)
# The source block lives under ``with marked_timer``.  The literal remains
# visually compact above; normalize all relative levels to that 12-space base.
INSERT = "    " + INSERT.replace("\n        ", "\n            ")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch(verl_root: Path, *, check_only: bool) -> dict[str, object]:
    target = verl_root / "verl/trainer/ppo/ray_trainer.py"
    if not target.is_file():
        raise FileNotFoundError(target)
    before_hash = sha256(target)
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        status = "already_applied"
    elif check_only:
        raise RuntimeError(f"Cycle09 rollout audit patch is absent: {target}")
    else:
        anchors = [anchor for anchor in (ANCHOR, LEGACY_ANCHOR) if text.count(anchor) == 1]
        if len(anchors) != 1:
            raise RuntimeError(
                f"patch anchor count is ambiguous; canonical={text.count(ANCHOR)} "
                f"legacy={text.count(LEGACY_ANCHOR)} in {target}"
            )
        text = text.replace(anchors[0], INSERT + ANCHOR, 1)
        target.write_text(text, encoding="utf-8")
        status = "applied"
    after = target.read_text(encoding="utf-8")
    required = (
        MARKER,
        'reward_extra_infos_to_dump["finish_reason"]',
        'reward_extra_infos_to_dump["prompt_id"]',
    )
    missing = [item for item in required if item not in after]
    if missing:
        raise RuntimeError(f"rollout audit patch verification failed: {missing}")
    compile(after, str(target), "exec")
    return {
        "status": status,
        "target": str(target),
        "sha256_before": before_hash,
        "sha256_after": sha256(target),
        "training_semantics_changed": False,
        "added_dump_fields": [
            "prompt_id",
            "response_token_length",
            "finish_reason",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-root", type=Path, default=Path("/root/autodl-tmp/verl"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(patch(args.verl_root, check_only=args.check), indent=2))


if __name__ == "__main__":
    main()
