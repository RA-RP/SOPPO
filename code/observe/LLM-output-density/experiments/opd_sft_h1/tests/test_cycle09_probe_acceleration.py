from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cycle09_r4_campaign as campaign
import cycle09_r4_common as common


class ToyAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        mixed = torch.tanh(
            self.q_proj(hidden) + self.k_proj(hidden) + self.v_proj(hidden)
        )
        return self.o_proj(mixed)


class ToyMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.up_proj = nn.Linear(width, width, bias=False)
        self.down_proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.sigmoid(self.gate_proj(hidden)) * self.up_proj(hidden))


class ToyLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_attn = ToyAttention(width)
        self.mlp = ToyMLP(width)
        self.calls = 0

    def forward(self, hidden: torch.Tensor):
        self.calls += 1
        hidden = hidden + self.self_attn(hidden)
        return (hidden + self.mlp(hidden),)


class ToyModel(nn.Module):
    def __init__(self, width: int = 6, layers: int = 3):
        super().__init__()
        self.config = SimpleNamespace(pad_token_id=0, eos_token_id=0)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(32, width)
        self.model.layers = nn.ModuleList([ToyLayer(width) for _ in range(layers)])

    def forward(self, input_ids: torch.Tensor, **_kwargs):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(logits=hidden)


def make_sample(sample_id: str, length: int) -> common.PreparedSample:
    positions = (1, length // 2, length - 1)
    bins = ("early", "mid", "late")
    weights = torch.zeros(length, dtype=torch.float32)
    windows = []
    for index, (position, bin_name) in enumerate(zip(positions, bins, strict=True)):
        weights[position] += 1.0 / len(positions)
        windows.append(
            common.WindowRecord(
                sample_id=sample_id,
                corpus_id="toy",
                window_index=index,
                start=position,
                end=position + 1,
                token_count=1,
                relative_start=position / length,
                relative_center=(position + 0.5) / length,
                relative_end=(position + 1) / length,
                position_bin=bin_name,
            )
        )
    return common.PreparedSample(
        sample_id=sample_id,
        input_ids=torch.arange(1, length + 1, dtype=torch.long).unsqueeze(0),
        attention_mask=torch.ones((1, length), dtype=torch.long),
        token_weights=weights,
        eligible_start=1,
        eligible_end=length,
        windows=windows,
        source={},
    )


def assert_nested_close(left, right) -> None:
    if isinstance(left, torch.Tensor):
        if left.dtype in (torch.float16, torch.bfloat16):
            torch.testing.assert_close(left, right, rtol=1e-3, atol=1e-5)
        else:
            torch.testing.assert_close(left, right, rtol=5e-5, atol=5e-6)
        return
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_close(left[key], right[key])
        return
    if isinstance(left, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            assert_nested_close(left_item, right_item)
        return
    assert left == right


def test_batched_profile_matches_single_sample_and_stops_after_target_layer():
    torch.manual_seed(7)
    baseline_model = ToyModel()
    accelerated_model = ToyModel()
    accelerated_model.load_state_dict(baseline_model.state_dict())
    samples = [make_sample("a", 4), make_sample("b", 6), make_sample("c", 5)]

    baseline = campaign.collect_profile(
        baseline_model,
        samples,
        [0, 1],
        "cpu",
        keep_factors=True,
        keep_residual_samples=True,
        factor_layers=(1,),
        forward_batch_size=1,
        early_stop=False,
    )
    accelerated = campaign.collect_profile(
        accelerated_model,
        samples,
        [0, 1],
        "cpu",
        keep_factors=True,
        keep_residual_samples=True,
        factor_layers=(1,),
        forward_batch_size=3,
        max_batch_tokens=32,
        early_stop=True,
    )

    for key in (
        "grams",
        "residual_second",
        "residual_mean",
        "position_second",
        "position_mean",
        "position_counts",
        "sample_factors",
        "residual_samples",
        "residual_sample_means",
    ):
        assert_nested_close(baseline[key], accelerated[key])

    assert [layer.calls for layer in baseline_model.model.layers] == [3, 3, 3]
    assert [layer.calls for layer in accelerated_model.model.layers] == [1, 1, 0]
    assert accelerated["forward_execution"] == {
        "requested_batch_size": 3,
        "max_batch_tokens": 32,
        "batch_count": 1,
        "early_stop": True,
        "early_stop_layer": 1,
    }


def test_probe_batch_token_budget_preserves_order():
    samples = [make_sample("a", 4), make_sample("b", 6), make_sample("c", 5)]
    batches = list(campaign._probe_batches(samples, 8, 10))
    assert [[sample.sample_id for sample in batch] for batch in batches] == [
        ["a"],
        ["b"],
        ["c"],
    ]


def test_real_llama_padding_matches_single_sample_profile():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(11)
    baseline_model = LlamaForCausalLM(config).eval()
    accelerated_model = LlamaForCausalLM(config).eval()
    accelerated_model.load_state_dict(baseline_model.state_dict())
    samples = [make_sample("a", 4), make_sample("b", 6), make_sample("c", 5)]

    baseline = campaign.collect_profile(
        baseline_model,
        samples,
        [0, 1],
        "cpu",
        keep_factors=True,
        keep_residual_samples=True,
        factor_layers=(1,),
    )
    tail_calls = 0

    def count_tail(_module, _inputs, _output):
        nonlocal tail_calls
        tail_calls += 1

    handle = accelerated_model.model.layers[2].register_forward_hook(count_tail)
    try:
        accelerated = campaign.collect_profile(
            accelerated_model,
            samples,
            [0, 1],
            "cpu",
            keep_factors=True,
            keep_residual_samples=True,
            factor_layers=(1,),
            forward_batch_size=3,
            max_batch_tokens=32,
            early_stop=True,
        )
    finally:
        handle.remove()

    for key in (
        "grams",
        "residual_second",
        "residual_mean",
        "position_second",
        "position_mean",
        "sample_factors",
        "residual_samples",
        "residual_sample_means",
    ):
        assert_nested_close(baseline[key], accelerated[key])
    assert tail_calls == 0


def test_real_qwen3_padding_matches_single_sample_profile():
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
    )
    torch.manual_seed(13)
    baseline_model = Qwen3ForCausalLM(config).eval()
    accelerated_model = Qwen3ForCausalLM(config).eval()
    accelerated_model.load_state_dict(baseline_model.state_dict())
    samples = [make_sample("a", 4), make_sample("b", 6), make_sample("c", 5)]

    baseline = campaign.collect_profile(
        baseline_model,
        samples,
        [0, 1],
        "cpu",
        keep_factors=True,
        keep_residual_samples=True,
        factor_layers=(1,),
    )
    tail_calls = 0

    def count_tail(_module, _inputs, _output):
        nonlocal tail_calls
        tail_calls += 1

    handle = accelerated_model.model.layers[2].register_forward_hook(count_tail)
    try:
        accelerated = campaign.collect_profile(
            accelerated_model,
            samples,
            [0, 1],
            "cpu",
            keep_factors=True,
            keep_residual_samples=True,
            factor_layers=(1,),
            forward_batch_size=3,
            max_batch_tokens=32,
            early_stop=True,
        )
    finally:
        handle.remove()

    for key in (
        "grams",
        "residual_second",
        "residual_mean",
        "position_second",
        "position_mean",
        "sample_factors",
        "residual_samples",
        "residual_sample_means",
    ):
        assert_nested_close(baseline[key], accelerated[key])
    assert tail_calls == 0
