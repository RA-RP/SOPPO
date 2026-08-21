import jsonlines

from src.data.dataset import PreferenceCollator, PreferenceDataset


class FakeTokenizer:
    pad_token_id = 0
    eos_token = "!"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        return "PFX"

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(value) for value in text]}


def test_qwen_style_response_mask_and_dynamic_padding(tmp_path):
    path = tmp_path / "pairs.jsonl"
    with jsonlines.open(path, "w") as writer:
        writer.write({"sample_id": "a", "prompt": "p", "response_a": "xy", "response_b": "z", "label": 1})
        writer.write({"sample_id": "b", "prompt": "q", "response_a": "x", "response_b": "uvw", "label": 0})
    dataset = PreferenceDataset(str(path), FakeTokenizer(), max_length=16, require_labels=True)
    first = dataset[0]
    assert first["loss_mask_a"][:3] == [0, 0, 0]
    assert sum(first["loss_mask_a"]) == 3
    batch = PreferenceCollator(0)([dataset[0], dataset[1]])
    assert batch["input_ids_a"].shape[1] == 6
    assert batch["attention_mask_a"][1, -1] == 0


def test_unlabeled_rejects_label_leak(tmp_path):
    path = tmp_path / "bad.jsonl"
    with jsonlines.open(path, "w") as writer:
        writer.write({"sample_id": "x", "prompt": "p", "response_a": "a", "response_b": "b", "label": 1})
    try:
        PreferenceDataset(str(path), FakeTokenizer(), require_labels=False)
    except ValueError as exc:
        assert "Label isolation" in str(exc)
    else:
        raise AssertionError("Label leak was not rejected")


def test_dataset_rejects_duplicate_ids_and_nonbinary_labels(tmp_path):
    path = tmp_path / "bad_labeled.jsonl"
    with jsonlines.open(path, "w") as writer:
        writer.write({"sample_id": "x", "prompt": "p", "response_a": "a", "response_b": "b", "label": 1})
        writer.write({"sample_id": "x", "prompt": "q", "response_a": "c", "response_b": "d", "label": 2})
    try:
        PreferenceDataset(str(path), FakeTokenizer(), require_labels=True)
    except ValueError as exc:
        assert "Duplicate sample_id" in str(exc) or "0/1" in str(exc)
    else:
        raise AssertionError("Malformed labeled rows were not rejected")
