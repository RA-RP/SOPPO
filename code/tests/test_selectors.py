import json
from pathlib import Path

from src.training.selectors import headroom, static_lambda


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(root: Path, name: str, accuracy: float, brier: float, raw_accuracy=None) -> None:
    run = root / name
    write_json(run / "complete.json", {"status": "succeeded"})
    best = {"step": 40, "val_accuracy": accuracy, "val_brier": brier}
    if raw_accuracy is not None:
        best.update(
            {
                "raw_mean_logp_val_accuracy": raw_accuracy,
                "raw_mean_logp_val_brier": 0.22,
                "raw_mean_logp_score_type": "simpo_mean_logp_delta_margin_free",
                "val_samples": 300,
            }
        )
    write_json(run / "best.json", best)
    write_json(run / "checkpoints" / "step_000040" / "adapter_config.json", {})


def test_headroom_compares_dpo10_with_its_frozen_base_not_dpo100(tmp_path):
    main = tmp_path / "main"
    make_run(main, "dpo10", 0.70, 0.20, raw_accuracy=0.64)
    write_json(
        main / "dpo10" / "initial_validation.json",
        {
            "raw_mean_logp_val_accuracy": 0.58,
            "raw_mean_logp_val_brier": 0.25,
            "raw_mean_logp_score_type": "simpo_mean_logp_delta_margin_free",
            "checkpoint": "frozen_qwen3_base_before_training",
            "val_samples": 300,
        },
    )
    output = tmp_path / "headroom"
    headroom(main, output)
    result = json.loads((output / "headroom_selection.json").read_text(encoding="utf-8"))
    assert result["status"] == "succeeded"
    assert result["baseline"] == "frozen_qwen3_before_training"
    assert abs(result["headroom"] - 0.06) < 1e-12


def test_static_lambda_selection_uses_validation_only(tmp_path):
    main = tmp_path / "main"
    for value, accuracy, brier in (
        (0.1, 0.60, 0.24),
        (0.3, 0.65, 0.23),
        (0.5, 0.65, 0.21),
        (1.0, 0.64, 0.20),
    ):
        make_run(main, f"soppo_pe_static_lambda_{value:.1f}", accuracy, brier)
    gate = tmp_path / "headroom.json"
    write_json(gate, {"status": "succeeded", "headroom": 0.06})
    output = tmp_path / "selection"
    static_lambda(main, gate, output)
    result = json.loads((output / "lambda_selection.json").read_text(encoding="utf-8"))
    assert result["selected_static_lambda"] == 0.5
    assert result["selected_method"] == "soppo_pe_static_lambda_0.5"
