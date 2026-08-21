"""Prepare GetSlice inputs/configs and derive Qwen L18 relative functional ranks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jsonlines


EQUAL5 = (
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
EQUAL7 = ("self_attn.q_proj", "self_attn.k_proj") + EQUAL5
EPSILONS = (0.01, 0.025, 0.05, 0.10)


def prepare(data_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite C_epsilon inputs: {output}")
    s_dir = output / "S" / "soppo"
    x_dir = output / "X"
    s_dir.mkdir(parents=True)
    x_dir.mkdir(parents=True)
    with jsonlines.open(data_dir / "labeled_train.jsonl") as reader, jsonlines.open(
        s_dir / "labeled_train.jsonl", "w"
    ) as writer:
        for row in reader:
            answer = row["response_a"] if int(row["label"]) == 1 else row["response_b"]
            writer.write({"question": row["prompt"], "answer": answer})
    with jsonlines.open(data_dir / "unlabeled_train.jsonl") as reader, jsonlines.open(
        x_dir / "unlabeled.jsonl", "w"
    ) as writer:
        for row in reader:
            side = int(row["sample_id"].encode().hex(), 16) & 1
            writer.write({"output": {"text": row["response_a"] if side else row["response_b"]}})
    print(f"C_epsilon inputs ready: {output}")


def write_config(model: Path, inputs: Path, output: Path, config_path: Path) -> None:
    if config_path.exists():
        raise FileExistsError(f"Refuse to overwrite GetSlice config: {config_path}")
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "model": str(model),
        "save_path": str(output),
        "tasks": ["soppo"],
        "mode": "split_whitened_svd",
        "DEV": "cuda",
        "model_seq_len": 2048,
        "seed": 42,
        "target_layer": 18,
        "layer_gpu_chunk_size": 6,
        "single_layer_task_group_size": 1,
        "epsilon": 0.001,
        "svd_singular_floor": 0.0,
        "cholesky_jitter": 0.00001,
        "activation_cache_device": "cpu",
        "uv_dtype": "float32",
        "cleanup_intermediate": False,
        "skip_existing_outputs": False,
        "model_dtype": "bfloat16",
        "trust_remote_code": False,
        "hf_cache_dir": None,
        "s_nsamples": 128,
        "s_jsonl_path": str(inputs / "S"),
        "s_jsonl_file": "labeled_train.jsonl",
        "s_batch_size": 1,
        "x_nsamples": 128,
        "x_jsonl_path": str(inputs / "X" / "unlabeled.jsonl"),
        "x_batch_size": 1,
        "save_s_json_path": str(output / "sMat_S.json"),
        "save_x_json_path": str(output / "xMat_X_S.json"),
        "save_s_pt_path": None,
        "save_x_pt_path": None,
        "save_s_uv_path": None,
        "save_x_uv_path": None,
        "save_metrics_pt_path": None,
        "save_metrics_json_path": None,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def functional_rank(values, epsilon: float) -> int:
    energy = [float(value) ** 2 for value in values]
    total = sum(energy)
    if total <= 0:
        return 0
    threshold = (1.0 - epsilon) * total
    cumulative = 0.0
    for index, value in enumerate(energy, 1):
        cumulative += value
        if cumulative >= threshold:
            return index
    return len(energy)


def spectra(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "layer_18" not in payload:
        raise KeyError(f"GetSlice output has no layer_18: {path}")
    return payload["layer_18"]


def derive(base: Path, methods_root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite C_epsilon aggregate: {output}")
    output.mkdir(parents=True)
    base_s = spectra(base / "sMat_S.json")
    base_x = spectra(base / "xMat_X_S.json")
    rows = []
    for method_dir in sorted(path for path in methods_root.iterdir() if path.is_dir()):
        for cell in sorted(path for path in method_dir.iterdir() if path.is_dir()):
            for domain, base_values, filename in (
                ("S", base_s, "sMat_S.json"),
                ("X_S", base_x, "xMat_X_S.json"),
            ):
                current_values = spectra(cell / filename)
                for module in EQUAL7:
                    if module not in base_values or module not in current_values:
                        raise KeyError(f"Missing L18 module {module} in {cell / filename}")
                    for epsilon in EPSILONS:
                        base_rank = functional_rank(base_values[module], epsilon)
                        current_rank = functional_rank(current_values[module], epsilon)
                        compression = (
                            (base_rank - current_rank) / base_rank if base_rank > 0 else 0.0
                        )
                        rows.append(
                            {
                                "method": method_dir.name,
                                "checkpoint": cell.name,
                                "domain": domain,
                                "layer": 18,
                                "module": module,
                                "epsilon": epsilon,
                                "base_rank": base_rank,
                                "current_rank": current_rank,
                                "c_epsilon": compression,
                            }
                        )
    detail = output / "c_epsilon_module_audit.csv"
    with detail.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate_rows = []
    keys = sorted({(row["method"], row["checkpoint"], row["domain"], row["epsilon"]) for row in rows})
    for method, checkpoint, domain, epsilon in keys:
        selected = [
            row for row in rows
            if (row["method"], row["checkpoint"], row["domain"], row["epsilon"])
            == (method, checkpoint, domain, epsilon)
        ]
        for label, modules in (("equal5_non_qk", EQUAL5), ("equal7", EQUAL7)):
            values = [row["c_epsilon"] for row in selected if row["module"] in modules]
            if len(values) != len(modules):
                raise ValueError(f"Incomplete {label} cell: {method}/{checkpoint}/{domain}/{epsilon}")
            aggregate_rows.append(
                {
                    "method": method,
                    "checkpoint": checkpoint,
                    "domain": domain,
                    "layer": 18,
                    "epsilon": epsilon,
                    "aggregation": label,
                    "c_epsilon": sum(values) / len(values),
                }
            )
    with (output / "c_epsilon_trajectory.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    (output / "complete.json").write_text(
        json.dumps({"status": "succeeded", "module_rows": len(rows), "aggregate_rows": len(aggregate_rows)}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data-dir", required=True)
    prep.add_argument("--output", required=True)
    config = sub.add_parser("config")
    config.add_argument("--model", required=True)
    config.add_argument("--inputs", required=True)
    config.add_argument("--output", required=True)
    config.add_argument("--config", required=True)
    post = sub.add_parser("derive")
    post.add_argument("--base", required=True)
    post.add_argument("--methods-root", required=True)
    post.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(Path(args.data_dir).resolve(), Path(args.output).resolve())
    elif args.command == "config":
        write_config(
            Path(args.model).resolve(),
            Path(args.inputs).resolve(),
            Path(args.output).resolve(),
            Path(args.config).resolve(),
        )
    else:
        derive(Path(args.base).resolve(), Path(args.methods_root).resolve(), Path(args.output).resolve())


if __name__ == "__main__":
    main()
