"""Cycle 07 think-format generative math runner (MATH500 / NuminaMath-test / AIME24).

Distinct from the Cycle 05/06 runners (`component/{math500,numina}/runner.py`),
which hardcode `enable_thinking=False` + `max_model_len=4096` for the no-think
instruct protocol. This runner is for the Qwen3-4B-Base + think-format SFT
trajectory (Cycle 07). Key differences:

  - chat template applied, `enable_thinking` NOT forced off — the model is trained
    to emit `<think>...</think>` naturally; the untrained base (step_000) just
    produces direct output and the extractor grabs the boxed answer wherever it is.
  - long generation budget (default 32768) for long-CoT reasoning chains.
  - records per-sample response **token** length (the dip-surge / recover-contract
    diagnostic required by the Cycle 07 design).
  - sampling temp=0.6 / top_p=0.9 (Rethink SFT eval settings).

Scoring reuses the validated `scorer_v2.score` + `scorer.extract_pred` (last
`\\boxed{}` + math_verify), identical to the instruct runners — `extract_pred`
already strips the `</think>` block before extracting, so think-format output is
handled correctly.

Importable API:
    from component.think_math.runner_think import run
    summary = run(task="math500", model="/path", label="step_000", outdir="/out")

CLI (subprocess-isolated by run_cycle07.py):
    python runner_think.py --task math500 --model PATH --label step_000 --outdir DIR
"""
import argparse
import gc
import hashlib
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

_COMPONENT = Path(__file__).resolve().parent.parent  # Eval/component/
if str(_COMPONENT) not in sys.path:
    sys.path.insert(0, str(_COMPONENT))

from scorer_v2 import score, is_mcq  # noqa: E402
from scorer import extract_pred       # noqa: E402

_EVAL_DIR = Path(__file__).resolve().parents[2]  # Eval/

# Match the Math-CoT-20k training `message` field exactly (single newline before the
# instruction) so eval prompts are in-distribution with what the model trained on.
INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."

MATH500_TEST = _EVAL_DIR / "tasks/data/hendrycks_math500/test.jsonl"
NUMINA_TEST = Path("/root/autodl-tmp/prepared/NuminaMath-1___5/test.jsonl")
NUMINA_JUNK = {"", "not found", "notfound", "none", "nan", "proof"}
SCORE_TIMEOUT_SECONDS = 60.0
SCORE_WORKER_MEMORY_GIB = 8
SCORE_WORKER_MAX_TASKS = 64


# --------------------------------------------------------------------------- #
# Task data loaders → list[{"problem": str, "answer": str}]
# --------------------------------------------------------------------------- #
def _load_math500(n: int) -> list[dict]:
    rows = [json.loads(l) for l in open(MATH500_TEST)]
    rows = [{"problem": r["problem"], "answer": r["answer"],
             "level": r.get("level"), "subject": r.get("subject")} for r in rows]
    return rows[:n] if (n and n < len(rows)) else rows


def _load_numina(n: int) -> list[dict]:
    rows = [json.loads(l) for l in open(NUMINA_TEST)]
    rows = [r for r in rows if str(r.get("answer", "")).strip().lower() not in NUMINA_JUNK]
    rows = [{"problem": r["problem"], "answer": str(r["answer"])} for r in rows]
    return rows[:n] if (n and n < len(rows)) else rows


def _load_aime24(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/aime_2024", split="train")
    rows = [{"problem": r["Problem"], "answer": str(r["Answer"]), "id": r.get("ID")}
            for r in ds]
    return rows[:n] if (n and n < len(rows)) else rows


_LOADERS = {"math500": _load_math500, "numina": _load_numina, "aime24": _load_aime24}


def _binom_se(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rows_fingerprint(rows: list[dict]) -> str:
    payload = [
        {"problem": row["problem"], "answer": str(row["answer"])}
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _score_worker(connection, memory_gib: int) -> None:
    # SymPy can expand a malformed expression until the whole runner is killed.
    # Keep the validated scorer, but contain each worker's address space.
    try:
        import resource

        limit = memory_gib * 2**30
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, OSError, ValueError):
        pass

    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            index, prediction, gold = request
            try:
                result = bool(score(prediction, gold))
                connection.send((index, result, None))
            except BaseException as exc:
                connection.send(
                    (index, False, f"{type(exc).__name__}: {exc}")
                )
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


def _start_score_worker(context):
    parent, child = context.Pipe(duplex=True)
    process = context.Process(
        target=_score_worker,
        args=(child, SCORE_WORKER_MEMORY_GIB),
        name="math-score-worker",
    )
    process.start()
    child.close()
    return parent, process


def _stop_score_worker(connection, process, *, force: bool) -> None:
    if process.is_alive() and not force:
        try:
            connection.send(None)
        except (BrokenPipeError, EOFError, OSError):
            force = True
    process.join(timeout=5.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=5.0)
    connection.close()


def _score_generations(generations: list[dict]) -> tuple[list[bool], list[str]]:
    context = mp.get_context("spawn")
    connection, process = _start_score_worker(context)
    tasks_in_worker = 0
    results: list[bool] = []
    statuses: list[str] = []

    try:
        for index, row in enumerate(generations):
            if tasks_in_worker >= SCORE_WORKER_MAX_TASKS:
                _stop_score_worker(connection, process, force=False)
                connection, process = _start_score_worker(context)
                tasks_in_worker = 0

            try:
                connection.send((index, row["gen"], row["gold"]))
                ready = connection.poll(SCORE_TIMEOUT_SECONDS)
                if not ready:
                    raise TimeoutError
                returned_index, result, error = connection.recv()
                if returned_index != index:
                    raise RuntimeError(
                        f"score worker index mismatch {returned_index} != {index}"
                    )
                results.append(bool(result))
                statuses.append("ok" if error is None else f"error:{error}")
                tasks_in_worker += 1
                if error is not None:
                    _stop_score_worker(connection, process, force=False)
                    connection, process = _start_score_worker(context)
                    tasks_in_worker = 0
            except TimeoutError:
                results.append(False)
                statuses.append("timeout")
                print(
                    f"[score] row={index} exceeded {SCORE_TIMEOUT_SECONDS:.0f}s; "
                    "recorded false and restarted worker",
                    flush=True,
                )
                _stop_score_worker(connection, process, force=True)
                connection, process = _start_score_worker(context)
                tasks_in_worker = 0
            except (BrokenPipeError, EOFError, OSError) as exc:
                results.append(False)
                statuses.append(f"worker_exit:{type(exc).__name__}")
                print(
                    f"[score] row={index} worker exited; recorded false and restarted",
                    flush=True,
                )
                _stop_score_worker(connection, process, force=True)
                connection, process = _start_score_worker(context)
                tasks_in_worker = 0

            if (index + 1) % 25 == 0 or index + 1 == len(generations):
                print(f"[score] {index + 1}/{len(generations)}", flush=True)
    finally:
        _stop_score_worker(connection, process, force=False)

    return results, statuses


def _shutdown_llm(llm) -> None:
    try:
        llm.llm_engine.engine_core.shutdown(timeout=30.0)
    except Exception as exc:
        print(f"[vllm] explicit shutdown warning: {exc}", flush=True)


def run(
    task: str,
    model: str,
    label: str,
    *,
    n: int = 0,
    outdir: str,
    max_tokens: int = 30720,
    max_model_len: int = 32768,   # Qwen3-4B-Base native max_position_embeddings
    gpu_mem: float = 0.85,
    temperature: float = 0.6,
    top_p: float = 0.9,
    seed: int = 42,
) -> dict:
    """Run one think-format generative eval. Returns summary dict."""
    if task not in _LOADERS:
        raise ValueError(f"unknown task {task!r}; expected one of {list(_LOADERS)}")

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    rows = _LOADERS[task](n)
    generation_path = out / f"{label}_generations.jsonl"
    generation_manifest_path = out / f"{label}_generations_manifest.json"
    generation_protocol = {
        "task": task,
        "label": label,
        "model": model,
        "n": len(rows),
        "rows_sha256": _rows_fingerprint(rows),
        "max_tokens": max_tokens,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_mem,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "prompt_instruction": INSTR,
    }

    generations: list[dict] | None = None
    if generation_path.is_file() and generation_manifest_path.is_file():
        try:
            generation_manifest = json.loads(
                generation_manifest_path.read_text(encoding="utf-8")
            )
            candidate = _read_jsonl(generation_path)
            if (
                generation_manifest.get("status") == "complete"
                and generation_manifest.get("protocol") == generation_protocol
                and len(candidate) == len(rows)
                and [item.get("row") for item in candidate] == list(range(len(rows)))
            ):
                generations = candidate
                print(
                    f"[generation] cached {generation_path} rows={len(generations)}",
                    flush=True,
                )
        except (OSError, ValueError, json.JSONDecodeError):
            generations = None

    generation_seconds = 0.0
    if generations is None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        prompts = []
        for row in rows:
            messages = [{"role": "user", "content": row["problem"] + INSTR}]
            # Keep the Cycle 07 think-format template default.
            prompts.append(
                tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        llm = None
        model_outputs = None
        started = time.monotonic()
        try:
            llm = LLM(
                model=model,
                dtype="bfloat16",
                gpu_memory_utilization=gpu_mem,
                max_model_len=max_model_len,
                trust_remote_code=True,
            )
            sampling = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
            )
            model_outputs = llm.generate(prompts, sampling)
            generations = []
            for index, (row, output) in enumerate(zip(rows, model_outputs)):
                candidate = output.outputs[0]
                generations.append(
                    {
                        "row": index,
                        "gold": row["answer"],
                        "resp_len": len(candidate.token_ids),
                        "finish": candidate.finish_reason,
                        "level": row.get("level"),
                        "subject": row.get("subject"),
                        "id": row.get("id"),
                        "gen": candidate.text,
                    }
                )
            generation_seconds = time.monotonic() - started
            _atomic_jsonl(generation_path, generations)
            _atomic_json(
                generation_manifest_path,
                {
                    "status": "complete",
                    "protocol": generation_protocol,
                    "generation_seconds": generation_seconds,
                    "output": str(generation_path),
                },
            )
            print(
                f"[generation] persisted {len(generations)} rows before scoring",
                flush=True,
            )
        finally:
            if llm is not None:
                _shutdown_llm(llm)
            model_outputs = None
            llm = None
            tok = None
            prompts = None
            gc.collect()

    if generations is None:
        raise RuntimeError("generation stage returned no records")

    score_started = time.monotonic()
    score_results, score_statuses = _score_generations(generations)
    scoring_seconds = time.monotonic() - score_started

    samples = []
    for row, ok, score_status in zip(generations, score_results, score_statuses):
        samples.append(
            {
                "gold": row["gold"],
                "pred": extract_pred(row["gen"]),
                "ok": ok,
                "resp_len": row["resp_len"],
                "finish": row["finish"],
                "level": row.get("level"),
                "subject": row.get("subject"),
                "id": row.get("id"),
                "gen": row["gen"],
                "score_status": score_status,
            }
        )
    _atomic_jsonl(out / f"{label}_samples.jsonl", samples)

    total = len(rows)
    correct = sum(score_results)
    boxed = sum("\\boxed" in row["gen"] for row in generations)
    trunc = sum(row["finish"] == "length" for row in generations)
    resp_lens = [int(row["resp_len"]) for row in generations]
    acc = correct / total if total else 0.0
    mean_len = sum(resp_lens) / len(resp_lens) if resp_lens else 0.0
    summary = {
        "task": task, "label": label, "model": model, "n": total,
        "acc": acc, "stderr": _binom_se(acc, total),
        "boxed_rate": boxed / total if total else 0.0,
        "trunc_rate": trunc / total if total else 0.0,
        "mean_response_len": mean_len,
        "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "seed": seed,
        "generation_cache": str(generation_path),
        "generation_seconds_this_invocation": generation_seconds,
        "scoring_seconds": scoring_seconds,
        "score_timeout_seconds": SCORE_TIMEOUT_SECONDS,
        "score_worker_memory_gib": SCORE_WORKER_MEMORY_GIB,
        "score_timeout_count": sum(
            status == "timeout" for status in score_statuses
        ),
        "score_error_count": sum(
            status != "ok" and status != "timeout" for status in score_statuses
        ),
    }
    _atomic_json(out / f"{label}.json", summary)
    print(f"\n===== {task} / {label} (N={total}) =====")
    print(f"acc {acc:.3f} ±{summary['stderr']:.3f} | boxed {summary['boxed_rate']:.3f} "
          f"| trunc {summary['trunc_rate']:.3f} | mean_resp_len {mean_len:.0f}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Cycle 07 think-format math eval")
    ap.add_argument("--task", required=True, choices=list(_LOADERS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=30720)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    run(
        task=args.task, model=args.model, label=args.label, n=args.n,
        outdir=args.outdir, max_tokens=args.max_tokens, max_model_len=args.max_model_len,
        gpu_mem=args.gpu_mem, temperature=args.temperature, top_p=args.top_p, seed=args.seed,
    )


if __name__ == "__main__":
    main()
