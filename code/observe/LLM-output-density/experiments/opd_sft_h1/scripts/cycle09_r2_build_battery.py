#!/usr/bin/env python3
"""Cycle09 R2 T9: build probe battery v2.1 (frozen, versioned). X-probe format follows GetSlice's strict reader: {"output": {"text": <text>}} per line.
Domains kept SEPARATE (user: do not merge into ID/OOD). X_teacher DEFERRED. Sources:
  X_ood_knowledge <- TIGER-Lab/MMLU-Pro (local HF cache)   [top priority]
  X_general       <- wikitext-103 (HF, network)
  X_math_hard     <- Maxwell-Jia/aime_2024 (local). Same-domain activation probe, treated uniformly with
                     X_ood_knowledge=MMLU-Pro (both are eval domains, used DELIBERATELY for same-domain
                     geometry<->behavior observation — the theory's own X_ood_knowledge rationale). The
                     "coupling" concern only bites an out-of-sample PREDICTION claim, which this project
                     does not make (it claims same-domain co-location). (user reconciliation 2026-07-09)
  X_math          <- reuse existing cycle07 X_base (base rollout on math)  [pointer, not rebuilt]
  X_bos, S        <- existing (reuse)
"""
import json, random
from pathlib import Path

OUT = Path("/root/autodl-tmp/cycle09_r2/getslice/inputs")
N_POOL = 128          # probe pool per domain; GetSlice config samples s/x_nsamples from this
SEED = 42
random.seed(SEED)


def write_probe(name, texts):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "x_probe.jsonl"
    with open(f, "w") as fh:
        for t in texts:
            fh.write(json.dumps({"output": {"text": t}}) + "\n")
    print(f"[battery] {name}: {len(texts)} -> {f}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    # X_ood_knowledge: MMLU-Pro (local cache), question + options as text
    mp = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    idx = random.sample(range(len(mp)), min(N_POOL, len(mp)))
    ood = []
    for i in idx:
        r = mp[i]
        opts = " ".join(f"({chr(65+j)}) {o}" for j, o in enumerate(r["options"]))
        ood.append(f"{r['question']}\nOptions: {opts}")
    write_probe("X_ood_knowledge", ood)

    # X_math_hard: AIME 2024 (local). CAVEAT recorded in battery_manifest.
    aime = load_dataset("Maxwell-Jia/aime_2024", split="train")
    hard = [str(aime[i]["Problem"]) for i in range(len(aime))]  # only ~30
    write_probe("X_math_hard", hard)

    # X_general: wikitext-103 (network)
    try:
        wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        gen, it = [], iter(wt)
        while len(gen) < N_POOL:
            t = next(it)["text"].strip()
            if len(t) > 200:
                gen.append(t[:2000])
        write_probe("X_general", gen)
        gen_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"[battery] X_general FAILED (network?): {e}", flush=True)
        gen_ok = False

    manifest = {
        "version": "v2.1-r2", "seed": SEED, "n_pool": N_POOL,
        "X_ood_knowledge": {"source": "TIGER-Lab/MMLU-Pro test", "n": len(ood)},
        "X_math_hard": {"source": "Maxwell-Jia/aime_2024", "n": len(hard),
                        "note": "same-domain activation probe; treated uniformly with X_ood_knowledge=MMLU-Pro "
                                "(both eval domains, deliberate same-domain geometry<->behavior observation). "
                                "Not an out-of-sample prediction claim. Reconciliation flagged to Theory "
                                "(AIME-coupling caution was inconsistent with the MMLU-Pro same-domain choice)."},
        "X_general": {"source": "wikitext-103-raw-v1", "ok": gen_ok},
        "X_math": {"reuse": "cycle07 getslice/inputs/X_base/x_probe.jsonl (base rollout on math)"},
        "X_bos": {"reuse": "existing"}, "S": {"reuse": "per-arm cycle07 S (dataset CoT)"},
        "X_teacher": "DEFERRED (off-KD control / Tier B)",
    }
    (OUT.parent / "battery_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[battery] manifest -> {OUT.parent/'battery_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
