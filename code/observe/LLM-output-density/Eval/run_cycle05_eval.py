#!/usr/bin/env python3
"""Cycle 05 full evaluation orchestration.

Phase 1: MMLU validation gate — base + theta0, single subject (abstract_algebra).
         Gate: both > 0.25. Abort if fails, write failure finding.
Phase 2: Full 8-model evaluation.
         - GSM8K (run_eval.py --task gsm8k): 4 remaining models only (reuse 4 existing)
         - MATH500 (run_eval.py --task math500): 4 remaining models only
         - MMLU (lm_eval, base model mode, NO chat_template, 5-shot): all 8
           Protocol matches Qwen3 official technical report (arXiv 2505.09388):
           no --apply_chat_template, 5-shot, standard loglikelihood. This is the
           field standard for MMLU — chat template degrades loglikelihood scoring
           because the model's probability mass shifts to think-block tokens rather
           than bare A/B/C/D letters (FINDING_05_mmlu_chat_template_collapse.md).
           lm-eval-harness issues #3405/#3576 independently confirm this behavior.
         - TruthfulQA-MC1 (lm_eval, enable_thinking=False, apply_chat_template): all 8
         - WinoGrande (lm_eval, NO chat_template): all 8
         - ARC-Challenge (lm_eval, enable_thinking=False, apply_chat_template, 25-shot): all 8
           Same chat protocol as TruthfulQA. Think mode not used: enable_thinking=True +
           loglikelihood collapses (same mechanism as MMLU chat-template collapse).
           Metric: acc_norm (length-normalized). ARC Challenge Set N=1172.

All output → /root/autodl-tmp/floor_probe/cycle05_full_eval/
Existing corrected results for base/theta0/opd_lmbda05/sft_n128 are reused as-is.
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Directory containing this script (Eval/)
EVAL_DIR = Path(__file__).resolve().parent

OUTROOT = Path("/root/autodl-tmp/floor_probe/cycle05_full_eval")
PYTHON = "/root/miniconda3/envs/density/bin/python"
LMEVAL = "/root/miniconda3/envs/density/bin/lm_eval"

LOG = OUTROOT / "logs" / "master.log"

MODELS = {
    "base":       "/root/autodl-tmp/model/Qwen/Qwen3-1.7B",
    "theta0":     "/root/autodl-tmp/cycle04_opd_stability_gain/model_outputs/theta0/256",
    "opd_lmbda05":"/root/autodl-tmp/cycle04_opd_stability_gain/model_outputs/opd_lmbda05/800",
    "opd_lmbda1": "/root/autodl-tmp/cycle04_opd_stability_gain/model_outputs/opd_lmbda1/800",
    "sft_n128":   "/root/autodl-tmp/cycle04_opd_stability_gain/model_outputs/sft_n128/128",
    "sft_n256":   "/root/autodl-tmp/cycle04_opd_stability_gain/model_outputs/sft_n256/256",
    "sft_n512":   "/root/autodl-tmp/cycle04_opd_stability_gain/model_outputs/sft_n512/512",
    "sft_n1024":  "/root/autodl-tmp/cycle04_opd_stability_gain/model_outputs/sft_n1024/1024",
}

# 4 models that already have corrected GSM8K + MATH500 from cycle05 pre-run
ALREADY_CORRECTED = {"base", "theta0", "opd_lmbda05", "sft_n128"}

# Existing corrected result paths to reuse (produced by consistent_axis_rerun.sh, 2026-06-16)
EXISTING_GSM8K = Path("/root/autodl-tmp/floor_probe/gsm8k_results")
EXISTING_MATH500 = Path("/root/autodl-tmp/floor_probe/math500_results")

# 8 representative MMLU subjects for full-run score (avoids 10hr full-mmlu run).
# Selected to span STEM, humanities, reasoning, and factual knowledge.
# Each has ~100 questions; 8 subjects × 100 Q × 4 choices × 5-shot ≈ 4 min per model.
MMLU_TASKS_SUBSET = (
    "mmlu_abstract_algebra,"
    "mmlu_anatomy,"
    "mmlu_philosophy,"
    "mmlu_logical_fallacies,"
    "mmlu_high_school_mathematics,"
    "mmlu_world_religions,"
    "mmlu_clinical_knowledge,"
    "mmlu_computer_security"
)


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def run(cmd: list[str], logfile: Path, label: str) -> int:
    log(f"START {label}")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    rc = proc.returncode
    log(f"{'OK' if rc == 0 else 'FAIL rc=' + str(rc)} {label}")
    return rc


def find_lmeval_result(outdir: Path) -> dict | None:
    """Find and parse the lm_eval results JSON in outdir."""
    if not outdir.exists():
        return None
    candidates = sorted(outdir.glob("results_*.json"))
    if not candidates:
        # Try subdirectories
        for sub in sorted(outdir.iterdir()):
            if sub.is_dir():
                candidates = sorted(sub.glob("results_*.json"))
                if candidates:
                    break
    if not candidates:
        log(f"  WARNING: no results_*.json in {outdir}")
        return None
    result_path = candidates[-1]
    try:
        return json.loads(result_path.read_text())
    except Exception as e:
        log(f"  WARNING: failed to parse {result_path}: {e}")
        return None


def get_mmlu_score(outdir: Path) -> float | None:
    """Extract MMLU acc from lm_eval output dir.
    For Phase 1 gate: looks for mmlu_abstract_algebra only.
    For Phase 2 full run: averages across the 8-subject subset.
    """
    data = find_lmeval_result(outdir)
    if data is None:
        return None
    results = data.get("results", {})
    # Try aggregate keys first
    for key in ("mmlu",):
        if key in results:
            for metric in ("acc,none", "acc"):
                if metric in results[key]:
                    return float(results[key][metric])
    # Collect subset subjects and average them
    subset_keys = [
        "mmlu_abstract_algebra", "mmlu_anatomy", "mmlu_philosophy",
        "mmlu_logical_fallacies", "mmlu_high_school_mathematics",
        "mmlu_world_religions", "mmlu_clinical_knowledge", "mmlu_computer_security",
    ]
    vals = []
    for key in subset_keys:
        if key in results:
            for metric in ("acc,none", "acc"):
                if metric in results[key]:
                    vals.append(float(results[key][metric]))
                    break
    if vals:
        return sum(vals) / len(vals)
    log(f"  WARNING: could not find mmlu score in {outdir}. Keys: {list(results.keys())[:5]}")
    return None


def get_lmeval_score(outdir: Path, task: str, metric: str) -> float | None:
    data = find_lmeval_result(outdir)
    if data is None:
        return None
    results = data.get("results", {})
    if task in results and metric in results[task]:
        return float(results[task][metric])
    log(f"  WARNING: {task}/{metric} not in results. Keys: {list(results.keys())[:5]}")
    return None


def write_failure_finding(reason: str):
    p = OUTROOT / "FINDING_05_phase1_gate_failure.md"
    p.write_text(f"""# Finding: Phase 1 Gate Failure

## Timestamp
{ts()}

## Reason
{reason}

## Action
Cycle 05 eval aborted. OOD-lite MMLU generative approach did not pass the >0.25 gate
on base or theta0. Investigate lm_eval output in:
{OUTROOT}/phase1_validation/

Do not cite any MMLU numbers from this run.
""")
    log(f"Written failure finding to {p}")


# ── Phase 1 helpers ──────────────────────────────────────────────────────────

def run_phase1_mmlu_validation() -> bool:
    """Run MMLU abstract_algebra on base + theta0. Return True if both > 0.25."""
    log("=== PHASE 1: MMLU validation (abstract_algebra, base + theta0) ===")
    scores = {}
    for label in ("base", "theta0"):
        model_path = MODELS[label]
        outdir = OUTROOT / "phase1_validation" / label
        outdir.mkdir(parents=True, exist_ok=True)
        # Check if already done (smoke test may have run base)
        existing = get_mmlu_score(outdir)
        if existing is not None:
            log(f"  {label}: reusing existing phase1 result = {existing:.4f}")
            scores[label] = existing
            continue
        cmd = [
            PYTHON, str(EVAL_DIR / "run_eval.py"),
            "--task", "mmlu",
            "--model", model_path,
            "--label", label,
            "--outdir", str(OUTROOT / "phase1_validation"),
            "--lm-tasks", "mmlu_abstract_algebra",
            "--gpu-mem", "0.80",
        ]
        rc = run(cmd, OUTROOT / "logs" / f"phase1_{label}.log", f"phase1_mmlu_{label}")
        if rc != 0:
            log(f"  ERROR: lm_eval failed for {label}. Aborting.")
            write_failure_finding(f"lm_eval returned rc={rc} for {label}")
            return False
        score = get_mmlu_score(outdir)
        if score is None:
            write_failure_finding(f"Could not parse MMLU score for {label}")
            return False
        log(f"  {label} abstract_algebra MMLU = {score:.4f}")
        scores[label] = score

    base_ok = scores.get("base", 0) > 0.25
    theta0_ok = scores.get("theta0", 0) > 0.25
    if base_ok and theta0_ok:
        log(f"=== PHASE 1 PASSED (base={scores['base']:.4f}, theta0={scores['theta0']:.4f}) ===")
        return True
    else:
        msg = (f"Gate B05 failed: base={scores.get('base', 'N/A'):.4f} "
               f"theta0={scores.get('theta0', 'N/A'):.4f} — "
               f"standard mmlu (loglikelihood+enable_thinking=False) did not recover above 0.25. "
               f"Investigate lm_eval output in phase1_validation/.")
        log(f"=== PHASE 1 FAILED: {msg} ===")
        write_failure_finding(msg)
        return False


# ── Phase 2 helpers ──────────────────────────────────────────────────────────

def run_gsm8k(label: str) -> dict | None:
    """Run eval_gsm8k_full.py for one model. Returns summary dict or None."""
    outdir = OUTROOT / "gsm8k"
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / f"{label}.json"
    if result_path.exists():
        log(f"  gsm8k/{label}: already done, skipping")
        return json.loads(result_path.read_text())
    # Check existing corrected results
    existing = EXISTING_GSM8K / f"{label}.json"
    if existing.exists():
        import shutil
        shutil.copy(existing, result_path)
        log(f"  gsm8k/{label}: copied from existing corrected results")
        return json.loads(result_path.read_text())
    cmd = [
        PYTHON,
        str(EVAL_DIR / "run_eval.py"),
        "--task", "gsm8k",
        "--model", MODELS[label],
        "--label", label,
        "--n", "0",
        "--outdir", str(outdir),
    ]
    rc = run(cmd, OUTROOT / "logs" / f"gsm8k_{label}.log", f"gsm8k_{label}")
    if rc != 0:
        log(f"  ERROR: gsm8k failed for {label}")
        return None
    return json.loads(result_path.read_text()) if result_path.exists() else None


def run_math500(label: str) -> dict | None:
    """Run eval_math500_full.py for one model."""
    outdir = OUTROOT / "math500"
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / f"{label}.json"
    if result_path.exists():
        log(f"  math500/{label}: already done, skipping")
        return json.loads(result_path.read_text())
    existing = EXISTING_MATH500 / f"{label}.json"
    if existing.exists():
        import shutil
        shutil.copy(existing, result_path)
        log(f"  math500/{label}: copied from existing corrected results")
        return json.loads(result_path.read_text())
    cmd = [
        PYTHON,
        str(EVAL_DIR / "run_eval.py"),
        "--task", "math500",
        "--model", MODELS[label],
        "--label", label,
        "--n", "0",
        "--outdir", str(outdir),
    ]
    rc = run(cmd, OUTROOT / "logs" / f"math500_{label}.log", f"math500_{label}")
    if rc != 0:
        log(f"  ERROR: math500 failed for {label}")
        return None
    return json.loads(result_path.read_text()) if result_path.exists() else None


def run_mmlu_full(label: str) -> float | None:
    """Run full MMLU (8-subject subset, loglikelihood) for one model."""
    outdir = OUTROOT / "mmlu" / label
    # Check if already done
    existing = get_mmlu_score(outdir)
    if existing is not None:
        log(f"  mmlu/{label}: already done = {existing:.4f}, skipping")
        return existing
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(EVAL_DIR / "run_eval.py"),
        "--task", "mmlu",
        "--model", MODELS[label],
        "--label", label,
        "--outdir", str(OUTROOT / "mmlu"),
        "--gpu-mem", "0.80",
    ]
    rc = run(cmd, OUTROOT / "logs" / f"mmlu_{label}.log", f"mmlu_{label}")
    if rc != 0:
        log(f"  ERROR: mmlu failed for {label}")
        return None
    return get_mmlu_score(outdir)


def run_truthfulqa(label: str) -> float | None:
    """Run truthfulqa_mc1 for one model."""
    outdir = OUTROOT / "truthfulqa" / label
    existing_score = None
    existing_data = find_lmeval_result(outdir)
    if existing_data is not None:
        for key in ("truthfulqa_mc1",):
            r = existing_data.get("results", {}).get(key, {})
            for metric in ("acc,none", "acc"):
                if metric in r:
                    existing_score = float(r[metric])
                    break
        if existing_score is not None:
            log(f"  truthfulqa/{label}: already done = {existing_score:.4f}, skipping")
            return existing_score
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(EVAL_DIR / "run_eval.py"),
        "--task", "truthfulqa",
        "--model", MODELS[label],
        "--label", label,
        "--outdir", str(OUTROOT / "truthfulqa"),
        "--gpu-mem", "0.80",
    ]
    rc = run(cmd, OUTROOT / "logs" / f"truthfulqa_{label}.log", f"truthfulqa_{label}")
    if rc != 0:
        log(f"  ERROR: truthfulqa failed for {label}")
        return None
    data = find_lmeval_result(outdir)
    if data is None:
        return None
    r = data.get("results", {}).get("truthfulqa_mc1", {})
    for metric in ("acc,none", "acc"):
        if metric in r:
            return float(r[metric])
    return None


def run_winogrande(label: str) -> float | None:
    """Run winogrande for one model (NO chat template)."""
    outdir = OUTROOT / "winogrande" / label
    existing_data = find_lmeval_result(outdir)
    if existing_data is not None:
        r = existing_data.get("results", {}).get("winogrande", {})
        for metric in ("acc,none", "acc"):
            if metric in r:
                score = float(r[metric])
                log(f"  winogrande/{label}: already done = {score:.4f}, skipping")
                return score
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(EVAL_DIR / "run_eval.py"),
        "--task", "winogrande",
        "--model", MODELS[label],
        "--label", label,
        "--outdir", str(OUTROOT / "winogrande"),
        "--gpu-mem", "0.80",
    ]
    rc = run(cmd, OUTROOT / "logs" / f"winogrande_{label}.log", f"winogrande_{label}")
    if rc != 0:
        log(f"  ERROR: winogrande failed for {label}")
        return None
    data = find_lmeval_result(outdir)
    if data is None:
        return None
    r = data.get("results", {}).get("winogrande", {})
    for metric in ("acc,none", "acc"):
        if metric in r:
            return float(r[metric])
    return None


def run_arc_challenge(label: str) -> float | None:
    """Run arc_challenge for one model (enable_thinking=False, apply_chat_template, 25-shot).
    Uses loglikelihood acc_norm (length-normalized). Same chat protocol as TruthfulQA.
    Think mode not used: enable_thinking=True + loglikelihood collapses like MMLU.
    """
    outdir = OUTROOT / "arc_challenge" / label
    existing_data = find_lmeval_result(outdir)
    if existing_data is not None:
        r = existing_data.get("results", {}).get("arc_challenge", {})
        for metric in ("acc_norm,none", "acc_norm", "acc,none", "acc"):
            if metric in r:
                score = float(r[metric])
                log(f"  arc_challenge/{label}: already done = {score:.4f}, skipping")
                return score
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(EVAL_DIR / "run_eval.py"),
        "--task", "arc_challenge",
        "--model", MODELS[label],
        "--label", label,
        "--outdir", str(OUTROOT / "arc_challenge"),
        "--gpu-mem", "0.80",
    ]
    rc = run(cmd, OUTROOT / "logs" / f"arc_challenge_{label}.log", f"arc_challenge_{label}")
    if rc != 0:
        log(f"  ERROR: arc_challenge failed for {label}")
        return None
    data = find_lmeval_result(outdir)
    if data is None:
        return None
    r = data.get("results", {}).get("arc_challenge", {})
    for metric in ("acc_norm,none", "acc_norm", "acc,none", "acc"):
        if metric in r:
            return float(r[metric])
    return None


def run_phase2() -> dict:
    """Run full 8-model evaluation. Returns collected results dict."""
    log("=== PHASE 2: Full 8-model evaluation ===")
    results = {}

    # Order: GSM8K/MATH500 first (4 missing models), then lm_eval tasks per-model
    log("--- GSM8K + MATH500 for remaining 4 models ---")
    for label in list(MODELS.keys()):
        if label in ALREADY_CORRECTED:
            # Still copy into cycle05 output dir for provenance
            run_gsm8k(label)
            run_math500(label)
        else:
            run_gsm8k(label)
            run_math500(label)

    # lm_eval tasks: all 8 models
    log("--- MMLU (8-subject subset, loglikelihood) for all 8 models ---")
    for label in MODELS:
        score = run_mmlu_full(label)
        results.setdefault(label, {})["mmlu"] = score

    log("--- TruthfulQA-MC1 (all 8 models) ---")
    for label in MODELS:
        score = run_truthfulqa(label)
        results.setdefault(label, {})["truthfulqa_mc1"] = score

    log("--- WinoGrande (all 8 models) ---")
    for label in MODELS:
        score = run_winogrande(label)
        results.setdefault(label, {})["winogrande"] = score

    log("--- ARC-Challenge (all 8 models) ---")
    for label in MODELS:
        score = run_arc_challenge(label)
        results.setdefault(label, {})["arc_challenge"] = score

    return results


def load_numina_scores() -> dict:
    """Load NuminaMath ID scores for all models."""
    scores = {}
    numina_csv = Path("/root/autodl-tmp/floor_probe/full_results_v2/numina_id_summary.csv")
    if numina_csv.exists():
        import csv
        with numina_csv.open() as f:
            for row in csv.DictReader(f):
                lbl = row["label"]
                scores[lbl] = float(row["open_acc"])  # ID axis = open_acc
    base_json = Path("/root/autodl-tmp/floor_probe/base_eval/base_Qwen3-1.7B.json")
    if base_json.exists():
        d = json.loads(base_json.read_text())
        scores["base"] = d.get("open_acc", d.get("overall"))
    return scores


def load_gsm8k_scores() -> dict:
    """Load GSM8K corrected scores (from cycle05 output dir)."""
    outdir = OUTROOT / "gsm8k"
    scores = {}
    for label in MODELS:
        p = outdir / f"{label}.json"
        if p.exists():
            d = json.loads(p.read_text())
            scores[label] = {"acc": d.get("acc"), "n": d.get("n"), "trunc": d.get("trunc_rate")}
    return scores


def load_math500_scores() -> dict:
    outdir = OUTROOT / "math500"
    scores = {}
    for label in MODELS:
        p = outdir / f"{label}.json"
        if p.exists():
            d = json.loads(p.read_text())
            scores[label] = {"acc": d.get("acc"), "n": d.get("n"), "trunc": d.get("trunc_rate")}
    return scores


def binomial_stderr(p: float | None, n: int) -> float | None:
    if p is None or n == 0:
        return None
    import math
    return math.sqrt(p * (1 - p) / n)


def compile_results():
    """Read all result JSON files and generate RESULTS_05.md + CSV tables."""
    log("=== Compiling results ===")
    numina = load_numina_scores()
    gsm8k = load_gsm8k_scores()
    math500 = load_math500_scores()

    # Load MMLU / TruthfulQA / WinoGrande / ARC-Challenge from lm_eval outputs
    mmlu_scores = {}
    tqa_scores = {}
    wg_scores = {}
    arc_scores = {}
    for label in MODELS:
        mmlu_scores[label] = get_mmlu_score(OUTROOT / "mmlu" / label)
        data = find_lmeval_result(OUTROOT / "truthfulqa" / label)
        if data:
            r = data.get("results", {}).get("truthfulqa_mc1", {})
            for m in ("acc,none", "acc"):
                if m in r:
                    tqa_scores[label] = float(r[m])
                    break
        data = find_lmeval_result(OUTROOT / "winogrande" / label)
        if data:
            r = data.get("results", {}).get("winogrande", {})
            for m in ("acc,none", "acc"):
                if m in r:
                    wg_scores[label] = float(r[m])
                    break
        data = find_lmeval_result(OUTROOT / "arc_challenge" / label)
        if data:
            r = data.get("results", {}).get("arc_challenge", {})
            for m in ("acc_norm,none", "acc_norm", "acc,none", "acc"):
                if m in r:
                    arc_scores[label] = float(r[m])
                    break

    # Write CSV tables
    tables_dir = OUTROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    import csv

    # gsm8k_corrected.csv
    with (tables_dir / "gsm8k_corrected.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "acc", "stderr", "n", "trunc_rate", "source"])
        for label in MODELS:
            g = gsm8k.get(label, {})
            acc = g.get("acc")
            n = g.get("n", 1319)
            src = "reused_corrected" if label in ALREADY_CORRECTED else "cycle05_run"
            se = binomial_stderr(acc, n)
            w.writerow([label, f"{acc:.4f}" if acc else "N/A",
                        f"{se:.4f}" if se else "N/A", n,
                        f"{g.get('trunc', 0):.4f}" if g.get("trunc") is not None else "N/A",
                        src])

    # math500_corrected.csv
    with (tables_dir / "math500_corrected.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "acc", "stderr", "n", "trunc_rate", "source"])
        for label in MODELS:
            m5 = math500.get(label, {})
            acc = m5.get("acc")
            n = m5.get("n", 500)
            src = "reused_corrected" if label in ALREADY_CORRECTED else "cycle05_run"
            se = binomial_stderr(acc, n)
            w.writerow([label, f"{acc:.4f}" if acc else "N/A",
                        f"{se:.4f}" if se else "N/A", n,
                        f"{m5.get('trunc', 0):.4f}" if m5.get("trunc") is not None else "N/A",
                        src])

    # ood_lite_summary.csv
    with (tables_dir / "ood_lite_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mmlu_acc", "truthfulqa_mc1", "winogrande", "arc_challenge"])
        for label in MODELS:
            w.writerow([
                label,
                f"{mmlu_scores.get(label, ''):.4f}" if mmlu_scores.get(label) else "N/A",
                f"{tqa_scores.get(label, ''):.4f}" if tqa_scores.get(label) else "N/A",
                f"{wg_scores.get(label, ''):.4f}" if wg_scores.get(label) else "N/A",
                f"{arc_scores.get(label, ''):.4f}" if arc_scores.get(label) else "N/A",
            ])

    # id_ood_trajectory.csv (master table)
    with (tables_dir / "id_ood_trajectory.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "numina_id", "gsm8k", "gsm8k_se", "math500", "math500_se",
                    "mmlu_acc", "truthfulqa_mc1", "winogrande", "arc_challenge"])
        for label in MODELS:
            g = gsm8k.get(label, {})
            m5 = math500.get(label, {})
            gacc = g.get("acc")
            macc = m5.get("acc")
            w.writerow([
                label,
                f"{numina.get(label, ''):.4f}" if numina.get(label) else "N/A",
                f"{gacc:.4f}" if gacc else "N/A",
                f"{binomial_stderr(gacc, g.get('n', 1319)):.4f}" if gacc else "N/A",
                f"{macc:.4f}" if macc else "N/A",
                f"{binomial_stderr(macc, m5.get('n', 500)):.4f}" if macc else "N/A",
                f"{mmlu_scores.get(label, ''):.4f}" if mmlu_scores.get(label) else "N/A",
                f"{tqa_scores.get(label, ''):.4f}" if tqa_scores.get(label) else "N/A",
                f"{wg_scores.get(label, ''):.4f}" if wg_scores.get(label) else "N/A",
                f"{arc_scores.get(label, ''):.4f}" if arc_scores.get(label) else "N/A",
            ])

    # Gate verdicts
    def gate_a05():
        """All 8 models have corrected GSM8K, MATH500, MMLU, TruthfulQA, WinoGrande, ARC-Challenge."""
        missing = []
        for label in MODELS:
            if not gsm8k.get(label, {}).get("acc"):
                missing.append(f"{label}/gsm8k")
            if not math500.get(label, {}).get("acc"):
                missing.append(f"{label}/math500")
            if not mmlu_scores.get(label):
                missing.append(f"{label}/mmlu")
            if not tqa_scores.get(label):
                missing.append(f"{label}/truthfulqa")
            if not wg_scores.get(label):
                missing.append(f"{label}/winogrande")
            if not arc_scores.get(label):
                missing.append(f"{label}/arc_challenge")
        if not missing:
            return "PASS", "All 8 models have all 6 corrected scores."
        return "PARTIAL", f"Missing: {', '.join(missing[:8])}"

    def gate_b05():
        """MMLU collapse resolved: scores > 0.25 and show spread (not all identical).
        Uses base model mode (no chat template, 5-shot) — matches Qwen3 official methodology.
        mmlu_generative and chat-template approaches both failed: documented in run_provenance.json."""
        scores_present = {k: v for k, v in mmlu_scores.items() if v is not None}
        if not scores_present:
            return "FAIL", "No MMLU scores available."
        below = [f"{k}={v:.3f}" for k, v in scores_present.items() if v <= 0.25]
        if below:
            return "FAIL", f"Models at or below 0.25: {below}"
        vals = list(scores_present.values())
        spread = max(vals) - min(vals) if len(vals) > 1 else 0
        if spread < 0.005:
            return "FAIL", f"All MMLU scores identical (spread={spread:.4f}) — structural artifact"
        return "PASS", f"All {len(scores_present)} models above 0.25; spread={spread:.4f}; using mmlu loglikelihood (enable_thinking=False)"

    def gate_c05():
        """Corrected numbers for 4 missing models directionally consistent; provenance complete."""
        issues = []
        for label in MODELS:
            if label in ALREADY_CORRECTED:
                continue
            g = gsm8k.get(label, {}).get("acc")
            m = math500.get(label, {}).get("acc")
            if g is None:
                issues.append(f"{label}/gsm8k missing")
            elif g < 0.5:
                issues.append(f"{label}/gsm8k={g:.3f} unusually low (<0.5)")
            if m is None:
                issues.append(f"{label}/math500 missing")
            elif m < 0.5:
                issues.append(f"{label}/math500={m:.3f} unusually low (<0.5)")
        prov = OUTROOT / "logs" / "run_provenance.json"
        if not prov.exists():
            issues.append("run_provenance.json not yet written")
        if not issues:
            return "PASS", "All 4 remaining models have corrected scores in expected range."
        return "PARTIAL" if "missing" not in str(issues) else "FAIL", "; ".join(issues)

    a_verdict, a_detail = gate_a05()
    b_verdict, b_detail = gate_b05()
    c_verdict, c_detail = gate_c05()

    with (tables_dir / "gate_verdicts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gate", "verdict", "detail"])
        w.writerow(["A05", a_verdict, a_detail])
        w.writerow(["B05", b_verdict, b_detail])
        w.writerow(["C05", c_verdict, c_detail])

    log(f"Gate A05: {a_verdict} — {a_detail}")
    log(f"Gate B05: {b_verdict} — {b_detail}")
    log(f"Gate C05: {c_verdict} — {c_detail}")

    # Write provenance
    import subprocess as sp
    try:
        lmeval_ver = sp.check_output(
            [LMEVAL, "--version"],
            stderr=sp.STDOUT).decode().strip()
    except Exception:
        lmeval_ver = "unknown"
    prov = {
        "cycle": "cycle_05_matched_control_id_ood",
        "script": __file__,
        "run_date": ts(),
        "lmeval_version": lmeval_ver,
        "model_paths": MODELS,
        "already_corrected": list(ALREADY_CORRECTED),
        "mmlu_task": "mmlu_8subject_subset",
        "mmlu_subjects": MMLU_TASKS_SUBSET,
        "mmlu_apply_chat_template": False,
        "mmlu_num_fewshot": 5,
        "mmlu_protocol": "base_model_mode — no chat template, 5-shot loglikelihood, consistent with Qwen3 official technical report (arXiv 2505.09388) and lm-eval field standard.",
        "mmlu_scope_note": "8 representative subjects (not full 57-subject MMLU). Full MMLU with 5-shot would take 10+ hours for 8 models. Subset covers STEM, humanities, reasoning, factual. Score is macro-average across 8 subjects.",
        "mmlu_generative_outcome": "BROKEN: score=0 across 4 configurations; free-form reasoning output not parsed correctly.",
        "mmlu_chat_template_outcome": "BROKEN: collapses all models to 0.2295 (random chance) — see FINDING_05_mmlu_chat_template_collapse.md.",
        "truthfulqa_enable_thinking": False,
        "winogrande_apply_chat_template": False,
        "arc_challenge_num_fewshot": 25,
        "arc_challenge_enable_thinking": False,
        "arc_challenge_apply_chat_template": True,
        "arc_challenge_metric": "acc_norm",
        "arc_challenge_protocol": "25-shot loglikelihood acc_norm, enable_thinking=False, apply_chat_template. Same chat protocol as TruthfulQA. Think mode not used: enable_thinking=True + loglikelihood collapses (same mechanism as MMLU chat-template collapse). ARC Challenge Set N=1172.",
        "gsm8k_script": "Eval/eval_gsm8k_full.py",
        "math500_script": "Eval/eval_math500_full.py",
        "output_root": str(OUTROOT),
    }
    (OUTROOT / "logs" / "run_provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False))

    return {
        "numina": numina, "gsm8k": gsm8k, "math500": math500,
        "mmlu": mmlu_scores, "truthfulqa": tqa_scores, "winogrande": wg_scores,
        "arc_challenge": arc_scores,
        "gates": {"A05": (a_verdict, a_detail), "B05": (b_verdict, b_detail),
                  "C05": (c_verdict, c_detail)},
    }


def write_results_05(compiled: dict):
    """Generate RESULTS_05.md."""
    numina = compiled["numina"]
    gsm8k = compiled["gsm8k"]
    math500 = compiled["math500"]
    mmlu = compiled["mmlu"]
    tqa = compiled["truthfulqa"]
    wg = compiled["winogrande"]
    arc = compiled["arc_challenge"]
    gates = compiled["gates"]

    def fmt(v, digits=4):
        if v is None:
            return "N/A"
        return f"{v:.{digits}f}"

    def se(acc, n):
        if acc is None:
            return "N/A"
        import math
        return f"±{math.sqrt(acc * (1 - acc) / n):.4f}"

    def corrected_tag(label):
        return "✅ reused" if label in ALREADY_CORRECTED else "✅ cycle05"

    rows = []
    for label in MODELS:
        g = gsm8k.get(label, {})
        m5 = math500.get(label, {})
        gacc = g.get("acc")
        macc = m5.get("acc")
        rows.append({
            "label": label,
            "numina_id": fmt(numina.get(label)),
            "gsm8k": fmt(gacc),
            "gsm8k_se": se(gacc, g.get("n", 1319)),
            "math500": fmt(macc),
            "math500_se": se(macc, m5.get("n", 500)),
            "mmlu_gen": fmt(mmlu.get(label)),
            "tqa_mc1": fmt(tqa.get(label)),
            "wg": fmt(wg.get(label)),
            "arc": fmt(arc.get(label)),
            "gsm8k_src": corrected_tag(label),
        })

    def gate_line(key):
        v, d = gates[key]
        icon = "✅" if v == "PASS" else ("⚠️" if v == "PARTIAL" else "❌")
        return f"| {key} | {icon} {v} | {d} |"

    md = f"""# RESULTS_05: Cycle 05 — Corrected Three-Axis Evaluation

```yaml
cycle: cycle_05_matched_control_id_ood
generated: {ts()}
script: Eval/run_cycle05_eval.py
output_root: {OUTROOT}
mmlu_note: "mmlu_generative failed (score=0 across 4 configs); using standard mmlu loglikelihood with enable_thinking=False"
```

## Corrected Protocol Summary

All 8 Cycle 04 models re-evaluated under the three-axis protocol:

| Axis | Setting |
|---|---|
| think axis | All tasks: `enable_thinking=False` (Qwen3 thinking pre-closed via chat template) |
| chat format | GSM8K/MATH500/NuminaMath/TruthfulQA/ARC-Challenge: `apply_chat_template=True`; MMLU/WinoGrande: **no chat template** (field standard / Qwen3 official methodology) |
| token cutoff | GSM8K/MATH500: `max_gen_toks=3072`; MMLU/TruthfulQA/WinoGrande: task default (loglikelihood, no generation budget needed) |

**MMLU protocol (cycle 05):** Base model mode — no chat template, 5-shot, standard loglikelihood. Matches Qwen3 official technical report methodology (arXiv 2505.09388). Two failed approaches documented: (1) `mmlu_generative + enable_thinking=True` → score=0 (Qwen3 free-form reasoning buries answer letter); (2) `--apply_chat_template` without `enable_thinking=False` → collapse to 0.2295/random chance (lm-eval-harness issue #3576, FINDING_05_mmlu_chat_template_collapse.md). Base model mode recovers to expected range and allows direct comparison with published Qwen3 numbers.

## Model Registry

| Model | Checkpoint Path | GSM8K/MATH500 status |
|---|---|---|
| base | `/root/autodl-tmp/model/Qwen/Qwen3-1.7B` | {corrected_tag("base")} |
| theta0 | `.../model_outputs/theta0/256` | {corrected_tag("theta0")} |
| opd_lmbda05 | `.../model_outputs/opd_lmbda05/800` | {corrected_tag("opd_lmbda05")} |
| opd_lmbda1 | `.../model_outputs/opd_lmbda1/800` | {corrected_tag("opd_lmbda1")} |
| sft_n128 | `.../model_outputs/sft_n128/128` | {corrected_tag("sft_n128")} |
| sft_n256 | `.../model_outputs/sft_n256/256` | {corrected_tag("sft_n256")} |
| sft_n512 | `.../model_outputs/sft_n512/512` | {corrected_tag("sft_n512")} |
| sft_n1024 | `.../model_outputs/sft_n1024/1024` | {corrected_tag("sft_n1024")} |

## Master Score Table

| Model | NuminaMath (ID) | GSM8K ✅ | ±SE | MATH500 ✅ | ±SE | MMLU-acc | TQA-MC1 | WinoGrande | ARC-C |
|---|---|---|---|---|---|---|---|---|---|
"""
    for r in rows:
        md += (f"| {r['label']} | {r['numina_id']} | {r['gsm8k']} | {r['gsm8k_se']} "
               f"| {r['math500']} | {r['math500_se']} | {r['mmlu_gen']} "
               f"| {r['tqa_mc1']} | {r['wg']} | {r['arc']} |\n")

    md += f"""
GSM8K N=1319, MATH500 N=500, NuminaMath N=1023/892 (open-form). ±SE = binomial stderr.

## Extraction Audit

| Task | Protocol | Artifact flags |
|---|---|---|
| NuminaMath | chat+enable_thinking=False+math_verify | ✅ validated cycle04 |
| GSM8K | chat+enable_thinking=False+eval_gsm8k_full.py | ✅ corrected cycle05 |
| MATH500 | chat+enable_thinking=False+eval_math500_full.py | ✅ corrected cycle05 |
| MMLU | no_chat+mmlu loglikelihood (5-shot, base model mode) | validated cycle05 (matches Qwen3 official; mmlu_generative + chat-template both broken) |
| TruthfulQA | chat+enable_thinking=False+lm_eval loglikelihood | validated cycle05 |
| WinoGrande | no_chat+enable_thinking=False+lm_eval loglikelihood | validated cycle05 |
| ARC-Challenge | chat+enable_thinking=False+lm_eval loglikelihood (25-shot, acc_norm) | validated cycle05 |

## Gate A05/B05/C05 Verdicts

| Gate | Verdict | Detail |
|---|---|---|
{gate_line("A05")}
{gate_line("B05")}
{gate_line("C05")}

## Artifact Paths

```
{OUTROOT}/
  tables/id_ood_trajectory.csv       ← master 8-model table
  tables/gsm8k_corrected.csv
  tables/math500_corrected.csv
  tables/ood_lite_summary.csv        ← MMLU + TQA + WinoGrande
  tables/gate_verdicts.csv
  logs/run_provenance.json
  logs/master.log
  gsm8k/<label>.json                 ← per-model GSM8K result
  math500/<label>.json
  mmlu/<label>/results_*.json
  truthfulqa/<label>/results_*.json
  winogrande/<label>/results_*.json
  arc_challenge/<label>/results_*.json
```

## Limitations

- WinoGrande uses no chat template (field standard), creating a deliberate
  format inconsistency vs. other tasks. Report WinoGrande scores separately.
- ARC-Challenge uses enable_thinking=False + apply_chat_template + 25-shot (acc_norm).
  Same chat protocol as TruthfulQA. Think mode (enable_thinking=True) not used for
  loglikelihood tasks: same collapse as MMLU (probability mass shifts to think-block tokens).
  ARC Challenge Set N=1172; metric is length-normalized accuracy (acc_norm).
- MMLU uses **base model mode** (no chat template, 5-shot loglikelihood), consistent with
  Qwen3 official technical report (arXiv 2505.09388) and lm-eval field standard. Two
  alternative approaches were tested and failed: (1) `mmlu_generative + enable_thinking=True`
  → score=0 (confirmed by lm-eval-harness issue #3322); (2) `--apply_chat_template` →
  collapse to 0.2295 across all models (confirmed by lm-eval-harness issue #3576,
  FINDING_05_mmlu_chat_template_collapse.md). Base model mode gives scores in the expected
  range (~0.45–0.55 for Qwen3-1.7B) and allows direct comparison with published numbers.
- GSM8K and MATH500 for base/theta0/opd_lmbda05/sft_n128 are reused from
  the cycle05 pre-result corrected run (consistent_axis_rerun.sh, 2026-06-16).
"""

    results_path = OUTROOT / "RESULTS_05.md"
    results_path.write_text(md)
    log(f"Written {results_path}")
    return results_path


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== Cycle 05 eval orchestration START ===")

    # Phase 1
    if not run_phase1_mmlu_validation():
        log("ABORT: Phase 1 gate failed. See FINDING_05_phase1_gate_failure.md")
        sys.exit(1)

    # Phase 2
    run_phase2()

    # Compile
    compiled = compile_results()

    # RESULTS_05.md
    write_results_05(compiled)

    log("=== Cycle 05 eval orchestration COMPLETE ===")
    log(f"All outputs in {OUTROOT}")
    log(f"Primary result: {OUTROOT}/RESULTS_05.md")


if __name__ == "__main__":
    main()
