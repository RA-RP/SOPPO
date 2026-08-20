"""Hardened scorer for NuminaMath free-form answers.

- math: math_verify (HF) parse+verify  (LaTeX / sympy / sets / intervals)
- MCQ : robust letter handling, incl. \\text{C}, (C), multi-select (B),(C),(D)
- fallback to the v1 normalize+sympy scorer if math_verify errors.
"""
import re
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")

# Ensure component/ directory is on path so scorer.py can be found
_COMPONENT = Path(__file__).resolve().parent
if str(_COMPONENT) not in sys.path:
    sys.path.insert(0, str(_COMPONENT))

# v1 fallback
from scorer import extract_pred as _v1_extract, score as _v1_score

try:
    from math_verify import parse, verify
    from math_verify import LatexExtractionConfig, ExprExtractionConfig
    _MV = True
except Exception as e:  # pragma: no cover
    print("math_verify import failed:", e)
    _MV = False


# ---------- MCQ ----------
def _strip_mcq_wrappers(s: str) -> str:
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("(", "").replace(")", "").replace(",", "")
    s = s.replace("\\", "").replace("$", "").strip()
    return s.replace(" ", "")


def mcq_gold_letters(gold: str):
    """Return set of choice letters if gold is purely MCQ-like, else None."""
    core = _strip_mcq_wrappers(gold)
    if core and all(c in "ABCDE" for c in core.upper()):
        return set(core.upper())
    return None


def mcq_pred_letters(pred_text: str):
    if pred_text is None:
        return set()
    txt = pred_text
    if "</think>" in txt:
        txt = txt.split("</think>")[-1]
    # prefer boxed content
    bx = re.findall(r"\\boxed\s*\{([^{}]*)\}", txt)
    region = bx[-1] if bx else None
    if region is None:
        m = re.findall(r"(?:answer|option|choice)\s*(?:is|:|：)?\s*([A-E,\s\(\)]+)", txt, re.I)
        region = m[-1] if m else txt[-80:]
    return set(re.findall(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", region.upper()))


# ---------- math via math_verify ----------
def _gold_variants(gold: str):
    yield gold
    # European ; -> , inside tuples/intervals: (0;0) -> (0,0), (0;4] -> (0,4]
    semi = re.sub(r"([\(\[][^\)\]]*?);", r"\1,", gold)
    semi = re.sub(r"([\(\[][^\)\]]*?);", r"\1,", semi)
    if semi != gold:
        yield semi
        gold = semi
    # strip leading "x\in" / "x =" / "x:" binding so "x\in(0,4]" matches "(0,4]"
    stripped = re.sub(r"^[a-zA-Z]\s*(\\in|=|:)\s*", "", gold)
    if stripped != gold:
        yield stripped


def _mv_parse_pred(pred_text: str):
    try:
        return parse(pred_text, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
    except Exception:
        return None


def _mv_verify(gold: str, pred_text: str) -> bool:
    """Verify pred against EVERY gold variant; True if any matches."""
    p = _mv_parse_pred(pred_text)
    if not p:
        return False
    for base in _gold_variants(gold):
        for cand in (base, "$" + base + "$", "\\boxed{" + base + "}"):
            try:
                g = parse(cand, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
                if g and verify(g, p):
                    return True
            except Exception:
                continue
    return False


def score(pred_text: str, gold: str) -> bool:
    if pred_text is None or gold is None:
        return False
    gold = str(gold).strip()

    letters = mcq_gold_letters(gold)
    if letters is not None:
        return mcq_pred_letters(pred_text) == letters

    if _MV and _mv_verify(gold, pred_text):
        return True
    # fallback to v1
    return _v1_score(_v1_extract(pred_text), gold)


def is_mcq(gold: str) -> bool:
    return mcq_gold_letters(str(gold)) is not None


if __name__ == "__main__":
    cases = [
        ("So the answer is \\boxed{10.5}.", "{10.5}", True),
        ("\\boxed{\\frac{4\\pi}{3}}", "\\frac{4\\pi}{3}", True),
        ("the volume is \\boxed{4\\pi/3}", "\\frac{4\\pi}{3}", True),
        ("Final answer: \\boxed{144}", "144", True),
        ("...thus \\boxed{12}", "144", False),
        ("The correct option is \\boxed{C}", "\\text{C}", True),
        ("answer is C", "\\text{C}", True),
        ("I think the answer is B.", "\\text{C}", False),
        ("\\boxed{5 + \\sqrt{7}}", "5 + \\sqrt{7}", True),
        ("\\boxed{\\sqrt{7}+5}", "5 + \\sqrt{7}", True),
        ("the solution set is \\boxed{(0,4]}", "x\\in(0;4]", True),
        ("answers are B, C and D", "(B), (C), (D)", True),
        ("only A and B", "(B), (C), (D)", False),
        ("\\boxed{0}", "0", True),
        ("</think>\nThe result is \\boxed{\\frac{1}{2}}", "0.5", True),
    ]
    ok = 0
    for i, (p, g, exp) in enumerate(cases):
        got = score(p, g)
        flag = "OK " if got == exp else "FAIL"
        ok += int(got == exp)
        print(f"[{flag}] case {i}: gold={g!r} -> {got} (expect {exp})")
    print(f"\nmath_verify available: {_MV}")
    print(f"{ok}/{len(cases)} unit cases passed")
