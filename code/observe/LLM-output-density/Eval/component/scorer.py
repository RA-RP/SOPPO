"""Floor-check scorer for NuminaMath-style free-form math answers.

Not a production grader. Goal: distinguish "floored (~0-5%)" from "meaningful (>=20%)".
Uses normalization + sympy equivalence (no math_verify/antlr4 available).
"""
import re
import sympy
from sympy import simplify, sympify, Rational


# ---------- answer extraction from model output ----------
def _last_boxed(text: str):
    """Return content of the LAST \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth = 0
    start = i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:j]
    return None


def extract_pred(text: str):
    """Best-effort final-answer extraction."""
    if text is None:
        return None
    # drop chain-of-thought
    if "</think>" in text:
        text = text.split("</think>")[-1]
    b = _last_boxed(text)
    if b is not None:
        return b.strip()
    # "answer is X" / "answer: X" / "the answer is"
    m = re.findall(r"(?:answer|answ?er is|final answer)\s*[:：=]?\s*\$?([^\n$]+)", text, flags=re.I)
    if m:
        return m[-1].strip().rstrip(".")
    # last non-empty line, stripped
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return lines[-1]
    return None


# ---------- normalization ----------
_MCQ = re.compile(r"^[A-E]$")


def is_mcq(gold: str) -> bool:
    return bool(_MCQ.fullmatch(gold.strip()))


def extract_mcq_letter(pred: str):
    if pred is None:
        return None
    # explicit single letter, or letter inside boxed/parens
    m = re.findall(r"\b([A-E])\b", pred.strip())
    if m:
        return m[-1]
    return None


def _latex_to_expr(s: str) -> str:
    s = s.strip()
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    s = s.replace("$", "").replace("\\,", "").replace("\\;", "")
    s = re.sub(r"\\d?frac\s*{([^{}]+)}\s*{([^{}]+)}", r"(\1)/(\2)", s)
    s = re.sub(r"\\d?frac\s*(\d)\s*(\d)", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\s*{([^{}]+)}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*(\w)", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "")
    s = re.sub(r"\\[a-zA-Z]+", "", s)            # drop remaining latex cmds
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("\\", "").replace(" ", "")
    # implicit multiplication (before ^->**): 4pi->4*pi, 2x->2*x, 2(->2*(, )(->)*(
    s = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", s)
    s = re.sub(r"(\))([a-zA-Z(])", r"\1*\2", s)
    s = s.replace("^", "**")
    return s


def _norm_str(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = s.replace(" ", "").replace("\\,", "").rstrip(".")
    s = s.replace("\\(", "").replace("\\)", "")
    return s.lower()


def score(pred: str, gold: str) -> bool:
    if pred is None or gold is None:
        return False
    gold = str(gold).strip()
    pred = str(pred).strip()
    # MCQ
    if is_mcq(gold):
        return extract_mcq_letter(pred) == gold.strip().upper()
    # exact normalized string
    if _norm_str(pred) == _norm_str(gold):
        return True
    # sympy equivalence
    try:
        pe = sympify(_latex_to_expr(pred), rational=True)
        ge = sympify(_latex_to_expr(gold), rational=True)
        if pe == ge:
            return True
        diff = simplify(pe - ge)
        if diff == 0:
            return True
        # numeric fallback
        if abs(float(pe) - float(ge)) < 1e-6:
            return True
    except Exception:
        pass
    return False


if __name__ == "__main__":
    # offline unit tests (no GPU)
    cases = [
        # (pred_text, gold, expected)
        ("So the answer is \\boxed{10.5}.", "{10.5}", True),
        ("\\boxed{\\frac{4\\pi}{3}}", "\\frac{4\\pi}{3}", True),
        ("the volume is \\boxed{4\\pi/3}", "\\frac{4\\pi}{3}", True),
        ("Final answer: \\boxed{144}", "144", True),
        ("...thus \\boxed{12}", "144", False),
        ("The correct option is \\boxed{D}", "D", True),
        ("I think the answer is C.", "B", False),
        ("answer is B", "B", True),
        ("\\boxed{5 + \\sqrt{7}}", "5 + \\sqrt{7}", True),
        ("\\boxed{\\sqrt{7}+5}", "5 + \\sqrt{7}", True),
        ("\\boxed{0}", "0", True),
        ("blah blah no answer here", "42", False),
        ("</think>\nThe result is \\boxed{\\frac{1}{2}}", "0.5", True),
    ]
    ok = 0
    for i, (p, g, exp) in enumerate(cases):
        pred = extract_pred(p)
        got = score(pred, g)
        flag = "OK " if got == exp else "FAIL"
        if got == exp:
            ok += 1
        print(f"[{flag}] case {i}: extracted={pred!r} gold={g!r} -> {got} (expect {exp})")
    print(f"\n{ok}/{len(cases)} unit cases passed")
