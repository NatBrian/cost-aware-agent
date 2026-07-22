"""SQuAD-style answer scoring (shared by collection draft-scoring and F6 eval).

One implementation for every arm and every stage — F4/F6 byte-reproduction
depends on this being the single scorer in the codebase.
"""

import re
import string
from collections import Counter


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred: str, golds: list[str]) -> float:
    p = normalize(pred)
    return float(any(p == normalize(g) for g in golds))


def f1(pred: str, golds: list[str]) -> float:
    p_toks = normalize(pred).split()
    best = 0.0
    for g in golds:
        g_toks = normalize(g).split()
        if not p_toks or not g_toks:
            best = max(best, float(p_toks == g_toks))
            continue
        common = Counter(p_toks) & Counter(g_toks)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        prec, rec = overlap / len(p_toks), overlap / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best
