"""Per-step quality q_t — paper_plan_v2 §2.1, §2.6.

QA: F1/EM of the step-t RUNNING DRAFT vs gold — a free string comparison at
collection time (the draft is template-emitted every step, §18.2).
ALFWorld: subgoal-completion fraction read directly from the environment state.

Label machinery only — q_t never enters x_t (§2.1 hard requirement).
Metrics follow the SQuAD/HotpotQA normalization convention so numbers are
commensurable with the Search-R1/OTC-PO line.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from cassi.common.schema import EMPTY_DRAFT


def normalize_answer(s: str) -> str:
    """Lower, strip punctuation/articles/extra whitespace (SQuAD convention)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    if prediction == EMPTY_DRAFT or not prediction.strip():
        return 0.0
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    if prediction == EMPTY_DRAFT or not prediction.strip():
        return 0.0
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_quality(draft: str, gold: str, metric: str = "f1") -> float:
    """q_t for QA domains. Headline terminal Q_τ uses EM; per-step labels use F1 (§2.4/§17)."""
    if metric == "f1":
        return f1_score(draft, gold)
    if metric == "em":
        return exact_match(draft, gold)
    raise ValueError(f"unknown QA metric {metric!r}")


def alfworld_quality(subgoals_done: int, subgoals_total: int) -> float:
    """q_t for ALFWorld = env subgoal-completion fraction (§2.1) — zero label cost."""
    if subgoals_total <= 0:
        return 0.0
    return max(0.0, min(1.0, subgoals_done / subgoals_total))
