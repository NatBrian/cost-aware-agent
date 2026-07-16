"""Running-draft machinery — paper_plan_v2 §2.6, §18.2.

The shared agent scaffold (ALL methods, training AND inference) requires every
step's output to end with one line:

    BEST ANSWER SO FAR: {one line; or EMPTY_DRAFT if none yet}

Its tokens are counted in c_t for every method. This module parses that line and
computes the draft-stability features that feed x_t (§11 progress group) — all
harness-computed string metrics, no ground truth, no executor-stated confidence.

The legacy answer-forcing probe (v5-style) is kept ONLY for validation ablation A5.
"""

from __future__ import annotations

import re

from cassi.common.schema import EMPTY_DRAFT

DRAFT_LINE_RE = re.compile(r"^\s*BEST ANSWER SO FAR:\s*(.*)\s*$", re.MULTILINE | re.IGNORECASE)

DRAFT_TEMPLATE_INSTRUCTION = (
    "At the END of every step, output exactly one line in this format:\n"
    "BEST ANSWER SO FAR: <your current best one-line answer, or EMPTY_DRAFT if you have none yet>"
)

# §18.2 — legacy probe, ablation A5 ONLY (never machinery)
LEGACY_PROBE_PROMPT = (
    "Based ONLY on the work so far, output your best final answer to the task now. "
    "One line, no explanation. If you have no answer yet, output your best guess."
)
LEGACY_PROBE_MAX_TOKENS = 64


def parse_draft(step_output: str) -> str:
    """Extract the running draft from a step's raw output. Missing/blank line → EMPTY_DRAFT.
    Multiple lines → the LAST one wins (the most recent statement of the draft)."""
    matches = DRAFT_LINE_RE.findall(step_output or "")
    if not matches:
        return EMPTY_DRAFT
    draft = matches[-1].strip()
    if not draft or draft.upper() == EMPTY_DRAFT:
        return EMPTY_DRAFT
    return draft


def normalized_edit_distance(a: str, b: str) -> float:
    """Levenshtein distance / max length, in [0,1]. Two EMPTY_DRAFTs are identical (0.0)."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    # O(len(a)*len(b)) dynamic program — drafts are one-liners, this is cheap.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def draft_stability_features(draft_history: list[str]) -> dict:
    """x_t progress features from the drafts seen so far (§11): how long the draft
    has been stable, and its recent edit distances. Harness-computed only.

    NOTE the honest caveat from §2.4: the executor authors the draft, so freezing a
    wrong draft can fake stability WITHIN an iteration; the defense is cross-iteration
    (label refresh) plus the V̂-vs-reward divergence diagnostic — not this function.
    """
    if not draft_history:
        return {"steps_since_draft_changed": 0, "draft_edit_distance_last3": [0.0, 0.0, 0.0]}
    steps_since = 0
    last = draft_history[-1]
    for d in reversed(draft_history[:-1]):
        if d == last:
            steps_since += 1
        else:
            break
    # edit distances of the last 3 consecutive-draft transitions, older→newer,
    # left-padded with 0.0 when fewer transitions exist yet
    dists = [
        normalized_edit_distance(draft_history[i - 1], draft_history[i])
        for i in range(max(1, len(draft_history) - 3), len(draft_history))
    ]
    dists = [0.0] * (3 - len(dists)) + dists
    return {"steps_since_draft_changed": steps_since, "draft_edit_distance_last3": dists[-3:]}


def retrieval_overlap(recent_docids: list[set]) -> float:
    """Jaccard overlap of the last retrieval's doc ids vs the union of the prior ones —
    high overlap = new searches are finding nothing new (§11 progress group)."""
    if len(recent_docids) < 2 or not recent_docids[-1]:
        return 0.0
    prior = set().union(*recent_docids[:-1]) if recent_docids[:-1] else set()
    if not prior:
        return 0.0
    inter = len(recent_docids[-1] & prior)
    union = len(recent_docids[-1] | prior)
    return inter / union if union else 0.0
