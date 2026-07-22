"""B2 — zero-training self-eval prompt + calibrated confidence exit (§5.2 row B2).

WHAT: a training-free stopping rule. Every k-th step the executor is probed with a
short self-evaluation prompt; the probe's scalar confidence is compared to a
threshold calibrated on dev, and the episode stops when confidence >= threshold.
Per §5.2 this is the "dangerous baseline" — LearnStop (2606.30852) shows calibrated
probes can beat learned stoppers; §1.2's caveat: probes know "am I right", not
"is the next step worth its price".

REIMPLEMENTS: Dynasor-style scalar probing (2412.20993). ADAPTATION DISCLOSED:
Dynasor's original signal is probe-in-the-middle answer consistency for CoT; our
agent adaptation asks for the current best answer plus an explicit 0-100 certainty
scalar (the "calibrated confidence exit" named in §5.2), calibrated on dev.

KILLS THE QUESTION: "why not just ask/probe?" (§1.2, reviewer FAQ §15).

COST KNOB (§5.3): the confidence threshold — sweeping it traces B2's frontier
(higher threshold => later stops => more cost, more accuracy).

BILLING SYMMETRY (§5.3): probe calls are auxiliary inference and MUST be billed
under the same price map as everything else. `bill_probe` returns token counts and
dollars so the harness adds them to c_t; nothing here is free.

TRAINING: none (needs_training=False) — threshold calibration on dev is a scan,
not a gradient step.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from cassi.budget.cost import token_cost

COST_KNOB = "confidence_threshold"

# The self-eval probe (auxiliary inference — billed; §5.3 billing symmetry).
PROBE_PROMPT = (
    "Pause your work. Based ONLY on the steps so far:\n"
    "1. On one line, state your current best final answer to the task "
    "(or EMPTY_DRAFT if you have none yet).\n"
    "2. On the next line, output exactly: CONFIDENCE: <integer 0-100> — your "
    "calibrated probability that this answer is correct and further work would "
    "not change it.\n"
    "Output nothing else."
)
PROBE_MAX_OUTPUT_TOKENS = 64

_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(\d{1,3})", re.IGNORECASE)

# Sentinel: no threshold on dev met the calibration target => never stop early.
NEVER_STOP = math.inf


def parse_confidence(probe_output: str) -> float | None:
    """Extract the 0-100 scalar from a probe response; None if absent/unparseable
    (an unparseable probe never triggers a stop — fail-open to CONTINUE)."""
    m = _CONFIDENCE_RE.search(probe_output or "")
    if m is None:
        return None
    return float(min(100, max(0, int(m.group(1)))))


def should_stop(confidence: float | None, threshold: float) -> bool:
    """The B2 exit rule: stop iff the probe's confidence reaches the calibrated
    threshold. None (unparseable probe) never stops."""
    if confidence is None:
        return False
    return confidence >= threshold


def calibrate_threshold(
    confidences: np.ndarray | list[float],
    correct: np.ndarray | list[bool],
    target_precision: float = 0.9,
) -> float:
    """Calibrate the exit threshold on dev (§5.2 "calibrated confidence exit").

    Given dev probes — confidence c_i and whether the probed answer was correct —
    return the SMALLEST threshold whose stop-set precision P(correct | c >= thr)
    reaches `target_precision`. Smallest-qualifying keeps the most cost savings at
    the required risk level (LearnStop matched lost-correct-risk logic, §5.6).

    Monotone by construction: the qualifying set {thr : precision(thr) >= p}
    shrinks as p grows, so the returned minimum is non-decreasing in
    `target_precision` (tested in tests/test_baselines_cpu.py). Sweeping
    `target_precision` (equivalently the returned threshold) is the §5.3 frontier.

    Returns NEVER_STOP (inf) when no threshold qualifies — B2 then degenerates to
    B1's single point, which is itself a reportable frontier endpoint.
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    if conf.shape != corr.shape or conf.size == 0:
        raise ValueError("confidences and correct must be same-length, non-empty")
    for thr in np.unique(conf):                       # ascending scan
        sel = conf >= thr
        if sel.any() and corr[sel].mean() >= target_precision:
            return float(thr)
    return NEVER_STOP


@dataclass
class ProbeBill:
    """One probe call's cost — added to c_t like any other spend (§5.3)."""
    input_tokens: int
    output_tokens: int
    dollars: float


def bill_probe(input_tokens: int, output_tokens: int, model: str | None = None) -> ProbeBill:
    """Price one probe call under the shared price map (cassi.budget.cost) —
    billing symmetry (§5.3): B2 pays for its own auxiliary inference."""
    return ProbeBill(
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        dollars=token_cost(int(input_tokens), int(output_tokens), model=model),
    )


def estimate_tokens(text: str) -> int:
    """~4 chars/token heuristic — CPU-test fallback when no tokenizer is wired.
    Production billing uses the serving stack's real token counts."""
    return max(1, len(text) // 4)
