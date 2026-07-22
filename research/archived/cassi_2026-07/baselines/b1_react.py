"""B1 — plain ReAct, no cost signal (paper_plan_v2 §5.2 row B1).

WHAT: the untrained base executor (Qwen3.5-9B) running the shared agent scaffold
(ReAct loop + mandatory running-draft line, §2.6/§18.2) with NO cost signal of any
kind — no monitor, no penalty, no shaping. Runs until it emits ANSWER or hits T_max.

REIMPLEMENTS: ReAct (Yao et al., 2210.03629) as instantiated by this repo's shared
agent template — the scaffold itself is identical for every method (§2.6), so B1
differs from the others only by the ABSENCE of any economics.

KILLS THE QUESTION: "how much slack exists" — the lower bound every cost-aware
method must beat on cost, and the accuracy reference for iso-accuracy readings.

COST KNOB (§5.3): none. B1 is a SINGLE operating point and is EXCLUDED from
iso-accuracy / iso-cost claims — §5.3 verbatim: "methods with no knob (B1 ReAct,
oracle) are reported as single points and excluded from iso-claims."

TRAINING: none (needs_training=False). B1 is pure inference of the base model;
there is no reward function to optimize. `reward` below exists only so evaluation
code can score B1 trajectories in the same quality units as everyone's Q_tau — it
is NOT a training signal.
"""

from __future__ import annotations

COST_KNOB: str | None = None
EXCLUDED_FROM_ISO_CLAIMS = True


def reward(terminal_quality: float) -> float:
    """Quality-only score of a finished B1 trajectory (Q_tau; §2.4's quality measure,
    EM headline / F1 variant on QA, success on ALFWorld). No cost term — B1 has no
    cost signal by definition. Evaluation bookkeeping only, never trained on."""
    return float(terminal_quality)
