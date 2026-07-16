"""B3 — SupervisorAgent-style training-free monitor (§5.2 row B3).

WHAT: an inference-time monitor over the (frozen) executor that force-stops the
episode when redundancy/budget triggers fire. The policy never changes — economics
stay OUTSIDE the weights, which is exactly what CASSI's internalization claim is
measured against (E1/E2).

REIMPLEMENTS: SupervisorAgent (2510.26585, ICLR'26 — the "-29.7% GAIA tokens at
parity" bar). ADAPTATION DISCLOSED (§5.2 requires this in the appendix):
SupervisorAgent supervises a MULTI-agent system with an LLM supervisor; our
single-agent envs (Search-R1 QA, ALFWorld) have no sub-agents to arbitrate, so we
reimplement its documented trigger PROTOCOL as harness-computed heuristics over
the §11 features: (i) repeated-tool-call detection, (ii) no-new-information via
retrieval overlap, (iii) budget threshold. This is a protocol adaptation, not a
faithful port — disclosed here and in the paper appendix.

KILLS THE QUESTION: "does a strong training-free runtime monitor already close
the gap?" — the published inference-time-control bar CASSI must beat.

COST KNOB (§5.3): trigger sensitivity s in [0,1]. Higher s => triggers fire
earlier (fewer repeats needed, lower overlap bar, earlier budget cut) => cheaper,
riskier frontier points. Sweep s over 3-5 values for the frontier.

BILLING (§5.3 symmetry): these triggers are pure harness arithmetic over already-
logged features — auxiliary inference cost ~= $0 (disclosed; unlike B2's probes,
there is no LLM call to bill). Any future LLM-judge trigger variant must be billed
under the shared price map.

TRAINING: none (needs_training=False).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from cassi.common.schema import Step

COST_KNOB = "trigger_sensitivity"

# Trigger names (returned in MonitorDecision.triggers)
REPEATED_TOOL_CALLS = "repeated_tool_calls"
NO_NEW_INFORMATION = "no_new_information"
BUDGET_THRESHOLD = "budget_threshold"


@dataclass
class MonitorDecision:
    stop: bool
    triggers: list[str]          # which heuristics fired (empty when stop=False)
    sensitivity: float


def trigger_thresholds(sensitivity: float) -> dict:
    """Map the scalar knob s in [0,1] to the three trigger thresholds.

    s=0 (laziest monitor): 4 identical calls / 95% overlap / 100% of wallet.
    s=1 (most aggressive): 2 identical calls / 50% overlap / 60% of wallet.
    Linear in between — one dial moves all triggers together, matching §5.3's
    "trigger sensitivity" (a single frontier knob, not three)."""
    s = float(min(1.0, max(0.0, sensitivity)))
    return {
        "min_repeats": int(round(4 - 2 * s)),          # 4 -> 2
        "overlap_threshold": 0.95 - 0.45 * s,          # 0.95 -> 0.50
        "budget_pct_threshold": 1.0 - 0.4 * s,         # 1.00 -> 0.60 of allowance spent
    }


def repeated_tool_call_trigger(steps: Sequence[Step], min_repeats: int) -> bool:
    """Fire when some tool call signature (action, observation digest) has occurred
    >= min_repeats times — the agent is re-running a call it already ran."""
    sigs = [(s.a, s.o) for s in steps if s.a == "tool_call"]
    if not sigs:
        return False
    return max(Counter(sigs).values()) >= max(2, min_repeats)


def no_new_information_trigger(
    steps: Sequence[Step], overlap_threshold: float, window: int = 2
) -> bool:
    """Fire when retrieval overlap (§11 progress feature: last retrieval vs prior
    ones) has stayed >= overlap_threshold for the last `window` steps — searches
    keep returning documents already seen."""
    if len(steps) < window:
        return False
    recent = [s.x.retrieval_overlap_last3 for s in steps[-window:]]
    return all(o >= overlap_threshold for o in recent)


def budget_trigger(dollars_pct: float, pct_threshold: float) -> bool:
    """Fire when the fraction of the allowance already spent reaches the threshold."""
    return dollars_pct >= pct_threshold


def should_stop(steps: Sequence[Step], sensitivity: float) -> MonitorDecision:
    """The B3 monitor decision on a trajectory prefix (called each step at
    inference; NEVER during training rollouts — B3 trains nothing).

    Any single trigger firing stops the episode; the fired trigger names are
    returned for the qualitative-analysis appendix."""
    if not steps:
        return MonitorDecision(stop=False, triggers=[], sensitivity=sensitivity)
    thr = trigger_thresholds(sensitivity)
    fired: list[str] = []
    if repeated_tool_call_trigger(steps, thr["min_repeats"]):
        fired.append(REPEATED_TOOL_CALLS)
    if no_new_information_trigger(steps, thr["overlap_threshold"]):
        fired.append(NO_NEW_INFORMATION)
    if budget_trigger(steps[-1].x.dollars_pct, thr["budget_pct_threshold"]):
        fired.append(BUDGET_THRESHOLD)
    return MonitorDecision(stop=bool(fired), triggers=fired, sensitivity=sensitivity)
