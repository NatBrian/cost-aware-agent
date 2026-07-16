"""Oracle stopping upper bound — stop at Snell τ* using ground truth (§5.2 last row).

WHAT: the headroom measurement. On forced-continuation eval replays (the §5.3
dual-run regret protocol), stop each trajectory at the Snell-optimal
τ* = min{t : U_t ≥ Cont(x_t)} computed WITH ground-truth quality — the optimal
non-anticipating stopping rule w.r.t. the collection policy's dynamics
(Hansen & Zilberstein AIJ'01 anytime-monitoring oracle; §2.2).

REIMPLEMENTS: no external paper — this is CASSI's own label machinery
(cassi.labels.snell) applied at evaluation with GT access; the classical
construct is cited, not claimed (§3 contribution 3, §4 theory row).

KILLS THE QUESTION: "headroom" — how far every implementable method is from
optimal stopping; the top line of every Pareto plot.

COST KNOB (§5.3): none. EVAL-ONLY, single point per λ label set, EXCLUDED from
iso-claims (§5.3 verbatim, together with B1). It uses ground truth (q_t enters
U_t), so it is never deployable and never trained (needs_training=False).
"""

from __future__ import annotations

import numpy as np

from cassi.labels.snell import LabelSet

COST_KNOB: str | None = None
EXCLUDED_FROM_ISO_CLAIMS = True
EVAL_ONLY = True


def tau_star(labelset: LabelSet, task_id: str, rollout_idx: int) -> int:
    """The oracle stop step for one trajectory — exactly snell_labels' τ*."""
    return int(labelset.tau_star[(task_id, rollout_idx)])


def should_stop(t: int, tau: int) -> bool:
    """Oracle decision at step t given τ*: stop iff t ≥ τ*."""
    return t >= tau


def utilities_at_tau_star(labelset: LabelSet) -> dict[tuple[str, int], float]:
    """(task_id, rollout_idx) → U_{τ*} — the utility the oracle banks per
    trajectory; the reference in the §5.3 stopping-regret metric
    (regret = U_{τ*} − U_{method's actual stop})."""
    out: dict[tuple[str, int], float] = {}
    for lab in labelset.labels:
        key = (lab.task_id, lab.rollout_idx)
        if lab.t == labelset.tau_star.get(key):
            out[key] = lab.u_t
    return out


def headroom_summary(labelset: LabelSet) -> dict:
    """The oracle's single reporting point for one λ label set: mean τ* and mean
    U_{τ*} (plus n). Feeds the Pareto plots' top line — never an iso-claim."""
    utils = utilities_at_tau_star(labelset)
    taus = [labelset.tau_star[k] for k in utils]
    return {
        "lam": labelset.lam,
        "n_trajectories": len(utils),
        "mean_tau_star": float(np.mean(taus)) if taus else 0.0,
        "mean_utility_at_tau_star": float(np.mean(list(utils.values()))) if utils else 0.0,
    }
