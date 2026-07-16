"""Headline metrics — paper_plan_v2 §5.3.

Four instruments, one module:

* stopping regret — UTILITY gap U_{τ*} − U_{τ_method}, not |t − t*| (v5's metric
  ignored magnitude; §5.3 / §14 changelog). The U_t curve comes from the dual-run
  forced-continuation replay protocol (§5.3): post-stop counterfactuals do not
  exist in normal eval traces, so replays supply them (replay cost billed to the
  analysis line of T4, never to any method — eval/overhead.py).
* frontier protocol — iso-metrics are undefined without it (§5.3): every method
  is swept over ITS OWN cost knob into a 3–5 point frontier; iso-accuracy cost /
  iso-cost accuracy are read by LINEAR INTERPOLATION between adjacent frontier
  points; knobless methods (B1 ReAct, oracle) are single points and EXCLUDED from
  iso-claims — this module raises `KnoblessFrontierError` rather than fabricate a
  frontier for them.
* matched lost-correct risk — the LearnStop protocol (2606.30852, §5.6): sweep
  each method's own threshold and compare cost savings at equal fractions of
  lost-correct answers (1%, 2%, 5%).
* internalization — % episodes self-terminated pre-monitor + monitor-off
  cost/accuracy deltas (§2.5/§5.3): the evidence that economics moved into the
  policy rather than being enforced at runtime.

All CPU, numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "stopping_regret",
    "KnoblessFrontierError",
    "Frontier",
    "cost_at_iso_accuracy",
    "accuracy_at_iso_cost",
    "pareto_auc",
    "MatchedRiskResult",
    "matched_lost_correct_risk",
    "internalization_metrics",
]


# ------------------------------------------------------------- stopping regret
def stopping_regret(u_curve: np.ndarray, tau_method: int, tau_star: int) -> float:
    """Stopping regret = U_{τ*} − U_{τ_method}  (§5.3 — utility gap, NOT |t − t*|).

    `u_curve` is the per-step stopping-utility curve U_1..U_T from the
    forced-continuation replay of the SAME task (dual-run protocol, §5.3);
    `tau_method` / `tau_star` are 1-based step indices (τ* from the Snell
    recursion on the replay, labels/snell.py).

    Positive regret ⇒ the method's stop lost utility vs the Snell-optimal stop.
    (Mildly negative values are possible on single paths: τ* is optimal in
    conditional expectation, not pathwise — §2.2 note (iii).)
    """
    u = np.asarray(u_curve, dtype=float)
    T = len(u)
    if T == 0:
        raise ValueError("empty utility curve")
    for name, tau in (("tau_method", tau_method), ("tau_star", tau_star)):
        if not 1 <= tau <= T:
            raise ValueError(f"{name}={tau} outside 1..{T} (1-based step index)")
    return float(u[tau_star - 1] - u[tau_method - 1])


# ----------------------------------------------------------- frontier protocol
class KnoblessFrontierError(ValueError):
    """Raised when an iso-metric is requested from a single-point (knobless)
    method — B1 ReAct / oracle are reported as single points and excluded from
    iso-claims (§5.3)."""


@dataclass
class Frontier:
    """One method's cost/accuracy frontier from its own knob sweep (§5.3).

    `points` = [(cost_dollars, accuracy), ...] — one point per knob setting
    (CASSI: inference-time λ; B2: confidence threshold; B3: trigger sensitivity;
    B4: tool-count coefficient; B5: penalty weight; B6/B7/B9: their λ).

    On construction the Pareto-efficient subset is extracted (sorted by cost,
    keep only strict accuracy improvements) — dominated knob settings never
    contribute to interpolation.
    """

    points: list[tuple[float, float]]
    method: str = ""
    # Pareto-efficient subset, filled in __post_init__ (increasing cost AND accuracy)
    eff_cost: np.ndarray = field(init=False, repr=False)
    eff_acc: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError(f"frontier '{self.method}': no points")
        pts = sorted(((float(c), float(a)) for c, a in self.points),
                     key=lambda p: (p[0], -p[1]))
        eff: list[tuple[float, float]] = []
        best = -np.inf
        for c, a in pts:
            if a > best:
                eff.append((c, a))
                best = a
        self.eff_cost = np.array([p[0] for p in eff])
        self.eff_acc = np.array([p[1] for p in eff])

    @property
    def is_single_point(self) -> bool:
        """True for knobless methods (one raw point) — excluded from iso-claims."""
        return len(self.points) < 2

    def _require_knob(self, what: str) -> None:
        if self.is_single_point:
            raise KnoblessFrontierError(
                f"'{self.method or 'method'}' is a single-point (knobless) frontier — "
                f"{what} is undefined; report it as a point and exclude from iso-claims (§5.3)."
            )

    @property
    def acc_range(self) -> tuple[float, float]:
        return float(self.eff_acc[0]), float(self.eff_acc[-1])

    @property
    def cost_range(self) -> tuple[float, float]:
        return float(self.eff_cost[0]), float(self.eff_cost[-1])


def cost_at_iso_accuracy(frontier: Frontier, acc: float) -> float | None:
    """Dollar cost the method needs to reach accuracy `acc`, by linear
    interpolation between adjacent (Pareto-efficient) frontier points (§5.3).
    Returns None when `acc` lies outside the frontier's accuracy range —
    NEVER extrapolate an iso-claim."""
    frontier._require_knob("cost-at-iso-accuracy")
    lo, hi = frontier.acc_range
    if acc < lo - 1e-12 or acc > hi + 1e-12:
        return None
    if len(frontier.eff_acc) == 1:      # sweep collapsed to one efficient point
        return float(frontier.eff_cost[0]) if abs(acc - lo) <= 1e-12 else None
    return float(np.interp(acc, frontier.eff_acc, frontier.eff_cost))


def accuracy_at_iso_cost(frontier: Frontier, cost: float) -> float | None:
    """Accuracy the method reaches at budget `cost`, by linear interpolation
    between adjacent frontier points; None outside the swept cost range (§5.3)."""
    frontier._require_knob("accuracy-at-iso-cost")
    lo, hi = frontier.cost_range
    if cost < lo - 1e-12 or cost > hi + 1e-12:
        return None
    if len(frontier.eff_cost) == 1:
        return float(frontier.eff_acc[0]) if abs(cost - lo) <= 1e-12 else None
    return float(np.interp(cost, frontier.eff_cost, frontier.eff_acc))


def pareto_auc(frontier: Frontier, cost_range: tuple[float, float]) -> float:
    """Pareto AUC (§5.3): mean achievable accuracy over `cost_range`
    (integral of the piecewise-linear frontier, normalized by the range width so
    methods integrated over the same range are directly comparable).

    Conventions outside the swept support: below the cheapest frontier point the
    method achieves nothing (0); above the most expensive point accuracy
    plateaus at the frontier maximum (a bigger budget can always run the best
    knob). State the shared `cost_range` in every caption.
    """
    frontier._require_knob("pareto-AUC")
    lo, hi = float(cost_range[0]), float(cost_range[1])
    if hi <= lo:
        raise ValueError(f"invalid cost_range {cost_range}")
    cmin, cmax = frontier.cost_range
    area = 0.0
    # segment below support: accuracy 0 → contributes nothing
    # segment inside support: trapezoid on the interpolated frontier
    seg_lo, seg_hi = max(lo, cmin), min(hi, cmax)
    if seg_hi > seg_lo:
        xs = np.unique(np.concatenate(
            [[seg_lo, seg_hi],
             frontier.eff_cost[(frontier.eff_cost > seg_lo) & (frontier.eff_cost < seg_hi)]]))
        ys = np.interp(xs, frontier.eff_cost, frontier.eff_acc)
        area += float(np.trapezoid(ys, xs))
    # segment above support: plateau at max accuracy
    if hi > cmax:
        area += float(frontier.eff_acc[-1]) * (hi - max(lo, cmax))
    return area / (hi - lo)


# ------------------------------------------------- matched lost-correct risk
@dataclass
class MatchedRiskResult:
    """Output of the LearnStop matched lost-correct-risk protocol (§5.3/§5.6)."""

    lost_fracs: np.ndarray       # fraction of full-run-correct answers lost, sorted asc
    savings: np.ndarray          # cost savings (1 − method$/full$) aligned with lost_fracs
    savings_at_risk: dict        # {risk_level: savings or None if outside swept range}


def matched_lost_correct_risk(
    per_task_correct_full: np.ndarray,
    per_task_cost_full: np.ndarray,
    per_task_correct_method: np.ndarray,
    per_task_cost_method: np.ndarray,
    risk_levels: tuple[float, ...] = (0.01, 0.02, 0.05),
) -> MatchedRiskResult:
    """LearnStop protocol (2606.30852; §5.3/§5.6 'stopping-rule fairness'):
    sweep the method's OWN threshold and read cost savings at equal fractions of
    lost-correct answers (default 1%, 2%, 5%).

    Inputs
    ------
    per_task_correct_full : (n,) 0/1 — correctness of the full (unstopped) run.
    per_task_cost_full    : (n,) dollars of the full run.
    per_task_correct_method : (k, n) 0/1 — correctness under the method at each
        of the k threshold settings of its sweep (row = one threshold).
    per_task_cost_method  : (k, n) dollars, aligned with the rows above.

    Per threshold row:
        lost-correct fraction = mean(correct_full ∧ ¬correct_method) / mean(correct_full)
        savings               = 1 − Σ cost_method / Σ cost_full
    Savings at each risk level are linearly interpolated on the sorted
    (lost_frac → savings) curve; None where the sweep never reaches that risk
    (no extrapolation — same discipline as the frontier protocol).
    """
    cf = np.asarray(per_task_correct_full, dtype=bool)
    costf = np.asarray(per_task_cost_full, dtype=float)
    cm = np.atleast_2d(np.asarray(per_task_correct_method, dtype=bool))
    costm = np.atleast_2d(np.asarray(per_task_cost_method, dtype=float))
    n = len(cf)
    if costf.shape != (n,) or cm.shape[1] != n or costm.shape != cm.shape:
        raise ValueError("shape mismatch: full arrays (n,), method sweeps (k, n)")
    base_correct = cf.mean()
    if base_correct <= 0:
        raise ValueError("full run has zero correct answers — lost-correct fraction undefined")
    full_cost = costf.sum()
    if full_cost <= 0:
        raise ValueError("full run has zero total cost")

    lost = (cf & ~cm).mean(axis=1) / base_correct
    savings = 1.0 - costm.sum(axis=1) / full_cost
    order = np.argsort(lost, kind="stable")
    lost, savings = lost[order], savings[order]

    at_risk: dict = {}
    for r in risk_levels:
        if r < lost[0] - 1e-12 or r > lost[-1] + 1e-12:
            at_risk[r] = None
        else:
            at_risk[r] = float(np.interp(r, lost, savings))
    return MatchedRiskResult(lost_fracs=lost, savings=savings, savings_at_risk=at_risk)


# ------------------------------------------------------------- internalization
def internalization_metrics(
    self_stop_flags: np.ndarray,
    cost_monitor_on: np.ndarray,
    acc_monitor_on: np.ndarray,
    cost_monitor_off: np.ndarray,
    acc_monitor_off: np.ndarray,
    baseline_cost: float | None = None,
) -> dict:
    """Internalization metrics (§2.5/§5.3) — did economics move INTO the policy?

    Inputs
    ------
    self_stop_flags  : (n,) bool — episode self-terminated (executor emitted
        ANSWER) BEFORE the monitor fired, monitor-on evaluation.
    cost/acc_monitor_on  : per-episode dollars / correctness with M_θ enforcing.
    cost/acc_monitor_off : per-episode dollars / correctness with the monitor
        DISABLED at test time (same frozen task list, §5.6).
    baseline_cost : optional mean per-episode dollars of the no-cost-signal
        reference (B1 ReAct) — enables the H5 savings-retention number
        (retention ≥ 0.7 ⇒ H5 holds; else 'partial internalization').

    Returns a flat dict (report all of it — no cherry-picking, §5.6):
      self_termination_rate, cost/acc means for both arms,
      cost_delta_monitor_off (off − on; ≈0 ⇒ fully internalized),
      acc_delta_monitor_off, and savings_retention_monitor_off
      = (baseline − off) / (baseline − on) when baseline_cost is given.
    """
    flags = np.asarray(self_stop_flags, dtype=bool)
    c_on = np.asarray(cost_monitor_on, dtype=float)
    c_off = np.asarray(cost_monitor_off, dtype=float)
    a_on = np.asarray(acc_monitor_on, dtype=float)
    a_off = np.asarray(acc_monitor_off, dtype=float)

    out = {
        "self_termination_rate": float(flags.mean()) if len(flags) else 0.0,
        "cost_monitor_on": float(c_on.mean()),
        "cost_monitor_off": float(c_off.mean()),
        "acc_monitor_on": float(a_on.mean()),
        "acc_monitor_off": float(a_off.mean()),
        "cost_delta_monitor_off": float(c_off.mean() - c_on.mean()),
        "acc_delta_monitor_off": float(a_off.mean() - a_on.mean()),
        "savings_retention_monitor_off": None,
    }
    if baseline_cost is not None:
        denom = float(baseline_cost) - out["cost_monitor_on"]
        if abs(denom) > 1e-12:
            out["savings_retention_monitor_off"] = (
                (float(baseline_cost) - out["cost_monitor_off"]) / denom
            )
    return out
