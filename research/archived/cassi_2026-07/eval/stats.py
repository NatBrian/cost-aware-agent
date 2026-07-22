"""Statistical & reporting protocol — paper_plan_v2 §5.6.

The rules this module enforces (every caller inherits them):

* uncertainty = bootstrap over TEST INSTANCES (10,000 resamples, 95% CI on every
  reported metric); seed variability is reported separately (mean ± s.d.).
* comparisons are within-instance paired — paired t-test AND Wilcoxon
  signed-rank, both reported; **Wilcoxon governs when normality fails** (cost
  distributions are heavy-tailed).
* multiple comparisons: Holm–Bonferroni across the 9 baselines within each
  domain (comparison family stated explicitly in the appendix).
* Pareto dominance: resample instances, recompute both frontiers, report the
  fraction of resamples in which A dominates at every shared accuracy level.
* effect sizes: Cohen's d for dollar-cost deltas; absolute risk difference for
  accuracy.
* SMALL-N POLICY (hard rule): hypothesis tests run ONLY on sets with n ≥ 500.
  GAIA-103 / Bamboogle-125 get point estimates + bootstrap CIs, labeled
  'transfer indicators, not hypothesis tests'. `paired_tests` enforces this by
  raising `SmallSampleError` — do not work around it, report a CI instead.

All CPU: numpy + scipy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from cassi.eval.metrics import Frontier, KnoblessFrontierError, cost_at_iso_accuracy

__all__ = [
    "SMALL_N_THRESHOLD",
    "SmallSampleError",
    "small_n_guard",
    "BootstrapCI",
    "bootstrap_ci",
    "paired_tests",
    "HolmResult",
    "holm_bonferroni",
    "pareto_dominance_bootstrap",
    "effect_sizes",
]

SMALL_N_THRESHOLD = 500   # §5.6 small-n policy


# --------------------------------------------------------------- small-n guard
class SmallSampleError(ValueError):
    """Hypothesis test requested on n < 500 (§5.6). Report point estimate +
    bootstrap CI instead, labeled 'transfer indicator, not hypothesis test'."""


def small_n_guard(n: int, threshold: int = SMALL_N_THRESHOLD) -> bool:
    """§5.6 small-n policy: True iff hypothesis tests are allowed at this n.
    Every test-running code path must call this (paired_tests does) and fall
    back to point-estimate + bootstrap CI when it returns False."""
    return int(n) >= int(threshold)


# ------------------------------------------------------------------- bootstrap
@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    n: int
    n_boot: int
    ci: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.point, self.lo, self.hi


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 95.0,
    seed: int = 0,
    statistic=np.mean,
) -> BootstrapCI:
    """Percentile bootstrap over test instances (§5.6): resample `values` with
    replacement `n_boot` times, return the point estimate and the `ci`%
    percentile interval of the statistic. Allowed at ANY n (unlike hypothesis
    tests) — it is exactly what §5.6 prescribes for GAIA-103/Bamboogle-125."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or len(v) == 0:
        raise ValueError("values must be a non-empty 1-D array")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = statistic(v[idx], axis=1)
    alpha = (100.0 - ci) / 2.0
    return BootstrapCI(
        point=float(statistic(v)),
        lo=float(np.percentile(boots, alpha)),
        hi=float(np.percentile(boots, 100.0 - alpha)),
        n=len(v), n_boot=n_boot, ci=ci,
    )


# ---------------------------------------------------------------- paired tests
def paired_tests(a: np.ndarray, b: np.ndarray, *,
                 normality_alpha: float = 0.05,
                 small_n_threshold: int = SMALL_N_THRESHOLD) -> dict:
    """Within-instance paired comparison (§5.6): paired t-test AND Wilcoxon
    signed-rank, both returned. `governing` tells the caller which p-value to
    quote: **Wilcoxon governs when the paired differences fail normality**
    (D'Agostino–Pearson at `normality_alpha`) — cost distributions are
    heavy-tailed, so expect Wilcoxon to govern on dollar metrics.

    Enforces the §5.6 small-n policy: raises SmallSampleError when n < 500 —
    use bootstrap_ci and label the number a transfer indicator instead.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays over the SAME task instances")
    n = len(a)
    if not small_n_guard(n, small_n_threshold):
        raise SmallSampleError(
            f"n={n} < {small_n_threshold}: hypothesis tests disallowed (§5.6 small-n "
            "policy). Report point estimate + bootstrap CI, labeled "
            "'transfer indicator, not hypothesis test'."
        )
    diff = a - b
    if np.std(diff) == 0.0:          # constant differences: tests are degenerate
        t_stat, t_p = 0.0, 1.0
        w_stat, w_p = 0.0, 1.0
        normal_p = 1.0
    else:
        t_stat, t_p = sps.ttest_rel(a, b)
        try:
            w_stat, w_p = sps.wilcoxon(a, b)
        except ValueError:           # e.g. all nonzero diffs tied at zero after dropping
            w_stat, w_p = 0.0, 1.0
        _, normal_p = sps.normaltest(diff)
    governing = "t" if normal_p >= normality_alpha else "wilcoxon"
    return {
        "n": n,
        "mean_diff": float(diff.mean()),
        "t_stat": float(t_stat), "t_pvalue": float(t_p),
        "wilcoxon_stat": float(w_stat), "wilcoxon_pvalue": float(w_p),
        "normality_pvalue": float(normal_p),
        "governing": governing,
        "governing_pvalue": float(t_p if governing == "t" else w_p),
        "note": "Wilcoxon governs when normality fails (§5.6); both reported.",
    }


# -------------------------------------------------------------- Holm–Bonferroni
@dataclass
class HolmResult:
    reject: list[bool]        # original order
    adjusted: list[float]     # Holm-adjusted p-values, original order
    alpha: float


def holm_bonferroni(pvals: list[float], alpha: float = 0.05) -> HolmResult:
    """Holm–Bonferroni step-down correction (§5.6 — replaces v1's plain
    Bonferroni) across the baseline family within each domain. Returns per-test
    reject decisions and Holm-adjusted p-values, both in the input order."""
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("pvals must be a non-empty 1-D sequence")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be in [0, 1]")
    m = len(p)
    order = np.argsort(p, kind="stable")

    reject = np.zeros(m, dtype=bool)
    adjusted = np.zeros(m, dtype=float)
    running_max = 0.0
    alive = True
    for rank, i in enumerate(order):            # rank 0 = smallest p
        adj = min(1.0, (m - rank) * p[i])
        running_max = max(running_max, adj)     # enforce monotonicity
        adjusted[i] = running_max
        if alive and p[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            alive = False                       # step-down stops at first failure
    return HolmResult(reject=reject.tolist(), adjusted=adjusted.tolist(), alpha=alpha)


# ------------------------------------------------------ Pareto dominance boot
def pareto_dominance_bootstrap(
    costs_a: np.ndarray, correct_a: np.ndarray,
    costs_b: np.ndarray, correct_b: np.ndarray,
    n_boot: int = 1000, seed: int = 0, n_grid: int = 25,
) -> float:
    """Pareto-dominance bootstrap (§5.6): resample test instances, recompute
    BOTH frontiers, return the fraction of resamples in which A's frontier
    dominates B's at EVERY shared accuracy level.

    Inputs are per-knob × per-instance matrices on the SAME frozen task list
    (§5.6 within-instance pairing):
        costs_a   : (k_a, n) dollars per instance at each of A's knob settings
        correct_a : (k_a, n) 0/1 correctness, aligned
        costs_b / correct_b : same for B, (k_b, n)

    Per resample: frontier points = (mean cost, mean accuracy) per knob;
    dominance is checked on an `n_grid` accuracy grid spanning the shared
    accuracy range (strictly lower A-cost required at every level). Resamples
    with no shared accuracy range, or where a frontier degenerates, count as
    NON-dominance (conservative).
    """
    ca, ra = np.atleast_2d(np.asarray(costs_a, float)), np.atleast_2d(np.asarray(correct_a, float))
    cb, rb = np.atleast_2d(np.asarray(costs_b, float)), np.atleast_2d(np.asarray(correct_b, float))
    n = ca.shape[1]
    if ra.shape != ca.shape or rb.shape != cb.shape or cb.shape[1] != n:
        raise ValueError("per-knob matrices must be (k, n) with the same n instances")
    rng = np.random.default_rng(seed)

    wins = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)        # paired resample: same instances for A and B
        fa = Frontier([(ca[k, idx].mean(), ra[k, idx].mean()) for k in range(ca.shape[0])], "A")
        fb = Frontier([(cb[k, idx].mean(), rb[k, idx].mean()) for k in range(cb.shape[0])], "B")
        lo = max(fa.acc_range[0], fb.acc_range[0])
        hi = min(fa.acc_range[1], fb.acc_range[1])
        if hi <= lo:
            continue                            # no shared accuracy level → non-dominance
        try:
            dominated = True
            for acc in np.linspace(lo, hi, n_grid):
                qa = cost_at_iso_accuracy(fa, acc)
                qb = cost_at_iso_accuracy(fb, acc)
                if qa is None or qb is None or not qa < qb:
                    dominated = False
                    break
        except KnoblessFrontierError:
            dominated = False                   # degenerate resample → conservative
        wins += dominated
    return wins / n_boot


# ---------------------------------------------------------------- effect sizes
def effect_sizes(a: np.ndarray, b: np.ndarray) -> dict:
    """Effect sizes (§5.6): Cohen's d (pooled s.d., for dollar-cost deltas) and
    the absolute risk difference mean(a) − mean(b) (for 0/1 accuracy arrays this
    is the difference in proportions; ×100 = percentage points)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("need at least 2 observations per group")
    diff = float(a.mean() - b.mean())
    pooled_var = (
        (len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)
    ) / (len(a) + len(b) - 2)
    pooled_sd = float(np.sqrt(pooled_var))
    if pooled_sd == 0.0:
        d = 0.0 if diff == 0.0 else float(np.sign(diff) * np.inf)
    else:
        d = diff / pooled_sd
    return {"cohens_d": d, "abs_risk_difference": diff}
