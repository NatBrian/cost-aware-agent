"""Algorithm 1 — empirical Snell-envelope stopping labels (paper_plan_v2 §2.2, §10).

Longstaff–Schwartz backward recursion with cross-sectional regression:

    V_T(x_T)  = U_T
    V_t(x_t)  = max( U_t ,  Ê[ V_{t+1} | x_t ] )      t = T−1 … 1
    Cont(x_t) = Ê[ V_{t+1} | x_t ]
    Δ*_t      = Cont(x_t) − U_t        (>0 ⇒ continue)
    τ*        = min{ t : U_t ≥ Cont(x_t) }

Ê is fit ACROSS all trajectories at each t (the G=8 GRPO rollouts give the
cross-section), so labels reflect the conditional expectation over continuations —
max-of-mean, not the prophet-biased mean-of-max (v5's argmax label; kept only as a
comparison arm for the E4 label study, `prophet_labels`).

Conventions (§2.2 precision notes):
  (i)  decision grid = every step on both domains;
  (ii) at t = T, Cont is undefined and a*_T := STOP by construction (Δ*_T := 0);
  (iii) τ* is optimal w.r.t. the COLLECTION policy's continuation dynamics — the
        loop (§2.7) re-solves it as the policy improves.

Output per step, per λ:  (a*_t, tanh(Δ*_t / s) ∈ [−1,1], V*_t unnormalized)
V* exists because the executor's shaping potential (§2.4) needs a value estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cassi.budget.cost import stopping_utilities
from cassi.common.schema import Trajectory
from cassi.stopper.features import feature_vector

MIN_CROSS_SECTION = 30   # below this, Ê_t falls back to the pooled mean of V_{t+1}


# ------------------------------------------------------------------ regressor
class _Regressor:
    """Cross-sectional Ê[V_{t+1}|x_t]: LightGBM by default, sklearn MLP fallback,
    constant-mean fallback when the step-t cross-section is too small (§10 Alg.1:
    'gradient-boosted trees or 2-layer MLP on x-features; NOT the stopper itself,
    to avoid label-model coupling')."""

    def __init__(self, params: dict | None = None):
        self.params = params or {
            "n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05,
            "min_child_samples": 20, "verbosity": -1,
        }
        self._model = None
        self._mean = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Regressor":
        self._mean = float(np.mean(y))
        if len(y) < MIN_CROSS_SECTION:
            self._model = None          # constant predictor
            return self
        try:
            import lightgbm as lgb
            self._model = lgb.LGBMRegressor(**self.params)
            self._model.fit(X, y)
        except ImportError:
            from sklearn.neural_network import MLPRegressor
            self._model = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=500,
                                        early_stopping=True, random_state=0)
            self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.full(len(X), self._mean)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            return np.asarray(self._model.predict(X), dtype=float)


# -------------------------------------------------------------------- outputs
@dataclass
class StepLabel:
    task_id: str
    group_id: str
    rollout_idx: int
    t: int                  # 1-based step index
    a_star: str             # STOP | CONTINUE
    delta_raw: float        # Cont − U (quality units); 0.0 at t=T by convention
    delta_norm: float       # tanh(delta_raw / s) ∈ [−1,1]
    v_star: float           # V_t — UNNORMALIZED (the shaping-potential label)
    u_t: float              # kept for regret evaluation & QC
    lam: float


@dataclass
class LabelSet:
    lam: float
    domain: str
    scale_s: float                     # tanh scale, fit once per domain then FROZEN (§17)
    labels: list[StepLabel] = field(default_factory=list)
    tau_star: dict = field(default_factory=dict)     # (task_id, rollout_idx) -> τ*
    backup_residuals: list[float] = field(default_factory=list)  # held-out |Ê−V| per t (§5.3 diagnostic)


# ---------------------------------------------------------------- Algorithm 1
def _utilities(traj: Trajectory, lam: float, median_pilot_spend: float,
               rule_table_off: bool) -> np.ndarray:
    return stopping_utilities(
        [s.q for s in traj.steps], [s.c for s in traj.steps], [s.tier for s in traj.steps],
        lam, median_pilot_spend, rule_table_off=rule_table_off,
    )


def fit_delta_scale(delta_raw: np.ndarray) -> float:
    """Tanh scale s per domain (Alg.1 line 7): the 90th percentile of |Δ*| on round-0
    data, so typical margins land in tanh's responsive range. Frozen after fitting."""
    nonzero = np.abs(delta_raw[np.abs(delta_raw) > 1e-12])
    if len(nonzero) == 0:
        return 1.0
    return float(max(np.percentile(nonzero, 90), 1e-6))


def snell_labels(
    trajectories: list[Trajectory], lam: float, median_pilot_spend: float,
    *, rule_table_off: bool = False, scale_s: float | None = None,
    regressor_params: dict | None = None, holdout_frac: float = 0.1, seed: int = 0,
) -> LabelSet:
    """Run Algorithm 1 over one collection round (one domain, one λ).

    trajectories must be FORCED-CONTINUATION collection rollouts (§2.1) — RL-mode
    trajectories that stopped at ANSWER would re-introduce the censoring flaw.
    Variable lengths are handled (ALFWorld can terminate early on env success):
    a trajectory's last step initializes its V and drops out of later backups.
    """
    if not trajectories:
        raise ValueError("no trajectories")
    domain = trajectories[0].domain
    rng = np.random.default_rng(seed)

    U = [_utilities(tr, lam, median_pilot_spend, rule_table_off) for tr in trajectories]
    X = [np.stack([feature_vector(s.x) for s in tr.steps]) for tr in trajectories]
    lengths = np.array([len(tr) for tr in trajectories])
    T_max = int(lengths.max())

    # V_T = U_T at each trajectory's own final step (§10 line 2)
    V = [u.copy() for u in U]
    Cont = [np.full(len(u), np.nan) for u in U]   # NaN at each trajectory's last step

    backup_residuals: list[float] = []
    for t in range(T_max - 1, 0, -1):             # t is 1-based step; backup for steps < len
        idx = [i for i in range(len(trajectories)) if lengths[i] > t]  # step t is non-terminal
        if not idx:
            continue
        Xt = np.stack([X[i][t - 1] for i in idx])
        yt = np.array([V[i][t] for i in idx])     # V_{t+1} (0-based: index t)
        # held-out backup residual (fitted-value-iteration error check, §5.3)
        n_hold = max(1, int(holdout_frac * len(idx))) if len(idx) >= MIN_CROSS_SECTION else 0
        perm = rng.permutation(len(idx))
        hold, fit = perm[:n_hold], perm[n_hold:]
        reg = _Regressor(regressor_params).fit(Xt[fit] if len(fit) else Xt,
                                               yt[fit] if len(fit) else yt)
        if n_hold:
            backup_residuals.append(float(np.mean(np.abs(reg.predict(Xt[hold]) - yt[hold]))))
        # refit on everything for the labels themselves
        reg = _Regressor(regressor_params).fit(Xt, yt)
        cont = reg.predict(Xt)
        for k, i in enumerate(idx):
            Cont[i][t - 1] = cont[k]
            V[i][t - 1] = max(U[i][t - 1], cont[k])   # §10 line 5

    # Δ*, a*, τ*
    all_delta = np.concatenate([
        (Cont[i][:-1] - U[i][:-1]) for i in range(len(trajectories)) if len(U[i]) > 1
    ]) if any(len(u) > 1 for u in U) else np.array([0.0])
    s = scale_s if scale_s is not None else fit_delta_scale(all_delta)

    out = LabelSet(lam=lam, domain=domain, scale_s=s, backup_residuals=backup_residuals)
    for i, tr in enumerate(trajectories):
        T = len(tr)
        tau = T                                    # default: stop at the end
        for t in range(1, T + 1):
            terminal = t == T
            d_raw = 0.0 if terminal else float(Cont[i][t - 1] - U[i][t - 1])
            stop = True if terminal else d_raw <= 0.0
            if stop and tau == T and t < T:
                tau = t
            out.labels.append(StepLabel(
                task_id=tr.task_id, group_id=tr.group_id, rollout_idx=tr.rollout_idx,
                t=t, a_star="STOP" if stop else "CONTINUE",
                delta_raw=d_raw, delta_norm=float(np.tanh(d_raw / s)),
                v_star=float(V[i][t - 1]), u_t=float(U[i][t - 1]), lam=lam,
            ))
        out.tau_star[(tr.task_id, tr.rollout_idx)] = tau
    return out


# --------------------------------------------- E4 comparison arm: prophet labels
def prophet_labels(trajectories: list[Trajectory], lam: float, median_pilot_spend: float,
                   *, rule_table_off: bool = False) -> dict:
    """v5's foresight-biased argmax label (t* = argmax_t U_t; CONTINUE before, STOP
    after) — NOT used for training; E4 comparison arm only (§5.4)."""
    out = {}
    for tr in trajectories:
        u = _utilities(tr, lam, median_pilot_spend, rule_table_off)
        out[(tr.task_id, tr.rollout_idx)] = int(np.argmax(u)) + 1
    return out


# ------------------------------------------------------------------- P3 QC
def qc_lambda_monotonicity(tau_by_lambda: dict[float, dict]) -> dict:
    """P3 sanity (c): higher λ ⇒ earlier τ*, checked per trajectory across the λ grid.
    Returns violation stats; the P3 memo must report them."""
    lams = sorted(tau_by_lambda)
    keys = set.intersection(*(set(tau_by_lambda[l].keys()) for l in lams))
    violations, pairs = 0, 0
    for k in keys:
        taus = [tau_by_lambda[l][k] for l in lams]
        for a in range(len(taus) - 1):
            pairs += 1
            if taus[a + 1] > taus[a]:            # larger λ stopped LATER — violation
                violations += 1
    return {"n_trajectories": len(keys), "n_pairs": pairs, "n_violations": violations,
            "violation_rate": violations / pairs if pairs else 0.0}


def qc_mean_tau(labelset: LabelSet) -> float:
    return float(np.mean(list(labelset.tau_star.values()))) if labelset.tau_star else 0.0
