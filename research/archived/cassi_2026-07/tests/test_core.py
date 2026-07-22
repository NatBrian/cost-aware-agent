"""CPU tests for the core CASSI math — schema, economy, Snell labels, shaping.

These assert the load-bearing invariants of paper_plan_v2:
  * Snell recursion: correctness on a hand-computable case, λ-monotonicity of τ*
  * PBRS: telescoping to −Φ(x_0); trajectory-level advantage inertness (§2.4)
  * step-level advantages: min-cohort guard engages on variable-length groups
  * one economy: U_t and R_base agree at τ (same Lagrangian)
"""

from __future__ import annotations

import numpy as np
import pytest

from cassi.budget.cost import (
    base_reward, calibrate_wallets, stopping_utilities, tier_from_remaining,
    tier_multiplier, token_cost, tool_cost,
)
from cassi.common.schema import (
    EMPTY_DRAFT, Step, StepFeatures, Trajectory, load_trajectories, save_trajectories,
)
from cassi.executor.shaping import (
    shaped_step_rewards, step_level_group_advantages, returns_to_go,
    trajectory_level_group_advantages,
)
from cassi.labels.drafts import (
    draft_stability_features, normalized_edit_distance, parse_draft, retrieval_overlap,
)
from cassi.labels.quality import alfworld_quality, exact_match, f1_score
from cassi.labels.snell import (
    fit_delta_scale, prophet_labels, qc_lambda_monotonicity, snell_labels,
)
from cassi.stopper.features import FEATURE_NAMES, feature_vector, serialize


# ------------------------------------------------------------------ fixtures
def make_traj(task_id: str, ridx: int, T: int = 8, plateau: int = 4,
              rng: np.random.Generator | None = None, noise: float = 0.02) -> Trajectory:
    """Synthetic trajectory: quality rises to a plateau, cost accumulates —
    the canonical 'more work stops paying' shape."""
    rng = rng or np.random.default_rng(ridx)
    steps, q = [], 0.0
    for t in range(1, T + 1):
        q = float(np.clip(q + (0.25 if t <= plateau else 0.01) + rng.normal(0, noise), 0, 1))
        x = StepFeatures(
            tokens_used=500 * t, tokens_pct=0.05 * t, tool_calls=t, tool_pct=0.1 * t,
            dollars=0.01 * t, dollars_pct=0.1 * t, burn_rate=0.01,
            tier=tier_from_remaining(0.01 * t, 0.1), step_idx=t,
            steps_since_draft_changed=max(0, t - plateau),
            draft=f"ans v{min(t, plateau)}", draft_len=6, question="q?", domain="qa",
        )
        steps.append(Step(x=x, a="tool_call", o="obs", c=0.01, tier=x.tier,
                          draft=x.draft, q=q))
    return Trajectory(task_id=task_id, domain="qa", allowance_B=0.1,
                      wallet_size="medium", group_id=task_id, rollout_idx=ridx,
                      steps=steps, outcome={"Q_tau": steps[-1].q,
                                            "collection_mode": "forced_continuation"})


@pytest.fixture(scope="module")
def trajs() -> list[Trajectory]:
    rng = np.random.default_rng(0)
    return [make_traj(f"t{i // 8}", i % 8, rng=rng) for i in range(80)]


# -------------------------------------------------------------------- schema
def test_schema_roundtrip(tmp_path, trajs):
    p = tmp_path / "t.jsonl"
    save_trajectories(trajs[:3], p)
    back = list(load_trajectories(p))
    assert len(back) == 3
    assert back[0].to_dict() == trajs[0].to_dict()


# ------------------------------------------------------------------- economy
def test_tiering():
    assert tier_from_remaining(0.0, 1.0) == "HIGH"
    assert tier_from_remaining(0.5, 1.0) == "MEDIUM"
    assert tier_from_remaining(0.75, 1.0) == "LOW"
    assert tier_from_remaining(0.95, 1.0) == "CRITICAL"
    assert tier_from_remaining(5.0, 1.0) == "CRITICAL"      # overspent
    assert tier_multiplier("CRITICAL") == 5.0
    assert tier_multiplier("CRITICAL", rule_table_off=True) == 1.0   # A8 economy


def test_costs():
    assert token_cost(1_000_000, 0) == pytest.approx(0.60)
    assert token_cost(0, 1_000_000) == pytest.approx(2.20)
    assert tool_cost("web_search", n_results=3) == pytest.approx(0.003 + 3 * 0.001)
    assert tool_cost("code_exec", seconds=10) == pytest.approx(0.0001 + 10 * 0.0001)


def test_one_economy_u_and_rbase_agree():
    """§2.4: U_τ computed by the labels equals R_base for a stop at τ — one Lagrangian."""
    q = [0.2, 0.5, 0.9, 0.92]
    c = [0.01, 0.02, 0.01, 0.03]
    tiers = ["HIGH", "MEDIUM", "LOW", "CRITICAL"]
    u = stopping_utilities(q, c, tiers, lam=1.0, median_pilot_spend=0.05)
    for tau in range(1, 5):
        r = base_reward(q[tau - 1], c[:tau], tiers[:tau], lam=1.0, median_pilot_spend=0.05)
        assert u[tau - 1] == pytest.approx(r)


def test_calibrate_wallets():
    spends = list(np.linspace(0.01, 1.0, 200))
    w = calibrate_wallets(spends)
    assert w["small"] < w["medium"] < w["large"]
    assert w["large"] == pytest.approx(2 * np.percentile(spends, 90))
    with pytest.raises(ValueError):
        calibrate_wallets([0.1] * 5)


# ------------------------------------------------------------------- quality
def test_qa_quality():
    assert exact_match("The Eiffel Tower", "eiffel tower") == 1.0
    assert f1_score("tower eiffel", "the eiffel tower") == pytest.approx(1.0)
    assert f1_score(EMPTY_DRAFT, "x") == 0.0
    assert 0 < f1_score("eiffel tower in paris", "eiffel tower") < 1
    assert alfworld_quality(2, 4) == 0.5


# -------------------------------------------------------------------- drafts
def test_parse_draft():
    assert parse_draft("thinking...\nBEST ANSWER SO FAR: 42") == "42"
    assert parse_draft("BEST ANSWER SO FAR: EMPTY_DRAFT") == EMPTY_DRAFT
    assert parse_draft("no line here") == EMPTY_DRAFT
    assert parse_draft("BEST ANSWER SO FAR: a\nmore\nBEST ANSWER SO FAR: b") == "b"


def test_draft_stability():
    feats = draft_stability_features(["a", "a", "ab", "ab", "ab"])
    assert feats["steps_since_draft_changed"] == 2
    assert len(feats["draft_edit_distance_last3"]) == 3
    assert feats["draft_edit_distance_last3"][-1] == 0.0     # last transition ab->ab
    assert normalized_edit_distance("abc", "abc") == 0.0
    assert normalized_edit_distance("", "abc") == 1.0
    assert retrieval_overlap([{1, 2}, {2, 3}]) == pytest.approx(1 / 3)


# ------------------------------------------------------------ features (§18.1)
def test_feature_vector_and_serialize(trajs):
    x = trajs[0].steps[2].x
    v = feature_vector(x)
    assert len(v) == len(FEATURE_NAMES)
    txt = serialize(x, lam=0.5, tokens_max=10_000, tool_calls_max=10,
                    allowance_dollars=0.1, t_max=10)
    assert "λ = 0.5" in txt and "[DRAFT]" in txt and "<stopper_input>" in txt
    # §18.1: no ground-truth leakage possible — q is not even on StepFeatures
    assert not hasattr(x, "q")


# ------------------------------------------------------------- Snell (Alg. 1)
def test_snell_hand_computable():
    """Deterministic identical trajectories: quality jumps to 0.9 at step 3 then
    flat; per-step penalized cost 0.1 → U = [-0.1, 0.3, 0.6, 0.5, 0.4].
    Optimal τ* = 3 (U peaks there and continuation only loses money)."""
    trajs = []
    q_seq = [0.0, 0.5, 0.9, 0.9, 0.9]
    for r in range(40):
        steps = []
        for t, q in enumerate(q_seq, 1):
            x = StepFeatures(step_idx=t, dollars=0.01 * t, tier="MEDIUM",
                             question="q", domain="qa", draft="d", draft_len=1,
                             tokens_used=t, tool_calls=t)
            steps.append(Step(x=x, a="tool_call", o="o", c=0.01, tier="MEDIUM",
                              draft="d", q=q))
        trajs.append(Trajectory(task_id=f"k{r}", domain="qa", allowance_B=0.1,
                                wallet_size="medium", group_id=f"k{r}", rollout_idx=0,
                                steps=steps, outcome={}))
    ls = snell_labels(trajs, lam=1.0, median_pilot_spend=0.1, seed=0)
    taus = list(ls.tau_star.values())
    assert all(t == 3 for t in taus), f"expected τ*=3 everywhere, got {set(taus)}"
    # labels at the stop step are STOP with Δ ≤ 0
    for lab in ls.labels:
        if lab.t == 3:
            assert lab.a_star == "STOP" and lab.delta_raw <= 1e-9
        if lab.t == 2:
            assert lab.a_star == "CONTINUE" and lab.delta_raw > 0


def test_snell_lambda_monotonicity(trajs):
    sets = {lam: snell_labels(trajs, lam=lam, median_pilot_spend=0.08, seed=0)
            for lam in (0.1, 1.0, 5.0)}
    mono = qc_lambda_monotonicity({l: s.tau_star for l, s in sets.items()})
    assert mono["violation_rate"] < 0.05
    means = [np.mean(list(sets[l].tau_star.values())) for l in (0.1, 1.0, 5.0)]
    assert means[0] >= means[1] >= means[2]


def test_snell_terminal_convention(trajs):
    ls = snell_labels(trajs[:8], lam=1.0, median_pilot_spend=0.08, seed=0)
    for lab in ls.labels:
        if lab.t == 8:                        # T_max step
            assert lab.a_star == "STOP" and lab.delta_raw == 0.0


def test_prophet_vs_snell_bias(trajs):
    """Prophet argmax may pick a late lucky peak; Snell τ* must not stop LATER on
    average than prophet under identical utilities (foresight bias, §2.2)."""
    snell = snell_labels(trajs, lam=1.0, median_pilot_spend=0.08, seed=0)
    prophet = prophet_labels(trajs, lam=1.0, median_pilot_spend=0.08)
    ms = np.mean(list(snell.tau_star.values()))
    mp = np.mean([prophet[k] for k in snell.tau_star])
    assert ms <= mp + 0.5


def test_fit_delta_scale():
    assert fit_delta_scale(np.array([0.0, 0.0])) == 1.0
    s = fit_delta_scale(np.array([0.1, -0.2, 0.3, -0.4, 0.5]))
    assert s == pytest.approx(np.percentile([0.1, 0.2, 0.3, 0.4, 0.5], 90))


def test_snell_variable_lengths():
    """ALFWorld-style early env termination: shorter trajectories must still label."""
    rng = np.random.default_rng(1)
    ts = [make_traj(f"v{i}", 0, T=5 + (i % 4), rng=rng) for i in range(40)]
    ls = snell_labels(ts, lam=1.0, median_pilot_spend=0.08, seed=0)
    by_len = {}
    for lab in ls.labels:
        by_len.setdefault(lab.task_id, 0)
        by_len[lab.task_id] = max(by_len[lab.task_id], lab.t)
    assert set(by_len.values()) == {5, 6, 7, 8}


# ----------------------------------------------------------- shaping (§2.4)
def test_telescoping():
    v = np.random.default_rng(2).normal(0, 1, size=10)
    r = shaped_step_rewards(v, gamma=1.0)
    assert r.sum() == pytest.approx(-v[0])                  # Σr = −Φ(x_0)


def test_returns_to_go():
    r = np.array([1.0, 2.0, 3.0])
    rtg = returns_to_go(r, terminal_reward=10.0)
    assert np.allclose(rtg, [16.0, 15.0, 13.0])


def test_trajectory_level_inertness():
    """§2.4: same Φ(x_0) across the group ⇒ trajectory-level advantages identical
    with and without shaping — shaping is invisible at trajectory level."""
    rng = np.random.default_rng(3)
    v0, term = 0.7, [1.0, 0.5, 0.2, 0.8]
    shaped = []
    for T in (5, 6, 7, 8):
        v = rng.normal(0, 1, size=T)
        v[0] = v0
        shaped.append(shaped_step_rewards(v))
    a_with = trajectory_level_group_advantages(shaped, term)
    a_without = trajectory_level_group_advantages([np.zeros(T) for T in (5, 6, 7, 8)], term)
    assert np.allclose(a_with, a_without, atol=1e-9)


def test_step_level_advantages_and_guard():
    rng = np.random.default_rng(4)
    sr = [rng.normal(0, 1, size=T) for T in (5, 6, 7, 8)]
    ga = step_level_group_advantages(sr, [1.0, 0.5, 0.2, 0.8], min_cohort=3)
    assert list(ga.cohort_sizes) == [4, 4, 4, 4, 4, 3, 2, 1]
    assert ga.guarded_steps == 3                             # steps with cohort 2 and 1
    # normalized: cohort steps have ~zero mean across alive members
    step0 = [ga.advantages[i][0] for i in range(4)]
    assert np.mean(step0) == pytest.approx(0.0, abs=1e-9)
    # guarded steps carry the trajectory-level baseline value
    tot = trajectory_level_group_advantages(sr, [1.0, 0.5, 0.2, 0.8])
    assert ga.advantages[3][7] == pytest.approx(tot[3])
