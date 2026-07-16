"""CPU tests for cassi.baselines — paper_plan_v2 §5.2 (B1–B9 + oracle), §5.3, §2.4.

Synthetic trajectories follow the canonical pattern: quality RISES then PLATEAUS
while cost accrues linearly, so U_t peaks near the plateau and the Snell τ* lands
there — cheap, deterministic, and economically meaningful.

Run from research/cassi/:  python -m pytest tests/test_baselines_cpu.py -q
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from cassi.baselines import BASELINES, REQUIRED_KEYS, load
from cassi.baselines import (
    b2_probe,
    b3_supervisor_monitor as b3,
    b4_otc_grpo as b4,
    b5_eapo as b5,
    b6_single_model_cost as b6,
    b7_cart_cost as b7,
    b8_agentprm_cost as b8,
    b9_direct_shaping as b9,
    oracle,
)
from cassi.budget.cost import base_reward
from cassi.common.schema import EMPTY_DRAFT, Step, StepFeatures, Trajectory
from cassi.executor.shaping import shaped_step_rewards, step_level_group_advantages
from cassi.labels.snell import snell_labels

MEDIAN_SPEND = 0.06     # frozen "pilot" constant for the synthetic economy
LAM = 1.0


# --------------------------------------------------------------- synthetic data
def make_traj(
    task_id: str = "task0",
    group_id: str = "g0",
    rollout_idx: int = 0,
    *,
    T: int = 6,
    plateau: int = 3,
    q_max: float = 0.8,
    cost: float = 0.01,
    allowance: float = 0.12,
    redundant: bool = False,
) -> Trajectory:
    """Quality rises linearly to q_max by `plateau`, then plateaus; constant
    per-step cost. `redundant=True` makes every tool call identical with high
    retrieval overlap (for the B3 trigger tests)."""
    steps: list[Step] = []
    for t in range(1, T + 1):
        q = q_max * min(t, plateau) / plateau
        draft = f"draft-{min(t, plateau)}"
        spent = cost * t
        obs = "same-doc-again" if redundant else f"obs-{task_id}-{t}"
        x = StepFeatures(
            tokens_used=120 * t, tokens_pct=t / (2 * T),
            tool_calls=t, tool_pct=t / (2 * T),
            dollars=spent, dollars_pct=spent / allowance, burn_rate=cost,
            tier="HIGH", step_idx=t,
            steps_since_draft_changed=max(0, t - plateau),
            draft_edit_distance_last3=[0.1, 0.1, 0.0 if t > plateau else 0.4],
            retrieval_overlap_last3=0.9 if redundant else 0.05,
            n_distinct_sources=1 if redundant else t,
            draft=draft, draft_len=len(draft),
            question="who wrote it?", domain="qa",
        )
        steps.append(Step(x=x, a="tool_call", o=obs, c=cost, tier="HIGH", draft=draft, q=q))
    outcome = {"Q_tau": q_max, "success": q_max >= 0.5, "tau": None, "gold": "gold",
               "collection_mode": "forced_continuation", "seed": 42, "iteration": 0}
    return Trajectory(
        task_id=task_id, domain="qa", allowance_B=allowance, wallet_size="medium",
        group_id=group_id, rollout_idx=rollout_idx, steps=steps, outcome=outcome,
    )


def make_group(G: int = 8, T: int = 6, plateau: int = 3) -> list[Trajectory]:
    """One GRPO group: G rollouts of the same task, slightly varied terminal quality
    (rising-then-plateau everywhere) so cross-sectional stats are non-degenerate."""
    return [
        make_traj(task_id="task0", group_id="g0", rollout_idx=i,
                  T=T, plateau=plateau, q_max=0.8 + 0.01 * i)
        for i in range(G)
    ]


@pytest.fixture(scope="module")
def group() -> list[Trajectory]:
    return make_group()


@pytest.fixture(scope="module")
def labelset(group):
    return snell_labels(group, LAM, MEDIAN_SPEND, seed=0)


def terminal_rewards(trajs: list[Trajectory], lam: float = LAM) -> list[float]:
    return [
        base_reward(tr.outcome["Q_tau"], [s.c for s in tr.steps],
                    [s.tier for s in tr.steps], lam, MEDIAN_SPEND)
        for tr in trajs
    ]


# ------------------------------------------------------------ registry (§5.2)
class TestRegistry:
    def test_completeness_vs_5_2(self):
        expected = {f"b{i}_" for i in range(1, 10)}
        names = set(BASELINES)
        assert "oracle" in names
        for prefix in expected:
            assert any(n.startswith(prefix) for n in names), f"missing §5.2 row {prefix}*"
        assert len(BASELINES) == 10   # B1–B9 + oracle, nothing else

    def test_required_keys_and_importable(self):
        for name, meta in BASELINES.items():
            assert set(REQUIRED_KEYS) <= set(meta), name
            mod = importlib.import_module(meta["module"])
            assert mod is load(name)
            # module-level knob constant must agree with the registry
            assert getattr(mod, "COST_KNOB") == meta["cost_knob"], name

    def test_frontier_protocol_knobs(self):
        # §5.3: knobless methods (B1, oracle) are single points, excluded from iso-claims
        knobless = {n for n, m in BASELINES.items() if m["cost_knob"] is None}
        assert knobless == {"b1_react", "oracle"}
        for n in knobless:
            assert getattr(load(n), "EXCLUDED_FROM_ISO_CLAIMS") is True
        # every other row has its OWN knob and the §5.2 names
        assert BASELINES["b2_probe"]["cost_knob"] == "confidence_threshold"
        assert BASELINES["b3_supervisor_monitor"]["cost_knob"] == "trigger_sensitivity"
        assert BASELINES["b4_otc_grpo"]["cost_knob"] == "tool_count_coefficient"
        assert BASELINES["b5_eapo"]["cost_knob"] == "penalty_weight"
        for n in ("b6_single_model_cost", "b8_agentprm_cost", "b9_direct_shaping"):
            assert BASELINES[n]["cost_knob"] == "lambda"
        assert BASELINES["b7_cart_cost"]["cost_knob"] == "label_lambda"

    def test_training_flags(self):
        trained = {n for n, m in BASELINES.items() if m["needs_training"]}
        assert trained == {"b4_otc_grpo", "b5_eapo", "b6_single_model_cost",
                           "b7_cart_cost", "b8_agentprm_cost", "b9_direct_shaping"}


# ------------------------------------------------------------------- B2 (§5.2)
class TestB2Probe:
    def test_parse_and_should_stop(self):
        assert b2_probe.parse_confidence("Paris\nCONFIDENCE: 85") == 85.0
        assert b2_probe.parse_confidence("no scalar here") is None
        assert b2_probe.should_stop(85.0, 80.0)
        assert not b2_probe.should_stop(70.0, 80.0)
        assert not b2_probe.should_stop(None, 0.0)   # fail-open to CONTINUE

    def test_calibration_monotone_in_target_precision(self):
        rng = np.random.default_rng(0)
        conf = rng.uniform(0, 100, size=400)
        # correctness correlated with confidence but noisy
        correct = rng.uniform(0, 100, size=400) < (0.2 * conf + 10)
        targets = [0.3, 0.5, 0.7, 0.85, 0.95, 0.999]
        thrs = [b2_probe.calibrate_threshold(conf, correct, p) for p in targets]
        for lo, hi in zip(thrs, thrs[1:]):
            assert hi >= lo, f"threshold not monotone: {thrs}"

    def test_never_stop_sentinel(self):
        thr = b2_probe.calibrate_threshold([10.0, 20.0], [False, False], 0.9)
        assert thr == b2_probe.NEVER_STOP
        assert not b2_probe.should_stop(100.0, thr)

    def test_probe_calls_are_billable(self):
        # §5.3 billing symmetry: probe returns token counts + positive dollars
        n_in = b2_probe.estimate_tokens(b2_probe.PROBE_PROMPT)
        bill = b2_probe.bill_probe(n_in, 32)
        assert bill.input_tokens == n_in and bill.output_tokens == 32
        assert bill.dollars > 0.0


# ------------------------------------------------------------------- B3 (§5.2)
class TestB3Monitor:
    def test_triggers_fire_on_redundant_trajectory(self):
        traj = make_traj(redundant=True, T=6)
        dec = b3.should_stop(traj.steps, sensitivity=0.5)
        assert dec.stop
        assert b3.REPEATED_TOOL_CALLS in dec.triggers
        assert b3.NO_NEW_INFORMATION in dec.triggers

    def test_no_fire_on_productive_trajectory(self):
        traj = make_traj(redundant=False, T=4)   # distinct calls, low overlap, cheap
        dec = b3.should_stop(traj.steps, sensitivity=0.5)
        assert not dec.stop and dec.triggers == []

    def test_sensitivity_knob_moves_the_frontier(self):
        # 3 identical calls, overlap 0.9: lazy monitor (s=0: needs 4 repeats,
        # overlap>=0.95) stays silent; aggressive monitor (s=1) fires.
        traj = make_traj(redundant=True, T=3)
        assert not b3.should_stop(traj.steps, sensitivity=0.0).stop
        assert b3.should_stop(traj.steps, sensitivity=1.0).stop

    def test_budget_trigger(self):
        traj = make_traj(T=6, cost=0.02, allowance=0.12)  # last step: 100% spent
        dec = b3.should_stop(traj.steps, sensitivity=0.0)
        assert b3.BUDGET_THRESHOLD in dec.triggers


# --------------------------------------------- B4/B5/B6 knob monotonicity (§5.3)
class TestRewardKnobMonotonicity:
    def test_b4_monotone_in_tool_coefficient(self):
        rewards = [b4.reward(1.0, m=5, m_star=1, alpha=a) for a in (0.0, 0.5, 1.0, 2.0)]
        assert rewards[0] == 1.0                       # alpha=0 → no penalty
        for lo, hi in zip(rewards[1:], rewards):
            assert lo < hi                             # strictly decreasing in alpha
        assert b4.otc_scale(1, 1, 2.0) == 1.0          # at m* → no penalty
        assert b4.otc_scale(0, 3, 1.0) == 1.0          # below m* capped, never a bonus

    def test_b4_group_m_star_from_correct_rollouts(self):
        assert b4.group_m_star([1.0, 0.0, 1.0], [4, 1, 2]) == 2   # min over CORRECT only
        assert b4.group_m_star([0.0, 0.0], [4, 3]) == 3           # fallback: group min
        g = b4.group_rewards([1.0, 1.0, 0.0], [2, 6, 1], alpha=1.0)
        assert g[0] == 1.0 and 0 < g[1] < 1.0 and g[2] == 0.0

    def test_b5_monotone_in_penalty_weight(self):
        rewards = [b5.reward(1.0, norm_cost=2.0, group_solve_rate=0.75, w=w)
                   for w in (0.0, 0.1, 0.5, 1.0)]
        assert rewards[0] == 1.0
        for lo, hi in zip(rewards[1:], rewards):
            assert lo < hi
        # adaptivity: zero solve rate → zero pressure regardless of w
        assert b5.reward(0.0, 2.0, 0.0, 5.0) == 0.0
        g = b5.group_rewards([1.0, 1.0, 0.0, 0.0], [1.0, 2.0, 1.0, 2.0], w=1.0)
        assert g[0] > g[1]                             # same group, costlier rollout pays more

    def test_b6_monotone_in_lambda(self):
        costs, tiers = [0.01] * 6, ["HIGH"] * 6
        rewards = [b6.reward(0.8, costs, tiers, lam, MEDIAN_SPEND)
                   for lam in (0.0, 0.5, 1.0, 2.0, 5.0)]
        assert rewards[0] == 0.8
        for lo, hi in zip(rewards[1:], rewards):
            assert lo < hi
        # default is the flat-λ CTA form: tiers must not matter when tier_scaled=False
        r_flat = b6.reward(0.8, costs, ["CRITICAL"] * 6, 1.0, MEDIAN_SPEND)
        assert r_flat == pytest.approx(rewards[2])
        assert b6.reward(0.8, costs, ["CRITICAL"] * 6, 1.0, MEDIAN_SPEND,
                         tier_scaled=True) < r_flat


# ------------------------------------------------------------------- B7 (§5.2)
class TestB7CartCost:
    def test_truncation_ends_at_tau_star_with_answer(self, group, labelset):
        sft = b7.build_cart_sft_dataset(group, labelset)
        assert len(sft) == len(group)                  # drafts are never empty here
        for orig, tr in zip(group, sft):
            tau = labelset.tau_star[(orig.task_id, orig.rollout_idx)]
            assert len(tr) == tau                      # ends at τ*
            assert tr.steps[-1].a == "answer"          # ANSWER appended
            assert tr.steps[-1].answered_flag
            assert tr.outcome["tau"] == tau
            assert tr.outcome["final_answer"] == orig.steps[tau - 1].draft
            assert tr.outcome["Q_tau"] == orig.steps[tau - 1].q
            # originals untouched (deep copy)
            assert orig.steps[tau - 1].a == "tool_call"

    def test_empty_draft_dropped(self, labelset):
        tr = make_traj(rollout_idx=0)
        for s in tr.steps:
            s.draft = EMPTY_DRAFT
        assert b7.build_cart_sft_dataset([tr], labelset) == []
        assert len(b7.build_cart_sft_dataset([tr], labelset, drop_empty_draft=False)) == 1

    def test_truncate_bounds(self, group):
        with pytest.raises(ValueError):
            b7.truncate_at_tau_star(group[0], 0)
        with pytest.raises(ValueError):
            b7.truncate_at_tau_star(group[0], len(group[0]) + 1)


# ------------------------------------------------------------------- B8 (§5.2)
class TestB8AgentPRMCost:
    def test_rtg_first_step_equals_base_reward(self, group):
        tr = group[0]
        rtg = b8.cost_inclusive_returns_to_go(tr, LAM, MEDIAN_SPEND)
        assert rtg[0] == pytest.approx(terminal_rewards([tr])[0])
        assert np.all(np.diff(rtg) > 0)                # constant costs → RTG rises toward Q_tau

    def test_pooled_targets_are_pooled_not_per_state_mc(self, group):
        targets = b8.pooled_rtg_targets(group, LAM, MEDIAN_SPEND)
        # EVERY visited state contributes its OWN observed RTG — no averaging
        assert len(targets) == sum(len(tr) for tr in group)
        same_state = [p.target for p in targets if p.t == 2]
        assert len(set(same_state)) > 1                # rollouts differ → targets differ
        # cost knob monotone: higher λ → lower targets
        hi = b8.pooled_rtg_targets(group, 5.0, MEDIAN_SPEND)
        assert all(h.target < t.target for h, t in zip(hi, targets))


# --------------------------------------- B9: the designed equivalence (§5.2, §2.4)
class TestB9DirectShaping:
    def test_identical_advantages_when_vhat_equals_vstar(self, group, labelset):
        """§2.4/§5.2: B9 = CASSI's exact step-level machinery, only the stopper
        deleted. When the stopper's V̂ equals v_star at labeled states (full
        coverage here), the advantage arrays must be bit-identical."""
        term = terminal_rewards(group)
        # CASSI's path: V̂ from the stopper — here set to v_star exactly
        lookup = b9.label_potential_lookup(labelset)
        v_hat = [
            np.array([lookup[(tr.task_id, tr.rollout_idx, t)]
                      for t in range(1, len(tr) + 1)])
            for tr in group
        ]
        shaped = [shaped_step_rewards(v, gamma=1.0) for v in v_hat]
        cassi_adv = step_level_group_advantages(shaped, term)
        # B9's path: identical calls, Φ read from the labels
        b9_adv = b9.direct_shaping_group_advantages(group, labelset, term, gamma=1.0)
        assert cassi_adv.guarded_steps == b9_adv.guarded_steps
        assert np.array_equal(cassi_adv.cohort_sizes, b9_adv.cohort_sizes)
        for a, b in zip(cassi_adv.advantages, b9_adv.advantages):
            assert np.array_equal(a, b)                # exact, not approx

    def test_unlabeled_states_fall_back(self, group, labelset):
        """What B9 lacks by design: off-support states get Φ=unlabeled_value."""
        fresh = make_traj(task_id="UNSEEN-task", rollout_idx=99)
        pot = b9.potentials_for_trajectory(labelset, fresh)
        assert np.all(pot == b9.DEFAULT_UNLABELED_VALUE)
        assert b9.label_coverage(labelset, [fresh]) == 0.0
        assert b9.label_coverage(labelset, group) == 1.0


# ------------------------------------------------------------------ oracle
class TestOracle:
    def test_tau_star_matches_snell_labels(self, group, labelset):
        for tr in group:
            assert oracle.tau_star(labelset, tr.task_id, tr.rollout_idx) == \
                labelset.tau_star[(tr.task_id, tr.rollout_idx)]

    def test_oracle_stops_at_the_plateau(self, group, labelset):
        # quality plateaus at step 3 while cost accrues → τ* near the plateau, before T
        taus = [oracle.tau_star(labelset, tr.task_id, tr.rollout_idx) for tr in group]
        T = len(group[0])
        assert all(1 <= tau < T for tau in taus)
        assert 2 <= float(np.mean(taus)) <= 4

    def test_headroom_summary_and_utilities(self, group, labelset):
        utils = oracle.utilities_at_tau_star(labelset)
        assert set(utils) == {(tr.task_id, tr.rollout_idx) for tr in group}
        # U at τ* is the trajectory's max utility over the decision grid
        for tr in group:
            key = (tr.task_id, tr.rollout_idx)
            traj_us = [l.u_t for l in labelset.labels
                       if (l.task_id, l.rollout_idx) == key]
            assert utils[key] == pytest.approx(max(traj_us))
        summary = oracle.headroom_summary(labelset)
        assert summary["n_trajectories"] == len(group)
        assert summary["mean_utility_at_tau_star"] > 0
        assert oracle.should_stop(5, 3) and not oracle.should_stop(2, 3)
