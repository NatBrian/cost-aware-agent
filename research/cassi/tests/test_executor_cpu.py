"""CPU-only executor tests — no torch/verl/vllm (paper_plan_v2 §16 P0/P2, §2.1,
§2.4, §2.5, §10 Alg.3–4).

Run from research/cassi/:  python -m pytest tests/test_executor_cpu.py -q
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from cassi.budget.cost import token_cost
from cassi.common.config import load_config
from cassi.common.schema import EMPTY_DRAFT, Step, StepFeatures, Trajectory, load_trajectories
from cassi.executor.collect import collect_round, forced_continuation_overhead, run_pilot
from cassi.executor.envs.base import MockSearchEnv
from cassi.executor.monitor import MockStopper, StopperMonitor, monitor_from_config
from cassi.executor.react_agent import ReactAgent, ScriptedLLMClient
from cassi.executor.shaping import trajectory_level_group_advantages
from cassi.executor.train_grpo import VerlCassiAdapter, compute_cassi_rewards

# ----------------------------------------------------------------- fixtures
CORPUS = {
    "d1": "Paris is the capital of France.",
    "d2": "Berlin is the capital of Germany.",
    "d3": "France is a country in Europe.",
}
TASK = {"task_id": "t1", "question": "What is the capital of France?", "gold": "Paris"}

STEP_SEARCH_1 = (
    "THOUGHT: I should look up the capital of France.\n"
    "ACTION: search[capital of France]\n"
    "BEST ANSWER SO FAR: EMPTY_DRAFT"
)
STEP_SEARCH_2 = (
    "THOUGHT: The passages say Paris; verify once more.\n"
    "ACTION: search[Paris France capital]\n"
    "BEST ANSWER SO FAR: Paris"
)
STEP_ANSWER = (
    "THOUGHT: Confirmed.\n"
    "ACTION: answer[Paris]\n"
    "BEST ANSWER SO FAR: Paris"
)
STEP_SEARCH_MORE = (
    "THOUGHT: Keep double-checking for robustness.\n"
    "ACTION: search[France Europe geography]\n"
    "BEST ANSWER SO FAR: Paris"
)

ANSWER_MIDWAY = [STEP_SEARCH_1, STEP_SEARCH_2, STEP_ANSWER, STEP_SEARCH_MORE]
NEVER_ANSWER = [STEP_SEARCH_1, STEP_SEARCH_2, STEP_SEARCH_MORE]

ALLOWANCES = {"small": 0.0005, "medium": 0.002, "large": 0.01}


def _run(outputs, *, mode, t_max=6, allowance=0.05, monitor=None):
    agent = ReactAgent(ScriptedLLMClient(outputs))
    return agent.run(
        TASK, MockSearchEnv(CORPUS), mode=mode, t_max=t_max,
        allowance_dollars=allowance, wallet_size="large",
        group_id="t1:iter0:g0", rollout_idx=0, monitor=monitor,
    )


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ======================================================== (a) react_agent modes
class TestReactAgentModes:
    def test_rl_mode_stops_at_answer(self):
        res = _run(ANSWER_MIDWAY, mode="rl")
        traj = res.trajectory
        assert res.stopped_by == "answer" and res.self_terminated
        assert len(traj.steps) == 3
        assert traj.steps[-1].a == "answer" and traj.steps[-1].answered_flag
        assert traj.outcome["tau"] == 3
        assert traj.outcome["collection_mode"] == "rl"
        assert res.final_answer == "Paris"

    def test_forced_continuation_runs_to_t_max(self):
        res = _run(ANSWER_MIDWAY, mode="forced_continuation", t_max=6)
        traj = res.trajectory
        assert res.stopped_by == "t_max"
        assert len(traj.steps) == 6                       # forced past the ANSWER (§2.1)
        assert traj.outcome["tau"] == 3
        assert traj.steps[2].answered_flag                # the free self-stop measurement
        assert all(not s.answered_flag for s in traj.steps[3:])
        assert all(s.a == "tool_call" for s in traj.steps[3:])
        assert traj.outcome["collection_mode"] == "forced_continuation"
        assert res.final_answer == "Paris"                # answer AT τ, not overwritten

    def test_draft_parsed_every_step(self):
        res = _run(ANSWER_MIDWAY, mode="forced_continuation", t_max=6)
        drafts = [s.draft for s in res.trajectory.steps]
        assert drafts == [EMPTY_DRAFT, "Paris", "Paris", "Paris", "Paris", "Paris"]
        # x_t carries the PRE-action draft (§11): step 2 still sees EMPTY_DRAFT
        assert res.trajectory.steps[0].x.draft == EMPTY_DRAFT
        assert res.trajectory.steps[1].x.draft == EMPTY_DRAFT
        assert res.trajectory.steps[2].x.draft == "Paris"
        assert res.trajectory.outcome["format_score"] == 1.0
        assert res.draft_line_tokens > 0                  # priced (§2.6, feeds T4)

    def test_budget_features_monotone(self):
        res = _run(ANSWER_MIDWAY, mode="forced_continuation", t_max=6)
        xs = [s.x for s in res.trajectory.steps]
        for a, b in zip(xs, xs[1:]):
            assert b.tokens_used > a.tokens_used
            assert b.dollars > a.dollars
            assert b.dollars_pct >= a.dollars_pct
            assert b.tool_calls >= a.tool_calls
            assert b.step_idx == a.step_idx + 1
        assert xs[0].tokens_used == 0 and xs[0].dollars == 0.0
        # costs: every step is priced (token cost > 0 always)
        assert all(s.c > 0 for s in res.trajectory.steps)
        assert all(s.tier == s.x.tier for s in res.trajectory.steps)

    def test_retrieval_features_populate(self):
        res = _run(NEVER_ANSWER, mode="forced_continuation", t_max=5)
        xs = [s.x for s in res.trajectory.steps]
        assert xs[-1].n_distinct_sources >= 2             # searches hit d1/d2/d3
        assert 0.0 <= xs[-1].retrieval_overlap_last3 <= 1.0
        assert len(xs[-1].history) > 0
        assert all(len(h["obs_digest"]) <= 256 for h in xs[-1].history)  # ≤64-tok digests

    def test_bad_mode_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            _run(ANSWER_MIDWAY, mode="monitor")


# ============================================================ (b) collect.py e2e
class TestCollect:
    @pytest.fixture(scope="class")
    def round_result(self, tmp_path_factory):
        cfg = load_config()
        out = tmp_path_factory.mktemp("collect") / "rollouts.jsonl"
        tasks = [dict(TASK, task_id=f"t{i}") for i in range(30)]
        llm = ScriptedLLMClient(ANSWER_MIDWAY)           # reset() per episode
        env = MockSearchEnv(CORPUS)
        report = collect_round(
            tasks, llm, env, cfg, domain="qa", out_path=out,
            allowances=ALLOWANCES, G=4, t_max=5, seed=0,
        )
        trajs = list(load_trajectories(out))
        return report, trajs

    def test_report_counts(self, round_result):
        report, trajs = round_result
        assert report["n_trajectories"] == len(trajs) == 30 * 4
        assert report["total_dollars"] > 0

    def test_schema_valid_jsonl(self, round_result):
        _, trajs = round_result
        for tr in trajs:
            assert tr.wallet_size in ALLOWANCES
            assert tr.allowance_B == ALLOWANCES[tr.wallet_size]
            assert tr.outcome["collection_mode"] == "forced_continuation"
            assert len(tr.steps) == 5                     # forced to T_max (§2.1)
            assert tr.outcome["tau"] == 3
            assert tr.steps[2].answered_flag
            for s in tr.steps:
                assert isinstance(s.x, StepFeatures)
                assert s.tier in ("HIGH", "MEDIUM", "LOW", "CRITICAL")
        # round-trip identity through the §11 JSONL schema
        d = trajs[0].to_dict()
        assert Trajectory.from_dict(d).to_dict() == d

    def test_group_shared_wallets_and_balance(self, round_result):
        report, trajs = round_result
        groups: dict[str, list[Trajectory]] = {}
        for tr in trajs:
            groups.setdefault(tr.group_id, []).append(tr)
        assert len(groups) == 30
        for members in groups.values():
            # ONE wallet per (task, group), shared by all G rollouts (§2.2)
            assert len({(m.allowance_B, m.wallet_size) for m in members}) == 1
            assert sorted(m.rollout_idx for m in members) == [0, 1, 2, 3]
        # roughly balanced strata (uniform draw over 30 tasks)
        assert all(v >= 1 for v in report["wallet_counts"].values())
        assert sum(report["wallet_counts"].values()) == 30

    def test_quality_scored_at_collection_only(self, round_result):
        _, trajs = round_result
        tr = trajs[0]
        assert tr.steps[0].q == 0.0                      # EMPTY_DRAFT scores 0
        assert tr.steps[1].q == 1.0                      # draft 'Paris' vs gold 'Paris'
        assert tr.outcome["Q_tau"] == 1.0 and tr.outcome["success"] is True
        # hard rule (§2.1): q never leaks into x_t
        assert "q" not in tr.steps[1].x.to_dict()

    def test_t4_accounting(self, round_result):
        report, trajs = round_result
        assert 0.0 < report["draft_token_share"] < 0.5
        assert report["forced_continuation_overhead_dollars"] > 0
        assert 0.0 < report["overhead_pct_of_total"] < 1.0
        assert forced_continuation_overhead(trajs[0]) == pytest.approx(
            sum(s.c for s in trajs[0].steps[3:]))

    def test_pilot_returns_spends(self):
        cfg = load_config()
        tasks = [dict(TASK, task_id=f"p{i}") for i in range(5)]
        spends = run_pilot(tasks, ScriptedLLMClient(ANSWER_MIDWAY),
                           MockSearchEnv(CORPUS), cfg, domain="qa", t_max=5)
        assert len(spends) == 5 and all(s > 0 for s in spends)
        # rl-mode pilot: 3 steps of spend (stops at ANSWER), same every task
        assert spends == pytest.approx([spends[0]] * 5)

    def test_uncalibrated_config_refuses(self, tmp_path):
        cfg = load_config()                              # allowances are null pre-P2
        with pytest.raises(RuntimeError, match="Pilot calibration missing"):
            collect_round([TASK], ScriptedLLMClient(ANSWER_MIDWAY),
                          MockSearchEnv(CORPUS), cfg, domain="qa",
                          out_path=tmp_path / "x.jsonl")


# ================================================================ (c) monitor
class TestMonitor:
    def test_stops_exactly_at_first_nonpositive_delta(self):
        stopper = MockStopper(delta_by_step={1: 0.5, 2: 0.10, 3: -0.05}, default_delta=1.0)
        monitor = StopperMonitor(stopper, lam=1.0, t_max=8)
        res = _run(NEVER_ANSWER, mode="rl", t_max=8, monitor=monitor)
        assert res.stopped_by == "monitor"
        assert len(res.trajectory.steps) == 2            # stopped PRE-action at step 3
        assert monitor.last_delta == pytest.approx(-0.05)
        assert res.final_answer == "Paris"               # the running draft is the answer
        assert monitor.stats()["monitor_stopped"] == 1

    def test_a8_rule_table_differs_from_learned(self):
        deltas = {1: 0.5, 2: 0.10, 3: -0.05}
        learned = StopperMonitor(MockStopper(delta_by_step=deltas), lam=1.0, t_max=8)
        res_learned = _run(NEVER_ANSWER, mode="rl", t_max=8, monitor=learned)
        rule = StopperMonitor(
            MockStopper(delta_by_step=deltas), lam=1.0, t_max=8,
            rule_table={"HIGH": 0.15, "MEDIUM": 0.15, "LOW": 0.15, "CRITICAL": 0.30},
        )
        res_rule = _run(NEVER_ANSWER, mode="rl", t_max=8, monitor=rule)
        # learned fixed threshold 0 stops at step 3; δ(tier)=0.15 already at step 2
        assert len(res_learned.trajectory.steps) == 2
        assert len(res_rule.trajectory.steps) == 1
        assert rule.stats()["mode"] == "rule_table_A8"
        assert learned.stats()["mode"] == "learned"

    def test_monitor_from_config_reads_a8_table(self, cfg):
        m = monitor_from_config(MockStopper(), cfg, ablation_a8=True, domain="qa")
        assert m.rule_table == cfg["inference"]["ablation_A8_rule_table"]
        assert m.every_k == cfg["inference"]["stopper_eval_every_k"]
        m2 = monitor_from_config(MockStopper(), cfg, domain="qa")
        assert m2.rule_table is None and m2.delta_threshold == 0.0
        assert m2.t_max == cfg["executor"]["horizon"]["qa"]

    def test_budget_exhausted_stop(self):
        monitor = StopperMonitor(MockStopper(default_delta=1.0), lam=1.0, t_max=8)
        res = _run(NEVER_ANSWER, mode="rl", t_max=8, allowance=1e-6, monitor=monitor)
        assert res.stopped_by == "budget"
        assert len(res.trajectory.steps) == 1            # step 1 spent past the wallet
        assert monitor.stats()["budget_stopped"] == 1

    def test_self_termination_tracking(self):
        # Δ̂ ≤ 0 first fires at step 2; episode A answers AT step 1 (beats the monitor)
        monitor = StopperMonitor(
            MockStopper(delta_by_step={1: 1.0, 2: -1.0}), lam=1.0, t_max=8)
        res_a = _run([STEP_ANSWER], mode="rl", t_max=8, monitor=monitor)
        res_b = _run(NEVER_ANSWER, mode="rl", t_max=8, monitor=monitor)
        assert res_a.stopped_by == "answer" and res_a.self_terminated
        assert res_b.stopped_by == "monitor" and not res_b.self_terminated
        s = monitor.stats()
        assert s["episodes"] == 2
        assert s["self_terminated"] == 1 and s["monitor_stopped"] == 1
        assert s["self_termination_rate"] == pytest.approx(0.5)   # §2.5 internalization

    def test_every_k_skips_queries(self):
        stopper = MockStopper(default_delta=-1.0)        # would stop at any query
        monitor = StopperMonitor(stopper, lam=1.0, t_max=8, every_k=2)
        res = _run(NEVER_ANSWER, mode="rl", t_max=8, allowance=10.0, monitor=monitor)
        # step 1 not a query step (1 % 2 != 0); first query at step 2 stops
        assert len(res.trajectory.steps) == 1
        assert stopper.n_queries == 1


# ==================================================== (d) compute_cassi_rewards
def _mk_x(t: int) -> StepFeatures:
    return StepFeatures(step_idx=t, question="q", domain="qa")


def _mk_traj(task_id: str, ridx: int, n_steps: int, q_tau: float,
             cost: float = 0.01) -> Trajectory:
    steps = [Step(x=_mk_x(t + 1), a="tool_call", o="obs", c=cost, tier="HIGH",
                  draft="d", q=0.0, answered_flag=(t == n_steps - 1))
             for t in range(n_steps)]
    return Trajectory(
        task_id=task_id, domain="qa", allowance_B=0.01, wallet_size="medium",
        group_id=f"{task_id}:iter0:g0", rollout_idx=ridx, steps=steps,
        outcome={"Q_tau": q_tau, "tau": n_steps, "collection_mode": "rl",
                 "format_score": 1.0},
    )


class TestComputeCassiRewards:
    MEDIAN = 1.0

    def test_telescoping_property(self, cfg):
        """§2.4: Σ_t r_t = −Φ(x_1); with a common start state, trajectory-level
        group advantages are provably unaffected by the shaping."""
        rng = np.random.default_rng(7)
        trajs = [_mk_traj("k1", i, 5, q_tau=float(i % 2)) for i in range(4)]
        v0 = 0.37                                        # same x_1 → same V̂(x_1)
        v_hats = [np.concatenate([[v0], rng.normal(size=4)]) for _ in range(4)]
        out = compute_cassi_rewards(trajs, v_hats, cfg, median_pilot_spend=self.MEDIAN)
        for r in out.step_rewards:
            assert r.sum() == pytest.approx(-v0)         # telescoped constant
        assert out.telescoped_constants == pytest.approx([-v0] * 4)
        shaped = trajectory_level_group_advantages(out.step_rewards, out.terminal_rewards)
        inert = trajectory_level_group_advantages(
            [np.zeros(5)] * 4, out.terminal_rewards)
        np.testing.assert_allclose(shaped, inert, atol=1e-9)
        # ...while STEP-level advantages do differ (dense credit is the point)
        no_shape = compute_cassi_rewards(trajs, [np.zeros(5)] * 4, cfg,
                                         median_pilot_spend=self.MEDIAN)
        assert any(not np.allclose(a, b) for a, b in
                   zip(out.advantages, no_shape.advantages))

    def test_min_cohort_guard_on_variable_lengths(self, cfg):
        rng = np.random.default_rng(11)
        lengths = [2, 2, 6, 6]
        trajs = [_mk_traj("k2", i, n, q_tau=float(i % 2)) for i, n in enumerate(lengths)]
        v_hats = [rng.normal(size=n) for n in lengths]
        out = compute_cassi_rewards(trajs, v_hats, cfg, median_pilot_spend=self.MEDIAN)
        np.testing.assert_array_equal(out.cohort_sizes, [4, 4, 2, 2, 2, 2])
        assert out.guarded_steps == 8                    # 2 trajs × steps 3..6
        for i in (2, 3):                                 # guarded steps share the
            tail = out.advantages[i][2:]                 # trajectory-level baseline
            np.testing.assert_allclose(tail, tail[0])
        for adv, n in zip(out.advantages, lengths):
            assert adv.shape == (n,)

    def test_base_reward_economy_matches_labels(self, cfg):
        """R_base = Q_τ − Σ λ·m(tier)·c̃ — the labels' own economy (§2.4)."""
        tr = _mk_traj("k3", 0, 3, q_tau=1.0, cost=0.02)
        out = compute_cassi_rewards([tr] * 4, [np.zeros(3)] * 4, cfg,
                                    median_pilot_spend=0.1)
        lam = cfg["executor"]["training_lambda"]
        expected = 1.0 - lam * 0.5 * (0.02 / 0.1) * 3    # m(HIGH)=0.5, 3 steps
        assert out.base_rewards[0] == pytest.approx(expected)
        fmt_w = cfg["executor"]["shaping"]["format_weight"]
        assert out.terminal_rewards[0] == pytest.approx(expected + fmt_w * 1.0)

    def test_error_paths(self, cfg):
        tr = _mk_traj("k4", 0, 3, q_tau=1.0)
        with pytest.raises(ValueError, match="length"):
            compute_cassi_rewards([tr], [np.zeros(2)], cfg, median_pilot_spend=1.0)
        bad = _mk_traj("k4", 1, 3, q_tau=1.0)
        bad.outcome["Q_tau"] = None
        with pytest.raises(ValueError, match="Q_tau"):
            compute_cassi_rewards([bad], [np.zeros(3)], cfg, median_pilot_spend=1.0)
        with pytest.raises(RuntimeError, match="median_pilot_spend"):
            compute_cassi_rewards([tr], [np.zeros(3)], cfg)   # cfg still null pre-P2

    def test_adapter_is_cpu_safe(self, cfg):
        """The verl seam builds its trainer config without importing verl (§16 P6)."""
        adapter = VerlCassiAdapter(cfg, domain="qa")
        tc = adapter.trainer_config
        assert tc["actor_rollout_ref"]["rollout"]["n"] == 8
        assert tc["actor_rollout_ref"]["actor"]["kl_loss_coef"] == pytest.approx(0.04)
        assert tc["algorithm"]["norm_adv_by_std_in_grpo"] is False   # Dr.GRPO hygiene
        tr = _mk_traj("k5", 0, 4, q_tau=1.0)
        with pytest.raises(RuntimeError, match="median_pilot_spend"):
            adapter.compute_group_rewards([tr], [np.zeros(4)])


# ======================================= (e) verl wiring (§16 P6, CPU-safe only)
class TestVerlWiring:
    """The P6 hooks, exercised WITHOUT GPUs. The dry-run runs in a subprocess so
    this file's in-process no-torch/no-verl rule stays intact; the encode/decode
    round trip importorskips the pinned verl stack (present in .venv per P0)."""

    def test_dry_run_cli_is_cpu_safe_and_validates_hooks(self, tmp_path):
        import subprocess
        import sys
        from pathlib import Path

        research_dir = str(Path(__file__).resolve().parent.parent.parent)
        env = dict(os.environ, PYTHONPATH=research_dir)
        proc = subprocess.run(
            [sys.executable, "-m", "cassi.executor.train_grpo", "--dry-run",
             "--domain", "qa", "--out", str(tmp_path)],
            capture_output=True, text=True, env=env,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "cassi_step_level" in proc.stdout          # adv estimator choice printed
        assert "norm_adv_by_std_in_grpo" in proc.stdout   # Dr.GRPO hygiene visible

    def test_advantage_encode_decode_round_trip(self):
        """rm_scores difference-encoding on step-final tokens must decode to the
        exact per-step advantages on every response token of that step (§2.4 —
        the equivalence the verl_hooks docstring documents)."""
        pytest.importorskip("verl")
        torch = pytest.importorskip("torch")
        from cassi.executor.verl_hooks import (
            compute_cassi_step_level_advantage,
            encode_step_values,
        )

        rng = np.random.default_rng(3)
        width = 40
        rows, masks, per_step = [], [], []
        for ends in ([4, 11, 19, 33], [7, 39], [0]):      # variable lengths incl. edges
            adv = rng.normal(size=len(ends))
            rows.append(encode_step_values(adv, ends, width))
            m = torch.zeros(width)
            m[: ends[-1] + 1] = 1.0                        # response region incl. obs tokens
            masks.append(m)
            per_step.append((ends, adv))
        token_level_rewards = torch.stack(rows)
        response_mask = torch.stack(masks)

        decoded, returns = compute_cassi_step_level_advantage(
            token_level_rewards, response_mask, config=None, index=None)
        assert torch.equal(decoded, returns)              # GRPO outcome convention
        for i, (ends, adv) in enumerate(per_step):
            start = 0
            for t, end in enumerate(ends):
                seg = decoded[i, start:end + 1].numpy()
                np.testing.assert_allclose(seg, adv[t], atol=1e-6)
                start = end + 1
            assert float(decoded[i, ends[-1] + 1:].abs().sum()) == 0.0  # masked tail


# ============================================== CPU import safety of lazy modules
class TestCpuImportSafety:
    def test_vllm_client_imports_without_gpu_stack(self):
        from cassi.executor.vllm_client import VLLMClient
        c = VLLMClient.from_config(load_config())
        assert c.model and c.enable_thinking is False

    def test_searchr1_env_unreachable_retriever_raises_with_instructions(self):
        pytest.importorskip("requests")
        from cassi.executor.envs.searchr1_qa import SearchR1QAEnv
        env = SearchR1QAEnv(retriever_url="http://127.0.0.1:9/retrieve", timeout=1.0)
        env.reset(TASK)
        with pytest.raises(NotImplementedError, match="p1_data.sh"):
            env.step("search", "capital of France")

    def test_alfworld_module_imports_cleanly(self):
        import cassi.executor.envs.alfworld as m
        assert hasattr(m, "ALFWorldEnv")
