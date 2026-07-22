"""CPU-only tests for cassi.stopper.{dataset,model,eval_regret,train_sft} —
paper_plan_v2 §2.3 (λ-conditioned dataset), §2.5 (stop policy), §5.3 (regret),
§16 P4 (done-criterion baselines). No torch required: the real model is lazily
imported (verified by an import-blocker subprocess) and eval runs on MockStopper.

Synthetic data follows the smoke-test pattern: per-step quality rises to a
plateau while cost accrues, so U_t = q_t − λ·Σ m·c̃ peaks mid-trajectory and
Algorithm 1 (cassi.labels.snell.snell_labels) produces interior τ*.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cassi.budget.cost import tier_from_remaining
from cassi.common.schema import Step, StepFeatures, Trajectory
from cassi.labels.snell import snell_labels
from cassi.stopper import dataset as ds
from cassi.stopper.eval_regret import (
    calibrate_draft_stability, compare_p4_baselines, evaluate_multi_lambda,
    evaluate_stopper, labels_by_trajectory, tau_star_of, write_records_csv,
)
from cassi.stopper.model import MockStopper

MEDIAN_PILOT_SPEND = 0.3
STEP_COST = 0.05
T = 8


# ------------------------------------------------------------- synthetic data
def make_synthetic_trajectories(n_tasks: int = 6, G: int = 4, seed: int = 7) -> list[Trajectory]:
    """Quality rises to a per-task plateau then flattens; cost accrues each step —
    the same shape the smoke tests use, guaranteeing interior Snell stops."""
    rng = np.random.default_rng(seed)
    wallets = {"small": 0.3, "medium": 0.5, "large": 0.8}
    trajs = []
    for i in range(n_tasks):
        task_id = f"task_{i:03d}"
        plateau = int(rng.integers(3, 6))                    # quality flat from here
        q_max = float(rng.uniform(0.7, 1.0))
        wallet_size = ["small", "medium", "large"][i % 3]    # per (task, group), §2.2
        allowance = wallets[wallet_size]
        for r in range(G):
            steps, spent, history = [], 0.0, []
            for t in range(1, T + 1):
                tier = tier_from_remaining(spent, allowance)
                q = q_max * min(1.0, t / plateau) + float(rng.normal(0, 0.005))
                draft_v = min(t, plateau)
                x = StepFeatures(
                    tokens_used=200 * t, tokens_pct=200 * t / 8192,
                    tool_calls=t, tool_pct=t / 20,
                    dollars=spent, dollars_pct=spent / allowance,
                    burn_rate=STEP_COST, tier=tier,
                    step_idx=t, steps_since_draft_changed=max(0, t - plateau),
                    draft_edit_distance_last3=[0.0 if t - j <= plateau else 0.0
                                               for j in range(3)],
                    retrieval_overlap_last3=min(1.0, 0.2 * max(0, t - plateau)),
                    n_distinct_sources=min(t, plateau),
                    draft=f"answer v{draft_v}", draft_len=10 + draft_v,
                    question=f"synthetic question {task_id}", domain="qa",
                    history=list(history[-3:]),
                )
                spent += STEP_COST
                history.append({"t": t, "action_type": "tool_call", "obs_digest": f"obs {t}"})
                steps.append(Step(x=x, a="tool_call", o=f"obs {t}", c=STEP_COST,
                                  tier=tier, draft=f"answer v{draft_v}", q=q))
            trajs.append(Trajectory(
                task_id=task_id, domain="qa", allowance_B=allowance,
                wallet_size=wallet_size, group_id=f"grp_{i:03d}", rollout_idx=r,
                steps=steps,
                outcome={"Q_tau": steps[-1].q, "success": True, "tau": None,
                         "collection_mode": "forced_continuation"},
            ))
    return trajs


@pytest.fixture(scope="module")
def trajs():
    return make_synthetic_trajectories()


@pytest.fixture(scope="module")
def labelsets(trajs):
    return [snell_labels(trajs, lam, MEDIAN_PILOT_SPEND, seed=0)
            for lam in (0.5, 2.0)]


@pytest.fixture(scope="module")
def examples(trajs, labelsets):
    return ds.build_examples(trajs, labelsets)


# ------------------------------------------------------------------ dataset
def test_build_examples_pools_all_lambdas(trajs, labelsets, examples):
    n_steps = sum(len(t) for t in trajs)
    assert len(examples) == n_steps * len(labelsets)          # ONE pooled dataset (§2.3)
    assert {e.lam for e in examples} == {0.5, 2.0}
    for e in examples:
        assert e.action in ("STOP", "CONTINUE")
        assert -1.0 <= e.delta_norm <= 1.0
        assert np.isfinite(e.v_star)


def test_serialized_input_carries_lambda_and_features(examples):
    by_lam = {}
    for e in examples:
        by_lam.setdefault(e.lam, e)
    assert "λ = 0.5" in by_lam[0.5].text                      # λ-conditioning in input (§18.1)
    assert "λ = 2" in by_lam[2.0].text
    e = by_lam[0.5]
    assert "[DRAFT]" in e.text and "[BUDGET]" in e.text and "[OBJECTIVE]" in e.text
    assert f"synthetic question {e.task_id}" in e.text


def test_split_by_task_never_splits_within_a_task(examples):
    train, hold = ds.split_by_task(examples, heldout_frac=0.34, seed=42)
    train_ids = {e.task_id for e in train}
    hold_ids = {e.task_id for e in hold}
    assert train_ids and hold_ids
    assert not (train_ids & hold_ids)                          # no leakage
    assert train_ids | hold_ids == {e.task_id for e in examples}
    assert len(train) + len(hold) == len(examples)
    # deterministic
    train2, hold2 = ds.split_by_task(examples, heldout_frac=0.34, seed=42)
    assert {e.task_id for e in hold2} == hold_ids
    # a task's examples land entirely on one side
    for tid in hold_ids:
        assert all(e.task_id != tid for e in train)


def test_examples_jsonl_roundtrip(examples, tmp_path):
    path = tmp_path / "examples.jsonl"
    n = ds.save_examples(examples[:50], path)
    assert n == 50
    loaded = ds.load_examples(path)
    assert len(loaded) == 50
    for a, b in zip(examples[:50], loaded):
        assert a == b


def test_labelset_jsonl_roundtrip(labelsets, tmp_path):
    ls = labelsets[0]
    path = tmp_path / "labels.jsonl"
    ds.save_labelset(ls, path)
    back = ds.load_labelset(path)
    assert back.lam == ls.lam and back.domain == ls.domain
    assert back.scale_s == pytest.approx(ls.scale_s)
    assert back.tau_star == ls.tau_star
    assert len(back.labels) == len(ls.labels)
    assert back.labels[0] == ls.labels[0] and back.labels[-1] == ls.labels[-1]


def test_serialize_context_from_config():
    from cassi.common.config import load_config
    ctx = ds.SerializeContext.from_config(load_config())
    assert ctx.t_max("qa") == 10 and ctx.t_max("alfworld") == 20   # §17 executor.horizon


# --------------------------------------------------------------- eval_regret
def test_labels_have_interior_stops(trajs, labelsets):
    """The synthetic economy must produce mid-trajectory τ* (rise-then-plateau)."""
    for ls in labelsets:
        taus = list(ls.tau_star.values())
        assert len(taus) == len(trajs)
        assert any(t < T for t in taus)


def test_oracle_mock_zero_regret_perfect_f1(trajs, labelsets):
    oracle = MockStopper.oracle(labelsets)
    for ls in labelsets:
        out = evaluate_stopper(oracle, trajs, ls)
        m = out["metrics"]
        assert m["mean_regret"] == pytest.approx(0.0, abs=1e-12)  # stops exactly at τ*
        assert m["f1_stop"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["n_trajectories"] == len(trajs)
        for rec in out["records"]:
            assert rec.tau_stopper == rec.tau_star
    multi = evaluate_multi_lambda(oracle, trajs, labelsets)
    assert multi["mean_regret"] == pytest.approx(0.0, abs=1e-12)


def test_majority_mock_is_worse(trajs, labelsets):
    for ls in labelsets:
        majority = MockStopper.majority_class(ls)
        m = evaluate_stopper(majority, trajs, ls)["metrics"]
        oracle_m = evaluate_stopper(MockStopper.oracle(ls), trajs, ls)["metrics"]
        assert m["mean_regret"] > oracle_m["mean_regret"] + 1e-6   # strictly worse utility
        assert m["f1_stop"] < 1.0


def test_tau_star_matches_labelset(trajs, labelsets):
    ls = labelsets[0]
    idx = labels_by_trajectory(ls)
    for traj in trajs:
        labs = idx[(traj.task_id, traj.group_id, traj.rollout_idx)]
        assert tau_star_of(labs) == ls.tau_star[(traj.task_id, traj.rollout_idx)]


def test_p4_baseline_comparison(trajs, labelsets):
    """§16 P4: the (here: oracle) stopper must beat majority-class and the
    calibrated draft-stability probe on held-out regret."""
    ls = labelsets[1]
    task_ids = sorted({t.task_id for t in trajs})
    train_ids, hold_ids = set(task_ids[:4]), set(task_ids[4:])
    train_trajs = [t for t in trajs if t.task_id in train_ids]
    hold_trajs = [t for t in trajs if t.task_id in hold_ids]
    from cassi.stopper.train_sft import filter_labelset
    train_ls, hold_ls = filter_labelset(ls, train_ids), filter_labelset(ls, hold_ids)

    k = calibrate_draft_stability(train_trajs, train_ls)
    assert 1 <= k <= 8

    out = compare_p4_baselines(
        MockStopper.oracle(ls), train_trajectories=train_trajs, train_labelset=train_ls,
        heldout_trajectories=hold_trajs, heldout_labelset=hold_ls)
    arms = out["arms"]
    assert set(arms) == {"stopper", "majority_class", f"draft_stability_k{out['probe_k']}"}
    stopper_regret = arms["stopper"]["mean_regret"]
    assert stopper_regret == pytest.approx(0.0, abs=1e-12)
    assert arms["majority_class"]["mean_regret"] > stopper_regret
    assert arms[f"draft_stability_k{out['probe_k']}"]["mean_regret"] >= stopper_regret


def test_records_csv(trajs, labelsets, tmp_path):
    out = evaluate_stopper(MockStopper.constant(1.0), trajs, labelsets[0])
    path = write_records_csv(out["records"], tmp_path / "regret.csv")
    lines = Path(path).read_text().strip().splitlines()
    assert len(lines) == len(trajs) + 1
    assert lines[0].startswith("task_id,")


def test_draft_stability_mock_is_deterministic():
    probe = MockStopper.draft_stability(2)
    x_hot = StepFeatures(steps_since_draft_changed=3)
    x_cold = StepFeatures(steps_since_draft_changed=1)
    for _ in range(3):
        assert probe.predict(x_hot, 1.0).action == "STOP"
        assert probe.predict(x_cold, 1.0).action == "CONTINUE"


# --------------------------------------------------------- torch-free imports
def test_stopper_modules_import_without_torch():
    """Import every cassi.stopper module in a subprocess where torch/transformers
    are BLOCKED — proves the lazy-import contract (model.py docstring)."""
    research_dir = str(Path(__file__).resolve().parents[2])
    script = f"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        root = name.split('.')[0]
        if root in ('torch', 'transformers', 'trl'):
            raise ImportError('blocked: ' + name)
        return None
sys.meta_path.insert(0, _Block())
sys.path.insert(0, {research_dir!r})
import cassi.stopper.dataset
import cassi.stopper.model
import cassi.stopper.eval_regret
import cassi.stopper.train_sft
from cassi.stopper.model import MockStopper
from cassi.common.schema import StepFeatures
p = MockStopper.constant(-0.5).predict(StepFeatures(), 1.0)
assert p.action == 'STOP' and p.delta == -0.5
print('TORCH_FREE_OK')
"""
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "TORCH_FREE_OK" in res.stdout


def test_train_sft_cli_parses():
    from cassi.stopper.train_sft import parse_args
    args = parse_args(["--labels", "a.jsonl", "b.jsonl",
                       "--trajectories", "t.jsonl", "--out", "/tmp/x"])
    assert args.labels == ["a.jsonl", "b.jsonl"] and args.out == "/tmp/x"
