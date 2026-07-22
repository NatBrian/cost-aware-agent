"""I5 tests: advantages (normalization + min-cohort guard), batch rewards,
divergence log, dry-run."""

import numpy as np
import pytest

from common import load_config
from train.advantages import group_step_advantages, trajectory_returns
from train.grpo_runner import NeutralJudge, dry_run
from train.reward_adapter import DivergenceLog


def test_advantages_normalize_within_cohort():
    rtgs = [[1.0, 0.5], [0.0, -0.5], [2.0, 1.5], [-1.0, -1.5]]
    adv = group_step_advantages(rtgs, min_cohort=3)
    for t in range(2):
        col = np.array([a[t] for a in adv])
        assert abs(col.mean()) < 1e-6           # zero-mean per step position
        assert col.std() == pytest.approx(1.0, abs=1e-3)
    order = np.argsort([r[0] for r in rtgs])
    assert np.argsort([a[0] for a in adv]).tolist() == order.tolist()


def test_min_cohort_guard_falls_back_to_trajectory_baseline():
    # 4 trajs; only ONE reaches step 3 -> its step-3 advantage must equal its
    # trajectory-level z-score, not a self-comparison (which would be 0/eps).
    rtgs = [[1.0, 0.8, 0.6], [0.5], [0.2], [-0.5]]
    adv = group_step_advantages(rtgs, min_cohort=3)
    totals = trajectory_returns(rtgs)
    expected = (totals[0] - totals.mean()) / (totals.std() + 1e-6)
    assert adv[0][2] == pytest.approx(float(expected))
    assert adv[0][1] == pytest.approx(float(expected))   # cohort of 1 at t=1 too
    col0 = np.array([a[0] for a in adv])                  # full cohort at t=0
    assert abs(col0.mean()) < 1e-6


def test_advantages_shapes_match_variable_lengths():
    rtgs = [[0.1] * 5, [0.2] * 2, [0.3] * 7, [0.0] * 4]
    adv = group_step_advantages(rtgs)
    assert [len(a) for a in adv] == [5, 2, 7, 4]


def test_divergence_log_records_and_saves(tmp_path):
    d = DivergenceLog()
    row = d.add([0.8, 0.6], [0.5], step=10)
    assert row == {"step": 10, "judge_score_mean": pytest.approx(0.7),
                   "f1_mean": pytest.approx(0.5)}
    p = tmp_path / "div.jsonl"
    d.save(p)
    assert p.read_text().count("\n") == 1


def test_dry_run_end_to_end():
    cfg = load_config()
    out = dry_run(cfg)
    assert out["episodes"] == 3 * cfg["grpo"]["group_size"]
    assert out["nonzero_advantages"] and out["divergence_rows"] == 1


def test_neutral_judge_is_nondegenerate():
    j = NeutralJudge()
    a = j.judge("p", ("x",))
    b = j.judge("p", ("x",))
    assert a != b     # alternates by design so wiring bugs can't hide
