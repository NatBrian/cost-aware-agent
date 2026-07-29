"""I4 tests: metrics rows, bootstrap, pairing, sanity checks, gate logic."""

import numpy as np
import pandas as pd
import pytest

from eval.gate_check import evaluate_gate
from eval.metrics import (aggregate, bootstrap_ci, check_row_counts,
                          check_utility_recompute, episode_row, paired_delta)

LAM = 0.5


def ep(task="t1", arm="a1", B=6, f1=0.8, steps=4, answered=4, forced=False,
       mode="none"):
    return {"task_id": task, "arm": arm, "mode": mode, "budget_B": B,
            "rollout": 0, "final_f1": f1, "final_em": float(f1 == 1.0),
            "steps_used": steps, "answered_at": answered, "forced_stop": forced,
            "config_hash": "x"}


def frame(rows):
    return pd.DataFrame([episode_row(e, LAM) for e in rows])


def test_episode_row_utility_and_selfstop():
    r = episode_row(ep(), LAM)
    assert r["utility"] == pytest.approx(0.8 - 0.5 * 4 / 6)
    assert r["self_stopped"] == 1.0 and r["hit_cap"] == 0.0
    r2 = episode_row(ep(answered=None, forced=True), LAM)   # enforced cut
    assert r2["self_stopped"] == 0.0 and r2["hit_cap"] == 0.0
    r3 = episode_row(ep(answered=None, forced=False), LAM)  # ran to t_max
    assert r3["hit_cap"] == 1.0


def test_bootstrap_ci_brackets_mean_and_deterministic():
    v = np.array([0.2, 0.4, 0.6, 0.8] * 25)
    lo, hi = bootstrap_ci(v, 2000, seed=7)
    assert lo < v.mean() < hi
    assert (lo, hi) == bootstrap_ci(v, 2000, seed=7)


def test_aggregate_groups_by_arm_and_budget():
    df = frame([ep(task=f"t{i}", arm=a, f1=0.5 + 0.1 * (a == "a3"))
                for i in range(20) for a in ("a1", "a3")])
    agg = aggregate(df, n_resamples=500)
    assert set(agg.arm) == {"a1", "a3"} and (agg.n == 20).all()
    assert agg[agg.arm == "a3"].f1.iloc[0] > agg[agg.arm == "a1"].f1.iloc[0]


def test_paired_delta_uses_common_tasks():
    rows = [ep(task=f"t{i}", arm="a3", f1=0.7) for i in range(10)]
    rows += [ep(task=f"t{i}", arm="a1", f1=0.5) for i in range(10)]
    d = paired_delta(frame(rows), "a3", "a1", budget=6, metric="f1",
                     n_resamples=500)
    assert d["n"] == 10 and d["mean_delta"] == pytest.approx(0.2)
    assert d["frac_tasks_a_wins"] == 1.0


def test_sanity_checks_catch_corruption():
    df = frame([ep(task=f"t{i}") for i in range(5)])
    check_row_counts(df, 5)
    with pytest.raises(AssertionError, match="rows"):
        check_row_counts(df, 6)
    dup = frame([ep(task="t1"), ep(task="t1")])
    with pytest.raises(AssertionError, match="duplicated"):
        check_row_counts(dup, 2)
    check_utility_recompute(df, LAM)
    bad = df.copy()
    bad.loc[0, "utility"] += 0.1
    with pytest.raises(AssertionError, match="stale utility"):
        check_utility_recompute(bad, LAM)


def _gate_cfg(dev_size=20):
    return {"gate": {"budget": "medium", "min_self_stop": 0.70,
                     "f1_margin": 0.05, "bootstrap_resamples": 100},
            "economy": {"lambda": LAM},
            "data": {"dev_size": dev_size},
            "episode": {"budgets": {"small": 3, "medium": 6, "large": 10}}}


def test_gate_go_and_nogo():
    rows = []
    for i in range(20):
        rows.append(ep(task=f"t{i}", arm="a1", f1=0.55, steps=8, answered=8))
        rows.append(ep(task=f"t{i}", arm="a2", f1=0.55, steps=6, answered=None,
                       forced=True, mode="enforce"))
        rows.append(ep(task=f"t{i}", arm="a3", f1=0.60, steps=4, answered=4))
    res = evaluate_gate(frame(rows), _gate_cfg())
    assert res["verdict"] == "GO"
    assert all(res[k]["passed"] for k in
               ("cond1_utility", "cond2_self_stop", "cond3_no_collapse"))

    # collapse scenario: a3 saves steps but f1 craters -> cond3 fails
    rows_bad = [r for r in rows if r["arm"] != "a3"]
    rows_bad += [ep(task=f"t{i}", arm="a3", f1=0.30, steps=2, answered=2)
                 for i in range(20)]
    res_bad = evaluate_gate(frame(rows_bad), _gate_cfg())
    assert res_bad["verdict"] == "NO-GO"
    assert not res_bad["cond3_no_collapse"]["passed"]


def test_gate_requires_all_arms():
    rows = [ep(task="t1", arm="a1")]
    with pytest.raises(ValueError, match="no rows for arm"):
        evaluate_gate(frame(rows), _gate_cfg(dev_size=1))


def test_gate_ignores_a3_harness_on_rows():
    """A3 harness-off is the claim (plan §6). F6 also collects A3 harness-ON
    into the same CSV; if the gate selected on arm alone, those rows would be
    averaged into the headline and a losing A3 could be dragged over the line
    (or a winning one under it)."""
    rows = []
    for i in range(20):
        rows.append(ep(task=f"t{i}", arm="a1", f1=0.55, steps=8, answered=8))
        rows.append(ep(task=f"t{i}", arm="a2", f1=0.55, steps=6, answered=None,
                       forced=True, mode="enforce"))
        rows.append(ep(task=f"t{i}", arm="a3", f1=0.60, steps=4, answered=4))
        # same arm, harness ON: must not touch the verdict
        rows.append(ep(task=f"t{i}", arm="a3", f1=0.10, steps=6, answered=None,
                       forced=True, mode="enforce"))
    res = evaluate_gate(frame(rows), _gate_cfg())
    assert res["verdict"] == "GO"
    assert res["cond3_no_collapse"]["a3_f1"] == pytest.approx(0.60)


def test_canonical_rows_excludes_offarm_modes():
    """Every headline number must come from each arm's DEFINING mode. F6 emits
    A3 in three modes under one arm label, and selecting on arm alone silently
    averaged them — it hit gate_check, then report.py and figures.py (the A3 row
    read F1 .530 / self-stop 39% instead of the harness-off .560 / .775)."""
    from eval.metrics import canonical_rows
    rows = []
    for i in range(5):
        rows.append(ep(task=f"t{i}", arm="a3", f1=0.9))                       # off
        rows.append(ep(task=f"t{i}", arm="a3", f1=0.1, mode="enforce"))       # on
        rows.append(ep(task=f"t{i}", arm="a3", f1=0.5,
                       mode="forced_continuation"))                          # oracle
        rows.append(ep(task=f"t{i}", arm="a2", f1=0.4, mode="enforce"))
    out = canonical_rows(frame(rows))
    a3 = out[out.arm == "a3"]
    assert set(a3["mode"]) == {"none"} and len(a3) == 5
    assert a3.f1.mean() == pytest.approx(0.9)      # not the 3-mode blend
    assert set(out[out.arm == "a2"]["mode"]) == {"enforce"}
