"""CPU tests for eval/run_frontier.evaluate_arm — the P5–P9 evaluation entry point.

Uses the same mock stack as the executor tests: ScriptedLLMClient + MockSearchEnv,
with the monitor's MockStopper. Verifies the §5.3 contracts: billing symmetry
(stopper queries are billed), self-termination tracking, and the dual-run regret."""

from __future__ import annotations

import csv

from cassi.eval.run_frontier import (
    INSTANCE_FIELDS, SUMMARY_FIELDS, _append_csv, evaluate_arm,
)
from cassi.executor.envs.base import MockSearchEnv
from cassi.executor.monitor import MockStopper, StopperMonitor
from cassi.executor.react_agent import ReactAgent, ScriptedLLMClient

CORPUS = {"d1": "Paris is the capital of France."}
TASKS = [{"task_id": f"t{i}", "question": "What is the capital of France?", "gold": "Paris"}
         for i in range(4)]

STEP_SEARCH = ("THOUGHT: look it up.\nACTION: search[capital of France]\n"
               "BEST ANSWER SO FAR: EMPTY_DRAFT")
STEP_SEARCH_DRAFT = ("THOUGHT: found it.\nACTION: search[France facts]\n"
                     "BEST ANSWER SO FAR: Paris")
STEP_ANSWER = "THOUGHT: done.\nACTION: answer[Paris]\nBEST ANSWER SO FAR: Paris"

ANSWERS_AT_3 = [STEP_SEARCH, STEP_SEARCH_DRAFT, STEP_ANSWER, STEP_SEARCH_DRAFT]
NEVER_ANSWERS = [STEP_SEARCH, STEP_SEARCH_DRAFT, STEP_SEARCH_DRAFT]


def _arm(outputs, monitor_factory=None, **kw):
    agent = ReactAgent(ScriptedLLMClient(outputs))
    defaults = dict(
        agent=agent, env=MockSearchEnv(CORPUS), tasks=TASKS,
        monitor_factory=monitor_factory, arm="test", lambda_dial=1.0,
        domain="qa", seed=42, t_max=6, allowance_dollars=0.05,
        self_termination=True, median_pilot_spend=0.001,
    )
    defaults.update(kw)
    return evaluate_arm(**defaults)


def test_self_terminating_policy_full_accuracy():
    summary, inst = _arm(ANSWERS_AT_3)
    assert summary["accuracy"] == 1.0                    # EM("Paris","Paris")
    assert summary["self_termination_rate"] == 1.0
    assert summary["n_tasks"] == 4 and len(inst) == 4
    assert all(r["stopped_by"] == "answer" and r["tau"] == 3 for r in inst)
    assert summary["stopper_cost_dollars"] == 0.0        # no monitor attached


def test_monitor_stops_and_is_billed():
    # monitor is consulted PRE-action: Δ̂≤0 first seen at t=3 → 2 executed steps
    factory = lambda: StopperMonitor(
        MockStopper(delta_fn=lambda t: 0.5 if t < 3 else -0.1), lam=1.0, t_max=6)
    summary, inst = _arm(NEVER_ANSWERS, monitor_factory=factory)
    assert all(r["stopped_by"] == "monitor" and r["tau"] == 2 for r in inst)
    assert summary["monitor_stop_rate"] == 1.0
    assert summary["self_termination_rate"] == 0.0
    assert summary["stopper_cost_dollars"] > 0.0         # billing symmetry (§5.3)
    # monitor stop at t=2: the draft exists, so accuracy survives
    assert summary["accuracy"] == 1.0


def test_regret_dual_run():
    summary, inst = _arm(ANSWERS_AT_3, with_regret=True)
    assert summary["mean_regret"] != ""
    assert float(summary["mean_regret"]) >= 0.0 - 1e-9   # replay frontier ≥ actual stop...
    # replay is billed to the analysis line, never the method (§5.3)
    assert summary["replay_cost_dollars_analysis_line"] > 0.0
    assert all(isinstance(r["regret"], float) for r in inst)


def test_csv_roundtrip(tmp_path):
    summary, inst = _arm(ANSWERS_AT_3)
    out = tmp_path / "frontier.csv"
    _append_csv(out, SUMMARY_FIELDS, [summary])
    _append_csv(out, SUMMARY_FIELDS, [summary])           # --append semantics
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    assert rows[0]["arm"] == "test" and float(rows[0]["accuracy"]) == 1.0
    inst_out = tmp_path / "frontier_instances.csv"
    _append_csv(inst_out, INSTANCE_FIELDS, inst)
    assert len(list(csv.DictReader(inst_out.open()))) == 4
