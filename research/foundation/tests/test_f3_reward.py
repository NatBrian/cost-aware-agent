"""I3 tests: rubric math, judge client (cache/parse/neutral), rewards, calibration."""

import json

import pytest

from agent.harness import EpisodeSpec, run_episode
from reward.calibration import agreement, make_labeling_sheet
from reward.judge_client import JudgeClient, neutral_bits
from reward.rewards import (episode_rewards, format_ok, returns_to_go,
                            step_rewards, terminal_reward)
from reward.rubric import (ANSWER_BITS, STEP_BITS, render_answer_prompt,
                           render_step_prompt, step_reward, step_score)
from tests.test_f2_harness import FakeLLM, FakeRetriever, spec, step_out

W_STEP = {"new_info": 0.4, "not_redundant": 0.3, "was_needed": 0.3}
RUBRIC_CFG = {"alpha": 0.2, "step_bits": W_STEP,
              "answer_bits": {"supported": 0.5, "nothing_left": 0.5},
              "calibration": {"per_bit_gate": 0.80, "per_bit_floor": 0.70}}


def make_episode(outputs):
    return run_episode(spec(), FakeLLM(outputs), FakeRetriever())


# ---------- rubric math (the 8-level table from the plan discussion) ----------

def test_step_reward_levels():
    r = lambda a, b, c: step_reward(
        {"new_info": a, "not_redundant": b, "was_needed": c}, W_STEP, 0.2)
    assert r(1, 1, 1) == pytest.approx(0.10)
    assert r(0, 0, 0) == pytest.approx(-0.10)
    assert r(1, 0, 0) == pytest.approx(-0.02)   # score 0.4
    assert r(0, 1, 1) == pytest.approx(0.02)    # score 0.6
    assert step_score(neutral_bits(STEP_BITS), W_STEP) == pytest.approx(0.5)  # -> reward 0


def test_worked_example_returns_to_go():
    """The exact numbers from the plan review: rewards [.1,-.04,.1,.1], F1=0.8,
    steps 4, B 6, lambda .5 -> RTG [0.727, 0.627, 0.667, 0.567]."""
    r_final = 0.8 - 0.5 * (4 / 6)
    rtg = returns_to_go([0.10, -0.04, 0.10, 0.10], r_final)
    assert [round(x, 3) for x in rtg] == [0.727, 0.627, 0.667, 0.567]


def test_terminal_reward_and_format():
    ep = make_episode([step_out("search", "q", "d"),
                       step_out("answer", "Rosie Mac", "Rosie Mac")])
    assert format_ok(ep) == 1.0
    assert terminal_reward(ep, lam=0.5, fmt_w=0.1) == pytest.approx(
        1.0 - 0.5 * (2 / 3) + 0.1)
    ep_bad = make_episode([step_out("search", "q", "d"), "x", "x",
                           step_out("answer", "Rosie Mac", "Rosie Mac")])
    assert format_ok(ep_bad) == 0.0            # malformed step present


# ---------- prompts are gold-free ----------

def test_judge_prompts_never_contain_gold():
    ep = make_episode([step_out("search", "who is X", "EMPTY_DRAFT"),
                       step_out("answer", "Someone Else", "Someone Else")])
    for i, s in enumerate(ep["steps"]):
        text = (render_answer_prompt(ep, i) if s["action_type"] == "answer"
                else render_step_prompt(ep, i))
        assert "Rosie Mac" not in text          # the gold answer never leaks


# ---------- judge client ----------

class ScriptedJudge(JudgeClient):
    """JudgeClient with transport replaced by a script; exercises real parse/cache."""

    def __init__(self, replies, tmp, **kw):
        super().__init__(endpoint="http://real:1", model="m",
                         rubric_version="rubric_v1", cache_dir=tmp, **kw)
        self._replies = list(replies)

    def _complete(self, prompt):
        return self._replies.pop(0)


def test_judge_parses_and_caches(tmp_path):
    good = json.dumps({"reasoning": "ok", "new_info": 1, "not_redundant": 0,
                       "was_needed": 1})
    j = ScriptedJudge([good], tmp_path)
    bits = j.judge("PROMPT-A", STEP_BITS)
    assert bits["new_info"] == 1 and bits["not_redundant"] == 0
    again = j.judge("PROMPT-A", STEP_BITS)     # no reply left -> must hit cache
    assert again["new_info"] == 1
    assert j.stats.cache_hits == 1 and j.stats.calls == 1


def test_judge_retries_then_neutral(tmp_path):
    j = ScriptedJudge(["not json", "still not json"], tmp_path)
    bits = j.judge("PROMPT-B", STEP_BITS)
    assert bits.get("_neutral") is True
    assert step_score(bits, W_STEP) == pytest.approx(0.5)   # zero reward
    assert j.stats.parse_failures == 1 and j.stats.calls == 2


def test_judge_rejects_placeholder_endpoint(tmp_path):
    with pytest.raises(ValueError, match="placeholder"):
        JudgeClient("PLACEHOLDER", "m", "rubric_v1", tmp_path)


# ---------- episode rewards end-to-end (mock judge) ----------

class ConstJudge:
    def __init__(self, val=1):
        self.val = val

    def judge(self, prompt, bit_names):
        return {b: self.val for b in bit_names}


def test_episode_rewards_shapes_and_malformed_penalty():
    ep = make_episode([step_out("search", "q1", "d"), "x", "x",
                       step_out("answer", "Rosie Mac", "Rosie Mac")])
    cfg = {"rubric": RUBRIC_CFG, "economy": {"lambda": 0.5},
           "reward": {"format_weight": 0.1}}
    out = episode_rewards(ep, ConstJudge(1), cfg)
    assert len(out["step_rewards"]) == len(ep["steps"]) == 3
    assert out["step_rewards"][0] == pytest.approx(0.10)
    assert out["step_rewards"][1] == pytest.approx(-0.10)   # malformed = worst
    assert out["step_rewards"][2] == pytest.approx(0.10)
    assert len(out["returns_to_go"]) == 3
    assert out["returns_to_go"][0] == pytest.approx(
        sum(out["step_rewards"]) + out["r_final"])


# ---------- calibration ----------

def test_labeling_sheet_and_agreement(tmp_path):
    eps = []
    for k in range(10):
        ep = make_episode([step_out("search", f"q{k}a", "EMPTY_DRAFT"),
                           step_out("search", f"q{k}b", "draft"),
                           step_out("answer", "final", "final")])
        eps.append(ep)
    pilot = tmp_path / "pilot.jsonl"
    pilot.write_text("".join(json.dumps(e) + "\n" for e in eps))
    sheet = tmp_path / "sheet.csv"
    n = make_labeling_sheet(pilot, sheet, n=12, seed=1)
    assert n == 12
    # simulate Brian labeling everything 1
    lines = sheet.read_text().splitlines()
    header = lines[1].split(",")
    out = [lines[0], lines[1]]
    import csv as _csv
    import io
    rows = list(_csv.DictReader(io.StringIO("\n".join(lines[1:]))))
    for r in rows:
        bits = ANSWER_BITS if r["action_type"] == "answer" else STEP_BITS
        for b in bits:
            r[f"label_{b}"] = "1"
    buf = io.StringIO()
    w = _csv.DictWriter(buf, fieldnames=header)
    w.writeheader()
    w.writerows(rows)
    sheet.write_text(lines[0] + "\n" + buf.getvalue())

    rep = agreement(sheet, ConstJudge(1), RUBRIC_CFG)
    assert rep["passed"] and rep["mean_agreement"] == 1.0
    rep_bad = agreement(sheet, ConstJudge(0), RUBRIC_CFG)
    assert not rep_bad["passed"]
