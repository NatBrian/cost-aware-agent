"""I2 tests: parser, QA scoring, episode loop x 3 modes, schema, budgets, resume."""

import json

import pytest

from agent.harness import EpisodeSpec, run_episode
from agent.prompts import EMPTY_DRAFT, parse_step, system_prompt, tracker_block
from collect.run_collection import completed_keys, draw_budget
from collect.schema import validate_episode
from eval.qa_metrics import em, f1


# ---------- fakes ----------

class FakeLLM:
    """Returns scripted step outputs in order; records every message list."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def chat(self, messages, temperature=0.0):
        self.calls.append([dict(m) for m in messages])
        return self.outputs.pop(0)

    def chat_with_logprobs(self, messages, temperature=0.0, want_logprobs=True):
        return (self.chat(messages, temperature),
                [-0.5, -0.1] if want_logprobs else None,
                {"prompt_tokens": 100, "completion_tokens": 20})


class FakeRetriever:
    def search(self, query):
        return [{"title": "Doc", "text": f"Info about {query}.", "score": 0.87}]

    def format_observation(self, hits):
        return "\n".join(f"[{i+1}] {h['title']}: {h['text']}" for i, h in enumerate(hits))


def step_out(action, content, draft):
    return f"THOUGHT: thinking.\nACTION: {action}[{content}]\nBEST ANSWER SO FAR: {draft}"


def spec(**kw):
    base = dict(task_id="t1", question="Who is X?", golds=["Rosie Mac"],
                arm="a1", mode="none", budget=3, t_max=5, temperature=0.0,
                seed=42, config_hash="abc", draft_retry=1)
    base.update(kw)
    return EpisodeSpec(**base)


# ---------- parsing ----------

def test_parse_valid_search_and_answer():
    p = parse_step(step_out("search", "capital of France", "EMPTY_DRAFT"))
    assert p == {"action_type": "search", "content": "capital of France",
                 "draft": EMPTY_DRAFT}
    p = parse_step(step_out("answer", "Paris", "Paris"))
    assert p["action_type"] == "answer" and p["draft"] == "Paris"


def test_parse_rejects_malformed():
    assert parse_step("THOUGHT: hm.\nno action here") is None
    assert parse_step("ACTION: search[]") is None


def test_parse_missing_draft_line_defaults_empty():
    assert parse_step("ACTION: search[x]")["draft"] == EMPTY_DRAFT


def test_qa_scoring_normalizes():
    assert em("The Apalachees", ["Apalachees"]) == 1.0
    assert f1("Rosie Mac was the double", ["Rosie Mac"]) > 0.5
    assert f1("", ["x"]) == 0.0


# ---------- episode modes ----------

def test_mode_none_ends_at_answer():
    llm = FakeLLM([step_out("search", "who is X", "EMPTY_DRAFT"),
                   step_out("answer", "Rosie Mac", "Rosie Mac")])
    ep = run_episode(spec(), llm, FakeRetriever())
    validate_episode(ep)
    assert ep["answered_at"] == 2 and ep["steps_used"] == 2
    assert ep["final_answer"] == "Rosie Mac" and ep["final_f1"] == 1.0
    assert not ep["forced_stop"]


def test_mode_enforce_cuts_at_budget_and_uses_draft():
    llm = FakeLLM([step_out("search", "q1", "EMPTY_DRAFT"),
                   step_out("search", "q2", "Rosie Mac")])
    ep = run_episode(spec(mode="enforce", budget=2), llm, FakeRetriever())
    validate_episode(ep)
    assert ep["forced_stop"] and ep["steps_used"] == 2
    assert ep["final_answer"] == "Rosie Mac" and ep["final_em"] == 1.0


def test_mode_forced_continuation_logs_answer_and_continues():
    llm = FakeLLM([step_out("search", "q1", "EMPTY_DRAFT"),
                   step_out("answer", "Rosie Mac", "Rosie Mac"),
                   step_out("search", "q3", "Rosie Mac"),
                   step_out("search", "q4", "Rosie Mac"),
                   step_out("search", "q5", "Rosie Mac")])
    ep = run_episode(spec(mode="forced_continuation"), llm, FakeRetriever())
    validate_episode(ep)
    assert ep["answered_at"] == 2 and ep["steps_used"] == 2
    assert ep["total_steps_run"] == 5          # ran to t_max
    assert ep["final_answer"] == "Rosie Mac"


def test_malformed_step_retries_then_consumes_step_and_keeps_draft():
    llm = FakeLLM([step_out("search", "q1", "Rosie Mac"),
                   "garbage", "still garbage",
                   step_out("answer", "Rosie Mac", "Rosie Mac")])
    ep = run_episode(spec(), llm, FakeRetriever())
    validate_episode(ep)
    assert ep["steps"][1]["action_type"] == "malformed"
    assert ep["steps"][1]["draft"] == "Rosie Mac"   # draft never erased
    assert ep["answered_at"] == 3


def test_a0_sees_no_budget_content_and_a1_does():
    llm0 = FakeLLM([step_out("answer", "x", "x")])
    run_episode(spec(arm="a0"), llm0, FakeRetriever())
    joined0 = json.dumps(llm0.calls[0])
    assert "<budget>" not in joined0 and "budget of" not in joined0
    llm1 = FakeLLM([step_out("answer", "x", "x")])
    run_episode(spec(arm="a1"), llm1, FakeRetriever())
    joined1 = json.dumps(llm1.calls[0])
    assert "<budget>" in joined1 and "Decide yourself" in joined1


def test_tracker_and_system_prompt_render():
    assert "Steps used: 2 of 6. Remaining: 4." in tracker_block(2, 6)
    assert "budget of 6 steps" in system_prompt(True, 6)
    assert "budget" not in system_prompt(False).lower()


# ---------- schema, budgets, resume ----------

def test_schema_rejects_bad_episode():
    llm = FakeLLM([step_out("answer", "x", "x")])
    ep = run_episode(spec(), llm, FakeRetriever())
    bad = dict(ep, final_f1=2.0)
    with pytest.raises(ValueError, match="final_f1"):
        validate_episode(bad)
    bad = dict(ep, steps=[])
    with pytest.raises(ValueError, match="no steps"):
        validate_episode(bad)


def test_draw_budget_deterministic_and_policy():
    budgets = {"small": 3, "medium": 6, "large": 10}
    assert draw_budget("medium", budgets, "t1", 42) == 6
    a = draw_budget("draw", budgets, "t1", 42)
    assert a == draw_budget("draw", budgets, "t1", 42)   # same task -> same wallet
    drawn = {draw_budget("draw", budgets, f"t{i}", 42) for i in range(50)}
    assert drawn == {3, 6, 10}                            # all wallets occur


def test_completed_keys_resume(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text(json.dumps({"task_id": "t1", "rollout": 0}) + "\n" +
                 json.dumps({"task_id": "t1", "rollout": 1}) + "\n")
    assert completed_keys(p) == {("t1", 0), ("t1", 1)}
    assert completed_keys(tmp_path / "missing.jsonl") == set()


def test_train_mode_captures_messages_logprobs_and_asst_idx():
    llm = FakeLLM([step_out("search", "q1", "d"),
                   step_out("answer", "Rosie Mac", "Rosie Mac")])
    ep = run_episode(spec(train_mode=True), llm, FakeRetriever())
    assert "messages" in ep and ep["messages"][0]["role"] == "system"
    for s in ep["steps"]:
        assert s["logprobs"] == [-0.5, -0.1]
        # asst_idx points at THIS step's reply in the message list
        assert ep["messages"][s["asst_idx"]]["role"] == "assistant"
        assert ep["messages"][s["asst_idx"]]["content"].startswith("THOUGHT")
    # non-train mode stays lean
    llm2 = FakeLLM([step_out("answer", "x", "x")])
    ep2 = run_episode(spec(), llm2, FakeRetriever())
    assert "messages" not in ep2 and "logprobs" not in ep2["steps"][0]


def test_retry_tokens_are_separable_from_work_tokens():
    """A retry re-sends the whole conversation, so prompt_tokens double on a
    malformed step. That is a real cost and is counted -- but n_retries and the
    first-attempt counts must also be recorded, or a token comparison between two
    arms with different malformed rates cannot be decomposed afterwards. The
    2026-08 token figures are permanently inflated because these were missing."""
    from agent.harness import EpisodeSpec, run_episode
    llm = FakeLLM([
        "garbage with no action line",                       # forces one retry
        "THOUGHT: t\nACTION: answer[Paris]\nBEST ANSWER SO FAR: Paris",
    ])
    spec = EpisodeSpec(task_id="t", question="q?", golds=["Paris"], arm="a1",
                       mode="none", budget=2, t_max=5, temperature=0.0, seed=1,
                       config_hash="h", draft_retry=1)
    ep = run_episode(spec, llm, FakeRetriever())
    s = ep["steps"][0]
    assert s["n_retries"] == 1, "retry count not recorded"
    # totals include both attempts; first-attempt fields isolate the original call
    assert s["prompt_tokens"] == 200 and s["first_attempt_prompt_tokens"] == 100
    assert s["completion_tokens"] == 40 and s["first_attempt_completion_tokens"] == 20
    # the overhead is therefore recoverable
    overhead = s["prompt_tokens"] - s["first_attempt_prompt_tokens"]
    assert overhead == 100


def test_clean_step_records_zero_retries():
    from agent.harness import EpisodeSpec, run_episode
    llm = FakeLLM(["THOUGHT: t\nACTION: answer[Paris]\nBEST ANSWER SO FAR: Paris"])
    spec = EpisodeSpec(task_id="t", question="q?", golds=["Paris"], arm="a1",
                       mode="none", budget=2, t_max=5, temperature=0.0, seed=1,
                       config_hash="h", draft_retry=1)
    s = run_episode(spec, llm, FakeRetriever())["steps"][0]
    assert s["n_retries"] == 0
    assert s["prompt_tokens"] == s["first_attempt_prompt_tokens"]
