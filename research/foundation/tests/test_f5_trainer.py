"""E-d trainer tests (CPU side): grouping + sample building with fake tokenizer."""

from train.grpo_trainer import build_samples, group_episodes


class FakeTok:
    def apply_chat_template(self, msgs, add_generation_prompt=True,
                            tokenize=True, **kwargs):
        return list(range(sum(len(m["content"]) for m in msgs) % 50 + 5))

    def encode(self, text, add_special_tokens=False):
        return [7] * len(text.split())


def mk_ep(task="t1", n_steps=2, lps=True, budget_B=4):
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "Question: q?"}]
    steps, advs = [], []
    for t in range(1, n_steps + 1):
        msgs.append({"role": "user", "content": f"<budget>{t}</budget>"})
        reply = f"THOUGHT: x\nACTION: search[q{t}]\nBEST ANSWER SO FAR: d"
        msgs.append({"role": "assistant", "content": reply})
        n_words = len(reply.split())
        steps.append({"t": t, "asst_idx": len(msgs) - 1,
                      "logprobs": [-0.1] * n_words if lps else []})
        advs.append(0.5 - t * 0.1)
        msgs.append({"role": "user", "content": "Results: ..."})
    return {"task_id": task, "messages": msgs, "steps": steps,
            "advantages": advs, "budget_B": budget_B}


def test_group_episodes_by_task_and_budget():
    eps = [mk_ep("a"), mk_ep("b"), mk_ep("a")]
    gs = group_episodes(eps)
    assert sorted(len(g) for g in gs) == [1, 2]


def test_group_episodes_never_mixes_budgets():
    """Same task at two wallets must NOT share an advantage group: z-scoring
    '3 steps of 8' against '3 steps of 2' bakes budget luck into the advantage
    (plan §2). One wallet per (task, group) makes this a tripwire, not a knob."""
    eps = [mk_ep("a", budget_B=2), mk_ep("a", budget_B=8), mk_ep("a", budget_B=2)]
    gs = group_episodes(eps)
    assert sorted(len(g) for g in gs) == [1, 2]
    for g in gs:
        assert len({e["budget_B"] for e in g}) == 1


def test_build_samples_aligns_tokens_and_advantages():
    samples, stats = build_samples([[mk_ep(n_steps=3)]], FakeTok())
    assert stats["kept"] == 3 and stats["len_mismatch"] == 0
    for s in samples:
        assert len(s["reply_ids"]) == len(s["old_logprobs"])
        assert isinstance(s["advantage"], float)
    # advantages preserved in order
    assert [round(s["advantage"], 1) for s in samples] == [0.4, 0.3, 0.2]


def test_build_samples_drops_missing_logprobs_and_counts():
    samples, stats = build_samples([[mk_ep(lps=False)]], FakeTok())
    assert samples == [] and stats["no_logprobs"] == 2


def test_build_samples_tolerates_small_mismatch():
    ep = mk_ep(n_steps=1)
    ep["steps"][0]["logprobs"] = ep["steps"][0]["logprobs"][:-1]  # off by one
    samples, stats = build_samples([[ep]], FakeTok())
    assert stats["kept"] == 1
    s = samples[0]
    assert len(s["reply_ids"]) == len(s["old_logprobs"])


def test_build_samples_drops_large_mismatch():
    ep = mk_ep(n_steps=1)
    ep["steps"][0]["logprobs"] = [-0.1] * 3   # reply has ~10 words
    _, stats = build_samples([[ep]], FakeTok())
    assert stats["len_mismatch"] == 1
