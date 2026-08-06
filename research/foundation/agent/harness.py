"""The episode loop: ReAct agent + step-budget harness, three modes (F2).

Modes:
  none                — nothing enforced (arm a1; RL rollouts). Ends at agent's
                        answer[...] or t_max.
  enforce             — hard stop at budget B: episode cut, final answer = last
                        draft (arm a2).
  forced_continuation — answer[...] is logged (answered_at, answer_draft) but the
                        episode continues to t_max (pilot + oracle analysis).

The model client is injected (LLMClient protocol: chat(messages, temperature)
-> str), so tests drive episodes with a scripted fake; scripts use OpenAIChat.
"""

from dataclasses import dataclass

from agent.prompts import EMPTY_DRAFT, parse_step, system_prompt, tracker_block
from eval.qa_metrics import em, f1

MODES = ("none", "enforce", "forced_continuation")


@dataclass
class EpisodeSpec:
    task_id: str
    question: str
    golds: list[str]
    arm: str                  # a0 | a1 | a2 | a3
    mode: str                 # MODES
    budget: int               # B; ignored by a0 prompts but recorded
    t_max: int
    temperature: float
    seed: int
    config_hash: str
    draft_retry: int = 1
    train_mode: bool = False   # capture messages + sampled-token logprobs (F5)


def _final_from(draft: str) -> str:
    return "" if draft == EMPTY_DRAFT else draft


def run_episode(spec: EpisodeSpec, llm, retriever) -> dict:
    """Returns one episode dict in the F2 JSONL schema."""
    assert spec.mode in MODES, spec.mode
    with_budget = spec.arm != "a0"
    messages = [{"role": "system",
                 "content": system_prompt(with_budget, spec.budget)},
                {"role": "user", "content": f"Question: {spec.question}"}]
    steps: list[dict] = []
    answered_at: int | None = None
    answer_draft = ""
    final_answer: str | None = None
    forced_stop = False
    last_draft = EMPTY_DRAFT

    t = 0
    while t < spec.t_max:
        t += 1
        if with_budget:
            messages.append({"role": "user",
                             "content": tracker_block(t - 1, spec.budget)})
        # Tokens accumulate across the retry loop below, and a retry re-sends the
        # WHOLE conversation -- so a retried step's prompt_tokens is roughly double
        # a clean one's. That is a real cost and is counted, but it must be
        # SEPARABLE afterwards: without n_retries recorded, a token comparison
        # between two arms with different malformed rates silently mixes
        # "did less work" with "needed fewer retries", and no re-analysis can undo
        # it. Measured 2026-08-06: malformed steps carried 9-12x the tokens of
        # clean ones and the arms' malformed rates differed by up to 2x. (audit)
        tok_in = tok_out = 0
        raw, step_lps, usage = llm.chat_with_logprobs(
            messages, spec.temperature, want_logprobs=spec.train_mode)
        tok_in += usage["prompt_tokens"]
        tok_out += usage["completion_tokens"]
        first_in, first_out = usage["prompt_tokens"], usage["completion_tokens"]
        parsed = parse_step(raw)
        retries = 0
        while parsed is None and retries < spec.draft_retry:
            retries += 1
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": "Invalid format. Reply with exactly the "
                                        "THOUGHT / ACTION / BEST ANSWER SO FAR lines."})
            raw, step_lps, usage = llm.chat_with_logprobs(
                messages, spec.temperature, want_logprobs=spec.train_mode)
            tok_in += usage["prompt_tokens"]
            tok_out += usage["completion_tokens"]
            parsed = parse_step(raw)
        if parsed is None:
            parsed = {"action_type": "malformed", "content": "",
                      "draft": last_draft, "raw_excerpt": raw[:300]}
        messages.append({"role": "assistant", "content": raw})
        draft = parsed["draft"]
        if draft == EMPTY_DRAFT and last_draft != EMPTY_DRAFT:
            draft = last_draft            # a dropped line never erases the draft
        last_draft = draft

        step = {"t": t, "action_type": parsed["action_type"],
                "query_or_answer": parsed["content"], "obs_digest": "",
                "draft": draft,
                "draft_f1_vs_gold": f1(_final_from(draft), spec.golds),
                "raw_len": len(raw),
                # real cost of this step, not a character proxy (plan v2.2 §12).
                # *_tokens INCLUDE retry attempts; n_retries and first_attempt_*
                # let an analysis separate work done from format overhead.
                "prompt_tokens": tok_in, "completion_tokens": tok_out,
                "n_retries": retries,
                "first_attempt_prompt_tokens": first_in,
                "first_attempt_completion_tokens": first_out,
                # retrieval productivity; populated on search steps below
                "retrieval_scores": []}
        if "raw_excerpt" in parsed:
            step["raw_excerpt"] = parsed["raw_excerpt"]
        if spec.train_mode:
            # index of this step's FINAL assistant reply in the message list
            # (appended just above); retried garbage replies stay in messages
            # for context fidelity but only the final reply is trained on.
            step["asst_idx"] = len(messages) - 1
            step["logprobs"] = step_lps or []

        if parsed["action_type"] == "answer":
            if spec.mode == "forced_continuation":
                if answered_at is None:       # first ANSWER only
                    answered_at, answer_draft = t, parsed["content"]
                step["obs_digest"] = "(answer logged; continue searching)"
                messages.append({"role": "user",
                                 "content": "Your answer was noted. Continue "
                                            "researching to improve or verify it."})
                steps.append(step)
            else:
                answered_at, final_answer = t, parsed["content"]
                steps.append(step)
                break
        elif parsed["action_type"] == "search":
            hits = retriever.search(parsed["content"])
            obs = retriever.format_observation(hits)
            step["obs_digest"] = obs[:2000]
            step["retrieval_scores"] = [float(h.get("score", 0.0)) for h in hits]
            messages.append({"role": "user", "content": f"Results:\n{obs}"})
            steps.append(step)
        else:  # malformed after retry: step consumed, neutral observation
            step["obs_digest"] = "(malformed action)"
            messages.append({"role": "user", "content": "No action taken."})
            steps.append(step)

        if spec.mode == "enforce" and t >= spec.budget:
            forced_stop = True
            final_answer = _final_from(last_draft)
            break

    if final_answer is None:                  # t_max reached without answer
        final_answer = (answer_draft if spec.mode == "forced_continuation"
                        and answered_at is not None else _final_from(last_draft))

    steps_used = (answered_at if spec.mode == "forced_continuation"
                  and answered_at is not None else len(steps))
    extra = {"messages": messages} if spec.train_mode else {}
    return {**extra,
        "task_id": spec.task_id, "question": spec.question,
        "arm": spec.arm, "mode": spec.mode,
        "budget_B": spec.budget, "seed": spec.seed,
        "config_hash": spec.config_hash, "steps": steps,
        "answered_at": answered_at, "forced_stop": forced_stop,
        "final_answer": final_answer,
        "final_f1": f1(final_answer, spec.golds),
        "final_em": em(final_answer, spec.golds),
        "steps_used": steps_used,
        "total_steps_run": len(steps),
    }
