# cost-aware-agent — Vision

> What we are building and why.

---

## The Problem

AI coding agents have no economic judgment.

They cannot assess whether their next action is worth its cost. They will improve, refactor, polish, and iterate forever — not because the work is valuable, but because they are trained to complete tasks and be helpful. The helpfulness training has no off switch. There is no sense of diminishing returns. There is no "good enough."

Every token the agent outputs costs real money. The agent does not know this. It does not factor this in. It will spend $2.50 polishing code that was good enough at $0.30 — and it will never notice the difference.

---

## How a Human Engineer Thinks

A senior engineer does not work until exhausted. They make constant cost-value decisions:

- "This refactor takes 3 hours but saves zero bugs → not worth it, skip"
- "This test takes 30 min and catches the critical path → worth it, write it"
- "The fix compiles and tests pass → ship it, don't polish further"
- "I've been debugging 4 hours with no progress → escalate, don't keep digging"
- "This feature adds 2 weeks of work for one edge-case user → reject it"

The engineer stops not because time ran out, but because they judged that **continuing produces less value than it costs**. This is economic reasoning. It is the core skill that separates senior engineers from junior ones.

---

## Why AI Cannot Do This

The model has no access to:

- What its output actually costs in dollars
- Whether the marginal improvement justifies the marginal cost
- When "good enough" is actually good enough
- When further work enters diminishing returns

This is not a capability problem. Frontier models — Opus 4.8, GPT-5, Gemini 3.1 Pro — are fully capable of economic reasoning when given the right inputs. The problem is structural: RLHF training rewards task completion and helpfulness. It never rewards the agent for stopping. There is no training signal for "I chose not to do this because it wasn't worth the cost." So the model never learns to say no.

When a user says "improve this again," the model improves it again. Always. Because every token of improvement is a reward-consistent output. Saying "this is not worth improving" is not.

---

## What We Are Building

We are building a harness that gives AI coding agents the cost-side of the trade-off equation, so they can make the same economic judgments a human engineer makes.

The agent should be able to reason:

> "I've already solved the core problem. The tests pass. What I'm about to do next will cost more than the value it produces. The right decision is to stop."

This is not about enforcing limits. This is not about cutting off the agent when money runs out. Any LLM API already has spend limits and rate limits — we do not need to rebuild that.

This is about teaching the agent that its decisions have a cost, and that the cost matters to the decision.

---

## What Success Looks Like

**Without this harness:**

User asks: "Improve the performance of this function."

Agent improves. Then improves again. Then micro-optimizes. Then refactors surrounding code. Then adds more tests. Never stops unless the user explicitly intervenes. Spends $2.50 on work that was complete and correct at $0.30.

**With this harness:**

Agent improves the function. Tests pass. The goal is met. Agent evaluates: "Current state satisfies the task. Further optimization would cost more than the improvement is worth." Agent stops and explains why.

Session cost: $0.30. Task complete.

---

## The Core Insight

Frontier models are capable of economic reasoning. They just never receive the inputs required to do it.

The harness provides those inputs.
