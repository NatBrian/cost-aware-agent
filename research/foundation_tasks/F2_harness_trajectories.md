# F2 — Step-budget harness, ReAct agent, and THE trajectory script

**Goal:** the load-bearing piece of the foundation: a ReAct agent with a step-budget
harness and a trajectory collection script whose JSONL output every later stage
(judge, RL, eval, analysis) consumes. If this schema is right, everything downstream
is plumbing.

## Design

**A step** = thought → one search call → observation (or the ANSWER emission, which
counts as a step and ends the episode). T_max = 10 hard cap for all arms.

**Shared scaffold (ALL arms — it must be a constant, never an advantage):**
- Each turn the harness injects a budget tracker block — the step-unit analog of
  `cost_aware_agent/prompts.py::render_budget_tracker` (the repo's tested design),
  inheriting its facts-not-advice philosophy (its §6 comment: "the harness states
  measured facts and delegates ALL judgment to the model — we measure the MODEL's
  economics, not our rule text"):

  ```
  <budget>
  Steps used: {t} of {B}. Remaining: {B−t}.
  Decide yourself what these numbers mean for your next step.
  </budget>
  ```

  The delegation line is fixed and identical every turn. NO prescriptive advice
  ("wrap up soon", "prefer cheap actions") anywhere in any arm's prompt — advice
  would mean measuring our rule text instead of the agent's economics, and would
  contaminate A1 (whose entire purpose is "facts only, nothing enforced").
  The repo's richer machinery (checklists, SELF-VERIFICATION, spend-audit
  checkpoints, streaks) is deliberately NOT ported — extra scaffold would be a
  confound shared by all arms and belongs to the harness product, not this
  experiment.
- The system prompt additionally states the task and requires each step's output
  to end with `BEST ANSWER SO FAR: {one line | EMPTY_DRAFT}`.
- The harness parses the draft line every step; malformed output → one retry, then
  `EMPTY_DRAFT` logged (edge case must be in tests).

**Harness modes** (flag on the collection script):
- `none` — nothing enforced (arm A1; also RL-training rollouts, which terminate
  naturally at ANSWER or T_max).
- `enforce` — at step B: episode cut, final answer = last draft (arm A2).
- `forced_continuation` — ANSWER logged as an event but the episode continues to
  T_max (carried from v2.1 §2.1; used ONLY for the pilot and the oracle analysis
  in F7, so we can see the full quality-vs-steps curve of each question).

## Trajectory JSONL schema (per episode)

```json
{"task_id": str, "arm": str, "budget_B": int, "seed": int, "config_hash": str,
 "steps": [{"t": int, "action_type": "search|answer", "query_or_answer": str,
            "obs_digest": str, "draft": str, "draft_f1_vs_gold": float,
            "tokens_in": int, "tokens_out": int}],
 "answered_at": int|null, "forced_stop": bool, "final_answer": str,
 "final_f1": float, "final_em": float, "steps_used": int}
```

Per-step `draft_f1_vs_gold` is a free string comparison at collection time
(training-side only; never shown to the agent or the judge).

## The 50-task pilot (decides the frozen constants)

Run 50 train tasks, G=4, `forced_continuation`, no budget. From the resulting
quality-vs-steps curves write a one-page pilot memo proposing:
- budget values {small, medium, large} (provisional {3, 6, 10}),
- headline λ (provisional 0.5) — pick λ so that at the medium budget, stopping at
  the observed knee of the quality curve beats both never-stopping and
  stopping immediately (sanity: the economy must make the problem non-trivial),
- T_max confirmation (is 10 enough for the hard stratum?).
User reviews the memo; constants then frozen in `configs/foundation.yaml`.

## Work items

1. ReAct agent (Qwen3.5-9B via vLLM) + prompts (`agent/prompts.py`, versioned).
2. Step-budget harness with the three modes.
3. `collect/run_collection.py`: task file in → JSONL out; G rollouts/task; budget
   drawn per (task, group) shared across the group's G rollouts; resumable
   (skips completed task_ids); config snapshot + git hash written next to output.
4. Schema validator (`collect/schema.py`) + pytest coverage: draft parsing,
   EMPTY_DRAFT, forced stop, forced continuation, resumability, F1 scoring.
5. Run the pilot; write the memo.

## Done criterion

Pilot memo written; one 20-task collection batch in each harness mode validates
against the schema; tests green.

Depends on: F0, F1. Feeds: F3 (judge reads these), F4, F5, F7.
