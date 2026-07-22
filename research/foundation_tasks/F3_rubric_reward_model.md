# F3 — The rubric & the prompted reward model (the piece that must be perfect)

**Goal:** the reward spec for the RL arm: a fixed, versioned rubric the frozen
Qwen3.6-27B judge scores every step against, with a calibration gate that must pass before any
RL run consumes its scores.

## Principles (the definition of "perfect" here)

1. **Exact things are computed exactly, never judged.** F1 vs gold, step counts,
   budget arithmetic, format compliance — all code, zero judge involvement. The
   judge only supplies genuine *judgments* no formula gives us.
2. **The judge never sees gold.** Its inputs are inference-available only (task,
   history digests, drafts, budget state). This keeps the reward signal honest to
   what a deployed monitor could compute, and preserves comparability with the
   full plan's RM design.
3. **Every rubric bit is binary, with a written decision criterion and 2 worked
   examples (one YES, one NO) inside the judge prompt.** No "rate 1–10".
4. **Weights are designed and documented, then frozen before RL.** Fitting weights
   to outcomes would be training a reward model by the back door (v2.1 §20's
   argument — kept).
5. **Versioned:** `rubric_v1` in the config and in every judge-output record; any
   edit after calibration bumps the version and re-runs calibration.
6. **Calibration gate:** the judge must agree with human labels before use (below).

## Prior art reviewed: PPTAgent trajectory-eval rubrics (lab-internal, well-tested)

Reviewed 2026-07-22 at `/mnt/src/code/PPT-GEN-Demo/eval_codes/trajectory_eval/`
(`trajectory_evaluation_prompts.md` + `evaluation_prompts.py`; also present under
`/mnt/src/jiageng/PPTAgent/eval_codes/`). Six LLM-judge dimensions for deep-research
trajectories (search query quality, planner, retrieval, search-integrated
reasoning, summarization, generation), 1–5/1–10 anchored scales, weighted composite.

**Adopted from them:**
- **Behaviorally-anchored definitions** (their strongest tested practice): every
  score level has a written description of what that behavior looks like. Our
  binary bits adopt the same discipline — each bit's YES/NO criterion is written
  as concrete behavior, plus worked examples (already principle 3).
- **Criteria language for our bits**: their "Retrieval Efficiency" criteria
  (duplicate searches, redundant/overlapping queries, search count vs task
  complexity, follow-ups refined from results) directly sharpen the definitions
  of `new_info` and `not_redundant` in the judge prompt.
- **Designed weighted composite** (their 0.15/0.20/... weights): same approach as
  our designed frozen weights — reassuring precedent that this is lab practice.
- **Their fuller rubric reused where it fits**: the six-dimension diagnostic is
  adapted (task-adjusted) for the F7 *analysis*, as an offline explanation layer —
  e.g., "did RL also improve query formulation, or only stopping?" Eval-time
  only, human-read, never a reward.

**Deliberately NOT copied, with reasons:**
- **1–10 / 1–5 absolute scales as reward**: their rubrics are *offline evaluation*
  read by humans; ours is a *reward consumed by RL*. Coarse absolute scales are
  noisy step-to-step (same behavior, different score), and RL will Goodhart a
  vague scale far more easily than crisp binary bits (v2.1 §20's argument).
- **Essay outputs** (strengths/weaknesses/recommendations): valuable in a report,
  pure cost and parse-surface in a reward call.
- **Their planner / multi-modal / generation dimensions**: our single-agent QA
  setting has no planner, no figures, no long-form generation.
- **The economics**: nothing in their rubrics asks "was this step worth its cost
  given the remaining budget?" — that question IS our paper; it has to be ours.

**Escalation path if the calibration gate fails:** first fix is pulling MORE of
their tested criteria language into the failing bit's definition and examples
(their wording survived real use; ours hasn't yet) — before inventing new bits
or loosening the gate. "Well-tested" transfers per-task only: their pedigree
certifies evaluation on PPT/deepresearch trajectories; OUR rubric earns its
robustness through the calibration gate on OUR trajectories, nowhere else.

Provenance note: these are colleagues' internal codes — fine to adapt in-lab
(recorded here); ask before any verbatim text ships in a public artifact.

## Rubric v1 — proposal (REVIEW THIS SECTION HARDEST)

Judge input per step t: task, steps 1..t history (action + observation digest ≤64
tokens each), draft before and after step t, budget state (t, B, remaining).

**For every non-ANSWER step — "was taking this step a good economic decision?"**

| Bit | Question the judge answers | Weight |
|---|---|---|
| `new_info` | Did this step's observation add relevant information not already in the history? | +0.4 |
| `not_redundant` | Is this step's query non-duplicative of an earlier step's (would a careful person have issued it)? | +0.3 |
| `was_needed` | BEFORE this step, was the draft still insufficient to answer the task? (If the draft already sufficed, the step was waste.) | +0.3 |

`step_score = 0.4·new_info + 0.3·not_redundant + 0.3·was_needed` ∈ [0,1];
per-step reward `r_t = α·(step_score − 0.5)`, α = 0.2 (config). A useless step is
mildly negative, a useful one mildly positive — dawdling never pays.

**For the ANSWER step — "was stopping now the right economic call?"**

| Bit | Question | Effect |
|---|---|---|
| `supported` | Is the final answer consistent with the collected evidence? | +0.5 |
| `nothing_left` | Is further search unlikely to change the answer (either evidence is complete, or the budget context makes more search a bad trade)? | +0.5 |

`stop_score` maps to `r_τ = α·(stop_score − 0.5)` the same way.

**Terminal reward (code, not judge):** `R_final = F1(answer, gold) − λ·(steps_used/B)
+ 0.1·format_ok`. Total trajectory reward = R_final + Σ r_t.

## Judge plumbing

- Client for the lab vLLM Qwen3.6-27B server; batched, retry-on-parse-failure
  (one reprompt, then the step scores 0.5 = neutral, and the failure is logged —
  neutral, not zero, so parser failures don't masquerade as "bad step" signals).
- Output format: strict JSON (`{"new_info": 0|1, ...}` after a short CoT field);
  parser + pytest with a mocked client.
- **Cache** keyed on (rubric_version, serialized input hash) — reruns are free.
- Every call logged (count + tokens) for the overhead report (F6).

## Calibration gate (before any RL)

1. Sample 50 steps from pilot trajectories, stratified: early/mid/late steps,
   ANSWER steps, obviously-redundant steps.
2. Brian hand-labels every rubric bit (a labeling sheet script makes this ~1 hour;
   plain-language instructions included — this doubles as the test that the rubric
   is answerable by a careful human at all).
3. Gate: **≥80% per-bit agreement** judge-vs-human, no bit below 70%. Fail → fix
   prompt/criteria, bump rubric version, re-run. Log the confusion table.

## Done criterion

Calibration gate passed and logged; cache + parser tests green; rubric v_final
frozen in config.

Depends on: F2 (pilot trajectories). Feeds: F5 (rewards), F6 (overhead numbers).
