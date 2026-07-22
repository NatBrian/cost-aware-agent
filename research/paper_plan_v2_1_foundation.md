# Paper Plan v2.1 FOUNDATION — the simplified pipeline-validation version

> **Status:** DRAFT for review (2026-07-22) — no code written yet; review this + the
> `research/foundation_tasks/F0–F7` docs, then we build.
> **Relationship to `paper_plan_v2_1.md`:** v2.1 stays the full paper plan and is NOT
> replaced. This document is the *foundation run*: a deliberately small, end-to-end
> version of the same idea, ordered by the senior's directive — prove the idea works
> and that RL proves it, on a robust pipeline, before spending the full v2.1 budget.
> **Where they conflict, this document governs the foundation work only;** v2.1 still
> governs the eventual full paper.
> **The old `research/cassi/` codebase will be archived** (moved under
> `research/archived/`, which is read-banned by CLAUDE.md). The foundation is built
> fresh so its correctness never silently depends on code we no longer trust.

---

## 0. Why this exists (plain language)

The v2.1 plan is a full ICLR paper: 2 training domains, ~10 baselines, a trained
reward model, dollar-denominated costs, 30+ training runs. Before spending that, we
want a small version that answers ONE question end-to-end:

> **Can RL training with per-step, budget-aware rewards from a prompted big reward
> model teach a ReAct agent to stop itself at the right time — better than just
> telling it about the budget (prompt) or forcing it to stop (harness)?**

If yes: the core bridge (economic step signal → RL → internalized stopping) works,
and every v2.1 component we later add (trained RM, dollars, more domains, more
baselines) rests on a pipeline we have already run end-to-end.
If no: we find out for a couple hundred tasks' worth of compute, and we learn
*which stage* failed, because every stage has its own gate.

## 1. The simplifications (v2.1 → foundation), each with its reason

| # | v2.1 (full plan) | Foundation | Why simpler is right for now |
|---|---|---|---|
| 1 | Cost = multi-dimensional dollars (tokens + tool fees + API), pilot-normalized, tier-scaled | **Cost = number of agent steps** | Steps are trivially countable, comparable across runs, and directly measure "how far the multi-hop loop went". No price map, no normalization pilot, no tiers. |
| 2 | Budget harness = dollar wallets w/ tiers | **Harness = step budget B** (hard stop + forced answer at B) | Same concept, one integer. |
| 3 | 10 baselines B1–B10 | **2 baselines**: ReAct + budget-in-system-prompt; ReAct + step-budget harness | The minimum pair that brackets the method: "just tell it" vs "just force it". |
| 4 | Reward model = trained 2B stopper (RM-T) on Snell-envelope labels; prompted 35B (RM-P) is baseline B10 | **Reward model = prompted Qwen3.6-27B (RM-P style) only**, per-step rubric scores; **no SFT/trained RM** | SFT takes the most wall-clock. The RL architecture stays identical (GRPO + per-step process rewards) — only the *source* of the step reward changes. The trained RM returns in the full plan. |
| 5 | Rubric = one baseline's scoring prompt | **Rubric = the method's reward spec — designed to be airtight** (own subtask doc F3, calibration gate before RL) | The rubric is now the reward source; a sloppy rubric invalidates the whole run. |
| 6 | 8–10K QA + 2K ALFWorld trajectories, 2 domains | **HotpotQA only, ~300 train tasks × G=8 rollouts, ~200 frozen dev tasks** | "Couple hundred training samples, multiple same runs" — enough for GRPO groups, small enough to iterate fast. |
| 7 | ≥2 training-loop iterations, λ sweep ×5, 3 seeds, Holm–Bonferroni stats | 1 iteration, 1 headline λ (+1 spot-check), 1 seed, bootstrap CIs only | Foundation proves direction and magnitude, not significance tables. |
| 8 | Kill-switches K1/K2, transfer evals, contamination protocol, OOD suite | One pre-registered gate (§6) + per-stage done-criteria | Same spirit (decide cheaply, log the decision), one page instead of ten. |

**Explicitly deferred to the full plan (not abandoned):** trained RM-T + Snell labels,
dollar costs, ALFWorld + OOD transfer, the loop (iteration 2), B2–B9 baselines,
potential-based shaping invariance machinery, full statistics protocol.

## 2. Setup (all shared scaffolding, all three arms)

- **Task:** HotpotQA multi-hop QA with local Wikipedia retrieval (search tool),
  the standard Search-R1-style env. Quality = F1/EM of final answer vs gold —
  free to compute, no judge needed.
- **A step** = one ReAct iteration: thought → one tool call (search) → observation.
  Emitting `ANSWER` ends the episode (that step is counted). Hard cap T_max = 10.
- **Step budget B** ∈ {small, medium, large} (provisionally {3, 6, 10}; calibrated
  by a 50-task unconstrained pilot in F2, then frozen). Every episode draws a B;
  within a GRPO group all G rollouts share the same B (never confound group
  advantages with budget luck — carried over from v2.1 §2.2).
- **Shared scaffold for ALL arms** (so it is a constant, never an advantage): the
  system prompt states the budget and steps used so far; the agent emits a
  running `BEST ANSWER SO FAR:` draft line each step (carried from v2.1 §2.6 —
  it is what the judge and the harness read).
- **Utility (the economy):** `U = F1(final answer) − λ·(steps_used / B)`.
  λ dimensionless; headline λ = 0.5 with a λ = 0.1 spot-check (both provisional,
  finalized after the pilot; logged before any RL run).
- **Models:** executor = Qwen3.5-9B (enable_thinking=False; drop to 4B only if
  GPU-constrained — decision logged). Judge/reward model = **Qwen3.6-27B** on the
  lab's vLLM server (frozen, inference only, never trained).

## 3. The three arms

| Arm | Name | What it is | What it answers |
|---|---|---|---|
| A0 | **Plain ReAct** (reference floor) | Frozen executor; NO budget information anywhere (no B in prompt, no tracker); stops only by its own ANSWER or T_max. One run of dev-200 (behavior can't depend on B), re-scored under each budget's utility | "How does the model behave with no cost signal at all?" — the no-information floor (v2.1's B1); without it, A1's behavior can't be attributed to the budget info |
| A1 | **Prompted-budget ReAct** (baseline) | Frozen executor; budget tracker block injected each step (§2/F2); nothing enforced | "Is telling the model about the budget enough?" |
| A2 | **Harness-enforced ReAct** (baseline) | Same frozen executor + prompt; harness force-stops at B and forces a final answer from the draft | "Is external throttling enough?" (this is the inference-time-control world) |
| A3 | **RL-internalized** (the method) | Same executor, GRPO-trained with per-step judge rewards + terminal economic reward (§4); evaluated **with the harness OFF** (harness-on also reported) | "Does training move the economics *into the policy*?" — the paper's core claim, miniaturized |

## 4. Reward design for A3 (summary — full spec in F3)

Two components, each computed by the cheapest *reliable* source:

1. **Terminal reward (exact, from gold — no judge):**
   `R_final = F1(answer, gold) − λ·(steps_used / B)`, plus a small format term.
   Gold answers exist at training time; anything computable exactly is computed
   exactly, never judged (v2.1 §20's own argument, kept).
2. **Per-step process reward (the prompted Qwen3.6-27B judge):** after rollouts finish,
   the judge scores every step against a **fixed, versioned binary rubric**
   (gold-free inputs: task, history digest, draft, budget state). The weighted
   rubric score maps to a small per-step reward
   `r_t = α·(step_score − 0.5)` (α provisionally 0.2). The judge also scores the
   stop decision at the ANSWER step (too early / appropriate / too late).
   Rubric bits, weights, judge prompt, parser, caching, and the **calibration
   gate** (judge must agree with ~50 hand-labeled steps at ≥80% per bit before
   any RL run) are all in `foundation_tasks/F3_rubric_reward_model.md`.

Credit assignment: per-step returns-to-go with group normalization (the same
step-level scheme v2.1 mandates), Dr. GRPO length hygiene, KL anchor. This is
honest per-step reward (not potential-based shaping); the invariance machinery
returns with the trained RM in the full plan, and the foundation report must say so.

**Honesty items carried over:** judge calls are counted and reported (billing
symmetry, in call-counts instead of dollars); judge-score-vs-true-reward divergence
is logged during training (the reward-hacking diagnostic — with a frozen prompted
judge this is the single most likely failure, and we want the curve either way).

## 5. Experiment sequence (maps 1:1 to subtask docs)

| Stage | Doc | What happens | Gate to pass before next stage |
|---|---|---|---|
| F0 | `F0_repo_restructure.md` | Archive `cassi/` → `research/archived/`; fresh `research/foundation/` package + test scaffold; CLAUDE.md updated | tests green; CLAUDE.md points here |
| F1 | `F1_data.md` | HotpotQA 300-train / 200-dev frozen samples + local retrieval index | manifest with counts + hashes committed |
| F2 | `F2_harness_trajectories.md` | ReAct agent, step-budget harness, **the trajectory collection script** (JSONL schema, per-step draft scoring, G=8); 50-task pilot → freeze B values and λ | one collected batch validates against schema; pilot memo written |
| F3 | `F3_rubric_reward_model.md` | Rubric spec + judge client + parser + cache | **calibration gate:** ≥80% per-bit agreement with hand labels |
| F4 | `F4_baselines.md` | A1 and A2 on dev-200 at all three budgets | both baselines' numbers on the frozen dev set, reported |
| F5 | `F5_rl_training.md` | GRPO on 300 tasks, judge rewards, 1 seed; micro-run (10 tasks) smoke gate first | training completes; divergence dashboard exists; checkpoint saved |
| F6 | `F6_evaluation.md` | A3 (harness-off and -on) vs A1/A2 on dev-200 × 3 budgets; metrics + bootstrap CIs | all numbers as CSVs with a generation script |
| F7 | `F7_analysis_report.md` | Figures (F1 steps-vs-F1 frontier, F2 internalization bars) + plain-language report in `experiments/reports/` | **the foundation GO/NO-GO verdict (§6), logged with date** |

## 6. The pre-registered gate (what "the idea works" means)

**GO** iff, on the frozen dev-200 at the medium budget:
1. A3 (harness OFF) achieves **higher utility U than both A1 and A2**, and
2. A3 self-terminates (emits ANSWER before B without enforcement) in **≥70% of
   episodes**, and
3. A3's F1 is not catastrophically below A2's (≥ A2 − 5 points) — savings must not
   come from answer-quality collapse.

**NO-GO** → diagnose in stage order (data → trajectories → rubric → RL) using each
stage's logged artifacts; fix and rerun the cheapest failing stage. Only if the
pipeline is verified sound and A3 still loses do we treat it as evidence against
the method — and that verdict, with traces, is exactly what the senior asked the
foundation to surface early. Either way the decision is logged in the F7 report.

**How to read the outcome (scope of the verdict — stated before running):**
- A **GO** proves the pipeline works end-to-end and that per-step economic RL beats
  prompting and enforcement *in this setup*. It does NOT prove v2.1's specific
  potential-based shaping mechanism (that math requires the trained RM, deferred).
- A **NO-GO with a sound pipeline** is evidence against the method *as instantiated
  with a frozen prompted judge* — the judge (v2.1's own predicted weak point) may be
  the culprit, not the idea. The cheap follow-up in that case: train the 2B RM-T on
  the foundation's already-collected trajectories (hours, not weeks, once this
  pipeline exists) and rerun F5 — i.e., the foundation's data and code are exactly
  what makes that next test cheap.

## 7. Metrics (all of them, nothing else)

- F1 / EM on dev-200 (mean, 95% bootstrap CI, 10k resamples, paired per task)
- Mean steps used; distribution of stop steps
- Utility U at each budget; steps-vs-F1 curve across the three budgets
- Internalization: % self-stopped pre-budget (A3 harness-off); A3 harness-off vs
  harness-on gap
- Judge overhead: total judge calls, calls per trajectory (billing symmetry)
- Training diagnostics: judge-score vs realized-F1 divergence curve; reward and
  entropy curves

## 8. Infrastructure notes

- RL stack: same as v2.1 — verl (GRPO, AgentLoop multi-turn) + the Search-R1-style
  retrieval env; judge via the lab vLLM server running Qwen3.6-27B. No new frameworks.
- GPU ritual (unchanged): `eval $(/mnt/src/zhanka/gpu_acquire.sh N)` before, 
  `/mnt/src/zhanka/gpu_release.sh` after; N=2 collection, N=4–8 GRPO; never kill
  occupier processes.
- Git identity: NatBrian only (repo rule; verify `git config user.name` →
  "Nathanael Brian" before committing).
- Every run writes a config snapshot + seed + git hash next to its outputs;
  every figure/table regenerates from CSVs by script. A per-experiment
  plain-language report goes in `experiments/reports/` (standing preference).

## 9. Rough timeline (after doc review)

- Week 1: F0 + F1 + F2 (restructure, data, harness, collection script, pilot)
- Week 2: F3 + F4 (rubric + calibration, baselines) — F5 micro-run smoke at end
- Week 3: F5 (full GRPO run) + F6 (evaluation)
- Week 4: F7 (analysis, report, GO/NO-GO) + buffer

## 10. Open items for YOUR review (flagged, not decided)

1. Budgets {3, 6, 10} and λ ∈ {0.5 headline, 0.1 spot} are placeholders until the
   F2 pilot — confirm you're happy deciding them from the pilot memo.
2. Rubric bits + weights in F3 are proposed, not sacred — please read F3 closely;
   it is the piece you asked to be "perfect".
3. Executor 9B vs 4B (speed/GPU trade) — default 9B unless you prefer faster loops.
4. Whether A2 (harness baseline) should also get a "wrap up now" warning one step
   before B (more realistic harness) or a plain hard stop — currently: plain hard
   stop, warning variant noted as optional.
