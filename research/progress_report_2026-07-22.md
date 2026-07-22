# Progress Report — CASSI Experiments (2026-07-22)

> For review. Previous status you knew: we were discussing which RL
> architecture to use (prompted 30B reward model vs trained small reward model).
> This report covers everything since: the decision, the implementation, the
> first experiments, and the first results.

---

## 1. Summary in five lines

1. The architecture discussion was **resolved by agreement**: we test **BOTH
   reward models** — your prompted large model (RM-P) and the small trained
   one (RM-T) — inside the same paper, as a measured comparison.
2. All code was implemented and verified (125 automated tests pass).
3. The **first four experiment stages ran on real GPUs** this week:
   pipeline smoke test ✅ → cost calibration ✅ → 9,600-trajectory data
   collection ✅ → stopping-label computation ✅.
4. **First real results:** the "cost dial" works (higher price sensitivity ⇒
   provably earlier stopping, zero violations in 38,400 checks), and a bias
   predicted by our theory was confirmed in data.
5. Two evaluations are **running right now**: the small reward model is
   training toward its pass/fail gate, AND the prompted big model (your
   endpoint) is being scored on the **same held-out exam** — so the next
   result is a direct trained-vs-prompted table. Nothing so far contradicts
   the paper plan — the plan stands.

---

## 2. The architecture decision (what we agreed)

Both proposed architectures turned out to be the **same pipeline** —
executor → trajectories → reward model → RL update — differing only in ONE box:

| Reward model              | What it is                                                                                   | Role in the paper                                                                                                   |
| ------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **RM-T (trained)**  | small 2B model, trained a few hours on labels computed from gold answers + real logged costs | the paper's method (this is also where the novelty lives)                                                           |
| **RM-P (prompted)** | your vLLM Qwen3.6-35B endpoint, chain-of-thought + a designed binary rubric                  | baseline**B10** — so the trained-vs-prompted question is answered by OUR OWN results table, not by citations |

Your vLLM server was wired in and **live-tested end to end**: on a real
mid-task state it produced the rubric (1,0,0,1), correctly flagged a redundant
verification step, and decided STOP — at **$0.0004 per call**. It also gets two
more jobs later: the frozen strong agent in the transfer experiment, and
optional pre-screening for label review.

Everything is written down in `paper_plan_v2_1.md` (changes marked [v2.1],
full changelog in its §20) and `architecture_comparison.md`.

---

## 3. What was implemented (code)

- Plan v2.1 + baseline **B10** module (the RM-P judge: prompt, parser, designed
  weights, billing, RL adapter) — registry now has 11 baselines, **125 CPU
  tests pass** (was 115).
- Full experiment stack staged on our lab machine: datasets (17K QA tasks,
  decontaminated), the 61 GB Wikipedia retrieval index, both model weights.
- Two notable engineering fixes (details in the reports):
  - the retrieval server searched 21M passages on CPU at **35 s/query** — we
    replaced the search internals with a GPU matrix multiply: **46 ms**
    (~760× faster, mathematically identical results);
  - several library/hardware incompatibilities (no CUDA compiler on the
    machine, new Qwen3.5 architecture quirks) were found and fixed.

## 4. What was run, and the results

### Stage 0 — Pipeline smoke test ✅ PASSED

One question through the full loop (agent → search → step-by-step answer
drafts → per-step dollar costs). Total cost of one 10-step episode: **$0.0098**.

### Stage 1 — Cost calibration (200 questions, no budget limit) ✅ DONE

Measured what tasks naturally cost, then froze the three "wallet" sizes used
everywhere later: small **$0.0018** / medium **$0.0099** / large **$0.0404**
(typical task ≈ $0.0035).

### Stage 2 — Data collection (the harvest) ✅ DONE

**9,600 trajectories** = 1,200 questions × 8 attempts, each with a randomly
drawn wallet, each forced to run all 10 steps (so every possible stopping
point's future is observable — required for unbiased labels). Total simulated
spend: **$143**. Bonus measurement: the untrained agent already tries to
answer early in **68%** of episodes — the raw material our training will shape.

### Stage 3 — Stopping labels (the "answer key") ✅ DONE, all checks passed

For each of **96,000 steps** we computed, by backward dynamic programming
(the same math used to price stock options): was stopping here better than
continuing? At five price-sensitivity levels λ:

| λ (how much a dollar hurts)  | 0.1 | 0.5 | **1 (default)** | 2   | 5   |
| ----------------------------- | --- | --- | --------------------- | --- | --- |
| average optimal stopping step | 5.0 | 3.4 | **2.8**         | 1.1 | 1.0 |

**Read it as a dial:** cheap money ⇒ work ~5 steps; expensive money ⇒ stop
immediately. Perfectly monotone: **0 violations out of 38,400 checked pairs.**

Two extra findings:

- **Theory confirmed by data:** the naive label ("pick the best step in
  hindsight") stops **+0.41 steps later** than our correct labels — exactly
  the foresight bias our plan predicted and designed against. This becomes
  paper evidence (experiment E4).
- **Honest caveat:** string-match scoring counts "Drug Enforcement
  Administration" as wrong when gold says "DEA" — a known limitation of the
  standard scoring convention (kept for comparability with all baselines).

### Stage 4 — Training the small reward model ⏳ RUNNING NOW

The 2B model is learning to predict the labels from only
inference-available information (budget state, step number, draft stability —
never the gold answer). When done, it faces a **hard gate**: it must beat two
simple baselines on held-out data, or the pipeline stops for fixes — by design,
before any expensive RL training. (Training is slower than planned — ~6+ hours —
because the machine lacks the CUDA compiler needed for the new model family's
fast kernels; the fallback path is correct, just patient.)

### Stage 5 — Prompted big model on the SAME exam ⏳ RUNNING NOW

In parallel, the prompted reward model (RM-P: your Qwen3.6-35B endpoint,
chain-of-thought + designed rubric) is being scored on **exactly the same
held-out tasks and the same metric** the small model must pass: ~1,500 judge
calls, under $1. Result: the next report contains a four-way table on
identical data —

| trained 2B | prompted 35B | always-majority | draft-stability probe |
| ---------- | ------------ | --------------- | --------------------- |

— which answers the trained-vs-prompted architecture question we debated,
with our own measurements instead of citations. Either outcome is useful:
if the prompted judge holds its own, we learned it cheaply; if it trails
(as the published benchmarks predict), the paper's core design choice gets
its first direct evidence.

## 4b. Findings so far, in one paragraph

Everything measurable to date says the method's foundations behave as
designed: the pipeline runs end to end at realistic costs (~$0.01/episode);
the economics dial works (price-sensitivity λ moves the optimal stopping
point smoothly from step 5.0 to step 1.0, zero ordering violations in 38,400
checks); labels are stable under noise (±0.04 steps); and one theoretical
prediction was confirmed empirically before any model was trained (naive
hindsight labels stop +0.41 steps too late — the foresight bias our label
method exists to remove). The untrained agent already wants to stop early in
68% of episodes, which is the behavior the training will calibrate. No
result so far falsifies any paper claim; the two decisive tests (gate, K1)
are the next two results.

---

## 5. What's next (in order)

1. **The four-way reward-model table** (hours away): trained 2B vs prompted
   35B vs two trivial baselines, identical held-out exam. The 2B must beat
   the trivial baselines (hard gate) or the pipeline stops for fixes before
   any RL money is spent.
2. **Kill-switch experiment K1** — the pre-registered GO/NO-GO: does training
   the executor with the reward model's signal beat just using it as a runtime
   monitor (and beat direct label-shaping with no reward model)? This decides
   the paper's headline claim. Needs ~4 GPUs for a few days — the main
   schedule risk is GPU availability (see below).
3. Then the full baseline suite (the prompted RM-P becomes baseline B10 with
   its own cost-knob sweep), ablations, and the evaluation grid per the plan.

**One practical issue to raise:** the lab machine's GPUs are shared, and one
colleague's automation deploys across all 8 cards outside the reservation
system (it displaced our servers twice). We work around it by waiting for
stable free windows, but a simple agreement on a card split (e.g., 2 cards for
us now, 4 during the K1 week) would remove the main schedule risk.

---

## 6. Where to read more

Plain-language reports, one per experiment (written for non-specialists):
`research/cassi/experiments/reports/` — start with `README.md` (index), then
`00_smoke_pilot.md`, `01_collection.md`, `02_labels.md`.
Plans: `research/paper_plan_v2_1.md` (source of truth), `research/architecture_comparison.md`
(the resolved architecture discussion).

**Bottom line: no result so far contradicts the plan; the paper stands. The
two decisive experiments (reward-model gate, kill-switch K1) are next.**

---

# Appendix A — The prompted big model (RM-P / baseline B10), explained slowly

This appendix is the long version of Stage 5. It answers three things:
**what the code does**, **how the experiment is set up**, and **what results
exist right now**. Everything here lives in two files:

- `research/cassi/baselines/b10_prompted_rm.py` — the judge itself
- `research/cassi/scripts/eval_rmp_heldout.py` — the exam that scores it
- settings: `research/cassi/configs/cassi.yaml`, block `prompted_rm`

## A.1 The one-sentence idea

A reviewer will ask: *"Why train your own small reward model? Why not just
**ask a big smart model** whether the agent should stop?"* RM-P **is** that
question turned into an experiment. We ask the big model, we score it on the
same exam as the small trained model, and we print both numbers in the same
table. We do not settle the argument by quoting other papers.

## A.2 The architecture, in plain words

Think of the agent as a worker, and the reward model as a **supervisor** who
watches the worker and says "keep going" or "that's enough, hand it in".

- **RM-T (trained)** — a small 2B model we train to be that supervisor.
- **RM-P (prompted)** — your big 35B model, **frozen** (never trained, we only
  send it text), asked in a prompt to be that supervisor.

Both supervisors are shown **exactly the same thing**: a text block describing
the worker's current situation. Same information, no cheating: neither one
ever sees the correct answer, and neither sees the worker's own claim of
confidence. That is what makes it a fair comparison.

The pipeline is unchanged; only one box is swapped:

```
task → worker agent → step by step → [ SUPERVISOR ] → stop? / reward signal
                                        ↑ RM-T (trained 2B)
                                        ↑ RM-P (prompted 35B)   ← baseline B10
```

## A.3 What the judge actually receives (real input format)

The situation text is called `x_t` ("the state at step t"). It is built by
`stopper/features.py: serialize()`. A real one looks like this:

```
<stopper_input>
[TASK] Which agency did the officer who signed the 1973 reorganization plan lead?
[BUDGET] tokens 3120/8192 (38%) | tool calls 4/20 | $0.004/$0.01
         | tier MEDIUM | burn $0.0009/step
[OBJECTIVE] cost-sensitivity λ = 1
[PROGRESS] step 5/10 | draft unchanged for 2 steps
           | draft edit-distance (last 3 steps): 0.31,0.00,0.00
           | retrieval overlap (last 3): 87% | distinct sources: 3
[HISTORY] 3: search: "1973 reorganization plan signed by"
          4: search: "DEA first administrator 1973"
          5: search: "DEA first administrator 1973"
[DRAFT] Drug Enforcement Administration
</stopper_input>
```

Read it like a dashboard: *how much money/tokens are left*, *how expensive
money is today (λ)*, *how much the draft answer is still changing*, *whether
the last searches are just repeating each other*, and *what the current answer
is*. Here the draft has stopped changing and the last two searches were nearly
identical — a strong hint that more work is wasted money.

## A.4 What we ask the judge to do (the prompt)

We wrap that block in this instruction (`build_judge_prompt`):

> You are a strict reviewer of a working AI agent. Below is the agent's current
> state … **Think step by step BRIEFLY (2–4 sentences)**, then end your reply
> with EXACTLY one final line of the form:
> `RUBRIC: [a,b,c,d]`
> where each letter is 0 or 1:
> a = the current draft answer already looks correct
> b = the most recent step added information that was actually needed
> c = one more step of work would likely improve the final answer
> d = the remaining budget comfortably affords more work

So the judge gives a short reason, then **four yes/no answers**. Four simple
questions are easier for a model to answer reliably than "give me a score from
0 to 100".

## A.5 What the judge gives back (real output)

From the live test on your endpoint on 2026-07-21:

```
The draft "Drug Enforcement Administration" is already a specific, plausible
answer and it has not changed for two steps. The most recent search repeated
the previous one and returned the same passages, so it added nothing. Another
step would very likely repeat the same retrieval. Budget is still healthy.

RUBRIC: [1,0,0,1]
```

Cost of that call: **$0.0004** (379 input tokens, 88 output tokens).

## A.6 Turning four bits into two numbers

The trained model has two outputs, so the prompted judge must produce two
comparable numbers. We combine the bits with **fixed, hand-designed weights**:

| number             | formula                                    | meaning                              | mirrors              |
| ------------------ | ------------------------------------------ | ------------------------------------ | -------------------- |
| `continue_score` | 0.40·(1−a) + 0.30·c + 0.15·b + 0.15·d | "how much is continuing still worth" | the stop margin Δ̂ |
| `state_value`    | 0.70·a + 0.30·d                          | "how good is this situation"         | the value head V̂   |

For our example bits `[1,0,0,1]`:

- `continue_score` = 0.40·(1−1) + 0.30·0 + 0.15·0 + 0.15·1 = **0.15** → very low
  → **STOP** (it is below the stop threshold).
- `state_value` = 0.70·1 + 0.30·1 = **1.00** → this is a great place to be.

Note `(1 − a)`: an answer that already looks correct is the strongest *reason
to stop*, so it enters flipped.

**Important rule: those weights are chosen by hand and written into the paper
appendix — we never tune them on data.** If we tuned them, we would be
*training* a reward model, and the whole "trained vs prompted" comparison would
collapse into comparing a trained model with another trained model.

Two safety behaviours in the code:

- **Fail-open**: if the judge's reply is garbled and we cannot find the
  `RUBRIC:` line, we **never** stop. A broken judge can waste money, but it can
  never throw away a task.
- **Everything is billed**: each judge call's tokens (thinking included) are
  charged to RM-P's own account (`bill_judge`). The big model does not get to
  look cheap by having its own cost hidden.

## A.7 The two experiments we run with it

**Arm (a) — "monitor"**: no training at all. While the agent works, we call the
judge each step and stop when `continue_score ≤ θ_p`. θ_p is the **cost dial**:
raise it and the agent stops sooner and cheaper (but riskier); lower it and it
works longer. θ_p is chosen on a development set by
`calibrate_threshold`, which picks the highest threshold that still keeps
stop-decisions ≥90% correct. Cheap: inference only.

**Arm (b) — "RL"**: we actually train the worker agent with GRPO, but the
per-step reward comes from the *judge's* `state_value` instead of the trained
model's V̂. The reward formula is literally the same code
(`executor.shaping`) — **only where the number comes from changes.** This arm
is gated: it runs only after the K1 kill-switch says GO, 1 seed, QA only, one
cost point. We log how far the judge's score drifts from the true reward — that
is the "reward hacking" curve.

## A.8 The exam (how we score it side by side)

`scripts/eval_rmp_heldout.py` runs RM-P on the **same held-out questions, same
split seed, and same metric** that the trained 2B model must pass at its gate.
It converts the judge into the stopper's own language with

```
Δ̂ = 2 · continue_score − 1      (stop when Δ̂ ≤ 0, i.e. continue_score ≤ 0.5)
```

so both models are read on one ruler. Default run: 150 trajectories ≈ 1,500
judge calls ≈ well under $1, 8 calls in parallel against your vLLM server.
Command:

```
python scripts/eval_rmp_heldout.py --round 0 --domain qa --lam 1.0 --sample 150
```

Output goes to `experiments/stopper/round0/rmp_heldout_qa_lam1.json` and records
the metrics plus the honest cost (`judge_calls`, `parse_failures`,
`tokens_in/out`, `est_dollars`).

## A.9 Results — the honest status

**Numbers: not yet.** As of this report, the exam has been *live-verified end to
end on a single real state* (the $0.0004 call in §A.5, which produced correct,
sensible bits and the right STOP decision), and the full 150-trajectory run is
executing alongside the trained model's training. There is no
`rmp_heldout_qa_lam1.json` on disk yet, so **any number for RM-P at this point
would be invented.** The four-way table lands in the next report.

Two small setup items must be true when it runs (both are now in place, worth
knowing because they are the usual failure causes): `prompted_rm.base_url` must
point at your live server (`http://122.11.227.227:6101/v1`, model
`Qwen3.6-35B-A3B`), and θ_p must come from dev calibration rather than a guess.

**What we predicted, in advance, before seeing the result** (this is written
into the plan so we cannot move the goalposts later): RM-P should **lose**, for
three concrete reasons —

1. Published benchmarks show big models judge "was this step redundant?" at
   ≤24.9% F1 (RedundancyBench) — barely better than guessing.
2. Frozen judges get gamed once you train against them (AgentPRM: 82% → 70%).
3. Yes/no bits are too coarse for the training signal. The reward is a
   *difference*, `r_t = V̂(next) − V̂(this)`. If two neighbouring steps get the
   same four bits — which happens constantly — the difference is exactly zero
   and the agent learns nothing that step.

If RM-P wins anyway, that is a genuinely useful finding and we learned it for
under a dollar. Either way the paper gets a real measurement instead of an
argument.

---

# Appendix B — What we can borrow from PPTAgent's judge code

We reviewed the team's PPTAgent evaluation code, which also uses a big model as
a judge. The up-to-date version is **`/home/yangqu/PPTAgent/eval_codes/ppt_eval/`**
(33 files, 2026-07-21 — note the copy in `/home/liangsheng/PPTAgent` is an older
generation and should not be used as the reference). It has been refined over
many real runs, and several of its tricks apply directly to our prompted judge.

## B.1 Their five good ideas

**1. Decide the worst problem first, then cap the score.**
Their strict prompt never asks for a score straight away. It first asks the
judge to label the worst visible defect as
`none / minor / moderate / severe / fatal`, and each label puts a **hard ceiling**
on the final score (minor ⇒ at most 7, moderate ⇒ at most 5, severe ⇒ at most 3,
fatal ⇒ at most 2). The prompt then says, in plain words:

> "Do not average away serious defects… A content-rich slide with severe
> unreadable content still cannot exceed the corresponding cap."

Why this matters: judges have a well-known bad habit. Given one bad thing and
three good things, they average and give a 7. The cap removes that option — one
serious problem decides the score by itself.

**2. Let the judge say "this does not apply."**
Their SVG judge outputs the single word `false` when the slide has no diagram to
judge, instead of being forced to invent a number. Most of that prompt is a list
of things that **do not** count, so the judge cannot drift.

**3. The reason field is a fill-in form, not free writing.**
`"Organization: <judgment>. Information richness: <judgment>. … Overall: <band
rationale>."` The judge must visit every dimension out loud before scoring.

**4. Control what the judge is allowed to see.**
"This is a pure VLM evaluation: judge only from the image. Do not use source
HTML, DOM, or intermediate descriptions." This is the same discipline as our
rule that the judge never sees the gold answer.

**5. They measure the judge, not only with it.**
Their results file runs each model **three times over the same 100 cases** and
reports mean / median / min / max. It also runs **three prompt variants side by
side** on identical inputs, reporting each variant's average score *and* how
often it flagged each defect level.

## B.2 The number that should worry us

In their three-run table, the same model on the same 100 slides scored the
"logic" dimension **7.21 on run 1 and 7.59 on run 2**. Nothing changed except
random sampling — the gap is 0.4 points.

Our RM-P exam is currently a **single run**. If RM-P loses to the trained model
by a small margin, a reviewer will fairly ask: *is that a real difference, or is
that just the judge being noisy?* We would have no answer.

## B.3 Three concrete improvements, in priority order

### Improvement 1 — run the RM-P exam three times (do this first)

Small change, applies to the run already in flight, and useful no matter what
the result is. Run `eval_rmp_heldout.py` three times with different seeds and
report mean and spread instead of one number.

Cost: three runs × ~1,500 calls × $0.0004 ≈ **$2**. Effectively free.

Effect on the paper: the trained-vs-prompted table stops being "2B got X,
35B got Y" and becomes "2B got X, 35B got Y ± Z" — which is the version a
reviewer can trust.

### Improvement 2 — a second prompted judge, "B10b", with severity caps

**Leave B10 exactly as it is.** Its prompt and weights were written down in
advance, together with a prediction that it would lose. Changing it now, after
seeing PPTAgent's better design, would look like we tuned the baseline until it
gave us the answer we wanted. That is the single easiest thing for a reviewer to
attack.

Instead we add a **second, stronger** prompted judge alongside it, using
PPTAgent's cap idea translated to our problem: judge the *redundancy of the last
step* first, and let that cap the "keep going" score.

Today (B10) asks for four flat yes/no bits and adds them up:

```
RUBRIC: [1,0,0,1]   →  continue_score = 0.15
```

The B10b version would ask the judge to name the problem first:

```
The last two searches used almost the same words and returned the same
passages, so the newest step added nothing at all. The draft has been
identical for two steps.

REDUNDANCY: fully_redundant
CONTINUE: 1
```

with the designed cap table:

| redundancy of the last step | example | cap on continue_score |
| --- | --- | --- |
| `none` | new step brought genuinely new facts | no cap |
| `mild` | mostly overlapping, some new detail | at most 0.6 |
| `repeated_query` | nearly the same search as before | at most 0.3 |
| `fully_redundant` | same search, same passages, draft unchanged | at most 0.1 |

Two reasons this is worth doing:

- **It answers our own weakest argument.** We currently claim RM-P must lose
  partly because yes/no bits are too coarse: two neighbouring steps get
  identical bits, so the training signal `V̂(next) − V̂(this)` is exactly zero.
  A reviewer will reply, correctly, *"that is a flaw in your rubric, not in
  prompted judges."* B10b removes that objection. We can then **show** how often
  the score fails to move (bits vs. graded), instead of asserting it.
- **A strong baseline that still loses is worth far more than a weak one.**
  Right now B10 is the obvious prompted design. B10b would be a design a
  colleague has already validated on hundreds of real cases.

The same rule carries over: **the caps are designed and written into the paper
appendix — never tuned on data.** Tuning them would make B10b a trained reward
model, which would destroy the comparison we are trying to make.

### Improvement 3 — let the judge say "I cannot tell"

Right now our code treats only *garbled output* as a non-answer (and then never
stops — the safe direction). Borrowing PPTAgent's `false` option would let the
judge deliberately abstain, and we would log that separately. The difference is
informative: a judge that says "I don't know" is behaving well; a judge that
confidently guesses is the failure mode we are testing for.

## B.4 What we deliberately will NOT copy

- **Their six evaluation dimensions.** PPTAgent judges a whole slide deck, which
  really does have many separate quality axes. We ask one small question — "stop
  now, or keep working?" More dimensions here would add cost and randomness, not
  information.
- **Their "describe first, then score" two-step.** That exists because slides are
  *pictures*, so something must convert the picture into words first. Our judge
  already receives a clean structured text block, so a description step would
  only add cost and lose detail.

## B.5 Summary

| # | Change | Cost | When |
| --- | --- | --- | --- |
| 1 | Run the RM-P exam 3× and report the spread | ~$2 | now — applies to the current run |
| 2 | Add B10b: severity-capped prompted judge (B10 untouched) | ~$1 per run + a day of work | after the first B10 number lands |
| 3 | Explicit "cannot tell" answer, logged separately | free | with B10b |

**One rule holds across all three: B10's existing prompt and weights are frozen.**
Everything above is added next to it, never on top of it. That is what keeps the
trained-vs-prompted comparison honest.

---

# Appendix C — How the agent actually runs (the rollout code and the harness)

A "rollout" is one attempt at one question: the agent thinks, searches, thinks
again, and eventually answers. Everything we measure — costs, labels, rewards —
comes out of rollouts, so it is worth knowing exactly how they are produced.

The important thing to understand up front: **we run rollouts in two completely
different places**, for two different purposes, and they use different code.

```
          OUR OWN LOOP                         VERL'S LOOP
   executor/react_agent.py                executor/verl_hooks.py
   ------------------------               --------------------------
   used for: data collection,             used for: RL training (GRPO)
             evaluation, monitors
   speed:    simple, one episode          speed: batched, many GPUs
             at a time over HTTP
   this is what produced the              this has NOT run yet
   9,600 trajectories
```

## C.1 The shared agent loop (`executor/react_agent.py`, 347 lines)

This is a **ReAct** agent — "Reason, then Act". The important design decision is
in the file's first line: this same scaffold is used by **CASSI and every single
baseline**. The plan puts it bluntly (§2.6): it is *"a constant, not an
advantage."* If our method won because it had a better agent loop, the paper
would prove nothing.

### What the agent is told (system prompt)

```
You are a careful research agent solving a task step by step.
Available tools:
search[query]: retrieve the top passages for the query from the
               local Wikipedia index (Search-R1 retriever).

At each step, respond in EXACTLY this format:
THOUGHT: <one short paragraph of reasoning>
ACTION: <tool>[<argument>]   (or  answer[<your final answer>]  to finish)
DRAFT: <your best answer so far>
```

### What one step looks like

```
THOUGHT: The 1973 reorganization plan created the DEA. I should confirm
         who its first administrator was.
ACTION: search[DEA first administrator 1973]
DRAFT: Drug Enforcement Administration
```

The agent replies, we parse three things out of it, and the loop continues:

| Parsed | How | Why it exists |
| --- | --- | --- |
| the action | regex `ACTION: tool[arg]`, **last match wins** | models often restate the format; the last one is the real intent |
| the draft | regex on the `DRAFT:` line | this is the running answer — the whole basis of "is it still changing?" |
| the cost | token count × price + tool fee | see C.3 |

**The `DRAFT:` line is the quiet centerpiece.** Every step must state its best
answer so far. That is what makes "the answer stopped changing for 2 steps"
measurable, and it is also *why* stopping early is even possible — there is
always an answer ready to hand in. Its tokens are charged like any others, for
every method, so it is not a free advantage.

### The two rollout modes (this is the subtle part)

```python
ROLLOUT_MODES = ("rl", "forced_continuation")
```

- **`rl` mode** — normal. The agent says `answer[...]`, the episode ends. This is
  what happens during real training and real evaluation.
- **`forced_continuation` mode** — the agent says `answer[...]`, we write down
  *"it wanted to stop at step 5"*, and then we **make it keep working anyway**
  to step 10. The observation it gets back is literally:

  > `(answer recorded; the episode continues — keep improving your answer if
  > further work is worth its cost)`

Why force it? Because to build the answer key, we must know **what would have
happened if it had kept going**. If we let the agent stop at step 5, steps 6–10
never exist and we can never check whether stopping at 5 was right. The plan
calls this censoring. Forced continuation removes it, at the price of paying for
steps nobody wanted — that overhead is tracked honestly by
`forced_continuation_overhead()` and reported.

Bonus: the "it wanted to stop at step 5" flag is a **free measurement**. That is
where the *68% of episodes try to answer early* number in Stage 2 came from —
no extra rollouts needed.

### Three rules the loop enforces

1. **The agent never sees the gold answer.** Quality `q_t` is scored afterwards,
   in `collect.py`, and never enters the state `x_t`.
2. **The agent never enforces its own budget.** It is told the budget but is
   never cut off. Reason (§2.1): *"exploration must be free; economics reach the
   policy only through rewards."* If we hard-stopped it, we would be measuring
   our cutoff rule, not learned judgment.
3. **The monitor is inference-only.** `monitor=None` during all training
   rollouts. A stopper is allowed to interrupt at evaluation time, never while
   the agent is learning.

## C.2 The environment (`executor/envs/searchr1_qa.py`)

One tool: `search[query]`. It POSTs to the **Search-R1 local retrieval server**:

```
POST http://127.0.0.1:8000/retrieve
{"queries": ["DEA first administrator 1973"], "topk": 3, "return_scores": true}
```

and gets back passages, which are turned into the observation:

```
[wiki_12345] The Drug Enforcement Administration was established in 1973... |
[wiki_67890] John R. Bartels was the first Administrator...
```

Two practical notes:

- The document IDs are kept. That is how we compute *"the last three searches
  returned 87% of the same documents"* — our redundancy signal.
- `return_scores: true` is **required**, not optional. The pinned upstream server
  unconditionally unpacks `(results, scores)` and crashes without it (upstream
  bug, verified 2026-07-21). This is the kind of thing that costs a day.

The retrieval index is the 21M-passage Wikipedia corpus — the one we moved from
35 s/query to 46 ms/query.

## C.3 The money harness (`budget/cost.py`, 148 lines)

Every dollar in the project is computed in **one file**, on purpose: the plan
calls it *"one Lagrangian, not three."* If the agent, the labels, and the
rewards each had their own idea of cost, they would be optimizing different
things and nothing would be provable.

**Prices** (per 1M tokens, local serving): input $0.60, output $2.20.
**Tool fees**: local retrieval $0.0001/query; web search $0.003/query +
$0.001/result.

**Budget tiers** — how much of the wallet is *left*:

| Remaining | Tier | Multiplier `m(tier)` |
| --- | --- | --- |
| > 60% | HIGH | 0.5 |
| 30–60% | MEDIUM | 1.0 |
| 10–30% | LOW | 2.0 |
| < 10% | CRITICAL | 5.0 |

This is the "money gets more precious as you run low" rule. A dollar spent at
CRITICAL hurts **ten times** more than the same dollar at HIGH. It is a designed
table, not learned.

**The one formula everything shares:**

```
U_t = q_t  −  Σ(i ≤ t)  λ · m(tier_i) · c̃_i
      ↑            ↑        ↑              ↑
   quality      price     tier         normalized
   of draft   sensitivity multiplier   cost of step i
```

`c̃` is the raw dollar cost divided by the median pilot spend — that division is
what makes λ a plain dimensionless dial instead of something in units of
dollars.

**Wallets** are drawn per task from the pilot measurements (small = P25,
medium = P75, large = 2×P90 of unconstrained spend). Critically,
`draw_wallet` is called **once per group of 8 rollouts, not once per rollout**:

> "group advantages must compare behavior under the same wallet, never confound
> behavior with wallet luck" (§2.2)

All 8 attempts at a question get the *same* wallet, so when we compare them, any
difference is behavior — not one attempt having gotten lucky with a bigger purse.

## C.4 Collection (`executor/collect.py`) — what actually produced our data

```
for each task:
    draw ONE wallet (shared)
    for g in 1..8:
        run the agent in forced_continuation mode to step 10
        afterwards: score each draft against gold  →  q_t
    write JSONL
```

1,200 tasks × 8 rollouts = **9,600 trajectories**, 10 steps each = **96,000
steps**, $143 of simulated spend. That is the file Stage 3's labels were
computed from.

## C.5 The training-time loop (`executor/verl_hooks.py`) — not yet run

For RL we cannot use the simple loop; we need thousands of rollouts batched
across GPUs. That is **verl**'s job (pinned at commit `7aed6b2`). Our code plugs
into it at three points, and every single reference carries a `# pin:` comment
with the exact upstream file and line number so the next person can re-verify
after an upgrade.

The awkward part, and why the file has a long justification comment: at the
pinned commit, verl computes rewards *per trajectory* and collapses each one to
**a single number on the final token**. But CASSI's whole idea is a *per-step*
reward. There was no clean hook for that, so:

- we override the **agent-loop manager** (the one batch-level hook the commit
  exposes) and overwrite the reward tensor after rollout;
- we smuggle per-step advantages through that tensor by **difference encoding**:
  write `A_t − A_{t+1}` on each step's last token, then a registered custom
  estimator decodes it with a reverse cumulative sum, so each step's advantage
  lands on exactly that step's tokens.

**Known consequence, already documented:** verl's own logged "reward" metric now
shows a meaningless number (it sums to `A_1`). The real economic rewards are
written to `<out>/divergence.csv` instead — which is also the file that will feed
the reward-hacking analysis. Worth knowing before someone panics at a dashboard.

## C.6 Why the reward needs step-level assignment (the non-obvious bit)

The shaping reward is `r_t = Φ(next state) − Φ(this state)`, where Φ is the
stopper's value head. Add those up across a whole episode and everything cancels
except the first term — the total is always `−Φ(x_0)`, the same constant for
every rollout in the group.

So if you compared rollouts only at the whole-episode level, **the shaping would
be mathematically invisible.** Its entire effect has to arrive through per-step
credit. That is not a preference; `tests/test_core.py` asserts the telescoping.
It is also why the difference-encoding gymnastics in C.5 are unavoidable rather
than merely clever.

## C.7 The map, in one place

| File | Lines | Job |
| --- | ---: | --- |
| `executor/react_agent.py` | 347 | the shared ReAct loop; both rollout modes |
| `executor/collect.py` | 238 | runs collection rounds, scores drafts, writes JSONL |
| `executor/envs/searchr1_qa.py` | 124 | the `search[]` tool over the Wikipedia index |
| `executor/vllm_client.py` | 78 | talks to the local vLLM server (thinking off) |
| `budget/cost.py` | 148 | **all** money: prices, tiers, wallets, the utility formula |
| `executor/shaping.py` | 108 | `r_t` and the step-level advantages (pure numpy) |
| `executor/monitor.py` | 190 | inference-time stopping + the self-stop metric |
| `executor/verl_hooks.py` | 518 | the GRPO plumbing (torch/verl; GPU only) |
| `executor/train_grpo.py` | 679 | the RL entry point (`--dry-run` is the regression check) |

Two habits in this code worth keeping: the money lives in exactly one file, and
every non-obvious line has a comment saying *which section of the plan* it
implements. That is why the pipeline can be re-derived from the paper plan
rather than from memory.
