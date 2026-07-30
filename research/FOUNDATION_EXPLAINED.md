# The Foundation Experiment, Explained Simply

**For: Brian, as the author who will publish this.**
**Date: 2026-07-29. Verdict: GO — but read §6 before you write the abstract.**

This document explains the whole experiment in plain language: what we asked,
how we did it, what came out, what we can honestly claim, and what to do next.
No AI background assumed.

---

## 1. The question, in one paragraph

An AI agent that answers questions by searching will keep searching. It does not
know when "good enough" has arrived. We gave it a **budget** — a number of search
steps — and asked: **can we teach it, through training, to manage that budget
itself?** Three ways to make an agent respect a budget:

1. **Tell it** the budget in its instructions and hope it complies.
2. **Force it** — a wrapper program cuts it off when the budget runs out.
3. **Train it** so the judgement lives inside the model itself.

The paper's claim is that option 3 beats options 1 and 2. This experiment was a
small, cheap test of that claim before committing to the full paper.

---

## 2. The setup, in plain terms

**The task.** HotpotQA: questions that need two facts joined together, e.g.
*"Which tomb is in a valley known as the Valley of the Gates of the Kings?"* The
agent cannot know the answer; it must search Wikipedia.

**A "step"** = one cycle of: think → do one search → read the result. The agent
also writes its current best guess after every step, so we always know what it
would answer if stopped right now.

**The "budget" (B)** = how many steps it is allowed. We used three: **2 (tight),
4 (medium), 8 (loose)**. Medium is the one the official test uses.

**"Utility" — the score that matters.** We need one number combining "good
answer" and "cheap":

```
Utility = answer_quality − 0.3 × (steps_used / budget)
```

*Answer quality* is F1, a 0–1 score for how well the answer matches the correct
one. The second part is the price of the steps. λ = 0.3 sets how expensive steps
are. Higher utility = better answer, fewer steps, or both.

**The four arms (the things we compare):**

| arm | what it is | plain description |
|---|---|---|
| **A0** | no budget info | The agent is never told a budget exists. The "floor". |
| **A1** | told the budget | Budget shown every step. Nothing enforced. |
| **A2** | forced to stop | Same agent, but a wrapper cuts it off at B and submits its current guess. |
| **A3** | **trained** | The model itself was retrained. Tested with the wrapper **off**. |

All four use the *same* prompt and scaffolding, so any difference comes from the
arm, not from wording.

---

## 3. How the RL training actually works

"RL" (reinforcement learning) means: let the model try, score the attempts, and
nudge it toward what scored well. Concretely, in three rounds:

**Step 1 — Let it play (collection).** Take 300 practice questions. For each, let
the model attempt it **8 different times** (it is random enough to behave
differently each time). That is 2,400 attempts per round. We record every step.

**Step 2 — Score every attempt.** Two kinds of score:

- **The final score (computed exactly, no AI involved).** We know the right
  answer for practice questions, so we compute
  `F1 − 0.3 × (steps/B)` directly. This is arithmetic, not opinion.
- **The per-step score (from a judge AI).** A separate, larger frozen model
  (Qwen3.6-27B) reads each step and answers yes/no questions: *did this search
  return new information? was the query different from earlier ones? was more
  work still needed?* And for the final answer: *is it supported by what was
  found? was stopping here reasonable?* These become small nudges (±0.1).

  The judge **never sees the correct answer** — otherwise it would just grade
  correctness, not process.

**Step 3 — Compare siblings.** This is the core trick (called GRPO). We do not
need to know what a "good" score is in absolute terms. We compare the 8 attempts
*at the same question* against each other. Attempts that beat their siblings get
reinforced; attempts that did worse get discouraged. Because all 8 share the same
question and the same budget, the comparison is fair.

**Step 4 — Nudge the model.** Adjust the model's weights slightly toward the
better-scoring attempts. "Slightly" is enforced by two safety devices: a small
learning rate, and a **KL anchor** that penalises drifting too far from the
starting model in one round.

**Step 5 — Repeat.** Serve the updated model, collect fresh attempts with it,
score, nudge. Three rounds total.

**A health check between rounds.** After each round we sample 40 fresh attempts
and check the model has not broken (garbled output, never answering). A previous
attempt at this experiment died exactly here: the model was damaged in round 1,
nobody checked, and round 2 trained on garbage. Now it is a hard gate.

---

## 4. The results

All numbers on **dev-200**: 200 questions the model never trained on, never
tuned on, and which were frozen before any of this started.

### Utility (higher is better)

| arm | B=2 | B=4 | B=8 |
|---|---|---|---|
| A0 no info | −.174 | .121 | .268 |
| A1 told | .035 | .205 | .352 |
| A2 forced | −.079 | .180 | .306 |
| **A3 trained** | **.116** | **.289** | **.386** |

**A3 wins at every budget.** The official test is at B=4, and it passed all three
pre-registered conditions:

- utility **.289** > A1's .205 and A2's .180 ✓
- stops on its own in **77.5%** of episodes (needed ≥70%) ✓
- answer quality **.560**, far above the .361 floor ✓

### Answer quality (F1)

| arm | B=2 | B=4 | B=8 |
|---|---|---|---|
| A1 told | .478 | .471 | .501 |
| A2 forced | **.221** | .411 | .455 |
| **A3 trained** | **.559** | **.560** | **.548** |

Two things stand out. A3 answers *better than every baseline*. And at the tight
budget, forcing (A2) collapses quality to .221 while A3 holds .559 — you cannot
make an answer ready early by cutting the agent off; only changed behaviour does
that.

### Is the difference real, or luck?

We compare **the same question** across arms and bootstrap the difference (resample
10,000 times to see how much it wobbles). At B=4:

- A3 − A1: utility **+.084**, 95% range **+.025 to +.147** — does not cross zero
- A3 − A2: utility **+.110**, 95% range **+.048 to +.171** — does not cross zero

"Does not cross zero" means the improvement is unlikely to be chance.

---

## 5. Did the number of steps go down after training?

**No. This is the most important thing in this document.**

You asked this directly, and the honest answer changes the story.

| arm | B=2 | B=4 | B=8 |
|---|---|---|---|
| A1 told | 2.96 | 3.54 | 3.98 |
| **A3 trained** | **2.96** | **3.61** | **4.32** |

A3 uses **the same or slightly more** steps than the untrained-but-informed
agent. Where its utility gain comes from, split into its two parts:

| budget | total gain | from better answers | from fewer steps |
|---|---|---|---|
| B=2 | +.081 | **+.081** | −.000 |
| B=4 | +.084 | **+.089** | −.005 |
| B=8 | +.034 | **+.047** | −.013 |

**Every bit of the improvement comes from answer quality. The step cost actually
works slightly against A3.**

And the stopping behaviour is nearly identical. Here is where each arm stopped at
B=4 (number of episodes stopping at each step):

```
A1 told:     step2: 47   step3: 79   step4: 31   step5: 31   ...
A3 trained:  step2: 48   step3: 77   step4: 30   step5: 32   ...
```

Practically the same distribution. And the self-stop rate went slightly **down**
after training (−.01 at B=4, −.06 at B=8).

**Plain conclusion: training made the agent better at answering, not better at
stopping.**

---

## 6. What this means for your paper — read this before writing

The gate passed honestly, on rules fixed in advance. But **the reason it passed is
not the reason the plan predicted.**

The reward has two parts: answer quality and step cost. Training improved the
first and left the second essentially untouched. Since F1 is *in* the reward, a
reviewer will immediately say:

> "You fine-tuned a model on HotpotQA with an F1-based reward, and F1 went up.
> That is ordinary task training. Where is the economic reasoning?"

On the current evidence, **you cannot answer that objection.** The step counts,
the stop-step distributions, and the self-stop rates all say stopping behaviour
did not change.

**I also have to correct something I wrote earlier.** In the first draft of the
run report I claimed the "harness-off beats harness-on" gap proved
internalization. Then I ran the control — A1 vs A2 is *the same untrained policy*
with the wrapper off vs on — and:

| budget | untrained off−on gap | trained off−on gap |
|---|---|---|
| B=2 | +.114 | +.185 |
| B=4 | +.026 | +.023 |
| B=8 | +.046 | **+.005** |

The untrained agent shows the same pattern. At B=4 and B=8 the trained agent's
gap is *no bigger* — at B=8 it is much smaller. So that gap mostly shows "cutting
an agent off hurts", which is true of any agent. **It is not evidence of
internalization.** The run report has been corrected.

### Why stopping probably did not change: the economics were too weak

At B=4 with λ=0.3, one extra step costs `0.3/4 = 0.075` utility. From the pilot,
a fourth step buys about +0.06 F1. So taking another step is roughly
**break-even**. The agent had almost no incentive to stop earlier — we priced
steps so cheaply that the optimum is flat.

That is a design finding, and a useful one: **λ = 0.3 is too small to change
behaviour.** It was chosen to put the optimum at the pilot's knee, but "where the
optimum sits" is not the same as "how strong the pull toward it is."

---

## 7. How do we know this is not overfitting?

Overfitting = looking good on the data you tuned on, and failing on fresh data.
Five defences, all built in before results were seen:

1. **The test set was frozen and never touched during development.** dev-200's
   200 questions were fixed before any training. The model never trained on them.
2. **A separate tuning set.** Every decision that needed data — which checkpoint
   to keep, whether the model was healthy — used **val-50**, a different 50
   questions. The rule was written down in advance: *tuning reads val-50, never
   dev-200.*
3. **A hard limit on test-set looks.** We allowed ourselves at most 3 evaluations
   on dev-200 ever, each logged with a date and reason. We used 2: one for the
   baselines, one for the final answer. If you re-run the test until you like the
   number, you have overfitted the test set even without training on it.
4. **The rules were fixed in advance.** The three pass conditions and their exact
   thresholds were written into a config file and turned into code *before*
   training. The verdict was computed by a script, not chosen by me.
5. **The training and test questions do not overlap.** Checked by ID and by
   question text; zero overlap between all three sets.

One honesty note: the training set had to be rebuilt after a machine wipe
destroyed the original, so A3 trained on a slightly different 300 questions than
originally planned. **dev-200 was recovered exactly** (by question ID), so the
test is unchanged.

---

## 8. How do we know the AI is not cheating?

"Cheating" here has a specific meaning: the model learns to please the **judge**
rather than actually do better. This is called **reward hacking**, and it is the
most common way results like this turn out fake.

We had a specific reason to worry. Before training, we measured the judge against
150 human-style labels and found it is **too generous about stopping** — it
approves stopping when more work was needed about twice as often as the reverse.

That gave us a **prediction we could check**: if the model learned to exploit
that generosity, its judge scores would climb while its *real* answer quality
stayed flat or fell.

Here is what happened over the three rounds:

| round | judge's score | real answer quality (F1) |
|---|---|---|
| 1 | .842 | .591 |
| 2 | .841 | .611 |
| 3 | .843 | .622 |

**Judge score: flat. Real quality: rising.** The exact opposite of the cheating
signature. The model got better at the task, not at flattering the judge.

Two further checks agree:

- The final test measured F1 against **known correct answers**, not against the
  judge. A3 scored highest of all arms — an AI judge cannot fake that.
- The "no quality collapse" condition existed to catch a model that saves steps
  by giving up. A3's quality went *up*, so it is not gaming cost by answering
  badly.

**Conclusion: the improvement is real, not judge-flattery.** The irony is that
the thing we proved is real (better answers) is not the thing we set out to prove
(better stopping).

---

## 9. Baselines vs RL — the honest scorecard

| question | answer |
|---|---|
| Does the trained agent beat "just tell it the budget"? | **Yes**, utility +.084 at B=4, statistically solid, and at every budget |
| Does it beat "force it to stop"? | **Yes**, +.110 at B=4 |
| Is it because it uses fewer steps? | **No.** Same or slightly more steps |
| Is it because it answers better? | **Yes.** All of the gain, +.089 F1 at B=4 |
| Does it stop on its own more often? | **No.** Slightly less often |
| Does it adapt to the budget more? | **Slightly.** Step range across budgets 1.36 vs A1's 1.02 |
| Which baseline was harder to beat? | **A1 (telling it)** beat A2 (forcing it) at every budget. Forcing is the weak strategy |

The A1-beats-A2 result is itself worth reporting: **telling a capable model about
a constraint works better than mechanically enforcing it** — enforcement cannot
improve what the agent has prepared by the cutoff, and at B=2 it is catastrophic
(F1 .221 vs .478).

---

## 10. What is solid, what is shaky

**Solid:**
- The pipeline works end to end and is reproducible from the repo.
- A3 beats both baselines on the pre-registered gate, with the rules fixed in
  advance and the verdict computed by script.
- The improvement is real, not reward hacking (§8), and not test-set overfitting (§7).
- A1 > A2: prompting beats enforcement.

**Shaky, or not shown:**
- **The stopping claim.** Not supported. Stopping behaviour barely moved.
- **Internalization.** The harness-off/on gap does not survive its control.
- Only **one training seed**. Re-running with different randomness might give a
  different size of effect.
- Only **one domain** (HotpotQA) and **one λ**.
- The judge's calibration passed the averaged rule (mean .847) but **not** the
  strict per-bit rule the plan wrote down (`nothing_left` .775 vs .80 required).
- The calibration "human" labels were produced by **AI labellers** with no context,
  not by you. That was the best available substitute for independence, but the
  plan's requirement for human labels is formally unmet.
- The final checkpoint was **chosen** using val-50, not fixed in advance.

---

## 11. What I would do next, in priority order

**1. The ablation that decides the paper's story (highest value, ~1 day).**
Retrain with **λ = 0** — no step cost at all, everything else identical. Then:
- If λ=0 gives the same F1 *and* the same step counts → the economic part of the
  reward did nothing, and this is task-skill RL. You would reframe the paper.
- If λ=0 uses noticeably more steps → the step cost *was* doing work, and you can
  claim it, with a clean control.

Right now you cannot tell these apart, and it is the first thing a reviewer will
ask. This single run resolves it.

**2. Raise the price of steps (~1 day).** λ=0.3 makes a step roughly break-even.
Try λ = 1.0 or 1.5, where stopping early clearly pays. If stopping behaviour then
moves, you have the paper's actual claim *and* a dose-response curve — the most
persuasive evidence possible ("we turned the price up and the behaviour changed").

**3. More seeds (~1 day each).** Two more training runs with different randomness.
Reviewers discount single-seed RL results heavily.

**4. Then the deferred v2.1 pieces:** the trained reward model (its training data
is already saved on disk), a second domain, dollar-based costs.

**Do not run the dev-200 test again** until you have a genuinely new method to
test. You have used 2 of your 3 permitted looks. Testing repeatedly until the
number improves is overfitting even without training on it.

---

## 12. Where everything is

| what | where |
|---|---|
| Full technical report | `research/foundation/experiments/reports/foundation_report.md` |
| Figures | `research/foundation/experiments/reports/figs/` |
| Complete decision history | `research/foundation_tasks/PROGRESS.md` |
| The plan being implemented | `research/paper_plan_v2_1_foundation.md` |
| All results as data | `research/foundation/experiments/results/foundation_eval.csv` |
| Trained models, training data | `/mnt/src/liangsheng/cassi_foundation/` |
| Code map | `research/README.md` |
| This run, tagged in git | `foundation-run-1` |

Every number in every report regenerates from the CSV by script. Nothing is
typed in by hand.

---

## 13. The one-paragraph summary

We trained an agent with a reward that mixes answer quality and step cost, and it
beat both "tell it the budget" and "force it to stop" on the frozen test set, by
a margin unlikely to be chance, with reward hacking and test-set overfitting both
checked and ruled out. **But it won by answering better, not by stopping sooner:
step counts did not fall, and stopping behaviour is nearly identical to the
untrained agent.** The most likely reason is that we priced steps too cheaply for
the agent to care. The pipeline is proven and the result is real — the claim just
needs to be either re-framed around answer quality, or re-tested with a stronger
step price before it can be about economic stopping.

---

## 14. UPDATE — the λ ablation answered §11's first question (2026-07-31)

§11 said the highest-value next step was a λ=0 ablation, because we could not
tell "the economics did something" from "this is ordinary task training". **We
ran it, plus a λ=1.0 arm. The answer is: the step-cost term does not change
stopping.**

Pre-registered rule (fixed before the data): the cost term counts as effective
only if mean steps at B=4 drops ≥0.5 between λ=0 and λ=1.0 with non-overlapping
confidence intervals.

| training λ | what a step costs at B=4 | mean steps |
|---|---|---|
| 0.0 | nothing | 3.500 |
| 0.3 | .075 | 3.500 |
| 1.0 | .25 (≈4× what it buys) | 3.460 |

**Δ = 0.040 steps. Verdict: NOT EFFECTIVE**, on both conditions.

**Why — and this is the useful part.** It is not that the penalty was too weak
(it is 82% of the training signal at λ=1.0) and not that the algorithm cancelled
it (checked). It is that **when the agent stops is mostly decided by the
question, not the agent**: easy questions finish in 2 steps, hard ones take 4,
and stopping early on a hard question just means answering wrong. Measured over
109 groups of 8 attempts at the same question — between-question SD of stop step
**1.220** vs within-question **0.666**. There is no free "stop sooner" behaviour
for a price to buy.

Two real findings alongside it:

- **Where the budget genuinely binds, the price does work.** At B=2 — where the
  untrained-cost policy overspends, taking 3.28 steps for a budget of 2 — λ=1.0
  cuts 0.70 steps with quality intact (CI excludes zero). Exploratory, not
  pre-registered, so weaker evidence; but it locates the regime where such
  rewards are worth using.
- **Too high a price damages the model.** λ=1.0 was the only arm to fail its
  health gate (11% malformed output), and had the lowest F1 of the three.

**What changes for the paper:** the "we taught economic stopping" claim is not
supported and should not be made. What replaces it is stronger than a null:
*pricing raw step count fails on step-budget benchmarks, and here is the variance
decomposition showing why.* Full detail, including the recommended fixes (price
steps relative to the minimum needed for **that** question, or reward the stop
decision against an oracle continuation), is in
`foundation/experiments/reports/ablation_report.md`.
