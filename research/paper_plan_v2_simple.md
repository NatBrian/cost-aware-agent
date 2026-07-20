# Paper Plan v2 — The Simple Version

**What this file is.** `paper_plan_v2.md` is the real plan, but it is written in
research shorthand. This file says the same things in plain words, with the
reasoning spelled out: what we are doing, why, what experiments we run, why each
one exists, and how to run it.

**Rule:** if this file and `paper_plan_v2.md` ever disagree, `paper_plan_v2.md`
wins. This is a translation, not a second opinion.

---

## 1. The problem, in one story

Imagine you hire a research assistant and pay them by the hour. You ask a
question. A good assistant does some searching, forms an answer, and then — this
is the important part — *notices when more searching is no longer worth your
money* and hands you the answer.

Today's AI agents cannot do that last part. They search, they reason, they call
tools, and nothing in their training ever told them "this next step costs money
and probably will not improve your answer." So they keep going.

This is not a made-up problem. Published measurements show:

- Longer thinking often makes accuracy *worse* past some point, not better.
- In agents specifically, how much an agent overthinks predicts whether it will
  fail the task.
- Picking the low-overthinking runs cut cost by 43% *and raised* success by 24%.

So there is real waste on the table.

**Why not just ask the model "are you done?"** People tried. It fails badly — on
a benchmark built to test exactly this, LLMs spot a redundant agent step only
~25% of the time. And the reason it fails is instructive: "am I done?" is a
question about *confidence*, but the real question is about *economics* — "is
the next step worth what it costs?" A model can be quite sure it is not done and
still be wrong to continue, because finishing is not worth the price.

**Our claim in one sentence:** knowing when to stop is an economic judgment, and
it can be *trained into* an agent rather than enforced on it from outside.

---

## 2. The method, in plain words

Think of two models: a **coach** and a **worker**.

- The **worker** (a 9B model) is the agent that actually does the task — searches,
  reasons, answers.
- The **coach** (a small 2B model) watches the worker at every step and estimates:
  *how much more value is left in continuing, given what continuing will cost?*

There are three ingredients plus a loop.

### Ingredient A — Where the coach's training answers come from

We need to teach the coach what the *right* stopping point was. We get this by
looking backward at trajectories the worker already produced.

For each step we compute a score:

```
value at step t  =  how good the answer is right now  −  λ × how much we've spent
```

`λ` (lambda) is just a knob for how much we care about money. λ high = cost
matters a lot. λ = 0 = money is free.

Now, the naive approach — "look at the whole trajectory, find whichever step had
the best score, call that the right answer" — is **wrong**, and it is worth
understanding why, because it is the single most common mistake in this area.

That approach lets the label peek at the future. It effectively asks "knowing how
this run turned out, when should we have stopped?" But a real agent, deciding in
the moment, does not know how the run will turn out. Optimal-stopping theory says
the peeking version *systematically overvalues continuing* — so a model trained
on it learns to stop too late. That would quietly destroy the whole paper.

The fix is a classical technique (the **Snell envelope**, computed by backward
recursion — the same math finance uses for pricing American options, where you
also must decide "exercise now or wait?" without knowing the future):

```
at the last step:  V = the score right now
at earlier steps:  V = max( score right now ,  average score of continuing )
```

That "average score of continuing" is estimated across *many* trajectories, not
one. So the label is "continuing is worth this much *on average* from a state
like this" — which is a judgment a real agent could actually make. The gap
between the two options is the **margin Δ**: positive means keep going, negative
means stop.

**Why this costs nothing extra:** we compute these labels from trajectories we
already collected. No additional model runs.

### Ingredient B — Training the coach

Train the small model to predict three things from what it sees mid-task: stop or
continue, the margin Δ, and the value V.

Two design notes worth understanding:

- **The coach only sees things a real deployment would have.** Budget spent, step
  count, how much the draft answer has changed lately, retrieval overlap, and so
  on. It never sees the ground-truth answer. This is a hard rule: if the coach
  needs privileged information to work, it is useless at deployment.
- **One coach handles every λ.** λ is part of its input. So at deployment you turn
  the cost dial without retraining anything. This is what makes "a principled cost
  knob" a real feature rather than a slogan.

### Ingredient C — Using the coach to train the worker

This is the actual novel step, and it needs care.

The obvious approach — "give the worker a bonus at every promising step" — is a
known trap. It pays the worker to *accumulate* promising steps, so it dawdles.
That is precisely backwards from what we want.

The correct approach is **potential-based shaping**, a 1999 result: if the reward
you add takes the form

```
reward at step t  =  (coach's value at the next state) − (coach's value at this state)
```

then you have provably not changed which behavior is optimal. You have only made
the feedback *denser* — instead of learning only at the end ("that run cost too
much"), the worker gets a signal at every step.

**One consequence you must not trip over.** Because of how these terms telescope,
they cancel out completely if you compute advantages at the whole-trajectory
level. The shaping would do literally nothing. Therefore **step-level credit
assignment is mandatory, not a nice-to-have.** If someone "simplifies" the code to
trajectory-level advantages, CASSI silently becomes a no-op and every result goes
flat. This is the #1 way this project can fail from a code change.

Put simply: **the coach converts a reward you only learn at the end into feedback
at every step, without changing what "good" means.**

### The loop (D)

Once the worker improves, the old labels are stale — they describe a worker that
no longer exists. So we redo it: collect with the new worker → new labels → new
coach → train again. At least twice.

**Important honesty control:** iteration 2 gains could just be "we trained
longer." So we run iteration 2 *twice* — once with a refreshed coach, once with
the old frozen coach at the same compute. The difference between those two is the
loop's real contribution. Without this control, reviewers will (correctly) reject
the loop claim.

---

## 3. What is actually new here

Everything we use exists somewhere in the literature. The *combination* does not.
Specifically, as of 2026:

| Already exists | But only... |
|---|---|
| Hindsight stop labels + small trained stopper | used at inference time, never to train the agent |
| Learned monitor → process reward → agent RL | quality-only, no notion of cost |
| Cost-aware agent RL | at the whole-trajectory level, not per step |

Nobody has turned an explicit *quality − λ·cost* optimum into a trained stopping
value that supervises a tool-using agent step by step. And nobody has asked
whether stopping economics can be *internalized into the weights* rather than
enforced by a runtime monitor.

**The one claim the paper lives or dies on:** a worker trained with coach-derived
rewards beats (i) the same coach used only as a runtime monitor, (ii) direct
shaping with no coach at all, and (iii) training-free monitors — measured as cost
at equal accuracy, on at least two domains.

---

## 4. The experiments — what, why, how

### First: the kill-switches (run these before anything else)

The whole project rests on one bet. So we test that bet cheaply, in weeks 2–3,
before spending months.

**K1 — the bridge test.** *Does training on the coach's signal actually beat just
using the coach as a runtime monitor?*

- **How:** 1,000 HotpotQA tasks, one seed, λ = 1.0. Three arms: shaped training,
  controller-only, and direct shaping with no coach. Each arm gets a small 3-point
  cost-dial frontier {0.5, 1.0, 2.0} so "equal accuracy" is well defined.
- **Pass condition:** shaped training cuts cost by ≥3% relative to controller-only
  at equal accuracy, AND is no worse than direct shaping (ties pass).
- **If it fails:** do not push on. Pivot to the honest smaller paper — "a trained
  cost-aware stopping monitor for frozen agents" — reusing the coach we already
  trained and dropping worker RL entirely.

**K2 — the separation test.** *Do we need two models, or would one model doing both
jobs work just as well?* Same task set, one 9B doing both vs 9B + 2B. Either
outcome is publishable; it changes the framing, not the viability.

### Then the main experiments

| ID | What it does | Why it exists |
|---|---|---|
| **E1** | CASSI against all 9 baselines on both training domains, 3 seeds | The headline result. Everything else supports this. |
| **E2** | Turn the monitor **off** at test time. Also transfer to unseen datasets and to a *different* worker model. | This is the internalization proof. If the worker stays cheap with no monitor watching, the economics really moved into the weights — something no inference-time-control paper can demonstrate. |
| **E3** | Sweep the λ dial on one fixed worker; plus run identical tasks under small/medium/large budgets | Shows the cost knob works without retraining, and that the agent responds to *how much money is left*, not just how much it spent. |
| **E4** | Compare our Snell labels against the naive peeking labels and other alternatives | Directly tests whether the careful label construction in §2A was worth it. A reviewer will ask. |
| **E5** | Two loop iterations, with the frozen-coach control described above | Proves loop gains are real, not just "more training." |
| **E6** | Do harder questions get more steps? | Sanity check. No novelty claimed. |

### The baselines, and what each one is there to kill

Baselines are not decoration. Each one exists to close off a specific "but
couldn't you just..." objection:

- **B1 ReAct, no cost signal** — establishes how much waste exists to begin with.
- **B2 confidence probe** — "why not just ask the model if it's confident?" This is
  a *dangerous* baseline; published work shows it sometimes wins. Take it seriously.
- **B3 training-free monitor** — the published bar we must clear (−29.7% tokens).
- **B4/B5** — existing cost-aware RL methods: is per-step signal needed at all, and
  is a learned *value* better than a simple adaptive penalty?
- **B6** — one model doing everything.
- **B7** — trained self-termination; includes an imitation-only arm answering "is RL
  even necessary?"
- **B8** — a quality value model with cost bolted on: is the *stopping* semantics the
  thing that matters, or would any value function do?
- **B9 — the pivotal one.** Our exact machinery with the coach deleted. If B9 matches
  us, the coach does not earn its existence. We implement it at full strength
  deliberately, because a weak version of this baseline would be dishonest.
- **Oracle** — the ceiling, so we know how much headroom is left.

### Measurement rules that keep this honest

- **Everyone pays for everything.** Our coach's calls, B2's self-eval prompts,
  B3's monitor triggers — all billed under the same price map. Otherwise we would
  be comparing our costs against their costs-minus-overhead.
- **Every method gets swept over its own cost knob.** "Cost at equal accuracy" is
  meaningless if one method is at a different operating point.
- **Everything in dollars**, not tokens, so tool fees and model prices are comparable.
- **Frozen eval sets, chosen before any method runs.** No moving targets.
- **Statistics only where n ≥ 500.** On a 103-question set the confidence interval
  is roughly ±9 points; claiming a 3-point win there invites a deserved rebuttal.

---

## 5. What we run on

- **Multi-hop QA + search** — primary training domain. Cheap to score (compare the
  draft answer to the gold answer — free), and it is the same data the methods we
  compare against used.
- **ALFWorld (embodied tasks)** — second training domain. Its success rate saturates,
  which is actually useful: when everyone succeeds, differences are pure efficiency.
- **Web research, math** — evaluation only, to test transfer.
- **SWE-bench — deliberately dropped.** Scoring quality mid-task would mean running
  the test suite at every step, and intermediate patches score ~0 until the very
  end, which collapses the labels. Cutting this was a decision, not an oversight.

---

## 6. What we honestly expect, and what breaks it

Target: **20–40% dollar-cost reduction at equal accuracy** on two domains.

Every hypothesis has a pre-registered fallback, so no single failure kills the
work — except one:

| If this fails | Then |
|---|---|
| Beating training-free baselines | Paper becomes "when does learned stopping help?" — still publishable |
| **Training beats monitoring (H2)** | **Kill-switch. Pivot to the monitor paper.** |
| Beating direct shaping | Coach becomes optional; paper is about the labels |
| Two models beating one | Claim rests on transfer and controllability instead |
| Keeping savings with monitor off | "Partial internalization" |

### The known ways this can go wrong

- **Reward hacking.** The worker writes the draft the coach reads, so it could freeze
  a wrong draft to fake "stability" and trick the coach into saying stop. We watch
  for this by tracking whether the coach's predicted value drifts away from actual
  measured reward, and by refreshing the coach every iteration against fresh ground
  truth. This is a documented failure mode elsewhere (one paper measured 82% → 70%).
- **Getting scooped.** Several closely adjacent papers are only weeks old. This is
  why the kill-switches run in week 2–3 rather than month 3.
- **Silent no-op.** The step-level advantage issue from §2C. Guard it in code review.

---

## 7. How to actually run it

Twelve phases, P0 through P11, in `research/cassi/scripts/`. Full detail with
done-criteria is `paper_plan_v2.md` §16; current status is `HANDOFF.md`.

The shape of it:

1. **P0–P1** — install the pinned stack, download and decontaminate data. *(Done.)*
2. **P2** — pilot run, then **freeze the wallet calibration into the config**. Nothing
   downstream is valid until this happens — the config loader physically blocks
   later phases until it does. Also: the prompt template must be frozen *before*
   this pilot, because the template affects spend and spend defines the wallets.
3. **P3–P4** — collect trajectories, build labels, train the coach.
4. **P5** — the kill-switches. **GO/NO-GO decision point.**
5. **P6–P8** — main training, baselines, evaluation.
6. **P9–P11** — figures, tables, paper.

Everything from P2 onward needs GPUs.

---

## 8. If you are new here, read in this order

1. This file — the intuition.
2. `research/cassi/PROJECT_GUIDE.md` — the same material with code symbols attached,
   plus a worked numeric example and the invariants you must not break.
3. `research/cassi/HANDOFF.md` — what is done, what runs next, exact commands.
4. `research/paper_plan_v2.md` — the source of truth for every detail.

Two standing rules: never read anything under `research/archived/` (stale, will
mislead you), and `git config user.name` must print `Nathanael Brian` before you
commit.
