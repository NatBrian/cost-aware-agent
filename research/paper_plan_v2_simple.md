# Paper Plan v2 — The Plain-Language Edition

**What this file is.** `paper_plan_v2.md` is the real plan, written in research
shorthand. This file covers *the same ground, completely* — nothing removed —
but in plain words with the reasoning spelled out. "Simple" here means *easy to
read*, not *less information*. If you read only one document to understand this
project end to end, read this one.

**Rule:** if this file and `paper_plan_v2.md` ever disagree, `paper_plan_v2.md`
wins. This is a translation, not a second opinion.

**Contents**

1. The problem
2. The method (coach and worker)
3. The competition — who did what, and how we differ
4. What we claim is new (and what we explicitly don't claim)
5. The models, datasets, and benchmarks — every choice and why
6. The training methods — exactly how each model is trained
7. The baselines — all nine, and what each one is there to kill
8. The experiments — what, why, how
9. How we measure, and the rules that keep it honest
10. What we expect, and everything that could go wrong
11. The twelve phases, start to finish
12. What the paper itself looks like
13. Where to go next

---

## 1. The problem

### The story

Imagine you hire a research assistant and pay them by the hour. You ask a
question. A good assistant searches a bit, forms an answer, and then — this is
the part that matters — *notices when more searching is no longer worth your
money* and hands you the answer.

Today's AI agents cannot do that last part. They search, reason, call tools, and
nothing in their training ever told them "this next step costs money and probably
won't improve the answer." So they keep going.

### This waste is measured, not assumed

- Reasoning accuracy follows an upside-down U as chains get longer. Past the peak,
  more thinking makes accuracy *worse* (Wu et al., 2502.07266).
- Out of 20 sampled reasoning chains, the shortest is **34.5 points more accurate**
  than the longest (Hassid et al., 2505.17813).
- In agents specifically, an "overthinking score" predicts task failure with
  R² = 0.892; picking the low-overthinking runs cut cost **43%** while *raising*
  success **24%** (Cuadron et al., 2502.08235).
- Outcome-only RL actively *causes* over-searching (DAS, 2602.03304, WWW'26).

But the opposite failure is real too: genuinely hard problems need up to **2.9×**
more tokens. So you cannot just cap everything — uniform throttling breaks the
hard instances. Whatever we build has to be *state-dependent*: spend where it
pays, stop where it doesn't.

### Why not just ask the model "are you done?"

People tried. It measurably fails.

- GPT-4 as a stopping judge leaves a large gap versus optimal under matched
  budgets (Token Economies, 2406.06461, EMNLP'24).
- On a benchmark built specifically for this, LLMs detect redundant agent steps at
  **≤24.88% step-level F1** (RedundancyBench, 2605.29893). That is close to useless.
- Calibrated confidence probes do better, but they answer *"am I right?"* — not
  *"is the next step worth its price?"* Those are different questions. A model can
  correctly know it isn't finished and still be wrong to continue, because
  finishing costs more than it's worth.

**This is the core insight:** stopping is an *economic* judgment — marginal value
versus marginal cost — and no existing training signal supplies it.

### How this connects to the rest of this repo

The `cost_aware_agent/` harness in this repo meters real dollar costs per step for
frontier agents at *inference* time. CASSI is the *training-side* counterpart: it
consumes exactly the budget features the harness already computes, and produces
agents that need less external throttling. The harness is the deployment story;
CASSI is the learning story. Same price map, same features — no duplication.

---

## 2. The method

Two models: a **coach** and a **worker**.

- The **worker** (Qwen3.5-9B) does the task — searches, reasons, answers.
- The **coach** (Qwen3.5-2B) watches every step and estimates: *how much value is
  left in continuing, given what continuing costs?*

Three ingredients, then a loop.

### Ingredient A — Where the coach's answers come from

We teach the coach what the *right* stopping point was, by looking backward at
trajectories the worker already produced.

For each step we compute a score:

```
value at step t  =  how good the answer is right now  −  λ × how much we've spent
```

`λ` (lambda) is the knob for how much money matters. λ high = cost matters a lot.
λ = 0 = money is free. We use λ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}, with **1.0 as the
headline**.

**The mistake we must not make.** The obvious approach is: look at the whole
finished trajectory, find whichever step scored best, call that the right answer.
This is **wrong**, and understanding why is essential.

That approach lets the label peek at the future. It asks "knowing how this run
turned out, when should we have stopped?" But a real agent deciding in the moment
doesn't know how the run will turn out. Optimal-stopping theory (prophet
inequalities, Krengel–Sucheston) proves the peeking version *systematically
overvalues continuing* — so a model trained on it stops too late, always. Our own
RL expert audit flagged this. Version 5 of this plan used it; v2 does not.

**The fix — the Snell envelope.** This is classical machinery (Longstaff–Schwartz;
the same math finance uses to price American options, where you also must decide
"exercise now or wait?" without knowing the future). Computed backward:

```
at the last step:  V = the score right now
at earlier steps:  V = max( score right now ,  expected score if we continue )
```

The crucial detail: "expected score if we continue" is fitted **across many
trajectories at once**, not read off one path. We collect 8 rollouts per task, so
at every step there's a cross-section to regress on. The label therefore says
"continuing from a state like this is worth about this much *on average*" — a
judgment a real agent could actually make. In technical terms we replaced
*mean-of-max* (impossible to implement) with *max-of-mean* (implementable).

The outputs per step are:

| Symbol | Meaning |
|---|---|
| `Δ*` (margin) | expected-continue-value minus stop-now-value. Positive → keep going. |
| `τ*` (tau star) | the first step where stopping beats continuing — the optimal stop point |
| `V*` | the value itself, which the worker's training needs later |

**Cost is nothing extra.** These labels come from trajectories the RL loop already
collects. No additional model calls. That said, we are honest about what *isn't*
free: the running-draft template tokens and the forced-continuation overhead are
real, and both are billed explicitly in our cost accounting (Table T4).

**Two subtleties worth knowing.** First, at the final step there's nothing to
continue into, so stopping is correct by construction. Second, τ* is optimal
*relative to the current worker's behavior* — which is exactly why we re-solve it
after the worker improves (the loop, below).

### Budget awareness, learned rather than hard-coded

A good assistant behaves differently when the budget is nearly gone. We want that,
but we want it *learned*, not written as an if-statement. Two mechanisms:

**1. Random wallets during collection.** Every (task, rollout-group) draws a wallet
— small, medium, or large — calibrated from a 200-task pilot: small = the 25th
percentile of observed spend, medium = the 75th, large = 2× the 90th.

One critical detail: **all 8 rollouts in a group share the same wallet.** Group
comparisons must compare behavior, not wallet luck. If rollouts in a group had
different wallets, the training signal would confound "this behavior was better"
with "this run happened to be richer." Wallets vary *between* groups, so the model
still sees budget variety.

**2. Cost gets more expensive as the wallet empties.** A dollar spent when you're
nearly broke hurts more than a dollar spent when you're flush. So costs are scaled
by a multiplier depending on how much remains:

| Budget remaining | Tier | Multiplier |
|---|---|---|
| >60% | HIGH | 0.5× |
| 30–60% | MEDIUM | 1.0× |
| 10–30% | LOW | 2.0× |
| <10% | CRITICAL | 5.0× |

This is a discretized Lagrangian relaxation of the budget-constrained problem — as
the wallet empties, the shadow price of further spend rises, so the same backward
recursion naturally produces earlier stopping. Version 5 applied this schedule as
a hand-written *inference-time* rule; here it goes into the **training data**, so
"stop earlier when nearly broke" becomes in-weights behavior. (A recent paper,
BAAR 2602.21227, explicitly names remaining-budget conditioning as future work —
we do it.)

The hand-written-rule version survives as ablation A8, and beating it is a
supporting result for the whole learned-over-rules thesis.

### Ingredient B — Training the coach

A small model with **three output heads**, trained to predict, from mid-task state:

1. **Stop or continue** — classification (cross-entropy loss, weight 1.0)
2. **The margin Δ** — regression (MSE, weight 0.5), squashed to [−1, 1]; this drives
   stop decisions
3. **The value V** — regression (MSE, weight 0.5), unsquashed; this feeds the worker's
   training

Two design decisions that matter:

**The coach only sees what a real deployment would have.** Its inputs are: the task
text; budget state (tokens used, tool calls, dollars, percent of allowance, burn
rate, tier); progress signals (step index, how many steps since the draft answer
changed, edit distance over the last three drafts, retrieval overlap, distinct
sources); the current draft answer; and compressed recent history. It **never** sees
the ground-truth answer, and — deliberately — **never sees the worker's stated
confidence** (that's a reward-hacking channel; see §10). This is a hard rule: if
the coach needs privileged information, it's useless at deployment.

**One coach covers every λ.** λ is part of its input text. So at deployment you turn
the cost dial without retraining anything. This is what makes "a principled cost
knob" a real feature rather than marketing. (That turning λ up should stop earlier
is classical comparative statics — Topkis — which we cite rather than claim.)

**Why a separate small model, honestly.** We do *not* claim a theorem here — single
cost-aware models demonstrably work (ALP, EAPO, DASH). The defensible reasons are:

1. **Privileged-information hygiene** (the SWEET-RL argument): the coach is trained
   using ground-truth quality that the worker must never see at inference. Separate
   models keep that boundary clean and auditable.
2. **Reusability:** one coach can supervise multiple workers, and can control frozen
   third-party agents you cannot retrain.
3. Whether separation *also* wins on raw performance is an **open empirical question
   we test at matched parameter count** (experiment A2 / kill-switch K2) — with a
   pre-registered fallback if it doesn't.

### Ingredient C — Using the coach to train the worker

This is the novel step, and it needs care.

The worker's base reward is the economic outcome — **the same economics the labels
use.** The coach, the worker, and the labels all optimize one objective, not three:

```
reward  =  final answer quality  −  λ × (tier-scaled, normalized total spend)
```

**The trap.** The obvious approach — "give a bonus at every promising step" — pays
the worker to *accumulate* promising steps, so it dawdles. That's exactly backwards.
Version 5 of this plan did this. It's a known failure (Ng, Harada & Russell, 1999).

**The correct approach — potential-based shaping.** If the added reward takes the
form of a *difference* in some potential function:

```
reward at step t  =  (coach's value at the next state) − (coach's value at this state)
```

then it is proven you have **not changed which behavior is optimal**. You've only
made feedback denser: instead of learning only at the end ("that run cost too
much"), the worker gets a signal every step. We set the potential to the coach's
value estimate, and by convention the potential at the terminal state is zero.

**The consequence you must not trip on.** These difference terms telescope — they
sum to a constant within a group. So if you compute advantages at the
whole-trajectory level, **the shaping cancels out completely and does literally
nothing.** Therefore **step-level credit assignment is mandatory, not optional.**

This is the single most dangerous line in the codebase. If someone "simplifies" it
to trajectory-level advantages, CASSI silently becomes a no-op and every result
goes flat with no error message. Guard it in code review.

Concretely: per-step returns-to-go, normalized within the group at each step index,
with a **minimum-cohort guard of 3** — because trajectories have different lengths,
late steps can have very few surviving group members, and normalizing over one or
two samples is noise. When fewer than 3 remain at a step, those steps fall back to
the trajectory-level baseline. That fallback applies *only* to those steps; it is
never a global fallback, for the reason above.

**Said plainly: the coach converts a reward you only learn at the end into feedback
at every step, without changing what "good" means.** Every step that moves the state
toward "stop-worthy with high value" gets paid its marginal contribution; steps past
the optimal stopping frontier get negative reward automatically.

**Honest positioning.** Potential-based shaping inside group-relative RL is already
established — we cite and borrow, we don't claim it. SHAPE (2604.06636, ACL'26) does
segment-level potential differences in GRPO and analyzes this exact telescoping
problem. TIPS (2603.22293) applies it to search agents. DIVER (ICLR'26) uses it for
invariance. **Our claim is what the potential *means*, not the shaping mechanics:**
every published potential is success-rate, teacher-likelihood, or diversity based.
None is a trained cost-aware stopping value.

### Ingredient D — The loop

Once the worker improves, the labels are stale — they describe a worker that no
longer exists. So: collect with the new worker → new labels → new coach → train
again. **At least two full iterations**, reported per-iteration.

**The forced-continuation trick.** During *label collection* we suppress
termination: when the worker says ANSWER, we log it as a draft event and force the
trajectory to run to the maximum step count anyway. Otherwise labels are censored by
the worker's own stopping choices — and, deliciously perversely, iteration-2 data
would vanish *precisely because internalization worked*. As a free bonus, those
logged would-have-stopped positions give us a self-stopping measurement at no cost.

During *RL training* rollouts, termination is normal — the worker must experience
real stopping economics. These two modes are easy to confuse and the distinction is
load-bearing.

**The honesty control.** Iteration-2 gains could just be "we trained longer." So we
run iteration 2 **twice at matched compute**: once with a refreshed coach, once with
the frozen old coach. The difference between them *is* the loop's contribution.
Without this control, no loop claim survives review. We also never call this "the
first self-reinforcing cycle" — that phrase is already taken (see §4).

### Inference — how it actually runs

The worker runs. The coach evaluates each step and stops the episode when the
margin goes to zero or below. **One fixed threshold, no rule table** — budget
sensitivity is already in the weights. The user's cost dial is the λ fed to the
coach; sweeping it traces the whole cost/quality frontier on a *fixed* worker, no
retraining.

**And the key measurement:** because the worker was *trained* with economics, it
should stop *itself* before the coach ever fires. So we report (i) what fraction of
episodes self-terminate, and (ii) cost and accuracy with the coach **switched off
entirely**. That's the cleanest possible evidence that economics moved into the
policy — and it is something no inference-time-control paper can produce.

---

## 3. The competition — who did what, and how we differ

This is the section reviewers care most about. Every component we use exists
somewhere. The *composition* does not. Here is the honest map, by family.

### Family 1 — Hindsight stop labels + a trained stopper

**Who:** TERMINATOR (2603.12529), OS-Pruner (2607.11089), LYNX (2512.05325),
LearnStop (2606.30852), Chen et al. (ICML'20).

**What they did:** exactly our Ingredients A and B. Look at finished trajectories,
derive exit labels, train a small stopper. OS-Pruner even optimizes
`accuracy − λ·tokens`, which is startlingly close to our objective.

**How we differ:** they are all **inference-time only** — the stopper controls a
*frozen* model, and economics never enter its weights. They price **tokens only**,
not tool fees or real dollars. And their labels are direct — TERMINATOR uses
first-answer-arrival positions, OS-Pruner does policy gradient on the stop
distribution — nobody does backward recursion. **Our delta: we train the agent with
it, we price real money, and our labels are non-anticipating.**

### Family 2 — Agent early-stopping / termination

**Who:** CaRT (2510.08517), BAGEN (2606.00198), SIM-RAG (SIGIR'25), DAS (2602.03304).

**What they did:** trained stop/continue decisions for agents specifically.

**How we differ:** none has a cost objective. CaRT uses a discount factor as a
proxy; BAGEN has no bridge to training; DAS trains via DPO but only on the stopping
boundary. **Our delta: an explicit priced objective, and the value feeding step-level
training.** CaRT is our baseline B7.

### Family 3 — Monitors over frozen agents

**Who:** SupervisorAgent (2510.26585, ICLR'26 — **−29.7% tokens on GAIA at parity**),
Ares (2603.07915), TAB (2604.05164), CoRL (2511.02755), VoI control (2605.05701),
Dynasor (2412.20993).

**What they did:** runtime cost control over agents they don't touch. SupervisorAgent
is the published bar we must clear, and it is a strong one.

**How we differ:** the policy stays frozen — economics *never* enter the weights.
This family is precisely what our headline comparison is against, and it's why the
monitor-off experiment exists: they structurally cannot run that experiment.
SupervisorAgent is baseline B3.

### Family 4 — Cost-aware agent RL

**Who:** OTC-PO (2504.14870), EAPO (2606.02132), SlimSearcher (2606.07074),
AdaTIR (2601.14696), Agent-Omit (2602.04284, ICML'26), RaM/VOC (2410.05563).

**What they did:** put cost terms into the reward and train on it. Closest in
*spirit* to us.

**How we differ:** their cost signal is a **single scalar at the end of the
trajectory**. There is no state-dependent stopping value, so no per-step credit,
and nothing transferable — you can't lift their cost signal onto a different agent.
**Our delta: the step-level economic value, and a coach that transfers.** These are
baselines B4 and B5.

### Family 5 — Adaptive length control for reasoning

**Who:** ALP (2506.05256, NeurIPS'25 spotlight), DAST, LASER-D, AdaptThink,
AdaCtrl, HAPO, L1 (COLM'25), Arora & Zanette (NeurIPS'25).

**What they did:** per-instance token budgets via RL — genuinely adaptive, scaled by
solve rate or difficulty tags.

**How we differ:** single-shot chain-of-thought tokens, no tools, no fees, and the
pressure is applied **before generation** rather than reassessed mid-trajectory. They
own per-instance difficulty adaptation and **we explicitly do not claim it.**

### Family 6 — Agent process reward models and step credit

**Who:** AgentPRM (2502.10325), AgentPRM-Fudan (2511.08325, WWW'26), Agent-RRM
(2601.22154), SWEET-RL (2503.15478), GiGPO (2505.10978), SPA-RL, PRIME, Implicit
PRM, DASH (2607.00482), RePro, MRT, PAV.

**What they did:** dense per-step signals for policy training — structurally our
Ingredient C.

**How we differ, and this is the sharpest line in the paper: *no PRM encodes cost.*
Every one of them is success- or progress-semantics.** There's no stopping margin
anywhere. DASH is closest on labels (post-hoc labels as advantages) and becomes our
pivotal baseline B9. AgentPRM becomes B8.

**Correction we must not repeat:** v5 claimed a complexity advantage over PRM
training. That was wrong — AgentPRM uses pooled return-to-go, not per-state Monte
Carlo, and rollout-free step labels already exist (Implicit PRM, PRIME, TD+GAE).
**We make no complexity claim.** Our efficiency point is narrow and true: cost-aware
*stopping* labels are free given intermediate quality measurements.

### Family 7 — Co-evolution loops

**Who:** iStar (2509.19199), Self-Guide (2604.03098), Cooper (2508.05613),
SPARK (2509.22624), Self-Rewarding, the ReST/STaR lineage.

**What they did:** iterate a policy and its evaluator against each other.

**How we differ:** we don't. **We instantiate a known loop; we do not invent one.**
V5 claimed "first self-reinforcing cycle" — those papers occupy that phrase verbatim
and AgentPRM already implements the loop. Claim dropped. Ours is a *cost-aware
stopping-centric instantiation*, and we say exactly that.

### Family 8 — Reward shaping mechanics

**Who:** SHAPE (2604.06636, ACL'26), TIPS (2603.22293), DIVER (ICLR'26),
Stop-RAG (2510.14337), SPAE (2601.03823).

**What they did:** SHAPE does PBRS in GRPO and fixes the telescoping issue. TIPS
does turn-level PBRS for tool agents with a public repo. **Stop-RAG is our closest
structural precursor** — it genuinely does fitted-Q backward recursion over completed
RAG trajectories with a max-bootstrap.

**How we differ:** all published potentials are success/teacher/diversity based —
none is a trained cost-aware stopping value. Stop-RAG is quality-only (no cost
term), inference-time-only, and has no optimal-stopping framing. **Our deltas: the
priced objective, multi-dimensional cost, and the training bridge.** We cite it
prominently; hiding it would be the kind of thing reviewers find.

### Family 9 — The classical theory

Hansen & Zilberstein (AIJ'01) on anytime monitoring — literally our oracle.
Weitzman '79, Russell & Wefald '91, Hay '12, prophet inequalities, Topkis, and
Cognitive Friction (2603.30031, HJB for tool agents). **We transfer this machinery
with citation. We do not claim to have invented any of it.** The contribution is the
LLM-agent instantiation.

### The one-paragraph summary

Hindsight stop labels plus a small trained stopper exist — **inference-time only**.
Learned monitor → process reward → agent RL exists — **quality-only**. Cost-aware
agent RL exists — **outcome-level only**. **Nobody** converts an explicit
quality−λ·cost hindsight optimum into a trained stopping-value model whose margin
trains a tool-using agent at the step level, and nobody has measured whether
stopping economics can be *internalized* rather than enforced. An orchestration
survey (2605.02801) independently observes that the stop decision is never an RL
target. That's the paper.

### Errors from v5 that must never come back

AgentPRM is *not* per-state Monte Carlo (it's pooled return-to-go) · IterResearch is
RL-trained (2511.07327), not a heuristic · BATS is Google's prompt-level budget
tracker (2511.17006) · "Learning to Stop While Learning to Predict" is Chen et al. ·
the two different AgentPRM papers must be disambiguated · CARL (2512.04949)
authorship is unverified.

---

## 4. What we claim, and what we don't

### The five contributions

1. **The cost-aware stopping bridge** — first method turning explicit quality−λ·cost
   hindsight optima into a trained stopping value that supervises a tool-using agent
   step by step. *Proven by E1 and kill-switch K1.*
2. **Internalized economic behavior** — first measurement of whether stopping
   economics can be trained *into* a policy rather than enforced at runtime.
   *Proven by E2 (monitor-off, self-termination, transfer).*
3. **Measurable stopping labels for agents** — Snell-envelope labels from
   already-collected trajectories, with a bias analysis showing why the naive
   argmax choice stops late. *Proven by E4.*
4. **A dollar-denominated objective with learned budget-awareness and a working λ
   dial** — tokens plus tool fees plus real dollars, wallet-awareness trained in
   rather than ruled in. *Proven by E3.*
5. **An honest map of when learned stopping helps** — matched-parameter
   single-vs-two-model comparison, matched-risk protocol, overhead by serving
   regime, and a low-slack control. *Framed as findings either way.*

### Explicit non-claims (stated in the paper, deliberately)

- **Not** "the first self-reinforcing cycle" — taken, verbatim, by iStar/Self-Guide.
- **No** complexity story about PRM training — the premise was factually wrong.
- **No** "representation conflict theorem" for two models — we test it empirically
  instead, and might lose.
- **No** novelty for per-instance difficulty adaptation — ALP/DAST/Ares own it.
- **No** novelty for a small model supervising a large one — Ares/TAB/SupervisorAgent
  own it.

Every one of these was a v5 claim that an audit falsified. Listing them protects us:
a reviewer who spots an overclaim rejects the paper; a paper that names its own
limits earns trust.

---

## 5. Models, datasets, benchmarks — every choice and why

The rule: **build what we claim, reuse what we compare on, install the rest.**
Everything reused is pinned to a commit hash and never modified internally.

### Models

| Role | Model | Why this one |
|---|---|---|
| **Worker** | `Qwen/Qwen3.5-9B` (Apache 2.0) | Newest open model in the 8–9B range, tool-native, and verl ships GRPO demos for it. Trained with thinking mode **off**. |
| **Coach** | `Qwen/Qwen3.5-2B` | Same family and chat template as the worker, which isolates the method from family differences. Ablations at 0.8B and 4B. |
| **Transfer (different family)** | `mistralai/Ministral-3-8B-Instruct-2512` | Tests that the coach isn't Qwen-specific. |
| **Transfer (different size)** | `Qwen/Qwen3.5-4B` | Tests transfer across scale. |

**If a reviewer challenges the model choice:** Qwen3.6 (Apr 2026) shipped nothing
below 27B, so Qwen3.5-9B is the direct successor of the backbones Search-R1 and
GiGPO used, and the community-standard agentic-RL backbone at submission time. For
Ministral: no open Llama below 27B has shipped since 2024 (Llama-3.1-8B would be 2.5
years old at review) and Gemma-4 E4B is only 4.5B effective — Ministral-3-8B is the
freshest like-for-like non-Qwen 8B.

### Infrastructure

| Piece | Choice | Why |
|---|---|---|
| RL framework | **verl ≥ 0.8.0** | Native multi-turn agent rollouts since 0.7; the framework of essentially every 2025–26 agent-RL paper. Alternatives (AReaL, SkyRL, slime, ROLL) are niche or async-first. |
| Search environment | **verl-tool** + Search-R1 | The *same environment as our baselines* — this is what keeps cost comparisons meaningful. |
| Embodied environment | **verl-agent** (GiGPO's official code) | Free comparability with GiGPO and AgentPRM. |
| Retriever | **E5 + Wikipedia-21M** (BM25 fallback) | Identical to baselines, so differences are the policy, not retrieval. One Qwen3-Embedding-0.6B ablation for robustness. |
| Coach training | **HF TRL v1.8** SFTTrainer + scalar head | Current standard. Note: the legacy value-head wrapper moved to `trl.experimental` — do not use it. |
| Cost accounting | This repo's price map + **HAL harness conventions** | HAL (2510.11977) is the 2026 standard for agent dollar accounting; adopting it preempts "how did you count cost?" |

Hardware: one 8×H200 node. Requires transformers v5 and vLLM ≥ 0.17.

### Training datasets

| Data | Amount | Role |
|---|---|---|
| Natural Questions + HotpotQA (train) | 8–10K sampled | Primary RL training |
| MuSiQue (train) | 5K sampled | Primary RL training (harder multi-hop) |
| ALFWorld train tasks | — | Second RL domain |

**Why this exact mix:** it is the standard corpus of the 2026 Search-R1 successors
(Search-R2, CoSearch, Search-E1; StepSearch trains on MuSiQue) *and* of OTC-PO, our
closest cost-aware baseline. Changing it would break comparability with the numbers
we most need to compare against.

### Evaluation benchmarks

| Benchmark | Size | Role | Why |
|---|---|---|---|
| HotpotQA dev | 1,000 | In-domain | Frozen subsample, chosen before any run |
| MuSiQue dev | 500 | In-domain | Harder multi-hop |
| **BrowseComp-Plus** | 830 | Out-of-domain | ACL'26 variant of BrowseComp with a *fixed local 100K-doc corpus* — no paid APIs, no live-search contamination. The reproducibility answer to "why not BrowseComp?" |
| **GAIA text-only dev** | 103 | Coach-as-monitor transfer | The benchmark of exactly the line we compare against (SupervisorAgent, SlimSearcher, Tongyi DeepResearch) |
| Bamboogle | 125 | Out-of-domain | Search-R1-line protocol; doubles as our fresh-data contamination check |
| 2WikiMultihopQA dev | 500 | Out-of-domain | Same protocol, fully local |
| **MATH-500 + AIME 2025** | 500 + — | Low-slack control | See below |
| RedundancyBench | — | Coach validation | Turns a threat paper into an eval for us |
| OptimalThinkingBench | — | Cheap sanity check | Agent-free over/under-thinking check |

**Why the low-slack control matters.** MATH-500 is kept for comparability with
ALP/DAST/LASER — but ALP's own results show savings concentrate there *because it's
the easy set*. AIME 2025 is the genuinely hard set where there is no slack to cut.
Its job is to prove **we don't strangle hard problems.** A method that saves money
everywhere including where it shouldn't is a broken method.

**Splitting the OOD evaluation by role** — a subtle but important fix. The *trained
worker* is tested on BrowseComp-Plus, Bamboogle, and 2Wiki, all of which use local
retrieval matching what it trained on. The *coach as a monitor* is tested on GAIA
over a frozen live-web agent, which is exactly SupervisorAgent's setup so the
comparison is direct. Running our trained worker on live-web GAIA as a headline
would confound "stopping transferred" with "the tool API changed" — so that's
appendix-only.

### What we deliberately dropped

**SWE-bench RL is out of scope**, and this was a decision, not an oversight:

1. 7B-scale SWE RL doesn't work — DeepSWE needed 32B and a cluster.
2. Scoring quality mid-task means running the test suite at *every step*.
3. Intermediate patches score ~0 until the very end, which collapses the labels to
   "stop at the last step" — degenerate and useless.

An inference-time monitor over a frozen 32B SWE agent may appear in the appendix.
Also considered and rejected: AppWorld (only 105 training tasks) and tau2-bench
(needs an LLM user simulator, which pollutes cost accounting).

---

## 6. The training methods — exactly how each model is trained

### Stage 1: Collecting trajectories

The base worker runs with **8 rollouts per task**, up to 10 steps for QA and 20 for
ALFWorld, in **forced-continuation mode**. Every step logs full state features,
dollar cost, budget tier, and the running draft answer.

**The running draft** is central and easy to overlook. Every agent — ours *and every
baseline* — must end each step with one line:

```
BEST ANSWER SO FAR: {one line; or EMPTY_DRAFT if none yet}
```

This solves the train/deploy symmetry problem: the draft exists identically at
collection, training, and inference. It makes quality scoring free — just compare
the logged draft to the gold answer, no extra generation. And because *every*
method emits it and pays for its tokens, it's a constant, not an advantage.

Target: ≥8,000 QA trajectories and ≥2,000 ALFWorld trajectories.

### Stage 2: Building labels

Run the backward recursion for each λ value. The regressor that estimates
continuation value is **LightGBM** (fallback: a small MLP) — deliberately *not* the
coach model itself, to avoid the label being coupled to the thing it trains.

Three quality checks before proceeding: manual review of 100 random trajectories
("does the chosen stop point look sane to a human?"), a label-noise sensitivity
re-run, and one sanity check that must hold — **higher λ must produce earlier
stopping, everywhere.** If that fails, something is wrong upstream.

### Stage 3: Training the coach (supervised fine-tuning)

| Setting | Value |
|---|---|
| Method | SFT, three heads |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 64 |
| Max sequence | 2,048 |
| **Early stopping on** | **held-out stopping regret — NOT cross-entropy** |

That last row matters. Cross-entropy measures whether it classifies stop/continue
correctly; stopping regret measures *how much utility the mistakes actually cost*.
Getting a close call wrong is cheap; getting a wildly-off call wrong is expensive.
Only the second metric knows the difference.

**Gate:** the coach must beat both a majority-class baseline and a calibrated
confidence probe on held-out regret. **If it can't, stop — fix features or labels
before touching RL.** A bad coach makes everything downstream meaningless.

### Stage 4: Training the worker (reinforcement learning)

| Setting | Value |
|---|---|
| Algorithm | **GRPO**, group size 8 |
| Learning rate | 5e-6 |
| KL penalty β | 0.04 |
| Clip ε | 0.2 |
| Length normalization | **Dr. GRPO** |
| Advantages | **step-level** (mandatory), min-cohort guard 3 |
| Rollout / eval temperature | 1.0 / 0.0 |
| Discount γ, format weight | 1.0, 0.1 |
| Iterations | ≥2 |

**Why GRPO:** it compares a *group* of rollouts on the same task against each other
rather than needing a separate learned critic. Since we already need multiple
rollouts per task for the label cross-section, the group structure is free.

**Why Dr. GRPO is non-negotiable:** standard GRPO has a documented length bias. Our
headline claim is "we made things shorter and cheaper." If our estimator has a
built-in preference for shorter outputs, a reviewer will — correctly — say our
savings are an artifact. Dr. GRPO's unbiased length handling closes that objection
before it's raised.

**Two step-credit variants, both run:** SHAPE-style segment credit, and per-step
returns-to-go with group normalization. **Kill-switch K1 picks the production
variant**, and both results are logged either way.

**Training λ:** the headline worker is trained at λ = 1.0 **only**. The frontier
comes from the inference-time dial on that one fixed worker, plus a single λ = 0.3
worker as a spot-check that the dial tracks reality. We never train five workers —
that matrix explosion is what made v5's budget ~10× over.

### Compute budget

One 9B multi-turn GRPO run is roughly **1–3 days on 8×H200**. The plan is 30–35
training runs, tiered:

- CASSI: 2 domains × (iteration 1 + iteration 2 × 2 control arms)
- Trained baselines B4–B9: 2–3 frontier points each, 1 seed except headline points
- Training ablations: A2, A3, A6, and A9's two negative controls, 1 seed
- **Inference-only ablations (A4, A5, A7, A8) reuse already-trained models at near-zero
  cost** — this tiering is the whole reason the matrix fits the budget

**Total: 10–12 weeks.** W1–2 setup and collection · W2–3 kill-switches · W4–6 main
experiment · W7–8 loop and label study · W9–10 transfer and ablations · W11–12
analysis and writing.

---

## 7. The baselines — all nine

Baselines are not decoration. Each closes off a specific "but couldn't you just…"
objection. Every one runs in the same environment, with the same worker model, and
pays the same prices.

| # | Baseline | The objection it kills |
|---|---|---|
| **B1** | ReAct, no cost signal | "How much waste is there even?" — establishes the ceiling on possible savings |
| **B2** | Self-eval prompt + calibrated confidence probe (Dynasor-style) | **"Why not just ask the model?"** — training-free. **This is a dangerous baseline: LearnStop shows it sometimes wins.** Take it seriously; don't weaken it. |
| **B3** | SupervisorAgent-style monitor | The published ICLR'26 bar: −29.7% tokens at parity on GAIA |
| **B4** | OTC-GRPO | "Is step-level economic signal needed at all, or is outcome-level enough?" |
| **B5** | EAPO (fallback: agentic-ALP) | "Is a learned *value* better than a well-tuned adaptive penalty?" |
| **B6** | Single model, GRPO with cost in the reward | "Why two models?" (matched-parameter, pairs with A2) |
| **B7** | CaRT + cost — **two arms: SFT-only and +RL** | The SFT-only arm asks: **"Is RL even needed, or does imitation suffice?"** |
| **B8** | AgentPRM-cost, honestly implemented | "Is *stopping* semantics what matters, or would any value function plus a cost term do?" |
| **B9** | **DASH-style direct shaping** | **The pivotal test: does the coach earn its existence?** |
| — | Oracle (labels with ground truth) | The ceiling — how much headroom remains |

**On B9, the one that could sink us.** It is our exact step-level machinery with the
coach simply deleted — labels used directly as advantages. We implement it at *full
strength* on purpose, because a weak version would be a strawman and reviewers find
strawmen. Note it *must* use step-level machinery: trajectory-level shaping is
provably inert, so a trajectory-level B9 would be a rigged comparison.

If B9 ties us, the honest conclusion is that the coach isn't earning its keep in the
training path, and the paper becomes "Snell-label direct shaping for agents" with
the coach kept for runtime control and transfer. That fallback is pre-registered.

---

## 8. The experiments

### First: the kill-switches (weeks 2–3, before anything else)

The whole project rests on one bet. So we test that bet cheaply and early, and we
log the decision with a date either way.

**K1 — the bridge test.** *Does training on the coach's signal actually beat just
using the coach as a runtime monitor?*

- **How:** 1,000 HotpotQA tasks, 1 seed, λ = 1.0. Three arms: shaped training,
  controller-only, and B9 direct shaping. The shaped arm runs *both* step-credit
  variants as sub-arms — this is also how we pick the production variant. Each arm
  gets a 3-point cost-dial mini-frontier at λ ∈ {0.5, 1.0, 2.0} so that "at equal
  accuracy" is actually well-defined.
- **Pass condition, exactly:** `(cost_controller − cost_shaped) / cost_controller ≥ 0.03`
  **and** `cost_shaped ≤ cost_B9` (ties pass). Read as direction and magnitude, not
  statistical significance — it's one seed.
- **If it fails:** do not push on. Pivot to the honest smaller paper — "a trained
  cost-aware, wallet-conditioned stopping monitor for frozen agents" — reusing the
  coach we already trained and dropping worker RL entirely.

**K2 — the separation test.** *Do we need two models, or would one do?* Same task
set: one 9B doing both jobs versus 9B + 2B, with parameters counted honestly in the
reporting. Either outcome is publishable; it changes framing, not viability. **Note:
the single-multitask arm is a deliberate stub in the current codebase and must be
implemented before this phase. K1 alone does not authorize a GO.**

### Then the main experiments

| ID | What it does | Why it exists |
|---|---|---|
| **E1** | CASSI vs all 9 baselines, both training domains, 3 seeds at headline points | The headline result. Everything else supports it. |
| **E2** | **Monitor switched off** at test time; plus transfer to unseen datasets and to *different* worker models (cross-family and cross-scale) | The internalization proof. If the worker stays cheap with nothing watching, the economics genuinely moved into the weights. **No inference-time-control paper can run this experiment** — that's the point. |
| **E3** | Sweep λ on one fixed worker; plus **identical tasks under small/medium/large wallets** | Shows the cost dial works without retraining, and that the agent responds to *how much is left*, not just how much was spent — stop points shift earlier, answers degrade gracefully, and no rule was written anywhere. |
| **E4** | Snell labels vs naive peeking labels vs TD/GAE vs Monte-Carlo, at matched label compute | Tests whether the careful label construction was worth it. Measures both coach accuracy *and* the downstream worker result — a label can look good and still train a worse agent. |
| **E5** | Two loop iterations with the **frozen-coach control at matched compute** | The (refreshed − frozen) difference *is* the loop's contribution. Without it, iteration-2 gains are confounded with "more training." |
| **E6** | Do 4-hop questions get more steps than 2-hop? | Consistency check only. **No novelty claimed** — ALP already owns this. |

### The nine ablations

A1 coach size (0.8B/2B/4B — ReMA warns ~1B meta-models can collapse; we verify the
floor) · A2 one model vs two **at matched total parameters** · A3 potential-based vs
naive additive shaping (**prediction: additive dawdles** — this validates the §2C
argument empirically rather than only theoretically) · A4 which input features
matter (budget only / +history / +draft stability) · A5 how often the coach runs
(every step vs every k-th) · A6 SFT-only coach vs +RL calibration · A7 rationale
text on/off · A8 **learned budget-awareness vs the hand-written rule table** · A9
**two negative controls**:

- **Random coach** — replace the coach's outputs with noise. If the worker *still*
  improves, our gains were generic dense-reward regularization, not economics.
- **Shuffled-label coach** — train on permuted labels. Isolates whether label
  *content* matters.

These two are cheap and decisive, and they're the first thing a skeptical reviewer
would demand. Better to run them ourselves.

---

## 9. How we measure, and the rules that keep it honest

### The headline metrics

- **Cost at equal accuracy, and accuracy at equal cost**, in dollars, plus the area
  under the Pareto frontier.
- **The frontier protocol** (without which "at equal accuracy" is meaningless): every
  method is swept over **its own** cost knob to produce a 3–5 point frontier — ours
  is the λ dial, B2's is its confidence threshold, B3's is trigger sensitivity, B4's
  the tool-count coefficient, and so on. Equal-accuracy cost is read by interpolating
  between adjacent points. **Methods with no knob (B1, oracle) are single points and
  are excluded from equal-accuracy claims** rather than being awkwardly compared.
- **Stopping regret** — the *utility gap* to the optimal stop, not the step distance.
  Being 3 steps early is not equally bad in all situations; magnitude matters.
  - *Measurement problem:* what would have happened after you stopped doesn't exist in
    a normal trace. *Solution:* on a fixed 500-task subsample, run everything twice —
    once normally (real cost and accuracy) and once forced to continue to the end
    (which reveals the full curve and the true optimum). **The replay cost is billed to
    the analysis line, not to any method** — otherwise we'd charge methods for
    measurement they didn't ask for.
- **Internalization** — monitor-off cost and accuracy, and the self-termination
  percentage, at iterations 0, 1, and 2.
- **End-to-end honest accounting** — total dollars including draft template tokens,
  forced-continuation overhead, coach training, and coach inference, reported **per
  serving regime** (KV-cache fork vs black-box re-prefill). This matters because
  overhead can *flip sign* between regimes. Plus amortization, since training-free
  baselines have none.
- **Overoptimization diagnostics** — the coach's predicted value versus realized
  reward over training, plus backup residuals of the label regressor on held-out
  data (an error-compounding check).
- **External validity** — RedundancyBench F1 for the coach, OptimalThinkingBench as a
  cheap sanity check. CostBench is cited but not run (it measures cost-optimal
  *planning*, a different construct — running it would invite confusion).

### Billing symmetry

**Every method pays for all the auxiliary inference it uses** — our coach's calls,
B2's self-eval prompts, B3's monitor triggers, any judge model — under one price
map. Otherwise we'd be comparing our costs against their costs-minus-overhead, which
is the most common way efficiency papers cheat, deliberately or not.

Prices: local model tokens at $0.60/M input and $2.20/M output; web search at
$0.003/query + $0.001/result; HTTP fetch $0.0001/request; code execution
$0.0001/exec + $0.0001/sec; local retrieval $0.0001/query. The *relative weights* are
what matter, and they're constant for everyone.

Costs are also **normalized per domain** by dividing by the median unconstrained
spend from the pilot, so λ means the same thing across domains. Raw dollars are
still reported everywhere.

### The statistics

- **Seeds 42/123/789** on every headline number; single seed allowed for ablations
  and non-headline frontier points, and every caption says which.
- **Bootstrap** over test instances, 10,000 resamples, 95% CI on every number. Seed
  variability reported separately as mean ± standard deviation.
- **Paired comparisons** — identical task sets per method. Both a paired t-test and
  Wilcoxon signed-rank, both reported; **Wilcoxon governs**, because cost
  distributions are heavy-tailed and normality fails.
- **Pareto dominance via bootstrap** — resample, recompute both frontiers, report the
  fraction of resamples where we dominate at *every* shared accuracy level.
- **Holm–Bonferroni** correction across the 9 baselines within each domain, with the
  comparison family stated explicitly.
- **Effect sizes** — Cohen's d for cost deltas, absolute percentage-point differences
  for accuracy.
- **Matched-risk curves** (the LearnStop protocol) — compare savings at equal
  fractions of lost-correct answers (1%, 2%, 5%). This is the fairest possible
  stopping comparison: any method can save money by stopping early and being wrong.
- **Small-n policy** — hypothesis tests **only** where n ≥ 500. GAIA-103 and
  Bamboogle-125 are reported as point estimates labeled *transfer indicators, not
  hypothesis tests*. At n = 103 an accuracy CI is roughly ±9 points, so a
  "significant" 3-point claim there is arithmetically impossible and claiming one
  invites a statistics rebuttal that would taint the whole paper.
- **Frozen evaluation subsamples** — chosen once, before any method runs. Every
  method, seed, and frontier point sees the identical task list.
- **No cherry-picking** — all λ values and all seeds appear in the appendix; the
  headline λ is chosen on dev *before* test runs; kill-switch decisions are logged
  with dates in the repo.

### Contamination

2018–2022 QA data under a 2026 base model is a *certain* reviewer question, and
there's documented evidence (2507.10532) that Qwen-family contamination can distort
RL conclusions. Four defenses:

1. N-gram and MinHash decontamination of all training prompts against every eval set.
2. Adversarially-fresh sets in the suite (Bamboogle; BrowseComp-Plus's fixed corpus).
3. A closed-book no-tool ablation showing parametric memory alone can't solve the
   evals — so gains must come from tool use.
4. Replication on a second model family (the Ministral arm doubles as this control).

Plus the claim-level defense, which is the strongest one: **our headline metric is
cost at matched accuracy.** Memorization inflates accuracy for all compared methods
equally; it does not create a cost *difference* between them.

---

## 10. What we expect, and what could go wrong

**Target: 20–40% dollar-cost reduction at equal accuracy on two agent domains.**

### The six hypotheses, pre-registered with fallbacks

| ID | Hypothesis | If it fails |
|---|---|---|
| H1 | We beat the training-free baselines (B2/B3) on ≥2 domains | Paper becomes "when does learned stopping help agents?" — a regime study. Still publishable, different claim. |
| **H2** | **Training beats monitoring (same coach)** | **Kill-switch. Pivot to the monitor paper.** |
| H3 | We beat B9 direct shaping | Coach demoted to optional; paper is about the labels |
| H4 | Two models ≥ one model at matched parameters | Claim rests on transfer, privileged-info hygiene, and controllability |
| H5 | Worker keeps ≥70% of savings with monitor off | Softened to "partial internalization" |
| H6 | Savings hold under both serving regimes | Regime-conditional recommendation |

Only H2 is fatal. That's by design — every other failure has a real paper on the
other side of it.

### The risks

| Risk | What we do about it |
|---|---|
| **Reward hacking** | The worker writes the draft the coach reads, so it could freeze a wrong draft to fake "stability" and trick the coach. We are honest that harness-computed ≠ ungameable. Three defenses: the coach never sees stated confidence; the coach is **refreshed every iteration** against fresh ground truth, which re-grounds the stability features; and we track predicted-value-vs-actual-reward divergence and refresh on divergence. This is a *documented* failure mode — AgentPRM measured 82% → 70%. |
| Training-free baselines match us | H1 fallback; the low-slack control shows we at least don't break hard tasks |
| Direct shaping matches us | H3 fallback; the coach still buys transfer and runtime control |
| One model matches two | H4 fallback |
| Label noise from mid-trajectory drafts | The regression pools 8 rollouts; validation check in A5; report noise sensitivity |
| GRPO pathologies (length bias, entropy collapse, Echo Trap) | Dr. GRPO / DAPO hygiene; GiGPO-style anchors if credit dilution appears |
| **Getting scooped** | Several adjacent papers (DASH, RePro, SlimSearcher, BAGEN) are ≤7 weeks old, and the CMU CaRT/MRT group is working nearby. **This is why kill-switches run in week 2–3, with a workshop preprint of the K1 result if it lands.** |
| **Silent no-op** | The step-level advantage issue from §2C. Not in the original risk table, but it belongs there — it's the one failure that produces no error message. |

---

## 11. The twelve phases

Full detail with done-criteria is `paper_plan_v2.md` §16; current status is
`HANDOFF.md`. **Never start a phase before the previous one's done-criterion is met**
(the one exception: baseline implementation may overlap the main training).

| Phase | What happens | Done when |
|---|---|---|
| **P0** ✅ | Install the pinned stack, pin every commit hash | One ReAct rollout runs end-to-end on both domains with draft lines and cost logging |
| **P1** ✅ | Download data, build the retrieval index, freeze splits, decontaminate | Dataset manifest with counts and split hashes committed |
| **P2** | 200-task pilot → **freeze wallet calibration into the config** → collect round 0 | ≥8K QA + ≥2K ALFWorld trajectories with scored drafts and balanced wallet strata |
| **P3** | Build labels for every λ | Labeled datasets + a one-page label-quality memo |
| **P4** | Train coach v0 | **Coach beats majority-class AND a confidence probe on held-out regret. If not, STOP.** |
| **P5** | **Kill-switches K1 + K2** | **GO/NO-GO decision, logged with a date either way** |
| **P6** | Worker GRPO, iteration 1, both domains | Beats B1 on cost-at-equal-accuracy on dev, both domains |
| **P7** | Loop iteration 2, both arms | Per-iteration table **+ the (refreshed − frozen) delta** |
| **P8** | Baselines B2–B9 (overlaps P6–P7) | All evaluated on the same frozen test sets with the same accounting |
| **P9** | Full evaluation, ablations, transfer, regret replays, 3 seeds | Every number for the tables and figures exists as CSV with a generation script |
| **P10** | Figures and tables | Everything regenerates from raw results with one `make` target — **no hand-edited numbers** |
| **P11** | Write the paper | Compiled PDF, 8 pages + appendix, every claim traceable |

**Everything from P2 onward needs GPUs.**

Two things about P2 that are easy to get wrong and expensive to discover late: the
prompt template must be **frozen before the pilot** (the template affects spend, and
spend defines the wallets — change it after and the pilot must rerun), and nothing
downstream is valid until the calibration is written into the config. The config
loader physically blocks later phases until it is.

---

## 12. What the paper looks like

Eight pages plus appendix, targeting **ICLR 2027** (~Sept 2026 submission; fallback
NeurIPS/ICML 2027).

| § | Section | Pages |
|---|---|---|
| 1 | Introduction — the waste empirics, why prompting fails, contributions | 1.25 |
| 2 | Related work — the families table as prose, with explicit deltas | 1.0 |
| 3 | Problem formulation — the stopping problem and classical citations | 0.75 |
| 4 | Method — the algorithms, the invariance statement, anti-hacking design | 1.5 |
| 5 | Experimental setup — domains, baselines, metrics | 0.75 |
| 6 | Results — headline tables, per-iteration gains, **hypothesis verdicts including failures** | 1.75 |
| 7 | Analysis — internalization deep-dive, hacking diagnostics, label study | 0.75 |
| 8 | Limitations and conclusion | 0.5 |

**Six figures:** F1 the pipeline · F2 the shaping intuition drawn on one real
trajectory · F3 cost-accuracy frontiers per domain · F4 internalization (monitor-off
bars + self-stop rates across iterations) · F5 the label study · F6 the
predicted-vs-actual divergence curve.

**Five tables:** T1 headline cost at equal accuracy · T2 all baselines × both domains
· T3 ablations · T4 the honest overhead accounting per serving regime · T5 transfer
by role.

**Writing order** (deliberate — write what you know first): Method → Setup → Results
→ Analysis → Related Work → Intro → Limitations → **Abstract last**, built from the
headline numbers once they exist.

**Two rules before every compile:** grep the draft against the list of dead v5 claims
in §4 — none may reappear — and verify every number in the text exists in a table or
figure. And build citations from the verified IDs in `research/lit_review/`; **never
cite from memory.**

---

## 13. Where to go next

1. **This file** — the complete picture in plain language.
2. `research/cassi/PROJECT_GUIDE.md` — the same material with code symbols attached,
   plus a worked numeric example and the invariants you must not break.
3. `research/cassi/HANDOFF.md` — what's done, what runs next, exact commands.
4. `research/paper_plan_v2.md` — the source of truth for every detail.

Two standing rules: **never read anything under `research/archived/`** (stale, will
mislead you), and `git config user.name` must print `Nathanael Brian` before you
commit anything.
