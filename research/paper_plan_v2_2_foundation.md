# Paper Plan v2.2 — FOUNDATION-2, the redesigned pipeline-validation run

> **Status:** DRAFT for review (2026-07-31) — no code written yet. This document
> supersedes `paper_plan_v2_1_foundation.md` (FOUNDATION-1) as the active plan.
> FOUNDATION-1 is **not deleted**: it ran to completion, produced a GO, and its
> results and failure modes are the entire input to this redesign. Read it for
> history; build from this.
>
> **Relationship to `paper_plan_v2_1.md`:** v2.1 remains the full ICLR paper plan.
> FOUNDATION-2 moves *back toward* it — the redesign is largely a matter of
> restoring two things FOUNDATION-1 simplified away (§4). Where they conflict,
> this document governs foundation work only.

---

## 0. Why this document exists (plain language)

FOUNDATION-1 asked: *can RL with per-step economic rewards teach an agent to stop
at the right time?* It ran end to end, passed its pre-registered gate, and then a
follow-up ablation showed the gate had passed **for the wrong reason** — the agent
got better at answering, not at stopping. A λ sweep from 0 to 1.0 moved stopping
by 0.04 steps.

The obvious conclusion was "the method doesn't work." The diagnostics say
something more useful: **the experiment could not have detected the effect even if
the method worked perfectly.** Three independent design errors each guaranteed a
null (§2). None of them is a property of the method.

FOUNDATION-2 fixes all three and re-asks the question on a design that can
actually answer it.

---

## 1. What FOUNDATION-1 established (keep these; they are real)

Run to completion 2026-07-22 → 2026-07-31. Full records:
`foundation/experiments/reports/`.

**Gate verdict: GO**, on the frozen dev-200 at B=4, harness off:

| arm | B=4 utility | B=4 F1 | B=4 steps | self-stop |
|---|---|---|---|---|
| A0 no budget info | .121 | .415 | 3.93 | 78% |
| A1 prompted | .205 | .471 | 3.54 | 78% |
| A2 enforced | .180 | .411 | 3.09 | 76% |
| **A3 RL-trained** | **.289** | **.560** | 3.61 | 78% |

Paired deltas at B=4 exclude zero (A3−A1 utility +.084, CI +.025…+.147).

**Findings that survive and carry into the paper:**

1. **Prompting beats enforcement.** A1 > A2 at every budget; at B=2 enforcement is
   catastrophic (F1 .221 vs .478). Telling a capable model a constraint beats
   mechanically cutting it off.
2. **The GO came from answer quality, not stopping.** A3's steps did not fall
   (3.61 vs A1's 3.54). Reported honestly in `foundation_report.md` §4 after the
   control (A1 vs A2, same policy) withdrew an earlier internalization claim.
3. **A correctly-calibrated cost term can exert no behavioural pull.** λ was
   calibrated so the utility optimum sat exactly at the observed quality knee —
   and moved nothing. *Where the optimum sits is not how hard the policy is pulled
   toward it.*
4. **A cost term works where the budget binds.** At B=2 only: −0.70 steps, CI
   excludes zero, F1 intact.
5. **The pipeline and its honesty machinery work.** Pre-registration, calibration
   gates, a frozen dev set with a look budget, a reward-hacking protocol that made
   a falsifiable prediction and then refuted it against its own data.

**Also established (infrastructure, all reusable):** GRPO trainer, ReAct harness
with three modes, retrieval env, judge client, eval/gate machinery, ~52 CPU tests.
FOUNDATION-2 changes what we *measure and reward*, not what we *run on* (§14).

---

## 2. The diagnosis — three design errors, each alone sufficient to force a null

### Error 1 — we targeted the wrong behaviour

We priced *"one step too many at the end."* The diagnostic (D1,
`pre_redesign_diagnostics.md`) measured how much of that exists on HotpotQA:
**0.31 steps mean, 0.00 median.** Our pre-registered threshold was 0.5 steps.
**The test required an effect larger than the effect that can physically exist.**

But D1 only looked at episodes that *succeeded*. Re-reading the per-step draft
curves (§3) shows the real cost sink is a different behaviour entirely:
**episodes that never get anywhere and keep going.**

### Error 2 — a scalar price cannot express a state-dependent rule

`λ·steps/B` applies identical pressure in every state. The optimal policy is
violently state-dependent: *keep going while progress is happening, quit fast when
it is not.* **No value of λ can encode that** — which is precisely what a 0→1.0
sweep found, and it is a stronger statement than the "difficulty confound" framing
in `ablation_report.md` §4.

Diagnostics ruled out the alternatives: GRPO is not cancelling the term (it is
81.7% of the advantage signal at λ=1.0), and stopping earlier *is* richly rewarded
(U(stop@2) = +0.131 vs U(stop@4) = −0.442). The signal was present, strong,
correctly signed — and structurally incapable of expressing the target behaviour.

**This is the argument for a value function instead of a price**, i.e. for the
mechanism v2.1 §2.2 specifies and FOUNDATION-1 simplified away.

### Error 3 — the objective was scaled so that success was invisible

At λ=0.3, B=4 one step costs 0.075 utility while a lost answer costs ~0.8 F1.
Consequence, measured (§3): **even a perfect oracle quit rule is worth +0.012
utility.** Our economy said "essentially never quit," and the policy complied.

Compounding it: budgets {2, 4, 8} against a natural stopping point of ~3.5 meant
**two of three budget conditions could not bind.** The only binding one (B=2) is
the only place the cost term worked.

---

## 3. The corrected empirical picture (measured 2026-07-31, existing rollouts)

Source: `experiments/results/pilot/pilot.jsonl` — 195 episodes with a self-chosen
ANSWER, arm A1 (frozen, untrained), `forced_continuation` to T_max = 10, so the
full per-step quality curve exists. `draft_f1_vs_gold` is recorded at every step
(`collect/schema.py:12`) — **this data was always there and we never read it.**

### 3.1 Half of all spend buys nothing

| | |
|---|---|
| episodes ending with F1 = 0 | **83 / 195 = 43%** |
| steps those episodes spent | **5.16 each** |
| **share of all steps that bought zero quality** | **52.7%** |

### 3.2 Hopelessness is visible in advance

Given the draft is still worthless after *k* steps:

| still worthless after | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 |
|---|---|---|---|---|---|---|---|
| P(eventually succeeds) | .574 | .511 | **.315** | **.241** | .213 | .171 | .160 |
| steps it will still spend | 3.16 | 2.98 | 3.58 | 3.16 | 2.64 | 2.11 | 1.56 |

By step 3 an unproductive episode has a 32% chance of ever working and burns
another 3.6 steps chasing it. **This is a decaying hazard, not noise** — it is the
signal an optimal-stopping rule consumes.

### 3.3 The headroom, valued under our own economy

Oracle quit rule "give up if no progress by step *k*", scored at λ=0.3, B=4:

| rule | steps | F1 | U | Δsteps | ΔF1 | ΔU |
|---|---|---|---|---|---|---|
| actual policy | 4.16 | .379 | .067 | — | — | — |
| quit if no progress by 3 | 2.83 | .287 | .075 | −1.34 | −.092 | +.009 |
| **quit if no progress by 4** | 3.23 | .321 | .079 | **−0.94** | −.058 | **+.012** |
| quit if no progress by 5 | 3.53 | .340 | .076 | −0.64 | −.039 | +.009 |

**−0.94 steps** — 3× D1's headroom and ~2× the threshold the ablation failed.

### 3.4 What is honest about this, and what is not

- **The step saving is real and large enough to detect. The utility gain is not**
  (+0.012, far under noise). Under the *current* economy the oracle rule is
  roughly Pareto-neutral: it trades −0.94 steps for −0.058 F1. That is Error 3
  restated, and it is why FOUNDATION-2 must fix the economy (§8) and report a
  frontier rather than one scalar (§9).
- **A global threshold rule is the crudest possible policy.** −0.94 steps is a
  *floor* on what a learned, per-episode rule could achieve, not a ceiling.
- **The rule as measured uses gold** (detecting "no progress" needs
  `draft_f1_vs_gold`). That is legitimate for *training labels* — it is exactly
  the privileged-information asymmetry v2.1 §2.3 argues for — but a deployed
  policy needs a gold-free proxy. **Whether that proxy exists is unproven and is
  the first thing FOUNDATION-2 tests (G0, §11).**
- **n = 195, one untrained arm, one dataset.** Must be replicated on the 2400
  λ=0 training rollouts (`experiments/results/train/`, on disk) before anything is
  pre-registered.

### 3.5 Amendment to `pre_redesign_diagnostics.md`

That report concluded *"HotpotQA cannot support a cost-aware-stopping claim."*
**Too broad.** Corrected: HotpotQA cannot support a **stop-when-done** claim
(0.23–0.31 steps of headroom, median 0). It **can** support a
**quit-when-hopeless** claim (~0.94 steps). D1 itself stands; the dataset verdict
does not.

---

## 4. What FOUNDATION-1 actually tested

FOUNDATION-1 replaced the trained stopping-value model with a frozen prompted
judge (simplification #4 in `paper_plan_v2_1_foundation.md` §1). In the full plan
that configuration already has a name: **baseline B10(b)** — *"executor GRPO
trained with RM-P rubric rewards in place of RM-T's V̂."*

And v2.1 §5.2 pre-registered a prediction for it:

> *Predicted outcome: RM-P loses — LLM step-redundancy judgment is ≤24.9% F1
> (RedundancyBench) and frozen judges get hacked under RL; binary rubric bits
> also cannot supply the continuous per-step differences r_t needs.*

**We ran the plan's own predicted-to-fail baseline, and it failed as predicted.**
That is a confirmed prediction, not a refuted method. The mechanism in v2.1 §2.2
— the Snell envelope / continuation value — **has never been run.**

This matters for framing: FOUNDATION-2 is not a rescue attempt after a negative
result. It is the experiment FOUNDATION-1 was supposed to de-risk, now run with
the component whose absence explains the null.

---

## 5. The redesigned question

> **Can an agent be trained to abandon unproductive work — to quit when the
> expected value of continuing falls below its cost — and does a learned
> continuation value do this where a scalar per-step price provably cannot?**

Two claims, one of which is a control on the other:

- **H1 (main).** A policy trained against a state-dependent continuation value
  spends materially fewer steps at matched answer quality than the same policy
  trained against a scalar per-step price.
- **H2 (mechanism).** The saving concentrates on episodes that would have failed —
  i.e. it comes from *abandonment*, not from truncating successful work. Falsified
  if the step reduction is uniform across succeeded/failed episodes.

H2 is what makes H1 interesting and is measurable per-episode.

---

## 6. Method — the continuation-value reward

### 6.1 The quantity

For each state `x_t` in a rollout, with the running-draft quality curve `q_t`
already recorded:

```
U_t   = q_t − cost(≤t)                      realized utility if we stop at t
V_T   = U_T
V_t   = max( U_t , Ê[ V_{t+1} | x_t ] )     Snell backward recursion
Cont(x_t) = Ê[ V_{t+1} | x_t ]              continuation value
Δ*_t  = Cont(x_t) − U_t                     stop margin; > 0 ⇒ continue
τ*    = min{ t : U_t ≥ Cont(x_t) }          optimal non-anticipating stop
```

`Ê[·|x_t]` is fitted cross-sectionally across the batch (Longstaff–Schwartz /
fitted value iteration), **not** taken as the max of one realized path. This is
the whole point: `argmax_t U_t` over a single trajectory is foresight-biased and
provably upper-bounds every implementable rule (Krengel–Sucheston; v2.1 §2.2).
Anything reported as achievable headroom must come from the Snell recursion, never
from a path maximum.

### 6.2 Why this fixes Error 2

`Δ*_t` compares continuing against stopping **from the same state**. Question
difficulty enters both terms identically and cancels by construction — the exact
confound `ablation_report.md` §4 quantified (between-question SD 1.220 vs
within-question 0.666) and could not remove with any λ.

### 6.3 Why the judge leaves the critical path

The per-step quality signal we need is **exactly computable from gold at training
time** — `q_t = draft_f1_vs_gold`, already logged. FOUNDATION-1 applied the
plan's own rule ("anything computable exactly is computed exactly, never judged")
only to the terminal reward. Applying it per-step:

- removes our single largest failure risk (the frozen judge — which cost a full
  recalibration cycle when the judge changed, and whose stop-decision bit was the
  only one to fail calibration at .775);
- removes the calibration gate from the critical path entirely;
- removes ~24k judge calls per round of GPU-served inference.

**The judge is not deleted — it is demoted to where v2.1 §5.2 always had it:**
the RM-P baseline (B10), run as a comparison arm rather than as the reward source.
That converts a liability into a paper table.

### 6.4 The counterfactual data already exists

`forced_continuation` mode (`agent/harness.py:104`) logs the agent's chosen
`answered_at` while continuing the episode to T_max — i.e. it records *what would
have happened had the agent not stopped*. That is precisely the input the Snell
recursion needs, and 195 such episodes are already on disk.

---

## 7. The arms

| Arm | Name | What it is | What it answers |
|---|---|---|---|
| **C0** | Plain ReAct | No budget information anywhere | The no-signal floor (carried from A0) |
| **C1** | Prompted budget | Budget tracker injected, nothing enforced | Is telling it enough? (the real bar — it beat enforcement in FOUNDATION-1) |
| **C2** | **Scalar-price RL** | GRPO on `R = F1 − λ·steps/B` — **FOUNDATION-1's A3** | The control that isolates the redesign. Without it, any C3 gain is confounded with "more training" |
| **C3** | **Continuation-value RL** (the method) | GRPO on the Snell-derived per-step signal (§6) | Does a state-dependent value do what a price cannot? |
| **C4** | Prompted-judge RL (**RM-P / B10**) | FOUNDATION-1's reward source, as a baseline | Trained-vs-prompted reward model, answered in our own table |

**C2 is non-negotiable.** FOUNDATION-1's headline error was attributing a gain to
the wrong cause because the control was missing; the internalization claim had to
be withdrawn when A1-vs-A2 was finally run. C2 is that control, pre-committed.

Enforcement (old A2) drops to an appendix — FOUNDATION-1 answered it decisively
(prompting beats it everywhere) and it need not be re-run at headline cost.

---

## 8. The economy — fixing Error 3

### 8.1 Cost becomes multi-dimensional

Cost = **tokens + tool calls + steps**, priced in real dollars via the harness's
own price map (`cost_aware_agent/cost.py`) — restoring v2.1 §2.1 and undoing
FOUNDATION-1 simplification #1.

Two reasons, one evidential and one structural:

- **Tokens carry unconfounded headroom.** At equal best quality the cheapest
  attempt uses **28% fewer characters** (D2) — same question, same outcome, pure
  verbosity. Steps and tool calls carry the difficulty confound; tokens do not.
- **Real prices make quitting worth it.** A long doomed episode costs
  quadratically in a growing context window, which a flat λ·steps/B cannot see.

**Reporting rule: always decompose.** A single dollar figure is the right *reward*
but the wrong *analysis unit* — merging one clean dimension with two confounded
ones hides the clean signal. Report $, tokens, tool calls and steps separately in
every table.

### 8.2 Budgets become policy-relative

Budgets are set as **percentiles of the policy's own unconstrained stopping
distribution** (50th / 75th / 100th), re-measured per dataset and per checkpoint,
instead of fixed integers.

Rationale: fixed {2,4,8} against a natural stop of ~3.5 left two of three
conditions non-binding, and the one binding condition is the only one where the
cost term worked (−0.70 steps, CI excludes zero). Policy-relative budgets make
binding a design invariant rather than an accident, and survive a dataset change
automatically.

### 8.3 λ is calibrated to the decision, not to the knee

FOUNDATION-1 set λ so the utility optimum sat at the observed quality knee. That
is a statement about *where* the optimum is, and §1.3 above is the lesson: it says
nothing about how hard the policy is pulled there. FOUNDATION-2 instead calibrates
λ so that **the oracle quit rule is worth a pre-specified, detectable amount of
utility** (target: ≥ 0.05, versus the +0.012 it was worth in FOUNDATION-1). λ is
frozen before training and reported with its derivation.

---

## 9. Metrics and the estimand

**The headline estimand changes from "mean steps" to two quantities:**

1. **Wasted spend** — cost incurred after `τ*` (the Snell-optimal stop), per
   episode. Paired within-question, so difficulty cancels.
2. **Cost at iso-F1** — read off a frontier swept over each method's own cost
   knob (v2.1 §5.3 protocol), not a single scalar U at one λ.

Mean steps is demoted to a descriptive statistic. It is substantially a
*consequence* of the question drawn (between-question SD 1.220) and should never
again be a headline.

**Full metric set:**

- Wasted spend vs τ*; **stopping regret** (utility gap, not |t − τ*|)
- Cost-at-iso-F1 and F1-at-iso-cost; Pareto AUC; all costs decomposed per §8.1
- **Abandonment rate** and **abandonment precision** — of episodes quit early, what
  fraction would have failed anyway? This is H2's test statistic
- F1 / EM with 95% bootstrap CIs (10k, paired per task)
- Distribution of stop steps, split by eventual success — the split is the point
- Internalization: % self-stopped, harness-off vs harness-on gap **with the
  same-policy control run alongside** (the FOUNDATION-1 lesson)
- Reward-hacking diagnostic: predicted-value vs realized-utility divergence
- Billing symmetry: every arm pays for all auxiliary inference it uses, including
  C4's judge calls and C3's value-fitting cost

---

## 10. Data

**Training stays on HotpotQA + adds MuSiQue.** Justification: §3 shows HotpotQA
retains ~0.94 steps of quit-headroom, so a dataset change is an *amplifier*, not a
prerequisite — and MuSiQue is 4-hop, has a real train split, and runs on the
existing retrieval index with zero new infrastructure. This also supplies v2.1's
E6 difficulty-stopping consistency check for free.

**GAIA and BrowseComp-Plus are transfer eval only.** GAIA has ~165 tasks and no
train split; it cannot be a training set, and v2.1 §5.1 already assigns web
research to transfer-eval-only for exactly this reason.

**Frozen dev discipline carries over unchanged**, with one addition: the
FOUNDATION-1 dev-200 has **1 of 3 looks remaining**, and FOUNDATION-2 starts a
**fresh look ledger** on its own dev set. Refinement happens on a val slice; every
dev evaluation is logged in `PROGRESS.md` with date and reason.

---

## 11. Stage sequence

Each stage has a gate. Stages G0–G2 are **CPU-only, on data already on disk**, and
together cost roughly a day — they decide whether the expensive stages are worth
running at all.

| Stage | What happens | Gate to pass |
|---|---|---|
| **G0** | **Gold-free predictability check.** Fit a classifier on the 2400 existing λ=0 rollouts: gold-free features (retrieval scores, draft churn/stability, repeated queries, entity coverage, step index) → eventual success. No GPU. | **AUC ≥ 0.65 held-out.** Below that, hopelessness is only visible in hindsight, H1 is unreachable on this data, and the dataset change becomes mandatory rather than optional — **this gate can kill the redesign, cheaply, in a day** |
| **G1** | **Headroom audit.** Replicate §3 at scale on the 2400 rollouts + MuSiQue pilot. Snell recursion offline; report τ*, wasted spend, and the achievable frontier | Measured headroom reported **before** any threshold is written down (§12) |
| **G2** | **Economy calibration.** Set λ and the policy-relative budgets so the oracle quit rule is worth ≥ 0.05 utility; freeze and commit | λ, budgets, and their derivation committed before any training |
| **G3** | **Reward implementation.** Snell/Longstaff–Schwartz label pipeline + per-step exact quality; CPU dry-run through the full reward path | `make test` green; dry-run reproduces hand-computed τ* on a fixture |
| **G4** | **Pre-registration.** Hypotheses, estimands, thresholds (derived from G1), decision rule, and the analysis script — committed before data | Committed, with the analysis script that will read the results |
| **G5** | **Training.** C2 and C3 trained identically except for the reward; C4 from FOUNDATION-1's existing checkpoints | Post-round temp-1.0 health probe passes (malformed < 10%) |
| **G6** | **Evaluation.** All arms on the frozen dev set × policy-relative budgets, harness off and on, frontier swept | All numbers as CSVs with a generation script |
| **G7** | **Analysis + verdict.** Figures, plain-language report, GO/NO-GO logged with date | The pre-registered rule applied by the G4 script, verbatim |

---

## 12. The pre-registered gate

**The thresholds are deliberately left blank here.** They are set at G4 **from the
headroom measured at G1**, and this ordering is itself the pre-registration.

FOUNDATION-1's threshold (0.5 steps) was chosen as "~14% of a 3.6-step baseline" —
a reasonable-sounding number that turned out to exceed the achievable maximum of
0.31. **A threshold picked before the ceiling is measured is a coin flip.**

Binding rules, fixed now:

1. **Detection threshold must be below the measured achievable headroom** — with
   the derivation shown, and above the CI half-width at the planned *n* (a
   threshold under the noise floor is equally useless: FOUNDATION-1's n=50 gave a
   half-width of 0.220 steps).
2. **Power is checked before running**, not reported after.
3. **The primary comparison is C3 vs C2**, not C3 vs prompting. The scalar-price
   control is what isolates the contribution.
4. **H2 is tested and reported whatever it says.** If the saving does not
   concentrate on failed episodes, the mechanism story is wrong even if H1 passes,
   and the report says so.
5. **The analysis script is committed before the data exists** and run unmodified.

**Scope of the verdict, stated before running:** a GO establishes that a learned
continuation value beats a scalar price *for abandonment behaviour, in this
setup*. It does not establish the full v2.1 potential-based shaping result (that
still needs the trained RM-T), and the report must say so.

---

## 13. Risks and kill conditions

| Risk | Evidence it is real | Response |
|---|---|---|
| **Hopelessness is not predictable gold-free** | Untested — the single biggest unknown | **G0 kills the redesign in one CPU-day** if AUC < 0.65 |
| **The saving is Pareto-neutral** | Measured: oracle quit trades −0.94 steps for −0.058 F1 | §8.3 re-calibrates λ; §9 reports a frontier. If the frontier shows no Pareto improvement at any λ, that is a genuine negative result and is publishable as one |
| **Abandonment is just answering early with a bad draft** | Plausible; would make H1 pass trivially | H2 + abandonment precision are pre-registered as required reporting, not optional |
| **Snell labels are noisy at our batch size** | Fitted value iteration compounds error | Per-step backup residuals on held-out trajectories (v2.1 §5.3); G3 fixture test |
| **Aggressive pricing degrades the policy** | Measured: λ=1.0 breached the health gate at 11% malformed | Health probe stays a hard gate between rounds; λ chosen at G2, not escalated ad hoc |
| **We attribute a gain to the wrong cause again** | Happened once (withdrawn internalization claim) | C2 control + same-policy harness control, both pre-committed |
| **Machine wipe** | Happened once, 79G lost | Unchanged policy: commit continuously to NatBrian; regenerables on `/mnt/src/liangsheng/` |

---

## 14. What carries over unchanged

FOUNDATION-2 changes the *reward and the measurement*, not the *machinery*. Reused
as-is:

| Component | Path | Change |
|---|---|---|
| ReAct episode loop, 3 harness modes | `agent/harness.py` | none — `forced_continuation` is now load-bearing (§6.4) |
| Per-step draft quality | `collect/schema.py` `draft_f1_vs_gold` | none — becomes the primary reward input |
| Retrieval env + E5/FAISS index | `envs/`, `scripts/serve_retrieval.py` | none |
| GRPO trainer, advantages, health probe | `train/` | reward source swapped; algorithm untouched |
| Eval, bootstrap CIs, paired deltas, gate-as-code | `eval/` | new estimands added |
| Judge client + rubric | `reward/` | demoted from reward source to the C4 baseline |
| Config single-source-of-truth | `configs/foundation.yaml` | extended, never bypassed |
| CPU test suite | `tests/` | extended; must stay green |

**New code required:** the Snell/Longstaff–Schwartz label pipeline, the gold-free
feature extractor (G0), the multi-dimensional cost accounting, and the frontier
sweep. Everything else is configuration.

---

## 15. Changelog — FOUNDATION-1 → FOUNDATION-2

| # | FOUNDATION-1 | FOUNDATION-2 | Forced by |
|---|---|---|---|
| 1 | Target: stop-when-done | **Target: quit-when-hopeless** | §3.1 — 52.7% of steps buy zero quality; stop-when-done headroom is 0.23–0.31 steps, median 0 |
| 2 | Reward: scalar `λ·steps/B` | **Reward: Snell continuation value** | §2 Error 2 — a price cannot express a state-dependent rule; λ sweep 0→1.0 moved 0.04 steps |
| 3 | Per-step signal: prompted judge | **Per-step signal: exact `draft_f1_vs_gold`**; judge → C4 baseline | §6.3 — computable exactly; also removes the top failure risk |
| 4 | Cost = steps | **Cost = tokens + tool calls + steps, in dollars** | D2 — 28% verbosity headroom, unconfounded |
| 5 | Budgets fixed {2,4,8} | **Budgets = policy-stop-distribution percentiles** | §2 Error 3 — 2 of 3 conditions never bound; the binding one is the only one that worked |
| 6 | Headline metric: mean steps, single U | **Wasted spend vs τ*, cost-at-iso-F1 frontier** | Mean steps is dominated by question difficulty (SD 1.220 vs 0.666) |
| 7 | λ calibrated to the quality knee | **λ calibrated so the oracle rule is worth ≥ 0.05 U** | §3.3 — oracle rule was worth +0.012, under the noise floor |
| 8 | Thresholds from intuition | **Thresholds from measured headroom (G1 before G4)** | Required 0.5 steps where 0.31 existed |
| 9 | No same-method control arm | **C2 scalar-price control, pre-committed** | An internalization claim had to be withdrawn for exactly this reason |
| 10 | HotpotQA only | **HotpotQA + MuSiQue**; GAIA/BrowseComp transfer-eval only | §10 — HotpotQA retains quit-headroom, so this is an amplifier; GAIA has no train split |

**Not changed, deliberately:** GRPO, the 9B executor (D4: it already responds to
budget signals by 1.02 steps — capability is not the binding constraint), the
retrieval env, the frozen-dev discipline, and the pre-registration culture.

---

## 16. Open items for review

1. **G0's AUC ≥ 0.65 threshold** is my judgement call. Too strict kills a viable
   redesign; too loose lets us spend weeks on an unlearnable signal. Confirm or
   move it.
2. **Dropping enforcement (old A2) to an appendix** — saves a full arm's compute.
   Confirm you are happy not re-running it at headline cost.
3. **MuSiQue as the second training set** vs going straight to a long-horizon
   deep-research benchmark. My recommendation is MuSiQue (zero new
   infrastructure); the counter-argument is that reviewers may want the harder
   setting in the headline rather than in transfer.
4. **Dollar-denominated cost now vs steps-plus-tokens now.** Full dollars restores
   v2.1 fidelity but reintroduces the price map and its normalization pilot, which
   FOUNDATION-1 simplified away for good reasons.
5. **Whether C4 (prompted-judge RL) is worth re-running** or whether
   FOUNDATION-1's existing numbers can be cited directly. Re-running is cleaner;
   citing is free.
