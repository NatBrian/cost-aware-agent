# Paper Plan v2.2 — FOUNDATION-2, the redesigned pipeline-validation run

> ## RESULTS ARE IN (2026-08-05) — read `foundation/experiments/reports/T5_SYNTHESIS.md`
>
> **Step 1 PASSED.** A scalar per-step price teaches abandonment of doomed work:
> −0.167 steps on HotpotQA (pre-registered gate), −0.228 on SimpleQA (out of
> distribution), −0.267 pooled on MuSiQue across two seeds. Every CI excludes zero.
> It scales with **failure rate**, not horizon.
>
> **Four things in this document are now WRONG and are corrected in place below:**
> §7.3/§12's dose-response rationale (falsified), §9's Step-2 trigger (not fired),
> §10's BrowseComp rationale (wrong reason), and the absence of a standing
> λ=0-control requirement. Sections are annotated rather than rewritten so the
> original reasoning stays auditable.
>
> **Do not claim from this plan that cost-aware training improves quality.** It
> does not; that effect is a separable confound (T1/T2/T4).
>
> **Status:** implementation-ready (2026-07-31, rewritten). Supersedes
> `paper_plan_v2_1_foundation.md` (FOUNDATION-1) as the active plan. FOUNDATION-1
> is **not deleted**: it ran to completion, produced a GO, and its results and
> failure modes are the entire input to this redesign.
>
> **Relationship to `paper_plan_v2_1.md`:** v2.1 remains the full ICLR paper plan.
> Where they conflict, this document governs foundation work only.
>
> **Rewrite note (2026-07-31).** The first draft of this document proposed ~12
> simultaneous changes. That was over-scoped and reproduced, at larger scale, the
> attribution error that already cost us a claim: change everything at once and no
> outcome is interpretable. This version is a **staircase** — one minimal
> experiment, then two conditional expansions, each triggered by a specific
> result. §5 explains why, §7 is the experiment we actually run first.

---

## 0. Why this document exists (plain language)

FOUNDATION-1 asked: *can RL with per-step economic rewards teach an agent to stop
at the right time?* It ran end to end, passed its pre-registered gate, and then a
follow-up ablation showed the gate had passed **for the wrong reason** — the agent
got better at answering, not at stopping. A λ sweep from 0 to 1.0 moved stopping
by 0.04 steps.

The obvious conclusion was "the method doesn't work." The diagnostics say
something more useful: **the experiment could not have detected the effect even if
the method worked perfectly.** Three design errors each guaranteed a null (§2).
None of them is a property of the method.

FOUNDATION-2 fixes them **in order of cost**, starting with the four cheapest
(§7), and only builds new machinery if the cheap fixes are not enough.

---

## 1. What FOUNDATION-1 established (keep these; they are real)

Run 2026-07-22 → 2026-07-31. Records: `foundation/experiments/reports/`.

**Gate verdict: GO**, frozen dev-200 at B=4, harness off:

| arm | B=4 utility | B=4 F1 | B=4 steps | self-stop |
|---|---|---|---|---|
| A0 no budget info | .121 | .415 | 3.93 | 78% |
| A1 prompted | .205 | .471 | 3.54 | 78% |
| A2 enforced | .180 | .411 | 3.09 | 76% |
| **A3 RL-trained** | **.289** | **.560** | 3.61 | 78% |

Paired deltas at B=4 exclude zero (A3−A1 utility +.084, CI +.025…+.147).

**Findings that survive into the paper:**

1. **Prompting beats enforcement.** A1 > A2 at every budget; at B=2 enforcement is
   catastrophic (F1 .221 vs .478).
2. **The GO came from answer quality, not stopping.** A3's steps did not fall
   (3.61 vs A1's 3.54). Reported honestly in `foundation_report.md` §4 after the
   A1-vs-A2 control withdrew an earlier internalization claim.
3. **A correctly-calibrated cost term can exert no behavioural pull.** λ was
   calibrated so the utility optimum sat exactly at the observed quality knee —
   and moved nothing. *Where the optimum sits is not how hard the policy is pulled
   toward it.*
4. **A cost term works where the budget binds.** At B=2 only: −0.70 steps, CI
   excludes zero, F1 intact. **This is the positive result Step 1 is built on.**
5. **The pipeline and its honesty machinery work.** Pre-registration, calibration
   gates, a frozen dev set with a look budget, and a reward-hacking protocol that
   made a falsifiable prediction and then refuted it against its own data.

---

## 2. The diagnosis — three design errors

### Error 1 — we targeted the wrong behaviour

We priced *"one step too many at the end."* D1
(`pre_redesign_diagnostics.md`) measured how much of that exists on HotpotQA:
**0.31 steps mean, 0.00 median**, against a pre-registered threshold of 0.5.
**The test required an effect larger than the effect that can physically exist.**

D1 only looked at episodes that *succeeded*. The per-step draft curves (§3) show
the real cost sink is a different behaviour: **episodes that never get anywhere
and keep going.**

### Error 2 — the budget did not bind

A budget can only change behaviour if the policy would otherwise exceed it.
Measured against the policy's own stopping distribution (n=195):

| budget | % of episodes that would overspend it | verdict |
|---|---|---|
| B=2 | 75% | **binds** |
| B=3 | 41% | **binds** |
| B=4 | 33% | borderline |
| B=8 | **6%** | dead — irrelevant to 94% of episodes |

We ran `{2, 4, 8}` and **gated at B=4**. A third of the evaluation went to a
condition where the budget was a no-op, and the gate sat on the borderline. The
one clearly-binding budget, B=2, is **the only place the cost term worked.**

### Error 3 — the objective was scaled so success was invisible

At λ=0.3, B=4 one step costs 0.075 utility while a lost answer costs ~0.8 F1.
Consequence, measured (§3.3): **even a perfect oracle quit rule is worth +0.012
utility.** Our economy said "essentially never quit," and the policy complied.

### A fourth, secondary error — the cost function had the wrong shape

Every step re-reads the whole conversation, so cost per step grows:

| step | total chars charged | vs step 1 |
|---|---|---|
| 1 | 1,674 | 1.00× |
| 5 | 9,742 | 5.82× |
| 10 | 16,281 | **9.73×** |

Flat `λ·steps/B` charges step 10 the same as step 1, **underpricing late steps by
up to 10×** — exactly the steps we want the agent to avoid. *(Upper bound: assumes
no prefix caching. The direction holds; the magnitude is smaller with vLLM prefix
caching, and must be measured once token counts are recorded — §12.)* This is
**deferred to Step 3**; it is a refinement, not a blocker.

### What is NOT the explanation

Ruled out by diagnostics on collected rollouts:

- **GRPO is not cancelling the cost term** — it is 81.7% of the advantage signal
  at λ=1.0, 24.5% at λ=0.3.
- **Stopping earlier is not under-rewarded** — U(stop@2) = +0.131 vs
  U(stop@4) = −0.442 on λ=1.0's own rollouts.
- **The executor is not too weak** — merely *telling* the 9B its budget moves it
  1.02 steps (D4), 25× the λ effect.

---

## 3. The corrected empirical picture

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
another 3.6 steps chasing it. **A decaying hazard, not noise.**

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

- **The step saving is real and detectable. The utility gain is not** (+0.012).
  Under the *current* economy the oracle rule is roughly Pareto-neutral. That is
  Error 3 restated and it is what §7.4 fixes.
- **A global threshold rule is the crudest possible policy.** −0.94 is a *floor*
  on what a learned per-episode rule could achieve, not a ceiling.
- **The rule as measured uses gold.** Legitimate for training labels (the
  privileged-information asymmetry of v2.1 §2.3), but a deployed policy needs a
  gold-free proxy. **Whether that proxy exists is unproven and is stage S1.**
- **n = 195, one untrained arm, one dataset.** Replicated at scale in S2 before
  anything is pre-registered.

### 3.5 Amendment to `pre_redesign_diagnostics.md`

That report concluded *"HotpotQA cannot support a cost-aware-stopping claim."*
**Too broad.** Corrected: HotpotQA cannot support a **stop-when-done** claim
(0.23–0.31 steps, median 0). It **can** support a **quit-when-hopeless** claim
(~0.94 steps). D1 itself stands; the dataset verdict does not.

---

## 4. What FOUNDATION-1 actually tested

FOUNDATION-1 replaced the trained stopping-value model with a frozen prompted
judge (simplification #4 of the FOUNDATION-1 plan). In the full plan that
configuration already has a name: **baseline B10(b)** — *"executor GRPO trained
with RM-P rubric rewards in place of RM-T's V̂."* And v2.1 §5.2 pre-registered a
prediction for it:

> *Predicted outcome: RM-P loses — LLM step-redundancy judgment is ≤24.9% F1
> (RedundancyBench) and frozen judges get hacked under RL; binary rubric bits
> also cannot supply the continuous per-step differences r_t needs.*

**We ran the plan's own predicted-to-fail baseline, and it failed as predicted.**
That is a confirmed prediction, not a refuted method. The mechanism v2.1 §2.2
specifies — the Snell continuation value — **has never been run**, and is Step 2
here.

---

## 5. The design principle: minimal first

**A foundation run exists to de-risk cheaply.** Twelve simultaneous changes have a
specific failure mode: if the result is positive we cannot attribute it, and if it
is negative we cannot localise it. FOUNDATION-1 already paid that price once — the
internalization claim had to be withdrawn because a control was missing.

**The minimal path is not a guess.** The λ ablation is routinely read as
condemning the scalar price. Look at where it ran (§2, Error 2): the price
**worked** at the one budget that bound (B=2: −0.70 steps, CI excludes zero,
quality intact) and did nothing at the two that did not. The ablation is evidence
that *pricing fails when the budget does not bind* — plus positive evidence that
it works when it does. Our gate happened to sit at B=4, one of the dead
conditions.

So Step 1 changes the **economy and the measurement** while holding the **method**
fixed. Whatever happens is attributable.

**Both outcomes are useful.** If Step 1 works, the paper's claim gets *stronger* —
"cost-aware stopping is learnable with a simple price, provided the budget binds
and you measure the right behaviour" beats a result that needs novel machinery. If
it fails, Step 2 has earned its complexity, with a clean argument: we fixed the
economy and the measurement and a price still could not do it.

---

## 6. Research question and hypotheses

> **Can an agent be trained to abandon unproductive work — to quit when the
> expected value of continuing falls below its cost?**

- **H1 (primary).** Under a binding budget and a correctly-scaled economy, a
  cost-trained policy spends materially less on episodes that yield nothing, at
  no cost to answer quality.
- **H2 (mechanism).** The saving concentrates on episodes that would have failed —
  abandonment, not truncation of successful work. **Falsified if the reduction is
  uniform across succeeded and failed episodes.**

H2 is what makes H1 interesting, is measurable per episode, and is reported
whatever it says.

---

## 7. STEP 1 — the minimal experiment

### 7.1 Everything that changes

> **Updated 2026-07-31 after S1/S2 ran.** Rows 1, 3 and 5 differ from this
> document's first draft. Both changes were forced by measurement, and both are
> recorded rather than quietly edited — see `s2_headroom.md`.

| # | change | where |
|---|---|---|
| **1** | **Budgets `{2, 4, 8}` → `{2, 3, 4}`; gate budget → B=2.** ~~B=3~~ — the 41%-binding figure for B=3 came from the *pilot's untrained* policy. Against the policy that is actually trained (mean stop 3.33) the binding fractions are B=2: **64.8%**, B=3: 25.5%, B=4: 17.2%. **Only B=2 binds** | `configs/foundation.yaml` |
| **2** | **λ recalibrated** to the size of the incentive → **λ\* = 0.568** (cap 0.6) | `configs/foundation.yaml` |
| **3** | **Primary estimand → paired Δsteps at iso-F1.** ~~Wasted spend `W`~~ is **not runnable**: 72% exact zeros give it a paired SD of 1.64, needing n≈2289 at B=2 and n≈108k at B=4. It is retained as the *economic reading*, reported with an explicit underpowered warning | `eval/metrics.py`, `scripts/s3_analyse.py` |
| **4** | **Detection threshold derived from S2-measured headroom** → **0.119 steps** (50% of the measured achievable 0.238) | `s3_preregistration.md` |
| **5** | **New frozen eval set `eval-600`** — n ≥ 479 required by the power analysis, so dev-200 is less than half of what the test needs | `scripts/s2_build_eval.py` |

Plus, from S1: **retriever similarity scores and per-step token counts are now
recorded.** The scores were being discarded in two places and are among the
strongest gold-free quit signals.

Plus two schema additions that cost nothing now and prevent re-collection later
(§12): **per-step token counts** and **retriever similarity scores**.

### 7.2 Everything that deliberately does NOT change

Dataset (HotpotQA 300/200/50) · reward form (scalar `F1 − λ·steps/B + format`) ·
per-step reward source (**the prompted judge stays**) · cost definition (steps) ·
GRPO trainer · executor (Qwen3.5-9B) · retrieval env · ReAct loop and harness
modes · frozen-dev discipline · config-as-single-source-of-truth.

**Byte-identical to what already ran, except the four rows above.**

### 7.3 Budgets

> **FALSIFIED 2026-08-05.** The reasoning below — that the effect lives where the
> budget *binds*, inherited from FOUNDATION-1's lone B=2 result — did not survive.
> The pre-registered dose-response prediction (|Δsteps| largest at B=2) **failed**:
> the effect was largest at B=3 on HotpotQA and is significant at *all three*
> budgets on MuSiQue without tracking the binding fraction (64.8 / 25.5 / 17.2%).
>
> Binding budgets are still a reasonable default — B=2 is where the pre-registered
> gate was set and it passed — but **"the budget must bind" is not the mechanism**,
> and should not be used to justify budget selection in future work. The mechanism
> is how much doomed work exists (T4, H-fail).

New: **`{small: 2, medium: 3, large: 4}`**, gate at **medium = 3** (41% of
episodes would overspend it). Derivation in §2 Error 2; re-measured at S2 against
the *current* policy before freezing.

**Freeze policy:** budgets are measured once, at round 0, and **frozen for the
whole run.** Re-deriving them per checkpoint would move the budget as the policy
improves and make arms incomparable. Named here because it is an easy trap.

Within a GRPO group all G rollouts share the same B — unchanged, and correct.

### 7.4 λ calibration

Replace *"put the utility optimum at the quality knee"* with a target on the
**size of the incentive**:

```
choose λ such that   ΔU(oracle quit rule)  ≥  0.05
where                ΔU = λ·(Δsteps / B) − ΔF1_loss
```

With the pilot's Δsteps = 0.94 and ΔF1_loss = 0.058, this gives λ ≥ 0.345 at B=3
and λ ≥ 0.459 at B=4 — **provisionally λ ≈ 0.5**, recomputed at S2 from measured
values and frozen before training.

**Hard cap λ ≤ 0.6.** λ = 1.0 breached the policy-health gate (11% malformed vs
the 10% limit) and produced the lowest F1 of the three arms. We are not
re-discovering that.

### 7.5 The primary estimand — wasted spend

```
W  =  E[ steps_used × 1(final_F1 = 0) ]        over ALL episodes
```

"How much of my budget went to nothing." Current value ≈ **2.20 steps/episode**
(5.16 × 0.426).

Why this and not the alternatives:

- **Unconditional**, so no selection bias — conditioning on failure would let a
  policy look good by failing more.
- **Directly measures the 52.7%** that §3.1 identified.
- **Only two ways to improve it:** succeed more often (good) or abandon failures
  faster (the target behaviour).

**Mandatory guard: F1 must not fall.** W can be gamed by quitting everything
immediately, so the gate pairs W with an F1 floor (§7.7). Reported at iso-F1.

**Secondary metrics:**

| metric | definition | purpose |
|---|---|---|
| abandonment rate | % of episodes self-stopping early with F1 = 0 | is it quitting at all? |
| **abandonment precision** | of episodes quit early, % that would have failed anyway (from `forced_continuation` replay) | **H2's test statistic** |
| stop-step distribution **split by eventual success** | — | the split *is* the finding |
| mean steps | — | **demoted to descriptive** (between-question SD 1.220 vs within 0.666) |
| judge-score vs realized-F1 divergence | unchanged | reward-hacking diagnostic |

### 7.6 Arms

| arm | what | source |
|---|---|---|
| **λ=0 control** | steps free, same everything else | **checkpoint exists** (`lam0_round3`) |
| **λ=λ\* treatment** | λ from §7.4 | one new training run |
| A1 prompted | reference | rows exist |
| A0 no-info | floor | rows exist |

**Primary comparison: treatment vs the λ=0 control**, paired per task. Both are
the same method, same data, same seeds, same rounds — only λ differs.

Enforcement (old A2) is **not re-run**; FOUNDATION-1 answered it decisively.

### 7.7 The pre-registered gate

Thresholds are **deliberately blank here** and are set at S3 from headroom
measured at S2. That ordering *is* the pre-registration. FOUNDATION-1's 0.5-step
threshold was picked as "~14% of a 3.6-step baseline" and turned out to exceed the
achievable maximum of 0.31.

Binding rules, fixed now:

1. **Threshold ≤ 50% of the S2-measured oracle ΔW**, and **> the CI half-width at
   the planned n** (FOUNDATION-1's n=50 gave a 0.220-step half-width — a threshold
   under the noise floor is as useless as one above the ceiling).
2. **Power checked before running**, not reported after.
3. **Primary comparison is treatment vs λ=0 control.**
4. **F1 guard:** F1 ≥ control − 0.02, paired. A W improvement bought with quality
   is not a pass.
5. **H2 reported whatever it says.** If the saving does not concentrate on failed
   episodes, the mechanism story is wrong even if H1 passes.
6. **The analysis script is committed before the data exists** and run unmodified.

---

## 8. Stage sequence

*(Status as of 2026-07-31. S1–S3 are done; their outcomes are recorded here
rather than left as intentions.)*

| stage | what happens | outcome / gate |
|---|---|---|
| **S1** | Gold-free predictability check — classifier on 2400 existing rollouts, features from `steps[:k]`, split by `task_id` | **PASS — held-out AUC 0.813** at k=3 (gate 0.65), replicated on three arms (.813 / .815 / .798). Strongest feature is the model's own confidence (`logprob_last`). `s1_predictability.md` |
| **S2** | Headroom audit + economy calibration | **DONE, and it changed the design:** gate budget → **B=2** (the only one that binds), estimand → **Δsteps** (W needs n≈2289), **λ\* = 0.568**, threshold **0.119**, **eval-600** built. `s2_headroom.md` |
| **S3** | Pre-registration + analysis script | **COMMITTED** before any S4 data existed. `s3_preregistration.md`, `scripts/s3_analyse.py` |
| **S0** | Re-score the FOUNDATION-1 λ arms at the new budgets | Diagnostic only, on val-50. Never touches dev |
| **S4** | **Both arms trained** — control λ=0, treatment λ=0.568, same base, data, seed, rounds, budgets | ~14h. Health probe after **every** round; a breach stops that arm at its last healthy checkpoint and is reported as a deviation, not hidden |
| **S5** | Evaluate both on eval-600 × {2,3,4}, harness off; apply `s3_analyse.py` verbatim | ~6h. The S3 rule decides |

**S1 and S2 cost about one CPU-day between them, on data already on disk, and S1
could have ended the redesign outright.** It did not, so Step 1 proceeds.

**Why the control is retrained rather than reused:** the FOUNDATION-1 λ=0
checkpoint exists, but it was trained under budgets `{2,4,8}`. Reusing it would
confound the λ change with the budget change — the exact error that forced a
claim to be withdrawn in FOUNDATION-1. It costs a full extra arm and is worth it.

---

## 9. STEP 2 — NOT TRIGGERED (2026-08-05)

> **Do not build this.** The trigger was "Step 1's gate fails **and** S1 cleared".
> S1 cleared (AUC 0.813) but **Step 1 passed**, so the condition never fired. A
> scalar price turned out to be sufficient, which makes the result simpler than
> the machinery designed for it. The section is kept for the reasoning and in case
> a future negative result revives it.

### 9-original. STEP 2 — conditional, only if Step 1 is null

**Trigger:** Step 1's gate fails *and* S1 cleared (AUC ≥ 0.65, so the signal
exists and the price simply cannot exploit it). That combination is a strong,
specific argument — *we fixed the economy and the measurement, and a scalar price
still could not do it* — and it is exactly the motivation the machinery needs.

| change | detail |
|---|---|
| **Reward form: price → value** | Snell backward recursion with cross-sectional regression: `V_t = max(U_t, Ê[V_{t+1}|x_t])`, `Δ*_t = Cont(x_t) − U_t`, `τ* = min{t : U_t ≥ Cont(x_t)}`. `Ê[·|x_t]` fitted across the batch (Longstaff–Schwartz), **never** a single path's max — that is the prophet bias v2.1 §2.2 warns about |
| **Why it fixes Error 2 structurally** | `Δ*_t` compares continuing against stopping **from the same state**, so difficulty cancels by construction — the confound quantified at between-question SD 1.220 vs within-question 0.666 |
| **Per-step signal: judge → exact** | `q_t = draft_f1_vs_gold`, already logged. Removes our most fragile component (it broke twice), removes the calibration gate from the critical path, removes ~24k judge calls per round, and frees a GPU |
| **Judge demoted to a baseline** | Where v2.1 §5.2 always had it (RM-P / B10) — converting a liability into a paper table |
| **Add the scalar-price control arm** | Step 1's treatment becomes Step 2's control, so the reward-form change is isolated |
| **Add stopping regret vs τ\*** | Utility gap, not `|t − τ*|` |

The counterfactual data this needs already exists: `forced_continuation`
(`agent/harness.py:104`) logs the agent's chosen `answered_at` while continuing to
T_max — precisely the Snell input, and 195 such episodes are on disk.

---

## 10. STEP 3 — the full paper (deferred, not abandoned)

> **CORRECTED 2026-08-05, then SOFTENED the same day (see `u2_mechanism_correction.md`).**
>
> An earlier version of this note said the effect tracks **failure rate, not
> horizon**, and that the next benchmark should be chosen for difficulty rather
> than length. **That claim is withdrawn.** It rested on failure rates measured in
> the *training rollouts*, where they peak at 3-hop; on the *evaluation set* —
> where the effect is actually measured — failure rate and horizon both rise
> monotonically (45.8/57.9/76.0% and 3.21/3.87/4.34 steps), so the two mechanisms
> make the same prediction and cannot be separated. The apparent 3-hop peak is
> also not significant (3-hop − 4-hop = −0.285, CI [−0.806, +0.210]).
>
> **What stands:** the effect grows on harder data (MuSiQue −0.267 vs HotpotQA
> −0.167). **What is undetermined:** why. **Practical guidance:** choose a
> benchmark where the agent *fails often* **and** episodes are long — both are
> higher where the effect was larger, and we cannot say which drives it.
> BrowseComp still qualifies on both counts.
>
> **Two additions to this step, both forced by results:**
>
> 1. **Any arm using λ needs a λ=0 control at matched training.** Without it,
>    generic training effects get silently credited to the method — which is what
>    happened to the quality gain until T1/T2/T4 separated it.
> 2. **MuSiQue is now the headline dataset**, not HotpotQA: larger effect
>    (−0.267 vs −0.167), no quality confound, selectivity intact, and it exposes
>    the reallocation behaviour that HotpotQA does not.
>
> **Already completed from this section:** SimpleQA (as the negative control —
> it did its job and separated two effects) and MuSiQue (as the headline).
> Remaining: token/dollar cost, frontier sweep, transfer evals, negative controls.

### 10-original — deferred items as originally written

| item | evidence it will matter | why deferred |
|---|---|---|
| **MuSiQue as primary training set** | **DiRe 68.8 for HotpotQA vs 37.8 for MuSiQue** — ~69 F1 of HotpotQA is reachable *without* multi-hop work, independently explaining our 0.31-step ceiling. Human–model gap 28.2 vs 9.6. 19,938 train / 2,417 dev, existing index | Changing the dataset and the economy at once makes Step 1 unattributable |
| **SimpleQA as low-slack negative control** | Single-hop by construction; the harness project's `real_cli` already found +2.8% (ns) on a low-slack set. **Pre-registered prediction: ~zero savings.** A positive result there means we are truncating necessary work | Needs a working positive result first to be a meaningful control |
| **BrowseComp-Plus transfer eval** | gpt-5/o3 issue **>20 search calls**; fixed local 100K-doc corpus (ACL 2026). **But** Qwen3-32B and SearchR1-32B issue **<2** despite prompting — a 9B may not engage the horizon, swapping a no-slack dataset for a no-engagement one. Also the one place D4 could reverse | Capability retest required first |
| **Cost = tokens + tool calls** | Step 10 costs 9.73× step 1 (§2); tokens carry **28% unconfounded headroom** (D2) at equal quality. Note tool calls ≡ steps in the current single-tool loop, so they are one dimension until a second tool exists | Needs token counts (§12) and a cached-cost measurement first |
| Dollar denomination | v2.1 §2.1 | Reintroduces the price map and normalization pilot FOUNDATION-1 dropped for good reasons; keep λ model-independent for now |
| Frontier sweep (cost-at-iso-F1) | v2.1 §5.3 | Many eval runs; needs a positive operating point to sweep around |
| Negative controls (random-value, shuffled-label coach) | v2.1 A9 | Only meaningful once a value function exists (Step 2) |
| 2WikiMultihopQA · FRAMES · GAIA | extra transfer evals | No train split, or human–model gap 3.7 (near-saturated) |

---

## 11. Data — Step 1

**Unchanged from FOUNDATION-1:** HotpotQA, frozen train-300 / dev-200, val-50
refinement slice, same retrieval index, same manifest discipline. Comparability
with the existing baseline rows is the point.

**Dev discipline:** FOUNDATION-1's dev-200 has **1 of 3 looks remaining**.
FOUNDATION-2 starts a **fresh look ledger** on its own dev set. Refinement happens
on val-50; every dev evaluation is logged in `PROGRESS.md` with date and reason.

The dataset roster and its evidence live in §10 and activate at Step 3.

---

## 12. Implementation manifest — Step 1

| file | change | kind |
|---|---|---|
| `configs/foundation.yaml` | `budgets: {small: 2, medium: 3, large: 4}`; `gate_budget: medium`; `economy.train_lambda` ← S2 value; new `gate:` thresholds ← S3 | edit |
| `envs/retrieval_client.py:31` | **stop discarding retriever scores** — currently keeps only `title`/`text`. Likely the strongest gold-free "am I finding anything?" feature, and S1 needs it | **bug-grade fix** |
| `agent/llm_client.py` | capture `usage` (prompt/completion tokens) from the vLLM response | edit |
| `agent/harness.py` | record per-step `prompt_tokens`, `completion_tokens`, and retrieval scores in the step dict | edit |
| `collect/schema.py` | add the new step fields; bump schema version; extend `validate_episode` | edit |
| `eval/metrics.py` | add `wasted_spend`, `abandonment_rate`, `abandonment_precision`; keep mean steps as descriptive | add |
| `eval/gate_check.py` | rewrite for the W estimand + F1 guard; thresholds from config | rewrite |
| `scripts/s0_rescore.sh` | re-score existing λ checkpoints at binding budgets | **new** |
| `scripts/s1_predictability.py` | gold-free feature extractor + classifier + held-out AUC | **new** |
| `scripts/s2_headroom.py` | scale replication of §3; oracle ΔW; λ and budget derivation | **new** |
| `analysis/figures.py` | figures for W and the success-split stop distribution | add |
| `tests/` | W and abandonment-precision unit tests on fixtures; schema round-trip with the new fields | add |

**No changes to:** `train/` (GRPO untouched), `reward/` (judge and rubric stay as
they are), `collect/sampling.py`, `scripts/f1_data.py`, the retrieval server.

---

## 13. Risks and kill conditions

| risk | evidence | response |
|---|---|---|
| **Hopelessness is not predictable gold-free** | untested — the biggest unknown | **S1 ends it in one CPU-day** if AUC < 0.65 |
| **The saving is Pareto-neutral** | measured: oracle quit trades −0.94 steps for −0.058 F1 | §7.4 re-scales λ so the trade is worth ≥0.05; the F1 guard (§7.7 rule 4) makes a quality-bought win a fail |
| **Abandonment is just answering early with a bad draft** | plausible; would make H1 pass trivially | H2 + abandonment precision are **required** reporting, not optional |
| **W is gamed by quitting everything** | structural | F1 floor in the gate; W reported at iso-F1 |
| **Aggressive pricing degrades the policy** | measured: λ=1.0 → 11% malformed, gate breached | λ ≤ 0.6 hard cap; health probe stays a hard gate between rounds |
| **We attribute a gain to the wrong cause again** | happened once | λ=0 control is the primary comparison; minimal change set keeps it attributable |
| **The judge breaks again** | happened twice | If calibration fails, **promote the exact-quality reward from Step 2 to Step 1** — the one pre-authorised scope change |
| **Machine wipe** | happened once, 79G lost | Unchanged: commit continuously to NatBrian; regenerables on `/mnt/src/liangsheng/` |

---

## 14. What carries over unchanged

| component | path | status |
|---|---|---|
| ReAct loop, 3 harness modes | `agent/harness.py` | unchanged; `forced_continuation` becomes load-bearing |
| Per-step draft quality | `collect/schema.py` `draft_f1_vs_gold` | unchanged; becomes the primary reward input at Step 2 |
| Retrieval env + E5/FAISS index | `envs/`, `scripts/serve_retrieval.py` | unchanged apart from the score fix |
| GRPO trainer, advantages, health probe | `train/` | **unchanged** |
| Judge client + rubric | `reward/` | **unchanged in Step 1**; demoted at Step 2 |
| Eval scoring path, bootstrap CIs, paired deltas | `eval/` | extended |
| Config single source of truth | `configs/foundation.yaml` | extended, never bypassed |
| CPU test suite | `tests/` | extended; must stay green |
| Executor Qwen3.5-9B | — | unchanged (D4: already responds by 1.02 steps) |

---

## 15. Changelog

### FOUNDATION-1 → Step 1 (what we run now)

| # | FOUNDATION-1 | Step 1 | forced by |
|---|---|---|---|
| 1 | Budgets `{2,4,8}`, gate B=4 | **`{2,3,4}`, gate B=3** | §2 Error 2 — B=8 was irrelevant to 94% of episodes; B=2 (the only binding one) is the only place the price worked |
| 2 | λ set so the optimum sits at the quality knee | **λ set so the oracle rule is worth ≥0.05 U** (λ ≈ 0.5, cap 0.6) | §3.3 — it was worth +0.012, under the noise floor |
| 3 | Headline: mean steps + single-λ utility | **Wasted spend W + abandonment precision** | §3.1 — 52.7% of steps buy zero quality; mean steps is dominated by question difficulty |
| 4 | Thresholds from intuition | **Thresholds from S2-measured headroom** | Required 0.5 steps where 0.31 existed |
| 5 | No same-method control | **λ=0 control is the primary comparison** | An internalization claim had to be withdrawn for exactly this reason |
| 6 | `raw_len` chars; retriever scores discarded | **Token counts + retriever scores recorded** | Cheap now, prevents re-collection; S1 needs the scores |

### Deferred, with triggers

| step | trigger |
|---|---|
| **Step 2** — Snell value reward, exact per-step quality, judge → baseline | Step 1 gate fails **and** S1 AUC ≥ 0.65 |
| **Step 3** — MuSiQue, SimpleQA control, BrowseComp transfer, token/dollar cost, frontier, negative controls | Step 1 or Step 2 passes |
| **Promoted early** — exact per-step quality replaces the judge in Step 1 | The judge fails calibration again |
| **Promoted early** — dataset change becomes mandatory | S1 AUC < 0.65 |

---

## 16. Open items for review

1. **S1's AUC ≥ 0.65 threshold** is my judgement call. Too strict kills a viable
   redesign; too loose spends a week on an unlearnable signal.
2. **The F1 guard at −0.02** (§7.7 rule 4) — tight enough that a quality-bought
   saving fails, loose enough to allow noise. Confirm the tolerance.
3. **λ cap at 0.6** — derived from λ=1.0 breaching the health gate, but 0.6 itself
   is untested. The alternative is to run a health probe at the chosen λ before
   committing to a full run.
4. **Whether S0 alone could be enough to decide.** If re-scoring the existing
   checkpoints at binding budgets already shows clean separation, we may be able
   to skip straight to writing rather than retraining — decide when S0 lands.
