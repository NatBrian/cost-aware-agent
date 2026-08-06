> # ⚠️ RETRACTED IN PART — 2026-08-06
>
> **Four independent audits found the headline here is substantially inflated and
> that three interpretive claims are artefacts. Read
> `CORRECTED_RESULTS.md` instead.** This document is kept unedited as a record of
> what was claimed and when.
>
> | claimed here | corrected |
> |---|---|
> | Δsteps −0.22 | **−0.03 to −0.11** on well-formed episodes; 37–90% of the original was format-collapse, not deliberation |
> | tokens −13% to −33% | **−10% to −13%** |
> | "selective — cuts dead ends, not good work" | **withdrawn.** 70–85% reproduced by a placebo with no treatment |
> | "unchanged / improved quality" | **withdrawn.** F1 rewards terseness; on a length-free metric quality is unchanged on 3 datasets and *worse* on HotpotQA |
> | "pre-registered gate PASS" | **fails** on the corrected estimand (−0.112 vs the −0.119 threshold) |
> | "grows on harder data" | **withdrawn.** p = 0.28–0.51 |
>
> Novelty is also pre-empted: arXiv 2510.01152 (MASH) and 2504.14870 (OTC-PO)
> anticipate the method, mechanism, benchmark and transfer protocol, at 2–12× the
> effect size.

# FOUNDATION-2 — final synthesis — 2026-08-05

What the paper can claim, with what evidence, and what it must not claim.
Covers Step 1 (S0–S5) and the follow-up phase (T1–T4, U1–U4).

**Outstanding:** seed 789 — blocked by another user holding all 8 GPUs;
auto-resume armed. Everything else is complete.

---

## 1. The claim

> **A scalar per-step cost price teaches a ReAct agent to abandon work that is
> going nowhere. The saving is selective, transfers to an unseen task
> distribution, and is ~3× larger in tokens than in steps.**

### In steps (the pre-registered, better-powered estimand)

| comparison | Δsteps | 95% CI | n |
|---|---|---|---|
| HotpotQA (pre-registered gate) | −0.167 | [−0.280, −0.057] | 600 |
| SimpleQA (**never trained on**) | −0.228 | [−0.362, −0.098] | 500 |
| MuSiQue seed 42 (matched r1) | −0.242 | [−0.435, −0.050] | 600 |
| MuSiQue seed 123 (matched r3) | −0.292 | [−0.490, −0.090] | 600 |
| **POOLED (corrected)** | **−0.220** | **[−0.295, −0.147]** | **1700** |

Every comparison excludes zero. Fig: `u4_fig3_forest.pdf`.

### In tokens (the unit that matters)

| dataset | rel. steps | **rel. tokens** | ratio |
|---|---|---|---|
| HotpotQA | −5.6% | **−13.4%** | 2.40 |
| SimpleQA | −7.6% | **−32.9%** | 4.34 |
| MuSiQue s42 | −6.8% | **−20.0%** | 2.97 |
| MuSiQue s123 | −8.0% | **−22.4%** | 2.79 |

**~91% of the saving is prompt tokens** — context no longer re-read. Every step
re-reads the whole conversation, so step 10 costs ~9.7× step 1; abandonment cuts
the expensive tail. The agent is not terser (completion tokens barely move), it
has fewer conversations to re-read.

**Honest caveat:** tokens are noisier — only 2 of 4 token CIs exclude zero, versus
4 of 4 on steps. Δsteps stays the primary result; Δtokens is the more meaningful
unit, reported with its wider intervals. Fig: `u4_fig2_steps_vs_tokens.pdf`.

## 2. It is selective, not hasty

Partitioned by the **control's** outcome — a fixed split the treatment cannot
influence:

| dataset | doomed work | successful work |
|---|---|---|
| HotpotQA | −0.486 ✱ | −0.031 |
| SimpleQA | −0.420 ✱ | −0.013 |
| MuSiQue s42 | −0.500 ✱ | +0.062 |
| MuSiQue s123 | −0.663 ✱ | +0.168 ✱ |

**The per-doomed-episode saving is stable at −0.42 to −0.66 across three
datasets**, while successful work is barely touched. If the price merely made the
policy hastier, both columns would fall together. They do not, anywhere.
Fig: `u4_fig1_selectivity.pdf`.

**Concrete counts** (U3): on MuSiQue, 44 episodes where the treatment quit and
nothing was lost against **3** where quitting cost a winnable answer — **14.7:1**.
On HotpotQA, 2:1. About **half of all episodes are byte-identical** between arms.

**What it looks like** — control burned all 10 steps and returned *no answer*;
treatment reached the same dead end in 3, guessed, and stopped. And the honest
downside: one case where the treatment quit one step before the lookup that would
have resolved the question, and the control got it right.

## 3. What the paper must NOT claim

1. **Not that it improves answer quality.** F1 rose +0.080 on HotpotQA, but the
   gain is statistically independent of the saving (T1), reproduces on single-hop
   questions where no efficiency gain is possible (T2), and vanishes on MuSiQue
   (T4). A confound to disclose, not a benefit.
2. **Not that λ stabilises training.** Refuted by seed 123 (§5).
3. **Not a mechanism for why the effect grows on harder data.** Withdrawn (§5).
4. **Not "reallocation."** The +0.114 steps on successful work is real in
   aggregate but corresponds to **zero** visible spend-more-and-win episodes
   (U3). Say: *spends marginally more on work the control also succeeded at*.
5. **Not a three-seed result** until seed 789 lands. Two seeds agree tightly
   (sd 0.035); the pre-registered bar was three and is reported as unmet.

---

## 4. Why FOUNDATION-1 concluded the opposite

It found Δ = 0.04, "NOT EFFECTIVE". Nothing about the method changed — four
measurement decisions did:

| | FOUNDATION-1 | Step 1 |
|---|---|---|
| gate budget | B=4 — slack for 67% of episodes | **B=2** — binds for 64.8% |
| n | 50 | **600** (power analysis: ≥479) |
| estimand | mean steps, single-λ utility | **paired Δsteps at iso-F1** |
| threshold | 0.5 steps (ceiling was 0.31) | **0.119** — half the measured headroom |

The original ran at a budget the agent never reached, with n=50 where n≈751 was
needed — **~15× underpowered**. Its verdict against its own rule stands; the
softer reading ("the effect is essentially zero") never did.

**The transferable lesson: measure the achievable ceiling and the noise floor
before choosing a threshold.** That single discipline caught both of
FOUNDATION-1's failures and one of ours (the `W` estimand needed n≈2289).

---

## 5. Six claims made and withdrawn — the honesty log

| # | claim | why withdrawn |
|---|---|---|
| 1 | "Cost-aware training improves quality" (S5) | T1: independent of the saving. T2: reproduces where cost-awareness cannot operate. T4: absent on MuSiQue |
| 2 | "λ is a general regulariser" (T2) | Too strong — no quality gain on MuSiQue |
| 3 | "λ stabilises training, replicated" (T4) | Seed 123's λ=0 control trained cleanly through all 3 rounds; its treatment was marginally *worse*. Two controls died, one didn't — noise read as a pattern at n=2 |
| 4 | "\|Δsteps\| is largest where the budget binds" (S3, pre-registered) | Failed. Largest at B=3; significant at all budgets without tracking the binding fraction |
| 5 | "The effect scales with failure rate, not horizon" (T4) | **Methodological error.** The premise was verified on training rollouts, where failure rate peaks at 3-hop; on the *eval set* both predictors rise monotonically, so the two hypotheses make the same prediction. The apparent peak is also not significant (3-hop − 4-hop CI [−0.806, +0.210]) |
| 6 | "Reallocation of budget" (U1/T5) | Zero clean spend-more-and-win episodes exist (U3). The aggregate is a diffuse sub-step shift, not a purposive behaviour |

Claims 1, 2, 3, 5 and 6 were mine and were stated more confidently than the
evidence bore. Claim 4 was pre-registered so it could fail visibly.

**A distinct failure mode worth naming (from #5):** pre-registration prevents
choosing a hypothesis *after* seeing results. It does **not** prevent validating a
test's premise on **different data from the measurement**. Check the premise where
you will measure.

**The step-reduction result is untouched by all six.** It is measured on paired
evaluation data, not health probes, and survived an out-of-distribution control, a
mechanism test, a scale test, a token re-analysis, trajectory inspection, and two
seeds.

---

## 6. Corrections to `paper_plan_v2_2_foundation.md`

1. **§7.3** — "the budget must bind" is falsified as a mechanism (still a fine
   default).
2. **§9 Step 2** — not triggered; the Snell machinery should not be built.
3. **§10** — BrowseComp is justified by the agent **failing often *and* episodes
   being long**; we cannot say which drives the effect.
4. **New standing requirement** — any arm using λ needs a λ=0 control at *matched
   training*, or generic training effects get credited to the method.

## 7. Open questions

- **Why λ improves F1 where it does.** Not better retrieval — the treatment
  fetches *fewer* distinct documents. Unexplained.
- **Whether failure rate or horizon drives the growth.** Needs an eval set where
  the two vary independently (e.g. fixed hop-count, varied distractor density).
- **Seed 789.**

## 8. Standing limitations

One executor (Qwen3.5-9B), one judge, one λ, three datasets, ≤2 seeds. Absolute
effects are fractions of a step on 3–4 step tasks. MuSiQue F1 is low (~0.27)
against a 2018 Wikipedia index — both arms equally, so the paired comparison is
unaffected. Health probes ran on HotpotQA val-50 even for MuSiQue arms: valid as a
format gate, but their F1 readings are out-of-domain. MuSiQue seed 42 is matched
at round 1 and seed 123 at round 3; that they agree across training depths is
reassuring but neither is a fully-powered fully-trained result alone.

## 9. Artifacts

**Reports** (24): `s0`–`s5`, `t1`, `t2`, `t4`, `u1`–`u3`, both pre-registrations,
this file.
**Figures**: `u4_fig1_selectivity.pdf`, `u4_fig2_steps_vs_tokens.pdf`,
`u4_fig3_forest.pdf`, `fig_s5a_dose_response.pdf`, `fig_s5b_h2_split.pdf`.
**Results**: `s5_eval/` · `t2_simpleqa/` · `t4_musique/` · `t3_seeds/`.

**Dev-look ledger:** FOUNDATION-1's dev-200 untouched in this phase; eval-600 read
once; MuSiQue eval-600 read once per seed.
