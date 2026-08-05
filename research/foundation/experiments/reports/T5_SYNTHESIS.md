# FOUNDATION-2 — final synthesis — 2026-08-05

What the paper can claim, with what evidence, and what it must not claim.
Covers Step 1 (S0–S5) and the follow-up phase (T1–T4), plus T3's two completed
seeds. **Seed 789 is outstanding** — blocked by another user holding all 8 GPUs;
auto-resume is armed.

---

## 1. The headline claim

> **A scalar per-step cost price teaches a ReAct agent to abandon work that is
> going nowhere. The saving is selective, transfers out of distribution, and
> grows with how often the agent gets stuck.**

| dataset | Δsteps (treatment − control) | 95% CI |
|---|---|---|
| HotpotQA (pre-registered gate) | **−0.167** | [−0.280, −0.057] |
| SimpleQA (out of distribution) | **−0.228** | [−0.362, −0.098] |
| MuSiQue seed 42 (matched r1) | **−0.242** | [−0.432, −0.050] |
| MuSiQue seed 123 (matched r3) | **−0.292** | [−0.488, −0.095] |
| **MuSiQue pooled (2 seeds)** | **−0.267** | **[−0.404, −0.129]** |

Every CI excludes zero. Roughly **6–8% fewer steps**.

The HotpotQA number cleared a threshold (0.119) that was derived from measured
headroom **before** the data existed, by a script committed before training
started and run unmodified.

## 2. It is selective — and on harder data it *reallocates*

Partitioned by the **control's** outcome (a fixed split the treatment cannot
influence):

| dataset | on doomed work | on successful work |
|---|---|---|
| HotpotQA | −0.486 ✱ | −0.031 (n.s.) |
| SimpleQA | −0.420 ✱ | −0.013 (n.s.) |
| **MuSiQue (pooled)** | **−0.582 ✱** | **+0.114 ✱** |

On the two easier sets the policy cuts dead ends and leaves productive work
alone. **On MuSiQue it goes further: it spends the saved budget back on questions
it can actually answer** (+0.114, CI excluding zero). That is a stronger and more
interesting behaviour than pure abandonment — budget *reallocation* — and it only
became visible on the hard dataset with a fully-trained pair.

Stated carefully: reallocation is a **two-seed MuSiQue observation**, not
established across datasets. The abandonment half is three-for-three.

## 3. It scales with failure rate, not horizon

Pre-registered before the treatment arm was trained (`t4_preregistration.md`):

| | predicted between | predicted within | observed |
|---|---|---|---|
| **H-fail** | ≈−0.24 (1.5–2×) | peaks at 3-hop | **−0.242, peaks at 3-hop** ✓ |
| H-horizon | ≈−0.19 | monotone in hops | rejected — not monotone |

By hops: −0.148 / **−0.429 ✱** / −0.180, tracking failure rate (51.7 / 61.9 /
50.7) rather than step count (3.50 / 3.63 / 3.74).

**The per-doomed-episode saving is stable at −0.42 to −0.58 across all three
datasets.** That constancy is the mechanism: the overall effect scales with how
much doomed work a dataset contains, not with how long its tasks are.

## 4. What the paper must NOT claim

**Not that cost-aware training improves answer quality.** F1 rose +0.080 on
HotpotQA — but T1 showed the gain is statistically independent of the step saving
(it lives on episodes whose step count never changed), T2 showed it reproduces on
single-hop questions where no efficiency gain is possible, and T4 showed it
**vanishes** on MuSiQue (−0.005, CI spans zero). It is a confound that can appear,
not a benefit of the method.

**Not that λ stabilises training.** See §5.

**Not a three-seed result** until seed 789 lands. Two seeds agree tightly
(sd 0.035) and the pooled CI excludes zero, but the pre-registered bar was three.

---

## 5. Claims made and then withdrawn — the honesty log

Four claims were stated during this work and later corrected by evidence. All are
recorded because a reader deserves to know which conclusions survived scrutiny.

| # | claim | why withdrawn |
|---|---|---|
| 1 | "Cost-aware training improves answer quality" (S5) | T1: independent of the saving. T2: reproduces where cost-awareness cannot operate. T4: absent on MuSiQue |
| 2 | "λ is a general regulariser" (T2) | Too strong. T4 found no quality gain on MuSiQue. Either the effect belonged to those particular HotpotQA-trained policies, or one round is too little training — not separable with what we have |
| 3 | "λ stabilises training, replicated on two datasets" (T4) | **T3 killed it.** Seed 123's λ=0 control trained cleanly through all three rounds (9.2 → 5.3 → 4.4% malformed) and its treatment was marginally *worse* at r3. Two controls died, one didn't. It was training noise read as a pattern at n=2 |
| 4 | "\|Δsteps\| is largest at the binding budget B=2" (S3 §6, pre-registered) | Failed. Largest at B=3; the effect appears at all budgets without tracking the binding fraction |

Claims 1–3 were mine and were stated more confidently than the evidence
supported. Claim 4 was pre-registered precisely so it could fail visibly.

**The step-reduction result is untouched by all four.** It is measured on paired
evaluation data, not on health probes, and it survived every check aimed at it:
an out-of-distribution control, a mechanism test, a scale test, and (so far) two
seeds.

---

## 6. Why FOUNDATION-1 got the opposite answer

FOUNDATION-1 concluded per-step economic rewards do **not** teach stopping
(Δ = 0.04, "NOT EFFECTIVE"). Nothing about the method changed. Four measurement
decisions did:

| | FOUNDATION-1 | Step 1 |
|---|---|---|
| gate budget | B=4 — slack for 67% of episodes | **B=2** — binds for 64.8% |
| n | 50 | **600** (power analysis: ≥479) |
| estimand | mean steps, single-λ utility | **paired Δsteps at iso-F1** |
| threshold | 0.5 steps (ceiling was 0.31) | **0.119** — 50% of measured headroom |

The original test ran at a budget the agent never hit, with n=50 where n≈751 was
needed — **~15× underpowered**. Its verdict against its own rule stands; the
softer reading ("the effect is essentially zero") never did.

**The transferable lesson: measure the achievable ceiling and the noise floor
before choosing a threshold.** Both of FOUNDATION-1's failures — and one of ours,
where the `W` estimand needed n≈2289 — were caught by that single discipline.

---

## 7. Corrections to `paper_plan_v2_2_foundation.md`

1. **§10 scale-up rationale is wrong.** The plan justifies BrowseComp by *episode
   length*. H-fail says the effect tracks **failure rate**. The target may be
   right; the reason is not. Next benchmark should be chosen for difficulty.
2. **Step 2 (Snell continuation value) is not triggered and should not be built.**
   Its trigger was an H1 failure. H1 passed, so a scalar price suffices — the
   result is simpler than the machinery designed for it.
3. **The dose-response prediction (§7.3, §12) is falsified** and should not be
   repeated as a rationale for budget selection.
4. **Add a standing requirement:** any arm using λ needs a λ=0 control at
   *matched training*, or regularisation is silently credited to the method.

## 8. Open questions

- **Why does λ improve F1 at all** (where it does)? Not better retrieval — the
  treatment retrieves *fewer* distinct documents. Unexplained.
- **Is reallocation real or MuSiQue-specific?** +0.114 on successful work is one
  dataset, two seeds.
- **The 3-hop peak rests on one significant cell**; 2-hop and 4-hop CIs span zero.
- **Seed 789** — outstanding.

## 9. Standing limitations

- One executor (Qwen3.5-9B), one judge, one λ value, three datasets.
- Absolute effects are fractions of a step on 3–4 step tasks.
- MuSiQue F1 is low (~0.27) against a 2018 Wikipedia index; both arms face this
  equally, so the paired comparison is unaffected.
- Health probes ran on HotpotQA val-50 even for MuSiQue arms — valid as a
  format gate, but their F1 readings are out-of-domain.
- The MuSiQue seed-42 comparison is two round-1 policies; seed 123 is round-3.
  That they agree across training depths is reassuring, but neither is a
  fully-powered fully-trained result on its own.

## 10. Artifacts

Reports: `s1_predictability.md` · `s2_headroom.md` · `s3_preregistration.md` ·
`s0_rescore.md` · `s5_verdict.md` · `t1_f1_gain.md` · `t2_negative_control.md` ·
`t4_preregistration.md` · `t4_musique.md` · this file.

Results: `s5_eval/` (9) · `t2_simpleqa/` · `t4_musique/` (9) · `t3_seeds/`.
Figures: `figs/fig_s5a_dose_response.pdf`, `fig_s5b_h2_split.pdf`.

**Dev-look ledger:** FOUNDATION-1's dev-200 was never touched in this phase.
eval-600 read once. MuSiQue eval-600 read once per seed.
