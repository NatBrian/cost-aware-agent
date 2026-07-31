# S3 — PRE-REGISTRATION for FOUNDATION-2 Step 1 — 2026-07-31

**Committed before the S4 training data exists.** The analysis script
(`scripts/s3_analyse.py`) is committed in the same commit and is run unmodified
against the S5 results. Every threshold below is derived from S2 measurements,
which were themselves taken before this document was written.

Git hash of this commit is the reference point: any later edit to this file or to
the analysis script is a protocol deviation and must be logged as one.

---

## 1. The question

Under a **binding** budget and a **correctly-scaled** economy, does a per-step
cost price teach the policy to spend less on work that returns nothing — without
costing answer quality?

FOUNDATION-1 answered "no" at B=4 with n=50. S2 showed that test was ~15×
underpowered for the effect available there, and that the budget did not bind.
Step 1 re-asks the question where it can actually be answered.

## 2. Arms

| arm | train λ | everything else |
|---|---|---|
| **control** | 0.0 | identical |
| **treatment** | **0.568** | identical |

Both are trained **from the same base checkpoint, on the same train-300, with the
same seed, the same 3 rounds, the same budgets {2,3,4}, and the same judge**.
**Only λ differs.**

**The λ=0 checkpoint from FOUNDATION-1 is NOT reused as the control**, even though
it exists. It was trained under budgets {2,4,8}; reusing it would confound the λ
change with the budget change and make any result unattributable — which is the
specific error that forced a claim to be withdrawn in FOUNDATION-1.

## 3. Evaluation protocol

- **Set:** `data/hotpotqa_eval_600.jsonl` (n=600), frozen, disjoint from
  train-300 / dev-200 / val-50 by id and normalized question, sha256 `42b8de8d…`
- **Budgets:** {2, 3, 4}. **Gate budget B=2** — the only one that binds (64.8% of
  episodes would overspend it, vs 25.5% and 17.2%)
- **Decoding:** temperature 0, G=1
- **Harness:** OFF (`mode=none`). Harness-on reported alongside, not gated on
- **Pairing:** per task, both arms on the identical 600 questions
- **CIs:** 10,000-resample paired bootstrap, seed 42

## 4. Primary hypothesis H1 — and the exact decision rule

> **H1.** At B=2, the treatment arm uses materially fewer steps than the control
> at no material cost to answer quality.

**PASS iff all three hold:**

1. `mean(Δsteps) ≤ −0.119` where `Δ = treatment − control`, paired per task
2. the 95% bootstrap CI for `Δsteps` lies **entirely below zero**
3. `mean(ΔF1) ≥ −0.02` — the quality guard

**Where 0.119 comes from:** S2 measured the achievable (gold-free, implementable)
effect at B=2 as **0.238 steps**; the pre-registered rule is *threshold = 50% of
achievable*. S2 also measured the paired within-task SD as 1.328, giving a CI
half-width of 0.119 at n=600 — so the threshold sits exactly at the resolution
limit, which is the tightest honest value. **n=600 was chosen to make this
threshold detectable (S2 required n ≥ 479); it was not chosen after seeing data.**

**Why the F1 guard is not optional:** the step saving can be bought by quitting
everything. Rule 3 makes that a FAIL, not a pass.

## 5. Secondary hypothesis H2 — the mechanism

> **H2.** The saving concentrates on episodes that were going to fail —
> abandonment, not truncation of successful work.

**Test.** Partition the 600 eval tasks by the **control arm's** outcome (a fixed
partition, not chosen by the treatment): `failed` = control F1 = 0, `succeeded` =
control F1 > 0. Compute `Δsteps` within each partition.

**H2 supported iff** `mean(Δsteps | control failed) < mean(Δsteps | control
succeeded)`, i.e. the saving is larger on doomed work.

**H2 is reported whatever it says.** If H1 passes and H2 fails, the economic claim
survives but the mechanism story does not, and the report must say so plainly. A
uniform reduction across both partitions means the policy simply got hastier.

## 6. Pre-registered dose-response prediction

Because the effect should live where the budget binds:

> `|Δsteps|` is largest at **B=2**, smaller at B=3, smallest at B=4.

Recorded now so it cannot be claimed post hoc. It is **supporting evidence, not
part of the gate** — B=3 and B=4 are underpowered by S2's own numbers (n≈751
needed at B=4), so a null there is uninformative and must not be reported as
evidence of absence.

## 7. Reported regardless of the verdict

- `W = E[steps × 1(F1=0)]` — the economic reading of the saving. **Not gated on:**
  S2 showed it needs n≈2289 at B=2. It is descriptive here, with its CI, and any
  claim about it must state that it is underpowered
- abandonment rate; self-stop rate; hit-cap rate
- F1, EM, mean steps, utility at the fixed reporting λ = 0.3
- token counts (newly recorded) and the realised cost-per-step curve
- judge-score vs realized-F1 divergence (reward-hacking diagnostic)
- malformed-output rate at temperature 1.0 per round (policy health)

## 8. Stopping and deviation rules, fixed in advance

- **Rounds:** exactly 3 per arm, as in FOUNDATION-1.
- **Health gate:** after **every** round, the temp-1.0 probe must show malformed
  < 10%. λ=0.568 is untested territory between a known-safe 0.3 and a
  known-harmful 1.0 (λ=1.0 breached this gate at 11%). **If a round breaches, that
  arm stops at its last healthy checkpoint and the deviation is reported in the
  results table** — as FOUNDATION-1 did for λ=1.0. Checkpoints are backed up
  *before* the probe runs, never after.
- **No λ retuning after seeing eval results.** λ* = 0.568 is frozen by S2.
- **One look.** eval-600 is read once, by the committed script. A second look for
  any reason is logged in `PROGRESS.md` with its justification.
- **No estimand substitution.** If H1 fails, we do not go looking for a metric
  that passes. The pre-registered secondary metrics are exactly those in §7.

## 9. What each outcome means, decided now

| outcome | reading | next |
|---|---|---|
| **H1 pass, H2 pass** | Cost-aware abandonment is learnable with a simple price, once the budget binds and the measurement is right | Step 3: scale up (MuSiQue, SimpleQA control, token cost, frontier) |
| **H1 pass, H2 fail** | The saving is real but is not abandonment — the policy got hastier | Report honestly; investigate before any mechanism claim |
| **H1 fail, CI excludes 0 but \|Δ\| < 0.119** | Real but smaller than the achievable estimate | Not a pass. Report the effect size; consider Step 2 |
| **H1 fail, CI contains 0** | Unresolved at n=600, or absent | **Step 2 is triggered** (Snell continuation value): the economy and measurement were fixed and a price still could not do it |
| **F1 guard breached** | Savings bought with quality | FAIL regardless of Δsteps |

## 10. Known limitations, stated before the result

- **The 0.238 achievable estimate is optimistic.** It came from a post-hoc filter
  over existing trajectories with k and θ chosen by scanning at their best. A
  trained policy may capture more (it can reshape the whole trajectory) or less
  (RL is noisy). The threshold is set at half of it partly for this reason.
- **B=3 and B=4 are underpowered** and cannot support claims either way.
- **eval-600 is 100% `hard` level** (HotpotQA dev is hard-only by design) while
  train-300 is mixed. This difficulty shift applies equally to both arms.
- **One seed.** FOUNDATION-2 Step 1 inherits FOUNDATION-1's single-seed
  simplification. Any headline claim in the paper needs the 3-seed protocol.
- **A pass establishes the effect on HotpotQA at a 3.3-step horizon**, where the
  absolute saving is a fraction of a step. It does not establish that the effect
  scales to long-horizon tasks; that is Step 3.
