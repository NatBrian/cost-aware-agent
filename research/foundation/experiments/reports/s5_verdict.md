# FOUNDATION-2 Step 1 — **H1 PASS, H2 SUPPORTED** — 2026-08-02

Decided by `scripts/s3_analyse.py`, committed before the training data existed and
run unmodified. Rule and thresholds: `s3_preregistration.md`. Frozen eval-600,
temperature 0, harness off, paired per task, 10k bootstrap.

## 1. The gate

> **H1 PASSES iff** mean(Δsteps) ≤ −0.119 **and** the 95% CI lies entirely below
> zero **and** mean(ΔF1) ≥ −0.02.  (Δ = treatment − control, at B=2)

| | control (λ=0) | treatment (λ=0.568) | Δ | 95% CI |
|---|---|---|---|---|
| steps | 2.977 | **2.810** | **−0.167** | [−0.280, −0.057] |
| F1 | 0.433 | **0.513** | **+0.080** | [+0.053, +0.108] |

| condition | requirement | result |
|---|---|---|
| 1 · effect size | Δsteps ≤ −0.119 | **PASS** (−0.167) |
| 2 · significance | 95% CI entirely below 0 | **PASS** (upper −0.057) |
| 3 · quality guard | ΔF1 ≥ −0.02 | **PASS** (+0.080) |

### **H1 VERDICT: PASS**

The quality guard did not merely hold — **F1 rose**, and its CI excludes zero. The
treatment is better on *both* axes: it spends less and answers better. This is a
Pareto improvement, not a trade.

## 2. H2 — the mechanism

> The saving must concentrate on episodes that were going to fail. Partition by
> the **control's** outcome, a fixed split the treatment cannot influence.

| partition | n | Δsteps | 95% CI |
|---|---|---|---|
| control **FAILED** (doomed work) | 179 | **−0.486** | [−0.832, −0.151] |
| control **SUCCEEDED** | 421 | −0.031 | [−0.093, +0.038] |

**H2 SUPPORTED.** The saving is **16× larger on doomed work**, and only the doomed
partition's CI excludes zero. The policy is not uniformly hastier — it abandons
work that was going nowhere and leaves successful work essentially untouched.

That is precisely the behaviour FOUNDATION-2 was redesigned to target, and it is
the behaviour the S1 predictability check said was learnable.

## 3. Robustness: the round-mismatch check

The control failed its round-3 health gate (20.5% malformed) and stopped at round
2; the treatment passed all three. So the protocol comparison comes from
differently-trained checkpoints, and a Δ could reflect *amount of training* rather
than λ. The treatment was therefore re-evaluated at round 2 as well.

| comparison | B=2 Δsteps | B=2 ΔF1 |
|---|---|---|
| protocol-matched, r3 vs r2 (**decides**) | −0.167 [−0.280, −0.057] | +0.080 |
| round-matched, r2 vs r2 (robustness) | −0.162 [−0.290, −0.035] | +0.068 |

**The two agree in sign and in magnitude (−0.167 vs −0.162).** The effect is not
an artefact of the treatment having had an extra round of training.

## 4. Full results

| arm | B | steps | F1 | W | fail% | self-stop | U |
|---|---|---|---|---|---|---|---|
| control | 2 | 2.977 | .433 | 1.092 | 29.8 | 32.8 | −.014 |
| control | 3 | 3.350 | .427 | 1.335 | 30.0 | 69.7 | .092 |
| control | 4 | 3.530 | .423 | 1.365 | 28.7 | 78.0 | .158 |
| **treatment** | 2 | **2.810** | **.513** | 1.073 | 33.3 | 40.0 | **.091** |
| **treatment** | 3 | **3.047** | **.531** | 1.148 | 31.7 | 76.0 | **.227** |
| **treatment** | 4 | **3.343** | **.526** | 1.330 | 32.0 | 82.3 | **.275** |

The effect holds at every budget, all three CIs excluding zero:

| B | Δsteps | 95% CI | ΔF1 |
|---|---|---|---|
| 2 (gate) | −0.167 | [−0.280, −0.057] | +0.080 |
| 3 | **−0.303** | [−0.420, −0.190] | +0.104 |
| 4 | −0.187 | [−0.288, −0.085] | +0.103 |

## 5. What did NOT go as predicted — stated plainly

**The dose-response prediction failed.** S3 §6 pre-registered that |Δsteps| would
be *largest at B=2*, because that is the only budget that binds. It is largest at
**B=3** (−0.303 vs −0.167). The prediction was recorded precisely so it could not
be quietly dropped, and it was wrong.

This does not affect the gate — the dose-response was explicitly supporting
evidence only — but it does undercut the "binding budget is the determinant"
story inherited from FOUNDATION-1's B=2 result. The honest reading now: the effect
is present at **all three** budgets and does not track the binding fraction
(64.8% / 25.5% / 17.2%). Whatever λ is doing here, it is not only relieving
budget pressure. That deserves investigation before the claim is repeated.

**W is null and stays null**: ΔW = −0.018, CI [−0.168, +0.132]. Exactly as S2
predicted (n≈2289 needed, we have 600). It is reported, not claimed. Switching the
primary estimand from W to Δsteps at S2 — before any of this data existed — is
what made the experiment answerable.

**The treatment fails slightly more often** (33.3% vs 29.8% at B=2). Abandonment
has a cost: some episodes that would have succeeded were given up on. Mean F1 is
still substantially higher, so the trade is favourable, but it is a real cost and
not a free lunch.

## 6. What this establishes, and what it does not

**Establishes, on HotpotQA at a ~3-step horizon:**

1. A per-step cost price **does** teach cost-aware abandonment — 0.167 fewer steps
   at B=2, CI excluding zero, against a threshold set from measured headroom
   *before* the data existed.
2. It is **abandonment, not haste**: −0.486 steps on doomed work vs −0.031 on
   successful work.
3. It costs **nothing in quality** — F1 rose by 0.080 (CI excludes zero).
4. The result survives the round-matched control.

**Does not establish:**

- That the effect scales to long-horizon tasks. The absolute saving is a fraction
  of a step on a 3-step task; whether it grows with horizon is Step 3's question.
- Any mechanism claim about *why* λ improves F1. That was not predicted and is not
  explained here.
- Anything about the trained stopping-value model (Step 2), which remains unrun.
- Generality across seeds: **one seed**, per the foundation's simplification.

## 7. How this differs from FOUNDATION-1, and why

FOUNDATION-1 asked the same underlying question and got a null. Four changes
account for the difference, each made because a measurement demanded it:

| | FOUNDATION-1 | Step 1 |
|---|---|---|
| gate budget | B=4 (slack for 67%) | **B=2** (binds for 64.8%) |
| n | 50 | **600** (S2 power: ≥479) |
| estimand | mean steps / single-λ utility | **paired Δsteps at iso-F1** |
| threshold | 0.5 steps (ceiling was 0.31) | **0.119** (50% of measured 0.238) |
| λ | set so the optimum sat at the knee | **set so the incentive was worth ≥0.05** |

The λ ablation's "NOT EFFECTIVE" was measured at B=4 with n=50, where S2 showed
n≈751 was needed — **roughly 15× underpowered**. The method was not the problem;
the measurement was.

## 8. Policy health — a finding in its own right

| round | ctrl (λ=0) | trt (λ=0.568) |
|---|---|---|
| 1 | 3.6% | 1.5% |
| 2 | 6.7% | 6.5% |
| 3 | **20.5% — FAIL** | **3.1% — PASS** |

**The λ=0 control degraded and failed; the priced arm did not.** FOUNDATION-1's
experience was the opposite (λ=1.0 was the arm that broke at 11%), which is what
motivated the 0.6 cap. λ=0.568 appears to sit in a regime where cost pressure
*stabilises* training rather than damaging it. Untested and unexplained — but it
is why the arms ended at different rounds, and it is worth a dedicated experiment.

## 9. Artifacts

`experiments/results/s5_eval/` — 9 JSONLs (3 arms × 3 budgets × 600), all
validated at 600 episodes / 600 unique tasks / 0 schema errors; identical task
sets across arms at every budget, so the paired bootstrap is well-founded ·
`s3_verdict.json` · `s5_matched.json` ·
figures `experiments/reports/figs/fig_s5a_dose_response.pdf`,
`fig_s5b_h2_split.pdf` · scripts `s3_analyse.py` (pre-registered, unmodified),
`s5_matched.py`, `analysis/s5_figures.py`.

**Dev-look ledger:** FOUNDATION-1's dev-200 was never touched. eval-600 has now
been read **once**.
