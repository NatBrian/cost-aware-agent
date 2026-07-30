# λ ablation — does the step-cost term change stopping? — 2026-07-31

**Pre-registered verdict: NOT EFFECTIVE.** Rule and thresholds fixed in
`ablation_preregistration.md`, committed before any ablation run.

Three arms, identical in every respect except the λ in the training reward:
λ = 0 (steps free), 0.3 (the original run), 1.0 (steps ~4× unprofitable).
Evaluated on **val-50** at budgets {2, 4, 8}, temperature 0, harness off, every
arm scored on one fixed yardstick (`U = F1 − 0.3·steps/B`). **dev-200 was not
touched** — one of its three permitted looks remains.

## 1. The pre-registered rule

> EFFECTIVE iff mean_steps(λ=0) − mean_steps(λ=1.0) ≥ 0.5 at B=4 **and** their
> 95% bootstrap CIs (10k, paired per task) do not overlap.

| quantity | value |
|---|---|
| mean steps λ=0.0 | 3.500 (CI 3.000–4.080) |
| mean steps λ=1.0 | 3.460 (CI 3.020–4.000) |
| **paired Δ** | **+0.040** (CI −0.140 … +0.300) |
| cond 1 — Δ ≥ 0.5 | **FAIL** |
| cond 2 — CIs disjoint | **FAIL** |

**At the gate budget, pricing a step from free to 4× unprofitable moves mean
steps by 0.04.** Power: with n=50 the Δ CI half-width is 0.220 steps, so effects
below ~0.22 are unresolvable here — but the observed 0.04 sits far below even
that, and the CI comfortably contains zero.

## 2. Full results (val-50, temp 0, harness off)

| arm | train λ | B | steps | F1 | self-stop | U |
|---|---|---|---|---|---|---|
| lam0 | 0.0 | 2 | 3.280 | .606 | .340 | .114 |
| lam0 | 0.0 | 4 | 3.500 | .693 | .780 | **.431** |
| lam0 | 0.0 | 8 | 3.840 | .713 | .900 | **.569** |
| lam03 | 0.3 | 2 | 2.820 | .697 | .380 | **.274** |
| lam03 | 0.3 | 4 | 3.500 | .687 | .820 | .424 |
| lam03 | 0.3 | 8 | 4.060 | .697 | .900 | .545 |
| lam10 | 1.0 | 2 | 2.580 | .597 | .420 | .210 |
| lam10 | 1.0 | 4 | 3.460 | .660 | .780 | .400 |
| lam10 | 1.0 | 8 | 3.620 | .676 | .960 | .541 |
| lam10_r2 † | 1.0 | 4 | 3.340 | .612 | .800 | .361 |

† sensitivity arm: λ=1.0 at round 2, its last **healthy** checkpoint. Its round-3
probe FAILED (malformed 11.0% vs the 10% gate). Both are reported because
measuring λ=1.0 only at round 2, while the other arms are measured at round 3,
would be an asymmetric comparison — and choosing the checkpoint that flatters a
conclusion is exactly the freedom pre-registration removes. The rule reads
round 3 (protocol-matched). The sensitivity arm is *worse* on every metric, so
the conclusion does not depend on the choice.

**Adding the cost term does not improve the objective it was added to optimise.**
At B=4 utility runs .431 (λ=0) → .424 (λ=0.3) → .400 (λ=1.0); at B=8, .569 →
.545 → .541. The CIs overlap heavily so this is not a significant *harm*, but it
is emphatically not a benefit.

## 3. The one place the cost term did work — and it was not pre-registered

Paired per-task deltas against the λ=0 control (\* = CI excludes zero):

| budget | arm | Δ steps | Δ F1 |
|---|---|---|---|
| **B=2** | λ=1.0 | **−0.700 \*** (CI −1.26…−0.24) | −0.008 (ns) |
| B=2 | λ=0.3 | −0.460 (ns) | +0.092 (ns) |
| B=4 | λ=1.0 | −0.040 (ns) | −0.034 (ns) |
| B=8 | λ=1.0 | −0.220 (ns) | −0.037 (ns) |

**At the tight budget the cost term produces a real, significant step reduction
of 0.70 steps with answer quality intact.** This is exploratory, not
pre-registered — the rule was specified at B=4 — so it is weaker evidence and
must be labelled as such in any write-up. But it is the only significant
behavioural effect in the whole sweep, and it points at the mechanism.

**Why B=2 and nowhere else: the budget has to actually bind.** With the harness
off, the λ=0 policy takes 3.28 steps at B=2 — it *overspends* a budget of 2. There
the cost term has something to pull against. At B=4 the policy stops at 3.5 of 4
and at B=8 at 3.8 of 8: it is already inside budget, so there is no pressure for
any λ to relieve. The determinant is not the size of λ but **whether the policy's
natural stopping point exceeds the budget.**

## 4. Why raising λ cannot fix this — the mechanism

Three diagnostics on already-collected rollouts, each of which eliminates a
candidate explanation:

**(a) GRPO is not cancelling the cost term.** Group-relative advantages remove
anything constant within a group, so a natural suspicion is that the step penalty
cancels. It does not: within-group SD of steps is ~0.7, making the cost term
**81.7%** of the advantage signal at λ=1.0 (24.5% at λ=0.3). *The signal is
present and strong.*

**(b) Stopping earlier is richly rewarded.** On λ=1.0's own B=4 rollouts:
U(stop@2) = +0.131, U(stop@3) = −0.211, U(stop@4) = −0.442. The utility-optimal
stop is step 2; the policy stops at 3.29. *A +0.34 utility gain sits unclaimed
after 450 updates.*

**(c) Stop step is confounded with question difficulty.** F1 is *highest* among
episodes stopping at step 2 (.631, against .539 at step 3) — early stops are not
quality sacrifices, they are the **easy questions**. Decomposing stop-step
variance over 109 groups of 8 rollouts on the same question:

| source | SD |
|---|---|
| within-question (what the **policy** varies) | 0.666 |
| between-question (what the **question** dictates) | 1.220 |

**Stopping earlier on a hard question means answering it wrong, and difficulty is
unknowable before searching.** Mean steps is therefore only partly a decision
variable; it is substantially a *consequence* of the question drawn. A price on
raw step count cannot move a quantity the policy only partly controls.

## 5. Aggressive pricing degrades the policy

Malformed-output rate at temperature 1.0, by round:

| arm | r1 | r2 | r3 |
|---|---|---|---|
| λ=0.0 | 2.2% | 3.5% | 5.8% |
| λ=0.3 | 4.3% | 2.0% | 9.0% |
| **λ=1.0** | 5.8% | 2.4% | **11.0% — probe FAILED** |

λ=1.0 is the only arm to breach the health gate, and its F1 at B=4 is the lowest
of the three (.660 vs .693). Turning the price up does not buy better stopping;
past a point it buys worse output.

## 6. Decision: λ=1.5 is NOT run

The pre-registration named a "real but weak" case — Δ positive, >0.15, under
threshold — as warranting a fourth point. **Δ is +0.040, far below that flag, so
the condition was not met.** Three further reasons:

1. λ=1.0 already breached the health gate; λ=1.5 would most likely degrade the
   policy further, testing pricing pressure's destructive limit rather than the
   hypothesis.
2. The B=2 result shows the determinant is **whether the budget binds**, not the
   size of λ. A fourth λ at a non-binding budget tests the wrong variable.
3. We already have the informative dose-response — across budgets, not across λ.

Cost avoided: ~10 GPU-hours. *(Earlier in this run I pre-justified skipping λ=1.5
from a mechanism argument alone, and a later round undercut it. This time the
pre-registered numeric condition decides, and the mechanism is corroboration.)*

## 7. What the paper can and cannot claim

**Cannot claim:** that per-step economic rewards teach cost-aware stopping. At the
gate budget, a 0→1.0 sweep of the cost coefficient moves stopping by 0.04 steps,
and the pre-registered rule fails on both conditions.

**Can claim, all evidenced here:**

1. **Prompting beats enforcement.** A1 > A2 at every budget, and at B=2
   enforcement is catastrophic (F1 .221 vs .478). Telling a capable model a
   constraint beats mechanically cutting it off.
2. **A correctly-calibrated cost term can exert no behavioural pull.** λ was
   calibrated so the optimum sat exactly at the observed quality knee — and moved
   nothing. *Where the optimum sits is not how hard the model is pulled toward
   it.* A concrete, non-obvious warning for reward design.
3. **The step-budget formulation confounds "when to stop" with "how hard the
   question was."** Quantified: between-question SD 1.220 vs within-question
   0.666. This is the structural reason (2) happens, and it generalises to any
   agent benchmark that prices raw step counts.
4. **A cost term does work where the budget binds** (B=2: −0.70 steps, CI
   excludes zero, F1 intact) — exploratory, but it locates the regime where such
   rewards are worth using.
5. **A reproducible pipeline and a reward-hacking protocol** that made a
   falsifiable prediction from calibration (judge over-approves stopping →
   expect judge score to rise while F1 stays flat) and then refuted it against
   three rounds of data.

## 8. Recommendation for v2.1

**Price something the policy controls.** Raw step count is confounded with
difficulty. Two designs that are not:

- **Steps relative to the minimum needed for *that* question** — estimate the
  oracle minimum from the group's own rollouts (the cheapest sibling that got the
  answer right), and price the excess. This removes the difficulty component by
  construction.
- **A stop-decision reward against an oracle continuation** — at each candidate
  stop, ask whether continuing actually improved the answer, and reward the
  decision rather than the count. The forced-continuation replay already
  collects exactly this counterfactual.

**Or choose binding budgets.** If step count must be priced directly, set B below
the policy's natural stopping point, which is the only regime where it moved.

## 9. Plain language, for Brian

We asked whether making each search step expensive would teach the agent to stop
sooner. **It does not** — at the standard budget, going from "steps are free" to
"steps cost four times what they're worth" changed the number of steps by 0.04.

The reason turned out to be interesting. It is not that the penalty was too small
or that the algorithm cancelled it (we checked both — the penalty is 82% of the
training signal). It is that **when the agent stops is mostly decided by the
question, not by the agent.** Easy questions finish in two steps, hard ones take
four. Stopping early on a hard question just means getting it wrong. So there is
no free "stop sooner" behaviour for a price to buy — and the agent can't tell in
advance which kind of question it has.

Two things did show up. When the budget is genuinely tight (B=2, where the agent
would naturally overspend), the price **does** work — 0.7 fewer steps with no
quality loss. And pushing the price too high **damages** the model: the λ=1.0 arm
failed its health check with 11% malformed output.

For the paper: the "we taught economic stopping" claim is not supported. But
"pricing raw step count fails, and here is exactly why, with the variance
decomposition to prove it" is a genuine contribution — it will save other people
this experiment. The pipeline works, and the two follow-up designs in §8 are
cheap because everything is already built.
