# Pre-registration — the λ ablation and dose-response

**Written 2026-07-29, BEFORE any ablation run. Committed before launch so the
prediction is timestamped ahead of the data.**

## Why this experiment exists

The foundation run passed its gate: the trained agent (A3) beat both prompting
(A1) and enforcement (A2) on utility. But the post-hoc analysis showed **the win
came entirely from answer quality, not from stopping**:

- steps at B=4: A1 3.54 → A3 3.61 (went *up*)
- utility gain +.084 = quality **+.089** + step-cost **−.005**
- stop-step distributions nearly identical; self-stop slightly *lower*

Since F1 is part of the reward, the result is equally consistent with "RL
fine-tuned the model on HotpotQA". **The gate could not distinguish the two
hypotheses.** This experiment is built to distinguish them.

**Suspected cause of the null:** at B=4 with λ=0.3 a step costs `0.3/4 = .075`
utility while the pilot showed a 4th step buys ≈ +.06 F1. Continuing is roughly
break-even, so almost no gradient pushed the policy to stop earlier. λ was chosen
to place the optimum at the pilot's knee — which is not the same as making the
pull toward it strong.

## Design

Three arms, **identical in every respect except the reward's λ**: same base
model, same 300 training tasks, same 3-round protocol, same lr / KL / accum /
update cap, same judge, same rubric, same health gates.

| arm | train λ | step price at B=4 | status |
|---|---|---|---|
| λ=0 | 0.0 | free | to run |
| λ=0.3 | 0.3 | .075 / step | **already trained** (foundation round 3) |
| λ=1.0 | 1.0 | .25 / step | to run |

`economy.train_lambda` moves; `economy.lambda` stays at 0.3 so every arm is
**scored on one fixed yardstick**. Raw metrics (mean steps, F1, self-stop) are
λ-independent and are the primary evidence.

**Evaluation: val-50 at budgets {2, 4, 8}, temperature 0, harness off.**
dev-200 is NOT touched — 1 of 3 permitted looks remains and it is reserved for a
final headline method, not a diagnostic.

## Hypotheses

**H1 (the cost term drives stopping).** Mean steps at B=4 falls monotonically as
λ rises:

```
steps(λ=0)  >  steps(λ=0.3)  >  steps(λ=1.0)
```

**H0 (the cost term is inert).** Steps are statistically indistinguishable across
λ, and any utility differences come from answer quality alone.

## Decision rule — fixed now, applied without amendment

**The cost term is declared EFFECTIVE iff both hold:**

1. `mean_steps(λ=0) − mean_steps(λ=1.0) ≥ 0.5` steps at B=4, **and**
2. their 95% bootstrap CIs (10,000 resamples, paired per task) do not overlap.

0.5 steps is ~14% of the ~3.6-step baseline: large enough to matter
behaviourally, small enough to be detectable at n=50.

**Secondary (supporting, not decisive):** self-stop rate rises with λ; the
steps-vs-λ curve is monotone across all three points.

**If the rule is not met**, the honest conclusion is that per-step economic
rewards at this scale do not change stopping behaviour, and the foundation's
result is task-skill RL. That is a publishable negative finding about reward
design, and it will be reported as such rather than reframed.

## Failure modes acknowledged in advance

- **λ=1.0 may break the policy** (steps are expensive enough that answering
  immediately could dominate). The health probe gates every round; a collapsed
  policy is reported as collapsed, not quietly dropped.
- **λ=0 may score higher utility on the fixed yardstick** simply by using more
  steps to get better answers. That is expected and is not evidence against H1;
  the decision rule reads steps, not utility.
- **n=50 is small.** A null result at this n is weak evidence of no effect; if
  the curve is directionally right but under-powered, that will be stated and the
  fix is more validation tasks, not a re-reading of the same data.

## What each outcome licenses for the paper

| outcome | claim the paper can make |
|---|---|
| rule met, curve monotone | Cost-aware training changes stopping behaviour, with dose-response — the paper's intended claim, now evidenced |
| rule met at λ=1.0 only | The mechanism works but needs a real price; λ=0.3 was mis-calibrated. Report the corrected operating point |
| rule not met | Per-step economic rewards did not produce economic behaviour at this scale. Report as a negative result; the foundation's GO stands as task-skill RL plus a proven pipeline |
