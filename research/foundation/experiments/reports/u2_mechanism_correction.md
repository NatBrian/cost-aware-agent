# U2 — the H-fail mechanism claim is WITHDRAWN — 2026-08-05

**T4 concluded "H-fail confirmed, H-horizon rejected." That conclusion rests on a
methodological error and does not survive.** The effect on MuSiQue is unchanged;
only the *explanation* for why it grows is retracted.

## The error

T4's pre-registration argued the two candidate mechanisms were distinguishable
because, within MuSiQue, **steps rose monotonically with hops while failure rate
did not** — giving H-fail and H-horizon different predictions.

Those failure rates came from the **training rollouts** (51.7 / 61.9 / 50.7%,
peaking at 3-hop). The effect is measured on the **evaluation set**. On the
evaluation set both predictors are monotonic:

| hops | n | fail % | steps | Δsteps | 95% CI |
|---|---|---|---|---|---|
| 2 | 622 | 45.8 | 3.21 | −0.140 | [−0.296, +0.014] |
| 3 | 378 | 57.9 | 3.87 | **−0.500** | [−0.765, −0.243] ✱ |
| 4 | 200 | **76.0** | **4.34** | −0.220 | [−0.670, +0.225] |

**Failure rate and horizon are not decoupled on the eval set — they rise
together.** The decoupling the whole test was built on does not exist where the
measurement happens. I verified the premise in the training data and then measured
somewhere else.

## The second problem: the "peak" is not real

Even taking the numbers at face value, the 3-hop peak is not statistically
distinguishable from 4-hop:

```
3-hop minus 4-hop:  −0.285   95% CI [−0.806, +0.210]
CIs overlap heavily; 4-hop has n=200 and a CI spanning zero.
```

A pattern of −0.140 / −0.500 / −0.220 across three cells, only one of which is
significant, is consistent with a monotonic effect plus noise.

And note the direction: **the highest-failure bucket (4-hop, 76%) shows the
*smallest* point estimate.** If H-fail were right that is the cell that should be
largest. It is not — though not significantly so either.

## What now stands

**Withdrawn:** "the effect scales with failure rate, not horizon."

**Retained:** the effect **grows on harder data** — MuSiQue −0.267 pooled vs
HotpotQA −0.167, and the per-doomed-episode saving is stable at −0.42 to −0.58
across three datasets.

**Undetermined:** *why* it grows. MuSiQue has both 2.2× HotpotQA's failure rate
and 1.15× its horizon; those two cannot be separated with the data we have. The
between-dataset comparison mildly favours failure rate (the failure difference is
far larger than the horizon difference), but that is an argument from proportion,
not a test.

## Consequence for the plan

The correction I made to `paper_plan_v2_2_foundation.md` §10 — *"choose the next
benchmark for difficulty, not for how long its episodes are"* — **is no longer
supported and must be softened.** The honest guidance is: choose a benchmark where
the agent **fails often and episodes are long**, since we cannot say which drives
the effect, and both are higher on the dataset where the effect was larger.

## How to test it properly

The clean design is a dataset where failure rate and horizon genuinely vary
independently *at evaluation time* — e.g. hold hop-count fixed and vary difficulty
by distractor density, or subsample eval questions into matched cells with equal
failure rate but different length. Neither is expensive; both were skipped because
the training-rollout statistics appeared to give the decoupling for free.

## Why this was caught

Pooling seeds doubled the per-bucket n and forced me to recompute the predictors
on the eval set rather than trusting the pre-registration's training-set figures.
Pre-registration protected against *choosing a hypothesis after seeing the result*
— it did **not** protect against **validating the test's premise on different data
from the measurement**. That is a distinct failure mode and worth naming.

**This is the fifth claim withdrawn in this project** (see `T5_SYNTHESIS.md` §5).
Each was caught by a check built into the process rather than by an outside
reviewer, which is the process working — but the rate is high enough that any
write-up should say so plainly.

Artifacts: pooled analysis over `t4_musique/` seed 42 and `t3_seeds/` seed 123.
