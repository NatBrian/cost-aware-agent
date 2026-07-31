# S2 — headroom audit and economy calibration — 2026-07-31

Run on `experiments/results/train/lam0_round1/rollouts.jsonl` (2400 episodes, 300
tasks × G=8, λ=0 arm). No GPU, no new collection. Everything here is measured
**before** any threshold is written down — that ordering is the pre-registration
(plan v2.2 §7.7 rule 1).

## Headline: the estimand had to change, and it is measurement that says so

| | |
|---|---|
| Primary estimand | ~~wasted spend `W`~~ → **paired Δsteps at iso-F1** |
| Gate budget | ~~B=3~~ → **B=2** |
| Required eval n | **≥ 479** (dev-200 was less than half) |
| Detection threshold | **0.119 steps** (50% of the 0.238 achievable) |
| **λ\*** | **0.568** (cap 0.6) |

## 1. Only B=2 binds — and my first budget choice was measured on the wrong policy

| budget | % of episodes that would overspend it | verdict |
|---|---|---|
| **B=2** | **64.8%** | **BINDS** |
| B=3 | 25.5% | slack |
| B=4 | 17.2% | slack |

I set budgets to `{2,3,4}` and gated at B=3 on the strength of a 41%-binding
figure. **That figure came from the pilot — an untrained A1 policy with mean stop
4.16.** The policy that will actually be trained stops at **3.33**, so B=3 binds
for only 25.5%.

Corrected: **gate at B=2.** B=3 and B=4 are kept as the non-binding contrast, so
the binding/non-binding dose-response is *shown* rather than assumed.

Lesson, and it is the same one twice now: **measure against the policy you will
actually run, not a convenient proxy.**

## 2. Achievable headroom is well under the oracle

Two rules, and the gap between them is the point:

| rule | uses gold? | B=2 ΔW | B=2 Δsteps | B=2 ΔF1 |
|---|---|---|---|---|
| oracle, quit at k=3 | **yes** — upper bound only | −0.226 | −0.337 | −0.020 |
| **learned** (S1 classifier, k=3, θ=0.5) | **no** — implementable | **−0.134** | **−0.238** | −0.018 |

The learned rule captures ~60% of the oracle. That is a healthy ratio, and it is
the honest number: it is what a policy could actually implement from gold-free
state. **The Step-1 threshold is derived from the learned rule, never the oracle.**

At B=4 the achievable effect collapses to Δsteps −0.164 / ΔW −0.019, and at B=8
to −0.052 / −0.011. The headroom lives almost entirely where the budget binds.

## 3. `W` is not runnable, and the power analysis is why

`W = E[steps × 1(F1=0)]` is a 0-or-steps mixture: **72% of episodes contribute
exactly zero**, and the rest contribute 2–8. Its paired within-task SD is **1.64**,
against an achievable effect of 0.134.

| B | estimand | effect | paired SD | threshold | **n needed** | |
|---|---|---|---|---|---|---|
| 2 | **steps** | 0.238 | 1.328 | 0.119 | **479** | **FEASIBLE** |
| 2 | W | 0.134 | 1.640 | 0.067 | 2,289 | too big |
| 4 | **steps** | 0.164 | 1.147 | 0.082 | **751** | **FEASIBLE** |
| 4 | W | 0.019 | 1.636 | 0.010 | 108,262 | too big |
| 8 | steps | 0.052 | 1.815 | 0.026 | 18,513 | too big |
| 8 | W | 0.011 | 2.406 | 0.006 | 675,204 | too big |

**W would have been unpassable by construction** — the same structural failure as
FOUNDATION-1's 0.5-step threshold against a 0.31-step ceiling, in mirror image: a
threshold *below the noise floor* is as unreachable as one above the ceiling. It
was caught this time because the plan requires measuring the ceiling and the
noise floor before writing the threshold. **The process worked.**

`W` is retained as the *economic interpretation* of the result — it is what the
saving means in budget terms — but the gate is decided on Δsteps.

## 4. This partly re-reads FOUNDATION-1's null

The λ ablation ran at **B=4 with n=50**. S2 says B=4 needs **n≈751**. The ablation
was therefore **roughly 15× underpowered** for the effect that is actually
available there.

Precisely what this does and does not change:

- **Unchanged:** the pre-registered rule required Δ ≥ 0.5 steps, and the
  achievable ceiling at B=4 is 0.164. The rule was unpassable, and "NOT
  EFFECTIVE" against *that rule* stands.
- **Changed:** the softer reading — "pricing moves stopping by 0.04, essentially
  zero" — is **not supported**. The observed +0.040 with a ±0.220 CI could not
  have resolved a 0.164 effect either way. The correct statement is *unresolved
  at that n*, not *absent*.

The one significant result in the whole ablation (B=2, −0.70 steps, CI excludes
zero) is exactly where S2 says the effect is largest and the budget binds. That is
now three independent lines pointing at the same place.

## 5. λ calibration

At the gate budget B=2, with the achievable rule saving 0.238 steps for 0.018 F1:

```
need λ ≥ (target_ΔU + ΔF1_loss)·B / Δsteps
       = (0.05 + 0.018)·2 / 0.238
       = 0.568          (cap 0.6)  ->  λ* = 0.568
```

**λ\* = 0.568, uncomfortably close to the 0.6 cap.** The cap exists because λ=1.0
breached the policy-health gate (11% malformed vs the 10% limit) and produced the
lowest F1 of the three arms. λ=0.568 is untested territory between a known-safe
0.3 and a known-harmful 1.0, so **S4 runs the temp-1.0 health probe after round 1
and aborts on breach** rather than discovering it at round 3.

## 5b. Validation: λ\* actually exerts pressure

A calibrated λ is worthless if GRPO cannot see it. Group-relative advantages
remove anything constant within a group, so only *within-group* variation counts.
Measured on the λ=0 rollouts:

| B | SD(F1) within group | SD(λ·steps/B) | **cost share of the signal** |
|---|---|---|---|
| 2 | 0.215 | 0.088 | 29.1% (λ=0.3) |
| **2** | **0.215** | **0.167** | **43.7% (λ\*=0.568)** |
| 4 | 0.186 | 0.042 | 18.5% (λ=0.3) |
| 4 | 0.186 | 0.080 | 30.1% (λ\*=0.568) |

At the gate budget the cost term carries **43.7%** of the advantage signal — a
real share, and comfortably below the **81.7%** at λ=1.0 that came with a
breached health gate. λ\* sits in the intended middle.

This also re-confirms the FOUNDATION-1 diagnostic: the term was never being
cancelled by GRPO. At λ=0.3 it was already 29% of the signal at B=2 and still
moved nothing there — because the *measurement* (mean steps at n=50, B=4) could
not have seen it.

## 6. Consequence: a new frozen eval set

n ≥ 479 makes dev-200 unusable as the eval set. Built **eval-600**
(`data/hotpotqa_eval_600.jsonl`, sha256 `42b8de8d…`), stratified and seeded,
**disjoint from train-300 / dev-200 / val-50 by id and by normalized question
text** (asserted, not assumed).

Disjoint rather than a superset of dev-200: FOUNDATION-2 changes the budgets, so
the old dev-200 rows are not comparable to the new arms anyway — reuse buys
nothing, while a fresh draw removes any question of adaptive contamination from
the looks dev-200 has already taken. 7,205 eligible questions made the draw free.

## Honest limitations

- **The "achievable" rule is a post-hoc filter, not a policy.** It truncates
  existing trajectories. A trained policy can do better (it can change the whole
  trajectory, not just cut it) or worse (RL is noisy). 0.238 is an estimate of
  what is available, not a promise of what training will capture.
- **Measured on the λ=0 arm at round 1.** A more-trained policy may have different
  slack. Direction is unlikely to reverse; magnitude may move.
- **Rollouts are temperature 1.0** training rollouts at budgets {2,4,8}. The eval
  runs at temperature 0 and {2,3,4}, so these are estimates from a related but
  not identical distribution.
- **The classifier threshold θ and step k were chosen by scanning** and reported
  at their best. That is legitimate for estimating a ceiling, but it means 0.238
  is optimistic as a prediction of trained behaviour.
- **eval-600 is 100% `hard` level** — HotpotQA's dev split is hard-only by design
  (logged since F1). The train set is mixed, so there is a train/eval difficulty
  shift that applies equally to all arms.

Artifacts: `experiments/results/s2/s2_headroom.json` · scripts
`s2_headroom.py`, `s2_build_eval.py`.
