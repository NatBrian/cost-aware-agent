# S0 — re-scoring the FOUNDATION-1 λ arms at binding budgets — 2026-07-31

**Question:** FOUNDATION-1 evaluated at `{2,4,8}` and gated at B=4, where the
budget was slack for 67% of episodes. If pricing works once the budget *binds*,
the already-trained checkpoints might show it — for the cost of an evaluation
instead of a week of training.

**Answer: no. S0 is null, and it does not shorten the path.** Step 1 proceeds to
S4 as planned.

val-50, temperature 0, harness off, budgets `{2,3,4}`. **dev-200 untouched.**

## Results

| arm | λ | B | W | F1 | steps | fail% | aband% | self-stop% |
|---|---|---|---|---|---|---|---|---|
| lam0 | 0.0 | 2 | 1.040 | .619 | 2.700 | 34.0 | 2.0 | 34.0 |
| lam0 | 0.0 | 3 | 1.180 | .680 | 3.180 | 28.0 | 10.0 | 72.0 |
| lam0 | 0.0 | 4 | 1.320 | .713 | 3.400 | 26.0 | 8.0 | 80.0 |
| lam03 | 0.3 | 2 | 0.880 | .699 | 2.760 | 26.0 | 2.0 | 38.0 |
| lam03 | 0.3 | 3 | 1.440 | .707 | 3.480 | 26.0 | 8.0 | 74.0 |
| lam03 | 0.3 | 4 | 1.600 | .673 | 3.500 | 30.0 | 12.0 | 82.0 |
| lam10 | 1.0 | 2 | 1.200 | .601 | 2.860 | 32.0 | 6.0 | 42.0 |
| lam10 | 1.0 | 3 | 1.040 | .642 | 2.860 | 30.0 | 14.0 | 78.0 |
| lam10 | 1.0 | 4 | 1.400 | .655 | 3.380 | 28.0 | 12.0 | 82.0 |

Paired per-task deltas vs the λ=0 control (\* = 95% CI excludes zero):

| B | arm | ΔW | 95% CI | ΔF1 | 95% CI |
|---|---|---|---|---|---|
| 2 | λ=0.3 | −0.160 | [−0.480, +0.160] | +0.080 | [−0.020, +0.180] |
| 2 | λ=1.0 | +0.160 | [−0.260, +0.680] | −0.018 | [−0.096, +0.057] |
| 3 | λ=0.3 | +0.260 | [−0.220, +0.840] | +0.027 | [−0.020, +0.093] |
| 3 | λ=1.0 | −0.140 | [−0.640, +0.220] | −0.038 | [−0.111, +0.033] |
| 4 | λ=0.3 | **+0.280\*** | [+0.020, +0.620] | −0.040 | [−0.100, +0.000] |
| 4 | λ=1.0 | +0.080 | [−0.320, +0.500] | −0.059 | [−0.131, +0.000] |

## Reading it honestly

**No consistent signal.** Signs flip across budgets within the same arm (λ=0.3 is
−0.160 at B=2 and +0.260 at B=3), which is the signature of noise, not effect.

**The one "significant" result points the wrong way** — λ=0.3 shows *more* waste
than λ=0 at B=4. It should not be over-read: it is one CI-excludes-zero result out
of six comparisons at α=0.05, which is roughly what chance produces. It is
reported because suppressing an inconvenient significant result is precisely the
practice pre-registration exists to prevent.

**This is what S2 predicted.** n=50 gives a ΔW CI half-width of ~0.45 against an
achievable effect of ~0.13. **S0 was underpowered by construction — by about 46×
on this estimand** — and was run anyway because it was half a day of GPU on
checkpoints that already existed, and a large effect would have been visible.
There is no large effect.

**What S0 rules out:** a free answer. There is no shortcut past S4.

## Two things S0 does show

**1. Looser budgets waste more.** For the control arm, W rises monotonically with
the budget: **1.040 → 1.180 → 1.320** at B = 2 → 3 → 4. Giving the agent more room
does not make it more accurate in proportion; a good part of the extra allowance
is spent on questions it never answers.

**2. B=2 genuinely binds.** Self-stop is only **34–42%** at B=2 versus 72–82% at
B=3 and B=4 — at the tight budget most episodes run past the allowance rather than
stopping on their own. That independently confirms S2's binding calculation from
the eval side, and confirms B=2 is the right gate budget.

## Limitations

- **n=50.** The dominant limitation; see above.
- **These checkpoints were trained under `{2,4,8}`**, so **B=3 is a budget none of
  them ever saw**. That column is out-of-distribution for all three arms and the
  weakest row in the table.
- λ=1.0's checkpoint is the round-3 one whose health probe failed in FOUNDATION-1
  (11.0% malformed vs the 10% gate) — mildly damaged, as documented then.
- Diagnostic only. No pre-registered rule attaches to S0, and nothing here can
  change the Step-1 verdict.

Artifacts: `experiments/results/s0_rescore/` · script `scripts/s0_analyse.py`.
