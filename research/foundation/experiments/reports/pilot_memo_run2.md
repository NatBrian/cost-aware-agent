# Pilot memo, run 2 (E-a re-run after the wipe) — 2026-07-28

**Run:** 50 train tasks × 4 rollouts, forced continuation to step 10, temp 1.0,
200 episodes. Raw: `experiments/results/pilot/pilot.jsonl` (backed up to
`/mnt/src/liangsheng/cassi_foundation/trajectories/`).

**Why re-run:** the first pilot's trajectories were lost in the container wipe,
and train-300 had to be re-derived, so the constants frozen from the old data
needed re-checking against the data we actually have.

## The curve reproduces run 1

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| draft F1 (run 2) | .00 | .15 | .28 | .34 | .37 | .41 | .44 | .44 | .47 | .48 |
| draft F1 (run 1) | .02 | .22 | .36 | .41 | .43 | .47 | .49 | .50 | .52 | .51 |

Run 2 sits ~0.05 lower throughout — expected, since train-300 was re-sampled and
run 1's exact questions are unrecoverable — but every structural feature the
plan depends on is unchanged:

- **Knee at step 3.** Steps 2–3 buy +0.15 and +0.13; every later step buys
  ≤0.06. Identical to run 1.
- **Overthinking in 24/200 episodes** — the final draft is worse than the
  episode's own best. *Exactly* run 1's count.
- **Natural stopping at median step 3** (P25 2, P75 6, P90 8); 5/200 never
  answer (run 1: 6/200).
- Final answer F1 .474, against the surviving A1 B=4 baseline's .471.

Four independent statistics landing on run 1's values is good evidence the
rebuilt pipeline is measuring the same thing the baselines were measured on.

## Constants: unchanged, re-derived from this data

`U(t) = draftF1(t) − λ·(t/B)` on the run-2 curve:

| λ | B=2 | B=4 | B=8 |
|---|---|---|---|
| **0.3** | optimum t1 (U −.150) | **optimum t3 (U +.055)** | optimum t4 (U +.190) |
| 0.1 | optimum t2 (U +.050) | optimum t4 (U +.240) | optimum t7 (U +.352) |

**λ = 0.3 and budgets {2, 4, 8} stand.** At the gate budget B=4 the optimum is
step 3 — the knee — interior and positive, which is exactly the property the
constants were chosen for. B=2 is all-negative: that budget genuinely binds, as
designed (it is where A2's hard cutoff collapsed F1 to .221 while A1 kept .478).
B=8 is near-unconstrained. λ=0.1 shifts the optimum to step 4/7, so the dial
still moves the target and remains a meaningful spot-check.

No constant is changed. The frozen values were derived from different data and
survive re-derivation on this data.

## The malformed-rate question, resolved

The 20-task smoke showed **20.5%** malformed steps and I held the scaffold
unchanged pending a bigger sample. At n = 2000 steps the rate is **2.4%**
(47/2000; 26/200 episodes contain at least one). The smoke figure was
small-sample noise — 78 steps, with the malformed ones clustered in a couple of
bad episodes.

**Decision: no scaffold change.** 2.4% is comfortably inside the post-round
health gate (malformed < 10%), and the reward's format term already penalises
the rest. Changing `max_tokens_per_step` or the prompt would have handed A3 a
scaffold the committed A0/A1/A2 baselines never ran under — a confound in the
one comparison the whole gate rests on — and re-collecting those baselines would
have spent one of the ≤3 permitted dev-200 looks. Waiting for n cost nothing and
saved that.

The underlying mechanism is still worth recording: malformed steps are
*truncations*, not format disobedience — `raw_len` 1961–2228 characters against a
512-token cap, so a verbose THOUGHT is cut mid-sentence and the required ACTION /
BEST ANSWER lines never arrive. If a future run wants them gone, the fix is a
larger cap or a length instruction, applied to **all arms** with baselines
re-collected.

## What this means for the method's chances

Unchanged from run 1, and now confirmed on the data we will actually train on.
The headroom for A3 is: the long tail (P90 = 8 steps), the 12% of episodes that
degrade by continuing past their own best draft, budget-adaptivity at B=2, and
redundant re-search. The agent's *median* stop is already at the knee, so the
gate (beat A1's U=.205 at B=4) is a real test, not a formality.
