# Pilot memo (E-a) — budgets and λ, derived from data — 2026-07-22

**Run:** 50 train tasks × 4 rollouts, forced continuation to step 10, per-step
draft scoring. 200 episodes. Raw: `experiments/results/foundation/pilot/`.

## What the curves showed (plain language)

Picture each question as "how good is the agent's draft answer after each
step?" Averaged over all 200 episodes:

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| draft F1 | .02 | .22 | .36 | .41 | .43 | .47 | .49 | .50 | .52 | .51 |

- **The knee is at step 3.** Steps 2–3 buy ~0.20 and ~0.14 F1 each; every step
  after buys ~0.02–0.05. Six extra steps past the knee buy just +0.10 total.
- **Overthinking is visible in our own data:** in 24/200 episodes the final
  draft is WORSE than the episode's best draft (the agent talked itself out of
  a right answer), and the average curve dips from step 9 to 10.
- **Natural stopping:** the untrained agent answers at median step 3
  (P25=2, P75=4, P90=7; 6/200 never answer). Steps-to-95%-of-own-peak matches
  almost exactly (mean 3.65) — its median stop is roughly calibrated, but the
  tail is long, and it cannot adapt to a budget it isn't told about.
- 60/200 episodes never reach any quality (retrieval-limited failures);
  T_max=10 is comfortably enough (gains ≈ 0 beyond step 8).

## Decisions (frozen in configs/foundation.yaml)

1. **Budgets {small: 2, medium: 4, large: 8}** — P25 / P75 / ~beyond-P90 of
   steps-to-95%-peak. Small genuinely binds (half of episodes want ≥3 steps);
   medium covers the typical case; large is near-unconstrained but below the
   hard cap. The provisional {3,6,10} was too loose — at medium 6, hardly any
   pressure existed.
2. **λ = 0.3 headline** (0.1 spot-check). At B=4 this prices a step at 0.075
   utility: with the curve above, utilities become U(2)=.07, U(3)=.14 (optimum),
   U(4)=.11, U(5)=.06 — an interior optimum at the knee, positive, non-trivial.
   At λ=0.1 the optimum shifts to ~step 5 — the dial demonstrably moves the
   target. (The old λ=0.5 made all utilities negative at medium — too punishing.)
3. T_max stays 10.

**Approval note:** constants frozen under Brian's full-autonomy mandate
(2026-07-22, "complete the entire end-to-end experiments"). Revisable until the
final dev-200 evaluation; revising after baseline runs costs a baseline rerun.

## What this means for the method's chances

The raw agent's median stop is already near the knee — so the headroom for the
RL arm is: (a) the long tail (P90=7, never-stoppers), (b) the 12% who degrade
by over-continuing, (c) budget-adaptivity (stopping at 2 when B=2 without being
cut), and (d) fewer redundant searches per episode (pilot trajectories show
query-rephrasing loops). The gate (A3 beats A1/A2 on utility at B=4) is
realistic but not free — exactly the honest test the foundation wants.
