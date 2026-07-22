# Report 02 — Stopping Labels (P3, Snell envelope)

**Date:** 2026-07-22 · **Hardware:** CPU only · **Status: ✅ DONE — all QC checks passed**

## What we computed, in plain words

For every one of the **96,000 steps** in the collected trajectories, we answered
one question with math instead of opinion: *"standing at this step, is it better
to STOP now (keep the current answer, keep the unspent money) or CONTINUE (pay
for more steps, hope the answer improves)?"*

The method walks **backwards** from the last step (the same dynamic-programming
idea used to price stock options — the Snell envelope / Longstaff–Schwartz
method). "What the future is worth" is estimated by a regression across all
9,600 trajectories, not read from one lucky run. Each step gets three labels:
STOP/CONTINUE, the margin Δ* (how much continuing is still worth), and the
state value V* — the three things the reward model will learn to predict.

We produced labels at **five price-sensitivity levels** (λ) and in **two
economies** (the main tier-scaled one, where money matters more as the wallet
empties, and a plain one kept for an ablation).

## Results — the cost dial works

Average optimal stopping step, by price sensitivity (main economy):

| λ (how much a dollar hurts) | 0.1 (cheap money) | 0.5 | 1 (headline) | 2 | 5 (expensive money) |
|---|---|---|---|---|---|
| mean optimal stop τ* | **step 5.0** | 3.4 | **2.8** | 1.1 | 1.0 |

Read it like a dial: when money barely matters, the optimal policy works ~5
steps; at the headline setting it works ~3; when money is precious, stop
immediately. Perfectly monotone — **0 violations in 38,400 checked pairs**
(quality check c).

Other checks:
- **Noise sensitivity (check b):** re-computing labels with subsampled quality
  readings moved τ* by only 0.036 steps on average — labels are stable.
- **Prophet bias confirmed (bonus result):** the naive "just pick the best step
  in hindsight" label stops **+0.41 steps later** than our non-psychic labels —
  exactly the foresight bias the plan predicted (§2.2) and the reason we use
  the Snell recursion. This becomes evidence in experiment E4.

## One honest caveat found in review

Example from the 20-trajectory review file: gold answer "**DEA**", agent's
answer "**Drug Enforcement Administration**" — string-matching scores this
**0.0** despite being semantically correct, so that trajectory's label says
"nothing was ever gained, stop at step 1." This is the standard scoring
convention of the entire baseline literature (we keep it for comparability),
but it adds some label noise on alias-style questions. Mitigations already in
the design: F1 is reported alongside exact-match, and the reward model
averages over ~10⁵ steps, diluting individual scoring quirks.

**Manual review file:** `experiments/labels/round0/qa_review_20.jsonl` (+ the
memo `label_quality_memo.md`) — 20 trajectories with per-step U, continuation
value, and the computed stopping point, for human eyeballing.

## What this unlocks

**Report 03:** train the small (2B) reward model to predict these labels from
only inference-available information (budget state, step count, draft
stability — never the gold answer), then the hard gate: it must beat two
simple baselines on held-out data, or the pipeline stops here by design.
