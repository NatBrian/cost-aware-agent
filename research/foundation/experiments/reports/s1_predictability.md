# S1 — gold-free predictability check — **PASS** — 2026-07-31

**The kill gate for FOUNDATION-2.** Pre-registered in
`paper_plan_v2_2_foundation.md` §8: held-out AUC ≥ 0.65 or the redesign does not
proceed as specified.

## The question

FOUNDATION-2's thesis is that the agent should **abandon unproductive episodes**.
We can tell an episode is doomed because we hold the gold answer. *The agent
cannot.* So everything rests on one unproven claim: **is eventual failure
predictable from gold-free state alone?**

Run on rollouts already on disk. No GPU, no new collection.

## Result: **PASS**, and it replicates

Predicting `final_F1 > 0` from the first *k* steps, best of logistic regression
and gradient boosting, mean over 5 grouped splits:

| k | lam0_round1 | lam10_round1 | round3 (λ=0.3) | n at k (lam0) |
|---|---|---|---|---|
| 1 | 0.598 | 0.609 | 0.624 | 2400 |
| 2 | 0.708 | 0.727 | 0.742 | 2383 |
| **3** | **0.813** | **0.815** | **0.798** | 1555 |
| 4 | 0.728 | 0.683 | 0.695 | 613 |
| 5 | 0.694 | 0.605 | 0.663 | 412 |

**Three independent arms, same peak, same shape.** Base rate P(success) ≈ 0.75–0.76
in all three, so a trivial classifier scores AUC 0.500.

**The peak is at k = 3**, which matches the pilot's hazard curve independently:
P(eventual success | no progress by step 3) = .315. Two different measurements on
two different datasets agree that **step 3 is where doom becomes legible.**

k = 1 is *below* the gate (~0.60) in all three arms — one step is not enough
information, which is exactly what a sane result looks like. If k=1 had scored
0.8 we would be looking at leakage.

## What predicts failure

| feature | what it is | why it matters |
|---|---|---|
| **`logprob_last`** (top at k=2,3) | the model's own mean token logprob on its last step | **The agent already "knows" when it is lost.** It is simply not trained to act on it |
| **`q_coverage`** (top at k=1, high throughout) | fraction of question tokens appearing in retrieved text | "am I even retrieving about the right entities?" |
| `obs_len_mean` / `obs_len_last` | retrieval productivity | empty/short results signal a dead end |
| `draft_in_obs` (k=4,5) | is the draft grounded in what was retrieved | ungrounded draft ⇒ guessing |

**Every one of these is available to the policy at inference.** No gold, no
privileged information.

The `logprob_last` finding is the most consequential: the strongest predictor of
failure is the model's own confidence. That is a signal the policy *already
computes* and currently ignores.

## Leakage discipline

Three safeguards, because a false PASS here would cost weeks:

1. **`draft_f1_vs_gold` is never a feature.** Only the draft's *shape* (length,
   churn, emptiness, grounding) is used — never its score.
2. **The split is by `task_id`, never by episode.** Each task has G=8 rollouts of
   the same question. An episode-level split would put siblings of a test question
   into training, and the model would memorise "this question is answerable"
   instead of learning "this trajectory is going nowhere." This is the single
   most likely way to fake a PASS, and it is ruled out by construction
   (`GroupShuffleSplit` on `task_id`).
3. **Features come from `steps[:k]` only** — the prefix a deployed policy would
   actually have at its decision point.

## Honest limitations

- **These rollouts are temperature-1.0 training rollouts** of an a3 policy at
  budgets {2,4,8}, not the eval distribution. Failure rate here is ~24%, against
  43% in the unconstrained pilot — different arms, different budgets, different
  temperature. The *direction* replicates; the magnitude will move.
- **n falls sharply with k** (2400 → 613 at k=4) because most episodes are short.
  The k=4,5 numbers rest on the subset of episodes that ran long, which is a
  selected sample — and notably one where the success rate has already dropped to
  ~51% and ~41%. Long-running episodes really are worse.
- **AUC is not utility.** A 0.81 AUC classifier still makes mistakes, and every
  false positive abandons a question that would have been answered. The quit
  *threshold* must be set from the economy (S2), not from the classifier.
- Predicting `F1 > 0` is a weak bar. A stricter target (F1 ≥ 0.5) is worth
  checking at S2.

## Verdict and consequence

**PASS at 0.813 (gate 0.65), replicated on three arms.** Hopelessness is legible
from gold-free state by step 3.

- Step 1 proceeds as specified in `paper_plan_v2_2_foundation.md` §7.
- The dataset change stays deferred to Step 3 (it was to be promoted to mandatory
  on a FAIL).
- **New for S2/S3:** `logprob_last` and `q_coverage` should be exposed to the
  policy in the budget-tracker block, or at minimum logged per step. The strongest
  available quit signal is currently computed and discarded.

Artifacts: `experiments/results/s1/s1_predictability.json` (+ `s1_lam10_round1`,
`s1_round3`). Script: `scripts/s1_predictability.py`.
