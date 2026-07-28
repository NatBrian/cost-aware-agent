# E-b re-run — judge switched to Qwen3.6-27B — **GATE NOT PASSED** — 2026-07-28

**Why re-run.** The judge changed (gemma-4-31B-it :6102 → Qwen3.6-27B :6101, Brian
2026-07-28). Rubric agreement is judge-specific, so the .848 from 2026-07-22 does
not transfer. Cost: 50 judge calls, no GPU — `sheet_v1.csv` survived the wipe with
the 50 human labels *and* each row's exact judging context.

## Results (agreement with the 50 human labels)

| bit | gemma / v1 | Qwen / v1 | Qwen / v2 | Qwen / v3 | n |
|---|---|---|---|---|---|
| new_info | .792 | .792 | .826 | **.833** | 24 |
| not_redundant | .833 | .875 | .826 | **.917** | 24 |
| was_needed | 1.000 | .792 | .783 | **.750** | 24 |
| supported | .846 | .769 | .654 | **.769** | 26 |
| nothing_left | .769 | .615 | .731 | **.692** | 26 |
| **mean** | **.848** | **.769** | **.764** | **.792** | |

Gate (plan §5, F3: ≥80% per bit): **FAIL** on all three rubric versions with the
new judge. Looser mean+floor reading (mean ≥.80, no bit <.70): also FAIL (v3 mean
.792; `nothing_left` .692 is a hair under the floor).

## What the rubric revisions did

- **v2** applied the audit's recommendations: a disjointness note (new_info judges
  the RESULT, not_redundant the QUERY), evidence-citation fields, an
  undecided→NO tie-break on redundancy, and a truncation-leniency rule on
  `supported`. Result: `nothing_left` +.116, but `supported` −.115 — the leniency
  rule pushed the judge toward YES where the humans had said NO. One prompt also
  failed to parse: the model returned a verbatim `evidence_quote` containing `""`,
  which is invalid JSON — a self-inflicted injection hazard in a field we added.
- **v3** keeps what measured better (disjointness note, `nothing_left` procedure,
  the `nearest_prior_step` integer) and reverts what measured worse (`supported`
  and `not_redundant` back to v1 wording), and drops the free-text quote fields.
  Best mean so far, and **0 parse failures, 0 neutral fallbacks**.

## The real finding: the instrument cannot resolve these differences

At n = 24–26 per bit, the 95% CI on a single bit's agreement is about **±0.17**:

| quantity | value | 95% CI |
|---|---|---|
| `nothing_left` (v3) | .692 | [.515, .869] |
| `was_needed` (v3) | .750 | [.577, .923] |
| mean (v1) | .769 | [.695, .843] |
| mean (v3) | .792 | [.721, .863] |

Moving one bit from .692 to the .80 gate takes **three label flips**. The v1/v2/v3
spread (±.05) sits well inside that noise, so continuing to tune prompt wording
against this sheet would be fitting noise, not improving the judge — the rubric
equivalent of tuning on the test set.

Two further reasons this sheet cannot arbitrate the gate:

1. **The labels are anchored to the previous judge.** `supported` only reached
   .846 in the original run after a review round in which the human labels were
   revised *while reading gemma's reasoning* (.692 → .846). Those labels are no
   longer independent of a judge, and Qwen is now being scored against them.
2. **They are author-labels, not Brian's**, which F3 asks for.

## Recommendation

Rebuild the instrument rather than keep tuning the rubric against it:

1. Draw a **larger sheet (~150 steps)** from the fresh pilot that Phase 3 has to
   run anyway — ±0.17 shrinks to about ±0.07 at n=150.
2. **Label blind** — labels written before any judge output for that row is
   requested, which removes the anchoring that inflated the original number.
3. Re-run agreement on **v1 and v3** against the new sheet and keep the winner.

Cost: one labeling round, no extra GPU. Until it passes, no RL run should consume
these scores — that is what the gate is for.

**Config now:** judge Qwen3.6-27B, `rubric_v3`, `max_tokens` 1024,
`enable_thinking: false`. The judge-vs-realized-F1 divergence log remains the
in-training safety net regardless of which rubric ships.
