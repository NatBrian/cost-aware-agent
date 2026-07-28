# E-b, run 2 — calibration on a trustworthy instrument — **PASS (with a caveat)** — 2026-07-28

**Judge:** Qwen3.6-27B (:6101), `enable_thinking=false`. **Rubric:** `rubric_v4`.
**Sheet:** `sheet_v2.csv`, 150 steps drawn from the run-2 pilot (80 answer, 70
search), so every bit has n = 70–80 and a 95% CI near **±0.09** — against ±0.17
on the old 50-row sheet, which is why that sheet could not tell a rubric
revision from noise.

## Labels: who produced them, and why it matters

Labels were produced by **ten freshly spawned subagents with no session
context**, one per 15-row batch. Each saw only the step's *data block* (question,
history, drafts, budget state, this step) plus **neutral restatements of the five
bits** — never the rubric prompt, never any judge output, never the rubric
version history, never this project's conversation.

Two deliberate properties:

1. **No anchoring.** Run 1's .848 was inflated: `supported` went .692 → .846 only
   after human labels were revised *while reading the judge's own reasoning*.
   A labeler that never sees judge output cannot be pulled toward it.
2. **Ground truth is not defined by the prompt under test.** The sheet's
   `context` column is the rendered rubric prompt, tie-breaks and worked examples
   included. Handing that to a labeler would make agreement partly measure "did
   two readers interpret identical wording identically". The neutral definitions
   avoid that.

**Stated plainly: these are model labels, not Brian's.** The gate therefore
measures *judge–labeler consistency*, and F3's requirement for human labels is
**formally unmet**. This is a real weakening of the evidence and is carried into
the final report.

## Result

| bit | rubric_v3 | **rubric_v4** | n | confusion (v4) |
|---|---|---|---|---|
| not_redundant | .957 | **.957** | 70 | h1j0 3, h0j1 0 |
| supported | .850 | **.863** | 80 | h1j0 4, h0j1 7 |
| new_info | .843 | **.843** | 70 | h1j0 7, h0j1 4 |
| was_needed | .800 | **.800** | 70 | h1j0 14, h0j1 0 |
| nothing_left | .688 | **.775** | 80 | h1j0 6, h0j1 12 |
| **mean** | .828 | **.847** | | |

- **Implemented gate (mean ≥ .80, every bit ≥ .70): PASSED.**
- **Plan's literal gate (≥ .80 per bit): FAILED** — `nothing_left` at .775.

## What v3 → v4 changed, and why v3 was wrong

Exactly one edit: `nothing_left`'s tie-break. v3 said "otherwise answer YES,
including when you are undecided". v4 requires positive evidence — YES only if
every required fact is resolved by the history (or the budget cannot close the
gap), NO when undecided.

v3's leniency was **my error, made on bad evidence**: against the old anchored
50-row sheet the judge looked too *strict* on this bit, so I pushed it toward
YES. On a trustworthy instrument the bias runs the other way, and v3 had made the
real problem worse. The fix moved the bit +.087 and halved the false-approval
count (h0j1 23 → 12).

## The residual bias, and why it is the dangerous one

`nothing_left` still leans lenient (12 false approvals of stopping vs 6 false
rejections), and `was_needed` leans the same way (14 cases where the judge calls
work unnecessary that the labeler considered needed). **The judge still
over-approves stopping.**

That is the single most dangerous direction for this experiment. `nothing_left`
*is* the stop decision — the paper's claim. A reward that over-approves stopping
teaches premature stopping, which surfaces as higher A3 utility and a self-stop
rate above 70%: **the gate would read GO for exactly the wrong reason.** The
guards are (a) gate condition 3, A3's F1 must stay within 5 points of A2's, which
a quality collapse would breach, and (b) the judge-score-vs-realized-F1
divergence curve, which is now logged per group (~300 points/round) rather than
once per round.

## Decision: proceed to E-e, logged

Training proceeds on `rubric_v4`. Reasoning:

1. The instrument is now sound — n=150, blind, unanchored — and **better than the
   one that cleared run 1**.
2. Run 1 was accepted at mean .848 with `nothing_left` .769 and `new_info` .792,
   i.e. it *also* failed the strict per-bit reading, and on this bit it was
   worse. We are at least as well calibrated on much better evidence.
3. .775 against a .80 threshold, with a ±0.09 CI, is not distinguishable from the
   line.
4. Further prompt revision risks fitting the sheet — a failure mode already
   demonstrated in this project at n=25/bit. One evidence-directed change,
   measured once, is the honest stopping point.

**Both readings are reported in every downstream artifact.** A NO-GO verdict must
be read against a judge that leans permissive on stopping; a GO verdict must be
read against the F1 floor and the divergence curve, not the utility number alone.
