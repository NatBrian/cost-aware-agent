# Report index — FOUNDATION-2

<!-- provenance -->
> **Run:** FOUNDATION-2 · **Plan:** `research/paper_plan_v2_2_foundation.md`
> **Lineage:** full paper plan `paper_plan_v2_1.md` → FOUNDATION-1
> (`paper_plan_v2_1_foundation.md`, completed 2026-07-29, returned a null) →
> **FOUNDATION-2** (the redesign, completed and closed 2026-08-11).
> **Status of this run:** closed. Current findings are in
> `foundation_v2_2_CORRECTED_RESULTS.md`.

Read in this order. Anything not listed is superseded.

## Start here

| file | what it is |
|---|---|
| **`../../../foundation_v2_2_PAPER_measurement_artefacts.md`** | **The paper draft.** Three measurement artefacts that break agent-efficiency experiments, with this project as the case study |
| **`foundation_v2_2_CORRECTED_RESULTS.md`** | **The current, corrected findings.** Supersedes the two documents below |

## Retracted in part — kept as a record of what was claimed

| file | status |
|---|---|
| `T5_SYNTHESIS.md` | ⚠️ headline inflated; carries a retraction banner |
| `../../../foundation_v2_2_COMPLETE_REPORT.md` | ⚠️ same; plain-language version |

## The experiment, in the order it ran

| file | stage |
|---|---|
| `s1_predictability.md` | Can the agent tell it is stuck without the gold answer? (AUC 0.813) |
| `s2_headroom.md` | How much saving is possible; how many questions are needed |
| `s3_preregistration.md` | Hypotheses, thresholds and analysis script, committed before data |
| `s0_rescore.md` | Re-scoring existing checkpoints at binding budgets — null |
| `s5_verdict.md` | The main result as originally reported |
| `t1_f1_gain.md` | Where the accuracy gain came from |
| `t2_negative_control.md` | SimpleQA, out of distribution |
| `t4_preregistration.md` / `t4_musique.md` | Does the effect scale with difficulty? |
| `u1_token_cost.md` | Cost in tokens rather than steps |
| `u2_mechanism_correction.md` | Mechanism claim withdrawn |
| `u3_examples.md` | Individual trajectories, wins and costs |

## FOUNDATION-1 (the earlier run that concluded the opposite)

`foundation_report.md` · `ablation_report.md` · `pre_redesign_diagnostics.md` ·
`baselines_report.md` · `calibration_report*.md` · `pilot_memo*.md` ·
`code_audit_2026-07-28.md`
