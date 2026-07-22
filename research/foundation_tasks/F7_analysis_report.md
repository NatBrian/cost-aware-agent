# F7 — Analysis, figures, and the foundation report

**Goal:** the human-readable outcome: two figures, one plain-language report, and
the logged GO/NO-GO verdict that decides whether we scale back up toward the full
v2.1 plan.

## Figures (one script each, CSV in → PDF out, no hand-edited numbers)

- **Fig 1 — the mini-frontier:** steps-used (x) vs F1 (y), one line per arm across
  the three budgets; oracle point if the F6 optional replay ran. The foundation's
  whole claim is visible here: A3 should sit up-and-left of A1 and A2.
- **Fig 2 — internalization:** (a) % self-stopped before budget per arm; (b) A3
  harness-off vs harness-on utility bars; (c) stop-step histograms per budget
  (does A3 stop earlier under smaller B — budget sensitivity without enforcement?).
- **Fig 3 (diagnostic, appendix-grade):** judge-score vs realized-F1 divergence
  during training, from the F5 dashboard.

## The report (`experiments/reports/foundation_run_1.md`, plain language)

Structure (kept simple, per the standing reporting preference):
1. What we tested, in three sentences.
2. The gate verdict (GO/NO-GO) with the three §6 conditions itemized, numbers
   inline, date-stamped.
3. What each arm did (table: F1, steps, U at medium budget + CIs).
4. What surprised us / qualitative trajectory examples (2–3, quoted).
5. Judge behavior: calibration table, divergence curve reading, hacking verdict.
6. What this means for the full v2.1 plan — an explicit adjustment list: which
   v2.1 decisions the foundation confirmed, which it contradicts, what to change
   before scaling (this section is the deliverable the senior asked for).
7. Costs of the run itself: GPU-hours, judge calls, wall-clock per stage.

## Optional diagnostic: the adapted PPTAgent six-dimension rubric

Per F3's prior-art decision: run the task-adapted PPTAgent trajectory-eval
rubric (`/mnt/src/code/PPT-GEN-Demo/eval_codes/trajectory_eval/`, dimensions
that apply to single-agent QA: query quality, retrieval, search-integrated
reasoning) as an OFFLINE diagnostic over before/after-training trajectories —
answers "did RL improve only stopping, or also search skill?" for the report's
§4. Human-read analysis only; its scores never touch rewards or the gate.

## Also in this stage

- Update the memory file (cassi-paper-status) with the verdict.
- Commit everything; tag the commit `foundation-run-1`.

## Done criterion

Figures regenerate via one make target; report reviewed by Brian; verdict logged.

Depends on: F6. Feeds: the decision about resuming the full plan.
