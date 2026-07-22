# F4 — The baselines (A0 reference + A1/A2)

**Goal:** A1 (prompted-budget ReAct) and A2 (harness-enforced ReAct) evaluated on
the frozen dev-200 at all three budgets, plus the A0 no-information reference,
with the same collection script, schema, and metrics code the method will use.
Baselines run FIRST — they debug the shared scaffold cheaply and set the bar the
RL arm must clear.

## Arms

- **A0 — plain ReAct, no budget information (reference floor).** Frozen
  Qwen3.5-9B with the draft line but NO budget content anywhere: no B in the
  system prompt, no tracker block. Stops only by its own ANSWER or T_max.
  Restores v2.1's B1 ("how much slack exists"). Because its behavior cannot
  depend on B, it runs ONCE (200 episodes) and the same trajectories are
  re-scored under each budget's utility formula — three numbers for free.
  Its role: the attribution control for A1 — "does budget info change behavior
  at all?" is only answerable against a no-info floor. A0 is a reference point,
  NOT part of the §6 gate (the gate stays A3 vs A1/A2).
  Scaffold caveat (logged): A0's prompt necessarily differs from A1's by the
  tracker block's absence, so A0-vs-A1 measures information + its packaging —
  acceptable for a reference arm; the controlled ladder is A1→A2→A3.
- **A1 — prompted budget, no enforcement.** Frozen Qwen3.5-9B, shared scaffold
  (per-step budget tracker block, draft line). The agent stops when it
  emits ANSWER, or at T_max. Measures: does *telling* the model about budgets
  produce economic stopping? (Expected per the literature: weakly.)
- **A2 — harness-enforced.** Same everything + `enforce` mode: hard stop at B,
  final answer = last draft. Measures: the external-throttling world. Note A2's
  F1 at small B is bounded by draft quality — this is the interesting number
  (external stops can't rescue a bad draft; internalized stopping should learn to
  produce answer-ready drafts earlier).
- Optional variant (flagged in the plan §10.4, default OFF): A2 with a "budget
  nearly exhausted, wrap up" warning at step B−1.

## Work items

1. Thin arm-runner wrappers over the F2 collection script (`--arm a0|a1|a2`),
   temperature 0 at eval (v2.1 convention).
2. Run A1/A2 × dev-200 × B ∈ {small, medium, large}; A0 × dev-200 once
   (re-scored per budget). G=1 at eval (greedy).
3. Metrics via the F6 metrics module (written against the same JSONL): F1/EM,
   steps used, utility U, % episodes hitting the cap.
4. Skim 10 trajectories per arm by hand; note failure patterns in a short memo
   (feeds rubric sanity + the F7 report's qualitative section).

## Done criterion

`experiments/results/baselines.csv` (one row per arm × budget × task) + a
10-line summary in `experiments/reports/`; numbers reproducible from the script.

Depends on: F1, F2 (+F6's metrics module, which should be written early).
Feeds: F6 comparison, F7 report, and the §6 gate.
