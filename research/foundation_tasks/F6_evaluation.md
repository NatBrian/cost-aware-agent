# F6 — Evaluation

**Goal:** every number the §6 gate and the F7 report need, as CSVs with generation
scripts, computed identically for all three arms.

## Runs

- A3 (trained checkpoint) on dev-200 × B ∈ {small, medium, large}, temperature 0:
  - **harness OFF** (headline — the internalization test), and
  - **harness ON** (`enforce`) for the harness-off-vs-on gap.
- A1/A2 numbers come from F4 (same script, same frozen dev set — nothing rerun).

## Metrics module (`eval/metrics.py` — written early, used by F4 too)

- Per task: F1, EM, steps_used, U = F1 − λ·(steps/B), self_stopped flag,
  hit_cap flag.
- Aggregation: mean + 95% bootstrap CI (10k resamples); **paired per task** for
  arm-vs-arm deltas (same frozen task list makes this valid).
- Curves: steps-vs-F1 across the three budgets per arm (the mini-frontier).
- Overhead: judge calls + judge tokens per trajectory (training-side, from F5
  logs) — reported for honesty even though eval-time A3 uses no judge.
- Optional (nice, cheap): oracle upper bound from `forced_continuation` replays
  of dev-200 at medium B — best-achievable U per task in hindsight; shows how
  much headroom every arm leaves. Flagged optional; ~1 extra collection run.

## Sanity checks (must pass before the report cites any number)

1. A1/A2 numbers reproduced from F4 CSVs byte-identically by the F6 aggregation.
2. No dev task missing/duplicated in any arm's CSV (200 rows × arms × budgets).
3. Utility recomputed from raw columns matches the stored U (no stale λ).

## Done criterion

`experiments/results/foundation_eval.csv` + per-arm summaries; the §6 gate
inputs computable by one script (`eval/gate_check.py` prints GO/NO-GO with the
three conditions itemized).

Depends on: F4, F5. Feeds: F7.
