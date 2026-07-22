# Foundation implementation progress

> Living tracker — updated at every stage boundary. Stage = one commit.
> Decisions locked 2026-07-22: GPUs allowed via ritual; judge = config placeholder
> (URL to be provided before F3 calibration); commit per stage, push to main
> (NatBrian); all five design defaults accepted as documented.

## Implementation stages (code-complete before experiments)

- [ ] **I0 — Restructure (F0):** commit plan docs · rescue `cassi/data` artifacts →
  `research/data_shared/` · rescue `cassi/experiments/reports/` →
  `research/reports_run1/` · `git mv research/cassi research/archived/cassi_2026-07` ·
  `foundation/` skeleton + fresh venv + Makefile + first test · CLAUDE.md rewrite ·
  memory updated
- [ ] **I1 — Data (F1):** sampling scripts (300 train / 200 dev, stratified, seeded) ·
  SHA256 manifest + overlap check · retrieval wrapper (rescued index; BM25 fallback) ·
  tests
- [ ] **I2 — Agent/harness/collection (F2):** `configs/foundation.yaml` ·
  prompts + budget tracker block · ReAct loop (vLLM executor client) · harness modes
  none/enforce/forced_continuation · collection script + JSONL schema validator +
  resumability · per-step draft F1 scoring · tests (mocked model) · 2-GPU smoke
- [ ] **I3 — Reward (F3):** rubric_v1 (bits, weights, anchored defs + worked examples) ·
  judge client (placeholder endpoint, batching, retry, neutral-on-parse-fail) ·
  cache keyed (rubric_version, input hash) · call logging · reward calc
  (r_t, R_final) · calibration labeling sheet + agreement/confusion calculator ·
  tests (mocked judge)
- [ ] **I4 — Metrics/eval (F4+F6):** shared `eval/metrics.py` (F1/EM/steps/U/
  self-stop) · arm runners a0/a1/a2/a3 · bootstrap CIs + per-task pairing ·
  sanity checks (row counts, U recompute, byte-reproduction) · `gate_check.py` ·
  tests (synthetic CSVs)
- [ ] **I5 — Training (F5):** verl GRPO launcher + config · step-reward injection ·
  per-step returns-to-go + group norm + min-cohort guard · Dr. GRPO hygiene ·
  KL β=0.04 · `--dry-run` (CPU, mocked judge) · wandb dashboard incl.
  judge-vs-F1 divergence · tests
- [ ] **I6 — Analysis (F7):** figure scripts (mini-frontier, internalization 3-panel,
  hacking curve) · report generator → `experiments/reports/` · adapted PPTAgent
  six-dim diagnostic (offline) · make targets · tests
- [ ] **I7 — Readiness:** full pytest green · README · experiment-start checklist
  (GPU ritual, judge URL slot, ordered run commands) → **READY-TO-EXPERIMENT**

## Experiment sequence (after I7 — each gated per the F-docs)

- [ ] E-a: 50-task pilot (forced continuation) → pilot memo → Brian approves
  budgets {small,medium,large} + λ → freeze in config
- [ ] E-b: judge URL wired → rubric calibration: 50-step labeling sheet →
  Brian's ~1h hand labels → ≥80%/70% gate
- [ ] E-c: baselines A0/A1/A2 on dev-200 → baselines CSV + 10-line report
- [ ] E-d: F5 micro-run (10 tasks, real judge) → non-degenerate rewards check
- [ ] E-e: full GRPO run (300 tasks, 1 seed) → checkpoint + dashboards
- [ ] E-f: F6 eval (A3 harness-off/on × 3 budgets) + oracle replay → gate_check
- [ ] E-g: F7 figures + report + GO/NO-GO verdict → tag `foundation-run-1`

## Log

- 2026-07-22: plan docs reviewed doc-by-doc with Brian; implementation started.
