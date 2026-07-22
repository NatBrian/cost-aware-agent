# Foundation implementation progress

> Living tracker — updated at every stage boundary. Stage = one commit.
> Decisions locked 2026-07-22: GPUs allowed via ritual; judge = config placeholder
> (URL to be provided before F3 calibration); commit per stage, push to main
> (NatBrian); all five design defaults accepted as documented.

## Implementation stages (code-complete before experiments)

- [x] **I0 — Restructure (F0)** ✅ 2026-07-22: plan docs committed (8aef0ee) ·
  79G data artifacts → `research/data_shared/` · run-1 reports + paid-measurement
  records (dataset_manifest.csv, RM-P heldout, trivial baselines) →
  `research/reports_run1/` · cassi archived → `research/archived/cassi_2026-07` ·
  `foundation/` skeleton + configs/foundation.yaml + common.py + venv
  (python3.12 via /mnt/src, get-pip bootstrap — system 3.10 lacks ensurepip) ·
  Makefile `make venv`/`make test` · 5 skeleton tests green · CLAUDE.md rewritten
- [x] **I1 — Data (F1)** ✅ 2026-07-22: `collect/sampling.py` (stratified, seeded,
  deterministic) + `scripts/f1_data.py` → frozen train-300 (194 med/55 easy/51 hard)
  + dev-200 + SHA256 manifest; rerun verified byte-identical; overlap check passes ·
  `envs/retrieval_client.py` + `scripts/serve_retrieval.py` (E5+FAISS over rescued
  64G index — dims verified 768/21M rows; live check deferred to I2 GPU smoke) ·
  14 tests green. **Finding logged:** HotpotQA dev split is 100% hard-level by
  design → dev-200 is all-hard while train is mixed (matches F1 limitation #3;
  report must mention it)
- [x] **I2 — Agent/harness/collection (F2)** ✅ 2026-07-22 (code): prompts +
  facts-not-advice tracker (a0 budget-blind verified by test) · episode loop w/
  3 harness modes, draft retry/persistence, per-step draft F1 · vLLM client +
  serve_executor.sh · schema validator · resumable collection CLI w/ seeded
  per-(task,group) wallet draw · 27 CPU tests green (commit 4bb6458).
  **Pending:** live 2-GPU smoke (20-task batch per mode) — scheduled as the
  pre-pilot checkpoint alongside retrieval-server first start
- [x] **I3 — Reward (F3)** ✅ 2026-07-22: rubric_v1 prompts (anchored bits +
  worked examples, gold-free — tested) · judge client (parse retry → neutral 0.5
  logged; disk cache; call/token stats; placeholder endpoint refused w/ hint) ·
  rewards (8-level table, malformed=worst, terminal economy, returns-to-go —
  plan worked example reproduced) · calibration sheet + agreement gate
  (mean≥.80, floor .70) · 36 tests green (commit b48ff3c)
- [x] **I4 — Metrics/eval (F4+F6)** ✅ 2026-07-22: `eval/metrics.py` (one scorer
  path, bootstrap CIs, per-task paired deltas, row-count + stale-utility guards) ·
  `eval/gate_check.py` (plan §6 itemized, thresholds in config `gate:`) ·
  `eval/build_rows.py` + `scripts/f4_baselines.sh` · 43 tests (commit 8395c82)
- [x] **I5 — Training (F5)** ✅ 2026-07-22 (CPU side): `train/advantages.py`
  (min-cohort guard tested incl. lone-survivor) · `train/reward_adapter.py`
  (+DivergenceLog = Fig-3 hacking curve) · `train/grpo_runner.py --dry-run`
  green (24 eps, nonzero advantages). **Pending:** verl glue in `_run_real`,
  wired + verified at the E-d micro-run against pinned verl (commit babfc2b)
- [x] **I6 — Analysis (F7)** ✅ 2026-07-22: 3 figure scripts (dataviz-validated
  entity-fixed palette, ALL PASS) · report generator (numbers auto-filled,
  prose TODOs) · adapted PPTAgent diagnostic (offline) · 52 tests (commit f915d73)
- [x] **I7 — Readiness** ✅ 2026-07-22: EXPERIMENT_CHECKLIST.md (ordered E-a…E-g
  w/ gates + Brian's 3 inputs) · make test/dry-run/data targets · 52 tests green,
  dry-run green → **CODE-COMPLETE; READY TO START EXPERIMENTS** (pre-flight GPU
  smoke is checklist step 1)

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
- 2026-07-22 (E-phase): judge = gemma-4-31B-it at 122.11.227.227:6102 (docs said
  Qwen 27B/35B — server reality wins); live rubric sanity PASSED (good search
  1,1,1 vs redundant rephrase 0,0,0). Standing 2-GPU-hold rule active (watcher
  polling; all 8 cards busy). **Corpus corruption found & repaired:** rescued
  wiki-18.jsonl was a botched tar extraction (512-byte header fused to record 0,
  padding tail) — caught by the retrieval server's ntotal==lines assertion;
  streaming repair validated all 21,015,324 records (ids 0…21015323), atomic
  swap, offsets rebuilt. GPU stack pinned (vllm 0.25.1/torch 2.11.0).
