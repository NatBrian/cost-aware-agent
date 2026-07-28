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

- [x] E-a ✅ 2026-07-22: 200-episode pilot; knee at step 3; overthinking in
  24/200; budgets {2,4,8} + λ=0.3 FROZEN from data (memo:
  foundation/experiments/reports/pilot_memo.md; auto-approved under mandate)
- [x] E-b ✅ 2026-07-22: calibration gate PASSED (mean .848; all bits ≥.70;
  was_needed 1.0). Label-review round documented (4 author-label fixes, 4
  judge attention-misses kept+logged). Judge biases for divergence read:
  lenient-on-redundancy, late-step supported noise
  (report: calibration_report.md)
- [x] E-c ✅ 2026-07-22 (**dev-200 look #1 of 3**): 1400 episodes. A1 beats A2
  everywhere (paired); A2 B=2 ceiling confirmed (F1 .221 vs .478); A1
  budget-adapts ~1 step. RL bar at B=4: U>.205, F1≥.361, self-stop≥.70
  (report: baselines_report.md)
- [x] E-d ✅ 2026-07-23 (GATE PASSED): verl replaced by custom lean GRPO trainer
  (rationale in commits e4fe6b6/18542f1). Micro-round end-to-end: 124/124
  samples zero token mismatch; update-1 ratio 1.003 / KL 6e-4 (math verified);
  KL-blowup tuned (shuffle + accum 32); multimodal-tensor merge at save;
  trained checkpoint SERVES ('READY')
- [ ] E-e **INTERRUPTED by container wipe 2026-07-28** (was: round-1b retraining
  from base on cached rollouts, launched 2026-07-23). Restarting from base —
  see the 2026-07-28 log entry for the full inventory. 3 rounds ×
  (300 tasks × G=8 sharded collect → judge → train → serve ckpt)
- [ ] E-f: F6 eval (A3 harness-off/on × 3 budgets) + oracle replay → gate_check
- [ ] E-g: F7 figures + report + GO/NO-GO verdict → tag `foundation-run-1`

## Autonomy mandate (Brian, 2026-07-22)

Full autonomy to run E-a…E-g end-to-end and REFINE UNTIL STABLE (RL beats
baselines; PPO fallback if GRPO collapses; fix any issue; websearch when
knowledge-limited). Standing 2-GPU hold at all times, even between experiments.

**Anti-overfitting policy (self-imposed, active from now):**
1. dev-200 stays untouched during ALL refinement. Tuning/iteration decisions use
   a 50-task VALIDATION slice carved from train-300's remainder pool (never the
   frozen dev set, never the 300 training tasks themselves where avoidable).
2. Every dev-200 evaluation is logged in this file with date + reason; target
   ≤3 total dev evaluations (baselines once, headline A3 once, +1 reserve).
   Iterating against dev until the gate passes would BE overfitting — the gate
   must pass on the first or second dev look, driven by validation-slice signals.
3. Refinement order on failure (cheapest first): rubric/judge → reward scales
   (alpha, lambda) → RL hyperparams (lr, KL, epochs) → algorithm (PPO fallback)
   → data size (300→500). One change per iteration, logged.

## Log

- 2026-07-28 **CONTAINER WIPE — all gitignored runtime state lost.** The pod came
  back fresh (every file dated 14:39); nothing was running and all 8 GPUs were
  idle, despite the E-e line above claiming RUNNING. `/home/liangsheng` is an
  ephemeral overlay; only committed files survived.
  - **Lost:** `research/data_shared/` (79G — HotpotQA sources incl. the
    *decontaminated* train file, 21M-passage corpus, E5/FAISS index) ·
    `foundation/data/` (frozen train-300/dev-200 + SHA256 manifest) · `.venv`
    and the GPU venv (`.venv-gpu` is now an empty shell: pip only) · every
    trajectory JSONL (pilot, baselines, E-e rollouts) · the judge cache · the
    **round-1b checkpoint**.
  - **Survived (committed):** all code and docs · all three reports ·
    `baseline_rows.csv` (all 1,400 A0/A1/A2 rows) · `sheet_v1.csv` (the 50
    calibration labels *with* each row's exact judge context).
  - **Consequences:** (1) training restarts from base — E-d/E-e survive only as
    code and commit messages; (2) **dev-200 is exactly recoverable** from the 200
    HotpotQA ids in `baseline_rows.csv`, so the surviving baseline numbers stay
    valid and comparable, and the gate is unaffected; (3) **train-300 is NOT
    reproducible** — its decontaminated source is gone and the generator lives in
    the read-banned archive, so it will be re-derived (same seed, same
    stratification, own overlap check vs dev-200) and A3 will train on a slightly
    different 300 questions than the lost run. Logged as a deviation.
  - **Policy fixes adopted (Brian, 2026-07-28):** everything regenerable now
    lives on `/mnt/src/liangsheng/` (persistent CPFS) and is symlinked into the
    repo, so a future wipe costs one symlink; commit and push continuously rather
    than at stage boundaries; GPU holds via `/home/liangsheng/brian/acquire_gpus.py`
    (never takes a card another process is on), 2 held at all times.
  - Dev-look ledger unchanged: **1 of ≤3 used** (E-c baselines).
- 2026-07-28 **Judge changed: gemma-4-31B-it (:6102) → Qwen3.6-27B (:6101)**
  (Brian). This invalidates the E-b gate, which was measured against gemma — so
  calibration re-runs against the new judge before any RL. Cheap: `sheet_v1.csv`
  carries the labels and the exact context, so `agreement()` re-judges from the
  sheet (50 calls, no GPU, no pilot rerun). **Verified integration hazard:**
  Qwen3.6-27B emits chain-of-thought into `message.content` (not
  `reasoning_content`) — a bare "output only JSON" request returned "Here's a
  thinking process: …" and hit the token cap. Strict-JSON parsing would fail on
  every step and fall back to neutral 0.5, silently zeroing the per-step reward
  while training continued on the terminal reward alone. Fix:
  `chat_template_kwargs: {"enable_thinking": false}`, which returns clean JSON.
  Also note `max_model_len` 32768 (gemma had 262144) — late-step prompt lengths
  must be checked against it.
- 2026-07-23 REFINEMENT ITERATION 1 (ladder tier: RL hyperparams): round-1
  training (lr 5e-6, 250 updates, full-FT) damaged the SAMPLING distribution —
  temp-0 val-50 healthy (F1 .607) but temp-1.0 rollouts 71.6% malformed, U
  -0.29; round-2 was training on that garbage -> killed. Changes: lr 2e-6,
  kl_beta 0.1, 150-update cap, KL log-ratio clamp, + NEW post-round temp-1.0
  health probe gate (scripts/probe_policy_health.sh: malformed<10%, cap<15%).
  Round-1b retraining from base on the original healthy rollouts (cached).
  Judge client hardened (transport retries + neutral fallback) after a
  connection-reset crash under 12-thread load.
- 2026-07-22: plan docs reviewed doc-by-doc with Brian; implementation started.
- 2026-07-22 (E-phase): judge = gemma-4-31B-it at 122.11.227.227:6102 (docs said
  Qwen 27B/35B — server reality wins); live rubric sanity PASSED (good search
  1,1,1 vs redundant rephrase 0,0,0). Standing 2-GPU-hold rule active (watcher
  polling; all 8 cards busy). **Corpus corruption found & repaired:** rescued
  wiki-18.jsonl was a botched tar extraction (512-byte header fused to record 0,
  padding tail) — caught by the retrieval server's ntotal==lines assertion;
  streaming repair validated all 21,015,324 records (ids 0…21015323), atomic
  swap, offsets rebuilt. GPU stack pinned (vllm 0.25.1/torch 2.11.0).
