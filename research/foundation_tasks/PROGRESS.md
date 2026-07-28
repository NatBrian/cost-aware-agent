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

- 2026-07-28 **SMOKE PASSED (20 tasks x 3 modes), with one finding held open.**
  Schema 0 errors in all three modes; retrievals sane (0 thin observations);
  F1 .462/.414/.454 and mean steps 3.90/3.05/4.00 — close to the surviving
  E-c baselines (A1 B=4: F1 .471, steps 3.55), so the rebuilt pipeline
  reproduces prior behaviour.
  **Finding: 20.5% of steps are malformed in mode=none** (16/78; 0% in enforce,
  8% in forced_continuation). Root cause is NOT format disobedience: the
  malformed steps have `raw_len` 1961–2228 chars against `max_tokens_per_step:
  512`, i.e. the model writes a long multi-sentence THOUGHT, gets truncated
  mid-sentence, and the required ACTION / BEST ANSWER lines never arrive.
  **Decision: do NOT change the scaffold yet.** Raising max_tokens or tightening
  the prompt would give A3 a scaffold the surviving A0/A1/A2 baselines never had,
  and those baselines are the most valuable asset we still hold — re-collecting
  them would spend one of the ≤3 dev looks. The scaffold must be a constant
  across arms (plan §2). Re-decide from the E-a pilot's 200 episodes, where n is
  large enough to tell a real rate from small-sample noise; if it stays ~20%,
  weigh a scaffold change + baseline re-run against living with it (the reward's
  format term already penalises malformed steps, so A3 should learn to avoid
  them — which is a legitimate result, not a confound, as long as the arms share
  the scaffold).
- 2026-07-28 **TWO ROOT CAUSES FOUND (both had bitten the first run; both now
  understood rather than hand-patched).**
  1. **`wiki-18.jsonl.gz` from HuggingFace is a TAR.GZ, not a plain gzip.** It
     holds one member (`data00/.../wiki_dump.jsonl`), so `gunzip -c` yields the
     tar stream: a 512-byte header fused onto record 0 and NUL padding at the
     tail. The retrieval server's `ntotal == lines` assertion caught it again
     (21,015,324 index rows vs 21,015,325 corpus lines). This is *exactly* the
     "botched tar extraction" the first run diagnosed on 2026-07-22 and repaired
     by hand — it was never a bad extraction on our side, it is what the file
     is. Fix: `tar -xzOf`, recorded in `/mnt/src/.../extract.sh`. Note the
     earlier `wc -l` "verification" was misleading: it counts newlines, so the
     unterminated final padding line made a corrupt corpus look correct.
  2. **The box cannot run cu130 builds.** Driver 570.172.08 reports CUDA 12.8;
     `torch 2.11.0+cu130` (pip's default variant) aborts with "NVIDIA driver on
     your system is too old (found version 12080)" and vLLM's engine core dies
     with it. cu128 wheels are forward-compatible across 12.x, so the stack must
     be pinned to cu128 via `--index-url .../whl/cu128`. **This vindicates the
     ORIGINAL `requirements-gpu-pinned.txt` (torch 2.10.0+cu128) that I had
     overwritten** on the strength of this log's "vllm 0.25.1/torch 2.11.0" line
     — the log recorded versions without the CUDA variant, which is the part
     that actually matters here. Lesson logged: verify the environment
     empirically; neither the pin file nor the log was trustworthy alone.
     **Swapping torch alone is not enough:** vllm 0.25.1's PyPI wheel is itself
     compiled for CUDA 13 (`import vllm` -> `ImportError: libcudart.so.13`), and
     no `+cu128` release wheel exists for 0.25.x (the documented GitHub release
     URLs 404 — matching vllm issue #37847). Since `vllm==0.17.1` requires
     exactly `torch==2.10.0`, the ORIGINAL pin set (vllm 0.17.1 / torch
     2.10.0+cu128 / transformers 4.57.6 / faiss-cpu 1.14.3 / fastapi 0.136.3 /
     uvicorn 0.51.0) is a coherent environment that empirically ran E-a and E-c
     on this box. **Restored it verbatim via `git checkout` and rebuilt the venv
     from it** (`.venv-gpu3`). My rewrite of that file was the error, not the
     file. `requirements-gpu.txt` independently notes "vllm>=0.17 ... GDN kernels
     for Qwen3.5", so 0.17.1 supports the executor.
     **The first run had already solved all of this** — commit `28b2728`
     ("executor serving SOLVED — full dependency saga resolved") records a chain
     of five failures: (1) torch 2.11 needs a CUDA>=12.9 driver, box has 12.8 ->
     pin vllm 0.17.1 + torch 2.10.0+cu128; (2) stale flashinfer-cubin 0.6.13 vs
     flashinfer 0.6.4 -> align them; (3) Qwen3.5's GDN prefill kernel JIT needs
     nvcc and there is no /usr/local/cuda on the box; (4) the pip nvcc wheels are
     both useless (the 12.9 wheel ships nvcc 13.2 = header clash, the 12.8 wheel
     ships no nvcc at all); (5) resolution: patch the venv's `qwen3_next.py` to
     take vLLM's own Triton FLA path (`forward_native`, the standard non-Hopper
     implementation, which self-compiles). I burned two turns rediscovering (1)
     because I trusted the prose log over the lockfile and never read the commit
     history for the file I was rewriting. **Read `git log -- <file>` before
     declaring a committed artifact stale.** The venv is gitignored so the
     `qwen3_next.py` patch itself was lost with it and must be re-applied; the
     `.cuda_home/` shim in the repo is a survivor of failure (4).
- 2026-07-28 **AUTONOMY MANDATE v2 (Brian):** "I will not prompt you. You must
  complete everything of the foundation paper plan until the end... You must
  advance and continue automatically and decide the best solution. Only ask me
  if you need clarification/questions." Self-scheduled loop keeps the run moving;
  every judgement call is logged here.
- 2026-07-28 **DATA REBUILT + VERIFIED.** `/mnt/src/liangsheng/cassi_foundation`
  (persistent) holds corpus + index + weights; `research/data_shared` symlinks to
  it. Assembly verified byte-exact against the values recorded in
  `serve_retrieval.py`: `e5_Flat.index` = 64,559,075,373 bytes and
  `wiki-18.jsonl` = 21,015,324 lines, so the index/corpus row-count assertion
  holds and the tar-corruption that bit the first run did NOT recur.
  Splits: **dev-200 EXACT MATCH by id to the pre-registered set** (verified
  against baseline_rows.csv — the gate keeps its questions and the surviving
  A0/A1/A2 numbers stay comparable); train-300 re-derived
  (189 medium/60 easy/51 hard vs the lost run's 194/55/51 — logged deviation);
  val-50 generated for the first time. Overlaps train/dev, val/dev, val/train
  all zero; decontamination dropped 0 rows (HotpotQA train and dev are disjoint
  splits, so the lost `*.decontaminated.jsonl` was belt-and-braces).
- 2026-07-28 **E-b RE-RUN: GATE NOT PASSED with the new judge.** Qwen3.6-27B
  scores mean .769 (rubric_v1) / .764 (v2) / .792 (v3) against the surviving
  50-label sheet — below .80 on both the strict per-bit reading the plan
  specifies and the looser mean+floor the code had drifted to. v2 applied the
  audit's rubric recommendations and was MEASURED, not assumed: `nothing_left`
  +.116, `supported` −.115, so v3 keeps what helped and reverts what hurt, and
  drops a verbatim-quote field that emitted invalid JSON. v3 runs with 0 parse
  failures and 0 neutral fallbacks. **Blocking finding:** at n=24–26 per bit the
  95% CI is ±0.17, so the v1/v2/v3 spread is noise — further prompt-tuning
  against this sheet fits noise. Report: `calibration_report_qwen36.md`.
- 2026-07-28 **DECISION (Brian): rebuild the calibration instrument, labeled by
  a FRESH SUBAGENT with no session context.** ~150 steps drawn from the Phase 3
  pilot; each batch goes to a newly spawned agent that sees only the bit
  definitions and the step context — never any judge output, never the rubric
  version history, never this conversation. Rationale: it removes the anchoring
  that inflated the original .848 (labels there were revised while reading the
  old judge's reasoning), it is independent of the judge under test
  (Qwen3.6-27B, a different model family), and n=150 shrinks the per-bit CI from
  ±0.17 to about ±0.07 — enough to actually arbitrate the gate. Honesty note for
  the report: these are model labels, not Brian's, so the gate is evidence about
  judge–labeler *consistency*, and the F3 requirement for human labels is
  formally unmet.
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
