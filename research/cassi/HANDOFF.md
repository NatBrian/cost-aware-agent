# CASSI Implementation — Handoff for the Next Agent

**Written 2026-07-16, updated same day after the execution session; updated 2026-07-21 for
plan v2.1.** The spec is **`research/paper_plan_v2_1.md`** (source of truth — an ADDITIVE
revision of paper_plan_v2.md; every change tagged [v2.1], §20 is the changelog; §16 is the
runbook). This document tells you what is already built/executed, what is pending, and the
exact next commands. Read `research/paper_plan_v2_1.md` §0, §2, §12, §16, §17, §20 before
touching anything. Design context for v2.1 (the two-reward-model agreement):
`research/architecture_comparison.md`.

**Per user instruction: never read anything under `research/archived/`** (stale AI outputs; that single directory is the whole ban).

---

## 1. Current state, in one paragraph

**Updated 2026-07-22 after the first real experiment run — read §1b for the full session
log.** Executed and DONE on this clone: P0 smoke (PASS), P2 pilot (wallet calibration
FROZEN in cassi.yaml) + full QA collection round 0 (9,600 forced-continuation
trajectories), P3 labels (96K steps × 5 λ, all QC passed, λ-dial monotone 5.0→1.0),
and the RM-P prompted-judge held-out evaluation (B10 preview — prompted 35B LOSES badly
at the neutral operating point). P4 stopper SFT was started but **KILLED BY USER ~7h in,
before the first checkpoint** — no trained stopper exists yet; that is the resume point.
Plain-language results: `experiments/reports/`. The paragraph below describes the
original 2026-07-16 code+staging state and remains accurate.

ALL code is written and verified (125 CPU tests pass in the venv: `python -m pytest tests/ -q`
from this dir — count raised from 115 on 2026-07-21 by the v2.1 B10 module's 10 tests),
**both** former wiring gaps are CLOSED, and phases **P0 and P1 are executed**:
the §19 stack is cloned+pinned (hashes in `configs/cassi.yaml pins:`) into a dedicated venv
(`.venv` — auto-activated by `scripts/common.sh`), all datasets are downloaded/decontaminated/
manifested, the wiki-18 retrieval corpus + assembled 64GB E5 index sit in
`data/searchr1_index/`, and Qwen3.5-9B/2B weights are prefetched in the HF cache. The verl
integration (`executor/verl_hooks.py` + full `train_grpo` CLI) dry-runs green against the
pinned verl commit. Core math is validated on synthetic data (hand-computable τ*;
λ-monotonicity; PBRS telescoping; B9≡CASSI when V̂≡V*). **No model is trained and no
experiment has run — every GPU step is still pending** (the machine's GPUs are fully held by
another user's training job; the user has said GPUs cannot be used for now).

## 1b. SESSION LOG 2026-07-21→22 — what was done, exact current state, what next

### Done, in order (each with its artifact)

| # | What | Result / artifact |
|---|---|---|
| 1 | Plan v2.1 + B10 module + config + 10 tests (see §1 note) | `paper_plan_v2_1.md` (§20 changelog), `baselines/b10_prompted_rm.py` |
| 2 | Full staging on THIS clone (see MACHINE NOTE in §3) | datasets, 61GB index, weights, GPU venv |
| 3 | **P0 smoke: PASS** — 1 rollout, 10 steps, $0.0098, drafts + costs verified | `experiments/smoke/qa.jsonl`, report 00 |
| 4 | **P2 pilot: DONE** — 200 unconstrained tasks; wallets FROZEN in cassi.yaml (qa: small 0.00182 / med 0.00988 / large 0.04043; median 0.00349). NEVER recalibrate | report 00 |
| 5 | **P2 collection round 0: DONE** — 1,200 tasks × G=8 = 9,600 forced-continuation trajectories, balanced wallets (382/417/401), $143.08 total, 46% forced-continuation overhead (feeds T4), 68% self-answered somewhere | `experiments/collect/round0/qa.jsonl` (merged from 8 parallel shard collectors, seeds 44–51 — SHARDED, so wallet draws differ from a single-seed run; shard files kept), report 01 |
| 6 | **P3 labels: DONE, all QC passed** — 96K steps × λ∈{0.1,0.5,1,2,5} × two economies. λ-dial mean τ*: 5.0/3.4/2.8/1.1/1.0 (tier-scaled; plain-λ A8 arm also built). QC: 0/38,400 monotonicity violations; noise |dτ*|=0.036; prophet-argmax stops **+0.41 steps later** (foresight bias CONFIRMED — E4 evidence). Caveat: EM/F1 alias noise ("DEA" vs "Drug Enforcement Administration" → 0.0) | `experiments/labels/round0/`, memo + `qa_review_20.jsonl` (manual review TODO still open), report 02 |
| 7 | **RM-P (prompted 35B judge) held-out eval: DONE** — B10 preview on the SAME split/metric as the P4 gate, 150 held-out trajectories, λ=1, θ_p=0.5: mean regret **5.60**, stops at step **6.43** vs optimal 2.79 (≈4 steps too late), STOP F1 0.52 (precision 0.91 / recall 0.37 — conservative+miscalibrated, not pure noise), 2.2% parse failures, $0.75 total. Same-sample trivial baselines: majority-class regret ≈ **−0.02** (λ=1 makes always-stop-at-1 near-optimal — single-λ slice only; the REAL gate pools all λ where always-stop dies), draft-stability probe regret 1.13 | `experiments/stopper/round0/rmp_heldout_qa_lam1.json`, `trivial_baselines_sample150.json`, `scripts/eval_rmp_heldout.py` |
| 8 | **P4 stopper SFT: STARTED, then KILLED BY USER** (2026-07-22 ~17:10) ~7h in, BEFORE the first checkpoint — **no stopper .pt exists**. Root causes of slowness: no nvcc ⇒ GDN fast kernels unavailable ⇒ torch fallback; seq 2048 × batch 64 × 480K examples | fix landed along the way: `stopper/model.py` reads width from `get_input_embeddings().embedding_dim` (Qwen3.5 config nests hidden_size under text_config) |

### Exact machine state at handoff

- **GPU hold ACTIVE**: locks on 6,7 (`/tmp/gpu_lock_{6,7}`, holder pid in
  `/tmp/cassi_gpu_hold.pid` — verify with `scripts/gpu_hold.sh status`; locks are
  COOPERATIVE only, user `yongyue` ignores them and cycles TP servers across all 8 cards;
  wait for several-minute-stable free windows, never race their loads).
- **torch retrieval server UP** on GPU 6 port 8000 (`scripts/torch_retrieval_server.py` —
  custom, because CPU faiss = 35 s/query and pip faiss-gpu has no L20X kernels; 46 ms/query,
  imports the pinned Search-R1 encoder unmodified). Client MUST send `return_scores: true`
  (upstream unpack bug, fixed on our side in `executor/envs/searchr1_qa.py`).
- **executor vLLM server DOWN** (was killed externally). Known-good relaunch:
  `CUDA_VISIBLE_DEVICES=<card> VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/python -m
  vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-9B --port 8901
  --gpu-memory-utilization 0.85 --gdn-prefill-backend triton` (both flags REQUIRED — no
  nvcc on this box; port 8001 is squatted by a foreign 404 service).
- Shell footgun that bit 3×: `pkill -f <pattern>` matches YOUR OWN wrapper shell if the
  pattern's text appears anywhere in your compound command — run pkill alone, bracket-trick
  the pattern, and never put the relaunch in the same command.

### WHAT NEXT (ordered; user must green-light training restarts — they said "stop our training")

1. **Retrain the stopper FAST, then the P4 gate.** Before rerunning
   `scripts/p4_stopper.sh 0`, cut the wall-clock ~10×: `--max-seq 512` (serialized states
   are ~400 tokens; check train_sft CLI, else edit §17 `stopper.sft.max_seq`), raise batch
   to fit, and consider subsampling train examples (~150K of 480K). Expected ~1–2 h on one
   L20X. The gate then auto-runs `p4_gate.py` (pooled multi-λ — the fair test vs
   majority-class; see #7's single-λ caveat before interpreting).
2. **Write report 03** (`experiments/reports/03_stopper.md`): the four-way table —
   trained 2B vs prompted 35B (#7) vs majority vs probe, same held-out exam. Template
   expectations and caveats are all in #7's row.
3. **P5 kill-switches K1/K2** (`scripts/p5_killswitch.sh`, §12): needs the trained stopper,
   the vLLM server back up, ~4 GPUs for GRPO (verl), and **K2's `--arm single_multitask`
   is still a NotImplementedError stub — implement BEFORE P5** (§6). K1 GO/NO-GO decides
   the paper; log to `GO_NO_GO.log` (not yet created).
4. Then per §3 queue: P6–P9, B10's two proper arms at P8 (frontier θ_p sweep — #7 was one
   operating point only), ALFWorld track (needs verl-agent env + its own pilot), GAIA
   (gated, needs HF login).

### Review pointers (for a reviewing agent)

Plain-language reports: `experiments/reports/README.md` → 00/01/02. Senior-facing summary:
`research/progress_report_2026-07-22.md`. Raw artifacts: paths in the table above. Logs:
scratchpad `smoke_pilot3.log`, `p4b.log`; server logs `experiments/logs/`. Every §16
done-criterion met so far is checkable from those files.

## 2. What is implemented (module map + test coverage)

| Where | What | Verified by |
|---|---|---|
| `common/schema.py`, `common/config.py` | §11 trajectory JSONL schema; §17 config loader + pilot-calibration guard | `tests/test_core.py` |
| `budget/cost.py` | The ONE economy (§2.2/§2.4): token/tool pricing, tiers, m(tier), wallets, pilot normalization, U_t, R_base; reuses the repo harness price map for API models | `tests/test_core.py` (U_τ ≡ R_base) |
| `labels/snell.py` | **Algorithm 1** (LightGBM backward recursion, Δ*, τ*, V*, tanh scale, backup residuals) + prophet labels (E4 arm) + QC (λ-monotonicity) | `tests/test_core.py` |
| `labels/quality.py`, `labels/drafts.py` | F1/EM/subgoal scoring; running-draft parse + stability features (§2.6/§18.2) | `tests/test_core.py` |
| `stopper/features.py` | §18.1 serialization (λ-conditioning) + numeric vector for the label regressor | `tests/test_core.py` |
| `stopper/dataset.py`, `model.py`, `train_sft.py`, `eval_regret.py` | SFT examples (pooled λ, split-by-task), 3-head model (Alg. 2), custom training loop (early-stop on held-out REGRET), regret eval + P4 gate baselines | `tests/test_stopper_cpu.py` (15) |
| `executor/react_agent.py`, `collect.py`, `monitor.py`, `shaping.py`, `train_grpo.py`, `envs/`, `vllm_client.py` | Shared scaffold (both §2.1 rollout modes), forced-continuation collection + wallets per (task, group), Alg. 4 monitor (fixed Δ̂≤0 + A8 mode + internalization tracking), PBRS + step-level RTG advantages + min-cohort guard, verl config builder + adapter | `tests/test_executor_cpu.py` (29 incl. verl dry-run), `tests/test_integration.py` (2), `tests/test_run_frontier_cpu.py` (4) |
| `baselines/` (b1–b10 + oracle) | One module per §5.2 row with registry, cost knobs, reward logic. **[v2.1] b10_prompted_rm** = "RM-P" prompted-judge baseline (CoT + designed binary rubric on the §18.1 serialization; continue-score = Δ̂ analog for the monitor arm, state-value = V̂ analog reusing `executor.shaping` for the rl arm; fail-open parsing; θ_p dev calibration; billing via the shared price map; `VLLMJudgeAdapter` for the lab 30B server). Config keys `prompted_rm` + `rl_algo_pilot` added to cassi.yaml; p8_baselines.sh runs the b10 monitor arm with the inference-only set | `tests/test_baselines_cpu.py` (36) |
| `eval/metrics.py`, `stats.py`, `overhead.py` | Frontier protocol + interpolation, regret, matched-risk, §5.6 stats (small-n guard ENFORCED), T4 ledger + serving regimes + billing symmetry (enforced) | `tests/test_eval_cpu.py` (20) |
| `analysis/` + `Makefile` | One script per F1–F6/T1–T5, CSV→PDF/tex, CVD-safe | `make figures tables` (skips gracefully pre-P9) |
| `scripts/` | P0–P9 runners; `p5_killswitch.sh` writes `GO_NO_GO.log`; GPU acquire/release with EXIT traps | shellcheck-style review; python drivers tested |
| `paper/` | main.tex + 8 §9-mapped section stubs + references.bib (41 seeded entries) + Makefile | compiles: `cd paper && make` |

## 3. PENDING — ordered work queue (this is your job)

Everything below needs GPUs and/or network downloads. **GPU protocol for this machine
(CLAUDE.md):** `eval $(/mnt/src/zhanka/gpu_acquire.sh N)` before, `/mnt/src/zhanka/gpu_release.sh`
after; N=2 for collection/stopper SFT, N=4–8 for GRPO. Never kill occupier processes.
**HOLD MODE (user policy 2026-07-21): keep GPUs held BETWEEN phases** —
`scripts/gpu_hold.sh start N` acquires once (via the sanctioned acquire script — flock'd,
skips others' locks/busy cards, never preempts) and a zero-footprint holder process keeps
the locks valid; every phase script then REUSES the held cards automatically (common.sh
detects the hold and skips its acquire + EXIT-trap release). `gpu_hold.sh status|stop`;
stop releases ONLY the held ids (NEVER the bare user-wide gpu_release.sh — the OS account
is shared with another person whose locks that would also drop). If a phase needs more
GPUs than the hold has, it exits with a clear pending message (stop + re-start with more;
never mix hold + extra per-phase acquires). A read-only availability watcher may already
be running in the user's session; on its notification: `scripts/gpu_hold.sh start 2` →
`scripts/smoke_and_pilot.sh`.

**MACHINE NOTE (2026-07-21): full staging was REDONE on the `/home/liangsheng/brian` clone**
(the original 2026-07-16 staging lives in a root-owned clone this account cannot read).
Now present here: GPU venv (pinned stack + verl deps incl. ray/tensordict/numpy<2 +
faiss-gpu), datasets (staged/decontaminated/manifested; GAIA still gated-pending),
assembled `data/searchr1_index/` (e5_Flat.index 61GB + wiki-18.jsonl 14GB), Qwen3.5-9B +
intfloat/e5-base-v2 in the HF cache. **Serving gotchas fixed live:** this box has NO CUDA
toolkit (no nvcc) → every flashinfer JIT path crashes vLLM: launch with
`VLLM_USE_FLASHINFER_SAMPLER=0` and `--gdn-prefill-backend triton` (both baked into
smoke_and_pilot.sh); executor port is 8901 (8001 is squatted by a foreign 404 service);
`--disable-log-requests` no longer exists in vLLM 0.19. **GPU contention:** user `yongyue`
cycles TP4 vLLM servers across ALL 8 GPUs ignoring the zhanka locks — do not launch onto a
card that just freed (their next cycle may land on it mid-load; happened twice); wait for a
several-minute-stable free window, and prefer human coordination for a fixed card split.

1. ~~P0 — installs~~ **DONE** (2026-07-16) except the GPU smoke rollout. Stack pinned; venv
   ready. NOTE: verl-agent and verl both claim the `verl` package name — the venv resolves to
   the PINNED verl (enforced install order in p0_setup.sh); ALFWorld's verl-agent harness
   therefore needs its own env or PYTHONPATH staging when that domain starts.
2. ~~P1 — data~~ **DONE** (2026-07-16): datasets + frozen subsamples staged in `data/`,
   decontamination ran (3 train items dropped), manifest committed, wiki-18 corpus + 64GB
   E5 index assembled in `data/searchr1_index/`. Leftovers: BrowseComp-Plus corpus staging
   (documented in p1_data.sh); GAIA text-only staged at 127 rows — the papers' 103-subset
   needs the annotator-metadata tool filter, verify before E2 (SupervisorAgent comparability).
3. **FIRST GPU SESSION → `scripts/smoke_and_pilot.sh`** (everything pre-staged): acquires 2
   GPUs, launches retriever + vLLM servers, runs the P0 smoke rollout + verify, then the
   200-task P2 pilot, and prints the wallet calibration → **write the printed values into
   `configs/cassi.yaml`** (they are `null` now; later phases refuse to run until filled).
   FOOTGUN: `gpu_acquire.sh` can grant locks while a foreign job still holds GPU memory —
   the script checks actual memory and aborts safely (never kill other users' jobs).
   Then `scripts/p2_pilot_and_collect.sh` for the full round-0 collection (G=8) — despite
   its name it SKIPS its pilot stage when calibration is already frozen in the config
   (verified: it checks `require_pilot_calibration` per domain), so it will not re-derive
   different percentiles after smoke_and_pilot.sh. Running it FIRST (nulls still in
   config) is also safe: its stage (a) runs the pilot itself, prints the values, and
   stops before collection until you write them into the config — the scripts differ in
   that smoke_and_pilot.sh additionally does the P0 smoke + server bring-up.
4. **P3 — labels** (`scripts/p3_labels.sh`): Algorithm 1 per λ ∈ {0.1,0.5,1,2,5} + QC memo.
5. **P4 — stopper v0** (`scripts/p4_stopper.sh`): SFT + the HARD GATE (beat majority-class AND
   the confidence probe on held-out regret, else STOP and fix features/labels).
6. **P5 — KILL-SWITCHES K1/K2** (`scripts/p5_killswitch.sh`, §12): the GO/NO-GO moment on
   HotpotQA-1K, 1 seed, λ=1.0 (the headline default). **Prerequisite: K2's
   `--arm single_multitask` is still a NotImplementedError stub — implement it BEFORE this
   phase (ideal no-GPU work, see §6). K1 alone must NOT gate a GO** unless the user
   explicitly approves that scope cut. K1's shaped arm runs BOTH step-credit variants
   (per_step_rtg and shape_segment) as sub-arms; the GO test applies to the better one and
   both results go in the log. Decision formula (exact arithmetic, as implemented in
   `scripts/killswitch_decision.py`): GO iff RELATIVE cost reduction
   `(cost_ctrl − cost_shaped)/cost_ctrl ≥ 0.03` at iso-accuracy AND
   `cost_shaped ≤ cost_B9` at iso-accuracy (tie passes). Iso-accuracy costs come from
   each arm's 3-point mini-frontier — every K1 arm is evaluated at inference λ-dial
   ∈ {0.5, 1.0, 2.0} (training stays at λ=1.0) so interpolation is possible at
   kill-switch scale. λ note: 1.0 is the PROVISIONAL headline; K1's GO stands even if
   the dev-chosen headline λ later differs (K1 is a direction+magnitude test, not
   λ-specific) — note any such difference in the log. Decision appended to
   `GO_NO_GO.log` (freeform dated UTC entries, written by killswitch_decision.py and
   by hand for later scope decisions) — never delete or rewrite past entries (§5.6
   no-cherry-picking). The log is CREATED at the first documented decision, whatever
   phase that is (an early P2 wallet-recipe deviation entry is fine).
7. **P6–P9** per §16 (iteration 1, loop iteration 2 with frozen-coach control, baselines,
   full eval incl. 500-task regret replays, 3 seeds on headline points). **Before P8:**
   read `research/lit_review/` for the baseline papers' actual mechanisms (B4/B5/B8
   "align to the official repo" decisions cannot be made from these docs alone).
   **[v2.1] B10 "RM-P" prompted-judge baseline (two arms, §5.2/§20):**
   (a) *monitor arm* runs with the P8 inference-only set — prerequisite: fill
   `prompted_rm.base_url` in cassi.yaml with the lab vLLM Qwen3.5-30B endpoint (the
   senior's server; frozen, never trained) and calibrate θ_p on dev via
   `b10_prompted_rm.calibrate_threshold` (write it into `prompted_rm.arms.monitor.threshold`);
   (b) *rl arm* — executor GRPO with the judge's state-value as Φ (same shaping machinery,
   only the potential source differs) — ONLY after a logged K1 GO, 1 seed, qa domain,
   1 frontier point; log the judge-score-vs-true-reward divergence (the F6 analog).
   The same 30B server is also the E2 stopper-as-monitor frozen agent and the optional
   P3 review pre-screener (v2_1 §19) — one deployment, three inference-only roles.
8. **P10–P11**: `make figures tables`, then write the paper into `paper/sections/`
   (writing order and claims-audit rule: §16 P11; every §14-dead-claim is banned).

### Former wiring gaps — BOTH CLOSED (2026-07-16)

- ~~run_frontier CLI~~ **DONE**: `eval/run_frontier.py` — full P5–P9 evaluation entry point
  (frontier summary rows + per-instance CSVs for the stats layer, billing symmetry, dual-run
  regret with the replay billed to the analysis line, `--regret-from-replays` offline mode).
  Tested in `tests/test_run_frontier_cpu.py`.
- ~~verl plumbing~~ **DONE**: `executor/verl_hooks.py` + full `train_grpo.py` CLI
  (`--tasks/--coach/--arm/--lambda/--step-credit/--max-steps/--init/--out/--dry-run`, §16
  contract) — custom `CassiAgentLoopManager` (group-level V̂ rewards), registered
  `cassi_step_level` adv estimator (difference-encoding on step-final tokens; decode proof in
  the module docstring), Dr.GRPO keys per the pinned commit, every touchpoint carrying a
  `# pin:` file/line reference. `--dry-run` is the regression check after any verl change.
  Known NotImplemented stubs (deliberate): `--arm single_multitask` (K2's A2 machinery) and
  the ALFWorld agent loop (verl-agent fork is API-incompatible with the pinned AgentLoop —
  needs its own env). Stopper V̂ serving defaults to CPU (`CASSI_STOPPER_DEVICE`).
  Logging quirk: verl's batch "reward" metric shows A₁ per trajectory (encoding artifact);
  TRUE economic rewards stream to `<out>/divergence.csv` (feeds F6); val batches carry the
  real terminal reward.
- `collect.py` re-collection with a trained policy: serve the checkpoint with vLLM and pass
  `--vllm-url` (no `--policy` flag; documented in p7/p9).
- P9's eval emitters log per-instance results per knob point — `run_frontier.py` already
  writes `*_instances.csv`; keep it that way (the stats functions consume per-instance
  matrices, not aggregates).

### Decisions made during implementation (so you don't re-litigate)

- §17 named a single-head TRL recipe but §2.3 needs three heads → custom torch loop at the
  same hyperparameters (`stopper/train_sft.py` docstring).
- §18.1 needs nominal caps the plan never fixed. Current state (KNOWN wrinkle — align at
  P4): the CANONICAL serialization caps are `stopper/dataset.py: SerializeContext` defaults
  (tokens_max=8192, tool cap=20) and `HFStopperPredictor` matches them — training and
  value-head inference are consistent. `executor/react_agent.py: DEFAULT_TOKENS_MAX=32768`
  is only the denominator for the `tokens_pct` FEATURE, not a serialization cap. The
  monitor's TEXT path serializes with its own caps — do not use the text path with the real
  stopper until the caps are unified into one config key (task in §6 queue, item 6).
- Alg. 4 "budget exhausted" quantified as spent ≥ allowance (both monitor modes, any k).
- §12 "≥3 points" interpreted as percentage-point cost reduction at iso-accuracy
  (`scripts/killswitch_decision.py`).
- B4 uses OTC's ratio reward form; B5 implements the solve-rate-scaled penalty family (EAPO
  primary / agentic-ALP fallback share the form); B6 defaults to flat-λ (published CTA form)
  with a `tier_scaled` fairness variant. All disclosed in module docstrings.
- Monitor accepts BOTH stopper protocols: feature-based `predict(x, λ)` = the value-head
  variant = THE DEFAULT in every phase (trained checkpoints load into it via
  `stopper/model.py: load_predictor`; K1 and all E-runs use it); text-based
  `evaluate(serialized_x)` exists for the generative variant (§18.3, ablation territory)
  and for this module's test mock. If unsure, you want `predict`.
- `LabelSet` persistence lives in `stopper/dataset.py` (`save_labelset`/`load_labelset`).
- `references.bib`: every entry must be verified against the real paper at P11 (authors
  marked TODO where the plan gave only arXiv ids).

## 3b. Post-experiment adjustment map — expected tuning once real numbers exist

These are NOT bugs. They are places where a value was chosen a priori and the first real data
will tell you the right one. Rules of the road: (i) anything frozen "after P2" changes ONLY
via a rerun of the freezing step, never by hand-editing downstream; (ii) the shared agent
scaffold and the economy are FAIRNESS INVARIANTS — see the "never change" list at the end.

**After the smoke run (first hours):**
- **Draft-line compliance** is the single biggest unknown: the whole label machinery assumes
  the model emits "BEST ANSWER SO FAR:" every step. Check `format_score` in the smoke JSONL —
  it lives in `Trajectory.outcome["format_score"]` (fraction of the episode's steps whose
  output contained a parseable draft line, 0.0–1.0); "compliance < 95%" means the mean
  format_score across smoke episodes is below 0.95. FREEZE BOUNDARY: the template must be
  final BEFORE the P2 pilot, not merely before collection — wallet dollars derive from
  pilot spend, and a template change changes spend. If the template changes after a pilot
  ran, the pilot MUST be rerun (cheap: 200 tasks) before its values are frozen.
  If compliance < ~95%, tune `SYSTEM_TEMPLATE` in `executor/react_agent.py` (add a few-shot
  example) BEFORE any collection — and then freeze it forever (all methods share it; changing
  it after any baseline has run invalidates the comparison).
- Action-parsing robustness (`search[...]` / `answer[...]`): expect to harden the regexes
  against real model output quirks. Same freeze rule applies.
- Qwen3.5 chat template: verify `<think>` is actually stripped and multi-turn is
  token-in-token-out (§19 stack note); adjust `vllm_client.py` chat_template_kwargs if not.

**After the P2 pilot (calibration step):**
- Fill `label.allowances` + `cost_normalization` (the script prints them). ALSO sanity-check:
  - **Do episodes actually traverse tiers?** If most episodes end still in HIGH, the wallets
    are too loose and tier-scaled costing is inert → tighten allowances (P25/P75/2×P90 is the
    default recipe, not scripture — document any deviation in GO_NO_GO.log).
  - **Does cost have variance beyond token count?** §17 tool fees are nominal; if retrieval
    fees are negligible vs tokens, the "multi-dimensional cost" claim weakens — consider a
    fee-scale sweep note, or report the dollar decomposition honestly in T4.
  - T_max=10 (QA): if many pilot episodes are still improving at step 10, raise it — but
    T_max changes invalidate any existing labels (rerun P3).
- `stopper/features.py` nominal caps (DEFAULT_TOKENS_MAX=32768, tool cap=T_max) → set from
  observed pilot distributions; add the chosen values to §17.

**After P3 (labels):**
- `fit_delta_scale` (tanh s) refits per domain — check the P3 memo's Δ histogram actually
  uses tanh's responsive range; if Δ* piles up at ±1, the scale heuristic (P90 of |Δ|) needs
  revisiting.
- Backup residuals (definition: at each backward step t, Algorithm 1 holds out ~10% of the
  cross-section and records the held-out mean |Ê[V_{t+1}|x_t] − V_{t+1}| — stored on
  `LabelSet.backup_residuals`; it measures how well the label regressor generalizes at that
  step) rising toward early steps = the LightGBM cross-section is too weak at
  late/rare steps → try pooled regressor with t as a feature (one-line change in snell.py's
  fallback logic) — this is E4-relevant, document either way.
- λ-monotonicity violation rate > 5% → feature leakage or regressor noise; STOP and debug
  before training the stopper on those labels.

**After P4 (stopper v0):**
- STOP/CONTINUE class imbalance (early steps are overwhelmingly CONTINUE): if stop-F1 is poor,
  add class weighting to the CE head (train_sft.py) — regret, not F1, remains the gate metric.
- If the 2B stopper underperforms the lightgbm label regressor on held-out regret, the text
  serialization is losing information vs the raw feature vector — check §18.1 formatting
  precision (rounding) before concluding "learned stopper doesn't work".

**During P6 (GRPO) — watch items:**
- The **difference-encoding** trick in `verl_hooks.py` (advantages on step-final tokens)
  depends on exact step→token alignment; the round-trip test covers it, but ANY verl upgrade
  or tokenizer change requires rerunning `--dry-run` + `tests/test_executor_cpu.py` first.
- Stopper V̂ serving is CPU by default (`CASSI_STOPPER_DEVICE`); if rollout throughput tanks,
  co-locate it on a GPU or raise the A5 every-k. `cuda:N` indexes into the ACQUIRED visible
  set (`CUDA_VISIBLE_DEVICES` remaps device ids) — `cuda:0` is the first acquired GPU, never
  physical GPU 0; you cannot accidentally address a foreign GPU while the env var is set.
- Watch `divergence.csv` (F6) from step 0: rising V̂-vs-reward divergence = the documented
  hacking channel — the §2.4 response is a stopper refresh, not a training restart.
- `--step-credit shape_segment` vs `per_step_rtg`: K1 picks; if SHAPE-segment wins, its
  implementation gets promoted from variant to default (one config key).

**Baseline alignment (P8):**
- B4 (OTC ratio form) and B5 (EAPO family form) were implemented from the papers' text; when
  official code differs, align to the official repo and note the change in each module
  docstring. B3's trigger sensitivity was designed a priori — calibrate its knob on dev like
  every other method (frontier protocol handles this).

**NEVER change after their freeze points** (reviewers can void the paper otherwise):
frozen dev/test subsamples (P1); the shared scaffold + draft template (after first baseline
run); the economy definition U_t/R_base (after P3); headline λ chosen on dev before test
(§5.6); GO_NO_GO.log entries (append-only). Refactors that are safe anytime: unifying the two
MockStopper classes (executor/monitor.py vs stopper/model.py — documented duplication), a
`--policy` flag for collect.py, a dedicated venv for the ALFWorld/verl-agent phase.

## 4. Schedule & budget reality (plan §7)

~30–35 training runs, 10–12 weeks on the 8×H200 node. K1/K2 (week 2–3) come FIRST — do not
build past P5 without a logged GO. Competitors are ≤3 weeks old at review time (DASH,
OS-Pruner); if K1 passes, consider the workshop-preprint hedge (§8).

## 5. Quick verification that nothing rotted (THE canonical ritual — same as PROJECT_GUIDE §7)

```bash
cd research/cassi && source .venv/bin/activate
python -m pytest tests/ -q       # expect: 125 passed IN THE VENV ("124 passed, 1 skipped"
                                 # = wrong env: outside the venv, verl is missing and the
                                 # dry-run test skips). Count history: 115 → 125 on
                                 # 2026-07-21 (v2.1 B10 tests). Update this count (and
                                 # PROJECT_GUIDE §7's) in the same commit whenever new
                                 # tests land.
python -m cassi.executor.train_grpo --dry-run \
    --config configs/cassi.yaml --domain qa                   # expect: "[dry-run] OK"
(cd paper && make)                                            # expect: main.pdf builds
git config user.name                                          # expect: Nathanael Brian
```

Do NOT re-run `scripts/p0_setup.sh` as a check — it is a network installer; P0 is DONE
(§3 queue item 1). Re-running it against moved upstream repos risks silently disturbing the pinned
stack (the "accidental upgrade" failure mode in PROJECT_GUIDE §17). Re-pin only
deliberately: update `pins:`, then rerun this whole ritual.

## 6. While GPUs are blocked — the sanctioned CPU work queue (in order)

The user has said GPUs cannot be used and must not be waited on. Useful work that is
explicitly sanctioned meanwhile:
1. **Close the K2 stub**: implement `--arm single_multitask` in `executor/train_grpo.py`
   (9B one-model multi-task comparator, A2/K2 machinery) — P5 NEEDS it (see §3 queue item 6).
2. Stage the dedicated ALFWorld env (verl-agent clashes with verl over the package
   name — separate venv or PYTHONPATH staging; see §3 queue item 1 note).
3. BrowseComp-Plus corpus staging (network/disk only, no GPU — large download, fine).
4. Resolve the GAIA exact-103 filter. Definition: from the 165-question validation set,
   keep text-only questions — no file attachment AND no annotator-metadata tool that
   implies files/multimodality (image/video/audio recognition, file readers). We staged
   127 (attachment filter only); the extra ~24 likely fail the tool filter. ACCEPTANCE:
   exactly the 103-question subset used by the SupervisorAgent/ODS line (needed for the
   direct ICLR'26 comparison in E2) — or, if irreproducible, a documented deviation note
   in the dataset manifest + paper appendix.
5. Safe refactors: unify the two MockStopper classes; add `--policy` to collect.py.
6. Align the §18.1 serialization caps (see the decisions list below).
Anything else GPU-shaped: don't. Don't poll for GPUs either.
