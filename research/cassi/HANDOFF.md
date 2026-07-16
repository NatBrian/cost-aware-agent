# CASSI Implementation — Handoff for the Next Agent

**Written 2026-07-16.** The spec is `research/paper_plan_v2.md` (source of truth; §16 is the
runbook). This document tells you what is already built, what is pending, and the exact next
commands. Read `research/paper_plan_v2.md` §0, §2, §12, §16, §17 before touching anything.

**Per user instruction: never read `research/archived*` folders** (stale AI outputs).

---

## 1. Current state, in one paragraph

Everything that can be built and verified WITHOUT a GPU is built and verified: the full
`cassi/` package (labels, stopper, executor, baselines, eval, analysis), the phase runner
scripts P0–P9 including the kill-switch gate, the LaTeX paper skeleton (compiles to
`paper/main.pdf`), and a 109-test CPU suite that passes in ~5 s
(`cd research/cassi && python -m pytest tests/ -q`). The core math is validated on synthetic
data: Snell recursion finds the hand-computable τ*, higher λ ⇒ earlier stopping (0 monotonicity
violations), PBRS telescopes to −Φ(x₀), trajectory-level advantages are provably shaping-blind
(hence step-level credit + min-cohort guard), and B9's advantages are bit-identical to CASSI's
when V̂ ≡ V* (the designed control). **Nothing has touched a GPU yet. No real data is
downloaded. No model is trained.** All experimental numbers do not exist yet.

## 2. What is implemented (module map + test coverage)

| Where | What | Verified by |
|---|---|---|
| `common/schema.py`, `common/config.py` | §11 trajectory JSONL schema; §17 config loader + pilot-calibration guard | `tests/test_core.py` |
| `budget/cost.py` | The ONE economy (§2.2/§2.4): token/tool pricing, tiers, m(tier), wallets, pilot normalization, U_t, R_base; reuses the repo harness price map for API models | `tests/test_core.py` (U_τ ≡ R_base) |
| `labels/snell.py` | **Algorithm 1** (LightGBM backward recursion, Δ*, τ*, V*, tanh scale, backup residuals) + prophet labels (E4 arm) + QC (λ-monotonicity) | `tests/test_core.py` |
| `labels/quality.py`, `labels/drafts.py` | F1/EM/subgoal scoring; running-draft parse + stability features (§2.6/§18.2) | `tests/test_core.py` |
| `stopper/features.py` | §18.1 serialization (λ-conditioning) + numeric vector for the label regressor | `tests/test_core.py` |
| `stopper/dataset.py`, `model.py`, `train_sft.py`, `eval_regret.py` | SFT examples (pooled λ, split-by-task), 3-head model (Alg. 2), custom training loop (early-stop on held-out REGRET), regret eval + P4 gate baselines | `tests/test_stopper_cpu.py` (15) |
| `executor/react_agent.py`, `collect.py`, `monitor.py`, `shaping.py`, `train_grpo.py`, `envs/`, `vllm_client.py` | Shared scaffold (both §2.1 rollout modes), forced-continuation collection + wallets per (task, group), Alg. 4 monitor (fixed Δ̂≤0 + A8 mode + internalization tracking), PBRS + step-level RTG advantages + min-cohort guard, verl config builder + adapter | `tests/test_executor_cpu.py` (27), `tests/test_integration.py` |
| `baselines/` (b1–b9 + oracle) | One module per §5.2 row with registry, cost knobs, reward logic | `tests/test_baselines_cpu.py` (26) |
| `eval/metrics.py`, `stats.py`, `overhead.py` | Frontier protocol + interpolation, regret, matched-risk, §5.6 stats (small-n guard ENFORCED), T4 ledger + serving regimes + billing symmetry (enforced) | `tests/test_eval_cpu.py` (20) |
| `analysis/` + `Makefile` | One script per F1–F6/T1–T5, CSV→PDF/tex, CVD-safe | `make figures tables` (skips gracefully pre-P9) |
| `scripts/` | P0–P9 runners; `p5_killswitch.sh` writes `GO_NO_GO.log`; GPU acquire/release with EXIT traps | shellcheck-style review; python drivers tested |
| `paper/` | main.tex + 8 §9-mapped section stubs + references.bib (41 seeded entries) + Makefile | compiles: `cd paper && make` |

## 3. PENDING — ordered work queue (this is your job)

Everything below needs GPUs and/or network downloads. **GPU protocol for this machine
(CLAUDE.md):** `eval $(/mnt/src/zhanka/gpu_acquire.sh N)` before, `/mnt/src/zhanka/gpu_release.sh`
after; N=2 for collection/stopper SFT, N=4–8 for GRPO. Never kill occupier processes.

1. **P0 — installs** (`scripts/p0_setup.sh`): clones verl/verl-tool/verl-agent/Search-R1 into
   `third_party/` (gitignored), installs `requirements-gpu.txt`, pins commit hashes into
   `configs/cassi.yaml pins:`. Done-criterion: one ReAct rollout on HotpotQA dev + one on
   ALFWorld with the draft line in every step (`scripts/verify_smoke.py`).
2. **P1 — data** (`scripts/p1_data.sh`): downloads NQ/HotpotQA/MuSiQue/Bamboogle/2Wiki/MATH-500/
   AIME-2025, stages GAIA-103 + BrowseComp-Plus, builds the Search-R1 wiki index + retriever
   server, runs `scripts/decontaminate.py`, emits the dataset manifest.
3. **P2 — pilot + collection** (`scripts/p2_pilot_and_collect.sh`): 200-task unconstrained pilot
   → **write the printed wallet/median values into `configs/cassi.yaml`** (they are `null` now;
   later phases refuse to run until filled) → round-0 forced-continuation collection (G=8).
4. **P3 — labels** (`scripts/p3_labels.sh`): Algorithm 1 per λ ∈ {0.1,0.5,1,2,5} + QC memo.
5. **P4 — stopper v0** (`scripts/p4_stopper.sh`): SFT + the HARD GATE (beat majority-class AND
   the confidence probe on held-out regret, else STOP and fix features/labels).
6. **P5 — KILL-SWITCHES K1/K2** (`scripts/p5_killswitch.sh`, §12): the GO/NO-GO moment on
   HotpotQA-1K, 1 seed. Requires finishing the two wiring gaps below first. Decision appended
   to `GO_NO_GO.log` — never delete or rewrite past entries (§5.6 no-cherry-picking).
7. **P6–P9** per §16 (iteration 1, loop iteration 2 with frozen-coach control, baselines,
   full eval incl. 500-task regret replays, 3 seeds on headline points).
8. **P10–P11**: `make figures tables`, then write the paper into `paper/sections/`
   (writing order and claims-audit rule: §16 P11; every §14-dead-claim is banned).

### Known wiring gaps (small, deliberate — finish before P5)

- **`cassi/eval/run_frontier.py` CLI does not exist yet.** Scripts p5/p9 call it to sweep a
  method over its knob and emit `arm,lambda_dial,accuracy,cost_dollars` CSVs (contract
  documented in `scripts/p5_killswitch.sh`). It's an aggregation runner over eval episodes —
  build it against `eval/metrics.py`'s `Frontier`.
- **`executor/train_grpo.py` exposes `--config/--domain/--dry-run` only.** The full §16 flag
  contract used by p5/p6/p7 (`--tasks/--coach/--arm/--max-steps/...`) plus the actual verl
  hookup is a documented 4-point TODO(P6) block in `VerlCassiAdapter` (reward tensor on
  step-final tokens; bypass verl's trajectory-level GRPO advantage with our
  `compute_cassi_rewards`; V̂ serving; SHAPE-segment variant). The reward/advantage SEMANTICS
  are done and tested — only the verl plumbing remains, against whatever verl ≥0.8 API is
  pinned at P0.
- `collect.py` re-collection with a trained policy: serve the checkpoint with vLLM and pass
  `--vllm-url` (no `--policy` flag; documented in p7/p9).
- P9's eval emitters must log **per-instance results per knob/threshold point** — the stats
  functions consume per-instance matrices, not aggregates.

### Decisions made during implementation (so you don't re-litigate)

- §17 named a single-head TRL recipe but §2.3 needs three heads → custom torch loop at the
  same hyperparameters (`stopper/train_sft.py` docstring).
- §18.1 needs nominal caps the plan never fixed → `DEFAULT_TOKENS_MAX=32768`, tool cap =
  T_max (feature-only, never enforcement); consider adding to §17.
- Alg. 4 "budget exhausted" quantified as spent ≥ allowance (both monitor modes, any k).
- §12 "≥3 points" interpreted as percentage-point cost reduction at iso-accuracy
  (`scripts/killswitch_decision.py`).
- B4 uses OTC's ratio reward form; B5 implements the solve-rate-scaled penalty family (EAPO
  primary / agentic-ALP fallback share the form); B6 defaults to flat-λ (published CTA form)
  with a `tier_scaled` fairness variant. All disclosed in module docstrings.
- Monitor accepts BOTH stopper protocols (text `evaluate` / feature `predict`) — see
  `executor/monitor.py` docstring.
- `LabelSet` persistence lives in `stopper/dataset.py` (`save_labelset`/`load_labelset`).
- `references.bib`: every entry must be verified against the real paper at P11 (authors
  marked TODO where the plan gave only arXiv ids).

## 4. Schedule & budget reality (plan §7)

~30–35 training runs, 10–12 weeks on the 8×H200 node. K1/K2 (week 2–3) come FIRST — do not
build past P5 without a logged GO. Competitors are ≤3 weeks old at review time (DASH,
OS-Pruner); if K1 passes, consider the workshop-preprint hedge (§8).

## 5. Quick verification that nothing rotted

```bash
cd research/cassi
python -m pytest tests/ -q          # expect: 109 passed
cd paper && make && cd ..           # expect: main.pdf builds
bash scripts/p0_setup.sh            # first pending step (network + disk)
```
