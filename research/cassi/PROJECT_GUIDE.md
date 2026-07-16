# CASSI Project Guide — read this to fully understand the project

**Audience:** an AI agent (or human) with ZERO prior context. This is the master
explanation: what the research is, why every piece of code exists, how the documents
relate, what's done, what's next, and what must never be changed. Reading order:
this file → `HANDOFF.md` (operational status + next commands) → `../paper_plan_v2.md`
(the authoritative spec) as needed.

**Hard rule inherited from the user:** never read `research/archived*` (stale outputs).

---

## 1. What this research is (plain language)

LLM agents work in loops: think → call a tool → look at the result → repeat. Nothing
in their training ever teaches them that **each extra step costs money and that at some
point the next step is no longer worth its price**. So they overwork: measured studies
show extra steps beyond the optimum actually REDUCE accuracy while burning cost.

Today's fixes are all external: a supervisor process watches the agent and cuts it off
(the agent itself never learns), or a blunt cost penalty is added to the final reward
(too coarse — it can't tell WHICH step stopped being useful).

**CASSI's idea, in one sentence:** train a small "coach" model that estimates, at every
step, whether continuing is still worth it economically — then use the coach's estimate
as a dense, step-by-step training signal for the big "worker" agent, so the worker
*internalizes* when to stop instead of being externally throttled.

- **Coach** = the stopper, Qwen3.5-2B. Learns from hindsight: after collecting many
  finished attempts, we can compute in retrospect where stopping would have been optimal
  (the "Snell envelope" — see glossary). The coach is trained to predict that.
- **Worker** = the executor, Qwen3.5-9B. Trained with RL (GRPO) where the coach's value
  estimate shapes the per-step reward (potential-based shaping — provably doesn't change
  what the best policy is, only makes credit assignment dense).
- **The loop:** better worker → fresh data → recompute optimal stopping → refresh coach
  → retrain worker. At least 2 iterations, with a control arm proving the loop itself
  (not just "more training") is what helps.

**The one claim the paper lives or dies on:** a worker TRAINED with coach-derived
rewards beats (i) the same coach used only as an inference-time supervisor, (ii) using
the hindsight labels directly as rewards with no coach (B9), and (iii) training-free
monitors — on cost at equal accuracy, across 2 domains. Kill-switch experiments K1/K2
test this cheaply BEFORE the expensive pipeline (paper_plan_v2 §12).

**Why it's novel (verified 2026-07-16 by a 94-paper review + 5 independent audits):**
every ingredient exists separately in the 2026 literature — hindsight stop labels
(inference-time only), monitor-as-reward (quality-only, no cost), cost-aware RL
(trajectory-level only). NOBODY converts an explicit quality−λ·cost hindsight optimum
into a trained stopping-value model that trains the executor at the step level, and
nobody measures whether stopping economics can be internalized into weights. That
composition + measurement is the paper.

Target: ICLR 2027 (submission ~Sept 2026). Fallback venues NeurIPS/ICML 2027.

---

## 2. Glossary — paper terms → code symbols

| Term | Meaning (plain) | In code |
|---|---|---|
| `q_t` | how good the agent's current draft answer is at step t (F1 vs gold; ALFWorld: fraction of subgoals done). Ground-truth-derived → label machinery ONLY | `labels/quality.py`; stored on `Step.q` (0.0 during live rollouts by design — filled at collection scoring) |
| `c_t`, tier, wallet | dollar cost of step t; budget tier = how much of the wallet (allowance) remains (HIGH/MEDIUM/LOW/CRITICAL); wallet drawn per (task, GRPO group) | `budget/cost.py` (`tier_from_remaining`, `draw_wallet`) |
| `U_t` | stopping utility = quality so far minus tier-scaled, normalized, λ-weighted spend: `q_t − Σ λ·m(tier_i)·c̃_i` | `budget/cost.py: stopping_utilities` |
| λ (lambda) | the cost-sensitivity dial. Higher λ = cost matters more = stop earlier. The stopper is λ-CONDITIONED (λ is in its input text), so one model serves the whole dial | config `label.lambda_values`; `stopper/features.py: serialize` |
| Snell envelope | the mathematically correct "should I stop now?" value computed backward over collected trajectories (max of stop-now vs expected-value-of-continuing). Avoids the "prophet bias" of just taking the best-in-hindsight step | `labels/snell.py: snell_labels` (Algorithm 1); prophet version kept only as E4 comparison |
| Δ* / Δ̂ | stop margin: continuation value − stop-now value. >0 ⇒ continue. Δ* = label; Δ̂ = stopper's prediction. Inference stops at Δ̂ ≤ 0 | `StepLabel.delta_*`; `executor/monitor.py` |
| V* / V̂ | the stopping VALUE (unnormalized). V̂ is the shaping potential Φ | third head of `stopper/model.py`; consumed by `executor/shaping.py` |
| PBRS / telescoping | potential-based reward shaping: r_t = Φ(next) − Φ(now). Provably doesn't change the optimal policy. With γ=1 the sum telescopes to −Φ(start) ⇒ trajectory-level advantages CANNOT see it ⇒ step-level credit is mandatory | `executor/shaping.py` (tests prove both facts) |
| min-cohort guard | at late steps few group members are still alive; below 3, fall back to trajectory baseline for those steps only | `shaping.py: step_level_group_advantages` |
| forced continuation | label-collection rollouts IGNORE the agent's ANSWER (log it, keep going to T_max) so we can observe what continuing would have yielded — otherwise labels are censored by the very behavior we train | `executor/react_agent.py` mode=`forced_continuation`; `answered_flag` on Step |
| running draft | every step, the agent must end with `BEST ANSWER SO FAR: ...` — makes per-step quality a free string comparison and gives the stopper stability features. ALL methods share it (fairness) | `labels/drafts.py`; `react_agent.py` scaffold |
| R_base | the worker's terminal reward, SAME economy as the labels (one Lagrangian): `Q_τ − Σ λ·m(tier)·c̃` | `budget/cost.py: base_reward` |
| GRPO / Dr.GRPO | the RL algorithm (group of G=8 rollouts per task, advantages relative to the group); Dr.GRPO = bias-hygiene settings so token savings aren't a length-bias artifact | `executor/train_grpo.py` + `verl_hooks.py` |
| frontier protocol | you can't compare methods at "equal accuracy" unless EVERY method is swept over its own cost knob → 3–5 point frontier, interpolate | `eval/metrics.py: Frontier`; `eval/run_frontier.py` |
| stopping regret | utility gap between where the method stopped and the Snell-optimal stop, measured via a second forced-continuation replay (dual-run protocol) | `eval/run_frontier.py`, `stopper/eval_regret.py` |
| internalization | THE headline measurement: turn the monitor OFF at test time — a worker that still stops well has internalized the economics; also % episodes self-stopped before the monitor fires | `executor/monitor.py` stats; E2 in the plan |
| K1 / K2 | kill-switches. K1: does the coach-as-training-signal beat coach-as-supervisor and labels-without-coach? K2: two models vs one at matched params. Run FIRST; NO-GO → pre-planned pivots | `scripts/p5_killswitch.sh`, plan §12, fallbacks §6 |

---

## 3. The pipeline, end to end (what runs, in order, and where)

```
P0  install/pin stack        scripts/p0_setup.sh          [DONE except GPU smoke]
P1  data + decontamination   scripts/p1_data.sh           [DONE]
──── first GPU session: scripts/smoke_and_pilot.sh ─────────────────────────────
P0' smoke rollout            collect.py --smoke → verify_smoke.py
P2  pilot → wallets          collect.py --pilot → WRITE VALUES INTO configs/cassi.yaml
P2' round-0 collection       scripts/p2_pilot_and_collect.sh (forced continuation, G=8)
P3  Snell labels per λ       scripts/p3_labels.sh (Algorithm 1 + QC memo)
P4  train coach v0           scripts/p4_stopper.sh (SFT; GATE: beat majority+probe on regret)
P5  KILL-SWITCHES K1/K2      scripts/p5_killswitch.sh → GO_NO_GO.log  ← decision point
P6  worker GRPO iteration 1  scripts/p6_grpo_iter1.sh (train_grpo.py + verl_hooks.py)
P7  loop iteration 2         scripts/p7_loop_iter2.sh (frozen-coach control arm = E5)
P8  baselines B2–B9          scripts/p8_baselines.sh
P9  full eval E1–E6 + A1–A9  scripts/p9_eval.sh (everything lands in experiments/results/*.csv)
P10 figures/tables           make figures tables (CSV in → PDF/tex out, no hand edits)
P11 write the paper          paper/ (skeleton compiles; writing order in plan §16)
```

Data flow: rollout JSONLs (schema `common/schema.py`, §11) → `labels/snell.py` →
labeled sets → `stopper/train_sft.py` → coach checkpoint → (a) `executor/monitor.py`
at inference, (b) V̂ into `verl_hooks.py` rewards at training → eval CSVs →
`analysis/` scripts → `paper/figures|tables` → `paper/main.tex`.

---

## 4. Repo map (research/cassi/)

```
common/    schema.py (§11 trajectory JSONL — THE data contract), config.py (loads §17 YAML;
           blocks post-P2 phases until pilot values are filled)
budget/    cost.py — THE economy. One module owns every dollar/tier/U_t/R_base computation
           so coach, worker, and labels provably optimize the same objective.
labels/    snell.py (Algorithm 1 + QC), quality.py (F1/EM/subgoals), drafts.py (draft parse
           + stability features + legacy probe kept only for ablation A5)
stopper/   features.py (§18.1 input text + numeric vector), dataset.py (SFT examples,
           task-level splits, LabelSet persistence), model.py (3-head model, MockStopper,
           load_predictor), train_sft.py (Alg.2; early-stop on held-out REGRET), eval_regret.py
executor/  react_agent.py (shared scaffold, both rollout modes), collect.py (P2/P7 + --pilot
           + --smoke), monitor.py (Alg.4 + internalization stats), shaping.py (PBRS + step
           advantages — the math core, fully CPU-tested), train_grpo.py (full §16 CLI +
           --dry-run), verl_hooks.py (ALL verl-touching code, pinned refs), vllm_client.py,
           envs/ (searchr1_qa.py, alfworld.py, base.py with MockSearchEnv)
baselines/ b1_react … b9_direct_shaping + oracle; registry in __init__.py; each docstring
           states the paper it reimplements, its cost knob, and what question it kills
eval/      metrics.py (frontier/regret/matched-risk), stats.py (§5.6, guards ENFORCED),
           overhead.py (T4 ledger, serving regimes, billing symmetry), run_frontier.py (CLI)
analysis/  figures/f1..f6, tables/t1..t5 (CSV → PDF/tex; Makefile at repo level)
scripts/   p0..p9 phase runners, smoke_and_pilot.sh (first GPU session), launch_grpo.sh,
           killswitch_decision.py, decontaminate.py, download_data.py, run_labels.py, ...
paper/     main.tex (compiles TODAY with article fallback), sections/01..08 (each headed by
           page budget + which plan §s feed it), references.bib (verify entries at P11)
tests/     114 CPU tests (~5s). test_core.py = the math invariants. Run before/after any change.
configs/   cassi.yaml — single source of truth (§17). `pins:` = installed stack versions.
data/      (gitignored) datasets, frozen dev subsamples, searchr1_index/ (64GB E5 + corpus)
third_party/ (gitignored) pinned clones: verl, verl-tool, verl-agent, Search-R1
.venv/     dedicated python env — scripts auto-activate it (scripts/common.sh)
```

---

## 5. Document map — which file answers which question

| Question | Read |
|---|---|
| What exactly is the method/experiments/claims? | `../paper_plan_v2.md` — THE spec. §2 method, §5 experiments, §10 algorithms, §12 kill-switches, §16 runbook, §17 config, §19 sourcing. Where anything disagrees, it wins. |
| What's done / what's next / what commands? | `HANDOFF.md` (+ §3b adjustment map: what to tune after real data, freeze rules) |
| Why is the field-positioning what it is? | `../lit_review/00_overview.md` (+ 10 area files); `../competitor_analysis.md` (beginner-friendly) |
| How harsh were the pre-rewrite reviews? | `../novelty_check/` (5 audit verdicts vs the OLD v5 plan — v2 already answers them; see paper_plan_v2 §14 changelog) |
| What was decided during experiments? | `GO_NO_GO.log` (append-only; created at P5) |
| Machine rules (GPUs, git identity)? | repo-root `AGENTS.md`/`CLAUDE.md`; `/root/dataDisk/liangsheng/Brian/AGENTS.md`; `/root/dataDisk/liangsheng/CLAUDE.md` |

---

## 6. Status snapshot (2026-07-16, end of build session)

DONE: all code (both former wiring gaps closed), 114 CPU tests green, verl dry-run green,
P0 installs+pins, P1 data (decontaminated, manifest), wiki-18 corpus + 64GB E5 index
assembled, Qwen3.5-9B/2B prefetched, paper skeleton compiles, everything pushed to
github.com/NatBrian/cost-aware-agent (private).

NOT DONE (all GPU-gated): smoke rollout, pilot calibration (config `null`s), collection,
labels-on-real-data, coach training, K1/K2, all RL, all experiments, all real numbers, the
paper text. **GPUs are currently off-limits** (machine held by another user's job; user
instruction: do not use or wait for GPUs). First command when that changes:
`bash scripts/smoke_and_pilot.sh`.

Deliberate stubs (documented, will raise NotImplementedError with instructions):
ALFWorld agent-loop under verl (package-name clash — needs own env), K2's
single-multitask arm, BrowseComp-Plus staging, GAIA exact-103 filter.

## 7. How to verify nothing rotted (run anytime, no GPU)

```bash
cd research/cassi
python -m pytest tests/ -q                                    # expect 114 passed, 1 skipped
.venv/bin/python -m cassi.executor.train_grpo --dry-run \
    --config configs/cassi.yaml --domain qa                   # expect "[dry-run] OK"
cd paper && make && cd ..                                     # expect main.pdf builds
```

## 8. The ten commandments (violating these voids the paper)

1. paper_plan_v2.md outranks every other document and any memory of past sessions.
2. Never read `research/archived*`.
3. x_t (stopper input) must NEVER contain ground truth or executor-stated confidence.
4. Label collection = forced continuation; RL rollouts = natural termination. Never mix.
5. One economy: labels, R_base, and eval utilities all come from `budget/cost.py`.
6. Step-level advantages are mandatory (trajectory-level shaping is provably inert).
7. Frozen things stay frozen: eval subsamples, scaffold after first baseline, economy
   after P3, headline λ chosen on dev, GO_NO_GO.log append-only.
8. Every method pays for its own auxiliary inference (billing symmetry) — enforced in code.
9. Kill-switches gate the expensive phases; a NO-GO is a documented pivot, not a failure.
10. Git identity: NatBrian only (enforced by folder config); no AI co-author lines;
    GPUs only via acquire/release, never kill foreign jobs.
