# Novelty Check — Agent 5: Devil's-Advocate Prior-Art Hunter

> Date: 2026-07-16. Mission: assume CASSI is NOT novel and try to prove it.
> Method: 17 query families (WebSearch; arXiv/S2 APIs were rate-limited so web + local PDFs used),
> then PDF-level verification of the 14 most dangerous papers (method + experiments read, not just abstracts).
> All verified PDFs are in `research/papers/`.

---

## My understanding of the proposal

CASSI trains a small (0.5B–3B) **stopping model** on **oracle stopping labels computed post-hoc from
completed executor trajectories**: `t* = argmax_t [quality_t − λ·cumcost_{1..t}]` — an O(T) computation with
zero extra rollouts (vs. AgentPRM's O(K×T²) Monte Carlo rollouts). Steps before t* are labeled CONTINUE,
after t* STOP, with a continuous margin Δ(s_t) = Q_continue − Q_stop. The stopper (SFT + GRPO, structured
budget-state input, STOP/CONTINUE/ADJUST + Δ + rationale output) is then used as a **cost-aware process
reward model** to train the 7B–72B executor via GRPO (step reward α·Δ + β·progress + γ·format + terminal
success), and enforced as a controller at inference (<3% overhead claimed).

Five claims: (1) first **self-reinforcing cycle** (executor → trajectories → oracle → stopper → process
rewards → better executor); (2) a **separate** stopping model is *necessary* (representation-conflict
argument, tested by ablation); (3) **O(T) oracle labels** replace O(K×T²) MC rollouts, with three formal
properties (uniqueness, λ-monotonicity, improvement under policy improvement); (4) **per-instance dynamic
cost adaptation** beats static length penalties (H5: stop step correlates with difficulty, r > 0.5);
(5) small stopper supervises large executor at <3% overhead with 20–40% cost savings at iso-accuracy.
Benchmarks: GAIA, WebWalkerQA, HotpotQA, MuSiQue, SWE-bench Verified, MATH-500, BFCL.

---

## Hunt log (query family → what surfaced)

| # | Query family | Key hits |
|---|---|---|
| 1 | learned stopping criterion / stop policy LLM RL | LACONIC (2602.14468, constrained length RL); "interwhen" (2602.11202); Answer-convergence early stop; **When Does Learning to Stop Help? (2606.30852)** |
| 2 | cost-aware process reward / efficiency reward model | **AgentPRM-WWW'26 (2511.08325, promise+progress TD PRM)**; xRouter (2510.08439, cost-aware routing RL); Tool-PRMBench (2601.12294); ATTNPO (2602.09953, attention-guided process supervision for efficiency) |
| 3 | value of computation / rational metareasoning | **Rational Metareasoning for LLMs (2410.05563, NeurIPS'24 wksp)** — VOC reward = quality − cost penalty in Expert-Iteration RL, 20–37% token cuts. CASSI's oracle objective *is* a hindsight VOC — must cite |
| 4 | optimal stopping / anytime reasoning | **AnytimeReasoner/BRPO (2505.13438)** — sampled budgets + verifiable dense rewards; **MRT (2503.07572)** — dense progress rewards, regret minimization; optimal-stopping-vs-best-of-N (2510.01394); Hansen & Zilberstein 2001 (anytime monitoring, classic) |
| 5 | hindsight optimal stop / post-hoc trajectory labeling | **TERMINATOR (2603.12529)**; **DASH (2607.00482)**; AgentHER (2603.21357); HCAPO (Tan 2026, hindsight step Q-values); Hindsight Credit Assignment for LLM agents (2603.08754); ECHO (2510.10304) |
| 6 | small model supervises large (reward) | AgentPRM 3B > GPT-4o (2502.10325); Prover-Verifier Games; RLT teachers; Agent-RRM (2601.22154) |
| 7 | budget-conditioned / deadline-aware policy | **BAGEN (2606.00198)**; BudgetThinker (2508.17196); **CoRL (2511.02755)**, budget-conditioned controller RL; TAB multiturn budgets (2604.05164); ContextBudget (2604.01664) |
| 8 | when to stop retrieval / adaptive search termination | AutoSearch (2604.17337, PPO w/ correctness+efficiency reward); RLStop (2405.02525); to-search-or-not-to-search (2602.03304); Probing-RAG/Adaptive-RAG family |
| 9 | tool-call budget RL / minimal tool calls | OTC (2504.14870); **EAPO / Learning When Not to Act (2606.02132)**; Alibaba "Metis" (redundant calls 98%→2%); knowledge-boundary RL (2605.26952) |
| 10 | monitor-guided RL / critic-guided termination | **MaR (2605.23384)**; **SupervisorAgent (2510.26585, ICLR'26)**; MGV (2511.04341); Dolores (2605.11388) |
| 11 | process reward without rollouts / single-trajectory credit | Implicit PRM (2412.01981, free process rewards from outcome labels); **GRPO is Secretly a PRM (2509.21154)**; GiGPO (2505.10978); SPA-RL (2505.20732); SWEET-RL; ESPO (2605.29860) |
| 12 | co-training reward model + policy loop | **Self-Guide (2604.03098)**; Cooper (2508.05613); SPARK (2509.22624); RLAnything; CoVerRL (2603.17775); PAG (2506.10406) |
| 13 | overthinking agents mitigation training | **DASH**; Overthinking Loops via MCP (2602.14798); danger-of-overthinking (2502.08235); **RePro (2606.14302)** |
| 14 | Pareto cost-accuracy / adaptive per-instance penalties | ALP (2506.05256); LASER-D (2505.15612); DAST (2503.04472); AdaCoT (2505.11896); DLER (2510.15110); **SlimSearcher (2606.07074)**; Think-Dense-Not-Long (2602.02099) |
| 15 | Chinese labs 2026 efficient agent RL | Qwen3-Coder-Next TR (2603.00729), Kimi-Dev, DeepSWE, SkyRL-Agent — RL training-efficiency focus; nothing on learned cost-aware stopping as reward |
| 16 | ICLR'26 / NeurIPS'25 OpenReview sweeps | SupervisorAgent (ICLR'26); Reason-Efficiently (NeurIPS'25); a 2605.02801 survey states: "as of May 4 2026, no explicit RL training method for the stopping decision was found" (orchestration domain) — supports the gap |
| 17 | SWE-agent cost-aware training | SWE-TRACE (2604.14820, rubric PRMs, no MC rollouts, SWE agents); SkyRL-Agent; none cost-aware stopping |

---

## Kill-shot candidates (verified from PDF unless noted)

### 1. DASH — "Know When to Stop: Segment-Level Credit Assignment for Reducing Overthinking" (2607.00482, Capital One, 1 Jul 2026) — **most dangerous single citation**
- **What it does (PDF verified):** Extracts intermediate answer commitments from completed reasoning traces, verifies each against ground truth (explicitly framed as a *cheap alternative to PRM step labels*), splits the rollout into segments, and assigns segment-level GRPO advantages: positive toward-correct, negative away-from-correct, **with an escalating length penalty on tokens past the point the correct answer was reached** ("the longer you continue past a correct answer, the worse it gets"). Trains Nemotron-4B on math (AIME/AMC/OlympiadBench); best accuracy where overthinking is worst.
- **How close:** This is CASSI's core mechanism minus the stopper: post-hoc per-step quality checks from a *single* trajectory (zero extra rollouts) converted into step-level training signal that teaches the executor to stop after the answer is good. For binary quality, CASSI's post-t* STOP labels ≈ DASH's negative post-correct segments.
- **What remains different (honest both ways):** DASH has **no separate stopping model, no cost model (no tool/dollar/λ), no budget state, no inference-time controller, no cycle, no agents/tools** — it is advantage shaping inside GRPO for math CoT. But it proves the "hindsight step-quality labels as process reward for executor RL" bridge works *without* any learned stopper — which threatens CASSI's claim that the stopper is the necessary middle piece. CASSI must include a "DASH-for-agents / direct oracle-as-advantage" baseline; if that baseline matches CASSI, contribution 2 collapses.

### 2. Self-Guide — "Co-Evolution of Policy and Internal Reward for Language Agents" (2604.03098, McGill/Mila, 3 Apr 2026)
- **What it does (PDF verified):** Agent generates short self-guidance signals during rollout; the same signal steers actions at inference AND is converted into step-level internal reward for GRPO. Explicitly claims a "co-evolving loop: better policy produces better guidance, and better guidance further improves policy," with a stage-wise trust schedule. ALFWorld/ScienceWorld/WebShop, ~8% over GRPO.
- **How close:** Directly falsifies the *phrasing* of Contribution 1 — a self-reinforcing policy↔reward cycle with step-level rewards for LLM agents exists (also Cooper 2508.05613, SPARK 2509.22624, RLAnything co-evolve RM+policy).
- **What remains different:** Self-generated signals (same model, no separate small stopper), **no cost-awareness, no stopping decision, no oracle supervision** (signals are not anchored to a quality−cost optimum), no budget state. The *cost-aware stopping* instantiation of the cycle is not done.

### 3. TERMINATOR — "Learning Optimal Exit Points for Early Stopping in CoT" (2603.12529, UT Austin/EPFL, Mar 2026)
- **What it does (PDF verified):** Defines **hindsight-optimal reasoning length (HORL)** — "the earliest position in the completed CoT at which the final answer has been logically reached... a retrospective property of the realized CoT" — builds an optimal-exit-label dataset from completed traces, trains a lightweight binary probe stopper, and early-exits at inference. Pareto-dominates DEER/Dynasor/Thought-Calibration on MATH-500/AIME25/HumanEval/GPQA (14–55% CoT cuts).
- **How close:** The oracle-labeling contribution in its binary-quality form: hindsight-optimal stop labels from completed trajectories → train a small stopper. Same family: **LYNX (2512.05325**, forced-exit GT labels + conformal control, PDF verified), **LearnStop (2606.30852**, logistic stopper on checkpoint GT labels, PDF verified), Thought Calibration.
- **What remains different:** All inference-time only — **none feeds the stopper back as a reward to train the executor**; CoT token-level, no tool/dollar cost, no λ trade-off (TERMINATOR is pure "first correct arrival", not quality−λ·cost), no budget conditioning, no agents.

### 4. SlimSearcher — "Training Efficiency-Aware Web Agents via Adaptive Reward Gating" (2606.07074, ZJU/Ant, 5 Jun 2026)
- **What it does (PDF verified):** Pareto-filtered SFT + **Adaptive Efficiency Anchoring**: reward shaping anchored to the "Minimal Necessary Path" found within each sampled trajectory group — an explicitly *per-task-adaptive* cost signal designed "to circumvent the brevity bias inherent in fixed penalties," gated by strict correctness. Cuts tool-call rounds 17–58% on **GAIA**, BrowseComp, XBench-DeepSearch.
- **How close:** Kills the exclusivity of Claim 4 on CASSI's own benchmark class: per-instance, non-static cost pressure in agent RL is published. EAPO (2606.02132, PDF verified: difficulty-aware redundant-tool-call penalties via tool-free rollouts) and ALP/LASER-D/DAST/AdaCoT/DLER (reasoning domain) pile on.
- **What remains different:** Single model, outcome-level shaping (no per-step stop labels), no learned stopping model, no Δ value, no budget state, no inference-time control, no mid-trajectory adaptation *signal* (the anchor is per-group, not per-step).

### 5. BAGEN — "Are LLM Agents Budget-Aware?" (2606.00198, Northwestern/RAGEN group, 29 May 2026)
- **What it does (PDF verified):** Formalizes budget-awareness as progressive interval estimation (predict remaining-budget interval or `impossible` each turn) via a rollout-replay protocol (labels derived by replaying logged prefixes of completed rollouts — hindsight labels). Internal (tokens) + external (money/inventory) budgets; Sokoban, Search-R1, SWE-bench, Warehouse. **Early-stop keyed on `impossible` saves 28–64% of tokens on failed trajectories at 1.6–4.2pp success cost; SFT+RL strengthens early-stop/alert.**
- **How close:** Owns the "agents lack economic judgment, and it's trainable" framing, with multi-dimensional budget states, on SWE-bench, before CASSI. Trains stop-relevant behavior with SFT+RL.
- **What remains different:** It's a capability benchmark + self-estimation training; early stopping mainly aborts *failing* trajectories (not optimal stopping of *successful overthinking*); no stopping model as PRM, no quality−λ·cost oracle, no executor process-reward training, no cycle.

### 6. CaRT — "Teaching LLM Agents to Know When They Know Enough" (2510.08517, CMU, Oct 2025) — plan's primary baseline, verified
- **PDF verified:** SFT on counterfactual termination pairs + verbal rationale ("verbalized value function") in the *same* model; medical diagnosis + math. Plan's characterization is accurate (SFT-based, self-termination, no cost λ, no executor RL, no separate stopper). Note CaRT's authors (Qu, Kumar) also wrote MRT — this group is one step from closing CASSI's loop themselves.

### 7. RePro — "Retrospective Progress-Aware Self-Refinement for LLM Agent Training" (2606.14302, SJTU/OPPO, 12 Jun 2026)
- **What it does (PDF verified):** Forward-then-reflect: after completing a trajectory and observing the outcome, the agent retrospectively labels its own per-step progress; RePro-PO turns these retrospective per-step signals into a composite dense reward complementing sparse outcome reward. WebShop/ALFWorld/Sokoban, up to +12pp.
- **How close:** "Retrospective per-step labels from completed trajectories → dense process rewards → agent RL" — the CASSI bridge, in agents, with hindsight labels.
- **What remains different:** Progress (quality-only), **no cost term, no stopping decision**, self-generated (no separate small model), no λ, no budget, no controller.

### 8. Honorable mentions (verified from PDF or abstract)
- **MaR (2605.23384, PDF verified):** frozen grader scores metacognitive knowledge/regulation as trajectory-level RL reward — monitor-as-reward genre, but cost-blind, stop-free.
- **SupervisorAgent (2510.26585, ICLR 2026, PDF verified):** lightweight runtime supervisor cuts 29.7% tokens on GAIA without training the base agents — the *inference-time* monitor value proposition already delivers ~30% savings; CASSI must beat this class or show training adds more.
- **Ares (2603.07915, PDF verified):** small router (Qwen3-1.7B) trained on "minimum reasoning effort that still yields the correct action" labels mined from collected trajectories — an oracle-style, trajectory-derived economic label for a small per-step controller over a frozen agent. Plan's characterization (discrete levels, no executor training) is accurate.
- **CoRL (2511.02755, PDF verified):** PPO-trained controller LLM with dual task+cost rewards and budget-conditioned modes over frozen expert pool — budget-conditioned cost-aware RL exists for orchestration; not cited in the plan.
- **AgentPRM-WWW'26 (2511.08325, abstract + intro):** TD-style "promise + progress" step scores for agent PRMs with beam search — a *second* AgentPRM the paper must disambiguate from Choudhury 2025; quality-only, inference-time guidance.
- **Rational Metareasoning for LLMs (2410.05563):** VOC reward (quality − compute cost) trained via Expert Iteration — the oracle objective's ancestor; single-model, non-agent, online estimate rather than hindsight argmax. Not in the plan's related work — must be.
- **Implicit PRM (2412.01981) + "GRPO is Secretly a PRM" (2509.21154) + GiGPO/SPA-RL/SWEET-RL:** rollout-free process rewards exist; the O(K×T²)-vs-O(T) framing cannot pretend MC rollouts are the only alternative.

---

## Verdict per CASSI claim

| CASSI claim | Verdict | Evidence |
|---|---|---|
| C1. First self-reinforcing cycle (oracle → stopper → process reward → executor RL → better trajectories) | **PARTIALLY DONE** | Generic policy↔reward co-evolution loops with step-level rewards exist for agents (Self-Guide 2604.03098; Cooper; SPARK; RLAnything). Hindsight step labels → process reward → executor RL exists (DASH math; RePro agents; HCAPO). **The cost-aware stopping instantiation of the loop: NOT FOUND.** "First cycle" phrasing is falsifiable as written. |
| C2. Separate stopping model is *necessary* (representation conflict) | **NOT FOUND as a tested claim — but empirically endangered** | No prior work argues/tests it. However, single-model cost-awareness succeeds broadly (EAPO, SlimSearcher, DASH, ALP, Rational Metareasoning, BudgetThinker), and separate *inference* stoppers are commodity (TERMINATOR, LYNX, LearnStop, Ares, SupervisorAgent). The ablation could easily go the wrong way; DASH shows the bridge works with *no* learned stopper at all. |
| C3. O(T) hindsight oracle labels vs O(K×T²) MC rollouts (+3 formal properties) | **PARTIALLY DONE** | Hindsight-optimal stop labels from completed trajectories: DONE for CoT (TERMINATOR HORL — near-identical to t* under binary quality; LYNX forced exits; LearnStop checkpoints; DASH) and for agents in adjacent form (BAGEN rollout-replay; Ares minimal-effort labels; RePro). Rollout-free process rewards: DONE (Implicit PRM, GRPO-as-PRM, GiGPO, SPA-RL, SWEET-RL, SWE-TRACE rubric PRMs). **The quality−λ·cost argmax oracle with λ-monotonicity properties and a trained Δ-value stopper: NOT FOUND verbatim.** The efficiency framing survives only as "cost-aware stopping labels for free," not "cheap PRM training" in general. |
| C4. Per-instance dynamic cost adaptation beats static penalties | **LARGELY DONE (as a phenomenon); CASSI's specific mechanism NOT FOUND** | ALP (per-prompt solve-rate-scaled penalty), LASER-D/DAST (difficulty-aware), AdaCoT (Pareto adaptive CoT triggering), DLER, EAPO (difficulty-aware tool penalties), SlimSearcher AEA (per-group minimal-path anchor on GAIA). The "static penalty" strawman is outdated; H5-style difficulty-compute correlation already demonstrated in this family. Mid-trajectory *stepwise* adaptation via a learned stopper remains open. |
| C5. Small stopper supervises large executor, <3% overhead | **DONE in components** | Small inference-time monitors over larger agents: Ares (1.7B router, 52.7% cost cut), SupervisorAgent (29.7% GAIA tokens), Dynasor/Certaindex probes. Small models providing *training* rewards for large: AgentPRM (3B), Prover-Verifier Games, RLT. The specific combination (small stopper as cost-aware PRM) is unclaimed, but this is not independently a contribution — it inherits novelty from C1/C3. |

---

## Closest prior work table

| Paper | Year | Venue | Overlap | Key Difference |
|---|---|---|---|---|
| DASH (2607.00482) | 2026 | arXiv (Capital One) | Post-hoc per-step GT checks on completed traces → segment-level GRPO advantages; escalating penalty past first-correct ≈ post-t* STOP; zero extra rollouts; explicit cheap-PRM framing | Math CoT only; no cost/λ/budget; no stopping model; no controller; no cycle |
| Self-Guide (2604.03098) | 2026 | arXiv (under review) | Explicit co-evolving policy↔internal-reward loop; step-level rewards in GRPO; language agents; inference + training use of same signal | Self-generated (no separate stopper); cost-blind; no stopping; no oracle anchoring |
| TERMINATOR (2603.12529) | 2026 | arXiv | Hindsight-optimal exit labels from completed trajectories; trains lightweight stopper; Pareto SOTA | Inference-time exit only; CoT; binary "first arrival," no quality−λ·cost; no executor training |
| LYNX (2512.05325) | 2025 | arXiv | Learned probe stopper on forced-exit GT labels + conformal guarantees | Inference-only, math CoT, hidden-state probe, no cost model |
| When Does Learning to Stop Help? (2606.30852) | 2026 | arXiv | Learned stopper (LearnStop) on GT checkpoint labels; cost accounting incl. probe overhead; matched-risk protocol | Study/benchmark; inference-only; finds learned stopping wins only in some regimes — a methodological bar CASSI must clear |
| SlimSearcher (2606.07074) | 2026 | arXiv (ZJU/Ant) | Per-instance adaptive efficiency reward (AEA) in agent RL on GAIA; anti-fixed-penalty motivation; 17–58% tool-call cuts | Single model; outcome-level; no learned stopper/Δ; no budget state |
| EAPO (2606.02132) | 2026 | arXiv (CASIA/ByteDance) | Difficulty-aware tool-abuse penalties in agentic RL; "when not to act" | Single model; tool-use frequency, not stopping; no monitor |
| BAGEN (2606.00198) | 2026 | arXiv (Northwestern) | Budget-awareness formalized + trained (SFT+RL); multi-dim budgets; hindsight replay labels; SWE-bench/Search-R1; early-stop saves 28–64% | Self-estimation capability; aborts failures, not optimal stopping; no PRM bridge |
| CaRT (2510.08517) | 2025 | arXiv (CMU) | Trains termination ("know when enough"); counterfactual pairs + rationale | Same-model SFT; no cost λ; no process reward; no executor RL |
| Ares (2603.07915) | 2026 | arXiv (UCSB) | Small learned per-step effort controller for agents, trained on trajectory-mined minimal-effort labels; 52.7% cost cut | Discrete effort routing; frozen executor; no stop decision; no reward bridge |
| AgentPRM (2502.10325) | 2025 | arXiv (Cornell) | PRM for agents; small RM supervises policy; iterated actor-critic | Quality-only; MC-rollout labels; no cost/stopping |
| AgentPRM-WWW (2511.08325) | 2026 | WWW 2026 | Step-wise promise+progress PRM for agents (TD-style, cheaper than MC) | Quality-only; inference-time beam-search guidance |
| Rational Metareasoning (2410.05563) | 2024 | arXiv/NeurIPS wksp | VOC reward = quality − cost penalty in RL; adaptive compute per instance | Single model; online VOC, not hindsight oracle; CoT, not agents |
| MRT (2503.07572) / AnytimeReasoner-BRPO (2505.13438) | 2025 | arXiv | Dense progress rewards / budget-sampled dense rewards for budget-agnostic reasoning | Single model; progress needs rollouts (MRT); no stopper; math |
| RePro (2606.14302) | 2026 | arXiv (SJTU/OPPO) | Retrospective per-step labels from completed trajectories as dense agent RL rewards | Progress-only; no cost; no stopping; self-generated |
| SupervisorAgent (2510.26585) | 2026 | ICLR 2026 | Lightweight monitor over agents cuts ~30% tokens on GAIA | Runtime intervention only; no training; no learned stop labels |
| CoRL (2511.02755) | 2025 | arXiv (UIUC/Apple) | RL controller with dual task+cost rewards, budget-conditioned modes | Orchestration/routing; frozen executors; no stopping |
| Implicit PRM (2412.01981) / GRPO-is-a-PRM (2509.21154) / GiGPO / SPA-RL | 2024–25 | ICLR'25 etc. | Process rewards without MC rollouts | Quality-only; undercut the O(K×T²) framing, not the cost-stopping idea |

---

## Overall: is there a kill-shot?

**No single kill-shot found.** No paper trains a cost-aware stopping value model on quality−λ·cost hindsight
labels and uses it as a process reward to train a tool-using executor, with budget-conditioned inference
enforcement. I verified the 14 closest candidates at PDF level; each is missing at least two of
{cost objective, stopping decision, separate trained stopper, executor training bridge, agents}.

**But the composite risk is severe — a "distributed kill":** a knowledgeable reviewer can stack
DASH (hindsight step labels → GRPO advantages, zero extra rollouts) + TERMINATOR/LYNX (hindsight-optimal
stop labels → small learned stopper) + Self-Guide/Cooper (policy↔reward co-evolution loop for agents) +
SlimSearcher/EAPO/ALP (per-instance adaptive cost rewards in agent RL) + BAGEN (trainable budget-aware
early stopping on SWE/search agents) and argue CASSI is a recombination. Three of the five stated
contributions are phrased in ways that are already falsifiable ("first cycle," "static penalties,"
"O(K×T²) is the alternative").

- **Novelty survival score: 5.5/10** (composite unclaimed; every pillar individually crowded; window closing fast — DASH is 15 days old, and CaRT/MRT authors are one step away).
- **Recommendation: PROCEED WITH CAUTION** — reposition claims, add the new baselines, and move fast.
- **Single most dangerous citation a reviewer could raise: DASH (arXiv 2607.00482)** — "hindsight
  step-quality labels from completed trajectories as step-level GRPO training signal that teaches the model
  when to stop, with zero extra rollouts, explicitly motivated as a cheap PRM alternative" — it removes the
  need for both the MC-rollout strawman and (potentially) the learned stopper. Runner-up: Self-Guide
  (2604.03098) against the "first self-reinforcing cycle" claim.

---

## What space remains (the defensible gap)

1. **The cost-aware stopping bridge:** no prior work converts an explicit quality−λ·cost hindsight optimum
   into (a) a trained stopping/value model that (b) emits a continuous stop-margin Δ used as (c) a process
   reward for executor RL on tool-using agents. DASH has (a)-without-cost directly as advantages; TERMINATOR
   has (a)+(b) inference-only; Self-Guide has (c) without cost or stopping. The triple composition is open.
2. **Explicit λ-controllable Pareto navigation** (λ-monotonicity of t*, budget-tier-conditioned λ) — no
   surveyed method exposes a principled per-instance cost-sensitivity dial tied to hindsight optima;
   BudgetThinker/L1 control length, not value-of-continuation.
3. **Multi-dimensional real cost (tokens + tool fees + dollars) in the stopping objective** — reasoning-domain
   work is token-only; agent-domain work (OTC, EAPO, SlimSearcher) counts calls but has no dollar-denominated
   stopping value; BAGEN measures but does not optimize stopping quality.
4. **Executor-internalized stopping vs. inference-time stopping, measured head-to-head** — the
   controller-only vs. reward-bridge ablation would be the first clean answer to whether training beats the
   (now strong: SupervisorAgent, TERMINATOR) inference-time monitor class.

---

## Improvement suggestions to dodge the closest prior art

1. **Rewrite Contribution 1.** Drop "first self-reinforcing cycle" (falsified by Self-Guide, Cooper, SPARK,
   RLAnything as a structure). Claim: "the first *cost-aware stopping* cycle — hindsight quality−cost optima
   supervise a stopping value model whose Δ trains the executor." Cite the co-evolution family explicitly.
2. **Add a DASH-style direct-shaping baseline (P0).** Oracle labels injected directly as step advantages in
   GRPO, *no learned stopper* (DASH adapted to agents). This is now the single most important ablation: if
   direct shaping matches CASSI, the stopper is dead weight; if CASSI wins (generalization to unlabeled
   states, inference-time control, transfer), the two-model design earns its complexity. Frame it as such.
3. **Add SlimSearcher-AEA, EAPO, and ALP as adaptive-penalty baselines** — the plan's home-made "Adaptive-α"
   baseline will be seen as a weak stand-in for published adaptive methods on the same benchmarks (GAIA).
4. **Reframe the O(K×T²) claim.** Acknowledge Implicit PRM, GRPO-as-PRM, GiGPO, SPA-RL, SWEET-RL,
   AgentPRM-WWW's TD variant, and SWE-TRACE rubric PRMs as existing rollout-free process signals; CASSI's
   efficiency point should be narrowly "cost-aware *stopping* labels are free given intermediate quality,"
   not "PRM training is otherwise O(K×T²)." Also disambiguate the two AgentPRMs (2502.10325 vs 2511.08325).
5. **Cite and absorb the learned-stopper family** (TERMINATOR, LYNX, LearnStop/2606.30852, Thought
   Calibration): position CASSI's stopper as the first *trained-for-training* stopper (reward model), not the
   first learned stopper. Consider adopting 2606.30852's matched lost-correct-risk protocol and probe-overhead
   accounting — reviewers from that line will demand it (and it preempts the "monitor overhead" question).
6. **Cite BAGEN, CoRL, xRouter, Rational Metareasoning (2410.05563), MRT/BRPO** — all absent or
   under-cited in the current plan; VOC in particular is the intellectual ancestor of the oracle objective
   (hindsight VOC is an honest one-line acknowledgment that costs nothing).
7. **Soften "representation conflict" from theory to hypothesis.** The single-model successes (DASH, EAPO,
   SlimSearcher, ALP) mean the two-model claim may fail; pre-register the fallback (stopper wins on transfer,
   controllability, and no-ground-truth inference rather than raw Pareto).
8. **Lead with the agent + real-dollar-cost setting** (SWE-bench, tool fees, budget tiers): every close
   competitor lives in math CoT or counts only calls; multi-dimensional dollar-denominated stopping value is
   the cleanest visible daylight — and MATH-500 should stay only as the low-slack control, never a headline.
9. **Move fast.** DASH (Jul 1), RePro (Jun 12), SlimSearcher (Jun 5), BAGEN (May 29) are all ≤7 weeks old;
   the composite gap is obvious enough that CMU (CaRT/MRT group) or the RAGEN group could close it within a
   cycle.
