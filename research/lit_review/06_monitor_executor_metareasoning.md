# Monitor-Executor Architectures & Meta-Reasoning

> Area 06 — systems where a separate module/agent monitors, critiques, or steers an executor;
> meta-reasoning and metacognition for LLMs. Researched 2026-07-16. All PDFs in `research/papers/`.
> Critical questions: (a) does any monitor make COST/BUDGET decisions? (b) is any monitor's output
> used as a TRAINING reward for the executor? (c) does any work argue for/against separating
> monitoring from execution (CASSI's "representation conflict")?

## Area overview

The monitor-executor idea now spans four distinct strands. (1) **Meta-thinking / metacognitive
architectures** (ReMA, MGV, Dolores, the Liu et al. 2026 survey) separate a meta-level that plans,
monitors, and regulates from an object-level that executes, explicitly invoking Flavell (1979) and
Nelson & Narens (1990). ReMA trains both levels with multi-agent RL; MGV is a pure formal theory;
Dolores is an inference-time decomposition scaffold. None of these is cost-aware — Dolores actually
spends 12.9x more tokens to buy accuracy. (2) **Runtime supervisors** (SupervisorAgent ICLR 2026,
plus safety monitors like CoT-Guard and ProbGuard) watch an executor's trajectory and intervene.
SupervisorAgent is the closest inference-time competitor to CASSI's controller mode: an untrained,
heuristically-triggered GPT-4.1 supervisor cuts GAIA token spend ~30% at unchanged success rate.
(3) **Reward-model monitors that train executors** (Agent-RRM, MaR, MISE): here the monitor's
judgment *is* the RL training signal — but every instance rewards reasoning *quality* (knowledge
coverage, regulation fidelity, critique-based scores), never cost, stopping, or budget. (4)
**Hierarchical / budget-controllable RL** (HiPER ICML 2026, CoRL, Plan-and-Act, CoAct): planner-
executor splits trained with RL or SFT. CoRL is the only one with an explicit budget in the reward
(a hard cost gate for a routing controller); HiPER shows the meta/object split can live inside a
*single* policy via structured decisions plus hierarchical advantage estimation.

Against CASSI's claims the landscape decomposes cleanly. Question (a): monitors that make cost
decisions exist at inference (SupervisorAgent) and cost-aware *controllers* trained with RL exist
for routing (CoRL), but no monitor issues per-step stop/continue decisions over an executor's
trajectory from a learned value-of-continuation. Question (b): monitor-output-as-training-reward is
now an established pattern for quality (Agent-RRM's GRPO scores + critiques; MaR's metacognitive
grader under DAPO; MISE's hindsight self-evaluation), so CASSI cannot claim the *bridge* itself —
only the cost-aware, stopping-specific, oracle-derived version of it. Question (c): separation is
argued repeatedly (ReMA: single-agent meta-thinking has "low efficacy"; Nelson-Narens' asymmetric
meta/object levels; Dolores' cognitive-load argument), but never as CASSI's specific
"execution-features vs economic-self-evaluation representation conflict," and two strong recent
results cut the other way — ReMA's multi-turn variant works best with *shared* weights, and HiPER
gets SOTA hierarchy with a *single* policy — so CASSI's two-model necessity claim needs its
ablation to carry real evidentiary weight.

The net picture: each ingredient of CASSI exists somewhere (separate meta-module; runtime cost
supervision; monitor-as-reward; budget-conditioned RL; continuation-value advantages), but no work
combines a *small trained economic monitor* + *per-step cost-aware process rewards* + *executor RL*
into one loop. The threat is a reviewer assembling the pieces; the opportunity is that the
intersection is genuinely unoccupied.

## Core papers

### ReMA: Learning to Meta-think for LLMs with Multi-agent Reinforcement Learning (Wan et al., 2025, NeurIPS 2025; arXiv:2503.09501)

- **Read from:** PDF pages 1-9 (`2503.09501_rema-meta-think-marl.pdf`)
- **Problem:** Single-agent training of meta-thinking (monitoring, evaluating, controlling one's
  own reasoning) is inefficient: one forward pass must master both meta-thinking and detailed
  problem solving, causing poor exploration and early convergence to local optima.
- **Method:** Decouples reasoning into a **high-level meta-thinking agent** π_h (plans, monitors,
  revises strategy; emits meta-thoughts m) and a **low-level reasoning agent** π_l (executes under
  those instructions). Single-turn: two *separate copies* of the same base LLM; iterative RL
  alternates freezing one and updating the other, each maximizing the shared task reward (answer
  correctness + format), REINFORCE++. Multi-turn: **parameter sharing** — one set of weights with
  two system prompts — trained with multi-turn GRPO using a novel turn-level importance ratio. The
  monitor sees prompt + interaction history; its output is *context* for the executor, not a reward.
- **Training / RL usage:** Both agents RL-trained against environment reward. The meta-agent is NOT
  a reward source; there is no cost/budget term anywhere.
- **Experiments & benchmarks:** MATH500, GSM8K, AIME24, AMC23, Gaokao2023En, Minerva Math,
  OlympiadBench; RewardBench970, JudgeBench. Llama-3-8B/3.1-8B-Instruct, Qwen2.5-7B-Instruct.
- **Key results:** Avg +6.68% over CoT for Llama3-8B math (26.73 vs 20.05); max +20.0 on AMC23
  (Llama3-8B), +13.33 AIME24 (Qwen2.5-7B), +14.23 RewardBench970 (Llama3.1-8B). Multi-turn: train
  acc 55%→70%, ~+5% avg across 7 test sets. Ablations: turn-level ratio beats token-level; **shared
  parameters with simultaneous updates converge faster than separate weights**; a 1B meta-agent
  collapses to the trivial EMPTY action while 8B adapts strategy to difficulty.
- **Limitations:** Math/judge tasks only (no tool-use agents); multi-turn highly hyperparameter-
  sensitive ("Echo Trap"); no efficiency objective — meta-thinking adds tokens; no stopping decision.
- **Relation to CASSI:** Overlap: separate monitor/executor modules, both RL-trained; explicitly
  argues single-model meta-thinking underperforms (supports separation). Differences: no cost, no
  stopping, monitor steers via text not rewards. Two findings cut against CASSI: shared-weights
  variant is *better* (weakens two-model necessity) and small (1B) meta-models collapse (challenges
  the 0.5B-3B stopper premise, though stopping is a narrower task than open-ended strategy
  generation). THREAT: **MEDIUM** — same architecture family and already in CASSI's plan as a
  baseline (ReMA-cost), but its evidence on weight-sharing and small-model collapse must be
  addressed, not just cited.

### Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems — SupervisorAgent (Lin et al., 2025/2026, ICLR 2026; arXiv:2510.26585)

- **Read from:** PDF pages 1-9 (`2510.26585_supervisoragent-stop-wasting-tokens.pdf`)
- **Problem:** Multi-agent systems waste tokens (verbose observations, repetitive loops) and
  propagate errors; prior work does post-hoc failure attribution, not real-time intervention.
- **Method:** Defines a Supervised Multi-Agent System: a **meta-level control agent (Supervisor)**
  monitors agent-agent, agent-tool, and agent-memory interactions *without altering the base
  agents*. An **LLM-free heuristic adaptive filter** flags three trigger types — error occurrence,
  inefficient behavior (e.g., repeated `page_down` loops), excessive observation length — then a
  prompted supervisor LLM with its own memory and a global-trace context window picks an action:
  `approve`, `provide_guidance`, `correct_observation`, `run_verification`. No stop/terminate
  action; it nudges and purifies rather than deciding termination. Supervisor and executor are
  both large prompted models (GPT-4.1 supervising GPT-4.1); nothing is trained.
- **Training / RL usage:** None. Purely inference-time; no reward to the executor.
- **Experiments & benchmarks:** GAIA validation (Smolagent testbed, GPT-4.1; also Gemini-2.5-pro,
  Qwen3-235B); GSM8k-Hard, AIME2024, HumanEval, MBPP, DROP (Qwen3-32B); MAS-agnostic tests on
  AWorld and OAgents.
- **Key results:** GAIA: **-29.68% avg tokens at identical pass@1 (50.91)**, 527.76K→371.12K;
  -32.39% on Level 2, -30.10% Level 3 (pass@3); token-cost variance -63% (L2). AIME +6.67% accuracy
  with -18.92% tokens; HumanEval -23.74% tokens at slightly higher accuracy. Ablation: observation
  purification drives efficiency (savings drop 29.68%→15.96% without it). Supervisor overhead
  **15.45% of total tokens** (all savings reported net); latency +<1.5 min/task. AWorld -36.54%,
  OAgents -39.36% tokens.
- **Limitations:** Hand-crafted heuristic triggers; supervisor is an expensive frontier model; no
  learning, no explicit budget state, no termination decisions, no per-step value estimate;
  occasionally over-compresses contexts and drops accuracy.
- **Relation to CASSI:** Strongest published instance of a *cost-motivated monitor* over a
  tool-using executor, on CASSI's own benchmark (GAIA), with headline savings (~30% at
  iso-accuracy) inside CASSI's promised 20-40% band. Differences: untrained, no stopping decision,
  no reward bridge, 15.45% overhead vs CASSI's <3% claim. THREAT: **HIGH** — an ICLR 2026 paper
  already delivering CASSI's headline efficiency numbers training-free; CASSI must include it (or a
  faithful variant) as a baseline and beat it on the Pareto frontier or show complementarity.

### Monitor-Generate-Verify (MGV): Formalising Metacognitive Theory for Language Model Reasoning (Oh & Gobet, 2025, NeurIPS 2025 Workshop on Foundations of Reasoning in Language Models; arXiv:2511.04341)

- **Read from:** PDF pages 1-7, v3 (`2511.04341_mgv-monitor-generate-verify.pdf`)
- **Problem:** Generate-Verify architectures omit the *monitoring* phase that precedes generation
  (difficulty assessment, confidence, strategy selection), contributing to the "prefix dominance
  trap" (~20% accuracy loss from early commitment to bad reasoning paths).
- **Method:** A computational translation of Flavell (1979) and Nelson & Narens (1990) into
  algorithmic form. Monitoring produces metacognitive experiences (difficulty, feeling-of-knowing)
  that select strategies before generation; verification feeds back to refine monitoring.
  Formalizes Nelson-Narens' **asymmetric meta/object split**: the meta-level maintains a model of
  the object-level, monitors progress, and "controls how long the process is allowed to continue
  (should the search persist or terminate?)" — dual-counter FOK evidence accumulation with
  satisficing thresholds for termination. Positions itself against resource-rational analysis
  (value-of-computation, VOC = expected quality gain minus deliberation cost), which it cites as
  the normative route (Callaway et al. 2022/2024 derive optimal stopping from cost-benefit
  principles; De Sabbata et al. 2024 RaM trains an LLM with reward U − C).
- **Training / RL usage:** None. v3 explicitly states "we present no empirical validation." (The v1
  preprint circulated preliminary GSM8K numbers — 75.42% vs 68.44% SELF-REFINE at 27-37% *higher*
  inference cost — which were removed; treat any such numbers as retracted/unvalidated.)
- **Experiments & benchmarks:** None (theory paper).
- **Key results:** A vocabulary/algorithmic skeleton (Algorithms 1-3) for monitoring, resource
  allocation inversely proportional to confidence, and termination via satisficing thresholds.
- **Limitations:** No implementation, no models, no data; the authors say so themselves.
- **Relation to CASSI:** Conceptual prior art for "a meta-level that decides continue-vs-terminate
  based on monitored signals," and its VOC framing is mathematically CASSI's Δ(s_t) = Q_continue −
  Q_stop. But nothing is learned or evaluated. THREAT: **LOW** — cite it (and the resource-rational
  line: Russell & Wefald, Lieder & Griffiths, Callaway) as theoretical grounding; it strengthens
  rather than blocks CASSI, though it shows the *idea* of learned economic stopping has clear
  intellectual precedent.

### Deep Reasoning in General Purpose Agents via Structured Meta-Cognition — DOLORES (Light et al., 2026, preprint; arXiv:2605.11388)

- **Read from:** PDF pages 1-8 (`2605.11388_dolores-structured-metacognition.pdf`)
- **Problem:** Agent scaffolds (ReAct, CodeAct, Deep Research) hard-code task decomposition in
  advance; they fail when the task requires adapting the *structure* of reasoning itself.
- **Method:** A formal language ("Deep Reasoning") that represents meta-reasoning as executable
  decompositions along three axes: associative vs formal (D1), object- vs meta-level (D2), atomic
  vs monolithic (D3). DOLORES instantiates it: a modeling LLM m decomposes each sentence-task into
  a small formal program mixing LLM calls (associative), Python (formal), and recursive DOLORES
  self-calls, executed in a REPL; in-context examples are formalized *human meta-reasoning traces*.
  Meta-level and object-level run in separate LLM context threads (same underlying model), lowering
  per-thread cognitive load. Inference-time only.
- **Training / RL usage:** None. No rewards, no fine-tuning.
- **Experiments & benchmarks:** SynthWorlds, PhantomWiki, DeepSearchQA, OOlong-real; Qwen3-8B/32B
  Thinking, Llama-3.3-70B; baselines ReAct, CodeAct, Deep Research, RLM.
- **Key results:** Beats the strongest baseline by **24.8% avg** (best in 11/12 settings; +36.4%
  Qwen3-32B, +12.8% Qwen3-8B, +25.4% Llama-3.3-70B); 8B DOLORES beats 32B baselines in >half the
  settings. Failure analysis of baselines: **premature termination in 78%** and hallucination in
  45% of failed traces. Cost: **12.9x more total tokens** than baselines (per-thread reasoning
  tokens -71%, non-reasoning -76%), acknowledged as a limitation.
- **Limitations:** Enormous token overhead; no learning; no budget awareness; decomposition
  quality depends on hand-formalized human traces.
- **Relation to CASSI:** Overlap: meta-level/object-level separation with the meta-level deciding
  structure; its premature-termination failure statistics are useful *motivation* evidence for
  CASSI's claim that stopping is a first-class failure mode. Differences: cost-increasing rather
  than cost-aware, inference-only, no monitor rewards. THREAT: **LOW** — orthogonal axis
  (decomposition, not economics); also a useful foil showing meta-reasoning without an economic
  term explodes cost.

### Metacognition as Reward: Reinforcing LLM Reasoning via Knowledge and Regulation Signals — MaR (Chen et al., 2026, preprint; arXiv:2605.23384)

- **Read from:** PDF pages 1-8 (`2605.23384_mar-metacognition-as-reward.pdf`)
- **Problem:** RLVR's outcome rewards leave intermediate reasoning unconstrained; rubric rewards
  need per-instance hand design. Wanted: *general* process-reward dimensions.
- **Method:** Scaffolds each rollout into Metacognitive Knowledge (MK: enumerate task-relevant
  knowledge units), Metacognitive Regulation (MR: an explicit executable plan), optional LOOKBACK
  (recover missing knowledge), and answer. A **frozen large grader LLM π_G (Qwen3.5-397B)** — the
  monitor — scores each rollout using fixed gold-knowledge annotations (GPT-5.1-generated):
  reward R = KMR + RMR + CR, where KMR = (k+r)/n knowledge coverage, RMR = a(1−λs) regulation
  fidelity with shortcut penalty (λ=0.3), CR = answer correctness. Trajectory-level reward feeds
  **DAPO** (G=8, group-normalized advantages) to train Qwen3.5-4B/9B. Monitor and executor are
  separate models; the monitor only exists at training time.
- **Training / RL usage:** YES — the monitor's metacognitive assessment IS the RL reward
  (explicitly inspired by Nelson 1990 monitoring-control). No cost/budget/length term at all.
- **Experiments & benchmarks:** 22 benchmarks across science (GPQA-Diamond, SuperGPQA, ...),
  medical, long-context (DocQA, LongMIT, LongReward), math (AIME 2024/25/26, MATH500), logic;
  ~32K training samples from RaR-Medicine/Science.
- **Key results:** Up to **+7.7% over base** and **+11.0% over vanilla DAPO**; Qwen3.5-9B+MaR avg
  67.6%, surpassing GPT-OSS-120B; DAPO alone gives only +0.4% (64.6→65.0). OOD process scores:
  KMR +22.1, RMR +19.1, CR +7.9 (DocQA). Math OOD: AIME24/25/26 +8.7/+8.0/+8.7 (9B).
- **Limitations:** Grader is 397B — far larger than the 4B/9B policy (opposite of CASSI's
  small-supervises-large); trajectory-level not per-step; static QA reasoning, not tool-using
  agents; no efficiency dimension (MK/MR scaffold adds tokens).
- **Relation to CASSI:** Occupies "metacognitive monitoring signal as RL reward." CASSI's wording
  ("monitor output as process reward") must be positioned against this: CASSI's monitor is small,
  cheap, per-step, *economic* (stop/continue value), and deployed at inference as a controller —
  none of which MaR does. THREAT: **MEDIUM** — paradigm overlap, dimension disjoint.

### Controlling Performance and Budget of a Centralized Multi-agent LLM System with Reinforcement Learning — CoRL (Jin et al., 2025, preprint; arXiv:2511.02755)

- **Read from:** PDF pages 1-8 (`2511.02755_corl-budget-controller-rl.pdf`)
- **Problem:** Decentralized multi-LLM systems call every model per query — uncontrolled inference
  cost. Want cost-efficient *and* cost-controllable coordination.
- **Method:** A centralized **controller LLM (Qwen2.5-7B-Instruct, trained)** decides per query
  whether to answer itself or decompose and dispatch sub-queries to a pool of **frozen expert
  LLMs** (o3, GPT-4.1, GPT-4.1-nano). Trained with **PPO** (GAE, learned value model, expert
  tokens loss-masked). **Reward r_φ = r_p x r_c**: task accuracy times a hard budget gate (r_c = 1
  iff dollar cost c(y) ≤ B, else total reward 0). Multi-budget training conditions the prompt on a
  budget level (low/medium/high), each with its own B, so one system exhibits budget-dependent
  behavior at inference.
- **Training / RL usage:** The *controller* is RL-trained with an explicitly cost-gated reward;
  executors receive no training signal (left to future work).
- **Experiments & benchmarks:** Deepscaler (40,315 train); MATH500, AMC2023, AIME2024, AIME2025;
  per-query dollar costs reported.
- **Key results:** High-budget CoRL beats the best single expert on all four sets: MATH500 **0.958
  vs o3 0.938** (cost $5.87 vs $5.64/set), AMC 0.997 vs 0.984, AIME24 0.877 vs 0.871, AIME25 0.867
  vs 0.842. Low-budget mode 0.900 on MATH500 at $4.65 mostly via the 7B controller (alone: 0.708).
  Expert-call ratio ordered low<medium<high; under B=0.001 the controller learns to avoid
  over-calling o3 (reward zeroed when over budget).
- **Limitations:** Single-turn routing on math only; binary cost gate (no marginal cost shaping);
  no monitoring of an *ongoing* trajectory, no stopping decisions; experts frozen.
- **Relation to CASSI:** Proves "small trained model + RL + explicit budget conditioning + frozen
  large executors" works — the closest thing to a *trained cost-aware supervisor*, but it routes
  before execution rather than monitoring during it, and no signal flows into executor training.
  THREAT: **MEDIUM-HIGH** — with Ares/SeqRoute/Router-R1 it crowds the "budget decisions by a
  small RL-trained controller" space; CASSI's wedge is per-step stopping + the reward bridge.

### Exploring Reasoning Reward Model for Agents — Agent-RRM / Reagent (Fan et al., 2026, preprint v2; arXiv:2601.22154)

- **Read from:** PDF pages 1-8 (`2601.22154_agent-rrm-reasoning-reward-model.pdf`)
- **Problem:** Agentic RL relies on sparse outcome rewards that can't differentiate intermediate
  reasoning quality; step-level PRMs are annotation-hungry and hackable; pairwise RMs give no
  actionable guidance.
- **Method:** **Agent-RRM**, a *trained* reward model (init Qwen3-8B; SFT on 28K then GRPO on 90K
  judgment data; trajectories from an ensemble, annotated by GPT-OSS-120B) that reads a full
  agentic trajectory and emits `<think>` (trajectory analysis) + `<critique>` (targeted flaws:
  missing browse steps, tool inefficiency, etc.) + `<score>` ∈ [0,1] — works without ground truth.
  Executor "Reagent" (Qwen3-8B, 6 tools: search, browse, Python, file, image, audio; SFT 55.6K then
  GRPO on 709K). Three integrations: **Reagent-C** — inference-time critique→refine (policy
  frozen); **Reagent-R** — GRPO with R_i = R_rule + λ·R_model (λ=0.3), the RM score added to the
  rule-based outcome reward; **Reagent-U** — critiques generate refined rollouts, initial+refined
  pooled with unified advantages in GRPO.
- **Training / RL usage:** YES — a separate trained monitor's scalar score (and critiques) directly
  shape GRPO training of a tool-using executor. Purely quality-based; no cost/length/budget terms;
  trajectory-level (not per-step).
- **Experiments & benchmarks:** GAIA (text + full), WebWalkerQA, HLE, xbench; HotpotQA, 2Wiki,
  Bamboogle, MuSiQue; AIME24/25, MATH500, GSM8K. 8x A800-80G.
- **Key results:** Reagent-U (8B): **GAIA text avg 43.7** (59.0/38.5/16.7 by level) and
  **WebWalkerQA 46.2**, vs 34.0/43.5 for Reagent w/o RRM; HLE 10.8, xbench 43.0, Bamboogle 76.8,
  AIME24 60.0. Reagent-R alone: GAIA 36.9, WWQA 45.3. λ sweep: plateau λ∈[0.2,0.4], decline at 0.5
  — over-weighting the monitor's process signal at the expense of outcome hurts.
- **Limitations:** RM is the same size as the policy (8B/8B); no efficiency dimension — nothing
  stops the agent from spending more; trajectory-level credit only; RM labels distilled from a
  120B annotator, not from hindsight-computable quantities.
- **Relation to CASSI:** The occupied bridge: separate trained monitor → reward → executor GRPO on
  GAIA/WebWalkerQA (CASSI's exact benchmarks). CASSI's remaining novelty on this axis is strictly
  the *content* of the signal (cost-aware stopping value from O(T) oracle labels, per-step) and the
  monitor's deployment as an inference-time controller. THREAT: **HIGH** — must be cited, compared,
  and ideally used as a baseline (Agent-RRM + cost term ≈ CASSI's "AgentPRM-cost" slot).

### HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for LLM Agents (Peng et al., 2026, ICML 2026, PMLR 306; arXiv:2602.16165)

- **Read from:** PDF pages 1-8 (`2602.16165_hiper-hierarchical-rl-credit.pdf`)
- **Problem:** Flat RL for multi-turn agents propagates sparse end-of-trajectory credit across tens
  of thousands of tokens — unstable optimization, inefficient credit assignment.
- **Method:** Plan-Execute factorization *inside a single autoregressive policy*: at every turn the
  LLM emits `<switch>` (KEEP/SWITCH the current subgoal — a binary termination decision for the
  running option), `<subgoal>`, `<action>`. **Hierarchical Advantage Estimation (HAE)**: a shared
  critic backbone with two heads — V_low(s,o) (return given current subgoal commitment) and
  V_high(s) (return at subgoal-boundary decision points) — yields three advantages: A_low (GAE
  within a segment, boundary-bootstrapped to V_high), A_high (segment-level GAE over macro-steps),
  and **A_switch = (q_t − β_t)·δ_switch with δ_switch = V_high(s_t) − V_low(s_t, o_{t−1})** — the
  estimated gain of terminating the current subgoal versus continuing it. PPO-style updates;
  proofs: unbiased gradient (up to GAE/critic error) and variance reduction vs flat GAE.
- **Training / RL usage:** Actor-critic; the "monitor" is a learned critic pair used only for
  training-time advantages (classic RL critic, not a deployable supervisor). No cost terms.
- **Experiments & benchmarks:** ALFWorld, WebShop; Qwen2.5-1.5B/7B-Instruct; baselines PPO, GRPO,
  GiGPO.
- **Key results:** 7B: **ALFWorld 97.4%** (+6.6 over GiGPO 90.8), **WebShop 83.3%** (+8.3 over
  75.2). 1.5B: ALFWorld 95.3 (GiGPO 86.7, GRPO 71.1, PPO 68.2), WebShop 71.4. Largest gains on
  long-horizon multi-subtask cases.
- **Limitations:** No cost/budget dimension; hierarchy is within one policy (no independent
  monitor); embodied/shopping environments, not open-web research agents.
- **Relation to CASSI:** δ_switch = V_high − V_low is the same mathematical object as CASSI's
  Δ = Q_continue − Q_stop, applied to subgoal termination instead of task stopping and without
  cost. Also demonstrates that hierarchical meta/object decisions do NOT require separate weights —
  direct counter-evidence CASSI's representation-conflict ablation must engage. THREAT: **MEDIUM**
  — quality-only and single-model, but a reviewer can ask "why not HiPER + cost-in-reward?"; CASSI
  needs an answer (e.g., single-model cost-penalty GRPO is exactly its planned ablation arm).

## Peripheral papers

**Plan-and-Act (Erdogan et al., 2025, ICML 2025; arXiv:2503.09572).** Separate **Planner** and
**Executor** models: the planner emits structured high-level plans, the executor grounds them into
environment actions; both fine-tuned on synthetic plan annotations reverse-engineered from
successful trajectories (plus dynamic replanning). SOTA **57.58% on WebArena-Lite** and 81.36%
text-only WebVoyager. Relation: canonical two-model planner-executor split with SFT, but the
planner is a *feedforward* strategist — it never monitors cost, never evaluates continuation value,
and provides no training signal to the executor. LOW threat; useful as the "separation without
economics" contrast.

**CoAct (Hou et al., 2024, preprint; arXiv:2406.13381).** Two-agent hierarchy: a **global planning
agent** (macro-plan, sub-task descriptions) and a **local execution agent**; on failure the global
agent *re-arranges the process trajectory* (replanning) on WebArena. Early (2024) evidence that a
meta-level watching an executor's failures helps; prompted only, no training, no cost. LOW threat.

**Metacognition in LLMs: Foundations, Progress, and Opportunities (Liu et al., 2026, preprint;
arXiv:2607.11881).** First comprehensive survey (97+ papers, 2021-2025; taxonomy: measuring,
findings, implementing — incl. for agents — improving capabilities, applications, broader
directions). Reports metacognitive interventions yield ~3-20% task-dependent gains; frames
monitoring→control as the core loop. Useful citation hub for CASSI's metacognition framing and for
checking nothing in its "implementing metacognition in agents" section does cost-aware stopping
(nothing listed there does — closest entries are confidence calibration and knowledge-boundary
work). LOW threat, HIGH citation value.

**TRIAGE: Evaluating Prospective Metacognitive Control in LLMs under Resource Constraints (Al Nazi
& Dipta, 2026, preprint; arXiv:2605.13414).** Evaluation-only framework: given a task pool and a
token budget calibrated to the model's own baseline cost, the model must commit — *before any
execution feedback* — to a plan of which problems to attempt, in what order, with what per-problem
token allocation; scored against a solvability/cost oracle (triage efficiency ratio). Finding:
frontier and open models show substantial gaps in prospective metacognitive control. Relation:
directly documents the capability gap CASSI's stopper is meant to fill (economic self-allocation),
but trains nothing and stays pre-execution. LOW threat, good motivation citation.

**Combining Cost-Constrained Runtime Monitors for AI Safety (Hua et al., 2025, NeurIPS 2025;
arXiv:2507.15886).** Given several safety monitors with known cost/performance, computes the
optimal *monitoring protocol* — when to call which monitor and when to intervene — under an
average-case budget, via exhaustive search + Neyman-Pearson allocation; more than doubles recall vs
naive baselines in a code-review setting. The budget governs the **monitoring pipeline itself**,
not the executor's task spend; still, it is the most formal "cost-aware monitor" result to date and
its framing (spend on monitors vs interventions) is adjacent to CASSI's monitor-overhead (<3%)
claim. LOW-MEDIUM threat.

**CoT-Guard: Small Models for Strong Monitoring (2026, preprint; arXiv:2605.12746).** Finds 4B-8B
models fail as CoT misbehavior monitors out of the box, then fixes them with **SFT (distilling from
stronger monitors) + RL on hard hidden-objective examples** — a small-trained-monitor pipeline
almost identical in shape to CASSI's stopper training (SFT→RL), but for security (detecting hidden
objectives in code tasks), not economics. Evidence that small trained monitors can supervise larger
executors when the monitoring task is narrow. LOW threat, useful feasibility precedent.

**MISE: Utilizing and Calibrating Hindsight Process Rewards via Reinforcement with Mutual
Information Self-Evaluation (2026, preprint; arXiv:2604.11611).** The agent's own **hindsight
generative self-evaluation** becomes a dense process reward, calibrated against environmental
feedback; proves hindsight self-evaluation reward ≡ mutual-information objective + KL to a proxy
reward policy ("first formal foundation for generative self-rewarding"). ~7B agents reach GPT-4o-
level validation performance without expert supervision. Relation: occupies "hindsight-computed
dense process rewards for agent RL" — adjacent to CASSI's post-hoc oracle labels — but
self-evaluation by the same model, quality-only, no cost, no stopping. MEDIUM-LOW threat: cite to
sharpen the claim that CASSI's hindsight signal is *economic* (quality − λ·cumcost) and produced by
a separate deployable controller.

## Synthesis

**Landscape.** Monitor-executor work has professionalized fast in 2025-2026 along three nearly
disjoint tracks: (i) *metacognitive separation* (ReMA, MGV, Dolores, survey) — strong on
architecture and psychology, silent on economics; (ii) *runtime supervision* (SupervisorAgent,
safety monitors, cost-constrained monitor scheduling) — increasingly cost-literate but untrained
and reward-free; (iii) *monitor-as-reward RL* (Agent-RRM, MaR, MISE; plus critic-style HiPER and
budget-gated CoRL) — trains executors or controllers, but every reward is about quality except
CoRL's binary routing budget gate. CASSI sits at the empty intersection of all three.

| Paper (year, venue) | Monitor decides what? | Cost-aware? | Monitor output trains executor? | Monitor vs executor size |
|---|---|---|---|---|
| ReMA (2025, NeurIPS) | strategy/plan text per turn | No | No (context only; both RL-trained on task reward) | equal 7-8B; multi-turn shares weights |
| SupervisorAgent (2026, ICLR) | approve / guide / purify / verify at flagged junctures | Yes (token efficiency; -29.7% GAIA) | No (inference-only) | equal & large (GPT-4.1 over GPT-4.1); 15.45% overhead |
| MGV (2025, NeurIPS WS) | (theory) strategy selection; termination via satisficing | conceptually (VOC) | No | n/a |
| Dolores (2026, preprint) | task decomposition into threads | No (12.9x tokens) | No | same model, recursive |
| MaR (2026, preprint) | scores MK/MR/answer per rollout | No | **Yes** (DAPO trajectory reward) | monitor 397B >> executor 4-9B |
| CoRL (2025, preprint) | route to expert vs answer itself | **Yes** (hard $ gate, multi-budget) | No (controller is the trainee; experts frozen) | controller 7B << experts (o3) |
| Agent-RRM (2026, preprint) | trajectory score + critique | No | **Yes** (GRPO reward + critiques) | 8B = 8B |
| HiPER (2026, ICML) | KEEP/SWITCH subgoal (critic advantages) | No | Yes (critic → advantages; standard actor-critic) | critic = value heads on shared backbone |
| Plan-and-Act (2025, ICML) | high-level plan (feedforward) | No | No (both SFT) | comparable |
| CoAct (2024, preprint) | global plan + replanning on failure | No | No | equal, prompted |
| TRIAGE (2026, preprint) | (eval) select/order/allocate under budget | Yes (eval-only) | No | n/a |
| Cost-constr. monitors (2025, NeurIPS) | when to call which safety monitor | Yes (monitoring budget) | No | monitors cheaper than intervention |
| CoT-Guard (2026, preprint) | flag hidden objectives in CoT | No (cost motivates small monitor) | No | **4-8B monitor for larger executors (SFT+RL)** |
| MISE (2026, preprint) | hindsight self-scores each step | No | **Yes** (self-reward, calibrated) | same model |

**Gaps (all verified empty as of 2026-07):** (1) no monitor makes *per-step stop/continue* economic
decisions over a running executor from a learned continuation-value; (2) no monitor-derived
*cost-aware* signal is used as a training reward — the reward-bridge papers are quality-only, and
the budget papers gate a router, not a process reward; (3) no work derives monitor training labels
from *hindsight oracle stopping points* (MISE's hindsight self-eval is the nearest neighbor);
(4) nobody closes the loop monitor→reward→better executor→better labels→better monitor; (5) the
"small monitor supervising a large executor" configuration exists only untrained (SupervisorAgent
is large-on-large; CoT-Guard is small but safety-only; MaR's grader is huge).

**Top threats to CASSI's novelty (ranked):**
1. **Agent-RRM** — already runs "separate trained monitor → GRPO reward → tool-using executor" on
   GAIA/WebWalkerQA; CASSI's bridge claim survives only as "first *cost-aware, stopping-focused,
   per-step*" version. Must cite + compare.
2. **SupervisorAgent (ICLR 2026)** — delivers CASSI's headline outcome (≈30% cost cut at
   iso-accuracy on GAIA) training-free; if CASSI cannot beat its Pareto point (and its 15.45%
   overhead with the promised <3%), the training machinery looks unjustified.
3. **CoRL (+ Router-R1/Ares/SeqRoute cluster)** — "small RL-trained controller making explicit
   budget decisions over frozen large models" exists; CASSI must sharpen that routing-before-
   execution ≠ monitoring-during-execution.
4. **HiPER** — V_high − V_low switching advantage ≈ Δ(s_t) inside a single policy with provable
   credit-assignment benefits; undermines "two models are necessary" unless CASSI's ablation
   (single-model cost-penalty GRPO vs two-model) is decisive.
5. **MaR** — "metacognition as reward" branding overlaps CASSI's story; disjoint in substance
   (no cost, giant grader) but must be positioned early to avoid reviewer conflation.
6. **ReMA** — its shared-weights superiority and 1B-meta-agent collapse are *empirical* challenges
   to CASSI's separation claim and small-stopper premise respectively.

**Opportunities:** (a) frame CASSI as the missing intersection with the table above — every row
has at most two of {separate monitor, cost-aware, trains executor, small-over-large}; (b) use
MGV/resource-rational citations (VOC, Callaway's optimal-stopping metalevel MDP) as normative
grounding for Δ; (c) use Dolores' 78% premature-termination failures + TRIAGE's prospective-control
gap + SupervisorAgent's loop/verbosity taxonomy as motivation evidence; (d) borrow CoT-Guard's
SFT→RL small-monitor recipe as feasibility precedent; (e) adopt Agent-RRM's λ-sweep finding
(process-signal weight plateau then decline) to justify CASSI's α tuning for the Δ term; (f) report
monitor overhead against SupervisorAgent's 15.45% to make the <3% claim land.
