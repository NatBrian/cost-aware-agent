# Budget-Constrained & Cost-Aware Agent Frameworks

> Area 04 — budget tracking/reporting, cost-aware tool use, dollar-cost optimization, cost-sensitive
> inference for LLM agents. Researched 2026-07-16. All core papers verified on arXiv and read from
> downloaded PDFs in `research/papers/`.
> Critical questions probed: (a) does any framework LEARN cost-awareness via RL rather than prompt
> heuristics? (b) does any work reward-shape tool-call cost during agent RL training?
> **Short answers: (a) YES — many (2024–2026); (b) YES — a fast-growing 2025–2026 line.** Details below.

## Area overview

The field has moved through three visible generations. **Gen 1 (2023–2024): system-level dollar-cost
optimization with frozen models.** FrugalGPT (cascades under an explicit dollar budget), EcoAssistant
(assistant hierarchy + solution reuse) and Budget-Constrained Tool Learning with Planning (knapsack/DP
tool-quota planning) treat cost as an orchestration problem around unmodified LLMs. **Gen 2 (2024–2025):
cost-aware behavior enters the model, first heuristically, then via RL.** SMART fine-tunes on curated
rationales for when tools are necessary; CATP-LLM (ICCV 2025) fine-tunes an LLM planner with *offline*
RL whose reward subtracts per-step tool execution cost; OTC-PO (Apr 2025) is the canonical *online* RL
work that multiplies the correctness reward by a tool-efficiency coefficient, explicitly penalizing
excess tool calls; xRouter and CoRL (Oct/Nov 2025) train routers/controllers with success-gated,
dollar-cost-penalized rewards. In parallel, Google's BATS (Nov 2025) shows that pure prompt-level
budget tracking ("Budget Tracker" plug-in) already shifts the cost-performance Pareto frontier for
frozen search agents.

**Gen 3 (2026): budget/cost-awareness as a first-class *trained* capability and as adaptive reward
shaping in agent RL.** By mid-2026 there are: benchmarks that measure whether agents *know* their
budgets (BAGEN, CostBench); inference-time planners that enforce hard monetary budgets with learned
world models (INTENT) or budget-conditioned tree search with step-level value estimates (BAVT); and —
most important for CASSI — agent-RL methods whose cost penalties are *adaptive* rather than static:
SlimSearcher anchors the tool/token penalty to the cheapest correct trajectory within each GRPO group;
EAPO scales the tool penalty by empirical query difficulty within the rollout group; AdaTIR does
difficulty-aware advantage shaping; IterResearch uses geometric reward discounting ("Efficiency-Aware
Policy Optimization"). The claim "prior cost penalties are static and no one learns cost-awareness"
is therefore no longer defensible as stated.

What remains genuinely open: all training-based methods above shape cost at the **trajectory/outcome
level** (group-normalized or gated scalar rewards); none (i) train a **separate lightweight
stopping/value model** on **post-hoc oracle stopping labels** (argmax_t quality−λ·cumcost), (ii) use
its cost-aware value margin as a **per-step process reward** for executor RL, or (iii) close a
**two-model self-reinforcing loop**. BAGEN comes closest in spirit on the labeling side (rollout-replay
prefix relabeling; SFT+RL of an early-stop/alert capability) but never feeds the estimator back into
executor training. That is the residual gap CASSI occupies — and it is narrower than the paper plan
currently assumes.

## Core papers

### Budget-Aware Tool-Use Enables Effective Agent Scaling — "BATS" (Tengxiao Liu et al., 2025, arXiv 2511.17006; Google Cloud AI Research + UCSB + Google DeepMind, preprint)
- **Read from:** PDF pages 1–6 (`2511.17006_bats-budget-aware-tool-use.pdf`)
- **Problem:** Test-time scaling for tool agents: granting more tool-call budget does not improve
  frozen agents — they "lack budget awareness" and hit a performance ceiling (ReAct saturates at
  budget 100 on BrowseComp).
- **Method:** (1) **Budget Tracker**: a plug-and-play prompt block appended each iteration showing
  per-tool used/remaining budget ("Tool1 Budget Used: ##, Remaining: ##…"). (2) **BATS** framework:
  budget-aware planning + verification that decides "dig deeper" vs "pivot" from remaining resources.
  Budget enters **purely at the prompt level — heuristic, zero training**. Also formalizes a
  **unified dollar cost metric** C_unified = token cost (provider pricing, cache-aware) + Σ tool
  calls × per-call price.
- **Training / RL usage:** None. Frozen Gemini-2.5-Pro/Flash, Claude-Sonnet-4.
- **Experiments & benchmarks:** BrowseComp, BrowseComp-ZH, HLE-Search; budgets 10–200 calls/tool;
  constrained optimization formulation max Acc s.t. c_i ≤ b_i.
- **Key results:** Budget Tracker lifts ReAct accuracy at same budget: Gemini-2.5-Pro 12.6→14.6
  (BrowseComp), 31.5→32.9 (ZH), 20.5→21.8 (HLE-Search); Claude-Sonnet-4 12.2→14.0. With 10× less
  budget (10 vs 100 calls): 12.8% vs 12.6% acc, 40.4% fewer searches, 21.4% fewer browses, unified
  cost −31.3% (6.8¢ vs 9.9¢/query). ReAct+Tracker keeps scaling to budget 200 while ReAct plateaus.
- **Limitations:** No learning; relies on the frozen model's ability to condition on a budget string;
  trajectory-level tool budgets only; no stopping-quality signal (only budget exhaustion or answer).
- **Relation to CASSI:** CASSI already plans BATS (+grid-searched variant) as a baseline — correct
  title/ID above. It supplies the unified dollar-cost metric CASSI should adopt for comparability.
  **THREAT: MEDIUM-LOW** — strong evidence budget awareness matters, but purely prompt-level; CASSI's
  learned stopper must beat it at iso-cost, which its own ceiling analysis suggests is feasible.

### Acting Less is Reasoning More! Teaching Model to Act Efficiently — "OTC-PO" (Hongru Wang et al., 2025, arXiv 2504.14870; CUHK + UIUC + Princeton, preprint under review)
- **Read from:** PDF pages 1–6 (`2504.14870_otc-acting-less.pdf`)
- **Problem:** Tool-integrated reasoning RL (Search-R1 style) optimizes only final correctness →
  excessive tool calls, "cognitive offloading," high cost.
- **Method:** Posits a per-(question, model) **optimal (minimal) tool-call count n**; reward =
  correctness **multiplied** by a tool-efficiency coefficient: r_φ^tool = α · r_tool · r_φ(q,y).
  OTC-PPO: r_tool = cos(mπ/(2m+c)). OTC-GRPO: approximates n by the **minimum tool calls k=min(C)
  among correct trajectories in the group** (updated across epochs toward a global optimum), then
  r_tool via a cos/sin map of f(m,n)=2nm/(m+n). Multiplicative gating chosen explicitly to prevent
  reward hacking (additive was unstable). Introduces **tool productivity** TP = #correct / #tool calls.
- **Training / RL usage:** Online RL (PPO and GRPO variants) on Qwen2.5-3B/7B-Base and
  Qwen2.5-Math-1.5B/7B. **Cost-awareness is fully learned via reward shaping** — self-claimed "the
  pioneering RL-based framework that explicitly optimizes for both efficiency and effectiveness of
  tool-integrated reasoning."
- **Experiments & benchmarks:** NQ + HotpotQA training (search agent, Search-R1 setting); ToRL code
  setting; baselines SFT, Base-RL, RAG, IRCoT, Search-R1, ToRL.
- **Key results:** Up to **68.3% fewer tool calls**, up to **215.4% higher tool productivity**, with
  comparable accuracy; e.g., the trained model answers trivial questions with 0–1 calls where
  Search-R1 uses 2–3.
- **Limitations:** Cost = raw tool-call count (no dollar/token heterogeneity); trajectory-level
  outcome reward (no per-step credit); the "oracle" n is the group empirical minimum, not a
  quality-cost trade-off; single-model; short-horizon QA.
- **Relation to CASSI:** The closest well-known ancestor of "reward-shape tool cost in agent RL."
  Its group-minimum k is a crude hindsight cost oracle — conceptually adjacent to CASSI's t* but with
  no quality−λ·cost trade-off, no stopping labels, no separate model, no process reward. **THREAT:
  HIGH** — reviewers will treat "cost-penalized agent GRPO" as solved by OTC; CASSI must include
  OTC-GRPO as a baseline (currently absent from the plan's baseline list) and differentiate on the
  process-reward/stopper axis.

### SlimSearcher: Training Efficiency-Aware Web Agents via Adaptive Reward Gating (Zequn Xie et al., 2026, arXiv 2606.07074; Zhejiang University + Ant Group, preprint)
- **Read from:** PDF pages 1–7 (`2606.07074_slimsearcher-reward-gating.pdf`)
- **Problem:** Deep-research agents trained with accuracy-only rejection sampling / RL exhibit "blind
  tool dependency" and "performative reasoning" (redundant loops), causing an "efficiency collapse"
  where RL scales up search rounds to force correctness.
- **Method:** Two stages targeting the **Minimal Necessary Path (MNP)**. (1) **Pareto-efficient SFT
  filtration**: from 13,863 seed trajectories, keep per query the correct trajectory maximizing
  r_tool×r_len. (2) **RL with multiplicative cascading gates**: R_final = r_correct · r_tool · r_len.
  Gate 1 binary correctness (zero reward if wrong — anti reward-hacking). Gate 2 **Adaptive Efficiency
  Anchoring**: tool cost C(τ)=Σ w_type(a) is compared to the **empirical minimum C_min within the
  sampled GRPO group**, δ=(C−C_min)/(C_min+ε), mapped to a bounded multiplier 2·exp(−δ)/(exp(−δ)+1) ∈
  (0,1]; Gate 3 same for token length vs group L_min. Explicitly designed to avoid the brevity bias of
  fixed/static penalties. Standard GRPO advantage on top.
- **Training / RL usage:** SFT + GRPO on Tongyi-DeepResearch-30B and Qwen3-30B-A3B-Instruct-2507;
  64 H800 GPUs; Serper API + Jina Reader tools. **Cost-awareness fully learned; penalty is adaptive
  (group-relative), not static.**
- **Experiments & benchmarks:** GAIA, BrowseComp, XBench-DeepSearch, HLE. Metrics: accuracy,
  tool-call rounds, tokens.
- **Key results:** Tool-call rounds −17%…−58% with equal or better accuracy. Tongyi-DeepResearch
  backbone: GAIA rounds 20.56→10.61 (−48.4%), tokens −33.4%, acc 0.682→0.709; BrowseComp rounds
  63.70→47.63, acc 0.410→0.447; XBench acc 0.713→0.790 with rounds 14.26→5.92. Qwen3 backbone HLE:
  rounds 27.86→19.51, acc 0.259→0.278. Prompt-control baseline fails to improve efficiency.
- **Limitations:** Outcome-level (trajectory) reward — no per-step process reward, no stopping
  decision head; anchor is within-group relative (needs G rollouts; no explicit budget state or λ);
  no dollar costs (uniform tool weights); no controller at inference; no difficulty conditioning
  beyond SFT data filtering.
- **Relation to CASSI:** The single most direct competitor on "cost-shaped agent RL for deep
  research," on CASSI's headline benchmark (GAIA) with a 30B executor and Pareto-improvement framing.
  Overlap: hindsight minimal-cost anchoring (their MNP ≈ empirical cost oracle), multiplicative
  correctness gating, GRPO. Differences: no learned stopping model, no oracle stopping labels, no
  process rewards, no budget conditioning, no inference-time controller. **THREAT: HIGH** — CASSI's
  "first to make agent RL cost-aware" framing is dead; the defensible claim is the stopper-as-PRM
  loop and per-instance λ/budget adaptation, and SlimSearcher must be a baseline (or at least its
  group-relative reward as the "single-model GRPO+cost penalty" ablation instantiation).

### Learning When Not to Act: Mitigating Tool Abuse in Agentic Reinforcement Learning — "EAPO" (Liuji Chen et al., 2026, arXiv 2606.02132; NLPR CAS + ByteDance + Zhejiang, preprint)
- **Read from:** PDF pages 1–6 (`2606.02132_learning-when-not-to-act.pdf`)
- **Problem:** Agentic RL induces *tool abuse*; but **uniform tool penalties or hard caps (i.e., OTC-style)
  suppress useful exploration on hard queries** — the right target is *unnecessary* tool use.
- **Method:** Efficient Agentic Policy Optimization: (1) **efficiency-aware rollouts** — force ≥n
  tool-free trajectories per group of M (M=16, n=2) so advantage estimation sees direct evidence of
  whether internal reasoning suffices; (2) **difficulty-aware reward shaping** — query difficulty
  d(q)=1−(1/M)Σ𝕀(correct) from the group; shaped reward R̃_i=[d(q)+(1−d(q))·ε(T_i)]·R_i with
  ε(T_i)=exp(−β(c_i−c_min)) anchored to the group's minimum tool count: **the cost penalty scales
  inversely with difficulty — easy queries get penalized hard for tool use, hard queries barely**;
  (3) confidence-aware token-level advantage reweighting (up-weight low-confidence tokens on correct
  trajectories, down-weight overconfident tokens on wrong ones).
- **Training / RL usage:** GRPO-family online RL; Qwen2.5-3B/7B-Instruct, Llama3.1-8B; 10K SFT cold
  start. **Adaptive (difficulty-conditioned) learned cost penalty.**
- **Experiments & benchmarks:** Math: AIME24/25, MATH500, GSM8K, MATH; knowledge: HotpotQA, 2Wiki,
  MuSiQue, Bamboogle. Baselines: GRPO, Reinforce++, ToolStar, ARPO, AEPO.
- **Key results:** vs GRPO: avg performance +10.45% / +7.27% / +9.69% on the three backbones while
  cutting avg tool calls −18.33% / −18.33% / −24.59%. Qwen2.5-7B knowledge avg F1 51.6 vs 46.8
  (GRPO) with TC 2.08 vs 2.87; Qwen2.5-3B math avg Pass@1 55.0 at TC 0.97 vs 51.7 at 1.13.
- **Limitations:** Difficulty estimated from group success rate (needs G rollouts per query, online
  only); outcome-level shaping; tool-count cost only; no stopping model or budget state; no
  long-horizon agents (≤ a few calls).
- **Relation to CASSI:** Directly undermines CASSI's blanket characterization of prior RL penalties
  as "static": EAPO (and AdaTIR) are per-instance difficulty-adaptive penalties, i.e., partial
  overlap with CASSI claim 4 (per-instance dynamic cost adaptation) — though via group statistics,
  not a learned per-state value estimate. No process reward, no separate model, no stopping.
  **THREAT: HIGH** (for claim 4 and the related-work framing; MEDIUM for the core loop).

### CATP-LLM: Empowering Large Language Models for Cost-Aware Tool Planning (Duo Wu et al., 2024→2025, ICCV 2025, arXiv 2411.16313; Tsinghua SIGS)
- **Read from:** PDF pages 1–6 (`2411.16313_catp-llm-cost-aware-tool-planning.pdf`)
- **Problem:** LLM tool planners (HuggingGPT-style, vision/audio tool DAGs) ignore tool execution
  costs (time, memory), producing plans whose cost outweighs benefit.
- **Method:** (1) Tool Planning Language: tools + dependencies as learnable tokens; DAG plans as
  token sequences (enables non-sequential/concurrent plans). (2) **Cost-Aware Offline RL (CAORL)**:
  fine-tunes the LLM as a **decision transformer** with **per-step intermediate rewards**:
  r_i = −(1−α)C(p_i) for non-terminal actions and αP(p_i)−(1−α)C(p_i) at end-of-plan (α=0.5) —
  i.e., every added tool immediately incurs its cost in the reward. Context augmentation embeds
  per-tool cost attributes (across input sizes) with importance encoding. Evaluation metric
  QoP = αP_task − (1−α)C_price with FaaS-style pricing (execution time + memory).
- **Training / RL usage:** Offline RL (decision transformer) + LoRA on Llama2-7B; reward model for
  plan performance. **Learned cost-awareness; cost enters per step.**
- **Experiments & benchmarks:** OpenCATP platform: 87 sequential (from OpenAGI) + 24 non-sequential
  tasks × 100 input sizes → 8,700 + 2,400 eval points; 10 open-source tool models; baselines GPT-3.5,
  GPT-4 (HuggingGPT prompting), fine-tuned and RL variants (TRICE, RLTF).
- **Key results:** Llama2-7B CATP-LLM beats GPT-4 planning: 1.02–1.41× higher QoP (sequential),
  1.92–5.99× (non-sequential); ~28.2–30.2% higher plan performance with 24.7–45.8% lower execution
  cost vs GPT-4 (v1 reporting); overall plan-quality gains 1.5–93.9%.
- **Limitations:** Not a ReAct/interactive agent: plans are generated then executed (no environment
  feedback loop, no stopping decision); costs are execution time/memory, not tokens/dollars; offline
  DT (no on-policy improvement cycle); small task space.
- **Relation to CASSI:** Historical precedent (first "coherent" cost-aware tool planning via RL) and
  the earliest *per-step cost reward* — CASSI's "no one gives step-level cost signals" phrasing must
  be scoped to *learned value-based process rewards for interactive executors*. **THREAT: MEDIUM** —
  different setting (static DAG planning), but a reviewer-citable "cost-aware RL for tools" prior.

### xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning (Cheng Qian et al., 2025, arXiv 2510.08439; Salesforce AI Research + UIUC, preprint)
- **Read from:** PDF pages 1–9 (`2510.08439_xrouter-cost-aware-routing.pdf`)
- **Problem:** Static escalation rules / heuristic routing under-utilize the cost-performance
  spectrum of model catalogs (GPT-5 family, o3, Gemini-2.5, open models).
- **Method:** A tool-calling **router LLM** (Qwen2.5-7B-Instruct) that answers directly or invokes
  external models (with prompt/temperature hints), trained end-to-end with an explicitly economic
  reward: **R_final = R_binary × (K − λC)** — success-gated ("no success, no reward; on success,
  cheaper is better"), C = total dollar-normalized cost of all model invocations, K success bonus,
  λ cost-penalty strength (three trained variants λ1/λ2/λ3). Full cost accounting (per-call token
  prices, per-turn and per-episode). Difficulty-stratified training data (Reasoning360, pass@k of
  Qwen3-32B).
- **Training / RL usage:** DAPO (GRPO-style) in VERL, max 3 turns. **Dollar-cost-awareness fully
  learned in the reward.**
- **Experiments & benchmarks:** Minerva, MATH-500, Olympiad Bench, AIME-24/25, AMC-23, Codeforces,
  Code-Contests, HumanEvalPlus, LiveCodeBench v5, GPQA-Diamond, MTBench, IFEval, LiveBench; baselines:
  untrained router models (GPT-4o, GPT-5-mini, Qwen…) and single models.
- **Key results:** xRouter-7B-λ2 reaches near-GPT-5 accuracy on Olympiad Bench at ~1/8 the cost;
  e.g., AIME25 0.767 @ $0.0106/query vs GPT-5 0.733 @ $0.0142; consistently on the cost-performance
  Pareto frontier vs static routing; trained router ≫ untrained Qwen2.5-7B router. Also documents
  failure modes: small open models struggle to learn sophisticated orchestration; some architectures
  train unstably.
- **Limitations:** Routing/delegation, not agentic tool loops; ≤3 turns; no stopping problem; cost
  shaping trajectory-level; router accuracy on hard reasoning limited by its 7B backbone.
- **Relation to CASSI:** Shows "learned dollar-cost trade-off via RL with an explicit λ" exists
  (their K−λC ≈ CASSI's quality−λ·cost at trajectory granularity) and that a *small* model can make
  economic decisions about *large* models. Differences: no per-step value/stopping, executor pool
  frozen, no process rewards, no loop. **THREAT: MEDIUM** — strong prior for "learned economic
  judgment," different mechanism and problem.

### Spend Less, Reason Better: Budget-Aware Value Tree Search for LLM Agents — "BAVT" (Yushu Li, Wenlong Deng, Jiajin Li, Xiaoxiao Li, 2026, arXiv 2603.12634; UBC + Vector Institute, preprint)
- **Read from:** PDF pages 1–6, 9–13 (`2603.12634_bavt-budget-aware-value-tree.pdf`)
- **Problem:** Test-time scaling treats compute as abundant; budget-aware prior work is either
  fine-tuning (doesn't transfer to agent workflows) or trajectory-level prompt heuristics (BATS)
  that cannot intervene mid-execution; agents burn budget on dead-end paths.
- **Method:** Training-free tree search over agent states with an explicit **budget state
  b_t=(b_tool, b_token)** decremented per action (resource-constrained deterministic decision
  process). (1) **Step-level value critic**: the same LLM prompted as critic predicts *residual value
  deltas* (relative progress / marginal information gain) rather than absolute quality — to combat
  self-evaluation overconfidence; enables pruning of uninformative tool calls. (2) **Budget-conditioned
  node selection**: sampling probability P ∝ V^(1/r) with r = remaining-budget ratio — smooth,
  parameter-free shift from broad exploration (high budget) to greedy exploitation (low budget).
  (3) Theorem 1: probabilistic convergence to a terminal answer under a finite budget bound
  (Chernoff-style argument, M ≥ (1/p_min)(K+√(2K log 1/ε)+2 log 1/ε)).
- **Training / RL usage:** None (explicitly training-free; positions post-training as expensive and
  non-transferable).
- **Experiments & benchmarks:** HotpotQA, 2Wiki, MuSiQue, Bamboogle; GPT-OSS-20B and
  Qwen3-30B-A3B-Instruct-2507; budget tiers Low (5 tool calls / 1–2K tokens), Mid (10 / 2–4K),
  High (20 / 4–8K); baseline: budget-matched parallel-sampling majority vote.
- **Key results:** OSS-20B: BAVT Low-budget avg EM 0.338 > baseline High-budget 0.334 (i.e., beats
  4× the resources); baseline Low 0.194 → BAVT 0.338. Qwen3-30B: baseline plateaus 0.289→0.293
  (Low→High) vs BAVT 0.386 at Low. Strictly superior Pareto frontier at every tier.
- **Limitations:** Prompted (not learned) critic — inherits base-model judgment; per-node critic
  calls add overhead; multi-hop QA only (no SWE/web); no learning signal ever persists; hard budget
  tiers rather than soft quality-cost trade-off.
- **Relation to CASSI:** The strongest *training-free* instantiation of "step-level value + budget
  state drives continue/expand/stop decisions" — structurally the inference-time shadow of CASSI's
  stopper-as-controller (CASSI's "zero-training self-eval prompt" baseline should be upgraded to
  BAVT-style). Differences: no trained stopper, no oracle labels, no executor training, no process
  reward. **THREAT: MEDIUM** — it occupies the "budget-aware value estimates for agents" conceptual
  ground at inference time; CASSI must show learning beats prompting here.

### Budget-Constrained Agentic Large Language Models: Intention-Based Planning for Costly Tool Use — "INTENT" (Hanbing Liu et al., 2026, arXiv 2602.11541; Renmin University Gaoling + SUFE + Baidu, preprint)
- **Read from:** PDF pages 1–8 (`2602.11541_intent-budget-constrained-planning.pdf`)
- **Problem:** Agents must solve multi-step tasks under a **hard monetary budget** in a dynamic tool
  marketplace (heterogeneous, changing per-call prices; stochastic tool outcomes). Prompted cost
  feedback still violates budgets in 32.8% of tasks (GPT-4.1-mini).
- **Method:** Inference-time planning, minimal intervention on a frozen agent. Trains a **language
  world model** W_φ to predict tool outcomes; **Monte Carlo Oracle** does a single lookahead rollout
  and rejects actions whose projected total cost exceeds remaining budget (feedback = simulated
  failure trace). **INTENT** adds intention decomposition: an **intention predictor** ρ̂ (probability
  a tool call satisfies the reasoning's intent), a conditional generator simulating the *ideal
  trajectory* (all calls succeed), and **geometric cost calibration** ĉ_k = Cost(a_k)/ρ̂_k (expected
  cost including retries, pessimistic), accepted iff γΣĉ ≤ B_t (risk preference γ). Reward formalism:
  R(τ) = J(a_K,q)·𝕀[ΣCost ≤ B] (zero if over budget). Notably **argues RL post-training is
  "fundamentally misaligned" with non-stationary tool markets** (prices/tools change at inference;
  retraining prohibitively expensive).
- **Training / RL usage:** Supervised training of the world model / intention predictor from
  interaction logs (a few thousand suffice; log-linear scaling); **no RL, agent frozen**.
- **Experiments & benchmarks:** Cost-augmented StableToolBench (765 tasks, B=50, per-call costs
  ~U(5,50), 20 recalled tools); backbones GPT-4.1-mini and GPT-5-nano; baselines Raw, Prompt,
  DFSDT, BTP (Zheng et al. 2024 quota planning), BATS, MCO.
- **Key results:** GPT-4.1-mini: pass rate 63.8 vs BATS 53.0, BTP 46.4, Prompt 30.9; feasible rate
  100% vs Prompt 67.2%; avg cost 24.9 vs Raw 102.1; token overhead 1.70× vs BATS 4.13×. GPT-5-nano:
  PR 76.0 (BATS 52.8), budget-optimal rate 92.6, FR 100%. Robust to tool-price shifts ±50% and
  budget scaling ¼–4×.
- **Limitations:** Inference-time only; needs world-model training data per market; hard-feasibility
  objective (no marginal quality-cost stopping — termination assumed when information sufficient);
  StableToolBench synthetic prices.
- **Relation to CASSI:** Complementary problem (hard dollar feasibility vs soft quality−λ·cost
  optimality). Its explicit anti-RL argument is a positioning threat CASSI should rebut (CASSI's
  budget-state input + tiers give some price-shift robustness; worth an experiment). **THREAT:
  MEDIUM-LOW** for the core loop; MEDIUM as a competing philosophy ("planning at inference beats
  training") reviewers may raise.

### BAGEN: Are LLM Agents Budget-Aware? (Yuxiang Lin, Zihan Wang et al., 2026, arXiv 2606.00198; Northwestern + UMich + Cornell + Stanford + UT Austin et al., preprint)
- **Read from:** PDF pages 1–6 (`2606.00198_bagen-budget-aware-agents.pdf`)
- **Problem:** Agent cost is measured only post-hoc; can agents *estimate mid-execution* how much
  budget remains needed and whether the task is still finishable ("budget awareness")?
- **Method:** Formalizes **progressive interval estimation**: at every turn k, predict
  [R̂_lo, R̂_hi] over remaining budget or declare `impossible`. **Rollout-replay protocol**: record an
  unconstrained rollout, then re-query the agent on every logged prefix and score predictions against
  realized remaining cost — i.e., *post-hoc relabeling of completed trajectories* (no extra task
  rollouts). Budget modalities: internal (tokens) and external (multi-dim: dollars, weeks, warehouse
  item-weeks). Sub-capabilities: feasibility prediction (macro-F1), early failure detection
  (Fail-F1), interval calibration (coverage × tightness). Then **trains** budget awareness:
  SFT and SFT+RL on Qwen2.5-7B-Instruct (combined reward to prevent collapse) to strengthen early-stop
  and alert behaviors.
- **Training / RL usage:** SFT + RL **of the budget estimator / early-stop behavior** (not of task
  execution). RL without SFT warm-start collapses.
- **Experiments & benchmarks:** Sokoban (2,500-token cap), Search-R1 (3,500), SWE-bench (160 turns),
  Warehouse (real enterprise supply-chain data, 3 coupled external budgets); 5 frontier models
  (GPT-5.2 Instant, Claude Opus 4.7 / Sonnet 4.6, Gemini 3.1 Pro, Qwen3-235B); 128 rollouts/model
  (64 SWE), 2–3K estimation samples per model-task pair.
- **Key results:** Budget awareness decoupled from task skill (success vs interval hit rate r≈0.35;
  Opus best actor on Search-R1 at 75.8% success but Sonnet better estimator, 36.5% vs 23.1% hit).
  All 20 model-env pairs systematically over-optimistic; on failed trajectories models predict >70%
  feasibility even after 60% of budget spent; alarms fire in the final 20%. **Early-stop policy on
  `impossible` predictions saves 28–64% of tokens on failed trajectories at 1.6–4.2 pts success
  cost.** SFT lifts Qwen-7B feasibility accuracy 25.5%→≈90%; interval coverage caps at 47% after
  SFT+RL.
- **Limitations:** Estimation studied offline (replay), not as an online controller (left to future
  work); trains the estimator only — **never bridges it into executor training**; no quality−cost
  trade-off (feasibility, not marginal value of continuing); calibration remains poor (47%).
- **Relation to CASSI:** Closest published relative of CASSI's *oracle-from-completed-trajectories*
  idea: replaying prefixes of finished rollouts to create supervision for a stopping/feasibility
  signal, then SFT+RL-ing a small Qwen on it. Overlap with the stopper's training recipe and the
  early-stop evaluation (their 28–64% savings vs CASSI's 20–40% target). Differences: labels are
  realized-cost/feasibility (not argmax_t[quality−λ·cumcost]), no Δ margin, no process-reward bridge,
  no executor RL, no self-reinforcing cycle. **THREAT: HIGH** on the "trained stopping/budget monitor"
  component and its evaluation protocol; the loop remains CASSI's.

## Peripheral papers

**FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance** (Lingjiao
Chen, Matei Zaharia, James Zou, 2023, arXiv 2305.05176; TMLR 2024). The canonical dollar-budget LLM
paper: prompt adaptation, LLM approximation, and a **learned LLM cascade** that sequentially queries
cheaper→dearer APIs with a trained scoring/stop function under an explicit dollar budget. Matches
GPT-4 accuracy with up to **98% cost reduction** or +4% accuracy at iso-cost. Learned (cascade router
+ scorer) but inference-system-level, single-turn, no agents, no RL, no stopping-in-trajectory.
Foundational citation for "cost as first-class objective"; THREAT LOW.

**EcoAssistant: Using LLM Assistant More Affordably and Accurately** (Jieyu Zhang, Ranjay Krishna,
Ahmed Awadallah, Chi Wang, 2023/2024, arXiv 2310.03046; AutoGen ecosystem). Code-driven QA with (1)
iterative code refinement vs an executor, (2) an **assistant hierarchy** (cheap model first, back off
to GPT-4), (3) retrieval of past successful solutions as demonstrations. Beats GPT-4 by ~10 pts
success at <50% of its cost. Heuristic system-level cascading for agents; no training, no budgets in
rewards. THREAT LOW.

**Budget-Constrained Tool Learning with Planning — "BTP"** (Yuanhang Zheng et al., 2024, arXiv
2402.15960; Findings of ACL 2024). Before execution, estimates tool usefulness from historical
experience, then **dynamic programming** allocates a per-tool invocation-quota plan under a hard
budget (multi-knapsack flavor); plugs into DFSDT etc. Improves tool-learning success under strict
budgets; used as the "BTP" baseline in INTENT (PR 46.4/57.7, feasible 100%). Heuristic planning +
frozen LLM; no RL; static quotas are conservative. THREAT LOW.

**SMART: Self-Aware Agent for Tool Overuse Mitigation** (Qian et al.-adjacent team: Wu, Qian et al.,
2025, arXiv 2502.11435; Findings of ACL 2025). SMART-ER dataset (three domains) with per-step
rationales for when tools are necessary; **supervised fine-tuning** yields SMARTAgent family:
**−24% tool use, +37% performance**, 7B models matching 70B/GPT-4o on this calibration, 1/5 the tool
calls on OOD (GSM8K, MINTQA). Learned but SFT-only (no RL, no reward shaping, no budget state);
addresses overuse, not stopping. THREAT LOW-MEDIUM (evidence "when-to-use-tools" can be distilled
without RL — a cheap baseline CASSI could add).

**IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling** (Alibaba Tongyi + RUC, 2025,
arXiv 2511.07327). Reformulates deep research as an MDP with **Markovian workspace reconstruction**
(evolving report instead of accumulating context, O(1) memory) and **Efficiency-Aware Policy
Optimization (EAPO)** — geometric reward discounting that incentivizes reaching answers with fewer
interactions + adaptive downsampling for distributed RL. +14.5 pp avg over open-source agents on six
benchmarks; scales to 2,048 interactions (3.5%→42.5%); as a prompting paradigm +19.2 pp over ReAct.
Efficiency enters RL only as *discounting* (implicit time cost), not explicit tool/dollar cost, and
no budget state; the plan's citation of IterResearch as "budget-aware heuristic agent" is slightly
off — it is an RL-trained efficiency-discounted agent. THREAT MEDIUM-LOW.

**CoRL: Controlling Performance and Budget of a Centralized Multi-agent LLM System with RL** (Bowen
Jin et al., 2025, arXiv 2511.02755; UIUC + Apple). Trains a **controller LLM** (Qwen2.5-7B-Instruct)
via PPO (expert tokens masked) to decompose/dispatch queries to frozen experts (o3, GPT-4.1,
GPT-4.1-nano). Reward r_φ = r_p · r_c with **r_c = 𝕀[c(y) ≤ B]** — any budget overrun zeroes the
reward — plus **multi-budget-mode training**: prompts conditioned "Answer under budget {low/med/high}"
with tier-specific B, yielding inference-time budget-controllable behavior. On Deepscaler-trained
math evals (MATH500, AMC23, AIME24/25) the system surpasses the best expert in high-budget mode and
stays economical in low-budget mode. Small-controller-supervises-big-experts + budget-tier
conditioning is architecturally the closest cousin of CASSI's tiered budget state; but it routes
rather than stops, and gives no process rewards. THREAT MEDIUM.

**CostBench: Evaluating Multi-Turn Cost-Optimal Planning and Adaptation in Dynamic Environments for
LLM Tool-Use Agents** (Jiayu Liu, Cheng Qian et al., 2025, arXiv 2511.02734). Travel-planning
benchmark with atomic/composite tools of customizable costs and four dynamic blocking events (tool
failures, price changes). Even GPT-5 stays **below 75% exact-match on cost-optimal plans** on hard
tasks, and performance drops ~40% under dynamic conditions. Pure evaluation — no method; useful as
evidence that cost-optimal reasoning is unsolved, and a candidate extra benchmark for CASSI. THREAT LOW.

**AdaTIR: Adaptive Tool-Integrated Reasoning via Difficulty-Aware Policy Optimization** (Zhaiyu Fang,
Ruipeng Sun, 2026, arXiv 2601.14696). RL with a **difficulty-aware efficiency reward** (penalty
proportional to task easiness) plus **Clipped Advantage Shaping** to fix the "sign reversal problem"
where tool penalties outweigh correctness. Tool calls −97.6% on simple / −28.2% on complex tasks at
equal-or-better accuracy; +4.8% on AIME-24 even with tools disabled (internalized reasoning).
Together with EAPO it establishes difficulty-adaptive cost penalties in 2026 RL — directly relevant
to CASSI's claim 4 framing. THREAT MEDIUM (framing).

**Translate-R1** (Jayarao et al., 2026, arXiv 2606.06835). Confidence-gated GSPO for **cost-sensitive
translation-tool use** in multilingual QA (Qwen3-4B, 22 languages): learn to call translation only
when comprehension fails. Reward gains +4.6/+23.5/+17.5 (High/Low/XLow resource tiers); 63% of full
reward at baseline cost; Pareto-optimal across 87% of the cost-sensitivity range. Another 2026 data
point that "cost-aware single-tool use via RL" is spreading domain-by-domain. THREAT LOW.

**Budget-Aware Agentic Routing via Boundary-Guided Training — BoPO** (Caiqi Zhang, Menglin Xia,
Xuchao Zhang et al. (Microsoft), 2026, arXiv 2602.21227). Trains a per-step small-vs-large model
routing policy for agent workflows: SFT on cost-efficient trajectories, then RL with
**boundary-relative rewards** (always-small vs always-large boundary policies define a difficulty
taxonomy) under strict per-task budgets; improves the cost-success frontier at substantially lower
cost. Learned budget-constrained step-wise routing (not stopping). THREAT LOW-MEDIUM.

**Reinforcing Real-world Service Agents: Balancing Utility and Cost in Task-oriented Dialogue — CMPO**
(Ning Gao et al., 2026, arXiv 2602.22697). Cost-aware Multi-turn Policy Optimization with a
**PID-Lagrangian cost controller** steering the policy along the Pareto boundary between user reward
and global cost constraints, plus "generative process credits" for multi-turn credit assignment;
persona-driven simulated users; outperforms baselines on real business scenarios and tool-agent-user
benchmarks. Notable as constrained-RL (Lagrangian) treatment of cost in an agent — a more principled
alternative to fixed λ that CASSI's λ-tier machinery should be compared against conceptually.
THREAT LOW-MEDIUM.

**ContextBudget / BACM-RL** (Yong Wu et al., 2026, arXiv 2604.01664). Treats **context management**
(what to keep/compress in the window) as a budget-constrained sequential decision problem; curriculum
RL learns compression strategies across varying context budgets; >1.6× gains over strong baselines in
high-complexity multi-objective QA and long-horizon browsing. Budget-conditioned RL, but over memory,
not actions/stopping. THREAT LOW.

Also noted (no full entries): **BAMAS** (2511.21572) and **AgentBalance** (2512.11426) — budget-aware
*multi-agent system topology* selection; **ZEBRA** (2605.20485) — zero-shot budgeted resource
allocation for LLM orchestration; **Timely Machine** (2601.16486) — time-budget-aware test-time
scaling; **Token Economics for LLM Agents** (2605.09104) — dual computing/economics survey (source of
the CoRL description; documents $47K-weekend multi-agent invoices); **ATLAS** (2606.01667) —
test-time learning-to-allocate scaling; **PruneTIR** (2605.09931) — inference-time tool-call pruning;
**AdaSearch** (2512.16883) — RL balancing parametric knowledge vs search.

## Synthesis

### Landscape classification

| Method (year) | Cost signal | Heuristic vs learned | Where cost enters | Inference-only vs training | Stopping? | Separate monitor model? |
|---|---|---|---|---|---|---|
| FrugalGPT (2023) | dollar | learned (cascade scorer) | router/cascade threshold | training (scorer) | cascade exit | scorer heads |
| EcoAssistant (2024) | dollar | heuristic | system hierarchy | inference-only | no | no |
| BTP / Zheng (2024) | dollar (per-call) | heuristic (DP over stats) | pre-execution plan | inference-only | quota exhaustion | no |
| SMART (2025) | tool calls | learned (SFT) | training data rationales | training (SFT) | no | no |
| CATP-LLM (ICCV'25) | exec time+memory | **learned (offline RL/DT)** | **per-step reward** | training | plan end token | reward model (offline) |
| OTC-PO (2025) | tool calls | **learned (PPO/GRPO)** | outcome reward coefficient | training | no | no |
| xRouter (2025) | **dollar** | **learned (DAPO)** | outcome reward K−λC | training | no | router is the trained model |
| CoRL (2025) | dollar vs budget B | **learned (PPO)** | outcome reward gate 𝕀[c≤B]; budget-tier prompts | training | no | small controller over frozen experts |
| BATS (2025) | unified dollar (tokens+tools) | heuristic (prompt) | prompt (budget block) | inference-only | budget exhaustion | no |
| IterResearch (2025) | interactions (implicit) | **learned (RL discounting)** | geometric discount | training | no | no |
| CostBench (2025) | dollar (synthetic) | — (benchmark) | — | — | — | — |
| AdaTIR (2026) | tool calls | **learned, difficulty-adaptive** | advantage shaping | training | no | no |
| BAVT (2026) | tool+token budget state | heuristic (prompted critic) | node-selection exponent V^(1/r) | inference-only | value-threshold termination | prompted critic (same LLM) |
| INTENT (2026) | dollar (hard budget) | learned world model; heuristic control | lookahead cost oracle (accept/reject) | inference-only (WM trained) | feasibility-based | world model + intention predictor |
| SlimSearcher (2026) | tool+token (group-relative) | **learned, group-adaptive** | multiplicative outcome gates | training (SFT+GRPO) | no | no |
| EAPO / When-Not-to-Act (2026) | tool calls | **learned, difficulty-adaptive** | shaped outcome reward | training | no | no |
| BAGEN (2026) | tokens + external dollars/time | **learned (SFT+RL) estimator** | replay-labeled estimation targets | training (of estimator) | **early stop on `impossible`** | trained 7B estimator (offline replay) |
| BoPO routing (2026) | dollar per-step model choice | **learned (SFT+RL)** | boundary-relative reward | training | no | routing policy |
| CMPO (2026) | global dialogue cost | **learned (Lagrangian RL)** | PID-Lagrangian constraint | training | dialogue end | no |
| **CASSI (proposed)** | tokens/tools/dollars, multi-tier | learned (SFT+GRPO stopper; GRPO executor) | **per-step process reward Δ(s_t) + oracle labels** | training (both models) | **learned marginal-value stopping** | **yes — small stopper supervises large executor** |

### Answers to the critical questions

**(a) Learned cost-awareness via RL?** Yes, extensively: CATP-LLM (offline DT, per-step cost),
OTC-PO (PPO/GRPO, tool coefficient), xRouter (DAPO, dollar λ), CoRL (PPO, budget gate + tier
conditioning), IterResearch (geometric discounting), SlimSearcher (GRPO, group-anchored gates), EAPO
(difficulty-scaled penalty), AdaTIR (difficulty-aware advantage shaping), Translate-R1 (GSPO
confidence gating), BoPO (boundary-relative routing rewards), CMPO (PID-Lagrangian), BACM-RL (context
budgets), BAGEN (SFT+RL of a budget estimator/early-stop). "Cost-awareness only via prompt
heuristics" is false as of 2025; any CASSI claim must be scoped past this.

**(b) Reward-shaping tool-call cost during agent RL?** Yes: OTC-PO is the canonical instance
(Apr 2025); SlimSearcher, EAPO, AdaTIR, and (implicitly) IterResearch extend it with *adaptive*
anchors (group minimum cost, group difficulty, geometric discounting) precisely to fix the
weak-gradient/brevity-bias problems of static penalties. All of these operate at
**trajectory/outcome granularity**: a scalar (possibly shaped) reward at the end, spread over tokens
by GRPO/PPO. None inject a **per-step, state-dependent cost-aware value signal** from a learned
critic (CATP-LLM's per-step cost is the exception, but offline, cost-only — no learned
value-of-continuing — and for static DAG planning, not interactive agents).

### Gaps this area leaves open (CASSI's remaining room)

1. **Per-step cost-aware process reward from a learned value model.** Cost shaping exists only as
   outcome-level scalars; nobody trains a critic that outputs Δ(s_t)=Q_continue−Q_stop and uses it as
   a dense step reward for executor RL.
2. **Oracle stopping labels t\* = argmax_t [quality_t − λ·cumcost_t] from completed trajectories.**
   The nearest relatives are OTC's group-minimum tool count, SlimSearcher's Minimal Necessary Path
   (also group-empirical), and BAGEN's rollout-replay realized-cost labels — none combines
   per-step quality traces with λ-weighted cumulative cost into stopping supervision.
3. **The two-model loop.** CoRL/xRouter train a small controller over frozen executors; SlimSearcher/
   OTC/EAPO train the executor with no monitor; BAGEN trains a monitor with no executor. No published
   work co-trains both and closes the cycle (oracle → stopper → process reward → executor → better
   trajectories).
4. **Learned marginal-value stopping.** BAGEN stops on predicted *infeasibility* of failing runs;
   BAVT stops on prompted value thresholds; nobody learns economically-optimal stopping ("good
   enough, stop now") for succeeding runs — the overthinking half of the problem.
5. **Iso-accuracy dollar accounting for trained agents.** BATS's unified cost metric exists only for
   frozen agents; trained-efficiency papers (SlimSearcher, EAPO, OTC) report rounds/tokens, not
   dollars, and never per-instance budget tiers.

### Top threats to CASSI's novelty (ranked)

1. **SlimSearcher (2606.07074) — HIGH.** Cost-shaped SFT+GRPO deep-research agent with *adaptive*
   reward gating, Pareto improvements on GAIA/BrowseComp/HLE (rounds −17–58%, accuracy up). Directly
   occupies "make agent RL cost-aware" on CASSI's headline benchmark; must be a baseline, and CASSI's
   contribution must be pitched as the stopper-as-PRM loop, not cost-aware agent RL per se.
2. **OTC-PO (2504.14870) — HIGH.** The pioneering, widely-cited RL tool-call penalty (up to −68.3%
   calls). Its group-minimum "optimal tool calls" is a hindsight cost oracle; absent from CASSI's
   current baseline list — reviewers will demand it.
3. **EAPO (2606.02132) + AdaTIR (2601.14696) — HIGH (framing).** Difficulty-adaptive cost penalties
   already exist, so CASSI's "static penalty" characterization of prior RL and part of claim 4
   (per-instance adaptation) need rewording: CASSI's adaptivity is *state-dependent and learned*, not
   group-statistical.
4. **BAGEN (2606.00198) — HIGH (component).** Post-hoc trajectory relabeling to train (SFT+RL) a
   budget-feasibility/early-stop estimator, with 28–64% token savings from early stopping —
   overlaps CASSI's stopper training recipe and headline savings range; CASSI must cite it and
   differentiate via quality−λ·cost labels, the Δ margin, and the executor-training bridge.
5. **CoRL (2511.02755) + xRouter (2510.08439) — MEDIUM.** Small trained controllers making learned
   dollar-cost decisions over big frozen models, with budget-tier conditioning (CoRL) and explicit
   λ-cost rewards (xRouter). Weakens "first small-supervises-large economic controller" claims;
   CASSI's distinction is supervision *of the same executor's stopping* + process-reward transfer.
6. **BAVT (2603.12634) — MEDIUM.** Training-free step-level values + budget-conditioned search;
   the strongest inference-time baseline for "budget-aware stepwise control"; also supplies a
   convergence-guarantee style CASSI's formal-properties section will be compared to.
7. **INTENT (2602.11541) — MEDIUM-LOW.** Hard-budget feasibility via learned world model at
   inference; its explicit "RL post-training is misaligned with dynamic tool markets" argument is a
   philosophical attack CASSI should preempt (e.g., budget-state generalization experiment).
8. **CATP-LLM (2411.16313) — LOW-MEDIUM.** Peer-reviewed (ICCV 2025) precedent for "cost-aware tool
   planning via (offline) RL with per-step cost rewards" — bounds how strongly CASSI can claim
   priority on step-level cost signals.

### Opportunities

- **Position precisely:** "First to convert *post-hoc oracle stopping labels* into a *learned
  cost-aware process reward* that trains the executor" survives this area's literature; every
  broader phrasing ("first cost-aware agent RL," "first budget-aware agent," "first small economic
  controller") does not.
- **Baselines to add from this area:** OTC-GRPO and SlimSearcher-style group-anchored cost GRPO (as
  the strong instantiations of "single-model GRPO + cost penalty"); EAPO/AdaTIR difficulty-adaptive
  penalty (upgrade of the "adaptive-α difficulty-classifier variant"); BAVT (upgrade of the
  "zero-training self-eval" baseline); BAGEN-style feasibility early-stopper (ablation of stopping
  labels: feasibility vs marginal-value).
- **Metrics/benchmarks:** adopt BATS's unified dollar cost for iso-accuracy-cost reporting; consider
  CostBench (dynamic prices) to answer INTENT's non-stationarity critique; report tool-call rounds
  and tokens separately (SlimSearcher convention) for comparability.
- **Differentiating experiments:** (i) outcome-level cost shaping (OTC/SlimSearcher) vs CASSI's
  per-step Δ process reward at matched compute — the process-reward bridge is the claimed win, so
  isolate it; (ii) stopping-error |t_stop − t*| against BAGEN-style feasibility stopping — show
  marginal-value stopping also trims *successful* overlong runs (overthinking), which feasibility
  stopping cannot; (iii) budget-tier conditioning vs CoRL-style prompt tiers.
- **Honest scoping:** the O(T)-vs-O(K×T²) efficiency claim is about PRM label construction and is not
  contradicted in this area (no cost-aware PRMs exist here at all) — but note OTC/SlimSearcher get
  their hindsight cost anchors for free from the same GRPO group, i.e., "zero extra rollouts" alone
  is not unique; the unique part is zero-extra-rollout *per-step value labels*.
