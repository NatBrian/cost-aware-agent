# Competitor Paper Analysis: CASSI vs. Top Related Work

> **CASSI:** "Scalable Cost-Aware Agent Training via Oracle-Guided Stopping Rewards"
> **Target Venue:** ICLR / NeurIPS / ICML
> **28 papers analyzed in full (not abstracts)** — detailed element-by-element comparison tables.
> **Split across 4 files:** Part 1 (Categories 1-2), Part 2 (Categories 3-4), Part 3 (Categories 5-7), Part 4 (Synthesis).

## Summary of CASSI

| Aspect | CASSI |
|---|---|
| **Problem** | LLM agents overthink — no training signal for "this is good enough, stop." Existing solutions: rigid budgets (s1), static length penalties (L1), heuristics (BATS), or Monte Carlo PRMs (AgentPRM) needing O(K×T²) extra executions |
| **Key Innovation** | Oracle-guided stopping labels: `t* = argmax_t [quality_t − λ × cumulative_cost]` — O(T) post-hoc computation requiring zero additional policy executions |
| **Architecture** | Two-model: small stopping model (0.5B–3B) + large executor (7B–72B) |
| **Training** | 3-phase: (1) Collect trajectories, (2) Train stopping model on oracle labels via SFT+GRPO, (3) Train executor via GRPO with stopping model's cost-aware value Δ(s_t) as process reward |
| **Domains** | GAIA, WebWalkerQA, HotpotQA, MuSiQue, SWE-bench Verified, MATH-500 |
| **Expected Results** | 20–40% cost reduction at iso-accuracy; stopping point correlates with task difficulty (r > 0.5); monitor overhead <3%; K×T fewer training executions than AgentPRM |

---

## Category 1: Cost-Aware & Budget-Constrained Agent Frameworks

### Paper 1: BATS — "Budget-Aware Tool-Use Enables Effective Agent Scaling"
**Authors:** Liu et al. (2025) | **URL:** https://arxiv.org/abs/2511.17006

| Element | CASSI | BATS | Contrast |
|---|---|---|---|
| **Background** | LLM agents overthink — no economic judgment. Oracle-guided stopping rewards solve scalability gap in cost-aware PRM training (O(T) vs O(K×T²)) | Test-time scaling for tool-augmented web search agents hits performance ceiling because agents lack "budget awareness." Granting more tool-call budget fails to improve performance | CASSI addresses general agent overthinking with learned solutions; BATS only web search domain with training-free prompts |
| **Motivation** | Compute `t* = argmax[quality − λ×cost]` post-hoc from completed trajectories — zero extra executions. Dynamic per-instance adaptation beats static penalties, making cost-aware PRM training tractable on long-horizon benchmarks | Agents saturate quickly — they prematurely conclude or give up — unaware of unused resources. Need dynamic strategy adaptation based on remaining budget, without any model training | CASSI: training-based solution (learn optimal stopping). BATS: training-free solution (prompt engineering). Both want dynamic adaptation, different means |
| **Approach/Method** | Post-hoc oracle labels for stopping via O(T) computation. Train small stopping model on (state → STOP/CONTINUE) via SFT+GRPO. Use stopping model's cost-aware Δ(s_t) as process reward for executor GRPO | Two-tier training-free: (1) Budget Tracker — prompt-level budget status after each round. (2) BATS Framework — budget-aware planning (constraint decomposition into exploration/verification), tree-structured plan, budget-aware self-verification (CONTINUE/PIVOT/ACCEPT) | CASSI: learned stopping via oracle labels + RL. BATS: heuristic stopping via prompts. Both use two-component design (planning+verification ↔ stopper+executor) |
| **Solution Implementation** | Two-model: small stopping model (0.5B–3B, Qwen2.5-Instruct) + large executor (7B–72B). Multi-dimensional budget (tokens, tools, dollars, iterations, tier). 3-phase pipeline | Pure prompt engineering — no SFT, no RL, no model modification. Gemini-2.5-Flash/Pro, Claude-Sonnet-4. Tools: Google Custom Search, Jina.ai, Crawl4AI. Unified cost metric: tokens + tool calls | CASSI: trained models, multi-domain. BATS: frozen models, web-only. CASSI's cost metric is multi-dimensional with tier system |
| **Training** | Phase 1: executor runs to completion, records per-step state/action/cost/quality. Phase 2: compute oracle t* per trajectory, SFT stopping model, GRPO fine-tune. Phase 3: executor GRPO with stopper rewards (α·Δ + β·progress + γ·format + outcome) | **No training whatsoever.** Improvements come only from better prompt engineering | CASSI learns; BATS can't improve over time |
| **Experiments** | 6 benchmarks across 4 domains: GAIA (466 questions), WebWalkerQA, HotpotQA, MuSiQue, SWE-bench Verified, MATH-500. 3 seeds, paired t-tests, bootstrap CI. 12 baselines | BrowseComp (1,266), BrowseComp-ZH (289), HLE-Search (200). Budget: 3–200 tool uses. Sequential + parallel scaling | CASSI: broader domain coverage. BATS: larger dataset in one domain |
| **Results** | Expected: 20–40% cost reduction at iso-accuracy. Stopping correlates with difficulty (r>0.5). Monitor overhead <3%. K×T fewer training executions than AgentPRM | Budget Tracker alone: 12.6%→14.6% BrowseComp (Gemini-2.5-Pro). 10× less budget: comparable accuracy, 31.3% lower cost. BATS outperforms training-based agents without training. budget=5 surpasses ReAct's best accuracy | CASSI aims higher (20-40% savings vs BATS's ~31% but at 10× budget reduction). BATS shows budget awareness alone helps — validates CASSI's premise |
| **Weaknesses (Theirs)** | N/A — CASSI's: oracle labels depend on executor trajectory quality; stopper may not transfer across executors; domain-specific risk; GRPO instability | Training-free: no mechanism to learn better behavior. Web search only. Verification depends on task structure (weaker on HLE-Search). Relies on heuristics — no learned stopping signal. Cannot learn per-instance adaptive thresholds | CASSI addresses: learned, not heuristic; multi-domain; continuous λ tradeoff; per-instance adaptation through training |
| **How CASSI Extends** | N/A | CASSI replaces heuristic stopping with learned oracle-guided stopping that adapts per-instance. CASSI trains the executor (not just prompts). CASSI covers 6 domains (not just web search). CASSI's λ parameter provides principled cost-quality tradeoff |
| **Future (Theirs → CASSI)** | N/A | Implies: extension beyond web search, training-based methods, learning adaptive thresholds → CASSI directly addresses all three |
| **Relevance** | 4/5 — Validates budget awareness improves efficiency. Strong training-free baseline. Unified cost metric (tokens+tools) is framework CASSI adopts. Limited scope, not a direct competitor |

---

### Paper 2: INTENT — "Budget-Constrained Agentic Large Language Models: Intention-Based Planning for Costly Tool Use"
**Authors:** Liu et al. (2026) | **URL:** https://arxiv.org/abs/2602.11541

| Element | CASSI | INTENT | Contrast |
|---|---|---|---|
| **Background** | LLM agents overthink. Oracle-guided stopping labels (O(T)) make cost-aware PRM training tractable | Formalizes budget-constrained tool use as sequential decision. Tool prices change dynamically; agent must operate under hard budget without retraining | Both address cost-constrained agents. Different scope: CASSI = stopping, INTENT = tool selection under budget |
| **Motivation** | Post-hoc oracle labels (O(T), zero extra executions) replace MC rollouts (O(K×T²)) | RL post-training can't adapt to dynamic tool markets. MCTS too slow. Simple prompting fails 32.8% of time — agents exceed budgets via repetitive retries | Both identify failures of naive approaches. INTENT's 32.8% budget violation rate strengthens CASSI's case |
| **Approach** | `t* = argmax[quality − λ×cost]`. Train stopper on oracle labels → executor GRPO | Inference-time planning: (1) Language World Model (3B) predicts tool outcomes. (2) Monte Carlo Oracle — lookahead, reject costly actions. (3) Intention-Based Oracle: factorizes into Intention Predictor (binary: will tool succeed?) + Conditional Generator. Geometric cost calibration: expected_cost = nominal_cost / success_probability | CASSI: training-time with oracle labels. INTENT: inference-time with ensemble of small models. INTENT's intention decomposition and geometric calibration are clever techniques CASSI could adopt |
| **Implementation** | Two-model: 0.5B–3B stopper + 7B–72B executor | Qwen2.5-3B LWM on ~100K trajectories. Qwen3-0.6B Intention Predictor on 86K GPT-annotated triples. Caching: Rollout Cache, Last Call Cache, Blacklist. GPT-4.1-mini/nano agents | INTENT requires training ensemble of oracle models — significant upfront cost. CASSI's oracle labels are free |
| **Training** | 3-phase: collect, train stopper (SFT+GRPO), train executor (GRPO with stopper rewards) | **Agent is frozen** — no executor training. Only oracle models trained via SFT on annotated data | CASSI trains executor; INTENT rejects actions from frozen policy |
| **Experiments** | 6 benchmarks across 4 domains | StableToolBench (765 instances), synthetic tool prices U(5,50), B=50. Baselines: RAW, PROMPT, DFSDT, BTP, BATS | CASSI: diverse domains. INTENT: single domain, synthetic prices |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Requires training 3 oracle models (LWM+Generator+Predictor). Synthetic prices. World model inaccuracies. Lookahead overhead. Single domain. Frozen policy | CASSI: zero extra model training for labels, real benchmarks, trains executor, no world model needed |
| **Future (Theirs→CASSI)** | N/A | Extend beyond tool use; periodic oracle retraining for market dynamics → CASSI already covers diverse domains; iterative training handles model drift |
| **Relevance** | 3/5 — INTENT's finding that prompting fails 32.8% strengthens CASSI's case for training. Geometric cost calibration is a technique CASSI could adopt for richer oracle labels |

---

### Paper 3: CaRT — "CaRT: Teaching LLM Agents to Know When They Know Enough"
**Authors:** Liu et al. (2025) | **URL:** https://arxiv.org/abs/2510.08517

| Element | CASSI | CaRT | Contrast |
|---|---|---|---|
| **Background** | LLM agents overthink with tool use — no stopping training signal. Oracle-guided stopping labels (O(T)) solve scalability | LLM agents need to know when to stop acquiring information. Models struggle to predict success probability, lack principled exploration | **Most directly aligned paper** — both address core problem of knowing when to stop |
| **Motivation** | Mathematically optimal oracle labels (O(T), zero extra executions) vs CaRT's expensive counterfactual generation. Dynamic λ-weighted cost adaptation | Teach LLMs implicit "verbalized value function" for termination. Existing solutions: fixed budgets (inflexible), length penalties (poor generalization), confidence (not optimized for termination) | Both want learned termination. CASSI: mathematically optimal, cost-aware. CaRT: implicit, binary |
| **Approach** | `t* = argmax[quality − λ×cost]`. SFT+GRPO stopper. Executor GRPO with stopper Δ rewards | (1) Hard Negative Counterfactuals: modify trajectory so termination would be wrong (success rate drops <30%). (2) Verbal Reasoning: GPT-4o generates reasoning for termination/continuation. SFT on counterfactual pairs + reasoning traces. Optional GRPO | CASSI: continuous Δ ∈ [−1,1] from oracle. CaRT: binary stop/continue from counterfactuals. CaRT's counterfactual reasoning could augment CASSI's oracle labels |
| **Implementation** | Two-model: 0.5B–3B stopper + 7B–72B executor. 3-phase pipeline | Medical: Qwen2.5-3B, 1,233 problems (MedQA-USMLE+MedMCQA), GPT-4o conversations. Math: Qwen3-1.7B, 2,000 problems | Both use small models for termination. CASSI: diverse domains. CaRT: 2 domains, no tool use |
| **Training** | Phase 1: collect. Phase 2: SFT+GRPO stopper. Phase 3: executor GRPO | SFT on counterfactual pairs + reasoning traces. Optional GRPO. RL leads to longer traces (didn't help in math — cautionary) | CaRT trains only termination model. CASSI trains both. CaRT's RL→longer traces is a caution for CASSI's GRPO |
| **Experiments** | 6 benchmarks: GAIA, WebWalkerQA, HotpotQA, MuSiQue, SWE-bench, MATH-500 | Medical: 100 in-dist + 200 OOD dermatology. Math: AIME 2025. Small-scale (100-200 per domain) | CASSI: much broader evaluation. CaRT: limited but deeper probe analysis |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality; stopper transfer; domain-specificity | Requires labeled trajectories per step (expensive). Counterfactual generation labor-intensive. Fixed policy — termination+exploration not jointly optimized. Small-scale. No cost model. No tool-use | CASSI addresses: O(T) oracle labels (not expensive counterfactuals), explicit λ×cost, trains both models, multi-domain, tool-use agents |
| **Future (Theirs→CASSI)** | N/A | Jointly optimize exploration + termination. Explicit value estimation beyond implicit counterfactuals → CASSI already provides explicit Δ(s_t)∈[−1,1]; GRPO enables joint optimization through process rewards |
| **Relevance** | **5/5** — Most directly competitive. Validates "learn when to stop" paradigm. CASSI's advantages: mathematically optimal oracle labels, explicit λ cost, executor training, broader domains. CaRT's counterfactual approach could augment CASSI's oracle labels |

---

### Paper 4: s1 — "s1: Simple test-time scaling"
**Authors:** Muennighoff et al. (EMNLP/ICLR Workshop 2025) | **URL:** https://arxiv.org/abs/2501.19393

| Element | CASSI | s1 | Contrast |
|---|---|---|---|
| **Background** | LLM agents overthink with tool use — cost-aware PRM training via oracle labels | Seeks simplest approach to test-time scaling. o1 used massive RL; s1 asks: what's the minimal recipe? 1K examples + budget forcing | CASSI: training-based solution. s1: minimal training (26 min SFT) + external budget forcing |
| **Motivation** | Oracle labels (O(T)) make cost-aware PRM training tractable for long-horizon agent tasks | Minimal data + simple test-time intervention. Careful data curation (difficulty + diversity + quality) matters more than quantity | Both aim for efficiency. s1 achieves it through simplicity; CASSI through learned adaptation |
| **Approach** | `t* = argmax[quality − λ×cost]`. Train stopper → executor GRPO | (1) s1K: 1,000 curated questions from 59K pool, Gemini Thinking distillation. (2) Budget Forcing: enforce max tokens by appending end delimiter; enforce min by suppressing delimiter + "Wait" (triggers self-correction) | CASSI: internal learned stopping. s1: external delimiter manipulation. s1's forcing leads to self-correction — potential hybrid |
| **Implementation** | Two-model: 0.5B–3B stopper + 7B–72B executor | Qwen2.5-32B-Instruct → SFT → s1-32B. 26 min, 16 H100s. SFT only — no RL | CASSI: SFT+GRPO for both models. s1: SFT only, external control |
| **Training** | 3-phase with GRPO | SFT only — 26 minutes. Budget control is entirely external (decoding-time). Model cannot self-regulate | CASSI's executor learns to self-regulate through GRPO with stopper rewards |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Budget forcing is crude and uniform — same "Wait" for all problems. No per-instance adaptation. Cannot self-regulate. Only reasoning tasks. No cost model. Distillation from Gemini limits independence | CASSI: learned per-instance adaptation; self-regulating; agent domains; λ cost model; no distillation dependency |
| **Future (Theirs→CASSI)** | N/A | Extend to other model families and tasks. Understand why short traces tend correct (inverse scaling) → CASSI extends to agent tasks and explains inverse scaling: oracle t* stops when quality plateaus |
| **Relevance** | 4/5 — Strong baseline for cost-aware inference. s1's data curation methodology models how CASSI should curate trajectory datasets. Inverse scaling supports CASSI's premise |

---

### Paper 5: AgentPRM — "Process Reward Models for LLM Agents: Practical Framework and Directions"
**Authors:** Choudhury (2025) | **URL:** https://arxiv.org/abs/2502.10325

| Element | CASSI | AgentPRM | Contrast |
|---|---|---|---|
| **Background** | Overthinking — cost-aware PRM training via O(T) oracle labels | Training LLM agents via PRMs. PRMs used in math (deterministic transitions) but not in agentic settings (stochastic environments) | **CASSI's most important comparison.** Both use PRMs for agent training |
| **Motivation** | O(T) oracle labels (zero extra executions) vs AgentPRM's O(K×T²) MC rollouts — K×T fewer executions. Makes cost-aware PRM training tractable on long-horizon tasks (SWE-bench) | RL difficult for agents due to long horizons + sparse rewards. PRMs can provide step-level signals but haven't been applied to stochastic agent environments | CASSI directly addresses AgentPRM's training cost bottleneck |
| **Approach** | `t* = argmax[quality − λ×cost]`. Train cost-aware stopping model → executor GRPO | 3-stage: (1) Roll out, compute MC Q̂(s,a) targets. (2) Train Q-function (soft BCE). (3) RL update (Online DPO) maximizing PRM with KL. InversePRM: IRL from demos without outcome rewards. Best-of-N at inference | CASSI: O(T) oracle labels, cost-aware. AgentPRM: O(K×T²) MC rollouts, no cost model. CASSI's stopping is binary/ternary (simpler); AgentPRM's Q-values are continuous (harder) |
| **Implementation** | Two-model: 0.5B–3B stopper + 7B–72B executor | Llama3.2-3B for PRM + policy. 3 iterations, 10K rollouts/iteration. ALFWorld (text game) | Both two-model design. CASSI: smaller stopper, real benchmarks |
| **Training Cost** | **O(T) — zero extra executions** (post-hoc argmax over recorded values) | **O(K×T²) — K rollouts per state**, each T−t steps. T=20, K=8 → 1,520 additional step-executions (~160 full trajectories) per training iteration. Reward hacking detected (success peaks then degrades while PRM reward increases) | **This is CASSI's core advantage.** ~160× fewer additional executions. CASSI's simpler task (binary stop) may reduce reward hacking |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | **O(K×T²) execution cost.** Single benchmark (ALFWorld). PRM generalizes poorly (reward hacking). Small model (3B). No cost modeling. Best-of-N increases test-time cost | CASSI addresses all: O(T) cost, 6 benchmarks, cost-aware Δ, separate stopper→no reward hacking, no Best-of-N overhead |
| **Future (Theirs→CASSI)** | N/A | Exploration strategies, process reward shaping, model-predictive reasoning, scaling to complex environments → CASSI's stopper can serve as PRM for lookahead; already scales to complex envs; oracle labels complement exploration |
| **Relevance** | **5/5** — Single most important comparison. CASSI's primary innovation (O(T) oracle labels) directly addresses AgentPRM's main weakness (O(K×T²)). Cost-aware value is natural extension of Q-values. Must cite prominently |

---

## Category 2: Agent Stopping, Early Exit & Adaptive Termination

### Paper 6: BAVT — "Spend Less, Reason Better: Budget-Aware Value Tree Search for LLM Agents"
**Authors:** Li et al. (2026) | **URL:** https://arxiv.org/abs/2603.12634

| Element | CASSI | BAVT | Contrast |
|---|---|---|---|
| **Background** | Overthinking in agents — learned cost-aware PRM via O(T) oracle labels | Test-time scaling wastes tokens. Existing budget-aware methods need expensive fine-tuning or coarse heuristics. Need mid-execution intervention within a single trajectory | Both want mid-execution budget-aware control. Different means: learned vs. heuristic |
| **Approach** | `t* = argmax[quality − λ×cost]`. Train stopper → executor GRPO | Training-free: (1) Budget-Conditioned Node Selection: `p(select i) ∝ V_i^(γ_t)` where γ depends on remaining budget. (2) Residual Value Predictor: scores relative progress (not absolute quality). (3) Theoretical convergence guarantee (probability ≥ 1−ε) | CASSI: learned. BAVT: clever parameter-free heuristic. BAVT's γ_t dynamic adjustment mirrors CASSI's budget-tier system |
| **Training** | 3-phase: collect, train stopper (SFT+GRPO), train executor (GRPO) | **No training** — purely inference-time heuristics | CASSI learns; BAVT can't improve |
| **Experiments** | 6 benchmarks across 4 domains | 4 multi-hop QA, 2 model families | BAVT: QA only. CASSI: diverse domains |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Training-free may underperform learned methods on complex tasks. Single LLM for everything (search+value+execution). QA only. No learned adaptation | CASSI addresses: learned adaptation, separate specialized stopper, diverse domains, principled cost model |
| **Key Difference** | CASSI's learned stopping model adapts to task patterns. BAVT's heuristic can't learn from data. CASSI's λ-weighted cost more principled than generic "resource ratio" |
| **Relevance** | 3/5 — Shows budget-conditioned search works, supporting CASSI's premise. But purely heuristic — CASSI's learned approach generalizes better |

---

### Paper 7: DEER — "Dynamic Early Exit in Reasoning Models"
**Authors:** Yang et al. (2025) | **URL:** https://arxiv.org/abs/2504.15895

| Element | CASSI | DEER | Contrast |
|---|---|---|---|
| **Background** | Overthinking in tool-using agents — cost-aware PRM | ~75% of reasoning samples contain "pearl reasoning" — a point where reasoning is sufficient. 36.7% need less than half the original reasoning. Overthinking wastes compute AND degrades accuracy | Both identify a "sufficient point." CASSI formalizes it mathematically; DEER finds it heuristically |
| **Approach** | `t* = argmax[quality − λ×cost]`. SFT+GRPO stopper → executor GRPO | Training-free: (1) Detect transition tokens ("Wait," "Alternatively," "Hmm"). (2) Induce trial answer via "final answer" delimiter. (3) If confidence high → stop. Otherwise → continue | DEER: heuristic at transition points. CASSI: learned at every step. CASSI's stopper essentially "learns to detect pearl reasoning" through oracle labels |
| **Training** | 3-phase with GRPO | **No training** — inference-time only | CASSI: trains stopping into model. DEER: external detection |
| **Results** | Expected: 20–40% cost reduction | 31–43% CoT length reduction while improving accuracy 1.7–5.7%. Coding: 64.9% reduction, +2.1 pass@1 | DEER's results validate "stop early = better" — strong baseline for CASSI |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Requires transition tokens (not all models). Trial answer adds overhead. CoT only — not agent tasks with tools. Heuristic confidence | CASSI doesn't need transition tokens; works on agent tasks; learned stopping (not heuristic); multi-dimensional cost |
| **Relevance** | 3/5 — Effective for reasoning but doesn't generalize to agents. CASSI's stopper learns DEER's "pearl detection" with cost-awareness + agent applicability |

---

### Paper 8: TALE — "Token-Budget-Aware LLM Reasoning"
**Authors:** Han et al. (ACL 2025 Findings) | **URL:** https://arxiv.org/abs/2412.18547

| Element | CASSI | TALE | Contrast |
|---|---|---|---|
| **Background** | Overthinking in tool-using agents | CoT increases tokens dramatically (258 vs 15 for simple arithmetic). RLHF may have biased models toward verbosity | Both identify unnecessary token waste |
| **Approach** | `t* = argmax[quality − λ×cost]`. Per-step stopping via trained stopper → executor GRPO | TALE-EP: zero-shot budget estimation + prompt "use less than [budget]." TALE-PT: binary search for optimal budget → SFT/DPO to internalize. **Token Elasticity:** too-small budget paradoxically increases tokens | TALE: one budget per question (trajectory-level). CASSI: per-step stopping (finer granularity). CASSI's stopper avoids Token Elasticity |
| **Training** | 3-phase: collect, SFT+GRPO stopper, GRPO executor | TALE-EP: no training. TALE-PT: offline budget search (354 min on A100) + SFT or DPO | CASSI: O(T) oracle labels are cheaper than binary search. CASSI trains stopper, not just main model |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Prompt brittleness at small budgets (Token Elasticity). Math only. Static pre-execution budget. No tool-use. | CASSI: per-step (not pre-set), multi-dimensional cost, agent tasks, learned stopper avoids token elasticity |
| **Relevance** | 3/5 — Token Elasticity is an important phenomenon CASSI should be aware of. Validates LLMs CAN be trained for token efficiency |

---
*Continue to Part 2...*
