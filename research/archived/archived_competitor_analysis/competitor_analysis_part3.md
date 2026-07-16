# Competitor Analysis Part 3: Categories 5-7 (Token-Efficient Reasoning, Efficient Tool Use & Deep Research, Foundational Insights)

> Continues from Part 1 (`competitor_analysis.md`) and Part 2 (`competitor_analysis_part2.md`).
> Covers: Token-Efficient Reasoning (3 papers), Efficient Tool Use & Deep Research (4 papers), Foundational Insights (3 papers).

---

## Category 5: Token-Efficient Reasoning

### Paper 19: SelfBudgeter — "SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning"
**Authors:** Li et al. (2025) | **URL:** https://arxiv.org/abs/2505.11274

| Element | CASSI | SelfBudgeter | Contrast |
|---|---|---|---|
| **Background** | Overthinking in agents — cost-aware PRM with separate stopper | QwQ-32B provides 13 solutions for "2+3=?" consuming 100× more tokens. Existing methods fail to self-estimate AND adhere to budgets simultaneously | Both want models to auto-estimate required compute. Different granularity: per-question vs per-step |
| **Motivation** | Oracle labels (O(T)) for cost-aware PRM with per-step dynamic adaptation | Let model autonomously estimate required budget then strictly follow it. No existing method does both self-estimation and adherence | CASSI: stopper estimates per-step continuation value. SelfBudgeter: model estimates total budget |
| **Approach** | `t* = argmax[quality − λ×cost]`. Per-step stopper → executor GRPO | (1) Cold-Start SFT: output `<budget>N</budget><solution>...</solution>`. (2) Budget-Guided GRPO: three rewards — Budget Penalty (penalize exceeding b_max), Precision Budget Control Reward (cosine-based, peaks when actual ≈ (1−α)×budget), Accuracy Reward. Dynamic α: 6.0→0.1 to prevent reward hacking | CASSI: step-level stopping via separate model. SelfBudgeter: question-level budget in one model. CASSI's λ is simpler, more intuitive than α scheduling |
| **Training** | 3-phase: collect, SFT+GRPO stopper, GRPO executor | Cold-start SFT (3,630 samples) → Budget-guided GRPO (30K, 3 epochs). 5 responses/round, T=0.6. Single model must learn budgeting AND solving | CASSI separates budgeting (stopper) from solving (executor). SelfBudgeter bundles both in one model |
| **Results** | Expected: 20–40% cost reduction | **61% length compression** maintaining accuracy. GSM8K: 84.10% at 1231 tokens (vs R1: 73.09% at 2865) — 11-point improvement while compressing to 43%. MATH500: 78.47% at 2327 (vs 74.93% at 5327). Generalizes to GPQA/SCoRE | CASSI: agent tasks (not just math). Both show dramatic compression |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality; stopper transfer | Math only. Cold-start needed. Single scalar budget (no per-step). α scheduling fragile (fixed α → reward hacking or collapse). No heterogeneous costs. `<budget>` tag not universal | CASSI: per-step (finer), multi-dimensional cost, separate stopper (no capacity conflict), agent domains |
| **Future (Theirs→CASSI)** | N/A | Validation on complex real-world scenarios → CASSI directly validates budget-aware training on real agent tasks (GAIA, SWE-bench) |
| **Relevance** | 4/5 — Most architecturally similar in token efficiency category. CASSI extends to: multi-step agents, heterogeneous costs, separate stopper, per-step granularity, oracle-guided labels. Strong reinforcement |

---

### Paper 20: BudgetThinker — "BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens"
**Authors:** Wen et al. (2025) | **URL:** https://arxiv.org/abs/2508.17196

| Element | CASSI | BudgetThinker | Contrast |
|---|---|---|---|
| **Background** | Overthinking in agents — cost-aware PRM | Long CoT achieves SOTA but prohibitive cost. Models need continuous budget awareness, not one-time instruction | Both want continuous cost awareness. Different mechanism: separate stopper vs control tokens |
| **Approach** | `t* = argmax[quality − λ×cost]`. Stopper evaluates per-step → executor GRPO | Control tokens periodically inserted (10%–100% of budget). SFT on 41K budget-annotated pairs. Curriculum GRPO: `R = k1·correctness + k2·format + k3·max(1−γ·(B−|y|)²/B², 0)`. Progressively tighter budgets, mixed final phase | CASSI: modular (stopper evaluates state). BudgetThinker: intrusive (control tokens in generation). CASSI's λ is simpler, more flexible |
| **Training** | 3-phase: collect, SFT+GRPO stopper, GRPO executor | SFT on 41K pairs → Curriculum GRPO → mixed-budget. Single model with modified generation pipeline | CASSI: stopper is separate, trained independently. BudgetThinker: model must be retrained with control token pipeline |
| **Results** | Expected: 20–40% cost reduction | +4.9% avg accuracy across budgets with precise adherence. MATH-500, AMC 2023, AIME 2024. Outperforms ThinkPrune | Both achieve budget-aware behavior. CASSI: through separate model; BudgetThinker: through embedded controls |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Math only. Generation pipeline modification needed. Model-intrusive (retraining per model). No tool use. Single model carries reasoning + budget tracking | CASSI: modular (plug into any executor), agent tasks, no pipeline modification, multi-dimensional cost |
| **Relevance** | 2/5 — Control tokens work but are model-intrusive. CASSI's separate monitor offers more modularity and applicability to agents |

---

### Paper 21: Budget Guidance — "Steering LLM Thinking with Budget Guidance"
**Authors:** Li et al. (2025) | **URL:** https://arxiv.org/abs/2506.13752

| Element | CASSI | Budget Guidance | Contrast |
|---|---|---|---|
| **Background** | Overthinking in agents — cost-aware PRM training | Deep-thinking LLMs produce excessively long reasoning. Fine-tuning costly; inference-time methods heuristic and degrading | Both want cost control. CASSI: training-based. Budget Guidance: inference-time Bayesian modulation |
| **Approach** | `t* = argmax[quality − λ×cost]`. Train stopper via SFT+GRPO → executor GRPO | Classifier-guidance-inspired: `p(Y_t | ...) ∝ p(Y_t) · Pr(L_t ≤ l_bar − t | ...)`. Lightweight Gamma predictor: auxiliary model predicts Gamma distribution over remaining length from LLM hidden states. Soft token-level steering (multiply logits by CDF scores, renormalize). No LLM fine-tuning | CASSI: trained stopping decisions. Budget Guidance: inference-time probability modulation. Both use lightweight auxiliary model. Budget Guidance's Gamma predictor is a novel technical contribution |
| **Training** | 3-phase: collect, SFT+GRPO stopper, GRPO executor | Only predictor trained (SFT on traces). Main LLM untouched | CASSI trains both models; Budget Guidance only trains predictor |
| **Results** | Expected: 20–40% cost reduction | Up to 26% accuracy gain under tight budgets. 63% of thinking tokens with competitive accuracy. Cross-domain generalization. Emerges difficulty estimation | Both achieve significant token reduction |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Math/coding only. Predictor needs target LLM data. Per-token overhead. Budget: reasoning tokens only. Inference-only — model doesn't learn | CASSI: agent domains, model learns better behavior through RL, multi-dimensional cost, no per-token overhead |
| **Relevance** | 2/5 — Clever no-fine-tuning approach but single-turn only. Gamma predictor is novel. CASSI's RL approach generalizes better to multi-turn agents |

---

## Category 6: Efficient Tool Use & Deep Research

### Paper 22: DeepResearcher — "Scaling Deep Research via RL in Real-world Environments"
**URL:** https://arxiv.org/abs/2504.03160

| Element | CASSI | DeepResearcher | Contrast |
|---|---|---|---|
| **Background** | Overthinking in agents — cost-aware PRM | Deep research agents: prompt-engineered (brittle) or RAG-based RL (unrealistic). No end-to-end real web training | Orthogonal research directions. CASSI could be applied to DeepResearcher agents |
| **Approach** | `t* = argmax[quality − λ×cost]`. Stopper → executor GRPO | End-to-end RL in real web environments. Multi-agent: browsing agents, specialized content agents. Real search/browse interactions | CASSI: cost-aware stopping. DeepResearcher: capability improvement via real-world RL |
| **Results** | Expected: 20–40% cost reduction | +28.9 points over prompt engineering. +7.2 over RAG-based RL. Emergent: plan formulation, cross-validation, self-reflection, honesty | DeepResearcher improves quality; CASSI reduces cost |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Multi-agent complexity. Real web non-determinism. High training cost. No cost-awareness or budget. No stopping policy — agents may over-search | CASSI's stopper could prevent DeepResearcher from over-searching. Clear integration path |
| **Future (Theirs→CASSI)** | N/A | More domains, complex coordination → CASSI could serve as cost-management layer for any multi-agent research system |
| **Relevance** | 2/5 — Orthogonal contribution. CASSI could be deployed on DeepResearcher as cost-saving layer. Not a competitor |

---

### Paper 23: ReTool — "ReTool: Reinforcement Learning for Strategic Tool Use in LLMs"
**Authors:** Feng et al. (2025) | **URL:** https://arxiv.org/abs/2504.11536

| Element | CASSI | ReTool | Contrast |
|---|---|---|---|
| **Background** | Overthinking — cost-aware PRM | Reasoning models struggle with structured problems (geometry, computation). Code interpreters help but rely on rigid hand-crafted schemas | Both use RL to induce strategic behaviors. CASSI: stopping. ReTool: tool use |
| **Approach** | `t* = argmax[quality − λ×cost]`. Stopper → executor GRPO | Cold-start SFT (synthetic code-augmented traces) → GRPO with multi-turn real-time code execution. Binary correctness reward. Model autonomously discovers tool invocation patterns | Both GRPO-based. Both multi-phase (SFT→RL). CASSI: stopping decisions. ReTool: tool invocation decisions |
| **Results** | Expected: 20–40% cost reduction | 67% at 400 steps vs 40% at 1080 (2.7× faster). 72.5% surpassing o1-preview by 27.9%. Emergent code self-correction ("aha moment") | Both show GRPO induces non-trivial strategic behaviors autonomously |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Math only. Code interpreter only. Sandbox required. No cost mechanism. Large model (32B). No stopping decisions | CASSI: multi-tool agent scenarios, cost awareness, explicit stopping, smaller models |
| **Future (Theirs→CASSI)** | N/A | Multi-tool scenarios, sophisticated rewards, broader evaluation → CASSI already evaluates multi-tool; CASSI's cost-aware process rewards are the "sophisticated rewards" they call for |
| **Relevance** | 3/5 — Proves GRPO induces strategic behaviors without human priors — supports CASSI's training claims. Complementary: ReTool for tool strategy + CASSI for cost-aware stopping |

---

### Paper 24: Ares — "Ares: Adaptive Reasoning Effort Scaling for LLM Agents"
**URL:** https://arxiv.org/abs/2603.07915

| Element | CASSI | Ares | Contrast |
|---|---|---|---|
| **Background** | Overthinking — cost-aware PRM with separate stopper | LLMs support configurable reasoning levels (high/medium/low). Uniform use degrades performance. Different steps need different reasoning depths | **Most architecturally similar paper.** Both use small control model for per-step decisions |
| **Motivation** | Oracle labels (O(T)) for cost-aware PRM | Lightweight router predicting per-step effort achieves near-optimal cost-performance. Intra-model switching preserves KV cache | Both: small model controls larger agent. CASSI: stop/continue. Ares: effort level selection |
| **Approach** | `t* = argmax[quality − λ×cost]`. Stopper → executor GRPO | Router (Qwen3-1.7B) predicts e_t ∈ {low, mid, high}. Data: (1) Collect successful trajectories. (2) Test each effort level per step → find minimum sufficient. (3) Teacher LLM generates rationales. SFT + GRPO with outcome + cost + format rewards. Plug-and-play | Ares: discrete levels via trial-and-error labeling. CASSI: binary stop via oracle labels. Ares's min-effort labeling requires testing all levels per step; CASSI's oracle labels are O(T) post-hoc |
| **Results** | Expected: 20–40% cost reduction | **52.7% token reduction** vs always-high on TAU-Bench. RL improves: Retail 54.8%→58.5% with 176K tokens saved. Airline 36%→42% with 80% tokens saved | Both significant savings. Ares on reasoning effort; CASSI on stopping decisions |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Requires configurable reasoning levels (specific models). Router needs successful trajectories. Single router per domain. No λ tradeoff. Reasoning effort only | CASSI: binary decisions (simpler, more general), λ parameter, trains executor too, no model-specific features needed |
| **Future (Theirs→CASSI)** | N/A | Not discussed → CASSI + Ares could combine: Ares adjusts effort per step, CASSI decides when to stop |
| **Relevance** | **5/5** — Strongest architectural parallel. Router-executor validates CASSI's approach. Both per-step, small control model, GRPO, plug-and-play. Different objectives and labeling. Must cite as contemporaneous |

---

### Paper 25: SeqRoute — "SeqRoute: Budget-Robust Query Routing for Multi-Turn LLM Conversations"
**URL:** https://arxiv.org/abs/2605.25424

| Element | CASSI | SeqRoute | Contrast |
|---|---|---|---|
| **Background** | Overthinking in agent trajectories — cost-aware PRM | Myopic routers cause "budget bankruptcy" — exhausting resources on early simple queries, leaving later complex ones with inadequate models | Both address cost-aware sequential decisions under budget. Different context: model routing vs step stopping |
| **Approach** | `t* = argmax[quality − λ×cost]`. Stopper → executor GRPO | Session-Budget MDP: State = [embedding || remaining budget]. Actions: {weak(cost=1), strong(cost=10)}. Offline CQL. **Hindsight Budget Relabeling (HBR):** simulate history under diverse hypothetical budgets → 2.38M transitions at zero cost. **λ-sweep:** `a* = argmax[Q(s,a) − λ·c(a)]` for zero-shot Pareto | HBR is powerful — could generate budget-annotated data for CASSI's stopper from unconstrained logs. λ-sweep is novel mechanism CASSI could adopt |
| **Results** | Expected: 20–40% cost reduction | Up to 73.5% cost reduction with near-zero bankruptcy. Learned delayed gratification: suppresses strong model early, deploys on final decisive queries | Both: cost-quality Pareto. SeqRoute on model selection; CASSI on trajectory stopping |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Binary action space only. Simple embedding state. Requires offline logs. Routes between models, not step-level | CASSI: continuous state (trajectory), per-step decisions, interactive RL |
| **Relevance** | 3/5 — HBR technique is powerful and complementary to CASSI's oracle labeling. λ-sweep for zero-shot budget adaptation could enhance CASSI. Different core problem |

---

## Category 7: Foundational Insights (Empirical & Theoretical)

### Paper 26: Don't Overthink It — "Preferring Shorter Thinking Chains for Improved LLM Reasoning"
**Authors:** Hassid et al. (2025) | **URL:** https://arxiv.org/abs/2505.17813

| Element | CASSI | Don't Overthink It | Contrast |
|---|---|---|---|
| **Background** | Overthinking in agents — cost-aware PRM via oracle labels | Within individual questions, shorter reasoning chains are significantly more likely correct — up to 34.5% more accurate than longest. Counters conventional "longer=better" wisdom | Both challenge "longer=better." CASSI provides training solution; this paper provides empirical proof |
| **Approach** | `t* = argmax[quality − λ×cost]`. Train stopper → executor GRPO | short-m@k: run k parallel, halt when m chains complete, majority-vote among m shortest. Short-1@k (most efficient), short-3@k (best balance). Training: SFT on short chains > SFT on long chains | CASSI: training-time stopping decisions. Paper: inference-time selection among completed chains. Both show shorter/earlier-stopped = better |
| **Results** | Expected: 20–40% cost reduction | Shorter chains 34.5% more accurate. 40% fewer tokens with same performance. 33% less wall time. Training on short chains: both shorter AND better | Key empirical validation for CASSI's thesis that `t*` exists and is earlier than full trajectory end |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Inference-only for main contribution. Math only. Requires `<think>` tags. k× parallel compute overhead. No cost/quality tradeoff. Cannot handle when longest is correct | CASSI: training-based, agent domains, no parallel overhead, λ tradeoff, handles all scenarios |
| **Future (Theirs→CASSI)** | N/A | Integrate with RL training, adaptive m selection, broader benchmarks → CASSI directly integrates "short is better" with RL via oracle labels; adaptive stopping is learned version of adaptive m |
| **Relevance** | 4/5 — Strong empirical evidence for CASSI's premise. "Shorter chain most likely correct" validates oracle stopping labels. CASSI extends to training-based and agent domains |

---

### Paper 27: When More is Less — "Understanding Chain-of-Thought Length in LLMs"
**Authors:** Wu et al. (2025) | **URL:** https://arxiv.org/abs/2502.07266

| Element | CASSI | When More is Less | Contrast |
|---|---|---|---|
| **Background** | Overthinking in agents — cost-aware PRM | Prevailing assumption: longer CoT = better. Most rigorous empirical + theoretical analysis of this assumption | **Strongest theoretical foundation for CASSI.** CASSI operationalizes their insights |
| **Motivation** | Oracle labels (O(T)) for cost-aware PRM. Dynamic per-instance adaptation | Move beyond "longer is better" to principled understanding. Controlled synthetic experiments, real LLMs, formal theory | CASSI: solving the problem. Paper: understanding/characterizing the problem. CASSI fills their gap |
| **Approach** | `t* = argmax[quality − λ×cost]`. Train stopper → executor GRPO | Three-pronged: (1) Real-world — Qwen2.5 1.5B–72B on MATH-L5 → inverted-U curves. (2) Synthetic — GPT-2 with controlled difficulty/step-granularity. (3) Theory — formal CoT model, error accumulates exponentially → proves existence of optimal CoT length N*, scaling laws: N*↑ with difficulty, N*↓ with capability | CASSI's `t*` formula directly aligns with their optimal CoT N*. Their inverted-U curve proves t* exists |
| **Key Results** | Expected: 20–40% cost reduction | **Inverted-U curve** (accuracy peaks then drops with CoT length). Optimal drops from 14→4 steps (1.5B→72B). Optimal-length CoTs beat longest by **40%** for 72B. GRPO training: CoT length decreases as accuracy improves ("simplicity bias"). p≈1e-8 correlation between difficulty and optimal length | CASSI's t* from oracle labels computes exactly what their theory predicts exists. Simplicity bias aligns with CASSI's cost-aware training |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Synthetic experiments simple (addition). "Step count" not tokens. Simple theory. Needs pre-computed optimal length. Heuristic inference. No cost model | CASSI addresses all: handles complex agent trajectories, actual token cost, O(T) computation from any trajectory, learned stopping model (not heuristic), explicit λ cost |
| **Future (Theirs→CASSI)** | N/A | Extend theory to complex reasoning; auto-determine optimal CoT; integrate with RL → CASSI directly addresses all three. CASSI IS the training-time operationalization of their insights |
| **Relevance** | **5/5** — Strongest theoretical foundation. Inverted-U directly motivates optimal stopping point. Simplicity bias aligns with CASSI. MUST cite prominently. CASSI fills exact gap they identify: trainable per-instance stopping |

---

### Paper 28: CTA — "Calibrate-Then-Act: Cost-Aware Exploration for LLM Agents"
**URL:** https://arxiv.org/abs/2602.16699

| Element | CASSI | CTA | Contrast |
|---|---|---|---|
| **Background** | Overthinking in agents — cost-aware PRM | LLM agents must balance exploration cost against information gain. Existing agents use static policies | Both want cost-aware decisions. CASSI: stopping. CTA: exploration actions |
| **Approach** | `t* = argmax[quality − λ×cost]`. Trained stopping model → executor GRPO | Sequential decision with latent variables. Explore (cost θ) or commit. Reward = I[success]·D_θ(actions). CTA-Prompted: zero-shot, prior injected in prompt. CTA-RL: trains agent with cost-incorporating rewards. Standard RL without priors fails | CASSI: trained stopper provides cost signals. CTA: explicit priors about environment state. CTA shows explicit cost signals change behavior even without training |
| **Training** | 3-phase: collect, SFT+GRPO stopper, GRPO executor | CTA-Prompted: no training. CTA-RL: standard RL with cost rewards. Small-scale tasks (Pandora's Box, RAG QA, file reading) | CASSI: large-scale agent benchmarks. CTA: proof-of-concept tasks |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Small-scale tasks. Prior predictor needed. Simple action space (explore/commit). Scaling unclear | CASSI: full-scale agent benchmarks, arbitrary action space, learned stopping model replaces prior predictor |
| **Relevance** | 2/5 — Validates core intuition: explicit cost signals change behavior even without retraining. CASSI takes this further with learned, dynamic stopping. CTA's task scale too limited to be baseline |

---

*Continue to Part 4 (Synthesis)...*
