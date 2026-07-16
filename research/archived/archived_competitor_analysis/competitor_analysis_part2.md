# Competitor Analysis Part 2: Categories 3-4 (ReMA, MGV, CORL, L1, Reason Efficiently, Agent-RRM, CSO, CARL, GiGPO, GRPO)

> Continues from `competitor_analysis.md`. Covers: Meta-Reasoning & Monitor-Executor Architectures (3 papers) + RLHF & Preference Optimization (7 papers).

---

## Category 3: Meta-Reasoning & Monitor-Executor Architectures

### Paper 9: ReMA — "Learning to meta-think for LLMs with multi-agent reinforcement learning"
**Authors:** Wan et al. (NeurIPS 2025) | **URL:** https://arxiv.org/abs/2503.09501

| Element | CASSI | ReMA | Contrast |
|---|---|---|---|
| **Background** | Overthinking in agents — cost-aware PRM via oracle labels | Reasoning models show meta-thinking as byproduct of RL, not explicit engineering. Single-agent RL forces meta-thinking + reasoning into one pass → inefficient exploration, early local optima | Both decompose reasoning into separate components. CASSI: for cost efficiency. ReMA: for reasoning quality |
| **Motivation** | Oracle labels (O(T)) for cost-aware PRM — dynamic per-instance adaptation | Decoupling meta-thinking from execution enables structured exploration and better generalization. About improving reasoning QUALITY, not reducing cost | CASSI improves efficiency; ReMA improves capability. Opposite goals with similar architecture |
| **Approach** | `t* = argmax[quality − λ×cost]`. SFT+GRPO stopper → executor GRPO | High-level agent πₕ generates meta-thoughts; low-level πₗ generates reasoning conditioned on meta-thoughts. Multi-turn: iterative interleaving via Markov Game, alternating RL (REINFORCE++). Parameter sharing (same weights, distinct prompts). Turn-level ratio clipping | Both two-agent RL. CASSI: sequential training (stopper first, then executor). ReMA: alternating training. CASSI's stopper is deliberately smaller |
| **Implementation** | Stopper: 0.5B–3B. Executor: 7B–72B. Stopper is deliberately smaller | Llama-3-8B, Qwen2.5-7B — same-size copies for both agents. Training: 7.5K MATH, 5K RewardBench | CASSI's stopper is much smaller → lower overhead. ReMA uses same-size models → 2× inference cost |
| **Training** | 3-phase: collect, SFT+GRPO stopper, GRPO executor | Alternating MARL: freeze one agent, train other via REINFORCE++. Multi-turn: GRPO with turn-level ratio clipping. 0.8K SFT from LIMO via GPT-4o for bootstrapping | Both use GRPO. CASSI: sequential (simpler). ReMA: alternating (more complex, unstable) |
| **Results** | Expected: 20–40% cost reduction | Single-turn: +6.68% avg math, +8.49% LLM-as-Judge. OOD: +20% AMC23, +13.33% AIME24. Multi-turn: ~5% gain but brittle ("Echo Trap" — agents echo each other). Small models (1B) collapse; 8B adapts | ReMA improves accuracy at cost of complexity. CASSI reduces cost with simpler binary task |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Multi-turn highly unstable (Echo Trap). Small gains on instruction-tuned models. Only ≤8B. Two-agent **doubles inference cost**. Not about cost reduction | CASSI's stopper is deliberately small (<3% overhead vs 2×). CASSI's binary task likely more stable than full meta-reasoning |
| **Future (Theirs→CASSI)** | N/A | Extend to complex reasoning scenarios. Multi-turn instability unresolved → CASSI suggests simpler architecture (binary stop/continue) avoids Echo Trap |
| **Relevance** | 3/5 — Philosophically similar two-agent design but opposite goals. Multi-turn brittleness informs CASSI's stability design. CASSI's simpler task + smaller stopper likely more stable |

---

### Paper 10: MGV — "Monitor-Generate-Verify: Formalising Metacognitive Theory for Language Model Reasoning"
**Authors:** Oh & Gobet (FoRLM 2025) | **URL:** https://arxiv.org/abs/2511.04341

| Element | CASSI | MGV | Contrast |
|---|---|---|---|
| **Background** | Overthinking in agents — cost-aware PRM with monitor-executor architecture | Current reasoning (Self-Verification, SELF-REFINE) follows Generate-Verify but omits pre-generation MONITORING. Human metacognition monitors BEFORE action | MGV provides theoretical justification for CASSI's two-model architecture from cognitive science |
| **Motivation** | Oracle labels for cost-aware PRM | "Prefix dominance trap" — once LLMs commit to strategy, verification rarely recovers (~20% accuracy loss). Fill theoretical gap — no prior work translated Flavell's metacognitive frameworks computationally | CASSI is an empirical instantiation; MGV is a theoretical foundation |
| **Approach** | `t* = argmax[quality − λ×cost]`. Stopper is the Monitor; executor is Generate; outcome is Verify | 3-phase: Monitor (metacognitive knowledge → difficulty assessment) → Generate (strategy selection → execution) → Verify (plausibility/coherence; feedback updates Monitor). Formal pseudocode only | CASSI's stopper IS MGV's Monitor (assessing whether to continue). CASSI's executor IS Generate. CASSI's outcome reward IS Verify feedback. CASSI provides the training methodology MGV only speculates about |
| **Implementation** | Full implementation: 0.5B–3B stopper, 7B–72B executor, 3-phase training, 6 benchmarks | **None.** Purely theoretical. "No empirical validation" | CASSI is a fully implemented version of MGV's Monitor-Generate architecture |
| **Training** | SFT+GRPO for stopper, GRPO for executor | **None.** No training methodology described | CASSI provides concrete training: oracle labels + GRPO for the Monitor component |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | **No experiments.** No implementation. No training recipe. No connection to modern RL (PPO, GRPO). No cost model | CASSI is the empirical answer MGV calls for |
| **Relevance** | 2/5 — Theoretical grounding for CASSI's architecture. Cite as motivation for monitor-executor split. Not a competitor — no working system |

---

### Paper 11: CORL — "Controlling Performance and Budget of a Centralized Multi-agent LLM System with RL"
**Authors:** Jin et al. (2025) | **URL:** https://arxiv.org/abs/2511.02755

| Element | CASSI | CORL | Contrast |
|---|---|---|---|
| **Background** | Overthinking in single-agent trajectories | Multi-LLM systems use decentralized debate — all models process every query, incurring massive costs. Existing multi-agent RL ignores cost | Both want cost control. Different context: CASSI = per-step stopping; CORL = per-query model routing |
| **Approach** | Stopping decisions within trajectory via stopper + executor | Controller LLM decides: answer directly or decompose into sub-queries dispatched to experts. RL with dual rewards: task + cost (which models called, tokens). Budget-conditioned via system prompts | Both use controller/monitor with cost-aware decisions. CORL routes between models; CASSI stops within trajectory |
| **Implementation** | Stopper 0.5B–3B, executor 7B–72B | Qwen2.5-7B controller + GPT experts. Controller is full 7B LLM | CASSI's monitor is deliberately smaller (0.5B–3B vs 7B). Different granularity: step vs query |
| **Training** | 3-phase: collect, SFT+GRPO stopper, GRPO executor | RL (PPO) on controller only — dual reward. Experts frozen | Both train only the control model. CASSI also trains executor |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Full 7B controller (not lightweight). Fixed experts. Routes between models, not per-step control. API call cost only | CASSI: smaller controller, per-step granularity, multi-dimensional cost, trains executor too |
| **Relevance** | 2/5 — Validates controller-executor architecture for cost management. Different domain and granularity. CORL's approach orthogonal to CASSI |

---

## Category 4: RLHF & Preference Optimization for Agent Behaviors

### Paper 12: L1 — "L1: Controlling How Long a Reasoning Model Thinks With RL"
**Authors:** Aggarwal & Welleck (2025) | **URL:** https://arxiv.org/abs/2503.04697

| Element | CASSI | L1 | Contrast |
|---|---|---|---|
| **Background** | Overthinking in tool-using agents — cost-aware PRM | Reasoning models' length is uncontrolled — cannot allocate specific compute budget. s1's budget forcing severely degrades performance | Both want controlled compute. CASSI: automatic per-instance. L1: user-specified budget |
| **Motivation** | Oracle labels (O(T)) for cost-aware PRM. Adaptive stopping per instance | Design method for user-specified length targets with minimal accuracy loss. RL trains model to follow length instructions (not s1's heuristic forcing) | CASSI: no user budget needed. L1: requires user to specify target length |
| **Approach** | `t* = argmax[quality − λ×cost]`. Stopper + executor GRPO | Length Controlled Policy Optimization (LCPO): append "Think for N tokens" to prompt. GRPO: `r = I(y=y_gold) − α·|n_gold − n_y|`. L1-Exact: exact targets. L1-Max: soft constraint preferring shorter up to budget. **SRM discovery:** long-CoT RL creates strong short-CoT models | CASSI: learned per-step stopping from oracle labels. L1: prompt-based length target with RL penalty. L1's SRM discovery is novel and relevant |
| **Implementation** | Stopper 0.5B–3B + executor 7B–72B | DeepScaleR-1.5B (RL from R1-Distill-Qwen-1.5B). 40K math QA. 700 GRPO steps, 4K context | Single model vs two-model. CASSI: agent tasks. L1: math only |
| **Results** | Expected: 20–40% cost reduction | 100–150% better than s1 at 512/1024 tokens. 2× fewer tokens than DeepScaleR-4K. 1.5B L1 surpasses GPT-4o at equal lengths. ~3% length error, <2.5% budget violation | Both ~50% token reduction. CASSI: per-instance automatic. L1: user-specified per problem |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Cannot exceed training budget (4K limit). Uniform budget for all problems (L1-Exact). User must specify budget. Math only | CASSI: automatic per-instance, no budget ceiling, agent domains, no user specification needed |
| **Future (Theirs→CASSI)** | N/A | Train models to use MORE inference compute than training. Separate rewards for reasoning vs answer tokens → CASSI's multi-dimensional budget already separates token types |
| **Relevance** | 4/5 — Closest in cost-aware RL space. CASSI's advantage: automatic optimization without user budgets, applicability to agents. Should cite as showing RL enforces length constraints |

---

### Paper 13: Reason Efficiently — "Training Language Models to Reason Efficiently"
**Authors:** Arora & Zanette (NeurIPS 2025) | **URL:** https://arxiv.org/abs/2502.04463

| Element | CASSI | Reason Efficiently | Contrast |
|---|---|---|---|
| **Background** | Overthinking in tool-using agents — cost-aware PRM | Reasoning models waste compute especially on easy problems. DeepSeek-R1 takes "more than a page" for "How much is 1+1?" Deployment efficiency critical | **Closest conceptual predecessor.** Both optimize cost-quality tradeoff via RL |
| **Motivation** | Oracle labels (O(T)) make cost-aware PRM training tractable. Dynamic per-instance adaptation beats static penalties | Train models to use fewer tokens for correct answers — navigate token-accuracy Pareto. Core: reasoning models waste compute, especially on easy problems | CASSI extends their approach: explicit stopping (not just length penalty), separate model, agent domains |
| **Approach** | `t* = argmax[quality − λ×cost]`. Stopper as cost-aware PRM → executor GRPO | RL reward: `R = I(y=y*) × (1 − α·σ((LEN − MEAN)/STD))`. Per-prompt normalization prevents over-penalizing hard problems. Sigmoid soft-clipping ensures correct (even long) preferred over incorrect. PPO+RLOO. **Theory:** maximizer maintains perfect accuracy, generates shortest correct solution | CASSI: explicit stopping decision. Reason Efficiently: implicit brevity incentive. Both have tunable parameter (α vs λ). CASSI: per-step. RE: total length |
| **Training** | 3-phase: collect, SFT+GRPO stopper, GRPO executor | ~100 RL steps (~200 gradient updates). PPO+RLOO. Single model. Length penalty embedded in reward | Both computationally efficient. CASSI: trains two models. RE: trains one model faster |
| **Results** | Expected: 20–40% cost reduction | 7B: **50% token reduction with <5% accuracy loss.** Difficulty-adaptive: 16% on AIME2024 (hard), 65% on GSM8K (easy). Outperforms SFT and DPO. Only 100 RL steps | Both ~50% savings. Both difficulty-adaptive. RE: math only. CASSI: agents too |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality; stopper transfer | No per-instance targeting (α uniform). Penalty on total output. Small accuracy loss unavoidable. Math only. RLOO bias partially confounds | CASSI: explicit per-instance stopping, separate model avoids capacity conflict, agent domains, cost-aware process rewards |
| **Future (Theirs→CASSI)** | N/A | Improve performance WHILE reducing compute (not trading). Precise length targeting → CASSI's per-step stopping provides precise control; stopper eliminates only unnecessary steps (preserves necessary ones) |
| **Relevance** | **5/5** — Closest conceptual predecessor. CASSI extends by: (1) explicit stopping via oracle labels, (2) separate stopper for principled per-instance decisions, (3) agent domains. MUST cite. Theoretical proof that maximizer generates shortest correct is directly relevant |

---

### Paper 14: Agent-RRM — "Agent-RRM: Exploring Reasoning Reward Model for Agents"
**Authors:** Fan et al. (2026) | **URL:** https://arxiv.org/abs/2601.22154

| Element | CASSI | Agent-RRM | Contrast |
|---|---|---|---|
| **Background** | Overthinking — cost-aware PRM via oracle labels | Agentic RL uses sparse outcome rewards. For long-horizon tasks, this fails to differentiate intermediate quality | Both address reward sparsity. CASSI provides cost-aware process rewards; Agent-RRM provides quality-focused process rewards |
| **Motivation** | Oracle labels (O(T)) for cost-aware PRM | Step-level rewards: prohibitive annotation cost + reward hacking risk. Pairwise RMs introduce biases. Textual critique signals unexplored | CASSI: automatic oracle labels (free). Agent-RRM: expensive GPT annotation |
| **Approach** | `t* = argmax[quality − λ×cost]`. Stopper produces Δ + STOP/CONTINUE | Multi-faceted RM: `<think>`, `<critique>`, `<score>`∈[0,1]. Reagent-C (text refinement, training-free), Reagent-R (scalar added to rule rewards for GRPO), Reagent-U (both, two-stage sampling) | CASSI: cost-aware stopping decisions. Agent-RRM: quality-focused critique+score. CASSI's RM is simpler (binary + value); Agent-RRM's is richer (text critique) |
| **Implementation** | Stopper 0.5B–3B + executor 7B–72B | Qwen3-8B for agent and RM. RM trained on 28K SFT + 90K RL (GPT-OSS-120B annotated). Tools: search, browse, code, file, image. 12 benchmarks | Shared benchmarks (GAIA, WebWalkerQA). CASSI's RM is cheaper to train (oracle labels vs GPT annotation) |
| **Results** | Expected: 20–40% cost reduction | 43.7% GAIA (text), 46.2% WebWalkerQA — best ≤8B open-source. +11.2% Bamboogle, +9.0% xbench over rule-based. 38.8% pass@1 full GAIA | Agent-RRM improves accuracy; CASSI reduces cost. Complementary — could be combined |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | RM requires expensive GPT annotation. RM adds inference overhead. Only 8B. λ needs per-domain tuning. Doesn't optimize cost — uses extra compute | CASSI: free oracle labels, <3% overhead, cost-aware, stopper size flexible |
| **Future (Theirs→CASSI)** | N/A | Scale to larger models, broader toolsets, real-world → CASSI's cost-awareness could make Agent-RRM's RM cost-efficient; CASSI's oracle labels could replace GPT annotation |
| **Relevance** | 4/5 — Validates fine-grained intermediate feedback for agents. Shared benchmarks enable direct comparison. Complementary: critiques could enhance stopper, cost-awareness could enhance their RM |

---

### Paper 15: CSO — "Verified Critical Step Optimization for LLM Agents"
**Authors:** Li et al. (2026) | **URL:** https://arxiv.org/abs/2602.03412

| Element | CASSI | CSO | Contrast |
|---|---|---|---|
| **Background** | Overthinking — cost-aware PRM via oracle labels | Agent post-training: coarse credit assignment (trajectory-level penalties hit good steps in failures), noisy step-level rewards, expensive MC methods | Both address credit assignment quality. Different focus: CSO = action quality; CASSI = when to stop |
| **Motivation** | Oracle labels (O(T)) for cost-aware PRM | Only small fraction of steps drive effective RL. ~10% are "critical" — alternate actions flip outcomes. Focus learning on verified critical points — more efficient than dense supervision | Both argue selective supervision > uniform. CSO: 16% of steps. CASSI: O(T) oracle labels |
| **Approach** | `t* = argmax[quality − λ×cost]`. Stopper → executor GRPO | Semi-online DPO: (1) Collect failed trajectories. (2) Identify critical steps via PRM scoring. (3) Expert alternatives (Claude-3.7-Sonnet). (4) Branch rollouts with policy. (5) Outcome verification. (6) DPO on verified pairs. (7) Iterate | CASSI: oracle labels for stopping. CSO: verified steps for action quality. Both computationally efficient in supervision (CSO: 16% steps; CASSI: O(T)). Both leverage completed trajectories |
| **Implementation** | Stopper 0.5B–3B + executor 7B–72B | CK-Pro-8B (Qwen3-8B + 47K SFT). Claude-3.7-Sonnet as PRM (5-dim rubric). β=0.5 (DPO), K=5, γ_high=0.65, γ_low=0.45, 2 rounds | CSO: single model + external closed PRM. CASSI: two trained models, open |
| **Results** | Expected: 20–40% cost reduction | 37% relative over SFT on GAIA (35.9%→49.5%). 8B matches GPT-4.1 (49.5% vs 45.6%). Only 16% steps need supervision (671 vs 4,126). Iterative improves while ETO degrades | CSO improves accuracy; CASSI reduces cost. Both on GAIA → direct comparison possible |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Outcome verification requires full trajectory execution (slow). Closed-source PRM dependency. Only 8B. No cost modeling. No stopping | CASSI: open models, explicit cost, stopping decisions, zero branch rollouts |
| **Future (Theirs→CASSI)** | N/A | Early stopping for verification, parallelized execution, joint PRM+policy training → CASSI's stopper can serve as early-stop heuristic; CASSI's approach directly enables joint training |
| **Relevance** | 4/5 — Validates selective supervision. Complementary: CSO for action quality + CASSI for cost-aware termination. Shared GAIA benchmark enables direct comparison |

---

### Paper 16: CARL — "CARL: Critical Action Focused Reinforcement Learning for Multi-Step Agent"
**Authors:** Shen et al. (2025) | **URL:** https://arxiv.org/abs/2512.04949

| Element | CASSI | CARL | Contrast |
|---|---|---|---|
| **Background** | Overthinking — cost-aware PRM | GRPO assumes equal contribution of all actions — in multi-step agents this is invalid. >50% of actions have near-zero impact. ~10% are "critical" | Both identify GRPO's uniform treatment problem. CARL: credit assignment. CASSI: cost-aware stopping |
| **Approach** | `t* = argmax[quality − λ×cost]`. Stopper → executor GRPO | (1) Critical Action via Entropy — MC entropy as criticality proxy (top-10% = high entropy). (2) Tree Differencing — Bellman-style averaging, advantage = E[R(child)] − E[R(parent)]. (3) Entropy-Guided Rollout + Selective Update — fork from high-entropy states, exclude low-criticality from updates | Both per-step granularity. Both process-level signals. CARL: tree differencing advantages. CASSI: oracle stopping labels. CARL's entropy-guided forking is novel |
| **Results** | Expected: 20–40% cost reduction | 3B: 34.0 vs 33.3 F1. 4B: 58.4 vs 57.0 F1. Uses 39.6% training samples vs GRPO. <50% tokens at inference (770 vs 1543). Higher policy entropy (no premature collapse). HotpotQA, MuSiQue | Shared benchmarks. CARL improves training+inference efficiency; CASSI directly targets inference cost |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Needs baseline capability (if confidently wrong, CARL wins). No large models or GAIA tested. Entropy estimation overhead. No cost model. No stopping | CASSI: stopping mechanism, cost model, diverse benchmarks (including GAIA) |
| **Future (Theirs→CASSI)** | N/A | Scale to larger models, GAIA, multi-agent → CASSI directly addresses (GAIA). CARL's action-level advantages could enhance CASSI's executor GRPO |
| **Relevance** | **4/5** — Most complementary. CARL improves "which actions to optimize"; CASSI determines "when to stop." Combining: CARL's advantages + CASSI's cost-aware stopper. Only ~10% actions matter aligns with CASSI's overthinking observation |

---

### Paper 17: GiGPO — "Group-in-Group Policy Optimization for LLM Agent Training"
**Authors:** Feng et al. (NeurIPS 2025) | **URL:** https://arxiv.org/abs/2505.10978

| Element | CASSI | GiGPO | Contrast |
|---|---|---|---|
| **Background** | Overthinking — cost-aware PRM | GRPO excels single-turn but struggles multi-turn (sparse/delayed rewards). Credit assignment across many steps is bottleneck | Both extend GRPO for multi-turn agents. Different angle: advantage estimation vs reward design |
| **Approach** | `t* = argmax[quality − λ×cost]`. Stopper → executor GRPO | Two-level: (1) Episode-level macro advantages (standard GRPO). (2) Step-level micro advantages via anchor state grouping — identifies repeated environment states across trajectories, groups actions from same state, computes relative advantages. Same GPU as GRPO | CASSI: cost-aware rewards from stopper. GiGPO: better advantage estimation via anchor grouping. Address orthogonal problems |
| **Results** | Expected: 20–40% cost reduction | +12% ALFWorld, +9% WebShop over GRPO. Same memory as GRPO, negligible overhead | Both improve over baseline GRPO. GiGPO: accuracy. CASSI: cost reduction |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Anchor state grouping needs shared states (not always available). Limited benchmarks. No cost/efficiency focus | CASSI doesn't need shared states. CASSI explicitly targets cost |
| **Relevance** | 3/5 — Anchor grouping could combine with CASSI's cost-aware rewards. GiGPO improves "credit assignment quality"; CASSI improves "reward content." Fully complementary |

---

### Paper 18: GRPO — "DeepSeekMath: Pushing the Limits of Mathematical Reasoning"
**Authors:** Shao et al. (2024) | **URL:** https://arxiv.org/abs/2402.03300

| Element | CASSI | GRPO (DeepSeekMath) | Contrast |
|---|---|---|---|
| **Background** | Overthinking — cost-aware PRM. Uses GRPO as RL engine | Introduces GRPO: eliminates critic model. Samples G outputs per question, normalizes rewards within group. 120B-token math corpus also contributed | **CASSI builds on GRPO.** CASSI's innovation is on top of GRPO's foundation |
| **Motivation** | Oracle labels (O(T)) for cost-aware PRM | PPO needs critic of comparable size → memory-intensive. GRPO: no critic → halves memory. Group normalization provides stable baselines without learned value function | CASSI extends GRPO with: (a) cost-aware rewards, (b) trained stopping model as process reward source, (c) two-model cooperative training |
| **Approach** | CASSI uses GRPO for both stopper and executor. Modifies reward: adds Δ(s_t) from stopper as process reward | For each q: sample G outputs, compute rewards, group-relative advantage: `A_i = (r_i − mean(r)) / std(r)`. Optimize: clip-ratio + KL penalty. Unified: RFT, DPO, PPO, GRPO are RL variants | CASSI keeps GRPO's core (clip-ratio, KL, group advantage). Extends reward with cost-awareness |
| **Key Finding (GRPO paper)** | N/A | Outcome reward beats process reward for math. Process reward experiments inconclusive — difficult to get right | CASSI directly addresses this: oracle stopping labels provide high-quality process rewards. CASSI may succeed where DeepSeekMath's process rewards failed |
| **Weaknesses (Theirs)** | Oracle labels depend on trajectory quality | Binary outcome reward may not scale to complex tasks. Process rewards hard. Math only. No formal convergence analysis | CASSI: cost-aware process rewards from oracle labels, agent domains (not just math), two-model architecture provides specialized reward signals |
| **Future (Theirs→CASSI)** | N/A | Extend to broader domains, process-level rewards → CASSI directly addresses both: agent domains, process rewards from stopper |
| **Relevance** | **5/5** — Algorithmic foundation. CASSI's key innovation: making GRPO cost-aware through oracle-guided stopping rewards + process rewards from trained stopper. Addresses GRPO paper's key shortcoming (difficulty of good process rewards) |

---

*Continue to Part 3...*
