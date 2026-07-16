# Novelty Assessment: Monitor Agent as Reward Model for Cost-Aware Stopping

## 1. What Existing Work Does That Overlaps with Our Proposal

### 1.1 Budget-Aware Agent Execution (Partial Overlap)

**BATS** (Liu et al., 2025) implements budget tracking and verification modules that decide whether to "dig deeper" or "pivot." This is the closest existing work to our monitor agent concept. However, BATS uses **fixed heuristics** — a prompt-level budget tracker and a hand-designed verification module — not a learned model. BATS has no training component and cannot improve its stopping decisions over time.

**BAVT** (Li et al., 2026) uses the remaining budget ratio as a scaling exponent for explore/exploit trade-offs. This is a mathematical formulation of the cost-benefit trade-off we want to learn, but it is **parameter-free and non-learned**. The exponent is a fixed function of remaining budget, not a learned function of reasoning quality, confidence, and problem difficulty.

**INTENT** (Liu et al., 2026) separates intention-level planning from execution-level decisions, which structurally resembles our monitor-executor split. But INTENT is purely inference-time with a world model learned from static trajectory data — no RL training, no reward model, no dynamic adaptation.

### 1.2 Training Models for Efficient Reasoning (Significant Overlap)

**Training Language Models to Reason Efficiently** (Arora & Zanette, NeurIPS 2025) and **L1** (Aggarwal & Welleck, 2025) both train reasoning models with length penalties via RL. These papers show that:
- Simple RL rewards (correctness - α * length) produce efficient reasoning
- Length-constrained policy optimization (LCPO) enables smooth cost-accuracy trade-offs
- Efficiency training is surprisingly cheap (~200 gradient updates)

**This is the strongest overlap with our proposal.** These papers already embed cost-awareness into the policy model itself. Why do we need a separate monitor agent?

**Key difference:** These methods optimize for static length targets — the model learns to produce shorter reasoning overall, but the length constraint is a **pre-specified budget**, not a dynamic, context-dependent stopping decision. The model cannot say "this problem is harder, I need more tokens" or "I've gathered enough evidence, I should stop now." The monitor agent provides exactly this dynamic, per-instance adaptation.

### 1.3 Process Reward Models (Partial Overlap)

**AgentPRM** (Choudhury, 2025) provides the architectural template for our monitor — an actor-critic framework where a PRM evaluates state-action pairs and the policy is updated via RL using these step-level rewards. Our monitor agent IS a process reward model specialized for cost-aware stopping.

**Agent-RRM** (Fan et al., 2026) produces structured feedback (scores + reasoning traces + critiques), showing that reward models can provide richer signals than scalar scores. This maps to our vision of a monitor that explains WHY it recommends stopping.

**CSO** (Li et al., 2026) focuses optimization on "critical steps" — decisions that demonstrably flip outcomes. The "stop vs. continue" decision is exactly such a critical step, and CSO's verification-by-policy-execution approach could be adapted for training our monitor.

### 1.4 Termination Learning (Direct Overlap)

**CaRT** (Liu et al., 2025) is the closest prior work in spirit — it explicitly trains LLMs to decide when to terminate information gathering using counterfactual pairs. However:
- CaRT trains the **same model** to both reason AND terminate (no separate monitor)
- CaRT uses SFT on counterfactual examples, not RL with a reward model
- CaRT's termination decision is binary (terminate/continue) without cost-awareness
- CaRT does not address the reward model aspect of our proposal

### 1.5 Separate Monitor/Controller Architectures (Partial Overlap)

**ReMA** (Wan et al., 2025) decouples reasoning into a high-level meta-thinking agent and a low-level reasoning agent, trained via multi-agent RL. This is architecturally similar to our proposal but ReMA's meta-agent provides strategic plans, not cost-aware stopping decisions.

**MGV** (Oh & Gobet, 2025) formalizes a Monitor-Generate-Verify framework based on metacognitive theory, arguing that monitoring should precede generation. This provides theoretical grounding for our approach but is purely a position paper with no empirical implementation.

**Stop Wasting Your Tokens** (2025) uses a SupervisorAgent that decides when to intervene in multi-agent systems. This is close to our monitor concept but operates at the multi-agent orchestration level, not within a single agent's reasoning process.

---

## 2. What Gaps Exist in the Literature

### Gap 1: No Learned Cost-Aware Stopping Model

Every existing budget-aware agent framework (BATS, BAVT, INTENT) uses training-free heuristics. No existing work trains a separate model whose sole purpose is to make cost-aware stopping decisions. All current approaches either (a) hardcode stopping rules, (b) embed budget signals in prompts, or (c) train the policy model itself with a length penalty — but none trains a dedicated monitor.

### Gap 2: No Process-Level Cost-Quality Reward Model for Agents

Process reward models exist (AgentPRM, Agent-RRM) but are trained for general task quality, not cost-quality trade-offs. Multi-component rewards (ToolRL) include format and execution success but not cost. No existing PRM outputs a signal like "the expected marginal benefit of continuing is less than the expected marginal cost."

### Gap 3: Static Budgets vs. Dynamic Stopping

All token-budget methods (TALE, SelfBudgeter, BudgetThinker, L1) estimate or specify a token budget **before generation begins**. None adapts the budget mid-trajectory based on observed reasoning quality. The distinction between "I'll give you 500 tokens" and "I'll watch your reasoning and tell you when you've done enough" is the key gap.

### Gap 4: No Integration of Monitoring with Reinforcement Learning for Agents

Meta-reasoning and reflection methods (Re-ReST, Agent-R, SELF) use separate reflector components but train them via SFT on reflection data, not via RL as a reward model. The idea of a monitor that is trained via RL to provide rewards that train the executor via RL — a kind of hierarchical RL — is unexplored.

### Gap 5: No Learned "Soft Stopping Curve"

Existing work treats stopping as binary (hard threshold: budget exhausted → stop) or as a static length penalty. No work has proposed learning a **continuous cost-benefit curve** that models the diminishing marginal utility of additional computation as a function of problem difficulty, reasoning progress, and remaining budget.

---

## 3. Is Our Proposal Novel? Assessment

**Yes, our proposal is novel as an integrated system**, though individual components have precedent:

| Component | Novelty Status | Precedent |
|-----------|---------------|-----------|
| Monitor agent as separate model | **Partially novel** | ReMA, MGV, Stop Wasting Tokens have separate controllers, but none for cost-aware stopping |
| Cost-aware stopping decisions | **Novel** | No existing work trains a model to make dynamic, cost-aware stop/continue decisions |
| Monitor as reward model for executor training | **Novel** | No existing work uses a cost-aware monitor to generate training signals for the executor |
| Learned soft stopping curve | **Novel** | All existing methods use hard thresholds or static length penalties |
| RL training of the monitor | **Partially novel** | AgentPRM trains PRMs for quality; no one trains a PRM for cost-quality trade-offs |

The **novel contribution** is the specific combination: a monitor agent trained via RL to serve as a cost-aware process reward model that provides dynamic, learned stopping signals during executor training, producing a soft stopping curve rather than hard budget enforcement.

---

## 4. What Is Unique vs. Incremental

### Unique Aspects

1. **Dynamic stopping within a trajectory:** Unlike all token-budget work (TALE, SelfBudgeter, BudgetThinker, L1) which pre-commits to a budget, our monitor observes the actual reasoning and decides in real-time.

2. **Separate monitor model:** Unlike all RL-for-efficiency work (Reason Efficiently, L1, BudgetThinker) which embeds cost-awareness in the policy model itself, we use a dedicated monitor. This enables the monitor to be smaller, specialized, and potentially shared across different executor models.

3. **Learned cost-benefit trade-off:** Unlike BAVT (fixed exponent), BATS (heuristic verification), and budget forcing (hard "Wait"/truncate), our monitor learns what "good enough" means from data.

4. **Training paradigm:** The monitor-as-reward-model creates a new training loop: the monitor learns to evaluate cost-quality trade-offs, and the executor learns to optimize against the monitor's signals. This is fundamentally different from both (a) training the executor directly with a length penalty and (b) using a generic quality PRM.

### Incremental Aspects

1. The **RL training methodology** (GRPO, DPO) is well-established (Category 4).
2. The **concept of process-level rewards** is well-established (AgentPRM, Agent-RRM).
3. The **empirical observation that overthinking exists** is well-established (Categories 2 and 5).
4. The **use of separate components for oversight** is established (Category 3).

---

## 5. Recommendations for Positioning the Paper

### 5.1 Frame the Problem as a New Training Paradigm

Position the paper as introducing a **new training paradigm for agents**: not "teach the agent to be efficient" (which L1, Reason Efficiently already do) but "teach a separate monitor to evaluate cost-quality trade-offs and use it to train the agent." This is analogous to how GANs introduced adversarial training or how actor-critic methods separated policy from value estimation.

### 5.2 Emphasize Dynamic vs. Static Budget Allocation

The strongest contrast with prior work is the shift from static budgets (TALE, L1, BudgetThinker) to dynamic, observation-based stopping. Frame this as moving from "open-loop" budget control to "closed-loop" cost-aware decision making.

### 5.3 Leverage the "Soft Stopping Curve" Concept

The term "soft stopping curve" is novel and intuitive. It captures the idea that stopping is not binary (budget=0 → stop) but a continuous function of cost, quality, and confidence. This contrasts with:
- Hard budget forcing (s1): "You have 500 tokens, then you MUST stop"
- Fixed length penalties (Reason Efficiently): "Shorter is always better"
- Heuristic verification (BATS): "If verification fails, dig deeper"

### 5.4 Use CaRT as the Primary Baseline

CaRT (Liu et al., 2025) is the closest prior work — it trains models to terminate information gathering. Position our work as extending CaRT by: (a) using a separate monitor instead of self-termination, (b) incorporating cost-awareness (not just correctness), (c) using RL instead of SFT, and (d) making the monitor a reward model for executor training.

### 5.5 Situate Within the PRM Literature

Frame the monitor agent as a new type of Process Reward Model — one optimized for cost-quality trade-offs rather than pure quality. This connects to the well-established PRM literature (AgentPRM, Math-Shepherd) while claiming a distinct contribution.

### 5.6 Target Venues

The paper spans multiple communities:
- **Agent training** (NeurIPS, ICML, ICLR) — primary venue
- **Efficient ML** (NeurIPS, ICML) — secondary
- **RL for LLMs** (ICLR, NeurIPS) — fits well
- **NLP/Reasoning** (ACL, EMNLP) — relevant but narrower

### 5.7 Potential Weaknesses to Address Preemptively

1. **"Why not just train the policy model with a length penalty?"** — Address this head-on: length penalties are static and instance-independent; a monitor provides dynamic, context-aware stopping that adapts to problem difficulty and reasoning quality.

2. **"Does the monitor add inference cost?"** — Yes, but the monitor can be a much smaller model (e.g., 0.5B parameters), making its cost negligible compared to the savings from stopping large models early.

3. **"How do you train the monitor without ground-truth stopping labels?"** — This is the key methodological challenge. Possible approaches: (a) outcome-based RL (reward the monitor when the executor succeeds with fewer resources), (b) curriculum learning starting from heuristic stopping rules, (c) counterfactual training similar to CaRT.

### 5.8 Experimental Recommendations

1. **Baselines:** (a) Standard ReAct without budget awareness, (b) BATS (prompt-level budget tracking), (c) L1/LCPO (length-constrained policy), (d) CaRT (SFT-based termination), (e) Fixed budget stopping.
2. **Metrics:** Accuracy vs. cost (tokens, tool calls, wall-clock time) Pareto curves; cost at iso-accuracy; accuracy at iso-cost.
3. **Domains:** Web search agents (GAIA, WebWalkerQA), coding agents (SWE-bench), multi-hop QA, math reasoning.
4. **Ablations:** Monitor size, training signal (outcome-only vs. process-level), static vs. dynamic stopping, monitor as reward model vs. monitor as stopping controller only.
