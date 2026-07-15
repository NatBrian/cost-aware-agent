# Category 1: Cost-Aware & Budget-Constrained Agent Frameworks

Papers that directly address making LLM agents budget/cost-aware during execution. These frameworks track resource consumption (tokens, tool calls, monetary cost) and adapt agent behavior accordingly — but all use **training-free heuristics** rather than learned stopping.

---

### BATS: Budget-Aware Tool-Use Enables Effective Agent Scaling

- **Authors:** Tengxiao Liu, Zifeng Wang, Jin Miao, I-Hung Hsu, Jun Yan, Jiefeng Chen, Rujun Han, Fangyuan Xu, Yanfei Chen, Ke Jiang, Samira Daruki, Yi Liang, William Yang Wang, Tomas Pfister, Chen-Yu Lee
- **Year/Venue:** 2025, arXiv preprint (Google Cloud AI / Google DeepMind / UC Santa Barbara)
- **URL:** https://arxiv.org/abs/2511.17006

**Background:** Test-time compute scaling has been extended to tool-augmented agents where "scaling" involves both thinking (tokens) and acting (tool calls). Prior work lacked systematic study of how agents behave under explicit tool-call budgets.

**Motivation:** Simply granting agents larger tool-call budgets fails to improve performance — they lack "budget awareness" and quickly hit a performance ceiling. Standard agents perform shallow searches and fail to utilize additional resources even when available.

**Approach:** Two-tiered approach. **(1) Budget Tracker:** A lightweight plug-and-play module compatible with any ReAct-based agent. It continuously inserts remaining budget signals into the prompt context at each step, including a policy guideline describing budget regimes (high/medium/low) and corresponding behavioral recommendations. This gives the agent continuous awareness of resource availability without any training. **(2) BATS (Budget-Aware Test-time Scaling):** A more advanced framework with two key modules — a Planning Module that adjusts stepwise effort to match current budget, and a Verification Module that dynamically decides whether to "dig deeper" into a promising lead or "pivot" to alternative paths based on remaining resources. BATS maintains a continuous signal of remaining resources and uses it to dynamically adapt behavior — flexibly switching between deepening and branching strategies. The framework formalizes a unified cost metric jointly accounting for token and tool consumption, enabling fair cost-performance comparison.

**Training Method:** No training required — purely inference-time prompting and orchestration. Budget Tracker is a prompt-level plugin. BATS adds planning and verification orchestration on top.

**Key Results:** Budget-aware methods produce more favorable scaling curves and push the cost-performance Pareto frontier. BATS achieves higher performance while using fewer tool calls and incurring lower overall cost than competing methods. The Budget Tracker alone provides effective scaling as a simple plug-in.

**Weakness:** Focused on search agents specifically; may not generalize to all agent types. Relies on prompt-level signals which may be fragile for very complex tasks. No learning component — the agent cannot improve its budget-aware behavior over time.

**Relevance to Our Work:** **High.** BATS directly addresses budget-constrained agent behavior with explicit budget tracking — analogous to what a monitor agent could provide. However, BATS uses fixed heuristics for the verification module while our approach uses a learned monitor agent as a reward model. The "dig deeper vs. pivot" decision in BATS is exactly the kind of stopping/continuing decision our monitor agent would learn to make.

---

### BAVT: Spend Less, Reason Better — Budget-Aware Value Tree Search for LLM Agents

- **Authors:** Yushu Li, Wenlong Deng, Jiatao Li, Xiaoxiao Li
- **Year/Venue:** 2026, arXiv preprint
- **URL:** https://arxiv.org/abs/2603.12634

**Background:** Test-time scaling treats compute as abundant, allowing agents to exhaust budgets on redundant steps or dead-end trajectories. Existing budget-aware methods either require expensive fine-tuning or rely on coarse trajectory-level heuristics that cannot intervene mid-execution.

**Motivation:** Current agents lack fine-grained budget control. They blindly allocate resources producing diminishing returns. Prior methods like BATS incorporate budget into prompts but rely entirely on LLM's implicit self-regulation and cannot detect/abandon failing trajectories in real time. The key gap is step-level budget-aware control.

**Approach:** BAVT is a training-free inference-time framework that models multi-hop reasoning as a dynamic search tree guided by step-level value estimation within a single LLM backbone. **(1) Budget-Conditioned Node Selection:** Uses the remaining resource ratio as a natural scaling exponent over node values, providing a principled parameter-free transition from broad exploration (when budget is high) to greedy exploitation (as budget depletes). When remaining budget r is high, the exponent r→1 makes node values nearly uniform (exploration); as r→0, selection becomes pure argmax (exploitation). **(2) Residual Value Predictor:** Instead of scoring absolute state quality (which suffers from LLM overconfidence), it scores relative progress — measuring how much closer a tool call moves the agent toward the answer compared to the current state. This enables reliable pruning of uninformative or redundant tool calls. **(3) Theoretical Guarantee:** Proves BAVT reaches a terminal answer with probability ≥ 1-ε under explicit finite budget bound.

**Training Method:** Training-free — all mechanisms operate at inference time. The residual value predictor uses the LLM itself for evaluation with a specially designed prompt asking it to estimate relative progress.

**Key Results:** BAVT under strict low-budget constraints surpasses baseline performance at 4× the resource allocation. Consistently outperforms parallel sampling baselines across benchmarks. Demonstrates that intelligent budget management fundamentally outperforms brute-force compute scaling.

**Weakness:** Relies on LLM self-evaluation for the residual value predictor, which can still be noisy. Single LLM backbone may limit diversity. The convergence proof assumes idealized conditions.

**Relevance to Our Work:** **Very High.** BAVT's budget-conditioned node selection mechanism (using remaining budget ratio as exponent for explore/exploit trade-off) is conceptually similar to our monitor agent providing a learned stopping signal. The residual value predictor (scoring relative progress rather than absolute quality) addresses the same overconfidence problem our monitor agent would handle. BAVT is training-free while we propose training the monitor — a natural extension.

---

### INTENT: Budget-Constrained Agentic LLMs — Intention-Based Planning for Costly Tool Use

- **Authors:** Hanbing Liu, Chunhao Tian, Nan An, Ziyuan Wang, Pinyan Lu, Changyuan Yu, Qi Qi
- **Year/Venue:** 2026, arXiv preprint (Renmin University of China)
- **URL:** https://arxiv.org/abs/2602.11541

**Background:** LLM agents increasingly access tool marketplaces (MCP, RapidAPI) with thousands of heterogeneous APIs. While this expands action spaces, it introduces economic cost — many tools expose scarce, monetized resources. Prior work overlooks the economic dimension of tool use.

**Motivation:** The central question: "Can we trust agentic models to make cost-sensitive tool-use decisions on our behalf?" Even when given budget feedback after each tool call, strong models frequently exceed budgets due to repetitive retries and unproductive exploration. Advanced reasoning models are overly conservative, leaving a large performance gap.

**Approach:** INTENT is an inference-time planning framework using an intention-aware hierarchical world model. It operates at two levels: **(1) Intention Level:** The world model predicts future tool usage patterns ("intentions") at a coarse granularity — what categories of tools will be needed later, not exact tool calls. This enables risk-calibrated cost estimation without exhaustive search. **(2) Execution Level:** The agent makes actual tool-call decisions guided by the intention-level predictions. The framework anticipates future spending through "intention-level" simulations, allowing the agent to reserve budget for critical later steps. It strictly enforces hard budget feasibility — if an action would exceed budget, it's blocked. The world model is learned from historical trajectories and can adapt to dynamic market shifts (tool prices changing, new tools appearing). INTENT requires no retraining of the base LLM agent.

**Training Method:** No LLM training required. The hierarchical world model is learned from historical agent trajectories. Operates purely at inference time.

**Key Results:** INTENT strictly enforces hard budget feasibility — never exceeds budget. Substantially improves task success over baselines under budget constraints. Remains robust under dynamic market shifts. The intention-level planning enables anticipatory budget reservation.

**Weakness:** The world model depends on historical data quality. Intention-level predictions are approximate and may miss fine-grained dependencies. Evaluated on synthetic benchmark (StableToolBench) — real-world tool marketplaces may be more complex.

**Relevance to Our Work:** **High.** INTENT's hierarchical approach — separating intention-level planning from execution-level decisions — maps to our proposal's separation between the agent and the monitor. The budget-reservation mechanism is analogous to a monitor signaling "stop now, save budget for later." INTENT shows that explicit forward-looking cost estimation dramatically improves budget compliance.

---

### IterResearch: Rethinking Long-Horizon Agents via Markovian State Reconstruction

- **Authors:** (Multiple authors, 2025)
- **Year/Venue:** 2025, arXiv preprint
- **URL:** https://arxiv.org/abs/2511.07327

**Background:** Long-horizon deep research agents accumulate extensive context, making it difficult to maintain coherent reasoning across many steps. Prior work either used fixed context windows or heuristic summarization.

**Motivation:** Long-horizon agents need efficient state representation to avoid context bloat and make cost-effective decisions about when to continue vs. stop researching. The Markovian property — that the current state should summarize all relevant history — is violated by naive context accumulation.

**Approach:** IterResearch introduces iterative deep research with Markovian state reconstruction. At each research round, the agent compresses its accumulated findings into a structured "workspace" that serves as a Markovian state — containing only the information relevant for future decisions. This workspace is reconstructed at each iteration, discarding irrelevant details. The framework includes **Efficiency-Aware Policy Optimization** — an RL training objective with discounted reward shaping that penalizes excessive computation. The discount factor γ and per-step cost penalty encourage the agent to find solutions efficiently. The policy is trained on multi-round trajectories using group-relative optimization, with rewards combining task success with efficiency penalties.

**Training Method:** RL with efficiency-aware reward shaping — discounted rewards with per-step cost penalties. Group-relative policy optimization (GRPO-style) on multi-round trajectories.

**Key Results:** Markovian state reconstruction reduces context length and improves long-horizon reasoning coherence. The efficiency-aware policy optimization produces agents that solve tasks with fewer interactions. The approach works as both a prompting strategy and a training method.

**Weakness:** Workspace reconstruction quality depends on the compression model's capability. Efficiency reward shaping requires careful tuning of discount and penalty parameters. Evaluated on search-based research tasks only.

**Relevance to Our Work:** **Medium-High.** IterResearch's efficiency-aware policy optimization directly addresses the same goal — training agents to be cost-effective. The workspace reconstruction acts as a form of "monitoring" (is the current state sufficient?). Our monitor agent could be applied to IterResearch's setting to provide more nuanced stopping signals than a fixed discount factor.

---

### Reducing Cost of LLM Agents with Trajectory Reduction

- **Authors:** (Multiple authors, 2025)
- **Year/Venue:** 2025, arXiv preprint
- **URL:** https://arxiv.org/abs/2509.23586

**Background:** LLM agents produce verbose trajectories with redundant tool calls and repetitive reasoning steps. The cost of these trajectories grows linearly with length, even when marginal information gain approaches zero.

**Motivation:** Can we reduce the cost of LLM agent trajectories by identifying and removing redundant or uninformative steps without retraining the agent?

**Approach:** A trajectory reduction framework that post-processes agent trajectories to remove redundant steps. The approach uses a **redundancy detector** that identifies steps where: (1) the same tool was called with similar parameters, (2) the information retrieved duplicates previous information, (3) the reasoning step reiterates previous conclusions without adding new insights. The detector uses similarity metrics (embedding cosine similarity, n-gram overlap) to identify redundant segments. Once identified, redundant segments are removed and the trajectory is reconstructed. This is a post-hoc optimization — it reduces stored/analyzed trajectory cost rather than live inference cost.

**Training Method:** No agent training. The redundancy detector uses embeddings and similarity metrics. No RL.

**Key Results:** Significant trajectory compression (up to 40-60% reduction) with minimal information loss. The compressed trajectories are as useful for downstream tasks (analysis, training) as full trajectories. Redundancy is most prevalent in later trajectory steps, confirming diminishing returns.

**Weakness:** Post-hoc reduction doesn't save inference cost — only storage/analysis cost. The redundancy detector may remove subtle but important information. Requires the full trajectory to be generated first.

**Relevance to Our Work:** **Medium.** The finding that later steps are most redundant provides empirical support for early stopping. If a monitor agent could detect this redundancy in real-time, it could prevent the wasted computation rather than removing it after the fact.

---

### SMART: Self-Aware Agent for Tool Overuse Mitigation

- **Authors:** (Multiple authors, 2025)
- **Year/Venue:** 2025, arXiv preprint
- **URL:** https://arxiv.org/abs/2502.11435

**Background:** LLM agents frequently overuse tools — calling tools when the answer is already known or when the model's internal knowledge would suffice. This wastes both computational resources and API costs.

**Motivation:** Tool overuse is a pervasive problem. Agents call tools even for simple knowledge that the base LLM already possesses. This is a form of "overthinking" at the tool-use level — the agent doesn't know when to stop using tools and rely on its own knowledge.

**Approach:** SMART introduces a self-awareness mechanism for tool use mitigation. The approach has three components: **(1) Data Collection:** Generates training data by running the agent on tasks and labeling whether each tool call was necessary (did the tool call change the final answer?). **(2) Reasoning Chain Construction:** For each tool call, constructs a reasoning chain that includes the agent's internal knowledge assessment and the necessity justification. This teaches the model to distinguish between "I need to look this up" and "I already know this." **(3) Agent Training:** Fine-tunes the agent on these reasoning chains, teaching it to self-assess its knowledge boundaries and use tools only when necessary. The model learns to ask "Do I already know this?" before making a tool call.

**Training Method:** SFT on constructed reasoning chains with knowledge boundary assessment. Data generated by comparing tool-augmented vs. tool-free agent performance.

**Key Results:** Significant reduction in unnecessary tool calls. Maintains or improves task accuracy while reducing tool usage costs. The model learns to recognize its own knowledge boundaries more accurately.

**Weakness:** Knowledge boundary assessment is trained on historical data — may not generalize to new domains. The "necessary vs. unnecessary" labeling is noisy. Only tested on knowledge-retrieval tasks.

**Relevance to Our Work:** **Medium.** SMART addresses the "when to stop using tools" problem from a self-awareness perspective. Our monitor agent generalizes this: instead of the agent learning to self-assess, we provide an external monitor that assesses whether continued tool use is worthwhile. The distinction between self-assessment and external monitoring is a key architectural choice in our proposal.

---

### Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems

- **Authors:** (Multiple authors, 2025)
- **Year/Venue:** 2025, arXiv preprint
- **URL:** https://arxiv.org/abs/2510.26585

**Background:** Multi-agent systems (MAS) with LLMs suffer from excessive token consumption as each agent independently generates verbose reasoning and communication. The supervisor agent must process all this output, creating a context window bottleneck.

**Motivation:** In multi-agent systems, most agent interactions are low-risk and don't need supervision. Only critical interaction points require supervisor intervention. The question is: what to supervise, when to supervise, and how to supervise?

**Approach:** A runtime supervision framework with three components: **(1) What to Supervise (High-Risk Interaction Points):** Identifies critical interaction points where agent outputs diverge, conflict, or contain high-uncertainty claims. These are flagged for supervision. **(2) When to Supervise (Adaptive Filter):** An adaptive filter that dynamically decides whether the current interaction warrants supervisor attention based on confidence scores, output diversity, and historical patterns. Low-risk interactions pass through without supervision. **(3) How to Supervise (Memory-Augmented, Multi-Level Intervention):** A memory-augmented supervisor that can intervene at different levels — from simple confirmation to full re-generation — based on the severity of the detected issue. The supervisor maintains a memory of past interventions to avoid repeating the same corrections.

**Training Method:** No LLM training. The adaptive filter uses heuristics and confidence scores. The supervisor is prompt-based.

**Key Results:** Significant token reduction in multi-agent systems by selectively supervising only high-risk interactions. The adaptive filter reduces supervisor context window usage while maintaining oversight quality. Multi-level intervention allows proportional responses.

**Weakness:** Heuristic-based risk detection may miss subtle issues. The supervisor's effectiveness depends on prompt engineering. Only tested on specific multi-agent debate and collaboration tasks.

**Relevance to Our Work:** **Medium.** The SupervisorAgent in this work is the closest existing instantiation of our monitor agent concept — a separate component that decides when and how to intervene. However, it operates at the multi-agent orchestration level with heuristic filters, while our monitor operates within a single agent's reasoning with learned stopping signals.

---

### AutoTool: Efficient Tool Selection for Large Language Model Agents

- **Authors:** (Multiple authors, 2025)
- **Year/Venue:** 2025, arXiv preprint
- **URL:** https://arxiv.org/abs/2511.14650

**Background:** LLM agents face high inference costs for tool selection when the tool space is large (hundreds or thousands of APIs). Describing all tools in the prompt consumes significant tokens.

**Motivation:** Tool invocation exhibits predictable, low-entropy inertia — agents tend to reuse the same tools in predictable patterns. Can we exploit this to reduce the cost of tool selection?

**Approach:** AutoTool builds a **tool invocation graph** where nodes are tools and edges represent sequential and parameter dependency relationships. The graph is constructed from historical tool-use trajectories. At inference time, instead of listing all tools in the prompt, the agent searches the graph using a **Cost-Informed Path Selection (CIPS)** algorithm that balances tool relevance against selection cost. The graph is searched in a hierarchical manner — first identifying the relevant tool cluster, then selecting specific tools within that cluster. Parameter filling is also hierarchical: dependency edges indicate which tool outputs feed into which tool inputs, enabling efficient parameter propagation.

**Training Method:** No LLM training. The tool graph is constructed offline from historical data. Graph search is a lightweight algorithm running alongside the LLM.

**Key Results:** Significant reduction in tool selection prompt length. The graph-based approach maintains or improves tool selection accuracy compared to listing all tools. Hierarchical parameter filling reduces redundant tool calls.

**Weakness:** The graph requires maintenance as tools change. Graph construction depends on historical data availability. The approach assumes tool usage patterns are stable.

**Relevance to Our Work:** **Medium-Low.** AutoTool addresses cost-efficiency in tool selection, which is one dimension of the overall cost budget our monitor agent would track. The graph-based approach could complement our monitor by providing structured information about tool usage patterns, but it doesn't address the question of when to stop using tools.