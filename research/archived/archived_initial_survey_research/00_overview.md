# Literature Review Overview: Cost-Aware Agent Training via Learned Stopping Curves

## Summary of the Overall Landscape

Our proposal — a **monitor agent** that tracks budget, judges whether results are "good enough," decides whether the executor agent should continue/adjust/stop, and acts as a **reward model** to train the executor — sits at the intersection of six active research areas. This review consolidates approximately **48 papers** from three parallel research subagent outputs, deduplicated and organized into six thematic categories.

The central insight emerging from this review is that **no existing work combines a learned quality monitor with real-time, dynamic stopping decisions within a single model's reasoning process**. While each of the six categories contains highly relevant individual contributions, the integration of cost-aware stopping as a training signal — rather than a hard budget constraint — appears genuinely novel.

### Our Proposal in Context

Existing work can be classified along two axes:
1. **What is controlled:** Token budgets (Category 5), tool-call counts (Category 1), model selection (misc.), or reasoning length (Category 2).
2. **How control is achieved:** Hard stopping thresholds, prompt-based budget enforcement, inference-time heuristics, or RL with length penalties embedded in the policy model itself.

Our proposal differs on both axes:
1. **What is controlled:** A learned "soft stopping curve" that balances cost (tokens, tool calls, iterations) against expected value, producing nuanced stop/continue/adjust decisions.
2. **How control is achieved:** A **separate monitor agent** trained via RL to serve as a reward model, providing process-level cost-quality signals during executor training.

---

## Six Categories with Paper Counts

| # | Category | Papers | Core Question |
|---|----------|--------|---------------|
| 1 | **Cost-Aware & Budget-Constrained Agent Frameworks** | ~8 | How do we make agents budget-aware during execution? |
| 2 | **Agent Stopping, Early Exit & Adaptive Termination** | ~8 | When should an agent stop reasoning/interacting? |
| 3 | **Meta-Reasoning, Monitor-Executor Architectures & Metacognition** | ~8 | How can a separate monitor guide an executor? |
| 4 | **RLHF, Preference Optimization & RL Training for Agent Behaviors** | ~8 | How do we train agents with fine-grained reward signals? |
| 5 | **Token-Efficient Reasoning & Budget-Aware Inference** | ~8 | How can reasoning be made more token-efficient? |
| 6 | **Efficient Tool Use & Long-Horizon/Deep Research Tasks** | ~8 | How to optimize tool calls and long-horizon tasks? |

---

## How the Categories Relate to Each Other and to Our Proposal

```
                         OUR PROPOSAL
                    (Monitor Agent as Reward Model
                     for Learned Soft Stopping)
                              |
          +-------------------+-------------------+
          |                   |                   |
    Category 3           Category 4          Category 2
  (Meta-Reasoning,    (RLHF/Preference     (Stopping & Early
   Monitor-Executor)   Optimization)           Exit)
          |                   |                   |
          +-------------------+-------------------+
                              |
                    +---------+---------+
                    |                   |
              Category 1           Category 5
           (Cost-Aware Agents)   (Token-Efficient
                                Reasoning)
                    |                   |
                    +---------+---------+
                              |
                         Category 6
                    (Efficient Tool Use &
                     Deep Research Tasks)
```

- **Category 1** (Cost-Aware Agent Frameworks) provides the "what" — the problem of budget-constrained agent execution. BATS, BAVT, and INTENT show that budget awareness improves agent performance, but all use **training-free heuristics** rather than learned stopping.

- **Category 2** (Stopping & Early Exit) provides the "when" — empirical and theoretical evidence that there exists an optimal stopping point (inverted U-shaped accuracy curves, diminishing returns). CaRT directly trains models to decide when to terminate but does so within a single model rather than via a separate monitor.

- **Category 3** (Meta-Reasoning & Monitor-Executor) provides the "architectural template" — the idea of separating monitoring from execution. MGV, ReMA, and Agent-R use separate components for oversight and execution, but none apply this to cost-aware stopping.

- **Category 4** (RLHF & Preference Optimization) provides the "how to train" — the algorithmic machinery (GRPO, DPO, PRMs) for training agents with fine-grained reward signals. AgentPRM and CSO are closest to providing the training framework our monitor agent would need.

- **Category 5** (Token-Efficient Reasoning) provides the "target metric" — evidence that token budgets matter, that per-problem estimation is valuable, and that RL with length penalties produces efficient reasoning. These papers optimize for efficiency but embed it in the policy model itself rather than using a separate monitor.

- **Category 6** (Efficient Tool Use) provides the "application domain" — long-horizon tasks with expensive tool calls are where cost-aware stopping has the highest potential impact. DeepResearcher, IterResearch, and ToolRL show that RL can optimize tool-use behavior.

---

## Master Bibliography (All ~48 Papers, Deduplicated)

1. **BATS** — Liu et al., "Budget-Aware Tool-Use Enables Effective Agent Scaling," arXiv 2025. https://arxiv.org/abs/2511.17006
2. **BAVT** — Li et al., "Spend Less, Reason Better: Budget-Aware Value Tree Search for LLM Agents," arXiv 2026. https://arxiv.org/abs/2603.12634
3. **INTENT** — Liu et al., "Budget-Constrained Agentic Large Language Models: Intention-Based Planning for Costly Tool Use," arXiv 2026. https://arxiv.org/abs/2602.11541
4. **IterResearch** — "IterResearch: Rethinking Long-Horizon Agents via Markovian State Reconstruction," arXiv 2025. https://arxiv.org/abs/2511.07327
5. **Trajectory Reduction** — "Reducing Cost of LLM Agents with Trajectory Reduction," arXiv 2025. https://arxiv.org/abs/2509.23586
6. **SMART** — "SMART: Self-Aware Agent for Tool Overuse Mitigation," arXiv 2025. https://arxiv.org/abs/2502.11435
7. **Stop Wasting Tokens** — "Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems," arXiv 2025. https://arxiv.org/abs/2510.26585
8. **AutoTool** — "AutoTool: Efficient Tool Selection for Large Language Model Agents," arXiv 2025. https://arxiv.org/abs/2511.14650
9. **CaRT** — Liu et al., "CaRT: Teaching LLM Agents to Know When They Know Enough," arXiv 2025. https://arxiv.org/abs/2510.08517
10. **Dynamic Early Exit** — "Dynamic Early Exit in Reasoning Models," arXiv 2025. https://arxiv.org/abs/2504.15895
11. **Diminishing Returns Early-Exit** — "The Diminishing Returns of Early-Exit Decoding in Modern LLMs," arXiv 2026. https://arxiv.org/abs/2603.23701
12. **s1** — Muennighoff et al., "s1: Simple test-time scaling," EMNLP/ICLR Workshop 2025. https://arxiv.org/abs/2501.19393
13. **Budget Forcing + RL** — Tarunokusumo & Cunha, "Boosting Accuracy and Efficiency of Budget Forcing via RL," arXiv 2025. https://arxiv.org/abs/2510.21398
14. **Don't Overthink It** — Hassid et al., "Don't Overthink It: Preferring Shorter Thinking Chains," arXiv 2025. https://arxiv.org/abs/2505.17813
15. **Over-Reasoning** — Chiang & Lee, "Over-Reasoning and Redundant Calculation of Large Language Models," arXiv 2024. https://arxiv.org/abs/2401.11467
16. **When More is Less** — Wu et al., "When More is Less: Understanding Chain-of-Thought Length in LLMs," arXiv 2025. https://arxiv.org/abs/2502.07266
17. **MGV** — Oh & Gobet, "Monitor-Generate-Verify: Formalising Metacognitive Theory for Language Model Reasoning," FoRLM 2025. https://arxiv.org/abs/2511.04341
18. **ReMA** — Wan et al., "ReMA: Learning to meta-think for LLMs with multi-agent reinforcement learning," arXiv 2025. https://arxiv.org/abs/2503.09501
19. **Structured Meta-Cognition** — "Deep Reasoning in General Purpose Agents via Structured Meta-Cognition," arXiv 2025. https://arxiv.org/abs/2605.11388
20. **Re-ReST** — "Re-ReST: Reflection-Reinforced Self-Training for Language Agents," arXiv 2024. https://arxiv.org/abs/2406.01495
21. **Agent-R** — "Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training," arXiv 2025. https://arxiv.org/abs/2501.11425
22. **SELF** — "SELF: Self-Evolution with Language Feedback," arXiv 2023. https://arxiv.org/abs/2310.00533
23. **PACER** — Zhang et al., "A Single Revision Step Improves Token-Efficient LLM Reasoning," arXiv 2026. https://arxiv.org/abs/2602.02828
24. **Budget Guidance** — Li et al., "Steering LLM Thinking with Budget Guidance," arXiv 2025. https://arxiv.org/abs/2506.13752
25. **GRPO** — Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning," arXiv 2024. https://arxiv.org/abs/2402.03300
26. **GiGPO** — Feng et al., "Group-in-Group Policy Optimization for LLM Agent Training," NeurIPS 2025. https://arxiv.org/abs/2505.10978
27. **Training-Free GRPO** — Cai et al., "Training-Free Group Relative Policy Optimization," arXiv 2025. https://arxiv.org/abs/2510.08191
28. **Step-DPO** — Lai et al., "Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning," ICLR 2025. https://arxiv.org/abs/2406.18629
29. **EntroPO** — Yu et al., "Building Coding Agents via Entropy-Enhanced Multi-Turn Preference Optimization," arXiv 2025. https://arxiv.org/abs/2509.12434
30. **CSO** — Li et al., "Verified Critical Step Optimization for LLM Agents," arXiv 2026. https://arxiv.org/abs/2602.03412
31. **AgentPRM** — Choudhury, "Process Reward Models for LLM Agents: Practical Framework and Directions," arXiv 2025. https://arxiv.org/abs/2502.10325
32. **Agent-RRM** — Fan et al., "Agent-RRM: Exploring Reasoning Reward Model for Agents," arXiv 2026. https://arxiv.org/abs/2601.22154
33. **Compute-Optimal Inference** — Snell et al., "Scaling LLM Test-Time Compute Optimally," ICLR 2025. https://arxiv.org/abs/2408.03314
34. **Inference Scaling Laws** — Wu et al., "Inference Scaling Laws: Compute-Optimal Inference for Problem-Solving," ICLR 2025. https://arxiv.org/abs/2408.00724
35. **TALE** — Han et al., "Token-Budget-Aware LLM Reasoning," ACL 2025 Findings. https://arxiv.org/abs/2412.18547
36. **SelfBudgeter** — Li et al., "SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning," arXiv 2025. https://arxiv.org/abs/2505.11274
37. **BudgetThinker** — Wen et al., "BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens," arXiv 2025. https://arxiv.org/abs/2508.17196
38. **DiffAdapt** — Liu et al., "DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference," arXiv 2025. https://arxiv.org/abs/2510.19669
39. **L1** — Aggarwal & Welleck, "L1: Controlling How Long a Reasoning Model Thinks With RL," arXiv 2025. https://arxiv.org/abs/2503.04697
40. **Reason Efficiently** — Arora & Zanette, "Training Language Models to Reason Efficiently," NeurIPS 2025. https://arxiv.org/abs/2502.04463
41. **Plan-and-Act** — "Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks," arXiv 2025. https://arxiv.org/abs/2503.09572
42. **Evolution of Tool Use** — "The Evolution of Tool Use in LLM Agents," arXiv 2025. https://arxiv.org/abs/2603.22862
43. **To Call or Not to Call** — "To Call or Not to Call: A Framework to Assess and Optimize LLM Tool Calling," arXiv 2025. https://arxiv.org/abs/2605.00737
44. **DeepResearcher** — "DeepResearcher: Scaling Deep Research via RL in Real-world Environments," arXiv 2025. https://arxiv.org/abs/2504.03160
45. **ReTool** — Feng et al., "ReTool: Reinforcement Learning for Strategic Tool Use in LLMs," arXiv 2025. https://arxiv.org/abs/2504.11536
46. **ToolRL** — Qian et al., "ToolRL: Reward is All Tool Learning Needs," arXiv 2025. https://arxiv.org/abs/2504.13958
47. **RRO** — Wang et al., "RRO: LLM Agent Optimization Through Rising Reward Trajectories," arXiv 2025. https://arxiv.org/abs/2505.20737
48. **CARL** — Shen et al., "CARL: Critical Action Focused Reinforcement Learning for Multi-Step Agent," arXiv 2025. https://arxiv.org/abs/2512.04949

### Additional Referenced Papers (Not in Main 48)

- **CoRL** — Jin et al., "Controlling Performance and Budget of a Centralized Multi-agent LLM System with RL," arXiv 2025. https://arxiv.org/abs/2511.02755
- **BAMAS** — Yang et al., "BAMAS: Structuring Budget-Aware Multi-Agent Systems," AAAI 2026. https://ojs.aaai.org/index.php/AAAI/article/view/40226
- **FrugalGPT** — Chen et al., "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance," TMLR 2024. https://arxiv.org/abs/2305.05176
- **RouteLLM** — Ong et al., "RouteLLM: Learning to Route LLMs with Preference Data," arXiv 2024. https://arxiv.org/abs/2406.18665
- **Hybrid LLM** — Ding et al., "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing," ICLR 2024. https://arxiv.org/abs/2404.14618
- **TREACLE** — Zhang et al., "TREACLE: Efficient Contextual LLM Cascades through Budget-Constrained Policy Learning," NeurIPS 2024.
- **Token Economies** — Wang et al., "Reasoning in Token Economies: Budget-Aware Evaluation of LLM Reasoning Strategies," EMNLP 2024. https://arxiv.org/abs/2406.06461
- **Stop Overthinking** — Sui et al., "Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models," TMLR 2025. https://arxiv.org/abs/2503.16419
- **DeepSeek-R1** — DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," Nature 2025. https://arxiv.org/abs/2501.12948
- **SWE-RL** — Wei et al., "SWE-RL: Advancing LLM Reasoning via RL on Open Software Evolution," arXiv 2025. https://arxiv.org/abs/2502.18449
- **Agent-RLVR** — Da et al., "Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards," arXiv 2025. https://arxiv.org/abs/2506.11425
- **Long-Context SWE** — Golubev et al., "Training Long-Context, Multi-Turn Software Engineering Agents with RL," arXiv 2025. https://arxiv.org/abs/2508.03501
- **Constitutional AI** — Bai et al., "Constitutional AI: Harmlessness from AI Feedback," arXiv 2022. https://arxiv.org/abs/2212.08073
- **RLAIF vs. RLHF** — Lee et al., "RLAIF vs. RLHF: Scaling RL from Human Feedback with AI Feedback," ICML 2024. https://arxiv.org/abs/2309.00267

---

## Key Themes Across All Categories

1. **Budget awareness improves performance** — Across all categories, the consensus is that making agents aware of resource constraints produces better cost-performance trade-offs than simply giving them more budget.

2. **LLM self-evaluation is unreliable** — The Token Economies paper, Over-Reasoning, and CaRT all demonstrate that LLMs cannot reliably assess their own reasoning quality or know when to stop. This is the single strongest motivation for a **separate** monitor agent.

3. **RL with length penalties works, but is coarse** — Categories 4 and 5 show that simple RL rewards combining correctness with length penalties produce efficient reasoning. But these are embedded in the policy model itself and provide no process-level feedback about when to continue vs. stop.

4. **Training-free approaches dominate current agent work** — BATS, BAVT, INTENT all use heuristics rather than learned stopping. No existing work trains a separate model for cost-aware agent stopping.

5. **Diminishing returns are universal** — Categories 2 and 5 provide strong empirical and theoretical evidence that more computation eventually stops helping, and that an optimal stopping point exists. The key gap is learning to detect this point in real-time.
