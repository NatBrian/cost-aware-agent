# Competitor Analysis Part 4: Grand Synthesis

> Continues from Part 1 (`competitor_analysis.md`), Part 2 (`competitor_analysis_part2.md`), Part 3 (`competitor_analysis_part3.md`).
> Cross-cutting comparative tables, CASSI's unique positioning, and paper-writing recommendations.

---

## Cross-Cutting: What Each Paper Contributes vs. What's Missing

| Paper | Strengths Relative to CASSI | Weaknesses CASSI Addresses | Complementarity |
|---|---|---|---|
| **BATS** (Cat 1) | Training-free simplicity, web at scale, unified cost metric | No learning, heuristic stopping, single domain | CASSI adds learned stopping + multi-domain. BATS's cost metric informs CASSI's budget state |
| **INTENT** (Cat 1) | Intention decomposition, geometric cost calibration, budget enforcement proof | Ensemble of oracle models, frozen policy, single domain | CASSI adds executor training + broader domains. INTENT's cost calibration could enhance oracle labels |
| **CaRT** (Cat 1) | Counterfactual reasoning for termination, verbal value function, OOD robustness | Labor-intensive labeling, binary stop, no cost model, no tool-use | **Closest competitor.** CASSI adds: O(T) labels, λ cost, executor training, agent domains |
| **s1** (Cat 1) | Sample efficiency (1K examples), simplicity (26 min SFT), budget forcing with self-correction | Uniform forcing, no per-instance adaptation, math only, external control | CASSI adds: learned stopping, per-instance adaptation, agent domains. s1's data curation methodology useful |
| **AgentPRM** (Cat 1) | PRM for stochastic agents, iterative policy iteration, InversePRM (demo-based) | **O(K×T²) cost** (core weakness), text game only, reward hacking, no cost model | **Most important comparison.** CASSI's O(T) oracle labels directly beat O(K×T²). Both train PRM+policy |
| **BAVT** (Cat 2) | Budget-conditioned tree search, residual value predictor, theoretical guarantees | Training-free (can't learn), single model, QA only, no cost function | CASSI adds learning + diverse domains. BAVT's γ_t scaling mirrors CASSI's budget tiers |
| **DEER** (Cat 2) | "Pearl reasoning" detection, 31-43% reduction while improving accuracy | Requires transition tokens, CoT only, heuristic confidence, no tools | CASSI's stopper learns DEER's "pearl detection" with cost-awareness + agent applicability |
| **TALE** (Cat 2) | Token Elasticity discovery, 68% reduction, both training-free and post-training | Static per-question budget, math only, no per-step, no tools | CASSI adds: per-step granularity, agent domains, learned stopper avoids Token Elasticity |
| **ReMA** (Cat 3) | Two-agent RL for meta-thinking, MARL formulation, OOD generalization | 2× inference cost, multi-turn instability, quality-focused (not efficiency) | Similar architecture but opposite goals. CASSI's simpler binary task likely more stable |
| **MGV** (Cat 3) | Cognitive science grounding, formal metacognitive framework, theoretical vocabulary | No implementation, no training, no experiments | CASSI is MGV's Monitor-Generate-Verify implemented for cost-aware stopping |
| **CORL** (Cat 3) | Controller-executor for multi-LLM routing with cost, budget-conditioned training | 7B controller (not lightweight), routes between models, not step-level | Validates architecture. CASSI refines to per-step + smaller controller |
| **L1** (Cat 4) | User-specified budgets via prompt, SRM discovery, strong length control precision | User must specify budget, math only, no per-instance automatic, training budget ceiling | CASSI: automatic per-instance, no user budget, agent domains, no ceiling |
| **Reason Efficiently** (Cat 4) | 50% token reduction, per-prompt normalization, theoretical proof, 100 RL steps | No per-instance targeting, math only, length penalty on total output, small accuracy loss | **Closest conceptual predecessor.** CASSI extends with explicit stopping, separate model, agents |
| **Agent-RRM** (Cat 4) | Multi-faceted RM (think+critique+score), 12 benchmarks, SOTA ≤8B agents | Expensive GPT annotation, extra inference cost, no cost optimization | Complementary: critiques → stopper; cost-awareness → their RM. Shared benchmarks |
| **CSO** (Cat 4) | Verified critical steps, 16% steps supervision, 37% GAIA improvement, iterative refinement | Closed-source PRM, branch rollouts, no cost model, no stopping | Complementary: CSO for actions + CASSI for stopping. Shared GAIA benchmark |
| **CARL** (Cat 4) | Action-level credit assignment, 39.6% training samples, entropy-guided forking | Needs baseline capability, no cost model, no stopping, single domain | Most complementary. CARL + CASSI = better actions + timely stopping |
| **GiGPO** (Cat 4) | Anchor state grouping, step-level micro advantages, same memory as GRPO | Needs shared states, limited benchmarks, no cost focus | GiGPO for credit quality + CASSI for reward content. Fully complementary |
| **GRPO** (Cat 4) | Eliminates critic (half memory), group normalization, unified RL paradigm, process reward exploration | Process rewards inferior to outcome (paper finding), math only | **Algorithmic foundation.** CASSI shows process rewards CAN work when derived from oracle labels |
| **SelfBudgeter** (Cat 5) | 61% compression, self-estimation + adherence, cold-start + GRPO, dynamic α scheduling | Math only, one budget per question, fragile α, no heterogeneous costs | CASSI extends to: agents, per-step, heterogeneous costs, separate stopper |
| **BudgetThinker** (Cat 5) | Continuous budget awareness via control tokens, curriculum RL, +4.9% accuracy | Model-intrusive, math only, pipeline modification, single model burden | CASSI: modular stopper (no pipeline change), agent domains, separation of concerns |
| **Budget Guidance** (Cat 5) | Zero-shot cost control, Bayesian formulation, Gamma predictor, no fine-tuning | Single-turn only, per-token overhead, reasoning only, model doesn't learn | CASSI's RL training generalizes better to multi-turn agents. Gamma predictor is novel |
| **DeepResearcher** (Cat 6) | Real-world web RL, multi-agent, emergent behaviors, +28.9 over prompt engineering | No cost-awareness, no stopping, high training cost, multi-agent complexity | CASSI could serve as cost-saving layer on DeepResearcher agents |
| **ReTool** (Cat 6) | RL-induced strategic tool use, 2.7× faster convergence, emergent code self-correction | Math only, sandbox required, single tool, no cost model | Proves GRPO induces strategies → supports CASSI. Complementary: tool strategy + stopping |
| **Ares** (Cat 6) | Per-step effort routing, 52.7% token reduction, plug-and-play, GRPO training | Needs configurable effort levels, trial-and-error labeling, reasoning effort only | **Strongest architectural parallel.** Different decision type (effort vs stop) but same structure |
| **SeqRoute** (Cat 6) | Hindsight Budget Relabeling, 73.5% cost reduction, λ-sweep, delayed gratification | Binary actions, offline RL, model routing not stopping | HBR technique + λ-sweep are valuable ideas CASSI could adopt |
| **Don't Overthink It** (Cat 7) | Empirical proof: shorter chains 34.5% more accurate, short-m@k efficiency, SFT on short > long | Inference-only, math only, parallel overhead, no cost/quality tradeoff | Validates CASSI's premise: short = better. CASSI extends to training + agents |
| **When More is Less** (Cat 7) | **Inverted-U proof**, simplicity bias discovery, theoretical scaling laws, 40% optimal vs longest gap | Characterizes problem, doesn't solve it. No trainable mechanism. Step count not tokens | **Strongest theoretical basis for CASSI.** CASSI IS the solution to what they characterize |
| **CTA** (Cat 7) | Cost-uncertainty formalization, explicit priors change behavior, standard RL fails without priors | Small-scale only, simple actions, prior predictor needed | Validates explicit cost signals matter. CASSI provides scalable version |

---

## Grand Comparison Matrix (Top 12 Most Relevant Papers)

| Dimension | CASSI | AgentPRM | CaRT | Reason Eff. | L1 | Ares | CARL | CSO | Self-Budgeter | WhenMore Less | Don't Overthink | GRPO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Cost-aware stopping** | ✓ (learned) | ✗ | ✓ (binary) | Implicit | ✓ (user) | Implicit | ✗ | ✗ | Implicit | ✗ | ✓ (heuristic) | ✗ |
| **Process rewards** | ✓ (cost-aware) | ✓ (Q-val) | ✗ | ✗ | ✗ | ✗ | ✓ (tree) | ✓ (PRM) | ✗ | ✗ | ✗ | Attempted |
| **Two-model arch** | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Oracle/label cost** | **O(T)** | O(K×T²) | Counter-factual | N/A | N/A | Min-effort | N/A | Branch rollouts | Heuristic | Pre-computed | N/A | N/A |
| **Trains executor** | ✓ (GRPO) | ✓ (DPO) | ✗ | ✓ (PPO) | ✓ (GRPO) | ✗ | ✓ | ✓ (DPO) | ✓ (GRPO) | ✗ | ✗ (SFT) | ✓ (GRPO) |
| **λ cost tradeoff** | ✓ | ✗ | ✗ | ✓ (α) | ✓ (α) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Agent domains** | 6 benchmarks | 1 (ALFWorld) | 2 | Math only | Math only | 3 (TAU, Browse, WebArena) | 6 (QA) | 2 (GAIA, XBench) | Math only | Math/arithmetic | Math only | Math only |
| **Per-step adapt** | ✓ (step) | ✓ (step) | ✓ (step) | Implicit | ✓ (prompt) | ✓ (step) | ✓ (step) | ✓ (step) | ✗ (question) | ✗ (model) | ✗ (chain) | ✓ (step) |
| **Reported savings** | 20-40% | — | — | ~50% | ~50% | 52.7% | ~50% tol | — | 61% | 40% gap | 40% | — |

---

## CASSI's 5 Unique Contributions (Validated Against All 28 Papers)

### 1. O(T) Oracle-Guided Stopping Labels (Zero Extra Executions)
- **AgentPRM** needs O(K×T²) MC rollouts (160 extra trajectories for T=20, K=8)
- **CaRT** needs expensive counterfactual generation
- **CSO** needs branch rollouts with expert alternatives
- **Ares** needs trial-and-error testing of all effort levels per step
- **No other paper** achieves zero additional executions for stopping label computation
- CASSI's `t* = argmax_t [quality_t − λ × cumulative_cost]` is O(T) post-hoc from already-collected trajectories

### 2. Explicit λ-Weighted Cost-Quality Pareto Optimization
- **L1** and **Reason Efficiently** have length penalties but not cumulative cost modeling
- **AgentPRM** models Q-values without cost
- **BudgetThinker/TALE/SelfBudgeter** use raw tokens, not weighted cost
- **No paper** formulates stopping as `quality − λ×cumulative_cost` with a tunable cost-sensitivity parameter
- CASSI's λ provides continuous Pareto navigation (different λ → different cost-accuracy points)

### 3. Stopping Model as Cost-Aware Process Reward Model for Executor GRPO
- **AgentPRM** provides Q-value process rewards but without cost awareness
- **CARL** provides action-level advantages but without cost awareness
- **Agent-RRM** provides critique+score rewards but without cost awareness
- **CSO** provides verified step pairs but without cost awareness
- CASSI uniquely combines: (a) cost-awareness in rewards, (b) process-level granularity, (c) derived from oracle labels, (d) used to train the executor via GRPO

### 4. Dynamic Per-Instance Cost Adaptation Across Heterogeneous Agent Tasks
- **BATS/BAVT** adapt but are training-free (one domain each)
- **Reason Efficiently/L1** adapt but only on math
- **Ares** adapts per-step but only on reasoning effort (not full agent trajectories)
- **SelfBudgeter/BudgetThinker** adapt per-question but only on math
- CASSI adapts stopping based on observed progress across **6 diverse benchmarks** spanning web research, multi-hop QA, software engineering, and math

### 5. Small Stopping Model (0.5B-3B) Supervises Large Executors (7B-72B) with <3% Overhead
- **Ares** has 1.7B router controlling 20B agent (comparable)
- **CORL** has 7B controller (too large)
- **ReMA** uses same-size models for both agents (2× inference cost)
- **No other paper** demonstrates such a large size gap (up to 144×) with near-zero overhead

---

## Five Most Important Baselines for the CASSI Paper

| Rank | Paper | Why Essential | Expected Advantage |
|---|---|---|---|
| 1 | **AgentPRM** | Direct O(K×T²) vs O(T) comparison. Both train PRM + policy. CASSI must show comparable or better results at dramatically lower training cost | Training efficiency (RQ5), comparable stopping accuracy |
| 2 | **Reason Efficiently** | Closest RL-based cost reduction paper. Static length penalty vs CASSI's dynamic stopping. Must show CASSI beats static approach on mixed-difficulty tasks | Better Pareto frontier on heterogeneous tasks (RQ2) |
| 3 | **CaRT** | Only prior paper training a model specifically for termination decisions. Most directly comparable objective | Better stopping accuracy, cost-aware adaptation (RQ4, RQ3) |
| 4 | **Ares** | Most similar architecture (small router + large agent). Contemporaneous work with different objective | CASSI's oracle labels are cheaper + handles full agent trajectories |
| 5 | **SelfBudgeter** | Most similar training approach (self-budgeting via GRPO). Trajectory-level budget vs CASSI's step-level | Per-step granularity supports better adaptation (RQ3) |

---

## Key Citations for CASSI Paper by Section

| Paper Section | Papers to Cite | Purpose |
|---|---|---|
| **Overthinking problem** | When More is Less, Don't Overthink It, s1 | Establish that overthinking is real (inverted-U) and costly |
| **Existing solutions & gaps** | BATS, INTENT (heuristics); s1, L1 (static budgets); CaRT (binary stopping); AgentPRM (O(K×T²) cost) | Show limitations of current approaches |
| **Monitor-Executor architecture** | MGV (theoretical), ReMA (RL-based), Ares (router-based) | Justify two-model design choice |
| **RL training methodology** | GRPO (algorithm foundation), Reason Efficiently (length penalty RL), AgentPRM (PRM training pattern) | Frame CASSI's training approach in context |
| **Process rewards** | AgentPRM, Agent-RRM, CSO, CARL | Position CASSI's stopping model as a novel cost-aware PRM |

---

## Recommendations for Combining Complementary Ideas

| Combination | How It Works | Benefit |
|---|---|---|
| CASSI + CARL | CARL's entropy-guided action-level advantages + CASSI's cost-aware stopper rewards in executor GRPO | Better credit assignment + cost-aware stopping |
| CASSI + CSO | CSO's verified critical steps for action correction + CASSI's stopping for termination | Both action quality AND timely stopping |
| CASSI + Ares | Ares's effort-level routing per step + CASSI's stop/continue decisions | "How hard" + "when done" for each step |
| CASSI + SeqRoute | SeqRoute's Hindsight Budget Relabeling to augment CASSI's trajectory dataset + λ-sweep for zero-shot cost adaptation | More training data + flexible deployment |
| CASSI + Agent-RRM | Agent-RRM's textual critiques as additional input to CASSI's stopper | Richer stopping decisions informed by explicit flaw detection |
| CASSI + INTENT | INTENT's geometric cost calibration (expected_cost = nominal / success_probability) to enrich CASSI's oracle labels | More accurate cost modeling for tool calls |

---

## Summary

CASSI occupies a unique position in the literature. Of the 28 papers analyzed:

- **0 papers** combine cost-aware process rewards from oracle labels with executor GRPO training
- **0 papers** achieve O(T) post-hoc stopping labels with zero extra policy executions
- **0 papers** evaluate cost-aware stopping across web search, QA, software engineering, and math
- **1 paper** (Ares) shares the small-router-large-agent architecture but for reasoning effort, not stopping
- **3 papers** (CaRT, Reason Efficiently, L1) directly address cost/efficiency in agent/reasoning but with limitations CASSI addresses

**CASSI's core differentiation:** Instead of (a) expensive MC rollouts (AgentPRM), (b) expensive counterfactuals (CaRT), (c) uniform penalties (Reason Efficiently/L1), or (d) training-free heuristics (BATS/DEER), CASSI computes mathematically optimal stopping labels from trajectories the agent already produces — O(T), zero additional cost — then trains both a stopping model AND the executor with these cost-aware signals.

---

*End of Competitor Analysis. 28 papers analyzed across 4 files.*
*Part 1: competitor_analysis.md (Categories 1-2) | Part 2: competitor_analysis_part2.md (Categories 3-4) | Part 3: competitor_analysis_part3.md (Categories 5-7) | Part 4: competitor_analysis_part4.md (Synthesis)*
