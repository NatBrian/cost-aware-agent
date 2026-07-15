# CASSI Novelty Refinement Plan

> **Current Score:** 3.0/5 (borderline ICLR)
> **Target Score:** 3.5–4.0/5 (solid accept)
> **Core Problem:** Individual components are known. The combination is novel but not framed strongly enough. Without refinement, aggressive reviewers will call this "HER + AgentPRM + Ares applied to stopping."

---

## The Root Cause: What's Actually Weak

The novelty assessment revealed three structural problems:

| Problem | Why It Hurts | Current Paper Plan |
|---|---|---|
| **1. Contribution framed as efficiency, not paradigm** | "Faster version of X" rarely gets ICLR acceptance. The paper currently sells "O(T) instead of O(K×T²)" as the main contribution | Section 7 is titled "Why Not Just Use AgentPRM with a Cost Term?" — the paper defines itself against AgentPRM |
| **2. No theoretical justification for two-model design** | Without theory, the two-model architecture looks like an arbitrary engineering choice. Reviewer: "Why not just add a cost penalty to the executor? SelfBudgeter does it in one model." | Section 4.3 says "practical design choice, not a contribution" — this is honest but weakens the paper |
| **3. The process-reward bridge buried** | Your irreducible novelty is the closed-loop cycle, but it's described as "Stage 3" of training, not as the paper's core thesis | The cycle appears only in Section 9.4, framed as implementation detail |

---

## Refinement #1: Reframe the Contribution Hierarchy

**Current order (paper_plan.md):**
1. Oracle-guided stopping labels (O(T) vs O(K×T²)) ← positioned as #1
2. Dynamic per-instance cost adaptation beats static penalties ← #2
3. Small stopping model supervises large executors ← #3

**New order:**
1. **The self-reinforcing cost-awareness training cycle** ← NEW #1
2. **Why two models are necessary** (not just a design choice) ← NEW #2 
3. Oracle-guided stopping labels (O(T) vs O(K×T²)) ← demoted to #3
4. Dynamic per-instance cost adaptation ← #4
5. Small model supervises large executor ← #5

**Why this matters:** Contribution #1 shifts from "we're faster than AgentPRM" to "we introduce a training paradigm no one has tried." The aggressive reviewer's "HER applied to stopping" argument now must contend with the full cycle, not just the labeling step.

**How to write it:**
> "We propose a new training paradigm for cost-aware LLM agents: a self-reinforcing cycle where an oracle-derived stopping objective labels completed trajectories, trains a lightweight stopping model, and the stopping model's cost-aware value estimates feed back as process rewards to train the executor — which then produces better trajectories, improving the oracle, and so on."

---

## Refinement #2: Add Theoretical Justification for Two Models

**The problem:** You currently say the two-model design is "a practical choice, not a contribution." A reviewer will ask: "If it's not a contribution, why not drop it? SelfBudgeter works in one model. Prove you need two."

**The fix:** Add a theoretical argument. Here's the claim you can make:

> **Proposition (informal):** A single model trained to both execute tasks AND evaluate its own cost-quality tradeoff faces a representation conflict — the features that predict what action to take next (execution) differ from the features that predict whether continuing is worth its cost (stopping). A separate stopping model avoids this conflict, leading to better stopping accuracy and better execution quality.

**How to support it empirically (new ablation):**
```
Compare:
(A) Single model trained with cost-penalized GRPO (like Reason Efficiently applied to agents)
(B) Single model trained with oracle stopping labels + regular task objective (multi-task)
(C) CASSI: separate stopper + executor with process-reward bridge

Prediction: (C) > (B) > (A), and the gap between (C) and (B) proves the two-model design matters.
```

**Why this strengthens novelty:** It transforms "we chose two models" (engineering) into "two models are necessary because of a fundamental representation conflict" (science).

---

## Refinement #3: Make the Process-Reward Bridge the Core Thesis

**Current structure:**
```
Section 4: CASSI Architecture (describes stopper, executor, budget state)
Section 5: Training Method (describes 3 phases)
```

**New structure:**
```
Section 4: The Cost-Aware Training Cycle (NEW)
  - 4.1: Why single-model cost penalties fail (static, instance-blind, representation conflict)
  - 4.2: The closed-loop cycle: oracle → stopper → process rewards → executor → repeat
  - 4.3: Why this cycle converges to better cost-quality Pareto frontiers
  - 4.4: The oracle objective: t* = argmax[quality − λ×cumulative_cost]
  
Section 5: Architecture (demoted, shorter)
  - 5.1: Stopping model design
  - 5.2: Executor design
  - 5.3: Budget state and λ adaptation

Section 6: Training Method
  - 6.1: Phase 1: Trajectory collection
  - 6.2: Phase 2: Oracle labeling + stopper training
  - 6.3: Phase 3: Executor training with cost-aware process rewards
```

**Why this matters:** The cycle becomes the headline. The oracle labeling becomes a component OF the cycle, not the contribution itself. This directly counters the "you just applied HER to stopping" criticism — HER is just one step in a larger system.

---

## Refinement #4: Add a "Negative Result" That Proves the Cycle Matters

This is a powerful rhetorical move. Show that a simpler version FAILS:

> **Expected negative result:** "If we train the stopping model on oracle labels but do NOT use its outputs as process rewards for executor training (i.e., the stopper only controls inference-time stopping), the executor does not learn cost-aware behavior — it continues to overthink. The process-reward bridge is necessary, not optional."

This would be **Ablation #4** from the novelty assessment (stopper as controller-only vs. stopper as process reward model). Make it prominent.

---

## Refinement #5: New Baselines to Add

Based on the novelty assessment, add these to Section 10.4:

| New Baseline | Why It Must Be There | Priority |
|---|---|---|
| **CaRT + cost + GRPO** | Already in plan. Elevate from P0 to **PRIMARY** comparison. Your entire novelty defense rests on beating this | P0 |
| **Single-model GRPO + cost penalty** (Reason Efficiently-style, applied to agents) | Proves the two-model design is necessary, not just a choice | P0 |
| **CASSI w/o process-reward bridge** (stopper for inference only, no executor RL) | Proves the bridge matters — this is your irreducible novelty | P0 |
| **BudgetThinker** | Closest single-model budget-aware RL approach | P1 |
| **Ares-style discrete effort router** | Same architecture, different oracle — proves your CONTINUOUS oracle is better than DISCRETE trial-and-error | P2 |

---

## Refinement #6: Strengthen Related Work Positioning

Current Section 5 organizes papers into 6 categories. That's fine, but it doesn't preempt the reviewer's argument. Add a dedicated subsection:

### New: Section 5.7 — "Why Existing Approaches Cannot Be Easily Extended to Cost-Aware Agent Stopping"

| Approach | Representative Paper | Why It Cannot Solve Our Problem |
|---|---|---|
| **Static length penalties** | Reason Efficiently, L1 | Penalty is instance-blind. Easy and hard tasks get same α. Cannot adapt mid-trajectory when progress stalls |
| **Self-termination** | CaRT | Trains only the stopper. No feedback to executor. Terminates trajectories but doesn't train the executor to produce better trajectories |
| **Quality-only PRMs** | AgentPRM | No cost awareness. Evaluates step quality but not whether that quality is worth the cost |
| **Training-free heuristics** | BATS, DEER | No learning. Same rules for every situation. Cannot improve from experience |
| **Hindsight relabeling for routing** | SeqRoute | Relabels with budget constraints for model selection, not with quality-cost optimality for within-trajectory stopping |
| **Small-controller architectures** | Ares | Controller predicts discrete effort levels via trial-and-error, not continuous stopping value. Does not train executor |
| **Single-model budget RL** | BudgetThinker, SelfBudgeter | Single model carries both reasoning and budget tracking — representation conflict. Cannot specialize for stopping decisions |

**This table is critical.** It tells the reviewer: "Yes, we know about all these papers. Here's exactly why none of them solve our problem, and why extending them wouldn't work." This preempts the "just combine X and Y" criticism.

---

## Refinement #7: Add Formal Properties of the Oracle

The novelty assessment noted "no fundamental theoretical contribution." Add a simple formal statement:

> **Property 1 (Monotonicity):** If the executor's trajectory quality is non-decreasing and cost is strictly increasing, then t* exists and is unique for any λ > 0.

> **Property 2 (λ-Sensitivity):** t*(λ₁) ≤ t*(λ₂) for λ₁ ≥ λ₂. As cost sensitivity increases, the optimal stopping point moves earlier (or stays the same). This gives us a principled way to navigate the cost-quality Pareto frontier.

> **Property 3 (Oracle Convergence):** As the executor improves during training (standard GRPO convergence), the oracle labels computed from its trajectories approach the true optimal stopping points for the improved policy.

These are simple, provable, and give the paper theoretical grounding. They also distinguish CASSI from purely empirical papers like Ares and BudgetThinker.

---

## Refinement #8: The "Killer Figure"

Every ICLR paper needs one figure that tells the whole story. Currently the plan has an architecture diagram (Section 8.1). Add this:

```
                    THE SELF-REINFORCING CYCLE
                    
    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │  ① Executor runs tasks, generates trajectories       │
    │     │                                                │
    │     ▼                                                │
    │  ② Oracle computes t* = argmax[quality − λ×cost]    │
    │     │  (O(T) post-hoc, zero extra executions)        │
    │     ▼                                                │
    │  ③ Stopping model trains on oracle labels            │
    │     │  (SFT: "copy correct stops" → GRPO: "practice")│
    │     ▼                                                │
    │  ④ Stopper provides Δ(s_t) as process rewards        │
    │     │  to executor during GRPO training              │
    │     ▼                                                │
    │  ⑤ Executor learns cost-aware behavior               │
    │     │  (better actions + knows when they're done)    │
    │     │                                                │
    │     └────────────► back to ① (better trajectories)   │
    │                                                      │
    └──────────────────────────────────────────────────────┘
    
    KEY INSIGHT: This cycle does not exist in any prior work.
    - AgentPRM: has ②→③→④ but no stopping, no cost, O(K×T²)
    - CaRT: has ①→②→③ but no ④→⑤ (no executor training)
    - Ares: has ①→②→③ but no ④→⑤ (no executor training, discrete labeling)
    - Reason Efficiently: has ⑤ only (cost penalty in single model, no ①→④)
```

---

## Summary: What Changes and What Stays

| Section | Current | Change |
|---|---|---|
| **Contributions** | #1 = O(T) oracle labels | #1 = Self-reinforcing cost-awareness cycle (the oracle is a component) |
| **Why two models** | "Design choice, not contribution" | Add theoretical argument + ablation proving necessity |
| **Architecture** | Section 4, before training | Demote. Make "The Cycle" the new Section 4. Architecture becomes Section 5 |
| **Related Work** | 6 categories | Add Section 5.7: "Why existing approaches cannot be easily extended" table |
| **Baselines** | 7 baselines | Add 3: single-model cost-penalty GRPO, CASSI w/o process-reward bridge, BudgetThinker |
| **Ablations** | 6 ablations | Add: single-model vs two-model, stopper-as-controller-only vs stopper-as-PRM |
| **Theory** | None | Add 3 formal properties of the oracle |
| **Figures** | Architecture diagram | Add "self-reinforcing cycle" figure as Figure 1 |
| **Title** | "Scalable Cost-Aware Agent Training via Oracle-Guided Stopping Rewards" | Consider: "Learning to Stop: Self-Reinforcing Cost-Aware Training for LLM Agents" (shorter, emphasizes the cycle, not the oracle) |

After refinements, the paper no longer reads as "we made AgentPRM faster." It reads as "we discovered a training paradigm that creates a virtuous cycle between stopping decisions and execution quality — something no prior approach achieves because they operate on only one side of the cycle."

---

*Refinement plan based on novelty assessment by 3 independent subagent reviewers.*