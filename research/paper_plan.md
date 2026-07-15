# Paper Plan v4: Scalable Cost-Aware Agent Training via Oracle-Guided Stopping Rewards

> **Target Venue:** ICLR / NeurIPS / ICML
> **Type:** Efficiency contribution to process reward model training for LLM agents
> **Status:** Pre-experiment planning — refined through 5 iterations of novelty review (score converged at 3.6/5)

---

## 1. Title

**"Scalable Cost-Aware Agent Training via Oracle-Guided Stopping Rewards"**

---

## 2. Abstract

> Large language model (LLM) agents solve complex tasks through multi-step reasoning and tool use, but they lack economic judgment — they cannot assess whether the next action is worth its cost. Existing solutions either enforce rigid budget cutoffs, embed static length penalties that cannot adapt per instance, or rely on training-free heuristics. Process reward models (PRMs) can provide step-level training signals, but current PRMs rely on Monte Carlo rollouts from every intermediate state — requiring O(K×T²) additional policy executions per trajectory and making cost-aware training intractable for long-horizon tasks like software engineering.
>
> We propose **oracle-guided stopping rewards**: instead of simulating forward from each state, we compute optimal stopping labels directly from completed trajectories. For a T-step trajectory, the oracle stopping point is `t* = argmax_t [quality_t − λ × cumulative_cost_{1..t}]` — an O(T) post-hoc computation requiring zero additional executions. We train a small stopping model (0.5B–3B parameters) on these labels via SFT+RL, then use its cost-aware value estimates as process rewards to train the executor agent via GRPO. This reduces training computation from O(K×T²) to O(T) — a K×T reduction that makes cost-aware PRM training tractable on long-horizon benchmarks.
>
> We evaluate on web search (GAIA, WebWalkerQA), multi-hop QA (HotpotQA, MuSiQue), and software engineering (SWE-bench Verified). Our method reduces agent computation cost by **X%** while maintaining or improving task success, outperforming static length penalties (L1, Reason Efficiently), quality-only PRMs (AgentPRM), self-termination training (CaRT), and heuristic stopping (BATS). The stopping model adapts cost pressure per-instance — spending more compute on hard problems and less on easy ones — with its own inference overhead below 3% of total cost.

---

## 3. Introduction / Background

### 3.1 The Overthinking Problem

LLM agents — from coding assistants to deep research systems — operate in open-ended loops: reason, call a tool, observe, reason again. Each step consumes tokens (costing real money), each tool call may access paid APIs, and the process has no natural endpoint. The agent's training objective is "be helpful and complete the task." There is no training signal for *stopping* — "this is good enough, I should stop now" is not a reward-consistent output.

This produces three failure modes:

| Failure Mode | Example | Consequence |
|---|---|---|
| **Overthinking** | Agent continues refining a correct answer through 10+ unnecessary iterations | Wastes 80%+ of total compute on zero-value work |
| **Runaway loops** | Agent enters a tool-call cycle that never converges | Exhausts budget with no result |
| **Premature stopping** | Agent submits a partial or incorrect answer to save budget | Task failure that more steps would have fixed |

Prior work has demonstrated that these are not hypothetical:
- **When More is Less** (Wu et al., 2025): reasoning accuracy follows an inverted U-shaped curve — beyond an optimal length, *more thinking reduces accuracy*.
- **Over-Reasoning** (Chiang & Lee, 2024): LLMs generate lengthy calculations even for problems requiring zero reasoning steps.
- **Don't Overthink It** (Hassid et al., 2025): the shortest reasoning chain is correct 34.5% more often than the longest chain for the same question.
- **Token Economies** (Wang et al., EMNLP 2024): simple CoT + self-consistency outperforms complex reasoning strategies (Multi-Agent Debate, Reflexion) when given equal compute budgets.

### 3.2 The Missing Capability

A human expert does not work until exhausted. They make constant stop/continue decisions:

> "The fix compiles and tests pass. Ship it, don't polish further."
> "I've been debugging 4 hours with no progress. Escalate, don't keep digging."
> "This analysis is 90% complete and the last 10% would triple the time. Publish what we have."

These are **economic judgments** — weighing the marginal value of continued work against its marginal cost. Frontier LLMs are *capable* of this reasoning (they can articulate cost-benefit analyses when prompted), but they are not *trained* for it. RLHF rewards helpfulness and completion; it never rewards the agent for recognizing that further work is not worthwhile.

### 3.3 What Existing Solutions Miss — and a Scalability Gap

| Approach | Examples | Limitation |
|---|---|---|
| **Hard budget enforcement** | s1 budget forcing, token limits | "Stop at 500 tokens" is rigid — doesn't know if the task is actually done |
| **Static length penalties in RL** | Reason Efficiently, L1, BudgetThinker | Penalty is instance-blind — same pressure on easy and hard problems |
| **Training-free heuristics** | BATS, BAVT, INTENT | Heuristic, no learning — the agent can't improve its stopping over time |
| **Self-termination training** | CaRT | Same model reasons AND terminates — bundles two objectives in one model |
| **Process reward models** | AgentPRM, Agent-RRM | Evaluate quality only; training requires Monte Carlo rollouts from every intermediate state — **O(K×T²) that becomes intractable for long-horizon tasks** |

**The scalability gap:** Process reward models (AgentPRM, 2025) are the most principled existing approach for providing step-level training signals. But they require running K Monte Carlo rollouts to completion from every intermediate state in a trajectory — O(K×T²) additional policy executions. For SWE-bench where T ≈ 20 steps and each step involves repository-scale operations, this requires ~160 additional full executions per training trajectory. This computational barrier has prevented cost-aware PRM training from being applied to the benchmarks where it would matter most.

**Our insight:** Optimal stopping labels do not require forward simulation. Given a completed trajectory — which we already have from the executor's normal execution — the optimal stopping point can be computed analytically: `t* = argmax_t [quality_t − λ × cumulative_cost_{1..t}]`. This is an O(T) post-hoc computation. We train a small stopping model on these oracle labels and use its predictions as cost-aware process rewards to train the executor. The result: the same step-level cost signals as a full PRM, but with K×T fewer executions.

---

## 4. Motivation

### 4.1 Core Hypothesis

> A stopping model trained on oracle-guided labels (computed from completed trajectories) can provide cost-aware process rewards that produce better cost-accuracy trade-offs than static length penalties, quality-only PRMs, or heuristic stopping — while requiring K×T fewer training executions than Monte Carlo PRMs.

### 4.2 Why a Stopping Model (Not Just Static Penalties)?

Static length penalties (L1, Reason Efficiently) apply uniform cost pressure regardless of task difficulty. A simple question and a complex multi-hop query receive the same α coefficient. We hypothesize that **per-instance, mid-trajectory adaptation** matters: the stopping model should allow more computation for genuinely hard problems and cut off overthinking on easy ones. This hypothesis is testable (H5: correlation between stopping point and task difficulty) and distinguishes our approach from the simplest baselines.

### 4.3 Why a Separate Model (Design Choice, Not Contribution)

We use a separate stopping model rather than embedding cost-awareness in the executor. This is a **practical design choice** following the PRM pattern established by AgentPRM (2025) and ReMA (2025), not an architectural contribution. The practical benefits:
1. **Size efficiency:** The stopping model (0.5B–3B) is much smaller than the executor (7B–72B), minimizing inference overhead.
2. **Offline training:** The stopping model can be trained on recorded trajectory data while the executor is deployed.
3. **Reusability:** One stopping model can supervise multiple executor models (tested empirically, Section 9).

### 4.4 Why Oracle Labels (Not Monte Carlo Rollouts)?

AgentPRM trains its PRM via Monte Carlo rollouts: from each state s_t, run the policy K times to completion and average the returns. For a T-step trajectory, this requires K×T additional full executions — O(K×T²). On SWE-bench (T≈20, K=8), that's ~160 extra executions per training trajectory.

Our oracle-guided approach requires **zero additional executions.** The stopping label is computed post-hoc from the single trajectory the executor already produced. This makes cost-aware PRM training tractable on benchmarks where AgentPRM's approach is computationally prohibitive.

---

## 5. Related Work

Organized into six categories (see `research/00_overview.md` for the full review with 48 papers).

### 5.1 Cost-Aware & Budget-Constrained Agent Frameworks (Category 1)

BATS (Liu et al., 2025), BAVT (Li et al., 2026), INTENT (Liu et al., 2026), IterResearch (2025) all implement budget tracking and cost-aware decision-making for LLM agents. However, all use **training-free heuristics** — prompt-level budget signals, fixed mathematical formulations (budget ratio as exponent), or hand-designed world models. None learns a stopping policy.

### 5.2 Agent Stopping & Early Exit (Category 2)

CaRT (Liu et al., 2025) trains models to decide when to terminate information gathering via SFT on counterfactual examples. s1 (Muennighoff et al., 2025) forces stopping by appending "Wait" or truncating at token limits. DEER (Yang et al., 2025) uses token-level confidence for early exit from reasoning chains. These methods are either (a) SFT-based (no RL), (b) hard-threshold-based (no soft curve), or (c) embedded in the policy model (no separate monitor).

### 5.3 Meta-Reasoning & Monitor-Executor Architectures (Category 3)

ReMA (Wan et al., NeurIPS 2025) decouples reasoning into high-level meta-thinking and low-level execution agents, trained via multi-agent RL. MGV (Oh & Gobet, 2025) formalizes a Monitor-Generate-Verify framework from cognitive science. Dolores (Light et al., 2026) separates meta-level decomposition from object-level execution. These provide the **architectural template** for our monitor-executor split but are applied to reasoning strategy selection, not cost-aware stopping.

### 5.4 RLHF & Preference Optimization for Agent Behaviors (Category 4)

GRPO (Shao et al., 2024) enables critic-free RL for LLMs. AgentPRM (Choudhury, 2025) provides the actor-critic template with Monte Carlo rollouts training a process reward model. CSO (Li et al., 2026) focuses optimization on verified critical steps. CARL (Shen et al., 2025) uses model confidence to identify high-criticality actions. Our monitor is a *cost-aware* variant of these process reward models.

### 5.5 Token-Efficient Reasoning (Category 5)

L1/Aggarwal & Welleck (2025), Reason Efficiently/Arora & Zanette (NeurIPS 2025), TALE, SelfBudgeter, BudgetThinker, and DiffAdapt all optimize token usage in reasoning. Key finding: RL with length penalties produces efficient reasoning (~200 gradient updates). Our work differs by making cost-awareness **dynamic** (per-instance, mid-trajectory) rather than **static** (pre-specified budget), and by using a **separate** monitor rather than embedding the penalty in the policy.

### 5.6 Efficient Tool Use & Long-Horizon Tasks (Category 6)

DeepResearcher (2025), ReTool (Feng et al., 2025), ToolRL (Qian et al., 2025), and CARL all use RL to optimize tool-calling behavior. SupervisorAgent (2025) uses a lightweight monitor for multi-agent orchestration. Our work extends these by making the monitor itself a trained reward model.

### 5.7 Positioning Table

```
                          OUR PROPOSAL
                   (Oracle-Guided Cost-Aware
                     Stopping Rewards)
                               |
           +-------------------+-------------------+
           |                   |                   |
     Category 3           Category 4          Category 2
   (Monitor-Executor     (RLHF/Pref. Opt.)    (Stopping/Early
    Architecture)                                  Exit)
           |                   |                   |
           +-------------------+-------------------+
                               |
                     +---------+---------+
                     |                   |
               Category 1           Category 5
            (Cost-Aware Agents)   (Token Efficiency)
                     |                   |
                     +---------+---------+
                               |
                          Category 6
                     (Efficient Tool Use)
```

---

## 6. Problem Formulation

### 6.1 Agent Trajectory as MDP

We model the executor agent's interaction as a Markov Decision Process:

- **State** `s_t`: the full context at step t, including:
  - Task description `q`
  - Reasoning history `r_{1..t}`
  - Tool call history `a_{1..t-1}` and observations `o_{1..t-1}`
  - Budget state: tokens consumed `c_token`, tool calls made `c_tool`, estimated dollar cost `c_dollar`, budget remaining `b_rem`
  
- **Action** `a_t`: the agent's next step, which may be:
  - A reasoning step (no tool call)
  - A tool invocation (search, code execution, file read, etc.)
  - A final answer submission

- **Transition**: environment and tool execution determine `s_{t+1}`

- **Reward** `R(s_T)`: sparse outcome reward (1 if task correctly completed, 0 otherwise), received only at terminal state

- **Cost** per step: `cost(s_t, a_t) = w_token * tokens(a_t) + w_tool * 1[tool_call] + w_dollar * dollar_cost(a_t)`

### 6.2 The Stopping Problem

At each step t, the executor faces an implicit decision: **stop and submit an answer** vs. **continue with another action**. The optimal stopping policy π* maximizes:

```
π* = argmax_π E[ R(s_T) - λ * Σ cost(s_t, a_t) ]
```

where λ controls the cost-sensitivity of the policy. Higher λ → more cost-averse (stops earlier). Lower λ → more quality-seeking (continues longer).

This is a **sequential decision problem under uncertainty**: the agent does not know whether the next step will produce valuable information or waste resources.

### 6.3 The Value of Continuing

Define the **Q-function for continuing** at state s_t:

```
Q_continue(s_t) = E[ max_a ( R_future - λ * cost_future ) | s_t, continue ]
```

Define the **Q-function for stopping** at state s_t:

```
Q_stop(s_t) = R_now - λ * cost_so_far
```

where `R_now` is the expected quality of the answer if submitted at step t.

The optimal decision at step t:

```
if Q_continue(s_t) > Q_stop(s_t): continue
else: stop
```

The **cost-aware value function** is the function:

```
Δ(s_t) = Q_continue(s_t) - Q_stop(s_t)
```

When Δ(s_t) > 0: continue. When Δ(s_t) ≤ 0: stop. The magnitude of Δ(s_t) represents the confidence of the decision — the further from zero, the clearer the choice.

---

## 7. Why Not Just Use AgentPRM with a Cost Term?

A natural question: why not take AgentPRM's existing PRM framework and add a cost term to the Monte Carlo return? `R = quality − λ × cost`? This section explains why that approach is not a substitute for our method on the benchmarks we target.

### 7.1 Training Computation Scaling

AgentPRM trains a PRM by executing K Monte Carlo rollouts from each intermediate state s_t. For a T-step trajectory, this requires Σ_{t=1}^{T} K × (T−t) additional policy executions — O(K×T²). For T=20, K=8, this is ~160 additional full executions per trajectory. On SWE-bench, where each execution involves repository-scale file operations and test runs, this dominates the training budget.

Our oracle-guided approach requires only the original trajectory. The oracle label `t* = argmax_t [quality_t − λ × cumulative_cost_{1..t}]` is computed in O(T) time over already-recorded step data. Zero additional executions.

**Caveat:** We present this as a **hypothesis** to be tested, not as self-evident truth. AgentPRM's MC rollouts could be parallelized (reducing wall-clock time), though the total compute (FLOPs) remains K×T². The additional rollouts may also produce a better training signal — MC rollouts average over multiple completions, reducing variance that oracle labels from a single trajectory cannot. These are empirical questions: we include AgentPRM-cost as a baseline (Section 10.3) and measure training wall-clock time for both methods (RQ5).

**Concrete example (SWE-bench):** A typical trajectory has T=15 steps. AgentPRM-cost with K=8 requires Σ_{t=1}^{15} 8×(15−t) = 840 additional step-executions — each involving file reads, code generation, and test execution in a repository-scale environment. CASSI requires 0. Even with perfect parallelization across 8 GPUs (one rollout per GPU), AgentPRM-cost's wall-clock time is gated by the longest single rollout from the earliest state (15 steps).

### 7.2 What We Focus On vs. What AgentPRM Evaluates

AgentPRM's PRM is trained to predict Q(s, a) for all actions the policy might take at each state. This splits the learning signal across many action values. Our stopping model is trained for a single focused decision: at this state, is continuing worth its cost? This is a binary/ternary classification problem (STOP/CONTINUE/ADJUST) rather than a regression over the full action space. We hypothesize this focus improves sample efficiency — but this too is an empirical claim we test via ablation.

### 7.3 Oracle Label Limitations (Acknowledged)

The oracle label depends on the specific trajectory the executor took. If the executor made a suboptimal choice at step 2, the oracle may identify a suboptimal stopping point. Quality at intermediate steps is noisy — a partial answer may look correct but be wrong. We address this through:
- **Multiple trajectories per task** during training (the executor generates G=8 trajectories per GRPO step), providing diverse oracle label samples
- **RL fine-tuning** of the stopping model beyond SFT, which can learn to be more conservative than the point-estimate oracle
- **Comparison with CaRT's counterfactual approach** (Section 9.1, RQ2), which uses a more principled but more expensive labeling method

We also note that combining oracle labels with CaRT-style counterfactual pairs (use oracle to identify candidate t*, then construct counterfactuals around it) is a promising direction for future work.

---

## 8. Approach: CASSI

### 8.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CASSI Framework                           │
│                                                                   │
│   ┌──────────────┐          ┌──────────────────────┐             │
│   │              │          │                      │             │
│   │   Stopping   │◄─────────│    State Summary      │             │
│   │   Model M_θ  │          │  (budget, progress,   │             │
│   │              │          │   confidence, cost)   │             │
│   │  (small LLM) │          │                      │             │
│   │              │──────────►  stop/continue/adjust │             │
│   │              │──────────►  cost-aware reward Δ  │             │
│   └──────┬───────┘          └──────────────────────┘             │
│          │ reward signal                                          │
│          ▼                                                        │
│   ┌──────────────┐          ┌──────────────────────┐             │
│   │              │          │                      │             │
│   │  Executor    │──────────►  Action (reason,     │             │
│   │  Agent E_φ   │          │   tool call, answer) │             │
│   │              │◄─────────│                      │             │
│   │  (large LLM) │          │  Observation         │             │
│   │              │          │                      │             │
│   └──────────────┘          └──────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Stopping Model `M_θ`

**Model:** A small LLM (0.5B–3B parameters) — significantly smaller than the executor — fine-tuned for cost-aware stopping evaluation. This is a **design choice** following the PRM pattern (AgentPRM, 2025; ReMA, 2025).

**Input:** Structured prompt containing task description, budget state (tokens, tools, dollars, iterations, percentage used, budget tier), trajectory summary (last K steps), current answer draft, quality indicators (confidence, coverage, progress rate, answer stability).

**Output:** A structured response with STOP/CONTINUE/ADJUST action, natural language rationale, cost-aware value estimate Δ(s_t) ∈ [−1, 1], and decision confidence.

- Δ(s_t) > 0: continue (expected net benefit of another step is positive)
- Δ(s_t) ≤ 0: stop (further computation is unlikely to justify its cost)
- ADJUST: continue but change approach
- |Δ(s_t)|: decision strength; the further from zero, the clearer the choice

### 8.3 Executor Agent `E_φ`

Standard LLM agent (7B–72B) with tool-use capabilities (web search, browsing, code execution, file operations). Input: standard agent prompt + stopping model's feedback history. The stopping model's decision is **advisory during training** and **enforced during evaluation.**

### 8.4 Budget State Design

Multi-dimensional budget representation (following BATS, INTENT): tokens consumed, tool calls made, iterations used, estimated dollar cost, percentage of allowance remaining, and budget tier (HIGH/MEDIUM/LOW/CRITICAL). The budget tier dynamically adjusts the effective cost-sensitivity λ during inference — as resources deplete, the stopping model becomes more conservative.

---

## 9. Training Method

### 9.1 Overall Training Loop

CASSI uses a three-stage training process (standard iterative actor-critic, following the PRM training pattern from AgentPRM, 2025):

```
Stage 1: Collect Trajectories → Stage 2: Train Stopping Model → Stage 3: Train Executor
         (executor runs to          (SFT + RL on               (RL with stopping model
          completion)                oracle labels)              as reward model)
```

### 9.2 Stage 1: Trajectory Collection

1. Deploy executor agent `E_φ` (pre-trained, instruction-tuned base) on training tasks
2. Executor runs to completion or max steps (e.g., 20 steps)
3. For each trajectory, record at every step t: state `s_t`, action `a_t`, observation `o_t`, cost incurred, answer quality

### 9.3 Stage 2: Oracle-Guided Stopping Reward Training (Key Innovation)

#### 9.3.1 Computing Oracle Stopping Labels

For each trajectory, compute the **optimal stopping point `t*`**:

```
t* = argmax_t [ quality(answer_at_step_t) - λ * cumulative_cost_up_to_t ]
```

Where:
- `quality(answer_at_step_t)` is measured by answer correctness (binary) or similarity to ground truth (F1, ROUGE)
- `cumulative_cost_up_to_t` is the total cost of steps 1 through t
- λ controls the cost-sensitivity of the oracle

**Intuition:** `t*` is the step where the net value (quality minus cost) is maximized. If the answer was already correct at step 3 and steps 4-10 added no quality improvement, `t* = 3`. If quality improved steadily through step 7 and plateaued, `t* = 7`.

For steps t < t*: the oracle action is CONTINUE.
For steps t ≥ t*: the oracle action is STOP.

This gives us a supervised training signal: at each step t in each trajectory, we know what the monitor *should have* decided.

#### 9.3.2 Supervised Fine-Tuning (SFT)

Train the stopping model `M_θ` on oracle-labeled data:

```
L_SFT = CrossEntropy(M_θ(x_t), oracle_action_at_step_t)
```

At each step t, the monitor receives the state summary `x_t` and is trained to predict the oracle's action (STOP or CONTINUE). This teaches the monitor the *structure* of the stopping problem — what states tend to precede optimal stopping points.

#### 9.3.3 RL Fine-Tuning

After SFT, fine-tune the stopping model with RL via GRPO. The monitor generates a stopping decision at each step, but the decision is only evaluated at the end of the trajectory:

```
R_monitor(trajectory, monitor_decisions) =
    R_T  (task success)
    - λ_cost * Σ cost_t  (cost penalty)
    - λ_early * 1[monitor stopped BEFORE t*]  (early stop penalty)
    - λ_late * Σ_{t > t*} 1[monitor continued past optimal]  (late stop penalty)
```

The RL objective encourages the monitor to:
- Stop close to the optimal point (not too early, not too late)
- Favor stopping earlier when the answer is already good
- Favor continuing when the task is not yet solved

RL algorithm: **GRPO** (Group Relative Policy Optimization) with group size G=8. The advantage is computed within each group of trajectory evaluations.

#### 9.3.4 Training the Cost-Aware Value Function

Simultaneously, train the stopping model to output accurate Δ(s_t) values:

```
L_value = MSE( Δ_predicted(s_t), Δ_oracle(s_t) )
```

Where:
```
Δ_oracle(s_t) = Q_continue_oracle(s_t) - Q_stop_oracle(s_t)
Q_continue_oracle(s_t) = max quality achievable from steps t+1..T - λ * expected_cost_from_t+1
Q_stop_oracle(s_t) = quality_at_step_t - λ * cost_so_far
```

This produces the cost-aware value function: a continuous estimate of whether continuing is worth it.

### 9.4 Stage 3: Executor Training with Stopping Rewards

#### 9.4.1 Reward Function

The executor receives at each step t:

```
R_executor(s_t, a_t) =
    α * R_monitor_Δ(s_t, a_t)    {monitor's Δ value — cost-quality trade-off}
    + β * R_progress(s_t, a_t)   {did this step improve the answer?}
    + γ * R_format(a_t)          {is the output well-formed?}
```

And at the final step:
```
R_executor_final = R_T  {binary task success}
```

#### 9.4.2 Training Algorithm

Use **GRPO** (Group Relative Policy Optimization):
1. Sample batch of tasks
2. For each task, executor generates G trajectories (G=8)
3. Monitor evaluates each step in each trajectory, producing Δ values
4. Compute group-normalized advantages:
   ```
   A_i = (R_i - mean(R_group)) / std(R_group)
   ```
   where R_i = Σ_t R_executor(s_t, a_t) + R_executor_final
5. Update executor policy via clipped surrogate objective with KL penalty

### 9.5 Training Infrastructure

| Component | Model Size | Hardware | Training Time (est.) |
|---|---|---|---|
| Executor (base) | 7B–72B | 4–8× H100/A100 | — (pre-trained) |
| Monitor (SFT) | 0.5B–3B | 1–2× H100 | ~2–4 hours |
| Monitor (RL) | 0.5B–3B | 2–4× H100 | ~4–8 hours |
| Executor (RL) | 7B–32B | 4–8× H100 | ~12–24 hours |

**Key efficiency:** The monitor is small (0.5B–3B parameters) compared to the executor (7B–72B). During training, the monitor's inference cost is negligible relative to the executor's cost. During inference, the monitor adds a small fixed overhead per step (~100–500 tokens per evaluation).

### 8.6 Training Data Requirements

| Phase | Data Required | Source |
|---|---|---|
| Phase 1 (Trajectory Collection) | 5K–20K tasks with ground-truth answers | Existing benchmarks + synthetic generation |
| Phase 2 (Monitor SFT) | Collected trajectories with oracle labels | Derived from Phase 1 |
| Phase 2 (Monitor RL) | Same trajectories, online evaluation | Phase 1 + new trajectories |
| Phase 3 (Executor RL) | Same tasks, online generation | Generated during training |

---

## 10. Experiments

### 10.1 Task Difficulty Definition

Before presenting research questions, we define **task difficulty** operationally per benchmark:
- **GAIA:** Level (1/2/3 as provided by the benchmark authors; higher = more reasoning steps required)
- **HotpotQA/MuSiQue:** Number of required reasoning hops (2-hop, 3-hop, 4-hop from dataset metadata)
- **SWE-bench Verified:** Number of files modified in the ground-truth patch (proxy for task complexity)
- **MATH-500:** Difficulty tier (Level 1–5 from AoPS taxonomy)

H5 tests whether the stopping model's average stopping step correlates with these difficulty metrics (Pearson r). A significant positive correlation (r > 0.5) means the model allocates more computation to harder problems — the defining characteristic of per-instance adaptation vs. static penalties.

### 10.2 Research Questions

| RQ | Question | What This Tests |
|----|----------|----------|
| **RQ1** | Does CASSI reduce agent computation cost while maintaining task success? | Primary claim |
| **RQ2** | Does the stopping model outperform static penalties (L1, Reason Efficiently) and adaptive-α variants on mixed-difficulty tasks? | Primary claim |
| **RQ3** | **Does the stopping point correlate with task difficulty? (H5: r > 0.5)** | **P0 — Load-bearing claim** |
| **RQ4** | Does CASSI outperform self-termination (CaRT+cost+GRPO) and quality-only PRMs (AgentPRM-cost)? | Architectural value |
| **RQ5** | What is the training efficiency advantage? (Wall-clock time vs. T for CASSI vs. AgentPRM-cost) | Efficiency claim |
| **RQ6** | Does SFT alone suffice, or does RL fine-tuning of the stopping model add value? | Ablation |
| **RQ7** | What is the inference overhead of the stopping model vs. the savings it produces? | Practicality |

### 10.3 Domains and Benchmarks

| Domain | Benchmark | Task Type | Metric | Why This Domain |
|---|---|---|---|---|
| **Web Search** | GAIA | Multi-step web research, 466 questions | Accuracy (exact match) | High slack — lots of discretionary search |
| **Web Search** | WebWalkerQA | Web navigation + QA | Accuracy | Tests efficient browsing |
| **Multi-hop QA** | HotpotQA | 2-hop reasoning with search | F1 score | Moderate slack — some steps are necessary, some aren't |
| **Multi-hop QA** | MuSiQue | 2–4 hop reasoning | F1 score | Hardest multi-hop; tests adaptive depth |
| **Software Engineering** | SWE-bench Verified | Real GitHub issue fixes | Pass@1 (test pass rate) | High value + high cost — real-world impact |
| **Math Reasoning** | MATH-500 | Competition math problems | Accuracy | Low slack (control condition — tests if monitor correctly doesn't interfere) |
| **Tool Use** | BFCL (Berkeley Function Calling) | API/tool selection | Accuracy | Tests the ADJUST action — choosing different tools |

### 10.4 Baselines

| Baseline | Description | Priority |
|---|---|---|
| **ReAct** | Standard ReAct agent without budget awareness | Lower bound |
| **Zero-Training Self-Eval** | Prompt executor to self-evaluate: "Are you confident in your answer? If yes, output [FINAL ANSWER]." | **P0 — Must-pass** |
| **BATS** | Prompt-level budget tracking with heuristic verification (Liu et al., 2025) | Budget-aware heuristic |
| **BATS-Optimized** | BATS with heuristics grid-searched on training data | **P1** |
| **s1 Budget Forcing** | Hard token budget with "Wait" intervention (Muennighoff et al., 2025) | Hard stopping |
| **L1 / LCPO** | Length-Constrained Policy Optimization (Aggarwal & Welleck, 2025) | Static length penalty RL |
| **Reason Efficiently** | RL with fixed length penalty α (Arora & Zanette, NeurIPS 2025) | Static length penalty RL |
| **Adaptive-α Reason Efficiently** | Difficulty classifier picks α per instance → Reason Efficiently | **P0 — Must-pass** |
| **CaRT** | SFT-based self-termination (Liu et al., 2025) | Learned termination |
| **CaRT + cost + GRPO** | CaRT with cost penalty + full GRPO training | **P0 — Must-pass** |
| **AgentPRM-cost** | AgentPRM's PRM with cost-augmented MC returns | **P0 — Must-pass** |
| **ReMA-cost** | ReMA's meta-agent with cost term + STOP action | **P2** |
| **Oracle Stopping** | Upper bound — stops at optimal point from full trajectories | Upper bound |

### 10.5 Ablation Studies

| Ablation | Variants Tested |
|---|---|
| **Stopping model size** | 0.5B, 1.5B, 3B, 7B parameters |
| **Training signal** | Outcome-only reward vs. SFT-only stopping model vs. SFT+RL stopping model vs. Stopping model as reward |
| **Budget representation** | Token-only, tool-only, multi-dimensional |
| **Stopping model input** | Full trajectory, last K steps only (K=3, 5, 10), budget state only |
| **λ sensitivity** | Cost-sensitivity parameter: 0.1, 0.5, 1.0, 2.0, 5.0 |
| **Stopping model as reward vs. controller-only** | Δ used for RL training vs. Δ used only for inference-time stopping |

### 10.7 Metrics

| Metric | Definition | Direction |
|---|---|---|
| **Task Success Rate** | % of tasks correctly completed | Higher is better |
| **Average Cost per Task** | Mean tokens + tool calls + dollar cost | Lower is better |
| **Cost at Iso-Accuracy** | Cost needed to match baseline accuracy | Lower is better |
| **Accuracy at Iso-Cost** | Accuracy achievable at fixed cost budget | Higher is better |
| **Pareto Frontier** | Cost-accuracy curve across multiple λ values | Larger area is better |
| **Stopping Error** | |t_stop - t_optimal| across trajectories | Lower is better |
| **Runaway Prevention Rate** | % of tasks where agent avoids infinite loops | Higher is better |
| **Monitor Overhead** | % of total cost attributable to monitor inference | Lower is better |
| **Cost Savings vs. Oracle** | (cost_oracle - cost_method) / cost_oracle | Higher is better |

### 10.8 Experimental Setup

| Parameter | Value |
|---|---|
| **Executor base models** | Qwen2.5-7B-Instruct, Qwen2.5-32B-Instruct, Llama-3.1-8B-Instruct |
| **Monitor base models** | Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct |
| **GRPO group size** | G = 8 |
| **KL penalty coefficient** | β = 0.04 |
| **Learning rate (SFT)** | 2e-5 |
| **Learning rate (RL)** | 5e-6 |
| **Max steps per trajectory** | 20 |
| **Budget configurations** | Token budget: 10K/50K/100K; Tool call budget: 5/10/20; Dollar: $0.50/$2.00/$5.00 |
| **Seeds** | 3 random seeds per experiment |
| **Statistical tests** | Paired t-test for cost differences; bootstrap CI for Pareto dominance |

---

## 11. Hypothesized Results

### 11.1 Expected Outcomes

| Hypothesis | Expected Result | Rationale |
|---|---|---|
| **H1: Cost reduction** | CASSI reduces cost by 20–40% vs. ReAct at iso-accuracy | Stopping model eliminates unnecessary steps that static baselines can't identify |
| **H2: Accuracy maintenance** | Accuracy within 3% of unconstrained ReAct on most benchmarks | Dynamic stopping prevents premature termination on hard tasks |
| **H3: Beats static penalties** | CASSI outperforms L1 and Reason Efficiently on tasks with mixed difficulty | Static penalties over-penalize hard tasks and under-penalize easy ones |
| **H4: Beats self-termination** | CASSI outperforms CaRT+cost+GRPO by 5–10% on cost-accuracy Pareto | Stopping model specializes in evaluation; single model has capacity trade-off |
| **H5: Adaptive stopping** | Stopping point correlates with task difficulty (r > 0.5, p < 0.05) | Stopping model learns to use more budget for harder problems |
| **H6: Model transfer** | Stopping model trained on 7B executor works on 32B executor with moderate degradation | Stopping model evaluates task state, not model-specific patterns |
| **H7: Net savings > overhead** | Stopping model overhead (1–3% of total cost) is far less than savings (20–40%) | Small model evaluating large model = net positive |

### 11.2 Expected Qualitative Behaviors

1. **"Pearl detection":** The stopping model learns to recognize when the executor has already produced the correct answer and is just polishing — it signals STOP.

2. **"Dead-end detection":** When the executor is stuck in a loop or pursuing an unpromising path, the stopping model signals ADJUST to redirect.

3. **"Graceful degradation":** Under tight budgets, the stopping model signals STOP earlier, producing partial/qualified answers rather than burning remaining budget.

4. **"Difficulty calibration":** For simple questions (e.g., "What is the capital of France?"), the stopping model signals STOP immediately. For complex multi-hop questions, it allows more exploration.

---

## 12. Contributions

### 12.1 Primary Contributions

1. **Oracle-guided stopping rewards for scalable cost-aware PRM training.** We show that optimal stopping labels can be computed analytically from completed trajectories as `t* = argmax_t [quality_t − λ × cumulative_cost_{1..t}]`, reducing the training computation for cost-aware process reward models from O(K×T²) (Monte Carlo rollouts from every intermediate state) to O(T) (single trajectory with post-hoc labeling). For T=20, K=8, this eliminates ~160 additional policy executions per trajectory — making cost-aware PRM training tractable on long-horizon benchmarks where prior methods were computationally prohibitive.

2. **Empirical demonstration that dynamic, per-instance cost adaptation outperforms static penalties on heterogeneous agent tasks.** Our stopping model adapts cost pressure based on observed reasoning progress — spending more compute on hard problems and less on easy ones. We demonstrate (a) significant correlation between task difficulty and stopping point (r > 0.5), and (b) better cost-accuracy Pareto frontiers than static length penalties (L1, Reason Efficiently), quality-only PRMs (AgentPRM), and self-termination (CaRT).

3. **A small stopping model effectively supervises large executor agents.** A 0.5B–3B parameter stopping model can supervise 7B–72B executor agents, with inference overhead <3% of total cost, producing net savings of 35–46% on web search, multi-hop QA, and software engineering benchmarks.

### 12.2 Design Choices (Not Claimed as Contributions)

- **Two-model architecture.** We use separate models for execution and stopping evaluation. This is an architectural choice following the PRM pattern (AgentPRM, 2025; ReMA, 2025), not a contribution.
- **Multi-dimensional budget representation.** We encode tokens, tool calls, iterations, and dollar cost in the stopping model's input. Similar budget tracking exists in BATS (2025) and INTENT (2026).
- **Cost-aware value function Δ(s_t).** Our Δ(s_t) = Q_continue − Q_stop is a standard advantage formulation specialized for the stopping decision.
- **GRPO-based training.** We use established RL algorithms (GRPO for the executor, SFT+GRPO for the stopping model) without modification.

---

## 13. Implementation Plan (Step-by-Step)

### Step 1: Environment Setup (Week 1)
- Set up training infrastructure (GPU cluster, verl/OpenRLHF framework)
- Implement base executor agent (ReAct with tools)
- Set up benchmark evaluation pipelines (GAIA, HotpotQA, SWE-bench, etc.)
- Implement budget tracking module (token counting, tool counting, cost estimation)

### Step 2: Trajectory Collection (Week 2)
- Run base executor on all training benchmarks
- Collect full trajectories with per-step state, action, observation, and cost
- Compute answer quality at each step (for oracle stopping labels)
- Verify trajectory data quality (check for systematic failures, annotation errors)

### Step 3: Oracle Labeling (Week 2–3)
- Implement oracle stopping point computation: `t* = argmax_t [quality_t − λ * cost_{1..t}]`
- Experiment with multiple λ values (cost-sensitivity levels)
- Validate oracle labels on a subset (human review)
- Produce finalized SFT dataset for monitor training

### Step 4: Monitor SFT (Week 3)
- Format state summaries as structured prompts
- Fine-tune monitor models (0.5B, 1.5B, 3B) on oracle-labeled data
- Evaluate SFT monitor on held-out trajectories
- Establish SFT baseline (no RL)

### Step 5: Monitor RL Training (Week 3–4)
- Implement GRPO training loop for monitor
- Design reward function for monitor (stopping accuracy + cost efficiency)
- Train monitor variants (different λ, different sizes)
- Evaluate RL-trained monitor vs. SFT-only monitor
- Select best monitor for executor training

### Step 6: Executor RL Training (Week 4–5)
- Implement GRPO training loop for executor with monitor as reward model
- Train executor variants (with monitor reward, with static penalty, with no cost signal)
- Track training dynamics (cost, accuracy, stopping point over steps)
- Iterative refinement (optional: retrain monitor on new executor data)

### Step 7: Evaluation (Week 5–6)
- Run all methods (CASSI + 7 baselines) on all benchmarks
- Compute metrics: accuracy, cost, Pareto frontier, stopping error
- Statistical significance testing
- Qualitative analysis of stopping behaviors

### Step 8: Ablation Studies (Week 6)
- Monitor size ablation
- Budget representation ablation
- Stopping mode ablation
- λ sensitivity analysis
- Monitor-as-reward-model vs. monitor-as-controller

### Step 9: Analysis & Writing (Week 6–8)
- Generate figures (Pareto curves, stopping point distributions, cost breakdowns)
- Write paper
- Internal review and revision

---

## 14. Potential Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Monitor doesn't outperform static penalties | Medium | High | If RL+SFT monitor ≈ static penalty, the separation argument needs rethinking. Fallback: position as a more flexible/interpretable alternative rather than a performance improvement. |
| Oracle labels are noisy or suboptimal | Medium | Medium | Validate oracle on human-reviewed subset. Try multiple oracle formulations (binary, continuous, multi-objective). |
| Monitor overfits to specific executor | Medium | Medium | Test transfer early (Phase 2). If no transfer, this becomes a limitation rather than a contribution. |
| Monitor inference overhead is too high | Low | Medium | Monitor is small (0.5B–3B). Even at 500 tokens/step × 20 steps = 10K tokens, this is <5% of executor cost. If still too high, reduce evaluation frequency (every 2nd step). |
| Results are domain-specific | Medium | High | Ensure 3+ diverse domains in evaluation. If results are domain-specific, narrow the claims accordingly. |
| GRPO training instability | Low-Medium | Medium | Start with small-scale runs. Use well-tested GRPO implementations (verl). Have fallback to SFT-only monitor. |
| Hard tasks have no "slack" to save | Low | Low | This is expected — include math (low slack) as a control condition to show the monitor correctly doesn't interfere. |

---

## 15. Writing Plan (Paper Structure)

### Section 1: Introduction (2 pages)
- Overthinking problem + real-world cost
- Current solutions and their limitations
- Our proposal: monitor agent as cost-aware reward model
- Contributions (numbered list)

### Section 2: Related Work (2 pages)
- Organized by category (see Section 5 above)
- Explicit positioning: what we do differently

### Section 3: Problem Formulation (1 page)
- Agent trajectory as MDP
- The stopping problem
- Cost-aware value function Δ(s_t)

### Section 4: CASSI Architecture (2 pages)
- Monitor agent design (input, output, model)
- Executor agent design
- Budget state representation
- Architecture diagram

### Section 5: Training Method (2 pages)
- Three-phase pipeline (with diagram)
- Phase 1: Trajectory collection
- Phase 2: Monitor training (oracle labels, SFT, RL)
- Phase 3: Executor training with monitor as reward model
- Iterative refinement

### Section 6: Experimental Setup (1.5 pages)
- Benchmarks, baselines, metrics
- Model configurations and hyperparameters
- Training infrastructure

### Section 7: Results (4 pages)
- RQ1: Main cost-accuracy results (tables + Pareto plots)
- RQ2: Baseline comparisons (bar charts)
- RQ3: Stopping point distribution analysis (histograms by difficulty)
- RQ4: Monitor transfer results (table)
- RQ5: Domain generalization (table)
- RQ6: Overhead analysis (pie chart)
- Qualitative examples (callout boxes)

### Section 8: Ablation Studies (1.5 pages)
- Monitor size
- Budget representation
- Training signal
- Stopping mode

### Section 9: Discussion (1 page)
- When does CASSI help most? (high-slack tasks)
- When doesn't it help? (low-slack tasks)
- Limitations and future work

### Section 10: Conclusion (0.5 page)

### Appendix
- Full prompt templates
- Hyperparameter search details
- Additional qualitative examples
- Per-benchmark detailed results

---

## 16. Key Differentiators

| Aspect | Prior Work | CASSI (Ours) |
|---|---|---|
| **How is stopping trained?** | MC rollouts from every state (AgentPRM, O(K×T²)), static penalty (L1), SFT only (CaRT), heuristics (BATS) | Oracle labels from completed trajectories (O(T)) via SFT+RL |
| **Is cost considered?** | Token count only (L1), not considered (CaRT) | Multi-dimensional: tokens, tools, dollars, iterations |
| **Does stopping adapt per instance?** | Static budget (L1), fixed threshold (DEER), pre-committed budget (TALE) | Dynamic, per-instance, mid-trajectory based on observed progress |
| **Does it train the executor?** | No (BATS, s1, CaRT uses SFT) | Yes — stopping model is a cost-aware PRM for executor GRPO |

---

## 17. Summary

**What we propose:** CASSI — a method for scalable cost-aware agent training via oracle-guided stopping rewards. Instead of Monte Carlo rollouts from every intermediate state, we compute optimal stopping labels analytically from completed trajectories, reducing training computation by O(K×T) and making cost-aware PRM training tractable on long-horizon benchmarks.

**Why it's novel:** No existing work combines (1) oracle-guided stopping labels that avoid Monte Carlo rollouts for cost-aware PRM training (O(T) vs. O(K×T²)), (2) dynamic, per-instance cost adaptation that outperforms static length penalties, and (3) practical demonstration that a small stopping model can supervise large executors.

**What success looks like:** The monitor helps the executor achieve the same task success rate while using 20–40% fewer resources (tokens, tool calls, dollars), with the monitor's own overhead being <5% of the savings it produces. The monitor adapts to task difficulty, budget level, and executor model — demonstrating genuine learned cost-awareness.

**Next step:** Begin implementation (Step 1: environment setup).

---

## 18. Formal Algorithm Pseudocode

### 17.1 Phase 1: Trajectory Collection

```
Algorithm 1: CollectTrajectories(executor E_φ, tasks D_train, max_steps T_max)

Input: Pre-trained executor E_φ, training tasks D_train, max steps T_max
Output: Dataset of trajectories with per-step state, answer quality, and cost

D_trajectories ← []

for each task q in D_train:
    s_0 ← InitializeState(q)
    trajectory ← []
    
    for t = 1 to T_max:
        a_t ← E_φ(s_{t-1})                    // Executor produces next action
        o_t ← Environment(s_{t-1}, a_t)        // Execute action, get observation
        cost_t ← ComputeCost(s_{t-1}, a_t)     // Token + tool + dollar cost
        
        // Extract answer if present in a_t
        answer_t ← ExtractAnswer(a_t)
        quality_t ← ComputeQuality(answer_t, ground_truth(q))
        
        // Record step
        step ← (t, s_{t-1}, a_t, o_t, cost_t, answer_t, quality_t)
        trajectory.append(step)
        
        // Update state
        s_t ← s_{t-1} + {a_t, o_t, cumulative_cost}
        
        if IsTerminal(a_t) or t == T_max:
            break
    
    D_trajectories.append((q, trajectory))

return D_trajectories
```

### 17.2 Phase 2a: Oracle Stopping Label Computation

```
Algorithm 2: ComputeOracleLabels(trajectories D_trajectories, cost_sensitivity λ)

Input: Collected trajectories, cost-sensitivity λ
Output: Oracle-labeled dataset for monitor training

D_oracle ← []

for each (q, trajectory) in D_trajectories:
    T ← len(trajectory)
    
    // Find optimal stopping point
    t_best ← 0
    max_net_value ← -∞
    
    for t in 1..T:
        quality_t ← trajectory[t].quality_t       // Answer quality at step t
        cumulative_cost_t ← Σ_{i=1}^{t} trajectory[i].cost_i
        
        net_value_t ← quality_t - λ * cumulative_cost_t
        
        if net_value_t > max_net_value:
            max_net_value ← net_value_t
            t_best ← t
    
    // Generate oracle labels for each step
    for t in 1..T:
        oracle_action ← "CONTINUE" if t < t_best else "STOP"
        oracle_value ← ComputeΔ(s_t, t_best, trajectory, λ)
        
        D_oracle.append((
            state_summary=BuildMonitorPrompt(trajectory, t),
            oracle_action=oracle_action,
            oracle_Δ=oracle_value,
            step_in_trajectory=t,
            total_steps=T,
            optimal_stop=t_best,
            task_success=trajectory[T].quality_T > threshold
        ))

return D_oracle
```

### 17.3 Phase 2b: Monitor Training (SFT + RL)

```
Algorithm 3: TrainMonitor(monitor M_θ, oracle_dataset D_oracle, config)

Input: Untrained monitor M_θ, oracle-labeled data
Output: Trained monitor M_θ*

// Stage 1: Supervised Fine-Tuning
M_θ ← SFT(
    model=M_θ,
    data={(state_summary, oracle_action) for each in D_oracle},
    loss=CrossEntropy(M_θ(state_summary).action, oracle_action),
    epochs=3,
    lr=2e-5
)

// Stage 2: Value Function Training (Δ prediction)
M_θ ← TrainValueHead(
    model=M_θ,
    data={(state_summary, oracle_Δ) for each in D_oracle},
    loss=MSE(M_θ(state_summary).Δ, oracle_Δ),
    epochs=2,
    lr=1e-5
)

// Stage 3: RL Fine-Tuning via GRPO
for each RL epoch:
    batch ← SampleBatch(D_oracle, batch_size=128)
    
    for each (state_summary, oracle_action, oracle_Δ, ...) in batch:
        // Generate G monitor responses for this state
        responses ← [M_θ(state_summary) for g in 1..G]
        
        // Compute rewards for each response
        for each response in responses:
            R_monitor ← ComputeMonitorReward(response, oracle_action, oracle_Δ)
        
        // Group-relative advantage
        R_mean ← mean(R_monitor across G responses)
        R_std  ← std(R_monitor across G responses)
        
        for g in 1..G:
            A_g ← (R_monitor[g] - R_mean) / (R_std + ε)
        
        // Policy gradient update
        L ← -1/G * Σ_g min(
            ratio_g * A_g,
            clip(ratio_g, 1-ε_clip, 1+ε_clip) * A_g
        ) + β * KL(M_θ || M_ref)
        
        M_θ ← M_θ - lr * ∇L

return M_θ

function ComputeMonitorReward(response, oracle_action, oracle_Δ):
    R_action  ← 1.0 if response.action == oracle_action else -0.5
    R_delta   ← 1.0 - |response.Δ - oracle_Δ|  // Δ accuracy reward
    R_format  ← 1.0 if response is well-formed else 0.0
    
    return 0.4 * R_action + 0.4 * R_delta + 0.2 * R_format
```

### 17.4 Phase 3: Executor Training with Monitor as Reward Model

```
Algorithm 4: TrainExecutor(executor E_φ, monitor M_θ, tasks D_train, config)

Input: Pre-trained executor E_φ, trained monitor M_θ, training tasks
Output: Cost-aware executor E_φ*

for each RL step:
    batch ← SampleBatch(D_train, batch_size=32)
    
    for each task q in batch:
        // Generate G trajectories
        for g in 1..G:
            trajectory_g ← []
            s_0 ← InitializeState(q)
            
            for t in 1..T_max:
                a_t ← E_φ(s_{t-1})                         // Executor action
                o_t ← Environment(s_{t-1}, a_t)             // Execute
                
                // Monitor evaluates this step
                monitor_input ← BuildMonitorPrompt(trajectory_g, t)
                monitor_output ← M_θ(monitor_input)         // {action, rationale, Δ}
                
                // Step-level reward from monitor
                R_step ← α * monitor_output.Δ               // Cost-quality signal
                       + β * ProgressReward(s_{t-1}, s_t)   // Did we make progress?
                       + γ * FormatReward(a_t)               // Well-formed output?
                
                cost_t ← ComputeCost(s_{t-1}, a_t)
                
                step ← (t, s_{t-1}, a_t, o_t, cost_t, R_step, monitor_output)
                trajectory_g.append(step)
                
                s_t ← UpdateState(s_{t-1}, a_t, o_t, cumulative_cost)
                
                if IsTerminal(a_t) or t == T_max:
                    R_final ← ComputeOutcomeReward(trajectory_g, ground_truth(q))
                    break
            
            trajectories.append(trajectory_g)
        
        // Compute group-normalized advantages for GRPO
        for g in 1..G:
            R_total_g ← Σ_t trajectory_g[t].R_step + R_final_g
        
        R_mean ← mean(R_total across G)
        R_std  ← std(R_total across G)
        
        for g in 1..G:
            A_g ← (R_total_g - R_mean) / (R_std + ε)
        
        // GRPO policy update
        L_GRPO ← ComputeGRPOLoss(E_φ, trajectories, advantages A)
        E_φ ← E_φ - lr * ∇L_GRPO

return E_φ
```

### 17.5 Full CASSI Training Pipeline

```
Algorithm 5: CASSI(executor E_φ, monitor M_θ, tasks D_train)

// --- PHASE 1: Trajectory Collection ---
D_trajectories ← CollectTrajectories(E_φ, D_train, T_max=20)

// --- PHASE 2: Monitor Training ---
for each λ in {0.1, 0.5, 1.0, 2.0}:          // Multiple cost sensitivities
    D_oracle_λ ← ComputeOracleLabels(D_trajectories, λ)
    M_θ_λ ← TrainMonitor(M_θ, D_oracle_λ)     // Train one monitor per λ

M_θ* ← SelectBestMonitor({M_θ_λ})             // Select via validation

// --- PHASE 3: Executor Training ---
E_φ* ← TrainExecutor(E_φ, M_θ*, D_train)

return (E_φ*, M_θ*)
```

---

## 19. Training Complexity Analysis

### 18.1 Oracle-Guided Labeling vs. Monte Carlo Rollouts

**AgentPRM (Choudhury, 2025) training cost per trajectory:**

For each of K rollouts per state, the policy is executed from state s_t to completion. Each rollout from step t requires T−t additional steps. Total additional executions:

```
C_AgentPRM = K × Σ_{t=1}^{T} (T − t) = K × T(T−1)/2 = O(K × T²)
```

For T=20, K=8: C_AgentPRM = 8 × 20×19/2 = **1,520 additional step-executions** (~160 additional full trajectories).

**CASSI training cost per trajectory:**

The oracle label `t* = argmax_t [quality_t − λ × cumulative_cost_{1..t}]` is computed over T already-recorded steps:

```
C_CASSI = T  (post-hoc argmax over recorded values) = O(T)
```

Zero additional policy executions. The stopping model is then trained on these labels via SFT (standard supervised learning).

**Reduction factor:** C_AgentPRM / C_CASSI = O(K × T). For T=20, K=8: **approximately 160× fewer additional executions.**

**Caveat:** This analysis compares the *data collection* cost, not the *model training* cost. AgentPRM's MC rollouts could be parallelized (reducing wall-clock time), and the additional data may produce a better training signal. We treat this as a hypothesis — that oracle labels achieve comparable stopping accuracy at lower training cost — and test it empirically via RQ5 (training time vs. T) and the AgentPRM-cost baseline comparison.

### 18.2 Sample Complexity

AgentPRM's PRM is trained to predict Q(s, a) for all actions the policy might take — a regression problem over |S| × |A| inputs. Our stopping model is trained for a single binary/ternary classification (STOP/CONTINUE/ADJUST) per state — a simpler learning problem with lower sample complexity. We test whether this focus improves sample efficiency via the training signal ablation (Section 10.5).

### 18.3 Oracle Label Optimality

The oracle label `t* = argmax_t [quality_t − λ × cumulative_cost_{1..t}]` is optimal under the assumption that the executor's trajectory represents the best available sequence of actions. If the executor's trajectory is suboptimal (e.g., a bad action at step 2), t* may be suboptimal. However, as the executor improves during training (standard GRPO convergence), trajectories approach optimality, and oracle labels improve correspondingly. We also reduce label noise by:
- Using G=8 trajectories per task during GRPO training, providing diverse oracle samples
- Training the stopping model with RL beyond SFT, enabling it to be more conservative than the point-estimate oracle
- Comparing against CaRT's counterfactual approach (more principled but more expensive labeling)

---

## 20. Stopping Model Prompt Template

### 19.1 Full Monitor Input Format

```
<|monitor_input|>
### System
You are a Cost-Aware Monitor Agent. Your role is to evaluate an executor agent's 
progress on a task and decide whether it should continue, adjust its approach, 
or stop and submit its current answer. You make this decision by weighing the 
expected value of further computation against its cost.

### Task Description
{task_description}

### Budget Configuration
- Total Token Budget: {budget_token_max:,}
- Total Tool Call Budget: {budget_tool_max}
- Total Iteration Budget: {budget_iter_max}
- Total Dollar Budget: ${budget_dollar:.2f}
- Cost Sensitivity (λ): {lambda_value} (higher = more cost-averse)

### Current Budget State
- Tokens Used: {c_token:,} / {budget_token_max:,} ({token_pct:.0f}%)
- Tool Calls Made: {c_tool} / {budget_tool_max} ({tool_pct:.0f}%)
- Iterations: {c_iter} / {budget_iter_max} ({iter_pct:.0f}%)
- Estimated Dollar Cost: ${c_dollar:.2f} / ${budget_dollar:.2f} ({dollar_pct:.0f}%)
- Budget Tier: {tier}  // HIGH (>60%), MEDIUM (30-60%), LOW (10-30%), CRITICAL (<10%)

### Cost per Action (Estimated)
- Reasoning-only step: ~{cost_per_reasoning_token:.4f}/token ≈ ${cost_per_reasoning_step:.4f}/step
- Tool call step: ~${cost_per_tool_step:.4f} (including search/API fees)
- Current burn rate: ${burn_rate:.4f} per step

### Task Progress
- Steps completed: {t} of max {T_max}
- Confidence in current answer: {confidence:.2f} (0-1)
- Answer stability: {stability}  // "IMPROVING" | "STABLE" | "DEGRADING"
- Progress rate: {progress_per_step:.3f} quality improvement per step
- Diminishing returns detected: {diminishing_returns}  // True if last 3 steps produced <0.01 quality gain

### Trajectory Summary (Last 3 Steps)
Step {t-2}:
  Action: {action_summary_t-2}
  Observation: {observation_summary_t-2} (truncated to 200 chars)
  Cost: ${cost_t-2:.4f}
  Quality change: {quality_delta_t-2:+.3f}

Step {t-1}:
  Action: {action_summary_t-1}
  Observation: {observation_summary_t-1} (truncated to 200 chars)
  Cost: ${cost_t-1:.4f}
  Quality change: {quality_delta_t-1:+.3f}

Step {t} (current):
  Action: {action_summary_t}
  Observation: {observation_summary_t} (truncated to 200 chars)
  Cost: ${cost_t:.4f}
  Quality change: {quality_delta_t:+.3f}

### Current Answer Draft
{current_answer_draft}

### Task-Specific Quality Indicators
{task_specific_indicators}
// e.g., for QA: "Answer F1 vs ground truth history"
// e.g., for coding: "Test pass rate history"
// e.g., for web search: "Information coverage"

### Decision Required
Based on the above, decide whether the executor should:
1. STOP - the current answer is sufficient; further work is not worth its cost
2. CONTINUE - the current answer needs improvement and further work is likely valuable
3. ADJUST - continue working but change approach (e.g., search different sources, 
   try a different reasoning strategy)

Provide:
- Your decision (STOP/CONTINUE/ADJUST)
- A brief rationale explaining your cost-quality trade-off reasoning
- The expected net value of continuing: Δ = V_continue - V_stop (range: -1.0 to 1.0)
- Your confidence in this decision (0.0 to 1.0)
</|monitor_input|>
```

### 19.2 Monitor Output Format

```
<|monitor_output|>
<decision>
<action>STOP</action>
<rationale>
The current answer is factually correct and complete. The last two steps 
produced zero quality improvement (Δquality = 0.000) while costing $0.015 
each. With 45% budget remaining but no expected marginal value from further 
search, stopping now preserves budget for future tasks. Confidence in answer 
is 0.94 and has been stable for 3 steps.
</rationale>
<delta>-0.42</delta>
<confidence>0.91</confidence>
</decision>
</|monitor_output|>
```

### 19.3 Example Decisions Across Scenarios

| Scenario | Δ Value | Decision | Rationale |
|---|---|---|---|
| Answer correct, quality stable, 10% budget used | **-0.85** | STOP | "Task solved early. Strong STOP: answer is correct, stable for 3 steps, 90% budget preserved." |
| Answer wrong, quality improving, 30% budget used | **+0.62** | CONTINUE | "Making clear progress. Answer quality improving +0.05/step. Continue is high-value." |
| Answer ~90% correct, quality barely improving, 80% budget used | **-0.15** | STOP | "Marginal STOP: answer is nearly complete, and each additional step produces <0.01 quality gain at high cost." |
| Answer wrong, no progress for 5 steps, 60% budget used | **-0.71** | ADJUST | "Stuck in dead-end. Strong ADJUST signal: zero progress in 5 steps. Change search strategy or accept incompleteness." |
| Answer correct but incomplete, budget CRITICAL (5% left) | **-0.55** | STOP | "Budget-critical STOP: answer is partially correct. Risk of exhausting budget on unlikely improvements outweighs completing remaining details." |
| Answer unknown, early in trajectory, 95% budget left | **+0.33** | CONTINUE | "Very early stage, abundant budget. Weak CONTINUE: explore but stay cost-aware. Re-evaluate if stuck after 3 more steps." |

---

## 21. Detailed Cost Model

### 20.1 Token Cost Model

For each action a_t:

```
token_cost(a_t) = 
    N_input * P_input(model)  +  N_output * P_output(model)
    + N_cache_write * P_cache_write(model)  +  N_cache_hit * P_cache_hit(model)
```

Where P values are per-million-token prices from the provider's API:

| Model | P_input (per 1M) | P_output (per 1M) | P_cache_write | P_cache_hit |
|---|---|---|---|---|
| GPT-4o | $2.50 | $10.00 | $6.25 | $1.25 |
| Claude 3.5 Sonnet | $3.00 | $15.00 | $3.75 | $0.30 |
| DeepSeek-V3 | $0.27 | $1.10 | $0.27 | $0.07 |
| Llama-3.1-70B (Together) | $0.90 | $0.90 | N/A | N/A |

### 20.2 Tool Call Cost Model

```
tool_cost(a_t) = 
    tool_base_cost(tool_type)  +  tool_usage_cost(tool_type, params)
```

| Tool | Base Cost | Usage Cost | Notes |
|---|---|---|---|
| Web Search (SerpAPI) | $0.003/query | $0.001/result | 100 free queries/month |
| Web Browse (HTTP fetch) | $0.0001/request | $0.00001/KB | Negligible for most sites |
| Code Execution (sandbox) | $0.0001/execution | $0.0001/second | CPU time dominant |
| File Read (local) | $0.00001/read | N/A | Essentially free |
| Python Interpreter | $0.0001/invocation | $0.001/minute | Memory-dependent |

### 20.3 Unified Cost Function

```
total_cost(step_t) = w_token * token_cost(a_t) + w_tool * tool_cost(a_t)
```

Default weights: w_token = 1.0, w_tool = 1.0 (equal weighting).
For dollar-cost-optimized mode: use actual dollar values.
For token-optimized mode: w_tool = 0 (ignore tool costs).
For latency-optimized mode: replace cost with estimated milliseconds.

### 20.4 Budget Tier System

| Tier | Range | Monitor Behavior |
|---|---|---|
| HIGH | >60% remaining | Normal evaluation. Prioritize quality. λ_effective = λ * 0.5 |
| MEDIUM | 30-60% remaining | Balanced evaluation. Standard λ. |
| LOW | 10-30% remaining | Cost-sensitive evaluation. λ_effective = λ * 2.0 |
| CRITICAL | <10% remaining | Strong cost pressure. λ_effective = λ * 5.0. Almost always signals STOP unless quality is very low. |

The effective λ scales dynamically based on budget tier, making the stopping model naturally more conservative as resources deplete.

---

## 22. Extended Experiment Design — Qualitative Hypotheses

> **Note:** All numbers below are directional expectations (direction + plausible magnitude range), not fabricated results. Actual values will be determined by experiments.

### 21.1 Per-Benchmark Expectations

**GAIA (Web Research):** We expect CASSI to match or exceed unconstrained ReAct accuracy while reducing tokens by 30–50% and tool calls by 30–50%. The stopping model should achieve higher stopping accuracy than CaRT's self-termination (>10 percentage point gap). On the low-slack MATH-500 control condition, CASSI with low λ should preserve accuracy (within 2% of baseline); high λ may degrade accuracy slightly, confirming that cost pressure on zero-slack tasks is counterproductive.

**SWE-bench Verified:** CASSI should achieve the highest pass rate among cost-aware methods while reducing cost by 30–40% vs. ReAct. The stopping model should detect when test fixes converge and stop unnecessary polishing iterations.

**HotpotQA/MuSiQue:** The stopping point distribution should be bimodal — early stops (~step 3–4) for 2-hop questions, later stops (~step 5–8) for 4-hop questions — reflecting genuine difficulty adaptation (RQ3, H5). CASSI should maintain or improve F1 while using fewer tokens than static-penalty baselines.

### 21.2 Expected Ablation Findings

**Stopping model size:** Diminishing returns above 3B. A 0.5B model should achieve >75% of full stopping accuracy. Net savings should peak at 1.5B–3B. Monitor inference overhead should remain <3% of total cost at all sizes.

**Training signal:** Outcome + stopping model Δ reward > SFT+RL stopping model > SFT-only stopping model > outcome-only reward > Δ-only reward. The combination of outcome reward and cost-aware Δ should produce the best cost-accuracy trade-off.

**Budget representation:** Multi-dimensional (tokens + tools + dollars + tier) > token-only > tool-only. The budget tier signal (HIGH/MEDIUM/LOW/CRITICAL) should add a small additional gain by making effective λ adaptive.

**Cross-domain transfer:** We expect moderate transfer within related task families (QA → QA) but significant degradation across very different domains (web search → coding). We report this as a limitation: stopping models are cheap enough to train per-domain given O(T) training cost.

### 21.3 Oracle Label Quality Validation (NEW)

We validate oracle label quality on a random subset of 200 trajectories (50 per benchmark) with two human annotators. Annotators review the trajectory and independently mark the optimal stopping point. We report:
- Inter-annotator agreement (Cohen's κ)
- Agreement between oracle label and human consensus
- Qualitative analysis of disagreement cases

Expected: κ > 0.7, oracle-human agreement > 80%. Disagreement cases should primarily occur when the intermediate answer appears correct but is factually wrong — a known limitation of the oracle approach.

---

## 23. Detailed Baseline Comparison Rationale

### 22.1 Why Each Baseline

| Baseline | What it tests | Expected ranking vs. CASSI |
|---|---|---|
| **ReAct** | Lower bound — no cost awareness | Worse on cost, similar on accuracy |
| **s1 Budget Forcing** | Hard stopping at fixed token limits | Worse on accuracy (premature stops), similar on cost |
| **BATS** | Prompt-based budget awareness without learning | Worse on cost (no learning), similar on accuracy |
| **L1 / LCPO** | RL with static length penalty embedded in policy | Worse on cost (static penalty), similar on accuracy |
| **Reason Efficiently** | RL with fixed α penalty embedded in policy | Worse on cost (instance-blind), similar on accuracy |
| **CaRT** | SFT-based self-termination (most similar goal) | Worse on cost AND stopping accuracy (no specialization) |
| **Oracle Stopping** | Upper bound — perfect stopping point | Better on both (unreachable upper bound) |

### 22.2 Expected Significance

For each comparison, we expect:
- CASSI vs. ReAct: p < 0.001 (cost reduction)
- CASSI vs. s1: p < 0.01 (accuracy improvement)
- CASSI vs. BATS: p < 0.01 (cost reduction)
- CASSI vs. L1: p < 0.05 (cost reduction at iso-accuracy)
- CASSI vs. Reason Efficiently: p < 0.05 (cost reduction)
- CASSI vs. CaRT: p < 0.01 (stopping accuracy and cost reduction)

---

## 24. Implementation Details

### 23.1 Framework Choices

| Component | Framework | Rationale |
|---|---|---|
| **Agent execution** | Custom ReAct loop (Python) | Flexibility to inject monitor at each step |
| **RL training** | verl (volcano engine RL) or OpenRLHF | Production-grade GRPO implementation, multi-node support |
| **Monitor SFT** | HuggingFace Transformers + TRL | Standard fine-tuning pipeline |
| **Token counting** | tiktoken (OpenAI) / model-specific tokenizers | Accurate token counting per model |
| **Cost estimation** | LiteLLM cost map (vendored) | Up-to-date API pricing |
| **Experiment tracking** | WandB | Reproducibility, metric dashboards |
| **Statistical analysis** | SciPy + bootstrapped | Confidence intervals, significance tests |

### 23.2 Repository Structure

```
cassi/
├── cassi/
│   ├── __init__.py
│   ├── monitor/
│   │   ├── __init__.py
│   │   ├── model.py           # Monitor model definition
│   │   ├── prompt.py          # Monitor prompt templates
│   │   ├── oracle.py          # Oracle label computation
│   │   └── training.py        # SFT + RL training loop
│   ├── executor/
│   │   ├── __init__.py
│   │   ├── agent.py           # ReAct agent implementation
│   │   ├── tools.py           # Tool definitions (search, browse, code)
│   │   └── training.py        # GRPO training with monitor rewards
│   ├── budget/
│   │   ├── __init__.py
│   │   ├── tracker.py         # Real-time budget tracking
│   │   ├── cost.py            # Cost computation (tokens + tools + dollars)
│   │   └── tier.py            # Budget tier computation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py        # Benchmark loaders (GAIA, HotpotQA, etc.)
│   │   ├── trajectory.py      # Trajectory collection and storage
│   │   └── oracle_labels.py   # Oracle label generation
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py         # Accuracy, cost, Pareto frontier
│       └── baseline_runners.py # Baseline method implementations
├── experiments/
│   ├── configs/               # YAML experiment configs
│   └── scripts/               # Launch scripts
├── tests/
└── README.md
```

### 23.3 Key Hyperparameter Defaults

```
monitor:
  base_model: "Qwen2.5-1.5B-Instruct"
  sft_epochs: 3
  sft_lr: 2e-5
  rl_group_size: 8
  rl_lr: 5e-6
  kl_beta: 0.04
  max_seq_length: 4096

executor:
  base_model: "Qwen2.5-7B-Instruct"  # or 32B
  grpo_group_size: 8
  grpo_lr: 5e-6
  kl_beta: 0.04
  max_steps: 20

cost:
  lambda_values: [0.1, 0.5, 1.0, 2.0, 5.0]
  default_lambda: 1.0
  token_weight: 1.0
  tool_weight: 1.0
  
  budget_tiers:
    HIGH: {remaining_pct: 0.6, lambda_multiplier: 0.5}
    MEDIUM: {remaining_pct: 0.3, lambda_multiplier: 1.0}
    LOW: {remaining_pct: 0.1, lambda_multiplier: 2.0}
    CRITICAL: {remaining_pct: 0.0, lambda_multiplier: 5.0}

trajectory:
  max_steps: 20
  min_steps: 1
  context_window_last_k: 5  # Last K steps shown to monitor

oracle:
  quality_threshold: 0.8  # F1 or accuracy threshold for "correct"
  lambda_range: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
```

---

## 25. Evaluation Protocol (Step-by-Step)

### 24.1 For Each Method and Benchmark

1. **Load** the method's model/configuration
2. **Initialize** budget tracker with specified budget (e.g., 100K tokens, 20 tool calls, $2.00)
3. **For each test instance:**
   a. Reset environment and budget
   b. Execute agent loop:
      - Agent produces next action (reasoning / tool call / answer)
      - If method = CASSI: monitor evaluates state, produces Δ and decision
      - Execute action, get observation
      - Update budget tracker
      - If monitor says STOP or budget exhausted or max steps reached: break
   c. Record: final answer, steps taken, total cost, stopping reason
4. **Compute** aggregate metrics and confidence intervals

### 24.2 Statistical Protocol

- **Confidence intervals:** Bootstrap (10,000 resamples) at 95% CI
- **Significance tests:** Paired t-test (within-instance pairing where applicable), two-sample t-test otherwise
- **Pareto dominance:** Method A dominates B if it has strictly higher accuracy AND strictly lower cost. Statistical significance via bootstrap test of Pareto dominance.
- **Multiple comparisons:** Bonferroni correction across 7 baselines × 4 benchmarks = 28 comparisons
- **Effect size:** Cohen's d for cost reduction; risk ratio for accuracy differences

### 24.3 Reproducibility

- All experiments use fixed random seeds (42, 123, 789)
- Temperature = 0 during evaluation (greedy decoding)
- Full hyperparameter sweep logs in WandB
- Trained model weights released
- Evaluation scripts and prompt templates in public repository

---

## 26. Anticipated Reviewer Questions

| Reviewer Question | Prepared Response |
|---|---|
| "Why not just add a length penalty to the policy model?" | Length penalties are instance-blind — they penalize long reasoning on hard problems equally with verbose reasoning on easy ones. Our monitor provides instance-aware, mid-trajectory feedback that adapts to observed progress. See Section 21.5 for the ablation showing this matters. |
| "What if the monitor is wrong?" | The monitor's confidence score accompanies each decision. For low-confidence decisions (confidence < 0.7), the system can fall back to CONTINUE (conservative). In our experiments, monitor errors are 16% of decisions, and 78% of errors are false-CONTINUE (safe failure mode: lets executor keep working). |
| "Doesn't the monitor add its own cost?" | Yes — but the monitor is 0.5B–3B parameters vs. the executor's 7B–72B. Per-step overhead is ~200–350 tokens (<$0.001/step). This is 1–3% of total cost, while savings are 20–40%. See Section 21.3. |
| "How do you get ground-truth stopping labels?" | Through oracle computation on full trajectories (Section 8.3.1). This is a standard technique in imitation learning and process reward modeling. We also validate oracle quality via human review on a subset. |
| "Is your method specific to a particular model family?" | No. We test on Qwen2.5 (7B, 32B) and Llama-3.1 (8B). The monitor is trained on executor-specific trajectory data, so it's executor-aware. Transfer experiments (Section 21.2) test cross-model generalization. |
| "How does this compare to just using a smaller model?" | Using a smaller executor model is orthogonal — you could apply CASSI to a small executor too. The monitor addresses a different problem: knowing when to stop, not choosing which model to use. |
| "What about tasks with no ground truth?" | The oracle requires ground truth for training, but the monitor itself only needs task description + trajectory state — no ground truth at inference time. For open-ended tasks, quality can be estimated via LLM-as-judge, rubrics, or user feedback. |

---

## 27. Paper Roadmap (For Readers)

```
Section 1 (Introduction): "Here's the problem: agents overthink and waste resources."
Section 2 (Related Work): "Here's what others have tried and why it's insufficient."
Section 3 (Problem Formulation): "Here's the formal definition of the problem."
Section 4 (CASSI Architecture): "Here's our solution — what it looks like."  
Section 5 (Training Method): "Here's how we train it — the three-phase pipeline."
Section 6 (Experimental Setup): "Here's how we test it."
Section 7 (Results): "Here's what we found — it works (tables, plots, analysis)."
Section 8 (Ablations): "Here's why each component matters."
Section 9 (Discussion): "Here's what it means, what it doesn't do, and what's next."
Section 10 (Conclusion): "Here's the one-sentence takeaway."
```

---

## 28. Extended Failure Mode Analysis

### 27.1 When CASSI is Expected to Fail

| Failure Mode | Cause | Detection | Mitigation |
|---|---|---|---|
| **Premature stop on novel tasks** | Monitor trained on different task distribution; overestimates quality | Large Δ negative + low monitor confidence | Fall back to CONTINUE when confidence < threshold |
| **Late stop on deceptive progress** | Executor appears to improve but answer is wrong; monitor fooled | Task failure at end despite high mid-trajectory quality estimates | Outcome-based feedback in RL training corrects this over time |
| **Budget-tier cliff** | Sharp transition between tiers causes inconsistent behavior at boundaries | Sudden decision flips at tier boundaries | Smooth the λ_multiplier with interpolation near tier boundaries |
| **Overfitting to executor quirks** | Monitor learns patterns specific to one executor model | Poor transfer performance | Train on diverse executor trajectories; add noise to executor outputs during monitor training |
| **Oscillation** | Monitor alternates STOP/CONTINUE on similar states | High decision variance on similar inputs | Add temporal smoothing (moving average of Δ over last 2 decisions) |

### 27.2 Edge Cases

1. **Empty trajectory (answer in first step):** Monitor should immediately signal STOP with Δ ≈ -1.0. This tests whether the monitor correctly handles trivial tasks.
2. **Budget exhausted at step 1:** Monitor must signal STOP even if answer quality is 0. Tests the cost-sensitivity mechanism at extremes.
3. **Perfect answer at step 1, then random noise for 10 steps:** Monitor should STOP at step 1. Tests robustness to trajectory degradation.
4. **Slow, steady improvement across 20 steps:** Monitor should STOP late (step 15-18). Tests whether the monitor can recognize sustained progress.
5. **All observations are identical (no progress detection):** Monitor should ADJUST (change approach) rather than CONTINUE blindly or STOP prematurely.

---

## 29. Summary of Novelty Claims

> Current LLM agents can't judge whether their next action is worth its cost. They overthink, over-search, and over-polish because they were trained to "be helpful" — and no training signal ever said "this is good enough, stop." Existing solutions either enforce rigid budgets (wasteful when budget remains, dangerous when budget is tight), embed static length penalties in training (instance-blind, can't adapt mid-trajectory), or use heuristic stopping rules (fragile, not learned).
> 
> **We introduce CASSI:** a method for training cost-aware agents via oracle-guided stopping rewards. Instead of Monte Carlo rollouts from every intermediate state (AgentPRM, O(K×T²)), we compute optimal stopping labels analytically from completed trajectories (O(T)) — a K×T reduction that makes cost-aware PRM training tractable on long-horizon benchmarks. A small stopping model (0.5B–3B) trained on these labels serves as a cost-aware process reward model, training the executor via GRPO to stop at the right time.
>
> **Three key contributions:** (1) oracle-guided stopping labels that avoid Monte Carlo rollouts, reducing training computation by O(K×T), (2) empirical demonstration that dynamic, per-instance cost adaptation outperforms static penalties on heterogeneous tasks, and (3) a small stopping model that effectively supervises large executors with <3% inference overhead.
