# CASSI Paper Plan — Compact Research Brief

> Purpose: compact, faithful summary of `research/paper_plan.md` (v5) for literature-research /
> novelty-check subagents. Read this instead of the full 95KB plan unless you need exact details.
> Full plan: `research/paper_plan.md`. Vision: `VISION.md`.
> **Do NOT read `research/archived_*` folders — previous AI conclusions must not bias fresh analysis.**

## One-sentence idea

Train a **small stopping model** (0.5B–3B) on **oracle stopping labels computed post-hoc from
completed agent trajectories** — `t* = argmax_t [quality_t − λ · cumulative_cost_{1..t}]`, an O(T)
computation requiring zero extra rollouts — then use that stopping model's **cost-aware value
estimate Δ(s_t) as a process reward** to train the executor agent (7B–72B) via GRPO, forming a
claimed "self-reinforcing cycle."

## Problem & motivation

- LLM agents (search, multi-hop QA, SWE agents) lack *economic judgment*: no training signal for
  "good enough, stop now." Failure modes: overthinking (polish a correct answer for 10+ steps),
  runaway loops, premature stopping.
- Cited motivation evidence: "When More is Less" (Wu et al. 2025, inverted-U accuracy vs. length);
  "Over-Reasoning and Redundant Calculation" (Chiang & Lee 2024); "Don't Overthink It" (Hassid et
  al. 2025, shortest chains more accurate); "Token Economies" (Wang et al., EMNLP 2024).

## Method (CASSI) — the 5-step cycle

1. **Executor** (ReAct-style tool-using agent, 7B–32B, Qwen2.5 / Llama-3.1) runs tasks →
   trajectories with per-step state, action, cost (tokens/tools/dollars), and intermediate answer
   quality (F1 / accuracy vs. ground truth at each step).
2. **Oracle labeling (O(T), key efficiency claim):** for each finished trajectory compute
   `t* = argmax_t [quality_t − λ·cumcost_t]`; steps < t* labeled CONTINUE, ≥ t* labeled STOP; a
   continuous margin Δ_oracle(s_t) = Q_continue − Q_stop normalized to [−1, 1].
3. **Stopping model** M_θ (Qwen2.5 0.5B–3B): SFT on oracle labels (+ MSE value head for Δ) then
   GRPO fine-tuning (reward = stopping accuracy vs. t* + cost terms). Input: structured state
   summary (task, multi-dim budget state incl. tier HIGH/MED/LOW/CRITICAL with λ multipliers, last-K
   steps, answer draft, confidence/stability indicators). Output: STOP/CONTINUE/ADJUST + rationale +
   Δ ∈ [−1,1] + confidence.
4. **Executor GRPO training with the stopper as cost-aware PRM:** step reward
   `R_t = α·Δ(s_t) + β·progress + γ·format`, plus terminal task-success reward; GRPO G=8,
   group-normalized advantages.
5. **Cycle claim:** better executor → better trajectories → better oracle labels → better stopper →
   better process rewards → better executor. At inference the stopper is enforced as a controller
   (its own overhead claimed <3% of total cost).

## Claimed contributions (as stated in the plan)

1. **First self-reinforcing cost-aware training cycle** (oracle → stopper → process rewards →
   executor RL → better trajectories). Claim: no prior work closes this loop (AgentPRM lacks
   cost/stopping; CaRT lacks executor training; Ares/SeqRoute do discrete budget routing only;
   L1/Reason Efficiently/BudgetThinker are single-model penalties).
2. **Separate stopping model is *necessary*** — "representation conflict" argument: execution
   features vs. economic self-evaluation features conflict in a single model; tested via ablation
   (single-model cost-penalty GRPO vs. multi-task vs. two-model).
3. **O(T) oracle labels vs. O(K×T²) Monte Carlo PRM training** (AgentPRM needs K rollouts from
   every state; ~160 extra executions per 20-step trajectory; CASSI needs 0). Plus 3 formal
   properties: uniqueness/monotonicity of t*, λ-monotonicity (higher λ → earlier t*), oracle
   improves as policy improves.
4. **Per-instance dynamic cost adaptation beats static penalties** — H5: stopping step correlates
   with task difficulty (r > 0.5).
5. **Small stopper supervises large executor** with <3% inference overhead, 20–40% cost savings at
   iso-accuracy.

## Experiments plan

- **Benchmarks:** GAIA, WebWalkerQA (web); HotpotQA, MuSiQue (multi-hop QA); SWE-bench Verified
  (coding); MATH-500 (low-slack control); BFCL (tool selection).
- **Key baselines:** ReAct; zero-training self-eval prompt; BATS (+grid-searched variant); s1
  budget forcing; L1/LCPO; Reason Efficiently (Arora & Zanette); adaptive-α difficulty-classifier
  variant; CaRT and CaRT+cost+GRPO (primary); single-model GRPO+cost penalty; CASSI w/o
  process-reward bridge (stopper as controller only); AgentPRM-cost; BudgetThinker; ReMA-cost;
  Ares-style discrete router; oracle stopping (upper bound).
- **Metrics:** success rate, cost/task, iso-accuracy cost, iso-cost accuracy, Pareto frontier,
  stopping error |t_stop − t*|, runaway prevention, monitor overhead.
- Ablations: stopper size (0.5B–7B), SFT vs SFT+RL, budget representation, λ sweep,
  reward-bridge vs controller-only, single- vs two-model, oracle vs MC labels.

## Related work the plan already cites (verify + extend these)

BATS, BAVT, INTENT, IterResearch (budget-aware heuristic agents); CaRT, s1, DEER (stopping /
early exit); ReMA, MGV, Dolores, SupervisorAgent (monitor/meta-reasoning architectures); GRPO,
AgentPRM, CSO, CARL (agent RL / PRMs); L1, Reason Efficiently, TALE, SelfBudgeter, BudgetThinker,
DiffAdapt (token-efficient reasoning); DeepResearcher, ReTool, ToolRL (tool-use RL); SeqRoute,
Ares (budget routing / hindsight relabeling).

## What the literature research must probe

1. Has *exactly this* been done — a learned stopping/value model used as a **cost-aware process
   reward** to train the executor? Any 2025–2026 paper closing a similar loop?
2. Is the O(T)-vs-O(K×T²) framing sound — do cheaper PRM-training methods already exist (e.g.,
   implicit PRMs, TD-style labels) that undercut the efficiency claim?
3. Is the "static penalty" characterization of L1/Reason Efficiently/newer 2025-2026 methods still
   accurate, or do adaptive/difficulty-aware penalty methods already exist?
4. Do "when to stop" learned controllers already exist in adjacent literatures (adaptive retrieval,
   early-exit reasoning, anytime algorithms, rational metareasoning)?
5. Where is the defensible gap, and what would reviewers cite as missed prior art?
