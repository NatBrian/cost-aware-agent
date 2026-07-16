# Competitor Analysis — CASSI vs. the Field (July 2026)

> **Sources:** `research/lit_review/00_overview.md` (master synthesis, authoritative tier list),
> `research/lit_review/00_paper_plan_summary.md` (what CASSI proposes), the per-paper entries in
> `research/lit_review/01_*.md`–`10_*.md`, and the five reviews in `research/novelty_check/`.
> All numbers below come from those files (which read the papers at PDF level). Where a detail was
> not captured there, this document says "not recorded in our review" rather than guessing.
>
> **Note:** this analysis evaluates the v5 plan (`paper_plan.md`) against the field — that is its
> job. The refined plan that responds to everything found here is **`research/paper_plan_v2.md`**
> (see its §14 changelog for the point-by-point response).

---

## 0. How to read this document

This document compares **CASSI** — our proposed research project — against every published or
preprinted paper that overlaps with it, as of July 2026. It is written for someone still learning
AI research, so here is the minimum vocabulary you need (a full glossary is in §7):

- **Agent / executor**: an LLM that solves a task over many steps, calling tools (web search, code
  execution) between reasoning steps. The "executor" is the big model doing the actual work.
- **Trajectory**: the full recorded sequence of an agent's steps on one task (thoughts, tool calls,
  results, costs), from start to finish. One task attempt = one trajectory.
- **SFT (Supervised Fine-Tuning)**: training a model by showing it input→output examples and
  making it imitate them. The simplest way to teach an LLM a new behavior.
- **RL (Reinforcement Learning)**: training a model by letting it act, scoring the result with a
  *reward*, and adjusting the model to get more reward. **GRPO** (Group Relative Policy
  Optimization) is the popular LLM flavor: sample a *group* of, say, 8 attempts per task, and
  reward each attempt relative to the group's average — no separate value network needed. **PPO**
  and **DPO** are older/alternative RL-style algorithms.
- **Outcome reward vs. process reward**: an *outcome* reward scores only the final result (did the
  task succeed?). A *process* reward scores every intermediate step. A **PRM (Process Reward
  Model)** is a trained model that produces those per-step scores.
- **Hindsight (post-hoc) label**: a training label computed *after* a trajectory is finished, by
  looking back at what actually happened ("in hindsight, step 6 was the best place to stop").
  Cheap, because it reuses work already done. CASSI calls its hindsight labels **oracle labels**.
- **Pareto frontier**: when you trade off two things (accuracy vs. cost), the frontier is the set
  of options where you cannot improve one without hurting the other. "Pareto-dominates" = better
  on at least one axis and no worse on the other.
- **Stopper / stopping model**: a (usually small) model whose only job is to watch the executor
  and decide "STOP now" or "CONTINUE".

**Tiers.** Following `00_overview.md`, competitors are split into:
- **Tier 1** — direct overlap. These 15 papers each occupy part of CASSI's territory; reviewers
  will expect CASSI to cite them, and most must also appear as experimental baselines ("must-beat").
  Each gets a deep dive in §4.
- **Tier 2** — adjacent work. Must be cited to show awareness, but not direct rivals. One-liners
  in §5.

**Threat levels** (HIGH / MEDIUM / LOW) are carried over from the literature review files. They
answer: "how much of CASSI's claimed novelty does this paper delete, and how likely is a reviewer
to raise it?" A HIGH threat does not mean CASSI is dead — it means a specific CASSI *claim* must
be rewritten, or the paper must be beaten experimentally.

---

## 1. CASSI in one paragraph

CASSI (working title of the plan "Learning to Stop: Self-Reinforcing Cost-Aware Training for LLM
Agents", `paper_plan.md` v5) proposes to train a **small stopping model** (0.5B–3B parameters) on
**oracle stopping labels computed post-hoc from completed agent trajectories**. The oracle is
`t* = argmax_t [quality_t − λ · cumulative_cost_{1..t}]` — for each finished trajectory, find the
step where quality-minus-cost peaked; everything before is labeled CONTINUE, everything after
STOP. This is an O(T) computation (one pass over a length-T trajectory) claimed to need **zero
extra rollouts**. The stopper is trained with SFT then GRPO, and outputs a STOP/CONTINUE decision
plus a continuous **cost-aware value margin Δ(s_t)** (how much better continuing is than stopping,
in [−1, 1]). CASSI's centerpiece move is then to use Δ(s_t) as a **per-step process reward to
train the large executor agent (7B–72B) via GRPO** — the "reward bridge" — claiming a
self-reinforcing cycle: better executor → better trajectories → better oracle labels → better
stopper → better process rewards → better executor. At inference the stopper doubles as a
controller enforcing stops, with claimed <3% overhead and 20–40% cost savings at iso-accuracy.
Planned benchmarks: GAIA, WebWalkerQA, HotpotQA, MuSiQue, SWE-bench Verified, MATH-500, BFCL.

---

## 2. The competitive landscape at a glance

### 2.1 The 2×2 map

Sort every relevant paper along two questions: (1) does it **train the executor's policy**, or
only control a frozen model at inference time? (2) is its signal **cost-aware** (does the money or
token budget enter the objective), or quality-only? From `00_overview.md`:

```
                      Is the signal COST-AWARE?
                       no                yes
                ┌──────────────────┬──────────────────────────────┐
 Inference-time │ DEER, Dynasor,   │ SupervisorAgent (ICLR'26),   │
 control only   │ Certaindex, ESC, │ BATS, BAVT, INTENT, VoI-     │
 (frozen policy)│ Thought Calib.,  │ budget (2605.05701), Ares,   │
                │ LYNX, TERMINATOR,│ SeqRoute, TAB, BAGEN,        │
                │ OS-Pruner        │ Plan-and-Budget, CoRL        │
                ├──────────────────┼──────────────────────────────┤
 Trains the     │ AgentPRM ×2,     │ OTC-PO, SlimSearcher, EAPO,  │
 policy         │ Agent-RRM, MaR,  │ AdaTIR, ALP, DAST, LASER-D,  │
 (RL / DPO)     │ PRIME, GiGPO,    │ AdaCtrl, HAPO, RaM (VOC),    │
                │ SPA-RL, DASH,    │ Agent-Omit, CTA, DAS, SAGE   │
                │ RePro, SWEET-RL  │   → all OUTCOME-level cost   │
                └──────────────────┴──────────────────────────────┘
                                          ▲
        EMPTY CELL: a LEARNED stopping-VALUE model (cost-aware, per-step)
        used as a PROCESS reward to TRAIN the executor — with the loop
        oracle → stopper → process reward → executor measured over iterations.
```

How to read the map: the top row is packed with stoppers, routers, and monitors that never train
anything (or only train themselves); the bottom-right cell is packed with RL methods whose cost
signal is a single number attached to the *whole trajectory* (outcome-level). Nobody sits in the
intersection CASSI targets: a **trained, per-step, cost-aware stopping value used as a process
reward to train the executor**.

The caution: **every component of CASSI now exists somewhere.** The hindsight stop label exists
(TERMINATOR, OS-Pruner), the small trained per-step controller for agents exists (Ares, TAB), the
cost-aware agent RL exists (OTC-PO, SlimSearcher), the trained-monitor-feeds-executor-GRPO bridge
exists (Agent-RRM, quality-only), the post-hoc-label-as-GRPO-advantage trick exists (DASH), and
the iterated PRM↔policy loop exists (AgentPRM). CASSI's novelty is the *composition*, not any
single part — and the paper must say so honestly.

### 2.2 What Tier 1 vs. Tier 2 means here

- **Tier 1 (15 papers, §3–§4):** each one overlaps CASSI on at least two of the five ingredients
  {cost objective, stopping decision, separate trained stopper, executor training, agents}. All
  must be cited; most must be baselines or explicit ablation targets.
- **Tier 2 (§5):** overlaps on one ingredient, or provides theory/evidence CASSI builds on. Cite,
  usually without running.

---

## 3. Master comparison table (Tier 1)

Legend: "Trains executor?" = does the *large task-solving model* get updated? "Stopping decision?"
= does the method explicitly decide stop-vs-continue mid-task? Cells are deliberately terse — full
explanations in §4.

| Paper (Year, Venue) | Problem it solves | Approach in ~5 words | Trains executor? | Cost-aware? | Stopping decision? | Separate small model? | Key benchmark + result | Biggest weakness | CASSI's delta |
|---|---|---|---|---|---|---|---|---|---|
| OS-Pruner (2026, preprint) | Overlong chain-of-thought | Optimal-stopping-trained CoT pruner | No (frozen) | Yes (acc − λ·tokens) | Yes | Yes (linear head + 2 layers) | Math CoT: −20–60% length, minimal acc loss | Single-model math, token cost only | Agents, multi-dim cost, reward bridge |
| TERMINATOR (2026, preprint) | Reasoning past answer arrival | Hindsight exit labels, small probe | No (frozen) | No (no cost in label) | Yes | Yes (1-block probe) | −14–55% CoT; Pareto on 14/16 pairs | Label ignores correctness & cost | Cost-aware label semantics + executor training |
| DASH (2026, preprint) | GRPO rewards overthinking | Post-hoc segment advantages in GRPO | Yes (GRPO) | Partial (length escalation, no λ) | Training-side only | No | AIME25 50.8 vs GRPO 45.4 | Needs ground truth per checkpoint; math only | Learned stopper as reward source; agents |
| Ares (2026, preprint) | Fixed reasoning effort wasteful | Small per-step effort router | No (frozen) | Yes (per-level cost reward) | No (effort levels only) | Yes (Qwen3-1.7B) | −35–45% tokens at iso-success (TAU/WebArena) | Cannot terminate; executor untouched | Stop semantics, budget state, Δ, co-training |
| TAB (2026, preprint) | Per-turn budget allocation | Small GRPO turn-budgeter | No (frozen solver) | Yes (acc − λ·overage) | No (sizes, never stops) | Yes (Qwen3-1.7B) | Up to −35–40% tokens, math multi-turn | Math sub-questions; no stopping | Stopping + heterogeneous costs + oracle labels |
| BAGEN (2026, preprint) | Do agents know remaining budget? | Replay-relabeled budget estimator | No (estimator only) | Yes (multi-dim budgets) | Partial (aborts failures) | Yes-ish (7B estimator) | Early-stop saves 28–64% tokens on failures | Aborts failures, not overthinking; 47% calibration | Optimal stopping of successes; reward bridge |
| Agent-RRM (2026, preprint) | Sparse agent rewards | Trained RM critiques → GRPO | Yes (GRPO) | No | No | No (8B RM = policy size) | GAIA text 43.7; WebWalkerQA 46.2 | No efficiency dimension | Cost-aware per-step signal; small stopper; controller |
| SlimSearcher (2026, preprint) | RL efficiency collapse | Cascading multiplicative reward gates | Yes (SFT+GRPO) | Yes (group-relative gates) | No | No | GAIA rounds −48.4%, acc 0.682→0.709 | Outcome-level; no stopper/controller | Learned stopping model, process rewards, budget state |
| OTC-PO (2025, preprint) | Tool over-calling | Hindsight optimal tool-call reward | Yes (PPO/GRPO) | Yes (tool-call count) | No | No | −68.3% tool calls, +215.4% tool productivity | Count-only cost; trajectory-level | Per-step Δ, continuous quality−λ·cost, stopper |
| AgentPRM — Cornell (2025, preprint) | Dense turn-level agent supervision | Pooled return-to-go PRM loop | Yes (Online DPO) | No | No | Yes (3B, = policy size) | ALFWorld 88.1%; BoN 91.0% (> Claude-3.5's 76.1%) | No cost/stopping; one benchmark | Cost+stopping semantics; CASSI must fix its own mischaracterization |
| AgentPRM — Fudan (2026, WWW 2026) | Cheap agent PRM labels | TD+GAE promise/progress PRM | Yes (PPO) | No | No | Yes (0.5B–3B PRM) | WebShop beam 76.0; ~8× compute-efficient | Bootstrap bias; no cost | Ground-truth-anchored economic labels; kills CASSI's label-cost claim as stated |
| SupervisorAgent (2026, ICLR 2026) | Runtime token waste | Training-free runtime monitor | No | Yes (cost-motivated) | No (nudges, never stops) | Yes (prompted, untrained) | GAIA −29.7% tokens at identical pass@1 | 15.45% overhead; no learning | Trained stopper + stop decisions + reward bridge; must beat on Pareto |
| CaRT (2025, preprint) | Mis-timed termination | Counterfactual SFT termination | SFT only, no RL | No (γ discount only) | Yes | Yes (medical: Qwen2.5-3B) | Optimal termination rate 0.32 vs 0.04 base | No explicit cost; RL variant unhelpful | λ·cost oracle, Δ margin, budget state, GRPO bridge |
| ALP (2025, NeurIPS 2025 spotlight) | Uniform penalties ignore difficulty | Solve-rate-scaled length penalty | Yes (GRPO) | Yes (adaptive token penalty) | No | No | ~50% token cut; 5.35× hard/easy adaptation | Single-turn math; outcome-level | Kills "static penalty" strawman; CASSI repositions on agent/step/stopper axes |
| Rational Metareasoning — De Sabbata (2024, preprint) | CoT cost on every query | VOC reward in Expert Iteration | Yes (Expert Iteration) | Yes (utility − γ·tokens) | No | No | 23–32% fewer tokens than STaR, equal acc | Not agentic; trajectory-level | Agentic per-step stopping + trained stopper + loop |

---

## 4. Per-competitor deep dives

### 4.1 OS-Pruner (Ehab, El Gadarri, Farias, Jozefiak, Moallemi; 2026; preprint, MIT/Columbia; arXiv 2607.11089)

- **Background & motivation.** Large reasoning models (LRMs) produce very long chains-of-thought
  (CoT — the model "thinking out loud" before answering). Existing early-exit methods treat
  "should I stop?" as a correctness classification with a hand-tuned threshold. The authors argue
  stopping is *naturally a sequential decision problem*: continue only while the expected quality
  improvement outweighs the token cost.
- **Problem definition.** Input: a frozen reasoning model generating a CoT, observed at paragraph
  boundaries. Output: a stop/continue decision at each boundary that maximizes an explicit
  reward `r(y_≤i | x) = A(y_≤i | x) − λ·L(y_≤i)` — per-prefix accuracy minus λ times token count.
  This is *the same functional form* as CASSI's oracle objective.
- **Research goal.** Show that a trained optimal-stopping policy beats threshold-based
  correctness classifiers for pruning CoT, with a single knob λ sweeping the accuracy–length
  trade-off.
- **Approach & architecture.** A **separate lightweight stopping policy** attached to the frozen
  base model: a linear head on the last hidden state plus light fine-tuning of the last 2
  self-attention layers, invoked at paragraph boundaries. Labels: force a final answer after
  *every* reasoning step of frozen-model traces, grade it, precompute per-prefix rewards once
  (no repeated on-policy rollouts — mirroring CASSI's O(T) labeling claim). Decision rule in
  value-function form: stop when current accuracy A ≥ the continuation value C_λ — the same
  quantity CASSI calls the margin Δ = Q_continue − Q_stop. **Theorem 1**: for any λ and any K>0
  there exists a stopping problem where the optimal stopping policy beats the *best possible*
  fixed-threshold correctness classifier by factor K — a formal argument for value-based stopping
  over threshold stopping.
- **Training methodology & RL usage.** Trains only the stopping policy by exact-expectation
  optimization of the stopping objective (possible because the frozen model's per-prefix rewards
  are precomputed). No RL on the generator; stopping is never used as a reward for the generator.
- **Experiments & evaluation metrics.** Math reasoning benchmarks over multiple base reasoning
  models; baselines include budget forcing/guidance, HALT-CoT, DEER, answer-convergence
  classifiers, FlashThink. Metrics: generation length vs. accuracy frontier.
- **Key results.** 20–60% generation-length reduction with minimal accuracy sacrifice across
  benchmarks and base models; still adds gains on top of a brevity-fine-tuned model; one scalar λ
  gives fine-grained frontier control.
- **Limitations & weaknesses.** Single-model math CoT only; token cost only (no tools, dollars,
  or multi-dimensional budgets); pure inference-time plug-in — never feeds back into generator
  training; probe is tied to the base model's hidden states, so it is not executor-agnostic.
- **How CASSI differs.** OS-Pruner occupies the cost-aware optimal-stopping *formulation* ground:
  same `quality − λ·cost` argmax labels, same value-margin stopping, same λ-frontier sweeps —
  CASSI's "formal properties" section would partially duplicate its theory. CASSI's deltas: agent
  trajectories with heterogeneous (token/tool/dollar) costs, budget-conditioned inference, and —
  decisively — the reward bridge that trains the executor. **To defend:** cite prominently;
  position CASSI's theory as an extension to agents; cite OS-Pruner's Theorem 1 *in support* of
  value-based over classifier stopping; do not claim the objective itself as new. Threat: HIGH.
- **Future work they name.** Not recorded in our review.
- **Research gap left open.** A cost-aware stopping value that leaves the math-CoT sandbox and
  changes how the executor itself is trained.

### 4.2 TERMINATOR (Nagle, Saydaliev, Garbaya, Gastpar, Makkuva, Kim; 2026; preprint v2, UT Austin/EPFL; arXiv 2603.12529)

- **Background & motivation.** Reasoning models keep "thinking" long after their final answer has
  logically arrived in the CoT. Fixed token budgets and calibrated confidence thresholds are
  brittle because optimal CoT length varies by task and model.
- **Problem definition.** Input: completed CoTs of a frozen LRM. Define the **hindsight-optimal
  reasoning length (HORL)**: the earliest position in a completed CoT at which the model's own
  final answer first logically arrived — a purely retrospective label needing no ground truth.
  Output: a per-token binary prediction "has the final answer arrived?" used to exit early.
- **Research goal.** Build an optimal-exit-label dataset from completed traces at scale, train a
  lightweight stopper on it, and Pareto-dominate training-free early exits.
- **Approach & architecture.** An LLM-based Extract–Identify–Verify pipeline (Qwen3-30B-A3B)
  locates the answer's earliest span. The stopper is a **separate binary probe**: one transformer
  block (weights copied from the LRM's final block) plus a prediction head on final-layer hidden
  states, trained with class-weighted cross-entropy. At inference, a sliding window over the 10
  most recent predictions injects the end-of-thinking token on majority vote (threshold 0.7) — no
  per-dataset calibration. Also documents observable signatures of answer arrival (confidence
  spikes; shifts in "thinking token" frequencies).
- **Training methodology & RL usage.** Trains only the probe on hindsight labels; LRM frozen; no
  RL anywhere; stopping never used as a generator reward.
- **Experiments & evaluation metrics.** Qwen3-8B/14B, Ministral-3-8B/14B-Reasoning; train mix
  AIME 1983–2024 + MATH + OpenCoder-SFT + OpenScience; eval on MATH-500, AIME 2025, HumanEval,
  GPQA vs. Vanilla, NoThinking, DEER, Dynasor, Thought Calibration.
- **Key results.** 14–55% average CoT-length reduction across the four datasets; >2× inference
  latency reduction; defines the Pareto frontier on 14/16 (model, benchmark) pairs; best or
  second-best on 28/32 metrics.
- **Limitations & weaknesses.** "Optimal" means first arrival of the model's *own* answer — a
  self-consistency proxy with no correctness and no cost in the label; single-model CoT;
  inference-only; probe tied to base-model hidden states.
- **How CASSI differs.** TERMINATOR directly precedents CASSI's headline trick — "oracle stopping
  labels computed post-hoc from completed trajectories, no extra rollouts → train a small
  stopper." The deltas are label semantics (first-answer-arrival, effectively CASSI's oracle with
  λ→0 and binary quality, vs. quality−λ·cost argmax), the state space (CoT tokens vs. agent
  tool-use trajectories with budget state), and the total absence of executor training. **To
  defend:** cite; stop presenting "post-hoc exit dataset + small learned stopper" as novel on its
  own; lead with cost-awareness and the training bridge. Threat: HIGH (for the labeling-framing
  contribution).
- **Future work they name.** Not recorded in our review.
- **Research gap left open.** Hindsight stop labels that encode an explicit quality–cost
  trade-off, and any path from the stopper back into training.

### 4.3 DASH (Lee, Dai, Zhou, Slavin, Zhang, Sahu, Campbell; 2026; preprint, Capital One; arXiv 2607.00482)

- **Background & motivation.** Overthinking is not just length: even length-controlled, incorrect
  traces show more unproductive self-reflection (hedging, abandoning approaches,
  self-contradiction). Standard GRPO broadcasts one scalar advantage to *all* tokens of a rollout,
  so a trace that found the right answer and then drifted away gets its good prefix punished
  together with its bad suffix — "GRPO's blind spot."
- **Problem definition.** Input: GRPO rollouts on math problems with ground-truth answers. Detect
  intermediate answer commitments (e.g., `\boxed{}`), compare each to ground truth, find **answer
  drift** (correct → incorrect). Output: segment-level advantages that reward the productive part
  and increasingly punish continuation past the correct answer.
- **Research goal.** Fix GRPO's uniform credit assignment so models learn *when to stop* without a
  separate stopper — explicitly framed as a cheap alternative to PRM step labels.
- **Approach & architecture.** No separate model, no inference-time stopper — a post-hoc proxy
  inside training. **DASH (Drift-Aware advantage SHaping)**: split each rollout into segments
  bounded by answer checkpoints; positive segments get +|A|·α₊; negative segments get −|A|·α₋·w(t)
  with an *escalating* length penalty w(t) that grows the longer the model continues past the
  correct answer (capped at w_max). Drift traces also get a shaped reward that improves as the
  post-drift tail shrinks. In words: "the longer you continue past a correct answer, the worse."
- **Training methodology & RL usage.** This *is* a training-side stopping signal: segment-level
  stop/continue quality directly shapes GRPO advantages for the reasoning policy (16 rollouts per
  problem, 4×8 H100). No learned stopper — labels come from ground-truth checks on intermediate
  answers.
- **Experiments & evaluation metrics.** Llama-3.1-Nemotron-Nano-4B (a high-drift model), 16.5K
  OpenR1-Math problems; eval OlympiadBench, AMC23, AIME24, AIME25 (avg@32) vs. GRPO, DR-GRPO, and
  GRPO + brevity bonus.
- **Key results.** On AIME25 (highest drift prevalence): DASH 50.8% vs. GRPO 45.4% vs. base 46.1%
  — plain GRPO *degrades* the base model (−0.7) while DASH improves it (+4.7). Averages: DASH
  67.22 vs. GRPO 66.83, but DR-GRPO wins the average at 68.38. DASH traces are 11.8% longer yet
  show the lowest overthinking-signal profile; removing the escalating penalty costs −2.2 AIME25.
- **Limitations & weaknesses.** Needs ground truth at every intermediate checkpoint (verifiable
  domains only); same-model, math-only, no tools or agents; no explicit cost/λ (drift, not
  dollars); no inference-time controller; DR-GRPO beats it on average.
- **How CASSI differs.** DASH is the closest existing instance of "stopping-relevant signal used
  as a process reward inside GRPO" — it partially occupies CASSI's reward-bridge contribution, but
  from ground-truth drift checks rather than a learned cost-aware stopping model, with no cost
  formalism, no agent costs, and no loop. It also independently validates CASSI's oracle
  intuition (post-t* continuation should be penalized increasingly). The novelty reviewers called
  it "the most dangerous single citation": it proves the hindsight-labels-as-GRPO-advantages
  bridge works *without any learned stopper*. **To defend:** implement a "DASH-for-agents /
  direct-oracle-as-advantage" baseline as a P0 ablation. If direct shaping matches CASSI, the
  stopper is dead weight; if CASSI wins (generalization to unlabeled states, inference-time
  control, transfer across executors), the two-model design earns its complexity. Threat: HIGH.
- **Future work they name.** Not recorded in our review.
- **Research gap left open.** Whether a *learned, cost-aware* stopping signal beats direct
  ground-truth advantage shaping — and whether any of this transfers to tool-using agents.

### 4.4 Ares (Yang, Hou, Wei, Bao, Chang; 2026; preprint, UCSB + Accenture; arXiv 2603.07915)

- **Background & motivation.** "Thinking" LLM agents accrue reasoning tokens at *every* step of
  multi-turn trajectories. Modern APIs expose fixed effort levels (high/medium/low), but a fixed
  level is wasteful on easy steps and damaging on hard ones (gpt-oss-20b loses ~20% accuracy when
  forced from high to low everywhere).
- **Problem definition.** Input: interaction history + current observation at every agent turn.
  Output: the *lowest sufficient* reasoning-effort level e_t ∈ {low, mid, high} for that step,
  configuring the frozen agent's thinking mode. Objective: maximize E[V(τ,x) − λ·Σ_t cost(e_t)] —
  task value minus λ times summed per-step effort cost.
- **Research goal.** Show a small learned router can select per-step reasoning effort for a large
  frozen agent, cutting tokens massively at iso-success.
- **Approach & architecture.** A lightweight router (Qwen3-1.7B) — a **per-step, mid-trajectory
  controller** — reads the history each turn, outputs a 3–5 sentence rationale plus a discrete
  effort label; the frozen executor's KV cache is preserved across levels (unlike model routing).
- **Training methodology & RL usage.** Three-phase hindsight pipeline: (1) sample N successful
  trajectories at high effort, keep the most concise as reference; (2) for each step and each
  effort level, re-execute that single step K=3 times and mark the level "sufficient" if it
  reproduces the reference action in ≥M of K trials; the label is the cheapest sufficient level;
  (3) SFT on teacher rationales, then **GRPO on the router** (executor frozen) with reward =
  task-success (+5) + per-turn effort costs (−0.2/−0.5/−1.0 for low/mid/high, applied only to
  successful trajectories to avoid "fail cheaply" degeneracy) + format term.
- **Experiments & evaluation metrics.** TAU-Bench (retail/airline), BrowseComp-Plus (deep
  research), WebArena; backbone gpt-oss-20b; accuracy and total/thinking tokens.
- **Key results.** TAU-Retail: 54.8% accuracy = fixed-High's 54.8% at 652k vs. 1007k tokens
  (−35%); BrowseComp-Plus 41.3 vs. 42.7 at −42% tokens; WebArena 46.5 vs. 45.0 (*better* than
  High) at −45% tokens; abstract claims up to 52.7% reasoning-token reduction; beats
  prompting-based routing using GPT-5 or Gemini-3 Pro as the router.
- **Limitations & weaknesses.** Effort levels are the only knob — the router **cannot terminate
  the episode**, switch models, or alter budgets; labeling needs ~9 extra single-step executions
  per step (cheaper than full-rollout Monte Carlo but *not* O(T)); per-level costs are hand-set;
  no explicit remaining-budget state; executor untouched.
- **How CASSI differs.** Ares is **the single closest system to CASSI's controller half**: small
  model (1.7B), per-step mid-trajectory decisions, hindsight labels from completed trajectories
  ("minimum sufficient effort" ↔ CASSI's t*), SFT→GRPO recipe, agentic benchmarks, a
  quality-minus-λ·cost objective. What Ares does not do: stop/continue decisions, budget-state
  conditioning, a continuous value margin Δ, and — decisively — its router never trains the
  executor: no process-reward bridge, no cycle. **To defend:** implement the plan's "Ares-style
  discrete effort router" baseline *faithfully* (the plan currently understates Ares as
  "trial-and-error" — it actually uses hindsight labels); differentiate on stopping semantics,
  budget state, O(T) labels, and executor co-training. Threat: HIGH (anticipates CASSI's
  contributions #4 and #5).
- **Future work they name.** Not recorded in our review.
- **Research gap left open.** A controller that can *end* the episode — and one whose judgment is
  distilled back into the executor rather than bolted on forever.

### 4.5 TAB (Jali, Nayak, Joshi; 2026; preprint v2, CMU; arXiv 2604.05164)

- **Background & motivation.** Token-efficiency RL was single-turn; in multi-turn reasoning,
  early verbosity compounds serving cost, and budget decisions have temporal dependency and
  delayed feedback — a *sequential compute allocation* problem.
- **Problem definition.** Input: conversation history + current sub-question at each turn. Output:
  a per-turn token budget b_t ∈ {256, 512, 1024, 2048, 4096} for a **frozen solver**, under a
  global per-problem budget B. Formulated as a multi-objective Markov decision process.
- **Research goal.** Show a separate small budgeter policy beats static budgets and LLM-judge
  budgeting on the accuracy–token frontier for multi-turn math.
- **Approach & architecture.** A **separate small budgeter** π_φ (Qwen3-1.7B, LoRA rank 64)
  observes the history and picks each turn's budget for a frozen L1-Qwen3-8B-Exact solver (chosen
  precisely because it obeys prompted budgets). Terminal reward `r = acc(x) − λ·max(0, Σ_t b_t −
  B)` — accuracy minus a hinge penalty on exceeding the global budget (λ = 0.001). A variant
  conditions on all past+future sub-questions when a plan exists.
- **Training methodology & RL usage.** GRPO on the 1.7B budgeter only (solver and question
  decomposer frozen), 125 steps, batch 64, MATH Level-5 problems; the same trajectory advantage
  is assigned to all turns (no value network — they note the credit-assignment problem and
  sidestep it).
- **Experiments & evaluation metrics.** MATH-500, AMC23, MATH Level-5, OlympiadBench, AIME25;
  baselines: static per-turn budgets, off-the-shelf LLM-judge budgeters.
- **Key results.** Up to **35% token savings** at maintained or improved accuracy (macro average
  over five benchmarks); the All-SubQ variant up to **40%**; superior accuracy–token frontier
  across all benchmarks.
- **Limitations & weaknesses.** Solver frozen — no executor training, no loop; allocates budgets
  but never decides to *stop* (turn count fixed by the decomposition); math sub-questions, not
  tool agents; terminal reward only; no oracle labels — the budgeter learns by pure RL
  exploration over budget choices.
- **How CASSI differs.** Structurally the closest paper in the token-efficiency area: a small
  RL-trained economic controller (1.7B) supervising a larger (8B) reasoner per step under a
  cost–accuracy objective. CASSI's deltas: stopping (not just sizing), heterogeneous costs,
  post-hoc O(T) oracle labels instead of pure RL exploration, the stopper-as-process-reward
  bridge, and the closed loop. **To defend:** cite and differentiate explicitly; TAB is the
  natural "budget-sizing-without-stopping" contrast. Threat: HIGH (for "small model supervises
  large model per-step" novelty).
- **Future work they name.** Not recorded in our review.
- **Research gap left open.** Sequential compute allocation that includes termination as an
  action, on tasks with real tool costs.

### 4.6 BAGEN (Lin, Wang, et al.; 2026; preprint, Northwestern + UMich + Cornell + Stanford + UT Austin et al.; arXiv 2606.00198)

- **Background & motivation.** Agent cost is usually measured only after the fact. BAGEN asks: can
  agents estimate *mid-execution* how much budget they still need, and whether the task is even
  finishable — "budget awareness"?
- **Problem definition.** At every turn k, the agent must predict an interval [R̂_lo, R̂_hi] over
  the remaining budget needed, or declare `impossible`. Budgets are both internal (tokens) and
  external, multi-dimensional (dollars, weeks, warehouse item-weeks). Sub-capabilities scored:
  feasibility prediction (macro-F1), early failure detection (Fail-F1), interval calibration
  (coverage × tightness).
- **Research goal.** Benchmark whether frontier agents are budget-aware, and test whether the
  capability can be *trained in*.
- **Approach & architecture.** A **rollout-replay protocol**: record an unconstrained rollout,
  then re-query the agent on every logged prefix and score its predictions against realized
  remaining cost — i.e., post-hoc relabeling of completed trajectories, no extra task rollouts
  (the same data-reuse trick as CASSI's oracle). Then trains budget awareness into Qwen2.5-7B.
- **Training methodology & RL usage.** SFT + RL of the budget estimator / early-stop behavior
  (not of task execution); RL without an SFT warm-start collapses; a combined reward prevents
  collapse.
- **Experiments & evaluation metrics.** Sokoban (2,500-token cap), Search-R1 (3,500), SWE-bench
  (160 turns), Warehouse (real enterprise supply-chain data, 3 coupled budgets); 5 frontier
  models; 128 rollouts/model.
- **Key results.** Budget awareness is decoupled from task skill (success vs. interval hit rate
  correlate only r≈0.35). All 20 model-environment pairs are systematically over-optimistic: on
  failed trajectories, models still predict >70% feasibility after 60% of the budget is spent;
  alarms fire only in the final 20%. An **early-stop policy keyed on `impossible` predictions
  saves 28–64% of tokens on failed trajectories at 1.6–4.2 points success cost.** SFT lifts
  Qwen-7B feasibility accuracy from 25.5% to ≈90%; interval calibration caps at 47% even after
  SFT+RL.
- **Limitations & weaknesses.** Estimation studied offline (replay), not as an online controller
  (explicitly left to future work); trains the estimator only — never bridges into executor
  training; feasibility, not marginal value of continuing (no quality−cost trade-off);
  calibration remains poor.
- **How CASSI differs.** BAGEN is the closest published relative of CASSI's
  *oracle-from-completed-trajectories* move: replaying prefixes of finished rollouts to supervise
  a stopping/feasibility signal, then SFT+RL-ing a small Qwen on it — and its 28–64% savings
  bracket CASSI's promised 20–40%. Differences: labels are realized-cost/feasibility rather than
  argmax_t[quality − λ·cumcost]; its early stopping mainly aborts *failing* runs, not successful
  runs that overthink; no Δ margin, no process-reward bridge, no executor RL, no cycle. **To
  defend:** cite; adopt or compare against its evaluation protocol; be explicit that CASSI stops
  *successful overthinking*, which BAGEN does not address. Threat: HIGH (on the trained
  stopper/monitor component and its evaluation).
- **Future work they name.** Deploying the estimator as an online controller (recorded in our
  review as "left to future work").
- **Research gap left open.** Turning budget estimates into an *optimal stopping* rule for
  trajectories that are succeeding, and letting that rule teach the executor.

### 4.7 Agent-RRM / Reagent (Fan et al.; 2026; preprint v2; arXiv 2601.22154)

- **Background & motivation.** Agentic RL leans on sparse outcome rewards that cannot
  differentiate intermediate reasoning quality; step-level PRMs are annotation-hungry and easy to
  hack; pairwise reward models give no actionable guidance.
- **Problem definition.** Input: a full agentic trajectory (reasoning + 6 tools: search, browse,
  Python, file, image, audio). Output: a structured judgment — `<think>` (analysis), `<critique>`
  (targeted flaws such as missing browse steps or tool inefficiency), and `<score>` ∈ [0,1] —
  produced *without ground truth*.
- **Research goal.** Build a trained "reasoning reward model" for agents and test three ways of
  plugging it into an executor.
- **Approach & architecture.** **Agent-RRM** (initialized from Qwen3-8B; SFT on 28K then GRPO on
  90K judgment examples; training trajectories from a model ensemble, annotated by GPT-OSS-120B).
  Executor "Reagent" (Qwen3-8B; SFT 55.6K then GRPO on 709K). Three integrations: **Reagent-C** —
  inference-time critique→refine (policy frozen); **Reagent-R** — GRPO with reward R = R_rule +
  λ·R_model (λ=0.3), the RM score added to the rule-based outcome reward; **Reagent-U** —
  critiques generate refined rollouts pooled with the originals under unified advantages.
- **Training methodology & RL usage.** Yes — a separate *trained* monitor's scalar score (and
  critiques) directly shape GRPO training of a tool-using executor. Purely quality-based;
  trajectory-level, not per-step.
- **Experiments & evaluation metrics.** GAIA (text + full), WebWalkerQA, HLE, xbench; HotpotQA,
  2Wiki, Bamboogle, MuSiQue; AIME24/25, MATH500, GSM8K. Hardware: 8×A800.
- **Key results.** Reagent-U (8B): GAIA text average **43.7** (59.0/38.5/16.7 by level) and
  WebWalkerQA **46.2**, vs. 34.0/43.5 without the RRM; HLE 10.8; Bamboogle 76.8; AIME24 60.0.
  Reagent-R alone: GAIA 36.9. The λ sweep plateaus at λ∈[0.2,0.4] and *declines* at 0.5 —
  over-weighting the monitor's signal at the expense of the outcome reward hurts.
- **Limitations & weaknesses.** The RM is the same size as the policy (8B/8B — nothing "small"
  about it); **no efficiency dimension at all** — nothing stops the agent from spending more;
  trajectory-level credit only; RM labels distilled from a 120B annotator, not from
  hindsight-computable quantities.
- **How CASSI differs.** Agent-RRM already occupies the bridge CASSI wants: separate trained
  monitor → reward → executor GRPO, on GAIA and WebWalkerQA — CASSI's exact benchmarks. CASSI's
  remaining novelty on this axis is strictly the *content* of the signal (cost-aware stopping
  value from O(T) oracle labels, per-step rather than per-trajectory) and the monitor's second
  job as an inference-time controller. **To defend:** cite, compare, and ideally run
  "Agent-RRM + cost term" as a baseline (it naturally fills the plan's "AgentPRM-cost" slot). Its
  λ-plateau finding is also a useful warning for CASSI's own reward-mixing coefficient α.
  Threat: HIGH.
- **Future work they name.** Not recorded in our review.
- **Research gap left open.** A monitor→reward→executor bridge whose signal knows what anything
  *costs*.

### 4.8 SlimSearcher (Xie et al.; 2026; preprint, Zhejiang University + Ant Group; arXiv 2606.07074)

- **Background & motivation.** Deep-research agents trained with accuracy-only rejection sampling
  or RL develop "blind tool dependency" and "performative reasoning" (redundant search loops) —
  an "efficiency collapse" where RL scales *up* search rounds to force correctness.
- **Problem definition.** Input: web-research agent trajectories. Output: a trained agent that
  follows the **Minimal Necessary Path (MNP)** — the fewest tool calls and tokens that still get
  the right answer.
- **Research goal.** Make deep-research agent RL cost-aware without the brevity bias of fixed
  penalties, and Pareto-improve accuracy and cost on GAIA-class benchmarks.
- **Approach & architecture.** Two stages. (1) **Pareto-efficient SFT filtration**: from 13,863
  seed trajectories, keep per query the correct trajectory maximizing r_tool × r_len. (2) **RL
  with multiplicative cascading gates**: R_final = r_correct · r_tool · r_len. Gate 1 is binary
  correctness (zero reward if wrong — anti reward-hacking). Gate 2, "Adaptive Efficiency
  Anchoring": compare the trajectory's tool cost C to the *empirical minimum C_min within the
  sampled GRPO group*, δ = (C − C_min)/(C_min + ε), mapped to a bounded multiplier in (0,1].
  Gate 3 does the same for token length. The anchor moves with each group — explicitly designed
  to avoid static-penalty brevity bias.
- **Training methodology & RL usage.** SFT + GRPO on Tongyi-DeepResearch-30B and
  Qwen3-30B-A3B-Instruct; 64 H800 GPUs; live Serper + Jina Reader tools. Cost-awareness is fully
  learned; the penalty is adaptive (group-relative), not static.
- **Experiments & evaluation metrics.** GAIA, BrowseComp, XBench-DeepSearch, HLE. Metrics:
  accuracy, tool-call rounds, tokens.
- **Key results.** Tool-call rounds −17% to −58% with equal or better accuracy. On the
  Tongyi-DeepResearch backbone: GAIA rounds 20.56→10.61 (−48.4%), tokens −33.4%, accuracy
  0.682→0.709; BrowseComp rounds 63.70→47.63 with accuracy 0.410→0.447; XBench accuracy
  0.713→0.790 with rounds 14.26→5.92. A prompt-only control fails to improve efficiency.
- **Limitations & weaknesses.** Outcome-level trajectory reward — no per-step process reward, no
  stopping-decision head; the anchor is within-group relative (needs G rollouts; no explicit
  budget state or λ); uniform tool weights (no dollar costs); no controller at inference; no
  difficulty conditioning beyond data filtering.
- **How CASSI differs.** The most direct competitor on "cost-shaped agent RL for deep research,"
  on CASSI's headline benchmark (GAIA), with a 30B executor and a Pareto-improvement framing.
  Overlaps: hindsight minimal-cost anchoring (MNP ≈ an empirical cost oracle), multiplicative
  correctness gating, GRPO. Differences: no learned stopping model, no oracle stopping labels, no
  process rewards, no budget conditioning, no inference-time controller. **To defend:** CASSI's
  "first to make agent RL cost-aware" framing is dead — SlimSearcher must be a baseline, or at
  minimum its group-relative reward becomes the honest instantiation of the "single-model
  GRPO+cost penalty" ablation. Threat: HIGH.
- **Future work they name.** Not recorded in our review.
- **Research gap left open.** Per-step economic signals and an explicit stopping decision on top
  of outcome-level efficiency gating.

### 4.9 OTC-PO (Wang, Qian, et al.; 2025; preprint (under review), CUHK + UIUC + Princeton; arXiv 2504.14870)

- **Background & motivation.** RL agents optimized only for final correctness over-call tools —
  the authors call it "cognitive offloading" — and the problem *worsens* with model size. They
  want correct answers with *minimal* tool calls, introducing **tool productivity (TP)** =
  correct answers per tool call.
- **Problem definition.** For each (question, model) there is assumed to be an optimal (minimal)
  tool-call count n. Since n is unknown, approximate it **in hindsight as the minimum tool calls
  among correct trajectories in the GRPO group** (updated across epochs toward a global optimum).
  Output: a policy that hits the answer with about n calls.
- **Research goal.** A plug-in reward for PPO/GRPO that teaches tool-using LLMs to act less while
  reasoning more.
- **Approach & architecture.** Optimal Tool Call-controlled Policy Optimization. Final reward
  `r = α · r_tool · r_φ(q,y)` — a **multiplicative efficiency coefficient on the base correctness
  reward**. The r_tool schedule (cosine/sine shapes with a remap f(m,n) = 2nm/(m+n)) peaks exactly
  at m = n calls and decays for both over- and under-calling. Multiplicative by design: an
  additive bonus was "unstable and sub-optimal" (the model earns reward by dropping tools without
  being correct); with multiplication, a wrong answer zeroes the efficiency bonus, preventing
  hacking.
- **Training methodology & RL usage.** OTC-PPO and OTC-GRPO variants; search setting follows
  Search-R1 (train on NQ + HotpotQA, Qwen2.5-3B/7B-Base); code setting follows ToRL
  (Qwen2.5-Math-1.5B/7B).
- **Experiments & evaluation metrics.** QA: NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue,
  Bamboogle; math/code: AIME24/25. Metrics: exact match (EM), tool calls (TC), tool productivity.
- **Key results.** Up to **68.3% fewer tool calls and up to 215.4% higher tool productivity at
  comparable accuracy**. Example (7B, NQ): Search-R1-PPO EM 0.449 / TC 3.282 vs. OTC-PPO EM 0.446
  / TC 1.040. Out-of-domain (7B OTC-PPO) beats Search-R1-PPO EM on all 5 QA sets with far fewer
  calls. Also documents that over-reliance on tools grows with scale (3B uses fewer calls than 7B).
- **Limitations & weaknesses.** Cost = tool-call *count* only (no tokens, latency, or dollars; no
  quality–cost margin); trajectory-level coefficient (no per-step signal, no stopping decision);
  the hindsight n is undefined when no group trajectory is correct; QA/math tools only.
- **How CASSI differs.** The closest reward-design competitor in agentic RL. Shared DNA:
  economics beyond correctness, and a hindsight per-question optimum computed from completed
  trajectories at zero extra rollout cost — structurally analogous to CASSI's post-hoc t* oracle.
  Differences: no learned stopping model, no per-step process reward, no continuous
  quality−λ·cost margin, no budget tiers, no inference-time controller, single-model. **To
  defend:** cite OTC-PO *specifically* (not just generically as "cost penalties") and beat
  OTC-GRPO, not just Search-R1 — reviewers will treat it as the first "hasn't this been done?"
  citation. Threat: HIGH.
- **Future work they name.** "Extend … to more complex agentic tasks … and longer-horizon
  reasoning" (recorded verbatim in our review).
- **Research gap left open.** Per-step, multi-resource cost signals with an explicit stop
  decision — everything OTC-PO compresses into one trajectory-level coefficient.

### 4.10 AgentPRM — Cornell (Choudhury; 2025; preprint "under review"; arXiv 2502.10325)

- **Background & motivation.** How can LLM agents improve through interaction without extensive
  human supervision? Large-scale RL is impractical for long horizons with sparse rewards; PRMs
  give dense turn-level signals but are underexplored for agents acting in external environments
  (math-PRM tricks like beam search over known transitions don't transfer).
- **Problem definition.** Turn-level MDP; the PRM is the Q-function Q^π(s,a) = expected
  discounted future success. Input: on-policy rollouts. Output: a PRM used to rank and train the
  policy.
- **Research goal.** A practical framework for agent PRMs: automatic labels, iterated
  PRM↔policy training, plus variants (InversePRM from expert demos, reward shaping).
- **Approach & architecture (corrected reading — important).** Three-stage loop, iterated K=3
  times: (1) roll out the current policy asynchronously, many times per task (10k trajectories
  per iteration on ALFWorld; 70k in a reward-hacking study); store every visited (s,a) in a
  hashed dictionary and compute targets as the *average discounted return-to-go over already
  collected trajectories passing through (s,a)* (their Eq. 1) — **pooled Monte Carlo from
  complete episodes, NOT fresh rollouts from each state**; the paper explicitly rejects
  MCTS-style per-state exploration as unscalable. (2) Train the PRM with soft binary
  cross-entropy. (3) Update the policy with Online DPO against the PRM, KL-regularized
  (a conservative policy iteration argument). Also proposes **InversePRM** — process rewards
  directly from expert demonstrations, no outcome rewards.
- **Training methodology & RL usage.** PRM and policy both Llama-3.2-3B; Online DPO; Best-of-N
  (N=16) ranking at inference; a reward-shaping variant blends PRM targets with a
  reference-policy advantage. **No cost anywhere; no stopping decision** — γ discounting is the
  only implicit length pressure.
- **Experiments & evaluation metrics.** ALFWorld only (134 out-of-distribution games); baselines
  BUTLER, ReAct (gpt-4o 65.7%, claude-3.5-sonnet 76.1%), Autogen, ExpeL, Reflexion, AdaPlanner.
- **Key results.** Success 85.8% (iteration 2) → 88.1% (iteration 3); Best-of-N 91.0% — a 3B
  model beating claude-3.5-sonnet's 76.1%. Average actions 12.0–12.7 vs. 19–25 for ReAct
  baselines (efficiency emerges without being optimized). InversePRM: 82.8% after ONE iteration.
  **Reward hacking measured:** with a PRM trained on only 10k rollouts, true success falls
  82%→70% while the PRM's own score keeps rising; mitigated at 70k rollouts.
- **Limitations & weaknesses.** Single benchmark with a discrete, revisitable state space (the
  hashed-dictionary trick needs repeated (s,a) visits and would degrade to single-sample
  estimates in open-ended web/SWE settings); 10k–70k rollouts per iteration is still a big sample
  bill; no cost or stopping semantics; preprint.
- **How CASSI differs — and the correction CASSI must absorb.** This is CASSI's primary framing
  target, and the plan currently **mischaracterizes it**: AgentPRM does *not* run K Monte Carlo
  rollouts from every state (that's Math-Shepherd-style annotation), so CASSI's "O(K×T²), ~160
  extra executions per trajectory" contrast is factually wrong and reviewers who know Eq. 1 will
  reject it. The honest residual deltas: CASSI's labels are *ground-truth-anchored* (per-step
  quality vs. gold) rather than pooled returns; they are *economic* (encode λ·cost and a stopping
  margin); and they need no state-revisit structure ("0 extra rollouts and no state-revisit
  requirement vs. 10k–70k pooled rollouts"). Note also that AgentPRM's iterated loop *is* already
  a self-reinforcing cycle for quality-only rewards — CASSI's cycle novelty must rest on the
  cost-aware stopping content, not the loop shape. Its measured reward-hacking collapse (82→70)
  is a direct warning for CASSI's frozen-stopper GRPO stage. **To defend:** rewrite contribution
  3 around label *semantics*, cite the hacking result, and add mitigations. Threat: HIGH (mostly
  self-inflicted).
- **Future work they name.** Not recorded in our review (the paper is itself a
  "framework and directions" piece).
- **Research gap left open.** PRM labels that mean something economic — cost, stopping,
  marginal value — rather than pooled expected success.

### 4.11 AgentPRM — Fudan (Xi, Liao, Li, et al.; 2026; **WWW 2026** (peer-reviewed); Fudan + Ant Group; arXiv 2511.08325)

- **Background & motivation.** Same name, different paper — a collision CASSI must disambiguate.
  PRMs for agent tasks face three obstacles: steps have no clear-cut "correctness"; steps are
  interdependent (a login detour is locally regressive but globally necessary); and existing PRM
  training "depends on either expert annotations or extensive Monte Carlo-based sampling…, both
  of which are costly."
- **Problem definition.** Input: agent trajectories. Output: a PRM with two heads — **promise**
  (Q-value: expected future success) and **progress** (advantage: how much this step improved
  the position) — trained cheaply, then used for inference-time search and executor RL.
- **Research goal.** Agent PRM labels *without* per-state Monte Carlo rollouts, at a fraction of
  the compute, published with explicit label-cost accounting.
- **Approach & architecture.** Labels via **TD (temporal-difference) estimation with GAE
  (Generalized Advantage Estimation)**: sample N=16 plain trajectories per query; bootstrap
  targets from the PRM's own predictions (δ_t = r_t + γM(s_t) − M(s_{t−1}); GAE λ=0.95); iterate
  batch-sample → estimate → update. Their explicit claim: "TD-based estimation with GAE does not
  require additional rollouts from each state like MC-based method." Their MC baseline is exactly
  the O(K×T²) scheme (16 fresh rollouts from every step's successor state).
- **Training methodology & RL usage.** PRMs on Qwen2.5-0.5B/3B (policies up to 7B/8B); used for
  (a) Best-of-N and step-level beam search at inference, and (b) **PPO training of the executor
  with the PRM as reward** (BabyAI + TextCraft, Qwen2.5-3B) — more stable and higher-scoring
  than outcome-reward or per-step-value baselines.
- **Experiments & evaluation metrics.** WebShop (≤6 turns), BabyAI (≤20), TextCraft (≤20) on
  AgentGym; GSM8K transfer; baselines SFT, RFT, ORM, PVM, Math-Shepherd-style MC.
- **Key results.** Qwen2.5-3B with beam search 8×8: WebShop 76.0 vs. PVM 54.5 / ORM 57.0; BabyAI
  89.8; TextCraft 56.7 vs. ORM 43.3. Claimed **~8× more compute-efficient** than ORM/PVM at
  matched Best-of-N. **Label-cost table: MC estimation uses 1.9× (WebShop), 2.8× (BabyAI), 1.5×
  (math) more sampled tokens than TD labels — and TD still scores higher** (WebShop BoN@64 74.0
  vs. 72.0).
- **Limitations & weaknesses.** TD targets bootstrap from the PRM's own initially-wrong
  predictions (bias; needs iterative re-estimation); assumes deterministic transitions and
  sparse terminal reward for the advantage identity; short horizons (≤20 turns); no ground-truth
  per-step quality; **no cost, no stopping**.
- **How CASSI differs.** This paper, peer-reviewed before any CASSI submission, has already
  claimed "cheap agent-PRM labels" with token-cost tables — CASSI's O(T)-vs-O(K×T²) framing
  cannot be presented as first. The defensible residual: CASSI's oracle labels are
  ground-truth-anchored (per-step quality vs. gold answer — no bootstrap bias, no iterative
  re-estimation) and *economic* (they encode λ·cost and a stopping margin), neither of which
  TD+GAE provides. **To defend:** cite; differentiate on label *semantics*, not label *cost*;
  disambiguate the two AgentPRMs explicitly; consider a TD+GAE-labels arm in the label-source
  ablation (prophet-argmax vs. Snell vs. MC vs. TD). Threat: HIGH (kills the label-efficiency
  contribution as stated).
- **Future work they name.** Not recorded in our review.
- **Research gap left open.** Step labels that are anchored to ground truth and carry an explicit
  cost trade-off, instead of bootstrapped quality-only estimates.

### 4.12 SupervisorAgent (Lin et al.; 2026; **ICLR 2026** (peer-reviewed); arXiv 2510.26585)

- **Background & motivation.** Multi-agent systems waste tokens (verbose observations, repetitive
  loops) and propagate errors; prior work does post-hoc failure attribution rather than
  real-time intervention.
- **Problem definition.** Input: the live message/tool/memory traffic of a multi-agent system.
  Output: targeted runtime interventions that cut token waste without altering the base agents.
- **Research goal.** Show a lightweight, training-free runtime supervisor can cut ~30% of tokens
  at unchanged success on GAIA-class agent workloads.
- **Approach & architecture.** A meta-level Supervisor watches agent-agent, agent-tool, and
  agent-memory interactions. An **LLM-free heuristic adaptive filter** flags three trigger types
  — errors, inefficient behavior (e.g., repeated `page_down` loops), excessively long
  observations — then a prompted supervisor LLM (with its own memory and a global-trace context)
  picks one of four actions: `approve`, `provide_guidance`, `correct_observation`,
  `run_verification`. **There is no stop/terminate action** — it nudges and purifies rather than
  deciding termination. Supervisor and executor are both large prompted models (GPT-4.1
  supervising GPT-4.1); nothing is trained.
- **Training methodology & RL usage.** None. Purely inference-time; no reward ever reaches the
  executor.
- **Experiments & evaluation metrics.** GAIA validation (Smolagents testbed; GPT-4.1, also
  Gemini-2.5-pro and Qwen3-235B); GSM8k-Hard, AIME2024, HumanEval, MBPP, DROP (Qwen3-32B);
  MAS-agnostic tests on AWorld and OAgents.
- **Key results.** GAIA: **−29.68% average tokens at identical pass@1 (50.91)**, 527.76K→371.12K;
  −32.39% on Level 2; token-cost variance −63% (L2). AIME +6.67% accuracy with −18.92% tokens;
  HumanEval −23.74% tokens at slightly higher accuracy. Ablation: observation purification drives
  the efficiency (savings drop 29.68%→15.96% without it). Supervisor overhead is **15.45% of
  total tokens** (savings reported net of it); +<1.5 min latency per task. AWorld −36.54%,
  OAgents −39.36% tokens.
- **Limitations & weaknesses.** Hand-crafted heuristic triggers; the supervisor is an expensive
  frontier model; no learning, no explicit budget state, no termination decisions, no per-step
  value estimate; occasionally over-compresses contexts and drops accuracy.
- **How CASSI differs.** This is the **training-free bar CASSI must clear**: an ICLR 2026 paper
  already delivering ~30% savings at iso-accuracy on CASSI's headline benchmark — inside CASSI's
  promised 20–40% band — with zero training. CASSI's deltas: a *trained* stopper, actual
  stop/continue decisions, a per-step value estimate, the reward bridge into executor training,
  and a much lower overhead target (<3% vs. 15.45%). **To defend:** include SupervisorAgent (or a
  faithful variant) as a baseline and beat it on the cost–accuracy Pareto frontier, or
  demonstrate complementarity. If CASSI's savings land in the same 20–40% band without a Pareto
  or accuracy advantage over this free method, the 3-stage training pipeline is unjustified.
  Threat: HIGH.
- **Future work they name.** Not recorded in our review.
- **Research gap left open.** Whether *learning* (a trained stopper + trained executor) buys
  anything beyond what heuristic runtime supervision already harvests.

### 4.13 CaRT (Liu, Qu, Schneider, Singh, Kumar; 2025; preprint, CMU; arXiv 2510.08517)

- **Background & motivation.** Strategic information gathering requires knowing not just how to
  acquire information but *when to stop and commit*. Off-the-shelf LLMs badly mis-time
  termination both in multi-turn information seeking (medical diagnosis dialogues) and in long
  CoT (math).
- **Problem definition.** Objective: maximize E[Σ_t γ^t · 1{a_t = terminate} · r(x, y_t)] — pick
  the termination time that maximizes discounted final reward. Note: **cost enters only
  implicitly through the discount factor γ** — there is no explicit price on steps.
- **Research goal.** Teach termination via supervised fine-tuning on counterfactually
  perturbed trajectories ("counterfactuals" = minimally edited versions of a trajectory that
  flip whether stopping is correct).
- **Approach & architecture.** Who decides differs by domain: in medicine, an explicitly
  **separate termination model** (Qwen2.5-3B) watches a *frozen* question-asking model; in math,
  the same model decides at episode boundaries. Labels are post-hoc: each conversation/reasoning
  prefix is scored with an external success estimate (an 8B judge for diagnosis; math prefixes
  labeled "terminate" if stopping early yields higher success than continuing). Two key
  ingredients: (1) **hard-negative counterfactual pairs** — minimally perturb a
  correct-termination trajectory (swap the final Q&A pair, or delete a span of reasoning) so that
  terminating becomes wrong; (2) GPT-4o-generated verbal rationales for each decision (a
  "verbalized value function" — an informal cousin of CASSI's Δ). Train with SFT; control the
  executor only at inference.
- **Training methodology & RL usage.** SFT is the main method. A CaRT+RL variant runs GRPO *on
  the termination decision itself* (binary reward for correct terminate/continue calls) — it
  "tends towards longer conversations" and gave no gains in math. The stopping decisions are
  **never used as rewards to train the executor** (per the RL-expert review, the precise
  statement: no RL and no separate reward model — in math, CaRT's SFT does fine-tune the model
  that reasons).
- **Experiments & evaluation metrics.** Interactive medical diagnosis built from craft-MD
  (MedQA-USMLE + MedMCQA, 1,233 problems, GPT-4o-simulated dialogues; 100 ID + 200 OOD test);
  math on 2,000 DeepScaleR problems with Qwen3-1.7B, evaluated on AIME 2025. Metrics: success
  rate, success-vs-fixed-baseline difference, optimal termination rate.
- **Key results.** Medical: optimal termination rate ≈0.32 (CaRT) vs. ≈0.17 (plain SFT) vs.
  ≈0.04 (base); best success-rate-difference (≈+0.027); holds out-of-distribution. Math (AIME25):
  higher success (~33 vs. ~32.5) with ~14.5K vs. ~18K response tokens; optimal termination rate
  ≈0.37 vs. ≈0.20 base. Ablations: counterfactuals matter most; rationales improve
  generalization (probe test accuracy 0.774 vs. 0.645 without).
- **Limitations & weaknesses.** No explicit cost in labels or objective (γ only); no per-instance
  budget or λ adaptation; the termination model never trains the executor; the RL variant was
  unhelpful; small models; two domains; no tool-cost accounting.
- **How CASSI differs.** CaRT is CASSI's declared primary baseline and the closest published
  *training-based termination* work: post-hoc per-prefix quality labels, a separate stopper
  watching a frozen executor, counterfactual data. CASSI's deltas: a cost-aware oracle
  (quality − λ·cumcost instead of γ-discounting), a continuous Δ value, budget conditioning, and
  the process-reward bridge into executor GRPO. **To defend:** the reviewers' first question will
  be "CaRT + cost term + your GRPO bridge = CASSI?" — so the plan's CaRT+cost+GRPO baseline is
  mandatory, and CASSI must win against it. Watch the authors: the CaRT group (Qu, Kumar) also
  wrote MRT and is "one step from closing CASSI's loop themselves." Threat: HIGH.
- **Future work they name.** Not recorded in our review.
- **Research gap left open.** Explicit cost in the termination label, and any mechanism by which
  learned termination improves the executor rather than just gating it.

### 4.14 ALP — Adaptive Length Penalty (Xiang, Blagden, Rafailov, Lile, Truong, Finn, Haber; 2025; **NeurIPS 2025 spotlight**; arXiv 2506.05256)

- **Background & motivation.** Exactly CASSI's H5 framing, published first: SFT on short traces,
  user budgets, and "RL with uniform penalties … treat all problems alike regardless of
  difficulty." Easy problems deserve short answers; hard problems deserve long ones.
- **Problem definition.** Input: reasoning prompts with verifiable answers. Output: a policy
  whose per-token penalty *scales with how easy the prompt is*, so compute concentrates on hard
  problems.
- **Research goal.** Difficulty-adaptive length penalties with zero extra compute, using the
  rollouts group-based RL already samples.
- **Approach & architecture.** During training, the empirical solve rate p_solved(q) is computed
  from the K rollouts already sampled per prompt. Reward:
  `r(y,q) = 1[answer correct] − β·N·max(p_solved(q), 1/K)·(per-token cost)` — the per-token
  penalty weight is proportional to the prompt's solve rate: high solve rate (easy) → strong
  penalty; hard prompts nearly unpenalized; a 1/K clip keeps some pressure on unsolved prompts.
  Explicitly "instance-adaptive," with no extra sampling cost.
- **Training methodology & RL usage.** GRPO (VeRL) on DeepScaleR-1.5B, 100 steps, batch 512,
  16,384-token context.
- **Experiments & evaluation metrics.** MATH-500, OlympiadBench, AIME 2024+2025; inference
  budgets {512, 1024, 2048, 4096}; baselines L1-Exact, L1-Max, ThinkPrune-2K, an Arora & Zanette
  variant; difficulty-mixture stress tests.
- **Key results.** ~50% average token reduction at maintained Pass@1; at a 1024-token budget,
  +40% Pass@1 over base on MATH; uses only 21% of tokens on the easiest 50% of problems;
  **adaptation ratio 5.35×** (tokens on hard vs. easy) vs. 1.01× for L1-Exact; most robust
  accuracy under difficulty shift.
- **Limitations & weaknesses.** Outcome-level reward (no step-level signal); difficulty = online
  solve rate (needs K rollouts and verifiable answers); single-model, single-turn math CoT;
  token cost only; no explicit stopping decision.
- **How CASSI differs.** ALP is **the single strongest counterexample to CASSI's "static,
  instance-blind penalties" claim** — per-instance, difficulty-adaptive, RL-trained,
  peer-reviewed (spotlight), with the same "adaptive computation allocation" narrative and even
  H5's difficulty–token correlation evidence. CASSI cannot claim adaptive penalties don't exist.
  Deltas that survive: agentic/heterogeneous-cost setting, per-step (mid-trajectory) signals, the
  separate stopper, and the stopping decision itself. **To defend:** cite; replace the plan's
  home-made "adaptive-α" baseline with published ALP (reviewers will read a home-made stand-in
  as dodging); reposition contribution 4 as *mid-trajectory, progress-conditioned stopping for
  agents with heterogeneous costs*. Threat: HIGH (framing-level).
- **Future work they name.** Not recorded in our review.
- **Research gap left open.** Difficulty adaptation that acts *during* a trajectory (step level)
  rather than via a per-prompt penalty weight, and costs beyond tokens.

### 4.15 Rational Metareasoning for LLMs (De Sabbata, Sumers, AlKhamissi, Bosselut, Griffiths; 2024, v3 June 2025; preprint under review; arXiv 2410.05563)

- **Background & motivation.** Chain-of-thought raises inference cost on *every* query regardless
  of difficulty. The authors import the classical AI idea of **rational metareasoning**
  (reasoning about whether reasoning is worth it) — specifically Russell & Wefald's **Value of
  Computation (VOC)**: compute only while the expected benefit exceeds the cost.
- **Problem definition.** Input: a question x, a candidate reasoning chain z, an answer y.
  Output: a policy that reasons only when beneficial, trained on the reward
  **R(x,y,z) = U(z|x,y) − C(z)**: utility = the likelihood gain of the correct answer given the
  chain (log π(y|z,x) − log π(y|x); the policy itself is the reward model — no external RM) minus
  cost C(z) = γ·length(z), γ = 0.1.
- **Research goal.** Train LLMs to internalize a cost–benefit trade-off for their own reasoning.
- **Approach & architecture.** Single model, no separate monitor. The VOC reward is optimized
  with **Expert Iteration** (rejection-sampling RL): sample K=4 chains per question, compute
  mean-centered advantages, discard negative-advantage (and optionally incorrect) chains, SFT on
  the survivors, repeat for 5 iterations.
- **Training methodology & RL usage.** Expert Iteration on Llama-3.2-3B and Llama-3.1-8B; no
  PPO/GRPO (explicitly left as future work).
- **Experiments & evaluation metrics.** ARC, CommonsenseQA, GSM8K, ProofWriter (mixed training);
  out-of-distribution on MMLU-CF; baselines Direct/CoT few-shot, STaR, instruct models.
- **Key results.** 23–32% fewer output tokens than STaR (35–42% vs. CoT few-shot) at equal or
  better accuracy; adaptivity emerges (up to 50.3% length reduction on the easy split — the
  hard/easy gap grows); OOD: 28–36% fewer tokens than STaR on MMLU-CF at similar accuracy.
- **Limitations & weaknesses (their own §7).** Not agentic — "adapting our method to this
  context would require incorporating the cost of tool use (e.g., API calls) into the reward
  function"; Expert Iteration only; trajectory-level reward (whole chain scored once — no
  per-step stop/continue, no mid-trajectory stopping, no value margin); no inference-time
  controller.
- **How CASSI differs.** This paper answers the question "has a cost-penalizing,
  VOC-style training signal been used for LLMs?" with **yes** — which falsifies CASSI's
  motivation phrasing that LLMs "have no training signal for 'good enough, stop now'" and any
  "first cost-aware training" claim. CASSI must claim the *loop* and the *agentic per-step
  stopping*, not cost-aware training per se. Deltas: separate trained stopper, per-step Δ, oracle
  t* labels, process rewards, tools/budgets, the cycle. **To defend:** cite as the VOC ancestor
  of the oracle objective; use a VOC-reward single-model arm as the "single-model GRPO+cost
  penalty" baseline; fix the motivation sentence. Threat: HIGH (framing-level).
- **Future work they name.** Extending to agentic settings by pricing tool use into the reward;
  moving from Expert Iteration to PPO/GRPO-style RL (both recorded in our review).
- **Research gap left open.** VOC applied per-step, by a trained monitor, in tool-using agents —
  precisely the cell CASSI targets.

---

## 5. Tier-2 quick reference

Adjacent work CASSI must cite but need not beat (grouped; IDs from `00_overview.md` §4).

**Adaptive length penalties for reasoning (kills any "static penalty" strawman; cite alongside
ALP):**
- DAST (2503.04472) — per-problem token budget from sampled solve rate, offline preference
  training; >30% token cuts.
- LASER-D (2505.15612) — dynamic difficulty-aware step-function length rewards.
- AdaptThink (2505.13417, EMNLP 2025) — RL choice of Think vs. NoThink per instance.
- AdaCtrl (2505.18822) — self-estimated difficulty tags + difficulty-aware budgets.
- HAPO (2505.11225, AAAI 2026) — reward vs. per-question historical minimum correct length.

**Cost-aware agentic RL (outcome-level; the crowd around SlimSearcher/OTC-PO):**
- EAPO (2606.02132) — difficulty-aware penalties on redundant tool calls ("learning when not to
  act").
- AdaTIR (2601.14696) — adaptive tool-integrated reasoning.
- DAS (2602.03304, WWW 2026) — causal-intervention alignment of the search/no-search decision
  boundary; shows outcome-RL *causes* over-search.
- CTA / Calibrate-Then-Act (2602.16699) — cost-aware exploration; key finding for CASSI:
  cost-discounted GRPO alone fails to internalize costs (supports the separate-stopper ablation).
- Agent-Omit (2602.04284, ICML 2026) — agent efficiency via omitting needless interaction.

**Budget routing & orchestration:**
- SeqRoute (2605.25424) — Hindsight Budget Relabeling + offline RL + λ-sweep Pareto navigation
  (the "hindsight relabeling + budget + λ" vocabulary is already taken).
- CoRL (2511.02755) — RL controller with dual task+cost rewards, budget-conditioned modes over
  frozen experts.
- xRouter (2510.08439) — cost-aware LLM orchestration via RL.

**Training-free budget-aware agents (inference-time):**
- BATS (2511.17006, Google) — prompt-level Budget Tracker + orchestration for tool agents
  (test-time scaling focus; the plan's earlier description needed this correction).
- BAVT (2603.12634) — budget-aware value tree search.
- INTENT (2602.11541) — intention-based planning under hard monetary tool budgets.
- VoI-budget (2605.05701) — per-step value-of-information under dual budgets; decides when to
  commit the answer (the training-free cousin of CASSI's Δ).

**Policy↔reward co-evolution loops (kills the "first self-reinforcing cycle" phrasing):**
- Self-Guide (2604.03098) — uses "self-reinforcing loop" essentially verbatim for agent
  policy↔internal-reward co-evolution with step-level GRPO rewards.
- SPARK (2509.22624), Cooper (2508.05613) — policy and reward model co-training genre.
- RePro (2606.14302) — retrospective per-step progress labels from completed trajectories as
  dense agent RL rewards (quality-only).

**Learned stopping / early exit (beyond Tier-1):**
- LYNX (2512.05325) — probe stopper on counterfactual forced-exit labels + conformal guarantees.
- LearnStop / "When Does Learning to Stop Help?" (2606.30852) — the referee paper: matched-risk
  protocol; learned stopping wins only in some regimes; **the same stopping policy saves +32.2%
  tokens under KV-fork serving but costs +120.9% extra under black-box re-prefilling** — probe
  overhead can flip the sign of savings; CASSI must adopt this evaluation style.

**Rollout-free step-level credit (kills "step signals require O(K×T²)"):**
- Implicit PRM (2412.01981, ICML 2025); PRIME (2502.01456); GiGPO (2505.10978, NeurIPS 2025);
  SPA-RL (2505.20732); SWEET-RL (2503.15478 — the honest "asymmetric information" argument for a
  separate critic); MRT (2503.07572 — progress-style dense rewards done right).

**Hierarchy / meta-reasoning architectures:**
- HiPER (2602.16165, ICML 2026) — SOTA hierarchical credit assignment in a *single* policy with a
  switch-advantage ≈ Δ (counter-evidence for "separation is necessary").
- ReMA (2503.09501, NeurIPS 2025) — multi-agent meta-thinking; shared weights converge faster;
  1B meta-agents collapse (evidence both ways on separation).
- MaR (2605.23384) — metacognition-as-reward, cost-blind.

**Adaptive retrieval (the RAG cousin of stopping):**
- SIM-RAG (SIGIR 2025) — trained sufficiency critic decides continue/stop per retrieval round
  (no cost term, never trains the generator).
- HiPRAG (2510.07794), StepSearch (2505.15107).

**Optimal-stopping & metareasoning theory (cite as the classical foundations of the oracle):**
- Hansen & Zilberstein 2001 (monitoring/control of anytime algorithms — CASSI's oracle and
  Properties 1–2 are essentially their Definition 4 + Theorem 1); Weitzman 1979 (Pandora's box);
  Chen et al., ICML 2020, "Learning to Stop While Learning to Predict" (post-hoc stop-label
  imitation — note the corrected attribution: Chen et al., not Xiao et al.); Hay et al. 2012;
  Callaway et al. 2018.

**Motivation ammunition (all verified; keep in the intro):**
- Cuadron 2502.08235 (agent overthinking predicts failure, R²=0.892); Wu 2502.07266 (inverted-U
  accuracy vs. length; difficulty–length r=0.57 — meaning H5 is a replication); Hassid 2505.17813
  (shortest-vs-longest +34.5pt); Gema TMLR'25 (inverse scaling); Chiang & Lee EACL'24;
  **Token Economies (2406.06461, EMNLP 2024) + RedundancyBench (2605.29893): prompted
  self-evaluation fails at economic judgment (≤24.9% step-level F1) — the strongest published
  justification for a *trained* stopper**; DAS (outcome-RL causes over-search); CTA
  (cost-discounted GRPO alone fails).

---

## 6. Synthesis: where the defensible gap is

### 6.1 The converged gap statement

Across 10 literature areas and 5 independent novelty reviews, the same conclusion
(`00_overview.md` §2):

> No prior work converts an explicit `quality − λ·cost` hindsight optimum into a **trained
> stopping-value model** whose continuous stop margin Δ serves as a **per-step process reward for
> training a tool-using executor**, with multi-dimensional (token/tool/dollar) costs and
> budget-conditioned inference.

The prior-art hunter's verdict: **no single kill-shot exists** — every one of the 14 closest
papers is missing at least two of {cost objective, stopping decision, separate trained stopper,
executor-training bridge, agents}. But the **composite risk is severe**: a reviewer can stack
DASH + TERMINATOR + Self-Guide + SlimSearcher/ALP + BAGEN and argue CASSI is a recombination.
Three of the five stated contributions are phrased in ways that are *already falsifiable* ("first
cycle," "static penalties," "O(K×T²) is the alternative"). Novelty survival score from the
harshest reviewer: ~4–5.5/10 as framed; ~6/10 after reframing. The window is closing — DASH was
15 days old at review time, and the CaRT/MRT group is one step from closing the loop themselves.

### 6.2 What CASSI v2 must change (claims to drop or reframe)

From the verdict table in `00_overview.md` §3 plus the five technical repairs:

1. **Contribution 1 ("first self-reinforcing cycle") — qualify.** The loop shape is generalized
   policy iteration; AgentPRM already iterates rollouts→labels→PRM→policy, and Self-Guide uses
   "self-reinforcing loop" verbatim. Defensible only as the "first **cost-aware
   stopping-centric** closed loop," and only if ≥2 full iterations are actually measured (the
   plan currently marks iteration "optional" — making the headline claim unfalsifiable).
2. **Contribution 2 ("separate stopper is necessary") — demote to hypothesis.** ReMA, HiPER, and
   SPARK give counter-evidence for single-model capability; PPG (2020) already described
   policy/value interference; the honest version is SWEET-RL's asymmetric-information argument
   (the stopper trains on privileged ground-truth quality that the executor never sees at
   inference). But CTA shows cost-discounted GRPO alone fails to internalize costs — so the
   three-way ablation (single-model vs. multi-task vs. two-model, at matched parameters) is
   winnable and should be the paper's centerpiece.
3. **Contribution 3 ("O(T) vs. O(K×T²)") — rewrite entirely.** Factually wrong about AgentPRM
   (pooled returns, no per-state restarts; O(K×T²) describes Math-Shepherd); Implicit PRM, PRIME,
   TD+GAE AgentPRM (WWW'26), GiGPO, SPA-RL are all rollout-free and currently uncited. Pivot from
   label *cost* to label *semantics*: cost-awareness + ground-truth anchoring + an explicit
   stopping margin.
4. **Contribution 4 ("dynamic beats static penalties") — retire the strawman.** Adaptive
   penalties are the 2025–26 norm (ALP, DAST, LASER-D, AdaCtrl, HAPO, AdaptThink, Agent-Omit,
   SlimSearcher, EAPO); Wu already measured the difficulty–length correlation (r=0.57), so H5 is
   a replication. Surviving angle: *mid-trajectory, progress-conditioned stopping* for *agents*
   with heterogeneous costs — vs. upfront or episode-level penalties.
5. **Contribution 5 ("small stopper, <3% overhead") — partially occupied.** Ares, TAB, and
   SupervisorAgent all put a small/cheap monitor over a bigger executor; LearnStop shows probe
   overhead can eat the savings. Surviving angle: the stopper *trains* the executor (nobody's
   controller does) plus transfer across executors; adopt LearnStop's matched-risk protocol.
6. **Formal properties — cite, don't claim.** Properties 1–2 are classical (Hansen & Zilberstein
   2001; Weitzman 1979; Topkis-style comparative statics; Chen et al. 2020); Property 1's
   uniqueness is actually false without a diminishing-returns assumption. Property 3 is the only
   open formal claim — make it precise or drop it.

**Five technical repairs the novelty agents demand:**
1. **Prophet bias.** The oracle regresses on the realized future maximum of the same trajectory —
   information not available at step t; by prophet-inequality arguments this systematically
   overestimates the value of continuing (late-stopping bias). Fix: Snell-envelope / backward
   fitted-Q labels — same O(T) cost, measurable, textbook-correct. (Bonus: the
   prophet-vs-Snell-vs-MC-vs-TD label ablation is itself novel.)
2. **Reward shaping.** Σ_t α·Δ(s_t) is non-potential-based — it *pays the executor to accumulate
   promising steps*, the opposite of the goal. Fix: potential-based Φ-differences or
   advantage/progress-style step rewards (PAV/MRT lineage).
3. **Reward hacking.** The executor writes the confidence/draft features the stopper reads — the
   exact PRM-hacking loop AgentPRM measured (82%→70%). Fix: objective-only stopper inputs,
   periodic stopper refresh, a held-out hacking probe.
4. **Train/inference leak.** The v5 monitor prompt includes ground-truth-derived quality
   indicators that cannot exist at inference. Restrict stopper inputs to inference-available
   features.
5. **Honest label-cost accounting.** Per-step quality is nearly free for QA (string F1) but
   equals a test-suite run per step on SWE-bench — the "zero extra cost" claim must be stated
   per benchmark; on SWE-bench the intermediate signal is also nearly flat at zero, giving the
   oracle no gradient.

**Benchmark rescoping (from `00_overview.md` §5):** GAIA, WebWalkerQA, BFCL, and MATH-500 have
**no train splits** — they cannot be RL-training benchmarks. Train on HotpotQA/MuSiQue (+ a
tool-use sim such as ALFWorld/WebShop); treat GAIA/WebWalkerQA as transfer evaluation. SWE-bench
Verified RL is only demonstrated at 32B–72B scale (DeepSWE) and per-step quality there means a
test run per step — rescope or drop.

### 6.3 The must-have baseline set

From the feasibility notes (`00_overview.md` §6): the planned 14 baselines × 7 benchmarks × 3
seeds is ~10× over an 8-week budget. Cut to **2 training domains and ~6 pivotal baselines**, with
transfer evals:

1. **SupervisorAgent-style training-free monitor** — the free method that already gets ~30%
   savings on GAIA; CASSI must beat it on the Pareto frontier.
2. **OTC-PO / EAPO cost-aware RL** — the published single-model cost-RL competitors (beating
   home-made variants is not enough).
3. **DASH-style direct shaping** — oracle labels injected straight into GRPO advantages, *no
   learned stopper*. The single most important ablation: it decides whether the stopper earns
   its complexity.
4. **Single-model GRPO + cost penalty** — the representation-conflict arm (instantiate honestly,
   e.g., with SlimSearcher's group-relative gates or a RaM-style VOC reward).
5. **Stopper-as-controller-only** — CASSI without the reward bridge; decides whether the bridge
   (the claimed core novelty) actually matters.
6. **Scalar-probe stopper** — a calibrated confidence/entropy exit (LearnStop-style), evaluated
   at matched lost-correct risk with per-serving-regime overhead accounting.

Supporting hygiene: use estimator-hygienic GRPO (Dr. GRPO / DAPO fixes) or reviewers will
attribute token savings to length-bias artifacts; guard the known pathologies (length-inflation
bias, entropy collapse, Echo Trap, credit dilution); pre-register and run the two kill-switch
experiments *first* at small scale — (a) does Δ-as-process-reward beat controller-only? (b) does
two-model beat single-model at matched parameter count? If either fails, pivot before building
SWE-bench infrastructure.

---

## 7. Beginner's glossary

One line each; terms as used in this document.

- **Ablation**: an experiment that removes one component of a system to measure how much that
  component actually contributes.
- **Advantage (RL)**: how much better an action did than the average/expected alternative; GRPO
  computes it relative to the sampled group.
- **Agent**: an LLM that acts over multiple steps — reasoning, calling tools, observing results —
  rather than answering in one shot.
- **Anytime algorithm**: an algorithm that can be stopped at any time and still return its best
  answer so far (the classical home of stopping theory).
- **Baseline**: an existing or simpler method you compare against to prove your method helps.
- **Best-of-N (BoN)**: sample N candidate solutions and keep the one a reward model scores
  highest.
- **Beam search (step-level)**: keep the top-scoring partial trajectories at each step and expand
  only those.
- **Budget forcing**: hard-capping generation (e.g., s1's trick of truncating or extending
  thinking) rather than learning when to stop.
- **Chain-of-thought (CoT)**: the model's step-by-step written reasoning before its final answer.
- **Conformal prediction**: a statistical wrapper that turns a model's scores into decisions with
  a guaranteed error rate (used by LYNX for safe early exits).
- **Credit assignment**: figuring out which of the many steps in a trajectory deserve
  praise/blame for the final outcome.
- **DPO (Direct Preference Optimization)**: trains a model directly from "A is better than B"
  pairs, skipping an explicit reward model; *Online DPO* regenerates pairs during training.
- **Early exit**: stopping a reasoning model's generation before it finishes naturally, once the
  answer is (probably) already determined.
- **Executor**: the large model that actually performs the task (vs. the monitor/stopper that
  watches it).
- **Expert Iteration**: RL-lite loop — sample, keep only the best outputs, fine-tune on them,
  repeat.
- **F1 score**: harmonic mean of precision and recall; standard partial-credit metric for QA
  answers.
- **Frozen (model)**: its weights are not updated during the procedure in question.
- **GAE (Generalized Advantage Estimation)**: a standard way to blend short- and long-horizon
  reward information when estimating advantages.
- **GAIA / WebWalkerQA / HotpotQA / MuSiQue / SWE-bench / MATH-500 / BFCL**: benchmarks for web
  agents, multi-hop QA, software engineering agents, math, and function calling, respectively.
- **GRPO (Group Relative Policy Optimization)**: LLM RL that samples a group of G responses per
  task and scores each relative to the group mean — no value network needed.
- **Hindsight (post-hoc) label**: supervision computed after the fact from a completed
  trajectory ("in hindsight, the best stop was step 6").
- **Iso-accuracy / iso-cost**: comparing costs at equal accuracy, or accuracy at equal cost.
- **KV cache**: stored attention states that let a model continue generating without recomputing
  the prefix; probing methods that break it get expensive.
- **λ (lambda)**: the price of cost in the objective `quality − λ·cost`; larger λ = stingier =
  earlier stopping.
- **LoRA**: low-rank adapters — a cheap way to fine-tune a model by training small added matrices.
- **LRM (Large Reasoning Model)**: an LLM trained to emit long chains-of-thought (o1/R1-style).
- **MDP (Markov Decision Process)**: the standard RL formalism — states, actions, rewards,
  transitions.
- **Monte Carlo (MC) rollouts**: estimating a state's value by actually running many completions
  from it and averaging the outcomes — accurate but expensive.
- **Oracle (label)**: the best-possible answer computed with information a deployed system won't
  have (here: the whole completed trajectory).
- **Outcome reward**: one score for the whole trajectory (did it succeed? how much did it cost?).
- **Overthinking**: continuing to reason/act after the answer is already good, wasting cost and
  sometimes destroying a correct answer.
- **Pareto frontier**: the set of methods/settings you can't improve on one axis (accuracy)
  without losing on the other (cost).
- **Policy**: the model being trained to act; "policy optimization" = RL on it.
- **PPO (Proximal Policy Optimization)**: the classic deep-RL algorithm; needs a value network,
  unlike GRPO.
- **PRM (Process Reward Model)**: a trained model that scores each intermediate step of a
  solution, giving dense training signals.
- **Probe**: a small classifier reading a frozen model's internal states (hidden activations) to
  predict something, e.g., "has the answer arrived?"
- **Process reward**: a per-step reward (from a PRM or rule) used during RL training.
- **Prophet bias**: the optimism that comes from training on the realized future maximum — a
  quantity no real-time decision-maker can see; biases stoppers toward stopping late.
- **Q-value / value function**: expected future reward from a state (or state-action pair);
  CASSI's Δ = Q_continue − Q_stop.
- **ReAct**: the standard agent loop format — alternate Reasoning text and Actions (tool calls).
- **Reward hacking (Goodhart)**: the policy learns to inflate the reward model's score without
  actually getting better (e.g., stating fake confidence).
- **Reward model (RM)**: a trained model that scores outputs; the generic parent of PRMs.
- **Reward shaping / potential-based shaping**: adding auxiliary rewards to speed learning;
  "potential-based" is the (safe) form that provably doesn't change the optimal policy.
- **RL (Reinforcement Learning)**: training by trial, reward, and adjustment rather than by
  imitation.
- **RLVR**: RL with verifiable rewards — domains where correctness can be checked automatically.
- **Rollout**: one sampled attempt (trajectory) generated by the current policy.
- **SFT (Supervised Fine-Tuning)**: training a model to imitate labeled examples.
- **Snell envelope**: the textbook-correct value function for optimal stopping, computed by
  backward recursion; the unbiased alternative to prophet-style labels.
- **Stopper / stopping model**: the model that decides STOP vs. CONTINUE for the executor.
- **TD (temporal-difference) learning**: estimating values by bootstrapping from the model's own
  next-step estimate instead of waiting for the final outcome.
- **Token**: the unit LLMs read/write (~¾ of a word); the unit most compute costs are billed in.
- **Tool call**: an agent action that invokes an external resource (search API, code runner);
  often the dominant real-world cost.
- **Tool productivity (TP)**: OTC-PO's metric — correct answers per tool call.
- **Trajectory**: the full recorded step sequence of one task attempt.
- **VOC (Value of Computation)**: classical metareasoning quantity — expected benefit of more
  computing minus its cost; compute only while VOC > 0.
- **Δ (delta) margin**: CASSI's continuous stopping value, Q_continue − Q_stop, normalized to
  [−1, 1]; positive = keep going, negative = stop.
