# CASSI — Explained Simply (Complete Guide)

> One-page idea: **We built a tiny cheap "coach" that watches AI agents work and tells them when to stop — saving 20-40% money without hurting quality. The coach costs almost nothing to train because we learn from work already done, not from expensive future simulations.**

---

## 1. THE PROBLEM (Why This Matters)

### Imagine You Hire a Researcher...

You say: "Find out everything about electric cars in 2025."

A **smart** researcher:
- Finds the key facts in 2 hours
- Says "I have enough for a solid report" and stops
- You pay for 2 hours of work ✅

A **dumb** researcher:
- Finds the key facts in 2 hours
- Keeps searching for 6 more hours finding slightly different articles saying the same thing
- You pay for 8 hours for the same report ❌

**This is exactly what happens with AI agents today.** They were trained to "be helpful" — nobody ever rewarded them for saying "this is good enough, I'll stop now."

### The Three Ways AI Agents Waste Your Money

| Problem | What Happens | Real Cost |
|---|---|---|
| **Overthinking** 🤔 | Agent has the right answer at step 3, but keeps "polishing" until step 15 | 80% of your compute bill is wasted |
| **Runaway Loops** 🔄 | Agent searches the same thing, gets same result, searches again... forever | Burns your entire budget with no result |
| **Stopping Too Early** ⏹️ | Agent panics about cost, submits a wrong answer at step 2 | Task fails — money already spent for nothing |

### Proof This Is Real (From Other Research Papers)

- **"When More is Less" (Wu et al., 2025):** Beyond a certain length of thinking, accuracy actually goes DOWN — like an upside-down U. More thinking literally makes things worse.
- **"Don't Overthink It" (Hassid et al., 2025):** The shortest correct answer chain is 34.5% more likely to be right than the longest chain.
- **DeepSeek-R1:** Takes "more than a page" of thinking to answer "How much is 1+1?"

---

## 2. WHAT OTHERS HAVE TRIED (And Why They Fall Short)

### Solution A: "Hard Cutoff" — Stop at exactly 500 tokens no matter what
- **Paper that does this:** s1 (Muennighoff, 2025)
- **The idea:** Set a fixed token limit. Agent MUST stop.
- **Why it fails:** No flexibility. "What's 2+2?" and "Prove Fermat's Last Theorem" get the same cutoff. Agent submitting a perfect answer gets cut off mid-sentence; agent with a wrong answer wastes the full budget.

### Solution B: "Tax Every Word" — Penalize long outputs in training
- **Papers that do this:** L1 (Aggarwal & Welleck, 2025), Reason Efficiently (Arora & Zanette, NeurIPS 2025)
- **The idea:** During training, subtract points for every extra token. The model learns to be concise.
- **Why it fails:** The tax is the SAME for every task. An easy problem and a hard problem get the exact same penalty rate. The model learns to be short everywhere — even when it NEEDS to think longer on hard problems.

### Solution C: "Gut Feeling Manager" — Use rules of thumb to decide when to stop
- **Paper that does this:** BATS (Liu et al., 2025)
- **The idea:** Track budget usage in the prompt. Use hand-crafted rules: "if less than 10% budget remains, stop."
- **Why it fails:** No learning. The same rules apply to every situation. Cannot improve from experience. Doesn't understand when the answer is "good enough" vs "still wrong."

### Solution D: "Simulate 160 Futures to Make One Decision" — Run expensive MC rollouts
- **Paper that does this:** AgentPRM (Choudhury, 2025)
- **The idea:** At every step, run 8 full simulations of "what would happen if we kept working" — each simulation goes all the way to the end. Average the results to decide.
- **Why it fails:** INCREDIBLY expensive. For a 20-step task: 8 simulations × 20 steps ≈ 160 extra full work sessions. You're spending more on deciding than on actually working. **This is O(K×T²) — costs grow quadratically with task length.**

---

## 3. CASSI's BIG IDEA (In One Sentence)

> **Instead of simulating the future 160 times (expensive), look at what already happened and figure out the best stopping point after the fact (free).**

### The Key Insight (With Numbers)

```
AgentPRM (competitor):  Run 8 simulations × 20 steps from every point = ~160 extra full executions per task
CASSI (our approach):   Look at the 1 trajectory already completed = 0 extra executions per task

Reduction:              160× fewer executions. Same quality of training signal.
```

### How We Compute the "Perfect Stopping Point" (For Free)

After the agent finishes a task, we have a log of every step:

```
Step 1: Agent searches "electric cars 2025"           | Cost so far: $0.01 | Answer quality: 0%   (no answer yet)
Step 2: Agent reads article, starts forming answer    | Cost so far: $0.02 | Answer quality: 30%  (partial answer)
Step 3: Agent has a solid answer                      | Cost so far: $0.03 | Answer quality: 95%  (almost perfect!)
Step 4: Agent searches more for "confirmation"        | Cost so far: $0.05 | Answer quality: 95%  (same as step 3)
Step 5: Agent searches even more, slightly rewords    | Cost so far: $0.07 | Answer quality: 96%  (1% improvement for $0.02)
Step 6: Agent searches again, answer unchanged        | Cost so far: $0.09 | Answer quality: 95%  (actually got slightly worse!)
```

Now we do a simple calculation for each step:

```
Value at step 3 = quality (95%) − λ × cost ($0.03) = 95 − λ×3   ← HIGHEST!
Value at step 6 = quality (95%) − λ × cost ($0.09) = 95 − λ×9   ← much lower

So: t* (optimal stop) = step 3. EVERYTHING AFTER STEP 3 WAS WASTE.
```

We now know: "At step 3, the agent SHOULD have stopped."

This entire calculation takes milliseconds — we just scan through numbers we already have. **Zero extra work, zero extra cost.**

---

## 4. THE ARCHITECTURE (What CASSI Looks Like)

```
┌──────────────────────────────────────────────────────────────────┐
│                        CASSI SYSTEM                                │
│                                                                    │
│   ┌───────────────────┐          ┌─────────────────────────┐      │
│   │   STOP COACH      │          │   WHAT DOES THE COACH    │      │
│   │   (tiny model)    │◄─────────│   SEE?                   │      │
│   │   0.5B - 3B params│          │                          │      │
│   │                   │          │ • Task description       │      │
│   │   Like a manager  │          │ • Budget used so far     │      │
│   │   who doesn't do  │          │   - Tokens: 4,500/50,000│      │
│   │   the work but    │          │   - Tools: 3/10 calls    │      │
│   │   decides when    │          │   - Money: $0.15/$2.00  │      │
│   │   to ship it      │          │   - Budget tier: HIGH    │      │
│   │                   │──────────►                          │      │
│   │   Output:          │          │ • Last 3 steps summary  │      │
│   │   STOP / CONTINUE  │          │ • Current answer draft  │      │
│   │   / ADJUST         │          │ • Quality indicators    │      │
│   │   + Score (-1..+1) │          │   - Confidence: 0.94   │      │
│   │   + Confidence     │          │   - Progress rate       │      │
│   │   + Reason         │          │   - Answer stability    │      │
│   └────────┬───────────┘          └─────────────────────────┘      │
│            │                                                        │
│            │ cost-aware reward signal                               │
│            ▼                                                        │
│   ┌───────────────────┐          ┌─────────────────────────┐      │
│   │   WORKER AGENT    │          │   WHAT DOES THE WORKER   │      │
│   │   (big model)     │──────────►   DO?                    │      │
│   │   7B - 72B params │          │                          │      │
│   │                   │          │ • Search the web         │      │
│   │   The person      │          │ • Browse pages           │      │
│   │   actually doing  │◄─────────│ • Run code               │      │
│   │   the task        │  result  │ • Read files             │      │
│   │                   │          │ • Submit final answer    │      │
│   └───────────────────┘          └─────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

### The Two Players — Explained Simply

**Player 1: The Worker (Executor Agent)**
- Size: 7 billion to 72 billion parameters (big AI brain)
- Job: Actually do the task — search, read, code, answer
- Models used: Qwen2.5-7B/32B, Llama-3.1-8B
- How it works: Takes the task + all previous steps → produces the next action
- It receives a "score" from the coach at every step during training

**Player 2: The Stop Coach (Stopping Model)**
- Size: 0.5 billion to 3 billion parameters (tiny AI brain, 14× to 144× smaller!)
- Job: Look at the situation and decide STOP, CONTINUE, or ADJUST
- Models used: Qwen2.5-0.5B/1.5B/3B
- Cost to run: Less than 3% of the worker's cost
- What it sees: Budget status, progress summary, current answer quality, last actions

### What the Coach Outputs (The "Decision Card")

Every step, the coach produces a structured answer like this:

```
Decision: STOP
Reason: "The answer is correct and hasn't improved for 3 steps. 
         Budget is at 45% but further work would add zero value. 
         Net value of continuing is negative."
Delta (Δ): -0.42     ← negative = stop, positive = continue
Confidence: 0.91     ← very sure about this decision
```

The **Delta (Δ)** is the coach's single-number summary:
- **Δ > 0:** "Continuing is worth it" → keep going
- **Δ ≤ 0:** "Stop — further work costs more than it's worth"
- **Far from 0:** Coach is very confident
- **Close to 0:** Coach is unsure

### The Budget State (What Makes CASSI "Cost-Aware")

The coach tracks FIVE dimensions of cost simultaneously:

| Cost Type | Example | Why Track It |
|---|---|---|
| **Tokens Used** | 4,500 of 50,000 | Each token costs real API money |
| **Tool Calls** | 3 of 10 allowed | Each search/API call costs money |
| **Iterations** | 5 of 20 max | Each step takes time |
| **Dollar Cost** | $0.15 of $2.00 budget | The actual money spent |
| **Budget Tier** | HIGH (60%+ remaining) | Determines urgency |

The **Budget Tier** automatically adjusts the coach's behavior:

| Tier | Remaining Budget | Coach's Behavior |
|---|---|---|
| 🔵 **HIGH** (>60%) | Lots of budget left | Relaxed — prioritize accuracy |
| 🟡 **MEDIUM** (30-60%) | Moderate budget | Balanced — normal cost sensitivity |
| 🟠 **LOW** (10-30%) | Running low | Cost-sensitive — be conservative |
| 🔴 **CRITICAL** (<10%) | Almost out | Very strict — almost always stops |

As the budget drains, the coach naturally becomes more conservative — just like a real manager would.

---

## 5. THE TRAINING PROCESS (How We Build CASSI)

CASSI is trained in **3 phases**. Think of it like training a sports team:

```
PHASE 1          →     PHASE 2          →     PHASE 3
Watch games      →     Train the coach   →     Train the player
(collect data)         (on recorded games)      (with coach's feedback)
```

### Phase 1: Collect Trajectories (Week 2)

**What:** Run the worker agent on 5,000–20,000 tasks. Let it work until it finishes or hits the 20-step limit.
**Record:** Every single detail at every step — what it did, what it cost, how good the answer was.

```
Example log for ONE task:
┌────────┬─────────────────────┬──────────┬─────────────────┐
│ Step   │ What happened       │ Cost     │ Answer quality   │
├────────┼─────────────────────┼──────────┼─────────────────┤
│ Step 1 │ Searched "EV 2025"  │ $0.005   │ 0% (no answer)  │
│ Step 2 │ Read 3 articles     │ $0.012   │ 40% (partial)   │
│ Step 3 │ Formed good answer  │ $0.018   │ 95% ✓           │
│ Step 4 │ Searched more...    │ $0.0025  │ 95% (same)      │
│ Step 5 │ "Double-checked"    │ $0.0035  │ 94% (worse!)    │
│ Step 6 │ Reworded answer     │ $0.0045  │ 95% (same)      │
└────────┴─────────────────────┴──────────┴─────────────────┘
```
- Data needed: 5,000–20,000 tasks with known correct answers
- Hardware: Standard (runs on the executor's normal hardware)
- Duration: ~1 week

### Phase 2: Train the Stop Coach (Weeks 3-4)

**This is CASSI's key innovation.** Two sub-steps:

#### Step 2a: Compute Oracle Labels (Free!)
For each completed task, calculate: "At which step was value = quality − λ×cost maximized?"

```
For the example above (with λ = 1.0):
Step 1: value = 0    − 1.0×0.005  = −0.005
Step 2: value = 40   − 1.0×0.012  = 39.988
Step 3: value = 95   − 1.0×0.018  = 94.982   ← HIGHEST → t* = 3
Step 4: value = 95   − 1.0×0.025  = 94.975
Step 5: value = 94   − 1.0×0.035  = 93.965
Step 6: value = 95   − 1.0×0.045  = 94.955
```

Result: t* = 3. Steps 1-2 should say CONTINUE. Steps 3-6 should say STOP.

#### Step 2b: Supervised Fine-Tuning (SFT) — "Copy the correct answers"
- Feed the coach: "Here's what the situation looked like at step X"
- Tell the coach: "The correct decision here was STOP/CONTINUE"
- The coach learns by copying: "Ah, when I see this pattern → I should say STOP"
- Hardware: 1-2× H100 GPUs
- Duration: ~2-4 hours
- Learning rate: 2e-5

#### Step 2c: RL Fine-Tuning (GRPO) — "Practice and get rewarded"
- Let the coach make decisions on new tasks
- Reward the coach for:
  - Stopping close to t* (not too early, not too late) ← main reward
  - Giving accurate Δ values (close to the oracle Δ) ← accuracy reward
  - Producing well-formatted output ← format reward
- The coach improves through trial-and-error
- Hardware: 2-4× H100 GPUs
- Duration: ~4-8 hours
- Algorithm: GRPO (Group Relative Policy Optimization), group size G=8

### Phase 3: Train the Worker with the Coach (Weeks 4-5)

Now we use the trained coach to teach the worker.

#### How the Worker Learns (GRPO Training)

```
For each task:
  1. Worker tries to complete the task (generates 8 different attempts)
  2. At every step, the Coach evaluates: "Is continuing worth it?" → gives a Δ score
  3. Worker gets rewarded for:
     a. The Coach's Δ score (cost-awareness)      —  how cost-aware was this step?
     b. Did this step make progress?               —  is the answer improving?
     c. Is the output well-formatted?              —  clean output?
     d. At the end: was the task solved?            —  final correctness
  4. Compare the 8 attempts. The ones that balanced cost AND quality best get reinforced
  5. Repeat with new tasks — worker improves over time
```

- Hardware: 4-8× H100 GPUs
- Duration: ~12-24 hours
- Group size (G): 8 (8 parallel attempts per task, compared against each other)
- KL penalty: β = 0.04 (prevents the worker from changing too drastically)
- Algorithm: GRPO with group-normalized advantages
- Max steps per trajectory: 20

### Training Infrastructure Summary

| What | Model Size | GPUs | Time |
|---|---|---|---|
| Worker (pre-trained, no training) | 7B–72B | — | — |
| Coach SFT (copy correct answers) | 0.5B–3B | 1–2× H100 | 2–4 hours |
| Coach RL (practice + rewards) | 0.5B–3B | 2–4× H100 | 4–8 hours |
| Worker RL (train with coach) | 7B–32B | 4–8× H100 | 12–24 hours |

---

## 6. THE EXPERIMENTS (How We Prove It Works)

### 6.1 Research Questions (7 Things We Must Prove)

| # | Question | Why It Matters |
|---|---|---|
| **RQ1** | Does CASSI actually reduce cost while keeping accuracy? | The whole point of the paper |
| **RQ2** | Is CASSI better than "just penalize long answers"? | Proves dynamic > static |
| **RQ3** | Does the coach stop earlier on easy tasks and later on hard tasks? | Proves the coach understands difficulty |
| **RQ4** | Is CASSI better than CaRT (train agent to stop itself) and AgentPRM (simulate futures)? | Proves architecture + training method matter |
| **RQ5** | How much faster/cheaper is CASSI's training vs AgentPRM? | Proves the efficiency claim |
| **RQ6** | Do we need RL training for the coach, or is SFT enough? | Ablation — what's necessary? |
| **RQ7** | How much does the coach itself cost to run vs how much it saves? | Practicality — is it worth it? |

### 6.2 Where We Test (7 Benchmarks Across 4 Domains)

| Domain | Benchmark | What The Agent Does | # Questions | Metric |
|---|---|---|---|---|
| 🌐 **Web Research** | GAIA | Multi-step web research (find facts, cross-reference) | 466 | Exact match accuracy |
| 🌐 **Web Research** | WebWalkerQA | Navigate websites and answer questions | Many | Accuracy |
| 🔗 **Multi-hop QA** | HotpotQA | Answer questions needing 2+ facts from different sources | Many | F1 score |
| 🔗 **Multi-hop QA** | MuSiQue | Harder multi-hop: 2-4 steps of reasoning | Many | F1 score |
| 💻 **Software Engineering** | SWE-bench Verified | Fix real bugs in real GitHub repositories | Many | Test pass rate (pass@1) |
| 🧮 **Math (Control)** | MATH-500 | Competition math problems | 500 | Accuracy |
| 🔧 **Tool Use** | BFCL | Pick the right API/tool for a task | Many | Accuracy |

**Why include MATH-500?** Math problems have zero wasted steps — every step is necessary. This is our "control" test: the coach should NOT interfere much on math. If the coach wrongly tells the worker to stop early on math, we know something is wrong.

### 6.3 Who We Compare Against (12 Baselines)

| Baseline | What It Does | Priority | What It Tests |
|---|---|---|---|
| **ReAct** | Standard agent, no cost awareness | Lower bound | Is CASSI better than doing nothing? |
| **Zero-Training Self-Eval** | Agent asks itself "are you confident?" | **P0 — Must beat** | Is self-evaluation enough? |
| **BATS** | Prompt-level budget reminders (no learning) | Budget heuristic | Is learning better than prompts? |
| **BATS-Optimized** | Grid-searched best BATS settings | P1 | Even optimized heuristics lose? |
| **s1 Budget Forcing** | "Stop after X tokens" (rigid cutoff) | Hard stopping | Is rigid cutoff worse than smart stopping? |
| **L1 / LCPO** | Train with exact token target penalty | Static penalty RL | Does CASSI beat uniform penalties? |
| **Reason Efficiently** | Train with length penalty (normalized) | Static penalty RL | Does CASSI beat normalized penalties? |
| **Adaptive-α Reason Eff.** | Pick best penalty per difficulty → Reason Efficiently | **P0 — Must beat** | Even adaptive static penalties lose? |
| **CaRT** | Train agent to self-terminate via SFT | Learned termination | Is separate coach better than self-termination? |
| **CaRT + cost + GRPO** | CaRT with cost penalty + full GRPO | **P0 — Must beat** | Best version of closest competitor? |
| **AgentPRM-cost** | AgentPRM's PRM with cost rewards | **P0 — Must beat** | Is oracle labeling better than MC rollouts? |
| **Oracle Stopping** | Perfect stopping (upper bound) | Upper bound | How close to perfect can CASSI get? |

### 6.4 What We Measure (9 Metrics)

| Metric | Definition | Which Direction Is Better |
|---|---|---|
| **Task Success Rate** | % of tasks correctly completed | ↑ Higher |
| **Average Cost per Task** | Mean (tokens + tool calls + dollar cost) | ↓ Lower |
| **Cost at Iso-Accuracy** | How much does CASSI cost to match a baseline's accuracy? | ↓ Lower |
| **Accuracy at Iso-Cost** | How accurate is CASSI at the same cost as a baseline? | ↑ Higher |
| **Pareto Frontier** | The curve of (cost, accuracy) across different λ values | ↑ Larger area under curve |
| **Stopping Error** | How far is the coach's stop from the optimal stop? | ↓ Lower |
| **Runaway Prevention** | % of tasks where agent avoids infinite loops | ↑ Higher |
| **Monitor Overhead** | What % of total cost is the coach? | ↓ Lower |
| **Cost Savings vs Oracle** | How much do we save vs the upper bound? | ↑ Higher |

### 6.5 The Hardest Test: H5 (Load-Bearing Claim)

This is the single most important hypothesis in the paper:

> **H5: The coach stops earlier on easy tasks and later on hard tasks — automatically, without being told.**

We measure this by checking whether the average stopping step correlates with task difficulty:

- **GAIA Level 1** (easy web research) → Coach should stop earlier
- **GAIA Level 3** (hard web research) → Coach should stop later
- **HotpotQA 2-hop** (simple multi-hop) → Coach should stop earlier
- **MuSiQue 4-hop** (complex multi-hop) → Coach should stop later

We require a Pearson correlation **r > 0.5** with statistical significance (p < 0.05). If this fails, CASSI doesn't work as claimed.

### 6.6 Ablation Studies (Taking Things Apart)

We test 6 "what if we changed X?" questions:

| What We Change | Variants Tested |
|---|---|
| **Coach size** | 0.5B, 1.5B, 3B, 7B params (how small can the coach be?) |
| **Training signal** | No coach, SFT-only coach, SFT+RL coach, Coach as reward (which training matters?) |
| **Budget tracking** | Track only tokens, only tools, or everything (multi-dimensional needed?) |
| **Coach's view** | See full trajectory, last 3/5/10 steps, or only budget (what info matters?) |
| **λ (cost sensitivity)** | λ = 0.1, 0.5, 1.0, 2.0, 5.0 (how does the dial affect performance?) |
| **Coach usage** | Use coach for training only OR training + inference (both needed?) |

### 6.7 Expected Results

| Hypothesis | What We Expect | Why |
|---|---|---|
| **H1: Cost reduction** | 20–40% less cost at same accuracy vs ReAct | Coach eliminates unnecessary steps |
| **H2: Accuracy preserved** | Within 3% of unlimited ReAct | Dynamic stopping prevents early cuts on hard tasks |
| **H3: Beats static penalties** | CASSI > L1 and Reason Efficiently on mixed-difficulty tasks | Static penalties are blind to difficulty |
| **H4: Beats CaRT + AgentPRM** | 5-10% better Pareto frontier | Separate coach + oracle labels > self-termination + MC rollouts |
| **H5: Adaptive stopping** | Stopping point correlates with difficulty (r > 0.5) | Coach learns when more work is worth it |
| **H6: Coach transfers** | Coach trained on 7B works on 32B (moderate loss) | Coach evaluates task state, not worker quirks |
| **H7: Net savings** | Coach costs 1-3% but saves 20-40% | Tiny coach watching big worker = profit |

### 6.8 Qualitative Behaviors We Expect

1. **"Pearl Detection"** 💎 — Coach recognizes when the answer is already perfect and signals STOP, preventing polishing
2. **"Dead-end Detection"** 🛑 — Coach spots when the worker is stuck in a loop and signals ADJUST to change approach
3. **"Graceful Degradation"** 📉 — Under tight budget, coach stops earlier with a partial answer rather than burning everything
4. **"Difficulty Calibration"** 📊 — "What's the capital of France?" → immediate STOP. Complex research question → allows more exploration

### 6.9 Experimental Configuration

| Setting | Value |
|---|---|
| Worker models | Qwen2.5-7B-Instruct, Qwen2.5-32B-Instruct, Llama-3.1-8B-Instruct |
| Coach models | Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct |
| GRPO group size | G = 8 (8 parallel attempts per task, compared against each other) |
| KL penalty | β = 0.04 (prevents too much change during training) |
| SFT learning rate | 2e-5 |
| RL learning rate | 5e-6 |
| Max steps per task | 20 |
| Token budget | 10K / 50K / 100K |
| Tool call budget | 5 / 10 / 20 |
| Dollar budget | $0.50 / $2.00 / $5.00 |
| Random seeds | 3 different seeds (42, 123, 789) — for reproducibility |
| λ values tested | 0.1, 0.5, 1.0, 2.0, 5.0 |
| Statistical tests | Paired t-test, bootstrap confidence intervals (10,000 resamples), Bonferroni correction |

---

## 7. THE CONTRIBUTIONS (What's New)

### Five Things We Claim As Contributions

| # | Contribution | Why It Matters |
|---|---|---|
| 1 | **The self-reinforcing training cycle** | No prior work connects oracle labeling → stopper training → process rewards → executor improvement → better trajectories. AgentPRM has pieces, CaRT has pieces, but nobody closes the loop |
| 2 | **Why two models are necessary (not just a choice)** | A single model doing both reasoning AND cost-evaluation faces a "representation conflict" — the two tasks need different features. We prove splitting them is strictly better |
| 3 | **O(T) oracle stopping labels (zero extra executions)** | Reduces training cost by 160× vs AgentPRM. Makes cost-aware agent training possible on long tasks (SWE-bench) where it was previously too expensive |
| 4 | **Dynamic per-instance cost adaptation beats static penalties** | Coach spends more on hard problems, less on easy ones — automatically. Static approaches can't do this |
| 5 | **Tiny coach supervises giant worker (<3% overhead)** | 0.5B coach managing a 72B worker. The overhead is less than what you save |

### Two Things We Do NOT Claim As Contributions

- **Multi-dimensional budget** — BATS and INTENT already track tokens + tools
- **GRPO-based training** — we use existing algorithms without modification

---

## 8. RISKS AND BACKUP PLANS

| Risk | How Likely | How Bad | What We'll Do If It Happens |
|---|---|---|---|
| Coach isn't better than static penalties | Medium | High | Reposition as "more flexible alternative" rather than "better performance" |
| Oracle labels are noisy/wrong | Medium | Medium | Validate with human review; try different oracle formulas |
| Coach only works with one worker model | Medium | Medium | Test transfer early; report as limitation if it fails |
| Coach costs more than expected | Low | Medium | Reduce evaluation frequency (check every 2nd step instead of every step) |
| Results only work on one domain | Medium | High | Test 3+ diverse domains; narrow claims if domain-specific |
| GRPO training is unstable | Low-Medium | Medium | Start small; use well-tested implementations; fall back to SFT-only coach |
| Math tasks have no waste to save | Low | Low | EXPECTED — included as control to show coach correctly doesn't interfere |

---

## 9. THE SELF-REINFORCING CYCLE (The Core Idea)

```
                     THE SELF-REINFORCING CYCLE
                     
   ① Executor runs tasks, generates trajectories
        │
        ▼
   ② Oracle computes t* = argmax[quality − λ×cost]
        │  (O(T) post-hoc, zero extra work)
        ▼
   ③ Stopping model trains on oracle labels
        │  (SFT: "copy correct stops" → GRPO: "practice")
        ▼
   ④ Stopper provides Δ(s_t) as process rewards
        │  to executor during GRPO training
        ▼
   ⑤ Executor learns cost-aware behavior
        │  (better actions + knows when they're done)
        │
        └────────► back to ① (produces better trajectories)

WHY THIS CYCLE IS NEW:
- AgentPRM: has ②→③→④ but no cost, no stopping, O(K×T²) 
- CaRT: has ①→②→③ but no ④→⑤ (coach only, worker not trained)
- Ares: has ①→②→③ with discrete levels, no ④→⑤ (no worker training)
- Reason Efficiently: has ⑤ only (cost penalty in one model, none of ①→④)
```

## 10. VISUAL SUMMARY (One Diagram)

```
                    THE OVERALL IDEA
                    ================

   OLD WAY (AgentPRM):                    OUR WAY (CASSI):
   ───────────────────                    ───────────────
                                          
   To decide "should I stop?"             After finishing, we look back:
   at step 5:                             
                                          Step 1: quality=0,  cost=$0.01  → value=-0.01
   Run 8 simulations from step 5          Step 2: quality=40, cost=$0.02  → value=39.98
   Each simulation goes to step 20        Step 3: quality=95, cost=$0.03  → value=94.97 ← BEST!
   That's 8 × 15 = 120 extra steps        Step 4: quality=95, cost=$0.05  → value=94.95
                                          Step 5: quality=95, cost=$0.07  → value=94.93
   Repeat this for EVERY step             
   (step 1, 2, 3, ... 20)                "t* = 3. The agent should have 
                                          stopped at step 3."
   Total: ~160 extra full work            
   sessions per training task             Cost: 0 extra executions.
                                          Time: milliseconds of math.
   
   COST: O(K × T²) — EXPENSIVE!           COST: O(T) — CHEAP!
```

---

## 11. KEY TERMS (Glossary)

| Term | Plain English |
|---|---|
| **LLM Agent** | An AI that uses tools (search, code, browse) to complete tasks — not just chat |
| **Trajectory** | The full sequence of steps: think → act → observe → think → act → observe → ... |
| **Stopping Model / Coach** | Our tiny AI (0.5B-3B params) that decides STOP/CONTINUE/ADJUST |
| **Executor / Worker** | The big AI (7B-72B params) that actually does the task |
| **Oracle Label** | The "correct" stopping point computed after the fact from completed work |
| **t*** | The step number where the agent should have stopped (optimal stopping point) |
| **λ (lambda)** | Cost sensitivity dial. Higher λ = save more money. Lower λ = maximize accuracy |
| **Δ (delta)** | Coach's score: positive = keep going, negative = stop. Range: [-1, +1] |
| **O(T)** | "Cost grows linearly with steps" — our approach. For 20 steps: ~20 operations |
| **O(K×T²)** | "Cost grows quadratically" — competitor. For 20 steps with K=8: ~1,520 operations |
| **SFT** | Supervised Fine-Tuning — "here are correct examples, copy them" |
| **GRPO** | Group Relative Policy Optimization — "try 8 different attempts, reward the best ones" |
| **RL** | Reinforcement Learning — "learn by trying things and getting scored" |
| **PRM** | Process Reward Model — scores each step's quality, not just the final answer |
| **MC Rollout** | Monte Carlo Rollout — "simulate the future many times and average the result" |
| **Pareto Frontier** | The best possible trade-off curve between cost and accuracy |
| **Ablation** | "What happens if we remove X?" — testing which parts of the system matter |
| **Budget Tier** | Urgency level: HIGH (>60% left), MEDIUM (30-60%), LOW (10-30%), CRITICAL (<10%) |
| **Self-Reinforcing Cycle** | The loop: executor → oracle labels → stopper training → process rewards → better executor. Closing this loop is CASSI's main innovation |
| **Representation Conflict** | Why one model can't do both reasoning and cost-evaluation well — the two tasks need different "mental features" to succeed |

---

*This covers everything. For the full technical details, see `research/paper_plan.md`.*  
*For how CASSI compares to 28 other papers, see `research/competitor_analysis.md` through `competitor_analysis_part4.md`.*
