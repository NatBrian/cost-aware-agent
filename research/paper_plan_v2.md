# Paper Plan v2 — CASSI: Internalizing Economic Judgment into LLM Agents via Cost-Aware Stopping Rewards

> **Target venue:** ICLR 2027 (submission ~Sept 2026) — fallback NeurIPS 2027 / ICML 2027
> **Status:** Post-independent-review rewrite (2026-07-16). Supersedes `paper_plan.md` (v5).
> **Provenance:** grounded in a fresh 10-area literature review (~94 papers read at PDF level,
> `research/lit_review/00_overview.md`) and 5 independent novelty audits
> (`research/novelty_check/`, scores 4–5.5/10 for v5 → this rewrite targets the unanimously
> identified surviving gap). Competitor detail: `research/competitor_analysis.md`.
> **Self-contained:** this document is written so an agent with zero prior context can implement
> the research end-to-end — §16 is the phase-by-phase execution runbook (setup → training →
> evaluation → figures → LaTeX paper), §17 the config reference, §18 the prompt templates.
> Where v5 is contradicted, THIS document wins.

---

## 0. Executive summary (read this first)

**One sentence.** We train a small stopping-value model on *measurable* hindsight labels
(quality − λ·cost Snell-envelope targets computed post-hoc from the agent's own trajectories,
zero extra rollouts), then use its cost-aware value as a *potential-based process reward* to
RL-train the executor agent — producing agents that *internalize* when to stop instead of being
externally throttled, and showing this internalization beats both inference-time monitors and
direct reward-shaping without a stopper.

**Why now / why us.** Every component exists in 2026 literature; the *composition* does not.
Verified across 5 independent novelty audits and 10 literature areas:

- Hindsight stop labels + small trained stopper exist **inference-time only** (TERMINATOR
  2603.12529, OS-Pruner 2607.11089, LYNX 2512.05325, BAGEN 2606.00198, CaRT 2510.08517).
- Learned monitor → process reward → executor RL exists **quality-only** (Agent-RRM 2601.22154,
  MaR 2605.23384, AgentPRM 2502.10325 & 2511.08325, SWEET-RL 2503.15478, RePro 2606.14302).
- Cost-aware agent RL exists **outcome-level only** (OTC-PO 2504.14870, SlimSearcher 2606.07074,
  EAPO 2606.02132, RaM/VOC 2410.05563).
- **Nobody** converts an explicit quality−λ·cost hindsight optimum into a trained stopping-value
  model whose continuous margin Δ trains a tool-using executor at the step level — and nobody has
  measured whether stopping economics can be *internalized* into the policy rather than enforced
  at runtime. An orchestration survey (2605.02801) independently states the stop decision is never
  an RL target. This is the paper.

**The one claim the paper lives or dies on (pre-registered):** an executor trained with
stopper-derived process rewards Pareto-dominates (i) the same stopper used as an inference-time
controller only, (ii) direct oracle-label reward shaping with no stopper (DASH-style), and
(iii) training-free monitors (SupervisorAgent-style, confidence probes) — on cost-at-iso-accuracy
across ≥2 agent domains. Kill-switch experiments (§12) test this in week 2–3 before any further
investment.

**What changed vs v5 (why v5 would have been rejected):** see §14 changelog. Headlines: the
"first self-reinforcing cycle" claim is dropped (falsified by iStar/Self-Guide/Cooper/SPARK +
AgentPRM's own loop); the O(K×T²) complexity story is dropped (mischaracterized AgentPRM;
rollout-free PRM labels already exist — Implicit PRM, PRIME, TD+GAE AgentPRM); the "static
penalty" strawman is dropped (ALP/DAST/LASER-D/EAPO are adaptive); the prophet-biased oracle is
replaced with Snell-envelope labels; the additive Δ reward is replaced with potential-based
shaping; the reward-hacking channel is closed; SWE-bench RL and no-train-split benchmarks are
rescoped; fabricated result-sounding numbers are removed.

---

## 1. Problem & motivation

### 1.1 The missing training signal

LLM agents operate in open-ended loops (reason → tool call → observe → repeat) with no natural
endpoint. RLHF rewards helpfulness and task completion; **no training signal ever rewards
recognizing that further work is not worth its cost**. The result (all figures verified from the
papers, see `research/lit_review/01_overthinking_empirics.md`):

- Overthinking is real and costly: reasoning accuracy follows an inverted-U in chain length —
  beyond the optimum, more thinking *reduces* accuracy (Wu et al. 2502.07266); the shortest of 20
  sampled chains is +34.5 points more accurate than the longest (Hassid et al. 2505.17813);
  inverse scaling in test-time compute is documented at TMLR level (Gema et al. 2507.14417).
- In *agents* specifically: overthinking scores predict task failure (R² = 0.892) and selecting
  low-overthinking runs cuts cost 43% while *improving* success +24% (Cuadron et al. 2502.08235);
  outcome-only RL *causes* over-search (DAS 2602.03304, WWW'26).
- The dual failure exists too: hard problems genuinely need up to 2.9× more tokens
  (Chiang & Lee 2401.11467; Wu 2502.07266) — uniform throttling breaks hard instances.

### 1.2 Why prompting is not enough (and why this justifies *training*)

The obvious cheap fix — ask the model "are you done?" — measurably fails at economic judgment:
GPT-4-as-stopping-judge leaves a large oracle gap under matched budgets (Reasoning in Token
Economies, 2406.06461, EMNLP'24), and on a 2026 benchmark purpose-built for this, LLMs detect
redundant agent steps at ≤ 24.88% step-level F1 (RedundancyBench, 2605.29893). Calibrated scalar
confidence probes are stronger but are quality-only — they know "am I right," not "is the next
step worth its price" (and LearnStop 2606.30852 shows when they win/lose). **The stopping decision
is an economic estimate — marginal value vs marginal cost — and no current signal supplies it.**

### 1.3 Positioning sentence (the accurate 2026 gap)

> Existing adaptive-efficiency methods set cost pressure *before* generation (solve-rate-scaled
> penalties: ALP, DAST, LASER-D; difficulty tags: AdaCtrl; budget pre-commitment: TALE,
> Plan-and-Budget) or shape *trajectory-level* rewards (tool-call counts: OTC-PO, EAPO,
> SlimSearcher), and existing stopping methods control a *frozen* policy at inference time
> (TERMINATOR, OS-Pruner, LYNX, SupervisorAgent, Ares, BAGEN). **None learns a state-dependent
> economic stopping value from observed mid-trajectory progress and reuses it as a process reward
> to train the agent.** We do, and we show the training is what makes the difference.

### 1.4 Relationship to the cost-aware-agent harness (this repo)

The repo's `VISION.md` and metering daemon supply *inference-time* budget state (real dollar
costs per step) to frontier agents. CASSI is the *training-side* counterpart: it uses exactly the
budget-state features the harness already computes (tokens, tool fees, dollars, burn rate) as the
stopper's input schema, and produces executors that need less external throttling. The harness is
the deployment story; CASSI is the learning story.

---

## 2. Method — CASSI (Cost-Aware Stopping Supervision, Internalized)

Three components: (A) measurable stopping-value labels from completed trajectories;
(B) a small stopping-value model trained on them; (C) executor RL with potential-based
process rewards derived from the stopper. Then the loop (D) iterates A→B→C ≥ 2 times.

### 2.1 Setup and notation

**The decision process (formal, briefly).** Episodic loop: at each step t the executor
π_φ (8–9B-class instruct model; §19 fixes the exact choice) chooses an action
a_t ∈ {reason, tool-call, ANSWER}; emitting ANSWER is the stop action and ends the episode;
episodes also end at T_max. τ denotes the step at which ANSWER was emitted (or T_max).
**During training rollouts the monitor never truncates** — exploration must be free, and
economics reach the policy only through rewards; monitor *enforcement* exists only at
inference (§2.5). **Two rollout modes (critical distinction):** *label-collection* rollouts
(P2/P7) **suppress termination** — ANSWER is logged as a draft event and the trajectory
force-continues to T_max, so continuation values are observable from every state; without this,
labels are censored by the policy's own stopping choices, and iteration-2 data would vanish
exactly because internalization worked. (The logged would-have-answered positions double as a
free self-stop measurement.) *RL-training* rollouts (P6) terminate normally at ANSWER, so the
policy experiences real termination economics. Ground-truth answers/checkers are available at
training time only. For each trajectory and step t we record:

- `x_t` — **inference-available state features only** (hard requirement; §2.5): task text, step
  index, budget state (tokens used, tool calls, estimated dollars, % of allowance, burn rate),
  compressed recent history (last-K actions + observation digests), the current **running
  answer draft** (emitted by the executor each step as part of the shared agent template, §2.6 —
  present at training AND inference), and *harness-computed* self-consistency/novelty signals
  (e.g., answer-draft stability across steps, retrieval-overlap ratio). **No ground-truth-derived
  quantities.**
- `q_t` — step-t quality, **defined per domain** (training-time label machinery only):
  QA domains = F1/EM of the step-t **running draft** (template-emitted, §2.6) vs ground truth —
  a free string comparison at collection; ALFWorld = **subgoal-completion fraction read directly
  from the environment state** (zero additional label cost on that domain).
- `c_t` — step cost; `C_t = Σ_{i≤t} c_i` — cumulative cost. Cost is **multi-dimensional and
  dollar-denominated**: `c_t = p_tok·(input+output tokens priced per model) + p_tool(tool type) +
  p_api(actual fees)`, reduced to dollars via the same price map the repo's harness vendors.
  For use in objectives, cost is **normalized per domain**: C̃_t = C_t / (median unconstrained
  spend from the P2 pilot), so λ is dimensionless and comparable across domains (raw dollars
  still reported everywhere).
- Stopping utility: `U_t = q_t − λ·C̃_t` (tier-scaled variant in §2.2; λ swept, §5).

### 2.2 (A) Measurable stopping-value labels — the Snell envelope, not the prophet

**v5's label (`t* = argmax_t U_t`, label CONTINUE before t*, STOP after) is foresight-biased and
we do not use it as the regression target.** Taking the argmax of the *realized future* of the
same trajectory trains the model to predict the expected *pathwise maximum*, which
prophet-inequality theory shows strictly upper-bounds every implementable (non-anticipating)
stopping rule — systematically overestimating the value of continuing, i.e., stopping too late
(Krengel–Sucheston; cf. Longstaff–Schwartz regression practice in optimal stopping). This was
independently flagged by our RL-expert audit and is fixable at identical O(T) cost:

Compute the **empirical Snell envelope** by backward recursion with cross-sectional regression
over the whole trajectory batch (Longstaff–Schwartz / fitted value iteration):

```
V_T(x_T)  = U_T
V_t(x_t)  = max( U_t ,  Ê[ V_{t+1} | x_t ] )        for t = T−1 … 1
Cont(x_t) = Ê[ V_{t+1} | x_t ]                       (continuation value)
Δ*_t      = Cont(x_t) − U_t                          (stop margin; >0 ⇒ continue)
τ*        = min{ t : U_t ≥ Cont(x_t) }               (optimal non-anticipating stop)
```

`Ê[·|x_t]` is fitted across all trajectories/tasks in the batch (G = 8 rollouts per task from
GRPO collection give the cross-section), so the label at x_t reflects the *conditional
expectation* over continuations — not the luck of one path. This directly answers the
hindsight-relabeling critique (Brandfonbrener et al. 2022; Dichotomy of Control): mean-of-max is
replaced by max-of-mean at each backup. Three precision notes: (i) the decision grid is **every
step on both domains** (the running draft is scored each step on QA; ALFWorld reads env subgoals
each step — §2.6), and the stopper evaluates on the same grid at inference (every k-th step only
in the overhead ablation A5); (ii) at t = T, Cont is undefined and a*_T := STOP by construction;
(iii) τ* is optimal **with respect to the collection policy's continuation dynamics** — stopping
is solved for the current policy, which is exactly why the loop (§2.7) re-solves it as the
policy improves.

*Closest structural precursor (verified 2026-07-16):* Stop-RAG (2510.14337, NeurIPS'25 workshop)
fits offline Q(λ) targets with a max{Q(STOP), Q(CONTINUE)} bootstrap over completed RAG
trajectories — a genuine fitted-Q backward recursion. It is quality-only (no cost term),
inference-time-only (controls a frozen pipeline), and has no optimal-stopping framing. Cite it;
our deltas are the λ·cost objective, the multi-dimensional cost, and the training bridge.
No LLM work uses Snell/Longstaff–Schwartz-style cost-aware continuation labels to train a
policy (searched 2026-07-16; OS-Pruner uses direct policy gradient on the stop distribution,
TERMINATOR uses first-answer-arrival positions — neither does backward recursion).

Label output per step, per λ: `(a*_t ∈ {STOP, CONTINUE}, Δ*_t normalized to [−1,1],
V*_t = V_t(x_t) unnormalized in quality units)` — the V* label exists because the executor's
shaping potential (§2.4) needs a value estimate, not just a decision margin.

**Learned allowance-conditioning (default; the rule-table variant is ablation A8).** The stopper
must react not only to what was *spent* but to what *remains* — and that sensitivity is learned,
not hand-written. Two ingredients:
1. **Randomized allowances at collection:** every **(task, GRPO group)** draws a wallet
   B ∈ {small, medium, large} (calibrated per domain from the round-0 spend distribution, §17),
   **shared by all G rollouts of that group** — group advantages must compare behavior under the
   same wallet, never confound behavior with wallet luck; B varies across groups, so allowance
   features in x_t (%, tier, burn rate) still vary independently of spend in the training data.
2. **Tier-scaled marginal costing in the labels:** cost is penalized at the shadow price
   prevailing when it is spent — `U_t = q_t − Σ_{i≤t} λ·m(tier_i)·c̃_i` with multipliers
   m = {HIGH: 0.5, MEDIUM: 1.0, LOW: 2.0, CRITICAL: 5.0}. This is a discretized Lagrangian
   relaxation of the budget-constrained stopping problem: as the wallet empties, the multiplier
   on further spend rises, so the Snell recursion (unchanged) yields earlier τ* under tighter
   remaining budgets. v5 applied this schedule as an *inference-time* heuristic (§22.4 there);
   here it enters the *training data*, so "stop earlier when nearly broke" is a learned,
   in-weights behavior — remaining-budget conditioning is exactly what BAAR (2602.21227) names
   as future work. Evaluated by the same-task-three-wallets study (E3).

**Complexity, stated honestly:** label *collection* is O(T) per trajectory on top of data the RL
loop already gathers, plus the running-draft template tokens and forced-continuation collection
overhead (§2.1, §2.6), which we price explicitly. We make
**no claim that other PRM training is O(K×T²)** — rollout-free step signals exist (Implicit PRM
2412.01981; PRIME 2502.01456; TD+GAE AgentPRM 2511.08325; GiGPO 2505.10978; SPA-RL 2505.20732).
Our efficiency point is narrow and true: **cost-aware *stopping* labels are free given
intermediate quality measurements** — and we benchmark our label quality against TD/GAE and
MC-style labels at matched compute (§5, E4).

### 2.3 (B) The stopping-value model M_θ

A small model (default Qwen3.5-2B; 0.8B/4B in ablations — same family as the executor to isolate
the method; SoTA-verified 2026-07-16, §19) with **three heads** trained on
(x_t → a*_t, Δ*_t, V*_t): cross-entropy on STOP/CONTINUE; MSE on the normalized margin Δ̂
(drives stop decisions, §2.5); MSE on the unnormalized value V̂ (supplies the shaping potential
Φ, §2.4). **The stopper is λ-conditioned:** λ is part of the input serialization (§18.1), and
ONE stopper is trained across all λ label sets — a single model that implements the whole
cost-sensitivity dial, rather than five per-λ models. (This is what makes the "principled λ
dial" real: at inference, changing λ in the input moves the stopping frontier without any
retraining.) Optionally emits a short
rationale for interpretability (ablation only — off in headline runs to keep overhead honest).
No RL stage for the stopper in the main method (v5's stopper-GRPO was unjustified complexity);
an SFT-vs-SFT+RL comparison remains as an ablation.

**Why a separate small model (honest version).** Not "representation conflict" as a theorem —
single cost-sensitive models demonstrably work (ALP, EAPO, DASH). The defensible reasons:
1. **Privileged information asymmetry** (the SWEET-RL argument): the stopper is trained on
   ground-truth quality (via labels) that the executor must never condition on at inference;
   a separate model cleanly separates what is learned from privileged data vs what is deployed.
2. **Reusability/transfer:** one stopper can supervise multiple executors (tested, E2) and
   provide inference-time control for frozen third-party executors.
3. Whether separation also wins on raw Pareto is an **empirical question we test at matched
   parameter count** (9B multi-task single model vs 9B executor + 2B stopper; §5, A2) — with
   the pre-registered possibility that it does not, in which case the paper's claim rests on
   transfer + controllability + privileged-information hygiene (fallback framing).

### 2.4 (C) Executor training — potential-based economic shaping

The executor's base objective is the economic outcome reward — **the same economy as the
labels** (same quality measure, same tier-scaled normalized cost; coach, worker, and labels
optimize one Lagrangian, not three):

```
R_base(trajectory) = Q_τ − Σ_{i≤τ} λ·m(tier_i)·c̃_i     (τ = executor's ANSWER step or T_max)
```

where Q_τ is the terminal quality in the *same measure the labels use* (QA headline: EM binary,
with an F1 variant reported; ALFWorld: task success).

v5's additive step reward `Σ_t α·Δ(s_t)` is **not used**: it is non-potential-based shaping that
pays the executor to *accumulate* promising steps (dawdling incentive — the opposite of the
goal; Ng, Harada & Russell 1999). Instead, define the potential as the stopper's cost-aware value
`Φ(x_t) = V̂_θ(x_t)` and shape:

```
Φ(x_t) = V̂_θ(x_t);   Φ(absorbing terminal) := 0     (the invariance convention)
r_t = γ·Φ(x_{t+1}) − Φ(x_t)          (potential-based ⇒ optimal policy preserved; γ = 1)
R_executor = Σ_t r_t + R_base + γ_fmt·format
```

**Consequence of the convention (stated so no one trips on it):** with γ = 1 and Φ(terminal) = 0,
the shaped terms telescope to −Φ(x_0) — a constant within each GRPO group (same task, same
start, same wallet). Trajectory-level group advantages are therefore **provably unaffected** by
the shaping; all of its effect arrives through **step-level advantage assignment**, which is
accordingly **mandatory, not optional**: per-step returns-to-go R_t = Σ_{t'≥t} r_{t'} + R_base
with group normalization (SHAPE-style segment credit / GiGPO-style step grouping). Dense credit
is the entire point of the bridge; the invariance guarantee is what makes it safe.

This is the paper's conceptual core, stated plainly: **the stopper converts the sparse economic
objective (success − λ·cost, known only at the end) into dense per-step economic feedback,
without changing what the optimal policy is.** Every step that moves the state toward "stop-worthy
with high utility" is paid its marginal contribution; steps past the Snell frontier receive
negative shaped reward automatically.

*Positioning (verified 2026-07-16):* PBRS inside group-relative RL is itself established — cite
and borrow, don't claim: SHAPE (2604.06636, ACL 2026) uses segment-level potential differences in
GRPO and explicitly analyzes/fixes the telescoping cancellation with a length-dependent dynamic
discount; TIPS (2603.22293) applies Ng-form turn-level PBRS to search tool agents (on PPO, with a
within-turn invariance proof and public repo); DIVER (2509.26209, ICLR 2026) injects intrinsic
rewards via PBRS for invariance. **Our claim is the potential's semantics, not the shaping
mechanics:** all published potentials are success-rate/teacher-likelihood/diversity-based —
none is a trained cost-aware stopping value, and no paper combines shaped value rewards with
stopping (nearest: SPAE 2601.03823, ad-hoc form, probe-based, no cost).

RL algorithm: GRPO with G = 8, with **estimator hygiene mandatory** (Dr. GRPO unbiased length
handling / DAPO fixes; token-level normalization), so token savings cannot be attributed to
length-bias artifacts — plus KL anchor β = 0.04, 3 seeds on headline tables.

**Anti-reward-hacking design (closes the Goodhart channel v5 left open):**
- M_θ input features are **harness-computed only** (budget arithmetic, retrieval-overlap,
  draft-stability string metrics); the executor's *stated confidence is never an input*.
  Honest caveat: harness-computed ≠ ungameable — the executor authors the draft, so the
  canonical within-iteration attack is *freezing a wrong draft to fake stability*. The channel
  is closed across iterations, not within them: the label refresh re-grounds stability features
  in GT quality, and the V̂-vs-reward divergence diagnostic below watches precisely this attack.
- The stopper is **refreshed on-policy each iteration** (§2.7) — frozen-RM overoptimization is
  the documented failure (AgentPRM measured 82%→70%; Gao et al. 2210.10760; PRIME's motivation).
- We track a **hacking diagnostic**: stopper-predicted V̂ vs measured task reward across training;
  divergence triggers stopper refresh (report the curves; reviewers will ask).

### 2.5 Inference protocol

Executor runs; M_θ evaluates x_t each step (or every k-th step; overhead ablation) and STOPs the
episode when Δ̂_t ≤ 0 — a single fixed threshold, because allowance-sensitivity is already *in
the weights* (§2.2 learned conditioning: the same mid-trajectory state yields a lower Δ̂ when
the remaining wallet is tight). The user's cost-sensitivity dial is the λ fed to the
λ-conditioned stopper at inference (§2.3) — sweeping it traces the stopping frontier on a fixed
executor, no retraining (λ-monotonicity of τ* is classical comparative statics — Topkis; cf.
2402.06999 — cited, not claimed). The v5-style alternative — allowance-blind stopper + hand-written δ(tier)
threshold table — is kept as ablation A8, and beating it is a supporting result for the
learned-over-rules thesis. **Internalization metric:** because the executor was *trained* with economic
shaping, it should stop itself before the monitor fires; we report (i) % episodes self-terminated
pre-monitor, (ii) cost/accuracy with the monitor disabled at test time — the cleanest evidence
that economics moved *into the policy*, which no inference-time-control paper can show.

### 2.6 The running draft & quality scoring (label machinery, honestly priced)

**The draft lives in the agent template, not in probes.** The shared agent scaffold — used by
CASSI and every baseline alike, so it's a constant, not an advantage — requires the executor to
emit a one-line "best answer so far" field at each step (§18.2). Its tokens are counted in c_t
for every method. This resolves the train/deploy symmetry requirement: the draft (and the
stability features derived from it) exists identically at collection, RL training, and
inference. Quality scoring is then free reading: `q_t` = F1/EM of the logged draft vs gold (QA)
or the env subgoal fraction (ALFWorld) — a string comparison at collection time, no extra
generation. The **decision grid is every step on both domains.** The v5-style forked
"answer-now" probe is retained only as a validation ablation (does forced-answer quality ≈
running-draft quality?), not as machinery. Scoring uses ground truth and runs at collection
time only; the draft itself is always present. On
domains where per-step quality is intrinsically expensive (SWE-bench: a test-suite run per step;
intermediate patches score ~0 until late, degenerating τ* to the last step), **we do not train —
this is exactly why SWE-bench RL is out of scope (§5.1)**.

### 2.7 (D) The measured loop

AgentPRM-style iteration (we cite it as the template — Algorithm 1 there; also the co-evolution
genre: Cooper 2508.05613, SPARK 2509.22624, Self-Guide 2604.03098, iStar 2509.19199): collect
with π_i → Snell labels → refresh M_θ → shape-train π_{i+1}. **v5 called iteration "optional";
here ≥2 full iterations are mandatory and reported per-iteration** (cost, accuracy, stopping
regret at each i), **with the frozen-coach control arm at matched compute (E5)** so loop gains
are separable from "more RL steps" — without this, no loop language survives review. We claim a *cost-aware
stopping-centric instantiation* of the known loop, never "the first self-reinforcing cycle."

---

## 3. Contributions (each mapped to the experiment that proves it)

1. **The cost-aware stopping bridge** — first method to convert explicit quality−λ·cost hindsight
   optima into a trained stopping-value model whose continuous margin trains a tool-using
   executor as a potential-based process reward. Proven by: bridge vs controller-only vs
   DASH-style direct shaping vs training-free monitors (E1; kill-switch K1).
   *Nearest misses:* TERMINATOR/OS-Pruner (labels+stopper, inference-only, token-cost only),
   DASH (post-hoc labels as advantages, no stopper/cost), Agent-RRM/MaR (monitor-as-reward,
   quality-only), OTC-PO/EAPO/SlimSearcher (cost-aware RL, outcome-level).
2. **Internalized economic behavior** — first measurement of whether stopping economics can be
   trained *into* the policy vs enforced at runtime: monitor-off evaluation + self-termination
   rates + transfer of the trained executor to unseen domains (E2).
3. **Measurable stopping-value labels for agents** — Snell-envelope/fitted-Q labels from
   already-collected trajectories (zero extra rollouts; running-draft tokens and
   forced-continuation overhead priced explicitly), with a bias
   analysis showing why prophet-style argmax labels (the naive choice) stop late; label quality
   benchmarked against TD/GAE and MC labels at matched compute (E4). *Framed as the correct
   transfer of Longstaff–Schwartz/Hansen–Zilberstein machinery to LLM-agent stopping — classical
   roots cited, not claimed; closest precursor Stop-RAG (2510.14337, fitted-Q stopping controller,
   quality-only, inference-only) cited, with our deltas being the λ·cost objective and the
   training bridge.*
4. **Multi-dimensional dollar-denominated stopping objective with learned allowance-conditioning
   and a controllable λ dial** — tokens + tool fees + real dollars in U_t; wallet-awareness
   trained in via randomized allowances + tier-scaled marginal costing (§2.2), not rule tables;
   λ-swept Pareto frontiers and the same-task-three-wallets study (E3). All close competitors
   price tokens only or count calls, and remaining-budget conditioning is explicitly named as
   future work by BAAR (2602.21227).
5. **An honest map of when learned stopping helps** — matched-parameter single- vs two-model
   comparison, matched lost-correct-risk protocol (LearnStop), overhead per serving regime
   (KV-fork vs re-prefill), low-slack control (ablations A1–A9 + the MATH-500/AIME-2025
   control). Framed as findings either way.

Explicit **non-claims**: no "first self-reinforcing cycle" (iStar/Self-Guide verbatim-occupy the
phrase; AgentPRM implements the loop); no O(K×T²)-vs-O(T) PRM-training complexity story; no
"representation conflict theorem"; no novelty for per-instance difficulty adaptation per se
(ALP/DAST/Ares own it) or for a small model supervising a large one (Ares/TAB/SupervisorAgent).

---

## 4. Related work — positioning table (full review: `research/lit_review/`)

| Family | Representatives (verified IDs) | What they do | What they lack (our delta) |
|---|---|---|---|
| Hindsight stop labels + trained stopper | TERMINATOR 2603.12529 · OS-Pruner 2607.11089 (objective `acc − λ·tokens`!) · LYNX 2512.05325 · LearnStop 2606.30852 · Chen et al. ICML'20 | post-hoc exit labels → small stopper, CoT | inference-only; token-cost only; never trains the executor |
| Agent early-stop / termination | CaRT 2510.08517 · BAGEN 2606.00198 · SIM-RAG (SIGIR'25) · DAS 2602.03304 (WWW'26) | trained stop/continue for agents | no λ·cost objective (CaRT: γ-discount; BAGEN: no bridge); DAS trains via DPO but boundary-only |
| Monitor/controller over frozen agents | SupervisorAgent 2510.26585 (ICLR'26, −29.7% GAIA tokens) · Ares 2603.07915 (per-step effort router, hindsight labels) · TAB 2604.05164 · CoRL 2511.02755 · VoI control 2605.05701 · Dynasor 2412.20993 | runtime cost control | policy frozen — economics never enter the weights |
| Cost-aware agent RL (single model) | OTC-PO 2504.14870 · EAPO 2606.02132 · SlimSearcher 2606.07074 · AdaTIR 2601.14696 · Agent-Omit 2602.04284 (ICML'26) · RaM/VOC 2410.05563 | cost terms in outcome reward | trajectory-level scalar; no state-dependent stopping value; no stopper to transfer |
| Adaptive length control (reasoning) | ALP 2506.05256 (NeurIPS'25 spot.) · DAST 2503.04472 · LASER-D 2505.15612 · AdaptThink 2505.13417 · AdaCtrl 2505.18822 · HAPO 2505.11225 · L1 2503.04697 (COLM'25) · Arora&Zanette 2502.04463 (NeurIPS'25) | per-instance token budgets via RL | single-shot CoT tokens; pre-generation pressure; no tools/fees; no mid-trajectory value |
| Agent PRMs / step credit | AgentPRM 2502.10325 (pooled return-to-go, iterative loop — NOT per-state MC) · AgentPRM-Fudan 2511.08325 (WWW'26, TD+GAE) · Agent-RRM 2601.22154 · SWEET-RL 2503.15478 · GiGPO 2505.10978 · SPA-RL 2505.20732 · PRIME 2502.01456 · Implicit PRM 2412.01981 · DASH 2607.00482 · RePro 2606.14302 · MRT 2503.07572 / PAV 2410.08146 | dense step signals for policy training | success/progress semantics only — **no PRM encodes cost**; no stopping margin |
| Loops (policy↔reward co-evolution) | iStar 2509.19199 · Self-Guide 2604.03098 · Cooper 2508.05613 · SPARK 2509.22624 · Self-Rewarding 2401.10020 · ReST/STaR lineage | iterate policy and evaluator | none cost-aware; we instantiate, not invent |
| Reward shaping in LLM RL | SHAPE 2604.06636 (ACL'26, PBRS in GRPO + telescoping fix) · TIPS 2603.22293 (turn-level PBRS for tool agents) · DIVER 2509.26209 (ICLR'26) · Stop-RAG 2510.14337 (fitted-Q stopping controller) · SPAE 2601.03823 | shaping mechanics / fitted-Q stopping | potentials are success/teacher/diversity-based — none is a trained cost-aware stopping value; Stop-RAG is quality-only + inference-only |
| Theory | Hansen & Zilberstein AIJ'01 (anytime monitoring = our oracle) · Weitzman '79 · Russell & Wefald '91 · Hay '12 · prophet inequalities · Topkis/2402.06999 · Cognitive Friction 2603.30031 (HJB for tool agents) | optimal stopping & metareasoning | we transfer (with citation) and add the LLM-agent instantiation |
| Motivation empirics | Cuadron 2502.08235 · Wu 2502.07266 · Hassid 2505.17813 · Gema TMLR'25 · Token Economies 2406.06461 · RedundancyBench 2605.29893 · CTA 2602.16699 (cost-discounted GRPO alone fails) | document overthinking + prompting's failure | we use them as motivation & baselines |

Corrections carried from the review (v5 errors that must not recur): AgentPRM ≠ per-state MC
(pooled return-to-go); IterResearch is RL-trained (2511.07327), not a heuristic; BATS =
Google 2511.17006 prompt-level budget tracker; "Learning to Stop While Learning to Predict" is
Chen et al.; two AgentPRMs must be disambiguated; CARL 2512.04949 authorship unverified.

---

## 5. Experimental design

### 5.1 Domains (rescoped to what is actually trainable — audit: novelty_agent_4 §feasibility)

| Domain | Role | Env/data | Why |
|---|---|---|---|
| Multi-hop QA + search | **Primary training** | NQ + HotpotQA + MuSiQue train splits (the Search-R1/OTC-PO corpus mix), Search-R1 retrieval env via verl-tool (local Wikipedia index; Qwen3.5-9B executor) | the exact training data of the 2026 Search-R1 successors (Search-R2, CoSearch, Search-E1) AND of OTC-PO — cost comparisons stay commensurable; cheap draft scoring (F1 vs gold ≈ free, §2.6) |
| Embodied/web tasks | **Second training domain** | ALFWorld via verl-agent (GiGPO harness) | train split + 2026 RL precedent (Agent2 RL-Bench, RWML); its success *saturation* is a feature for us — differences become pure efficiency (report steps/cost at matched success) |
| Web research | **OOD transfer eval only — split by role (tool-stack mismatch resolved)** | **Trained-executor transfer:** BrowseComp-Plus (830 Qs, fixed 100K-doc local corpus, ACL 2026 — local retrieval matches the training tool type) + Bamboogle (125) + 2WikiMultihopQA (500-dev). **Stopper-as-monitor transfer:** GAIA text-only 103-Q dev (validation used as test — disclosed; test set hidden), stopper monitoring a strong frozen live-web agent — exactly SupervisorAgent's setup, so the ICLR'26 comparison is direct; search-time-contamination caveat (2606.05241) reported. Trained-executor-on-GAIA-with-live-tools: exploratory appendix only (the executor never saw live-web tools in training — running it as headline would confound stopping transfer with tool-API shift) | no train splits (v5 error); each eval's tool stack now matches what the evaluated component saw |
| Math | **Low-slack control** | MATH-500 **+ AIME 2025**, inference-time monitor only | MATH-500 kept for ALP/DAST/LASER comparability — but ALP shows savings *concentrate* there (it's the easy set); AIME 2025 is the genuine can't-compress control |
| SWE | **Dropped from RL** (appendix at most: inference-time monitor on frozen 32B agent) | — | 7B SWE RL doesn't work (DeepSWE needed 32B+cluster); per-step quality = test-suite run per step; intermediate quality ≈ 0 degenerates τ* |

### 5.2 Baselines (~9, each with its purpose; all IDs verified)

| # | Baseline | Type | Kills the question |
|---|---|---|---|
| B1 | ReAct (no cost signal) | lower bound | how much slack exists |
| B2 | Zero-training self-eval prompt + **calibrated confidence exit** (Dynasor-style scalar probe) | training-free | "why not just ask/probe?" — *dangerous baseline, LearnStop shows it can win* |
| B3 | **SupervisorAgent-style training-free monitor** (2510.26585; a documented protocol *adaptation* to our single-agent envs — appendix) | training-free monitor | the ICLR'26 bar: −29.7% tokens at parity on GAIA |
| B4 | **OTC-GRPO** (2504.14870) | cost-aware RL, outcome-level | is step-level economic signal needed at all |
| B5 | **EAPO** (2606.02132; primary — agentic tool domain, same env family) — agentic-ALP (2506.05256) only as fallback if EAPO proves unreproducible | published adaptive penalty (replaces v5's homemade "adaptive-α") | is a learned VALUE better than adaptive scalar pressure |
| B6 | Single-model GRPO + cost-in-reward (CTA-style, 2602.16699 showed it under-internalizes) | single-model | two-model necessity, matched params w/ A2 |
| B7 | CaRT + cost (2510.08517) — **two arms: SFT-only and +GRPO** | trained self-termination | v5's primary comparison, retained; the SFT-only arm (imitate oracle-truncated trajectories, no RL) answers "is RL even needed, or does imitation suffice?" |
| B8 | **AgentPRM-cost, honestly implemented** (pooled return-to-go + cost term; optionally TD+GAE per 2511.08325) | quality-PRM+cost | is the *stopping-value semantics* what matters vs generic value+cost |
| B9 | **DASH-style direct shaping** (2607.00482 adapted: Snell labels → advantages directly, NO stopper — implemented with CASSI's exact step-level machinery, only the stopper deleted; trajectory-level shaping is provably inert (§2.4), so anything less would be a strawman) | ablation-as-baseline | **the pivotal test: does the stopper earn its existence** |
| — | Oracle stopping (Snell τ* with GT) | upper bound | headroom |

### 5.3 Headline metrics

- Cost at iso-accuracy & accuracy at iso-cost (dollar-denominated, all-inclusive), Pareto AUC.
  **Frontier protocol (iso-metrics are undefined without it):** every method is swept over ITS
  OWN cost knob to produce a 3–5 point frontier — CASSI: the inference-time λ dial; B2:
  confidence threshold; B3: trigger sensitivity; B4: tool-count coefficient; B5: penalty weight;
  B6/B7/B9: their λ. Iso-accuracy cost is read by linear interpolation between adjacent frontier
  points; methods with no knob (B1 ReAct, oracle) are reported as single points and **excluded
  from iso-claims**. Trained-baseline frontier points are 1-seed except each method's headline
  operating point (3 seeds).
- **Stopping regret** vs Snell-optimal τ* (utility gap, not |t−t*| — v5's metric ignores
  magnitude), and matched lost-correct-risk curves (LearnStop protocol 2606.30852).
  **Measurement protocol (post-stop counterfactuals don't exist in normal eval traces):** on a
  fixed 500-task subsample, dual runs — one normal (real cost/accuracy) + one
  forced-continuation replay to T_max (yields the U_t curve and τ*); regret = replay-frontier
  utility minus the method's actual-stop utility. Replay cost is billed to the analysis line of
  T4, not to any method.
- **Internalization**: monitor-off cost/accuracy; % self-terminated episodes; both across
  iterations i = 0, 1, 2.
- **End-to-end honest accounting** (audit demand): total $ incl. running-draft template tokens,
  forced-continuation collection overhead, stopper training, stopper inference, per **serving
  regime** (KV-cache-fork vs black-box re-prefill —
  overhead can flip sign: 2606.30852). Amortization vs training-free baselines' zero amortization.
  **Billing symmetry:** every method pays for ALL auxiliary inference it uses — B2's self-eval
  prompts, B3's monitor triggers, our stopper's calls, any judge — under the same price map.
  Reporting follows the HAL harness conventions (princeton-pli/hal-harness, 2510.11977) — the
  2026 standard for agent cost/token accounting and multi-run consistency (pass^k across seeds).
- Overoptimization diagnostics: V̂ vs realized reward divergence curves; plus per-step backup
  residuals of the label regressor on held-out trajectories (fitted-value-iteration error
  compounding check — feeds E4).
- External validity: RedundancyBench (2605.29893) step-redundancy detection F1 of our stopper
  (turns a threat paper into an eval); OptimalThinkingBench (2508.13141) as a cheap agent-free
  over/under-thinking check. CostBench (2511.02734) is cited, not run (cost-optimal *planning* —
  a different construct).

### 5.4 Experiments → claims map

| ID | Experiment | Proves |
|---|---|---|
| E1 | Full grid on the two training domains: CASSI vs B1–B9 (3 seeds at headline operating points; frontier points 1-seed per §5.3) | Contribution 1 |
| E2 | Monitor-off + self-termination; OOD transfer **by role** (§5.1): trained executor → BrowseComp-Plus/Bamboogle/2Wiki; stopper-as-monitor → GAIA-103 over a frozen live-web agent (SupervisorAgent-comparable); stopper transferred across executors (Qwen3.5-9B → Ministral-3-8B cross-family; → Qwen3.5-4B cross-scale) | Contribution 2, H-transfer |
| E3 | λ-frontier protocol: **executor trained at headline λ=1 only**; the Pareto frontier comes from the inference-time λ dial (λ-conditioned stopper, §2.3) on that fixed executor, + ONE λ=0.3 executor as a spot-check that the inference dial tracks the trained frontier (never 5 executor trainings — that's the v5 matrix explosion). Plus the **same-task-three-wallets study** (identical test tasks under small/medium/large allowances: stop-steps shift earlier, answers degrade gracefully, no rule anywhere) | Contribution 4 |
| E4 | Label study: Snell vs prophet-argmax vs TD/GAE vs MC-style labels — stopper accuracy + downstream executor result at matched label-compute (draft-token + forced-continuation costs included) | Contribution 3 |
| E5 | ≥2 loop iterations **with the decisive control**: iteration 2 is run twice at matched compute — (a) frozen iteration-1 coach ("more RL steps" control) vs (b) refreshed coach; **the (b)−(a) delta IS the loop's contribution** (without it, iteration-2 gains are confounded with extra training) | loop honesty |
| E6 | Difficulty-stopping correlation (2-hop vs 4-hop MuSiQue), demoted to consistency check (precedents: ALP; Wu r=0.57) | supports C1/C2, no novelty claim |

### 5.5 Ablations

A1 stopper size (0.8B/2B/4B; ReMA warns ~1B meta-models can collapse — verify floor) ·
A2 single-model multi-task vs two-model **at matched total params** ·
A3 potential-based vs additive Δ reward (predicted: additive dawdles — validates §2.4) ·
A4 stopper input families (budget-only / +history / +draft-stability) ·
A5 stopper eval frequency (every step vs every k-th) + legacy-probe vs running-draft label-source check (overhead-savings frontier) ·
A6 SFT-only stopper vs +RL calibration · A7 rationale on/off ·
A8 **learned allowance-conditioning vs rule table** (default §2.2 vs v5-style: fixed-allowance
training, plain λ labels, hand-written δ(tier) thresholds at inference) ·
A9 **negative controls** (cheap, decisive): (i) random-coach — Δ̂/V̂ replaced by noise; if the
executor still improves, gains were generic dense-reward regularization, not economics;
(ii) shuffled-label coach — trained on permuted labels; isolates whether label *content*
matters. Both on the primary domain, 1 seed.

### 5.6 Statistical & reporting protocol (restored from v1 §26, adapted)

- **Seeds:** 42/123/789 for every headline number (T1/T2); single-seed allowed for ablations
  A1–A9 and non-headline frontier points, per §7's tiering (say so in every caption).
  Evaluation decoding: temperature 0 for executors; the stopper is a deterministic threshold
  rule at eval time.
- **Uncertainty:** bootstrap over test instances (10,000 resamples), 95% CI on every reported
  metric; seed variability reported separately as mean ± s.d. across seeds.
- **Comparisons:** within-instance pairing (identical task sets per method) — paired t-test AND
  Wilcoxon signed-rank, both reported; Wilcoxon governs when normality fails (cost distributions
  are heavy-tailed).
- **Pareto dominance:** bootstrap test — resample instances, recompute both frontiers, report the
  fraction of resamples in which CASSI's frontier dominates at every shared accuracy level.
- **Multiple comparisons:** Holm–Bonferroni across the 9 baselines within each domain (replaces
  v1's plain Bonferroni; state the comparison family explicitly in the appendix).
- **Effect sizes:** Cohen's d for dollar-cost deltas; absolute risk difference (percentage
  points) for accuracy.
- **Stopping-rule fairness:** matched lost-correct-risk curves (LearnStop 2606.30852 protocol):
  sweep each method's own threshold and compare savings at equal fractions of lost-correct
  answers (1%, 2%, 5%).
- **No-cherry-picking clauses:** all λ values and all seeds appear in the appendix; the headline
  λ is chosen on dev *before* test runs; kill-switch GO/NO-GO decisions (§12) are logged with
  dates in the repo.
- **Small-n policy:** hypothesis tests run ONLY on sets with n ≥ 500 (HotpotQA/MuSiQue eval
  subsamples, 2Wiki, BrowseComp-Plus). GAIA-103 and Bamboogle-125 are reported as point
  estimates with bootstrap CIs, explicitly labeled *transfer indicators, not hypothesis tests*
  (at n=103 an accuracy CI is ≈±9pp — "significant" 3-point claims there are impossible and
  claiming them invites a statistics rebuttal).
- **Frozen eval subsamples:** chosen once, before any method runs — HotpotQA-dev 1,000 /
  MuSiQue-dev 500 / small sets in full; every method, seed, and frontier point evaluates the
  identical task lists.
- **Contamination protocol** (2018–2022 QA data under a 2026 base model is a certain reviewer
  question; documented risk: 2507.10532 shows Qwen-family benchmark contamination can distort RL
  conclusions): (1) n-gram/MinHash decontamination of all training prompts against every eval
  set; (2) adversarially-fresh OOD sets in the suite (Bamboogle; BrowseComp-Plus fixed corpus);
  (3) a closed-book no-tool ablation showing parametric memory alone cannot solve the evals —
  gains must come from tool use; (4) replication on a second model family (the Ministral transfer
  arm doubles as this control). Plus the claim-level defense: **our headline metric is cost at
  matched accuracy — memorization inflates accuracy for all compared methods equally, not the
  cost delta between them.**

---

## 6. Hypotheses (pre-registered, falsifiable, with fallback framings)

| ID | Hypothesis | If it fails → |
|---|---|---|
| H1 | CASSI Pareto-dominates B2/B3 (training-free) on ≥2 domains | paper becomes "when does learned stopping help agents" regime study (LearnStop-style; still publishable, different claim) |
| H2 | Bridge > controller-only (same stopper) | **kill-switch: abandon/pivot (§12)** |
| H3 | CASSI > B9 direct shaping (stopper generalizes beyond labeled states + gives inference control) | stopper demoted to optional; paper = "Snell-label direct shaping for agents" |
| H4 | Two-model ≥ single multi-task at matched params on Pareto | claim rests on transfer + privileged-info hygiene + controllability (pre-registered fallback) |
| H5 | Trained executor retains ≥70% of cost savings with monitor off | internalization claim softened to "partial internalization" |
| H6 | Savings hold under both serving regimes | report regime-conditional recommendation |

---

## 7. Training infrastructure & budget (realistic)

- Framework: verl ≥ v0.8.0 (AgentLoop multi-turn rollouts; Dr. GRPO/DAPO hygiene) with the
  Search-R1 env run via verl-tool, and ALFWorld via verl-agent (GiGPO's official harness);
  stopper via HF TRL v1.8 (SFTTrainer + scalar reward head — not the legacy value-head wrapper).
  Cost/consistency reporting follows the HAL harness conventions. Exact repos + rebuttal lines:
  §19 manifest. Hardware: the 8×H200 node.
- One 9B multi-turn GRPO run ≈ 1–3 days on 8×H200 (agentic-RL survey estimates). Budget ≈ 30–35
  training runs, tiered: CASSI (2 domains × [iter-1 + iter-2×2 arms]) + trained baselines B4–B9
  (each with 2–3 frontier points, 1 seed except headline points) + training ablations (A2, A3,
  A6, A9's two controls) on the primary domain, 1 seed + 3 seeds on headline operating points.
  **Inference-only ablations (A1's controller part, A4, A5, A7, A8) reuse trained models at
  near-zero cost** — this tiering is what keeps the matrix inside budget.
  **Total ~10–12 weeks**, not v5's 8 (audit: v5 matrix was ~10× over budget).
- Week-by-week: W1–2 env + draft-template scaffold + collection; W2–3 **kill-switches K1/K2**; W4–6 main E1;
  W7–8 iterations E5 + label study E4; W9–10 transfer/ablations; W11–12 analysis + writing.
  Full phase-by-phase execution detail with done-criteria: **runbook §16** (P0–P11, ending at
  the compiled LaTeX paper).

---

## 8. Risks (updated — the ones v5 missed)

| Risk | Mitigation |
|---|---|
| Reward hacking of learned Δ (documented: 82%→70% in AgentPRM) | objective-only stopper inputs; per-iteration stopper refresh; divergence diagnostics (§2.4) |
| Training-free baselines match us (B2/B3) | H1 fallback framing; low-slack control shows we don't break hard tasks |
| Direct shaping matches us (B9) | H3 fallback; stopper still yields transfer + runtime control |
| Single-model parity (A2) | H4 fallback (transfer/hygiene/controllability) |
| Label noise from mid-trajectory drafts | Snell regression pools G=8; legacy-probe validation check in A5; report label-noise sensitivity |
| GRPO pathologies (length bias, entropy collapse, Echo Trap) | Dr. GRPO/DAPO hygiene; GiGPO-style anchors if credit dilution appears |
| **Scooped** (DASH/RePro/SlimSearcher/BAGEN are ≤7 weeks old; CMU CaRT/MRT group adjacent) | move fast; kill-switches at W2–3; workshop preprint of K1 result early |

---

## 9. Paper plan (8 pages + appendix) — sections, figures, tables, sources

The writing *process* (LaTeX workflow, drafting order, compile loop) is Phase 10–11 of the
runbook (§16). This section fixes WHAT the paper contains and where each piece comes from.

| § | Section (page budget) | Content & source |
|---|---|---|
| 1 | Introduction (1.25) | overthinking/agent-waste empirics with numbers (§1.1); prompting fails at economics (§1.2); positioning sentence verbatim (§1.3); contributions C1–C5 (§3). Fig. F1. |
| 2 | Related work (1.0) | §4's families table rendered as prose+table; explicit deltas; the corrected AgentPRM reading stated in text |
| 3 | Problem formulation (0.75) | episodic MDP; stopping utility U_t; Snell envelope, Cont, Δ, τ*; classical citations (Hansen–Zilberstein '01, Weitzman '79); prophet-bias motivation for the label choice |
| 4 | Method (1.5) | Algorithms 1–3 (compact); potential-based bridge + policy-invariance statement; anti-hacking design; the measured loop. Fig. F2 |
| 5 | Experimental setup (0.75) | domains (§5.1), baselines B1–B9 table (§5.2), metrics incl. regret & matched-risk (§5.3), protocol pointer (§5.6) |
| 6 | Results (1.75) | T1–T2, F3–F5; per-iteration loop gains (E5); hypothesis verdicts H1–H6 stated explicitly, including any failures |
| 7 | Analysis (0.75) | internalization deep-dive (monitor-off, self-stop rates); hacking diagnostics (F6); label study (E4); difficulty-consistency check (E6); qualitative stop examples |
| 8 | Limitations + Conclusion (0.5) | domains where per-step quality is expensive (SWE); serving-regime dependence of overhead; scooping-window honesty |

**Figure inventory:** F1 pipeline (labels → stopper → shaped GRPO → loop) · F2 shaping intuition
(U_t, Cont(x_t), Δ, τ* drawn on one real trajectory) · F3 cost-accuracy Pareto per domain (all
baselines, λ sweep) · F4 internalization (monitor-off bars + self-termination % across iterations
0/1/2) · F5 label study (Snell vs prophet vs TD/GAE vs MC: stopper regret + downstream executor
cost) · F6 V̂-vs-realized-reward divergence during training (hacking diagnostic).

**Table inventory:** T1 headline cost@iso-accuracy / accuracy@iso-cost (3 seeds, CIs) · T2 all
baselines × 2 training domains · T3 ablations A1–A9 · T4 end-to-end overhead accounting per
serving regime (incl. running-draft tokens + forced-continuation collection + stopper serving +
training amortization) · T5 transfer by role (executor swap; trained executor →
BrowseComp-Plus/Bamboogle/2Wiki; stopper-as-monitor → GAIA-103).

**Appendix:** stopper I/O + draft template (§18), feature schema (§11), configs (§17),
statistical detail (§5.6), full λ/seed grids, per-domain breakdowns, qualitative examples,
GO/NO-GO logs.

---

## 10. Algorithm summaries (implementation-ready)

```
ALGORITHM 1 — Label construction (per collection round, per λ)
input: trajectories D = {(x_t, q_t, c_t)_{t=1..T_i}}, regressor class R (default: gradient-boosted
       trees or 2-layer MLP on x-features; NOT the stopper itself, to avoid label-model coupling)
1  U_t ← q_t − Σ_{i≤t} λ·m(tier_i)·c̃_i   # tier-scaled, pilot-normalized costing (§2.1–2.2); m ≡ 1 in ablation A8
   # decision grid = every step on both domains (running draft scored per step; ALFWorld q from env state)
2  V_T ← U_T for each trajectory
3  for t = T_max−1 down to 1:
4      fit Ê_t ∈ R on {(x_t, V_{t+1})} pooled across ALL trajectories   # cross-sectional
5      V_t ← max(U_t, Ê_t(x_t)) per trajectory
6  Δ*_t ← Ê_t(x_t) − U_t ;  a*_t ← STOP if Δ*_t ≤ 0 else CONTINUE
7  return {(x_t, a*_t, tanh(Δ*_t / s), V_t)}   # s = per-domain scale, fit once; V_t unnormalized (shaping potential label)

ALGORITHM 2 — Stopper training
SFT M_θ on (x_t → a*_t) CE + (x_t → Δ*_t) MSE + (x_t → V*_t) MSE, 3 epochs, lr 2e-5, early stop
on held-out stopping regret (not CE).

ALGORITHM 3 — Executor GRPO with economic shaping (one iteration)
for each task batch: sample G=8 trajectories with current π_φ
    per step: r_t = γ·V̂_θ(x_{t+1}) − V̂_θ(x_t)   with V̂_θ(absorbing terminal) := 0, γ = 1
    terminal:  R_base = Q_τ − Σ_{i≤τ} λ·m(tier_i)·c̃_i  (+ format term)   # same economy as labels
    advantage assignment: STEP-LEVEL (mandatory — §2.4): per-step returns-to-go
        R_t = Σ_{t'≥t} r_{t'} + R_base, group-normalized (SHAPE/GiGPO-style)
        min-cohort guard: if <3 group members alive at step t, fall back to the
        trajectory-level baseline for those steps (variable-length groups make late-step
        cohorts tiny; K1 also picks the step-credit variant: SHAPE-segment vs per-step RTG)
    GRPO update with Dr.GRPO normalization, KL β=0.04
every iteration end: recollect, rerun Alg.1–2 (stopper refresh), log V̂-vs-reward divergence

ALGORITHM 4 — Inference
each k-th step (k=1 headline): serialize x_t (harness features + user's λ, §18.1) → M_θ → Δ̂_t
stop if Δ̂_t ≤ 0  or  executor emits final answer  or  budget exhausted
   # fixed threshold — allowance-sensitivity is in the weights (§2.5);
   # the δ(tier) rule table exists only in ablation A8
```

---

## 11. Data & feature schema (for the implementing agent)

`x_t` feature groups (ALL computable by the repo's harness at inference):
budget: {tokens_used, tokens_pct, tool_calls, tool_pct, dollars, dollars_pct, burn_rate_$_per_step,
tier} · progress: {step_idx, steps_since_draft_changed, draft_edit_distance_last3,
retrieval_overlap_last3, n_distinct_sources} · draft: {running draft (template-emitted, §2.6), draft_len} ·
task: {question text, domain tag} · history: {last-K (action-type, obs-digest-64tok) pairs}.
**Excluded by design:** any GT-derived quality, any executor-stated confidence.

Datasets: NQ + HotpotQA train (sample 8–10K combined, Search-R1/OTC-PO mix), MuSiQue train
(sample 5K), ALFWorld train tasks; eval: HotpotQA/MuSiQue dev, GAIA text-only 103-Q dev,
BrowseComp-Plus (830 Qs + its fixed 100K-doc corpus), Bamboogle, 2Wiki, MATH-500 + AIME 2025.
Trajectory store: JSONL per trajectory {task_id, allowance_B, steps[{x_t, a_t, o_t, c_t,
tier_t, draft_t, q_t, answered_flag}], outcome} — `answered_flag` marks forced-continuation
ANSWER events (§2.1, the free self-stop measurement); draft-line tokens are inside c_t; legacy
probe answers are logged only in A5 ablation runs.

---

## 12. Kill-switch protocol (run FIRST — weeks 2–3, small scale)

K1 (**bridge test**): HotpotQA 1K-task subset, 2B stopper + 9B executor, 1 seed:
Δ-shaped GRPO vs controller-only vs B9-direct-shaping. Proceed iff shaped-GRPO beats
controller-only by ≥3 points cost-at-iso-accuracy AND ≥ B9 (1 seed — read as direction +
magnitude, not significance). Else pivot per H2/H3 fallbacks.
K2 (**separation test**): same subset: 9B single multi-task (task+stopping heads) vs 9B+2B
(params counted in reporting). Any outcome is publishable via H4, but the framing changes.
GO/NO-GO review after K1+K2 before building anything else.

---

## 13. What success looks like

Headline: **20–40% dollar-cost reduction at iso-accuracy on two agent domains, surviving
training-free baselines (B2/B3), with ≥70% of savings retained monitor-off** — plus transfer of
one stopper across two executors, per-iteration loop gains, and honest overhead accounting per
serving regime. Secondary: Snell labels beat prophet labels (validating the fix), stopping regret
near-oracle on easy strata, no degradation on the MATH-500 + AIME 2025 controls.

---

## 14. Changelog v5 → v2 (what changed and the evidence that forced it)

| v5 claim/design | Status | Forced by |
|---|---|---|
| "First self-reinforcing cycle" | **dropped** → "cost-aware stopping-centric instantiation" of the known loop, ≥2 iterations measured | iStar 2509.19199 + Self-Guide 2604.03098 (verbatim phrase), AgentPRM Alg.1, Cooper/SPARK; v5 §10.1 self-contradiction |
| O(T) vs O(K×T²) headline | **dropped** → narrow "stopping labels are free given quality probes" + label-quality benchmark | AgentPRM misread (pooled return-to-go); Implicit PRM/PRIME/TD+GAE AgentPRM/GiGPO/SPA-RL rollout-free |
| Static-penalty strawman | **dropped** → positioning sentence §1.3 | ALP/DAST/LASER-D/AdaCtrl/HAPO/EAPO/SlimSearcher adaptive |
| Prophet argmax oracle t* | **replaced** by Snell-envelope labels (same O(T)) | prophet-inequality foresight bias (novelty_agent_2 Q1) |
| Additive Σα·Δ step reward | **replaced** by potential-based shaping | Ng et al. 1999; dawdling incentive; PAV/MRT |
| Stopper reads executor-stated confidence | **removed** (objective features only) + stopper refresh | AgentPRM 82%→70% hacking; Gao 2210.10760 |
| GT-derived features in monitor prompt | **removed** (train/inference mismatch) | novelty_agent_4 audit |
| Stopper GRPO stage | **demoted to ablation** (SFT value model default) | unjustified complexity |
| "Representation conflict" as theory | **reframed**: privileged-information asymmetry + matched-param empirical test | SWEET-RL argument; PPG both-ways evidence; HiPER/SPARK counter-evidence |
| Properties 1–3 "formal grounding" | **recast as classical citations** (Hansen–Zilberstein Def.4/Thm.1; Topkis) + P3 dropped | theory audit (07 file); P1 uniqueness false as stated |
| 7 benchmarks incl. SWE-bench RL, GAIA training, BFCL | **rescoped** per §5.1 | no train splits; SWE per-step tests; 10× over budget |
| 14+ baselines incl. ReMA-cost/BudgetThinker reimpls | **cut to 9 purposeful** incl. OTC-GRPO, EAPO/ALP, SupervisorAgent, DASH-style, confidence probes — all previously missing | audits 1/3/4/5 |
| H5 difficulty correlation as P0 contribution | **demoted** to consistency check E6 | ALP owns the story; Wu r=0.57 precedent |
| |t−t*| stopping error metric | **replaced** by utility regret + matched-risk protocol | LearnStop 2606.30852 |
| §27 "16% errors / 78% false-CONTINUE" (no experiments exist) | **deleted** (credibility bomb) | novelty_agent_4 |
| Missing citations | **added throughout**: OTC-PO, TERMINATOR, OS-Pruner, DASH, Ares (correct reading: hindsight labels), TAB, BAGEN, Agent-RRM, SupervisorAgent, SlimSearcher, EAPO, SIM-RAG, DAS, CTA, RaM/VOC, iStar, Self-Guide, LearnStop, LYNX, Cognitive Friction 2603.30031, classical stopping theory | all audits |
| v5 §22.4 inference-time tier λ-multipliers (rule table) | **moved into training**: randomized allowances at collection + tier-scaled marginal costing in labels → allowance-sensitivity is learned in-weights; rule table demoted to ablation A8; adds same-task-three-wallets eval to E3 | internal consistency (learned>rules thesis); BAAR 2602.21227 names remaining-budget conditioning as future work; SeqRoute precedent for budget-context relabeling |
| Qwen2.5-7B/1.5B + Llama-3.1-8B model choices (late-2024 models) | **refreshed to July-2026 SoTA**: executor Qwen3.5-9B, stopper Qwen3.5-2B (0.8B/4B ablations), transfer Ministral-3-8B-Instruct-2512 + Qwen3.5-4B; verl ≥0.8 with verl-tool + verl-agent; TRL v1.8 scalar head; HAL-style cost reporting; full manifest + rebuttal lines in §19 | outdated-reuse rebuttal risk (web-verified 2026-07-16: Qwen3.6 has no open <27B model; no open Llama <27B since 2024; verl-tool reproduces Search-R1 on modern verl) |
| Benchmark suite (v5-era: GAIA/WebWalkerQA/BFCL + MATH-500-only control) | **refreshed & defended (web-verified 2026-07-16)**: training = NQ+HotpotQA+MuSiQue (the Search-R1-2026/OTC-PO corpus) + ALFWorld; OOD eval = GAIA-103 + BrowseComp-Plus (new, reproducible) + Bamboogle + 2Wiki; control = MATH-500 + AIME 2025; contamination protocol added to §5.6; per-dataset rebuttal lines in §19 | ALP shows MATH-500 is where easy savings live → alone it can't prove hard tasks survive; live-web GAIA has documented search-time contamination (2606.05241); HotpotQA-era data needs the §5.6 contamination defense under a 2026 base model |
| Shaping mechanics claimed implicitly as ours | **repositioned (web-verified 2026-07-16)**: PBRS-in-GRPO + telescoping fix exist (SHAPE ACL'26; TIPS; DIVER ICLR'26) — cited and borrowed as recipes; label construction's closest precursor Stop-RAG (fitted-Q stopping controller) cited; claim narrowed to the potential's SEMANTICS (trained cost-aware stopping value) and the composition | final training-mechanism de-risking check: no paper combines shaped value rewards with cost-aware stopping (nearest: SPAE, ad-hoc/no-cost); P6 implementation now follows a published recipe instead of an untested one |
| Experimental-design flaws (internal review round 3, 2026-07-16) | **12 repairs**: per-baseline frontier protocol (iso-accuracy was ill-defined against single-point baselines — every method now swept over its own cost knob, interpolation rule fixed, knobless methods excluded from iso-claims); E5 loop control arm (iteration-2 with frozen vs refreshed coach at matched compute — the delta IS the loop's contribution; without it the loop claim was confounded with "more training"); GAIA transfer split by role (stopper-as-monitor over frozen live-web agent = SupervisorAgent-comparable; trained executor → BrowseComp-Plus/Bamboogle/2Wiki where tools match training; live-tool executor run = exploratory only); stopping-regret dual-run protocol (forced-continuation replays on a 500-task subsample — post-stop counterfactuals don't exist in normal traces); small-n significance policy (no hypothesis tests on n<500; GAIA/Bamboogle = CIs only); billing symmetry statement; ablation tiering + honest budget (~30–35 runs); A9 negative controls (random-coach, shuffled-label coach); B3 adaptation disclosure; GAIA val-as-test disclosure; frozen eval subsamples; K1 1-seed caveat | third internal review (experiments/results/analysis as a measurement system): the worst findings were a headline metric that couldn't be computed as written and a loop experiment whose positive result would have been dismissible in one review sentence |
| Method-pipeline flaws (internal review round 2, 2026-07-16) | **10 repairs**: forced-continuation collection mode (label rollouts run to T_max; ANSWER logged as draft event — kills the censoring flaw that would silently destroy iteration-2 labels as internalization succeeds); running draft moved into the shared agent template for ALL methods (kills the inference-time draft paradox; probes demoted to a validation ablation; quality scoring is now a free string compare); λ-conditioned single stopper + frontier-via-inference-dial protocol (kills a hidden 5×-executor-trainings explosion in E3); min-cohort guard for step-level group normalization; B9 specified as step-level (fair fight); draft-freezing attack named honestly with its cross-iteration defense; backup-residual diagnostic; EMPTY_DRAFT edge; collection overhead priced into T4 | second internal review of §2 as a pipeline: the worst finding was a data-generation flaw that improves at iteration 1 and silently degrades at iteration 2 — the method's success would have destroyed its own training data |
| Formulation inconsistencies (internal review, 2026-07-16) | **11 repairs**: V̂ third head added (the shaping potential was referenced but never trained); invariance convention fixed (Φ(terminal)=0, telescoping ⇒ step-level advantages MANDATORY); R_base aligned to the labels' tier-scaled normalized economy (one Lagrangian, not three objectives); per-domain q_t (ALFWorld = env subgoal fraction, no probes); wallet B drawn per (task, group) not per episode; formal decision-process statement added (ANSWER = stop action; monitor never truncates training rollouts); recursion runs on the probe grid; cost pilot-normalized so λ is dimensionless; Cont(x_T) edge + policy-relative τ* noted | internal §2 review found the label→value→potential→objective chain didn't type-check across sections; an implementing agent would have built a system that doesn't run (missing V̂) or silently optimizes a mixed objective (plain-λ R_base vs tier-scaled labels) |

## 15. Reviewer FAQ (rewritten honestly)

- *"Why not just prompt / probe confidence?"* — Measured failure: ≤24.9% redundancy-F1
  (RedundancyBench), oracle gap (Token Economies); and B2 is in every table. If B2 wins, H1's
  fallback framing activates — we report it.
- *"Why a stopper instead of shaping directly with your labels?"* — B9 tests exactly this; the
  stopper's value-add is generalization to unlabeled states, executor-transfer, and runtime
  control (E2). If B9 ties, we say so (H3).
- *"Isn't this AgentPRM with a cost term?"* — B8 is that. Our delta is stopping-margin semantics
  + measurable Snell targets + potential-based bridge + internalization measurement (E1/E2/E4).
- *"PRM label efficiency is solved (TD/GAE/implicit)."* — Agreed; we claim none of it. E4
  compares label *semantics* at matched compute.
- *"Learned monitors get hacked."* — Objective-feature inputs, per-iteration refresh, divergence
  curves reported (§2.4).
- *"Overhead?"* — End-to-end dollars incl. running-draft tokens, forced-continuation collection,
  and stopper serving, per serving regime, vs zero-amortization baselines (§5.3). LearnStop
  protocol adopted.

---

## 16. Execution runbook — end-to-end phases for the implementing agent

Each phase lists goal → key steps → outputs → done-criterion. Phases map onto §7's weeks.
An agent executing this document should work phase-by-phase and not start a phase before the
previous phase's done-criterion is met (exception: P8 baseline implementation can overlap P6–P7).

**P0 — Environment setup (W1).**
Install (exact repos + why: §19 manifest; pin every commit hash into §17): verl ≥ v0.8.0 (GRPO
backend, AgentLoop multi-turn), verl-tool (Search-R1 env on modern verl), verl-agent (ALFWorld /
GiGPO harness), HF TRL v1.8 (stopper SFT + scalar reward head), lightgbm (label regressor),
wandb. Stack requirements for Qwen3.5: transformers v5, vLLM ≥ 0.17 (GDN kernels); train with
enable_thinking=False; note the chat template strips <think> from history — use token-in-token-out
handling for multi-turn. Ministral (transfer model) needs vLLM's mistral tokenizer mode + tool
parser.
GPU note for this machine: acquire before any run with `eval $(/mnt/src/zhanka/gpu_acquire.sh N)`
(N=2 for stopper SFT / collection, N=4–8 for executor GRPO) and release with
`/mnt/src/zhanka/gpu_release.sh` when done — never kill occupier processes.
Reuse the repo harness's price map (`cost_aware_agent/cost.py`) for dollar costing.
✅ Done: one ReAct rollout runs end-to-end on HotpotQA dev and one on ALFWorld, with the
running-draft template line present in every step and per-step cost logging into the trajectory
JSONL schema (§11).

**P1 — Data & environments (W1).**
Download NQ + HotpotQA train (sample 8–10K combined) + MuSiQue train (sample 5K); build the
retrieval index (Search-R1 recipe: local Wikipedia dump + E5/BM25 — kept identical to baselines
on purpose; one Qwen3-Embedding-0.6B retriever ablation for robustness); ALFWorld train task
list; freeze dev/test splits; stage eval-only sets (GAIA text-only 103-Q dev, BrowseComp-Plus +
its fixed 100K-doc corpus with local retriever, Bamboogle, 2WikiMultihopQA, MATH-500, AIME 2025).
Run the decontamination pass (n-gram/MinHash, train prompts vs all eval sets — §5.6).
✅ Done: dataset manifest with counts + split hashes committed.

**P2 — Collection round 0 (W1–2).**
Base executor (Qwen3.5-9B, enable_thinking=False), G=8 rollouts/task, T_max=10 (QA) / 20 (ALFWorld),
**forced-continuation mode** (§2.1: ANSWER logged as draft event, trajectory runs to T_max — no
censoring), running-draft template active (§2.6/§18.2), per-step draft scoring vs gold, full x_t
features logged (§11).
**Each (task, GRPO group) draws one allowance B ∈ {small, medium, large}, shared by all G
rollouts of that group** (never per-episode within a group — §2.2); calibrate the three
values first from a 200-task unconstrained pilot (small = P25 of observed spend, medium = P75,
large = 2×P90; freeze in §17 — the same pilot also fixes the cost-normalization constant
median_spend for C̃, §2.1).
✅ Done: ≥8K QA + ≥2K ALFWorld trajectories with per-step scored drafts, per-step dollar costs,
and roughly balanced allowance strata; running-draft token share + forced-continuation overhead
reported (feeds T4).

**P3 — Label construction (W2).**
Run Algorithm 1 per λ ∈ {0.1, 0.5, 1, 2, 5} with tier-scaled marginal costing (§2.2; plain-λ
variant kept for ablation A8); fit tanh scale s per domain; label-quality checks:
(a) manual review of 100 random trajectories (does τ* look right?), (b) label-noise sensitivity
(re-run with step-subsampled draft scoring), (c) sanity: higher λ ⇒ earlier τ* everywhere.
✅ Done: labeled datasets per λ + a one-page label-quality memo.

**P4 — Stopper v0 (W2).**
Algorithm 2 SFT on the 2B base; eval on held-out trajectories: stopping regret (utility gap to
Snell τ*), STOP/CONTINUE F1; external check on RedundancyBench (2605.29893).
✅ Done: stopper beats (i) majority-class and (ii) a calibrated confidence probe on held-out
stopping regret; if it cannot, STOP — fix features/labels before touching RL.

**P5 — KILL-SWITCH GATE (W2–3) — §12.**
K1 (bridge test) and K2 (separation test) on the 1K-task HotpotQA subset, 1 seed.
✅ GO if K1 passes per §12 thresholds. NO-GO → pivot per H2/H3/H4 fallback framings; write the
decision log either way (feeds appendix).

**P6 — Executor GRPO, iteration 1 (W4–6).**
Algorithm 3 on both training domains. **Implementation note (with a published recipe):** under
the invariance convention (Φ(terminal)=0, γ=1, §2.4) the shaped terms telescope to −Φ(x_0) —
constant within a group — so trajectory-level GRPO advantages are **exactly unaffected** by the
shaping; step-level advantage assignment is therefore **mandatory**, not a fallback. Two
step-level variants, both run: (a) SHAPE-style segment credit (2604.06636, ACL 2026 — their
dynamic-discount analysis is the published treatment of this exact telescoping issue) adapted to
our Φ; (b) per-step returns-to-go with group normalization (GiGPO-style). TIPS (2603.22293)
provides a working turn-level PBRS reference implementation for tool agents. Report which
variant headline numbers use. Dr. GRPO length-bias hygiene mandatory. Log the
V̂-vs-realized-reward divergence dashboard from step 0 (feeds F6).
✅ Done: iteration-1 executor beats B1 on cost@iso-accuracy on dev in both domains.

**P7 — Loop iteration 2 (W7–8).**
Re-collect with the iteration-1 executor **in forced-continuation mode again** (§2.1 — the
internalized executor stops early; without forcing, late-step label data would vanish exactly
because the method worked) → rerun P3–P4 (stopper refresh) → **GRPO iteration 2 TWICE at
matched compute: arm (a) frozen iteration-1 coach (the "more RL steps" control), arm (b)
refreshed coach.**
✅ Done: per-iteration deltas table (cost, accuracy, stopping regret at i=0,1,2) **+ the
(b)−(a) loop-contribution delta** — this IS E5.

**P8 — Baselines B2–B9 (overlaps W4–8).**
B2 self-eval prompt + Dynasor-style scalar probe (no training). B3 SupervisorAgent-protocol
monitor (reimplement its trigger heuristics on our envs). B4 OTC-GRPO (reward from 2504.14870,
same env). B5 EAPO (primary; agentic-ALP only if EAPO unreproducible — §5.2). B6 single-model
GRPO with cost-in-reward. B7 CaRT+cost — BOTH arms: SFT-only and +GRPO (§5.2). B8 AgentPRM-cost
(pooled return-to-go + cost; optional TD+GAE variant). B9 Snell-labels-as-advantages direct
shaping, no stopper (step-level machinery, §5.2). Budget: B4–B9 are training runs (~1–3 days
each on 8×H200), each with 2–3 frontier points per §5.3; schedule accordingly.
✅ Done: every baseline evaluated on the same frozen test sets with the same cost accounting.

**P9 — Full evaluation + ablations (W9–10).**
E1–E6 grids, A1–A9 ablations, transfer evals (executor swap; OOD sets by role §5.1), the
500-task forced-continuation regret replays (dual-run protocol §5.3), 3 seeds on headline
tables, statistics per §5.6, serving-regime overhead measurements (KV-fork vs re-prefill).
✅ Done: all numbers for T1–T5 and F3–F6 exist in `experiments/results/` as CSVs with a
generation script per figure/table.

**P10 — Figures & tables (W11).**
One script per figure/table (matplotlib; CSV in → PDF out; no hand-edited numbers). Inventory
and content fixed in §9. Style: colorblind-safe palette, error bars = 95% bootstrap CI.
✅ Done: `paper/figures/*.pdf` + `paper/tables/*.tex` regenerate from raw results with one make
target.

**P11 — LaTeX paper (W11–12).**
1. Skeleton: ICLR 2027 style files; `paper/main.tex` with §9's section structure and page budget.
2. Writing order: §4 Method → §5 Setup → §6 Results → §7 Analysis → §2 Related work → §1 Intro →
   §8 Limitations/Conclusion → Abstract LAST (from §0's one-sentence + headline numbers).
3. Citations: build `references.bib` from the verified IDs in `research/lit_review/00_overview.md`
   §4–5 and `competitor_analysis.md` (every claim about a competitor traces to a lit_review
   entry; no citation invented from memory).
4. Claims audit before every compile: grep the draft against §14's changelog — none of v5's dead
   claims ("first self-reinforcing cycle", O(K×T²), "static instance-blind", "structurally
   necessary") may reappear; every number in the text must exist in a table/figure.
5. Compile loop: `latexmk -pdf`; fix errors; check page budget; internal review pass with the
   reviewer-FAQ (§15) as the checklist; revise.
✅ Done: compiled PDF, 8 pages + appendix, all claims traceable, FAQ objections pre-answered.

---

## 17. Configuration reference (single source of truth; restores v1 §22–23 content, adapted)

```yaml
label:
  lambda_values: [0.1, 0.5, 1.0, 2.0, 5.0]     # dimensionless (cost is pilot-normalized, §2.1)
  default_lambda: 1.0                           # headline; chosen on dev (see §5.6)
  tier_multipliers: {HIGH: 0.5, MEDIUM: 1.0, LOW: 2.0, CRITICAL: 5.0}   # m(tier) in U_t (§2.2); m≡1 in A8
  allowances: calibrate per domain from 200-task unconstrained pilot —
              {small: P25_spend, medium: P75_spend, large: 2x P90_spend}; freeze after P2
  cost_normalization: C̃ = C / median_pilot_spend per domain (frozen after P2 pilot; §2.1)
  quality_scoring: {qa: F1/EM of per-step running draft vs gold (free string compare),
                    alfworld: env subgoal fraction}          # decision grid = every step (§2.6)
  collection_mode: forced_continuation_to_Tmax               # §2.1 — label rollouts only
  legacy_probe_ablation: answer_now_v1, max_tokens 64        # validation only (§2.6/§18.2)
  regressor: {type: lightgbm, fallback: mlp_2x256, features: x_t schema (§11)}
  delta_scale: fit tanh scale s per domain on round-0 data, then freeze

stopper:
  base_model: Qwen/Qwen3.5-2B                   # ablation: 0.8B / 4B (post-trained models carry no "-Instruct" suffix)
  head_impl: TRL v1.8 SFTTrainer + scalar head (AutoModelForSequenceClassification, num_labels=1)
  sft: {epochs: 3, lr: 2.0e-5, batch: 64, max_seq: 2048,
        early_stop_metric: heldout_stopping_regret}   # NOT cross-entropy
  heads: {action: CE weight 1.0, delta: MSE weight 0.5 (normalized, decisions),
          value: MSE weight 0.5 (unnormalized V*, shaping potential Φ)}
  lambda_conditioning: λ in input serialization (§18.1); ONE stopper across all λ label sets
  rationale: off                                # A7 ablation only

executor:
  base_model: Qwen/Qwen3.5-9B                   # enable_thinking=False
  training_lambda: 1.0                          # headline; +one λ=0.3 spot-check executor (E3); frontier via inference dial
  transfer_models: {cross_family: mistralai/Ministral-3-8B-Instruct-2512, cross_scale: Qwen/Qwen3.5-4B}
  grpo: {G: 8, lr: 5.0e-6, kl_beta: 0.04, clip_eps: 0.2,
         length_norm: dr_grpo,
         advantage: step_level (SHAPE-segment or per-step RTG — K1 picks; §2.4),
         min_cohort_guard: 3,   # <3 alive at step t → trajectory baseline for those steps ONLY
                                # (whole-trajectory-level shaping is provably inert — §2.4, never a fallback)
         rollout_temp: 1.0, eval_temp: 0.0}
  horizon: {qa: 10, alfworld: 20}
  shaping: {gamma: 1.0, format_weight: 0.1}
  iterations: 2                                 # minimum; report per-iteration

inference:
  stopper_eval_every_k: 1                        # A5 ablation: 2, 3
  delta_threshold: 0.0                           # fixed; allowance-sensitivity is learned (§2.2/§2.5)
  budget_tiers: {HIGH: '>60% remaining', MEDIUM: '30-60%', LOW: '10-30%', CRITICAL: '<10%'}
  ablation_A8_rule_table: {HIGH: 0.00, MEDIUM: 0.05, LOW: 0.15, CRITICAL: 0.30}   # v5-style comparator only

cost_model:                                      # dollars; relative weights are what matter
  token_prices_per_1M: {reference_local: {input: 0.60, output: 2.20},   # constant across methods
                        api_models: use repo harness price map (cost_aware_agent/cost.py)}
  tool_costs: {web_search: 0.003/query + 0.001/result, http_fetch: 0.0001/request,
               code_exec: 0.0001/exec + 0.0001/sec, retrieval_local: 0.0001/query}
  draft_line_tokens: charged at executor prices, ALL methods   # in c_t (§2.6); feeds T4
  legacy_probe_cost: same pricing                               # A5 ablation runs only

seeds: [42, 123, 789]
tracking: wandb project cassi-v2; every run tagged {phase, domain, lambda, seed, iteration}
```

**Implementation repo layout** (updates v1 §23.2 to the v2 method):

```
cassi/
├── labels/        # snell.py (Algorithm 1), drafts.py (running-draft parsing + scoring; legacy probe for A5), quality.py (F1/EM/subgoals)
├── stopper/       # model.py (three heads: action/Δ̂/V̂), features.py (§11 x_t serialization), train_sft.py, eval_regret.py
├── executor/      # react_agent.py, envs/ (searchr1_qa.py, alfworld.py), train_grpo.py (Algorithm 3, shaping + hygiene)
├── budget/        # reuse cost_aware_agent/{cost.py,tracker} — do not duplicate the price map
├── baselines/     # b2_probe.py … b9_direct_shaping.py (one module per §5.2 row)
├── eval/          # metrics.py (regret, Pareto, matched-risk), stats.py (§5.6), overhead.py (serving regimes)
├── analysis/      # figures/ (one script per F1–F6), tables/ (one per T1–T5)
└── paper/         # main.tex, sections/, figures/, tables/, references.bib
```

---

## 18. Stopper I/O templates (replaces v1 §21 — inference-available features ONLY)

### 18.1 Stopper input (serialized x_t; identical at training and inference)

```
<stopper_input>
[TASK] {question or task description}
[BUDGET] tokens {used}/{max} ({pct}%) | tool calls {used}/{max} | ${dollars:.3f}/${max:.2f}
         | tier {HIGH|MEDIUM|LOW|CRITICAL} | burn ${rate:.4f}/step
[OBJECTIVE] cost-sensitivity λ = {lambda}        # λ-conditioning (§2.3): the inference-time dial
[PROGRESS] step {t}/{T_max} | draft unchanged for {n} steps
           | draft edit-distance (last 3 steps): {d1},{d2},{d3}
           | retrieval overlap (last 3): {o}% | distinct sources: {k}
[HISTORY] {t-2}: {action_type}: {observation digest ≤64 tok}
          {t-1}: {action_type}: {observation digest ≤64 tok}
          {t}:   {action_type}: {observation digest ≤64 tok}
[DRAFT] {executor's running draft, template-emitted (§2.6); EMPTY_DRAFT token at t=1}
</stopper_input>
```

Deliberately absent (v1 §21 leaked these): any ground-truth-derived quality/Δquality numbers,
any executor-stated confidence, "answer stability: IMPROVING/DEGRADING" judgments computed from
gold answers. Everything above is computable by the repo harness at inference time.

### 18.2 The running-draft template line + legacy probe

**Running draft (mandatory scaffold, ALL methods, training and inference):** the agent template
requires each step's output to end with one line:
```
BEST ANSWER SO FAR: {one line; or EMPTY_DRAFT if none yet}
```
(tokens counted in c_t for every method; this is what [DRAFT] serializes and what quality
scoring reads at collection time — §2.6).

**Legacy answer-forcing probe (validation ablation A5 only — never machinery):**
```
Based ONLY on the work so far, output your best final answer to the task now.
One line, no explanation. If you have no answer yet, output your best guess.
```
(≤64 output tokens; used only to check forced-answer quality ≈ running-draft quality.)

### 18.3 Stopper output

Generative variant (headline): `<decision action="STOP|CONTINUE" delta="{float in [-1,1]}"/>`
— optional one-sentence rationale only in ablation A7.
Value-head variant: no text; the three heads (action logit, Δ̂ regression for decisions, V̂
regression for the shaping potential) read the serialized input directly. Enforcement: STOP
when Δ̂ ≤ 0 (§2.5; ablation A8 uses the δ(tier) rule table instead). The generative variant
emits `<decision action delta/>` for control; V̂ always comes from the value head (never parsed
from text).

---

## 19. Build-vs-Reuse manifest (web-verified 2026-07-16; re-verify before submission)

Decision rule: **build what we claim, reuse what we compare on, install the rest.** Reused repos
are cloned, version-pinned (commit hashes recorded in §17 at P0), and never modified internally —
our additions are wrapper hooks with a documented diff.

### Reuse (clone/install as-is)

| Component | Exact choice | Why this one + rebuttal line if challenged |
|---|---|---|
| Executor model | `Qwen/Qwen3.5-9B` (Apache 2.0, Feb–Mar 2026; train with enable_thinking=False) | Newest full-range open generation at 8–9B; tool-native; verl ships Qwen3.5 GRPO demos. *Rebuttal: "Qwen3.6 (Apr 2026) has no open model below 27B; Qwen3.5-9B is the direct successor of the Qwen2.5/Qwen3 backbones used by Search-R1/GiGPO and the community-standard agentic-RL backbone at submission time."* |
| Stopper model | `Qwen/Qwen3.5-2B` (0.8B/4B ablations) | Same family/chat template as executor isolates the method; GRPO-training precedent at 2B (WinDOM, 2026). |
| Transfer model | `mistralai/Ministral-3-8B-Instruct-2512` (Apache 2.0, Dec 2025; native function calling) | *Rebuttal: "No open Llama below 27B has shipped since 2024 — Llama-3.1-8B would be 2.5 years old at review; Gemma-4 E4B is only 4.5B effective. Ministral-3-8B is the freshest like-for-like non-Qwen 8B."* |
| RL framework | `verl-project/verl` ≥ v0.8.0 (Jun 2026) | AgentLoop multi-turn rollouts native since v0.7; the modal framework of 2025–26 agent-RL papers (verl-agent, verl-tool, AgentRL line). Alternatives (AReaL, SkyRL, slime, ROLL) are niche/async-first. |
| Search-agent env | `TIGER-AI-Lab/verl-tool` + `PeterGriffinJin/Search-R1` (retriever + data) | Same env as the baselines (OTC-PO/Search-R1 line); verl-tool (TMLR 2026, ICLR 2026 SPOT best paper) reproduces Search-R1 with higher performance on modern verl. |
| Embodied env | `langfengQ/verl-agent` (ALFWorld; GiGPO official code) | GiGPO/AgentPRM comparability for free; ALFWorld still standard in 2026 papers (Agent2 RL-Bench, EnvRL); ships GiGPO/GRPO/DAPO/GSPO. |
| Retriever | E5 + Wikipedia-21M dump (Search-R1 recipe), BM25 fallback | *Rebuttal: "Identical retriever to all baselines isolates the policy contribution; a Qwen3-Embedding-0.6B ablation shows robustness to retriever choice."* |
| Stopper SFT | HF TRL v1.8 — SFTTrainer + scalar head (`AutoModelForSequenceClassification`, num_labels=1) | Current standard; the legacy value-head wrapper was moved to `trl.experimental` — do not use it. |
| Cost accounting | This repo's `cost_aware_agent/cost.py` price map + HAL-harness reporting conventions (`princeton-pli/hal-harness`) | HAL (2510.11977) is the 2026 standard for agent dollar/token accounting + multi-run consistency; adopting it preempts "how did you count cost?" reviews. |
| Label regressor / misc | lightgbm, wandb, matplotlib, latexmk | Commodity; no currency risk. |

### Reuse — benchmarks & datasets (web-verified 2026-07-16)

| Dataset/benchmark | Role | Why + rebuttal line |
|---|---|---|
| NQ + HotpotQA + MuSiQue (train) | RL training | *Rebuttal: "The standard corpus of the 2026 Search-R1 successors (Search-R2 2602.03647, CoSearch 2604.17555, Search-E1 2605.22511; StepSearch trains on MuSiQue) and of the closest cost-aware baseline OTC-PO — changing it would break commensurability. Contamination is handled by the §5.6 protocol, and our headline metric (cost at matched accuracy) is robust to uniform accuracy inflation."* |
| ALFWorld (train) | second RL domain | *Rebuttal: "Still standard for agent RL in 2026 (Agent2 RL-Bench 2604.10547, RWML, maintained verl-agent with ICML'26 follow-ups); alternatives fail our criteria (AppWorld: 105-task train split; tau2-bench: needs LLM user-sim). Its success saturation is precisely the slack our method exploits — we report cost at matched success, where saturation makes efficiency the only differentiator."* |
| GAIA text-only 103-Q dev | OOD research-agent eval | *Rebuttal: "The primary benchmark of the exact line we compare to (SlimSearcher, SupervisorAgent ICLR'26, Tongyi DeepResearch). Gaia2 (Meta ARE) tests dynamic/async smartphone scenarios — a different task family, orthogonal to stopping. Live-search inflation (2606.05241) is disclosed and mitigated by BrowseComp-Plus."* |
| BrowseComp-Plus (830 Qs, fixed 100K-doc corpus) | OOD research-agent eval, reproducible | ACL 2026 variant of BrowseComp with a local retriever — no paid live APIs, no search-time contamination; the reproducibility answer to "why not BrowseComp?" |
| Bamboogle (125) + 2Wiki (500-dev) | OOD multi-hop eval | The Search-R1-line protocol through 2026 (R-Search ACL'26, REAP, Agentic-R); fully local (wiki-18 + E5); Bamboogle doubles as the adversarially-fresh contamination check |
| MATH-500 + AIME 2025 | low-slack control | *Rebuttal: "MATH-500 kept for ALP/DAST/LASER comparability; but ALP's own results show savings concentrate on MATH-500 (it is the easy set), so AIME 2025 — the 2026 standard hard set (Leash, CLORE, DISPO) — is the genuine low-slack control showing we don't strangle hard reasoning."* |
| RedundancyBench · OptimalThinkingBench · HAL conventions | efficiency/stopper validation | RedundancyBench: external stopper validation; OptimalThinkingBench: agent-free over/under-thinking check; HAL: the 2026 cost-reporting convention. CostBench cited, not run (cost-optimal planning — different construct). |

Stack pins for Qwen3.5: transformers v5, vLLM ≥ 0.17 (GDN kernels); chat template strips <think>
from history → token-in-token-out multi-turn handling; known verl issue with 3.5/3.6-35B-A3B
tool-call rollouts does NOT affect the 9B. Ministral: vLLM mistral tokenizer mode + tool parser.

### Build (ours — each line maps to a §2/§5 claim)

| Component | Where | Maps to |
|---|---|---|
| Snell-envelope label pipeline + draft scoring | `cassi/labels/` | Contribution 3 (§2.2, Alg. 1) |
| Stopper features/serialization + training | `cassi/stopper/` | Contributions 1, 4 (§2.3, §18) |
| Potential-based reward wiring in verl's reward fn | `cassi/executor/train_grpo.py` | Contribution 1 (§2.4, Alg. 3) |
| Cost/draft/monitor hooks around the agent loop | `cassi/executor/envs/` wrappers | method plumbing (§2.5, §2.6) |
| Baseline reward variants B2–B9 | `cassi/baselines/` | §5.2 fairness (same env/model for all) |
| Eval, stats, overhead, figures | `cassi/eval/`, `cassi/analysis/` | §5.3, §5.6, §9 |

**Re-verify this manifest in the week before submission** — model/framework churn in 2026 is
fast enough that a February choice can be stale by September; this table was accurate on
2026-07-16.
