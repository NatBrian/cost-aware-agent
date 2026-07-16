# Literature Review — Master Overview

> **Generated:** 2026-07-16 · Independent review (archived_* folders not consulted)
> **Scope:** 10 research areas, ~94 core papers read at PDF level, ~100 peripheral papers verified.
> 101 PDFs stored in `research/papers/`. Per-area details live in `01_…` through `10_…` files.
> **Target of analysis:** CASSI proposal (`research/paper_plan.md` v5, summarized in `00_paper_plan_summary.md`).

---

## 1. File map

| File | Area | Core papers | Biggest threat found |
|---|---|---|---|
| `01_overthinking_empirics.md` | Overthinking evidence & efficiency-accuracy empirics | 9 | FCS truncate-at-first-correct labels (2412.21187) |
| `02_token_efficient_reasoning_rl.md` | Length-penalty / budget RL for reasoning | 10 | ALP adaptive penalty (2506.05256, NeurIPS'25 spotlight); TAB separate budgeter (2604.05164) |
| `03_learned_stopping_early_exit.md` | Learned stopping, self-termination, early exit | 10 | **OS-Pruner (2607.11089)** — trained stopper w/ `acc − λ·tokens` objective |
| `04_budget_cost_aware_agents.md` | Budget/cost-aware agent frameworks | 9 | SlimSearcher (2606.07074) cost-aware agent RL on GAIA; BAGEN early-stop estimator |
| `05_agent_prms_credit_assignment.md` | Agent PRMs & step-level credit | 10 | WWW'26 AgentPRM TD+GAE rollout-free labels (2511.08325) |
| `06_monitor_executor_metareasoning.md` | Monitor-executor & metacognition | 8 | Agent-RRM (2601.22154) trained monitor→reward→executor GRPO |
| `07_optimal_stopping_metareasoning_theory.md` | Optimal stopping & metareasoning theory | 9 | Hansen & Zilberstein 2001 (oracle is classical); De Sabbata RaM (2410.05563) |
| `08_agentic_rl_tool_use.md` | Agentic RL for tool use / long horizon | 11 | OTC-PO (2504.14870) cost-aware tool-call RL |
| `09_adaptive_compute_routing.md` | Adaptive compute, difficulty-aware budgets, routing | 9 | **Ares (2603.07915)** — small per-step effort router, hindsight labels, SFT+GRPO |
| `10_hindsight_selfimprove_loops.md` | Hindsight relabeling & self-improvement loops | 9 | **TERMINATOR (2603.12529)** — hindsight-optimal stop labels + small stopper |

---

## 2. The landscape in one picture

Every *component* of CASSI now exists somewhere. No paper occupies the *intersection*.

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

**The defensible gap (converged across 10 lit areas + 4 novelty agents so far):**
> No prior work converts an explicit `quality − λ·cost` hindsight optimum into a **trained stopping-value model** whose continuous stop margin Δ serves as a **per-step process reward for training a tool-using executor**, with multi-dimensional (token/tool/dollar) costs and budget-conditioned inference.

---

## 3. Verdict on CASSI v5's five claimed contributions

| # | v5 claim | Verdict | Evidence |
|---|---|---|---|
| 1 | "First self-reinforcing cycle (oracle→stopper→process rewards→executor)" | **OVERCLAIM — must be qualified** | AgentPRM already iterates rollouts→labels→PRM→policy; Self-Guide (2604.03098) uses "self-reinforcing loop" verbatim for agent policy↔reward co-evolution; Cooper/SPARK/Mutual-Taught = established genre. Defensible only as "first **cost-aware stopping-centric** closed loop," with ≥2 measured iterations. |
| 2 | "Separate stopping model is *necessary* (representation conflict)" | **CONTESTED — keep as hypothesis, not theorem** | ReMA: shared weights converge faster, 1B meta-agents collapse; HiPER: SOTA hierarchy in a single policy with a switch-advantage ≈ Δ; SPARK: single-model synergy. BUT CTA (2602.16699) shows cost-discounted GRPO alone fails to internalize costs — the ablation is winnable, make it the paper's centerpiece. |
| 3 | "O(T) oracle vs O(K×T²) MC rollouts (AgentPRM)" | **FACTUALLY WRONG — must be rewritten** | AgentPRM pools existing rollouts (O(N×T), hashed return-to-go, Eq. 1) — no per-state restarts; O(K×T²) describes Math-Shepherd-style annotation. Implicit PRM (ICML'25), PRIME, WWW'26 AgentPRM (TD+GAE, "no additional rollouts"), GiGPO, SPA-RL are all rollout-free and uncited. Pivot from label *cost* to label *semantics* (cost-awareness + GT anchoring + stopping margin). |
| 4 | "Per-instance dynamic adaptation beats static penalties" | **STRAWMAN DEAD — reframe** | Adaptive penalties are the 2025-26 norm: ALP, DAST, LASER-D, AdaCtrl, HAPO (AAAI'26), AdaptThink, Agent-Omit (ICML'26), SlimSearcher, EAPO. Wu (2502.07266) already measures difficulty-length r=0.57 → H5 is a replication. Surviving angle: *mid-trajectory, progress-conditioned* stopping for *agents* with heterogeneous costs — vs. upfront/episode-level penalties. |
| 5 | "Small stopper supervises large executor, <3% overhead" | **PARTIALLY OCCUPIED** | Ares (1.7B router, per-step, −35-45% tokens), TAB (1.7B budgeter over frozen 8B), SupervisorAgent (training-free, −29.7% GAIA tokens at iso-acc, ICLR'26). LearnStop warns probe overhead can eat savings (adopt its matched-risk protocol). Surviving angle: the stopper *trains* the executor (nobody's controller does) + transfer across executors. |

**Formal properties:** Properties 1–2 are classical (Hansen & Zilberstein 2001 Def. 4 + Thm 1 ≈ the oracle & uniqueness; λ-monotonicity = standard comparative statics; also Weitzman 1979, Chen et al. ICML 2020 for post-hoc stop-label imitation). Cite as transfers. Property 3 (labels improve as policy improves) is the only open formal claim — make it precise or drop it.

**Technical repairs demanded by reviewers (from novelty agents, to fold into v2):**
1. **Prophet bias:** `Δ_oracle` regresses on the realized future max of the same trajectory — non-measurable at time t, provably optimistic (prophet inequality), biases toward late stopping. Fix: Snell-envelope / backward fitted-Q labels — same O(T).
2. **Reward shaping:** `Σ_t α·Δ(s_t)` is non-potential-based (pays the executor to *accumulate* promising steps — opposite of the goal). Fix: potential-based Φ-differences or advantage-style progress rewards (PAV/MRT lineage).
3. **Reward hacking:** executor writes the confidence/draft features the stopper reads → the exact PRM-hacking loop AgentPRM measured (82%→70%). Fix: objective-only stopper inputs, periodic stopper refresh.
4. **Train/inference leak:** v5's monitor prompt includes GT-derived quality indicators unavailable at inference. Restrict inputs to inference-available features.
5. **Honest label-cost accounting:** per-step quality is free-ish for QA (F1 vs GT) but = a test-suite run per step on SWE-bench; the "zero extra cost" claim must be per-benchmark.

---

## 4. Top competitors (feeds `competitor_analysis.md`)

**Tier 1 — direct overlap, must cite + compare (most also baselines):**
| Paper | ID | One-line | Why it matters to CASSI |
|---|---|---|---|
| OS-Pruner | 2607.11089 | Optimal-stopping-trained CoT pruner, `acc − λ·tokens` | CASSI's exact objective + trained stopper, single-shot CoT |
| TERMINATOR | 2603.12529 | Hindsight-optimal exit labels → small stopper | CASSI's oracle move, λ→0, no agents/RL bridge |
| DASH | 2607.00482 | Post-hoc segment credit vs overthinking in GRPO | The "no-stopper direct shaping" ablation, ready-made |
| Ares | 2603.07915 | 1.7B per-step effort router for frozen agent (SFT+GRPO on hindsight labels) | Small-supervises-large, per-step, agentic |
| TAB | 2604.05164 | Separate GRPO-trained turn-budget allocator | Two-model budget control, frozen solver |
| BAGEN | 2606.00198 | Replay-relabeled SFT+RL early-stop estimator for agents | Closest component stack, no executor training |
| Agent-RRM | 2601.22154 | Trained reasoning RM → executor GRPO (GAIA/WebWalkerQA) | Occupies monitor→reward→executor bridge (quality-only) |
| SlimSearcher | 2606.07074 | SFT+GRPO cost-gated deep-research agent (GAIA Pareto win) | Occupies "cost-aware agent RL on GAIA" |
| OTC-PO | 2504.14870 | RL w/ hindsight-optimal tool-call count reward | Cost-aware agent RL, hindsight anchor, outcome-level |
| AgentPRM (Cornell) | 2502.10325 | Agent PRM via pooled return-to-go + iterated loop | Primary framing target — currently mischaracterized |
| AgentPRM (Fudan, WWW'26) | 2511.08325 | TD+GAE agent PRM, "no additional rollouts", feeds PPO | Kills the label-efficiency contribution as stated |
| SupervisorAgent | 2510.26585 | Training-free runtime monitor, −29.7% GAIA tokens iso-acc | The training-free bar CASSI must beat |
| CaRT | 2510.08517 | SFT self-termination via counterfactuals | v5's primary comparison — now one of many |
| ALP | 2506.05256 | Adaptive length penalty ∝ per-prompt solve rate | Kills "static penalty" strawman |
| De Sabbata RaM | 2410.05563 | VOC reward (gain − γ·tokens) in Expert Iteration | Prior cost-aware *training signal*; defers agents to future work |

**Tier 2 — adjacent, cite:** DAST 2503.04472 · LASER-D 2505.15612 · AdaptThink 2505.13417 · AdaCtrl 2505.18822 · HAPO 2505.11225 · EAPO 2606.02132 · AdaTIR 2601.14696 · DAS 2602.03304 (WWW'26) · CTA 2602.16699 · SeqRoute 2605.25424 · CoRL 2511.02755 · xRouter 2510.08439 · BATS 2511.17006 · BAVT 2603.12634 · INTENT 2602.11541 · Self-Guide 2604.03098 · SPARK 2509.22624 · Cooper 2508.05613 · RePro 2606.14302 · LYNX 2512.05325 · LearnStop 2606.30852 · OS theory: Hansen & Zilberstein 2001, Weitzman 1979, Chen et al. ICML 2020, Hay 2012, Callaway 2018 · GiGPO 2505.10978 · SPA-RL 2505.20732 · PRIME 2502.01456 · Implicit PRM 2412.01981 · SWEET-RL 2503.15478 · MRT 2503.07572 · HiPER 2602.16165 · ReMA 2503.09501 · MaR 2605.23384 · Agent-Omit 2602.04284 · SIM-RAG (SIGIR'25) · HiPRAG 2510.07794 · StepSearch 2505.15107.

**Motivation ammunition (keep, all verified):** Cuadron 2502.08235 (agent overthinking predicts failure, R²=0.892) · Wu 2502.07266 (inverted-U; r=0.57) · Hassid 2505.17813 (+34.5pt shortest-vs-longest) · Gema TMLR'25 (inverse scaling) · Chiang & Lee EACL'24 · **Token Economies EMNLP'24 (2406.06461) + RedundancyBench (2605.29893): prompted self-evaluation fails at economic judgment (≤24.9% step-level F1)** — the strongest published justification for a *trained* stopper · DAS: outcome-RL *causes* over-search · CTA: cost-discounted GRPO alone fails to internalize costs.

---

## 5. Corrections to v5's citations (carry into v2)

- **AgentPRM** does NOT run K MC rollouts from every state; it pools complete on-policy rollouts (hashed return-to-go). The "~160 extra executions/trajectory" figure is unsupportable.
- **IterResearch** (2511.07327) is RL-trained with efficiency-aware shaping — v5 wrongly lists it as a training-free heuristic.
- **BATS** = "Budget-Aware Tool-Use Enables Effective Agent Scaling" (Google, 2511.17006) — prompt-level Budget Tracker + orchestration, test-time scaling focus.
- **Ares** = 2603.07915 (UCSB/Accenture); **SeqRoute** = 2605.25424 (UT Austin) — both real, IDs now known.
- **"Learning to Stop While Learning to Predict"** is Chen et al. (ICML 2020), not Xiao et al.
- **"Token Economies"** = "Reasoning in Token Economies" (2406.06461, EMNLP 2024 main); distinct from the "Reasoning on a Budget" survey (2507.02076).
- Venue updates: L1/LCPO → COLM 2025; Reason Efficiently → NeurIPS 2025; AdaptThink → EMNLP 2025; HAPO → AAAI 2026; Plan-and-Budget → ICLR 2026; SupervisorAgent → ICLR 2026; HiPER → ICML 2026; DAS → WWW 2026; AgentPRM-Fudan → WWW 2026; GiGPO/ToolRL/DAPO/SWE-RL/Arora&Zanette → NeurIPS 2025; CARL authorship unverified (2512.04949) — check before citing.
- GAIA/WebWalkerQA/BFCL/MATH-500 have **no train splits** — v5's use of them as RL training benchmarks must be rescoped (train on HotpotQA/MuSiQue/NQ + tool-use sims; treat GAIA/WebWalkerQA as transfer-eval).
- SWE-bench Verified RL: only demonstrated at 32B–72B (DeepSWE-scale); per-step quality there = test run per step. Rescope or drop.

## 6. Feasibility notes for the v2 experiment plan (from area 08)

- Standard cheap setup: Qwen2.5-3B/7B executor + Search-R1-style retrieval env; HotpotQA/MuSiQue train, Bamboogle/GAIA transfer-eval.
- Use estimator-hygienic GRPO (Dr. GRPO / DAPO fixes) or reviewers will attribute token savings to length-bias artifacts.
- Known pathologies to guard: length-inflation bias, entropy collapse, Echo Trap, credit dilution (GiGPO).
- 14 baselines × 7 benchmarks × 3 seeds ≈ 10× over an 8-week budget → cut to: 2 training domains, ~6 pivotal baselines (SupervisorAgent-style monitor, OTC-PO/EAPO cost-RL, DASH-style direct shaping, single-model cost-GRPO, stopper-as-controller-only, scalar-probe stopper), transfer evals.
