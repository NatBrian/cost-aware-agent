# Real-CLI experiment — HotpotQA through real Claude Code and real OpenCode

**Run completed 2026-07-03.** 120 eval runs + 24 calibration runs, every row
audit-clean, zero timeouts. Committed snapshots: `budgets.json`, `analysis.json`.

## What this experiment is

The production question: does the cost-aware-agent daemon change agent behavior
when installed the way a user would install it — real Claude Code and real
OpenCode, their own agent loops, native tools, production injection channels?

- **Claude Code**: `claude -p` (Sonnet), native `Read`/`Grep`/`Glob`, live
  `PreToolUse`/`SessionStart`/`PostToolUse`/`Stop` hooks loaded via `--settings`.
  Injection channel: `additionalContext` (accumulating) under the daemon's
  `on_change` policy.
- **OpenCode**: `opencode run` (deepseek-v4-flash-free, retail-priced by the
  daemon), production plugin auto-loaded from the sandbox project. Injection
  channel: system-transform (rebuilt every LLM call).

Dataset: HotpotQA distractor. Each question runs in a fresh temp sandbox
containing that question's own 10 passages (2 gold + 8 distractors) as
`corpus/*.txt`. The agent answers with `ANSWER: <...>` and is graded EM/F1
(SQuAD normalization).

## Bias controls (fixes from the repo audit)

1. **Held-out budget calibration.** The ON-arm budget is set by a rule fixed
   before any eval run: *budget = median OFF cost per question on the
   calibration set* (`data/calib.json`, 6 questions, disjoint from the 10-question
   eval set, same closed-book screen: closed-book F1 = 0.0).
2. **Paired significance reported.** `analyze_real.py` prints paired t + 95% CI
   for the money and tool-call deltas; non-significant savings are labeled as such.
3. **Cheat audit with teeth.** Every run logs every tool call by name + input.
   `audit_clean` requires: no mutation/exec/web tool, zero server web requests,
   and **no tool input referencing a path outside the run's sandbox**.

## Integrity incidents during bring-up (all caught by the audit, all fixed)

These are worth reading — they are exactly the failure modes a "did the agent
cheat" audit exists for:

1. **OpenCode `permission: deny` did not block bash.** A calibration run
   executed `find /home/... -name corpus` through a config that denied bash.
   Fix: `tools: {bash: false, ...}` (removes the tool from the model entirely;
   verified live — a forced bash attempt returns "unavailable tool").
2. **OpenCode bound the repo, not the sandbox, as project root** (two causes,
   fixed in sequence: no `.git` in the sandbox; then a stale `PWD` env var
   inherited through `subprocess` — OpenCode resolves the project from `PWD`,
   not the real cwd). Consequence before the fix: the model grep'd
   `data/calib.json` in the repo and **read the gold answer** (`"answer": "TOGO
   company"` came back in its grep results). The path audit flagged every such
   run; all contaminated runs were deleted and re-run after the fix, and the
   fix was verified by re-driving the same question (sandbox-only reads, no
   escapes, legitimate retrieval answer).
3. **Silent OpenCode session truncation** (provider-side error, exit 0, no
   final text): retried once, and a run that still ends with no model text is
   recorded as `failed` (a platform reliability event, excluded from paired
   scoring, counted in the reliability stats).

## Results

Budgets from the pre-registered rule (median OFF cost/question on the held-out
calibration set): **Claude $0.18**, **OpenCode $0.0025** (deepseek retail-priced).
10 paired eval questions × 3 repeats × OFF/ON per platform.

### Claude Code (Sonnet) — no significant behavior change; injection tax fixed

| arm | $/q | tools | F1 | EM | injections/q |
|---|---|---|---|---|---|
| OFF | 0.19873 | 4.23 | 0.748 | 0.567 | 0 |
| ON ($0.18) | 0.20433 | 3.87 | 0.750 | 0.533 | 3.4 |

ON vs OFF: **+2.8% cost, not significant** (paired t=−1.20, 95% CI of the
saving [−$0.016, +$0.005]); tools −0.37/q n.s.; ΔF1 +0.003, ΔEM −0.033 (one
question flipping on one seed — noise at this n).

What this does and doesn't show:

- **The on_change policy works.** 3.4 injections/run (was 22/run per-tool-call
  in the cc_adapter experiment), and the tier escalated live inside real
  sessions — all four tiers (HIGH→MEDIUM→LOW→CRITICAL) observed in delivered
  injections. Injection overhead is now ~2.8% and statistically
  indistinguishable from zero, versus the measured +44% before the policy.
- **No behavioral saving on this task shape.** These questions need ~4 tool
  calls of required retrieval — there is little discretionary slack to trim,
  and Sonnet doesn't cut required work under an advisory budget (consistent
  with the cc_adapter finding). The one question with slack behaves as
  predicted: q5 — the same runaway-prone question as the ../hotpotqa
  experiment — is the ONLY question where ON is cheaper (−$0.030/q).
- Honest bottom line for Claude Code: **budget injection is now approximately
  free to run, steers discretionary work, and does not make low-slack sessions
  meaningfully more expensive — but it does not save money on tasks without
  slack.**

### OpenCode (deepseek-v4-flash-free @ retail) — plumbing proven; model unresponsive; cache-bust tax found and fixed-in-part

| arm | $/q | tools | F1 | EM | injections/q |
|---|---|---|---|---|---|
| OFF | 0.00265 | 4.40 | 0.740 | 0.600 | 0 |
| ON ($0.0025) | 0.00450 | 4.67 | 0.750 | 0.600 | 4.7 |

ON vs OFF: **+70% cost, significant** (t=−13.49) — the harness itself, not the
model's behavior. Accuracy unchanged (ΔEM 0.000, ΔF1 +0.009); tools flat.

The mechanism, measured from the daemon dumps (seed 0 totals):

| condition | fresh input tokens | cache-read tokens | cache-hit rate |
|---|---|---|---|
| OFF | 164k | 423k | 72% |
| ON, exact per-call tracker (first run) | 356k | 280k | 44% → **+92% cost** |
| ON, byte-stable quantized tracker (fix) | 271k | 331k | 55% → +70% cost |

The system-prompt tracker changes bytes when tier/bucket transitions, and every
change invalidates deepseek's prompt cache from that point (cache reads are
~50× cheaper than fresh input, so the % penalty is large even though the
absolute overhead is $0.00185/question). Quantizing the displayed spend to
bucket floors and dropping the per-call tool counter (shipped in
`prompts.render_budget_tracker` / daemon rebuilt-channel path) recovered part
of the cache hit rate; the residual is structural at THIS budget scale — each
deepseek call spends ~20% of the $0.0025 budget, so the quantized signal still
legitimately moves almost every call. On budgets where a single call is a small
fraction of the budget (the intended regime), the text is stable for many calls
at a stretch.

Behaviorally, deepseek ignored the budget entirely (tool calls flat, no early
finalization) — the same weak-model non-compliance documented in ../hotpotqa.
Cross-agent **plumbing** is fully proven here: plugin loads project-scoped,
usage pushed and deduped, retail pricing applied, all four tiers delivered into
the system prompt, every injection recorded and exported.

### Takeaways

1. **Money-metric plumbing works end to end on both real CLIs** — real $ in,
   tiered advisory pressure out, full audit trail (events, transcripts, hook
   logs, daemon dumps) for every run.
2. **The harness's own cost is now a first-class, measured quantity.** Two
   different taxes were found and addressed (accumulating-channel re-injection
   → on_change policy; rebuilt-channel cache-bust → byte-stable quantized
   text). Rule of thumb shipped in the code: injected text may only change
   when the signal it carries changes.
3. **Advisory budgets don't cut required work** (Sonnet, both experiments) and
   **don't move non-compliant models at all** (deepseek). The saving lives in
   discretionary/runaway work — q5 here, q5 in ../hotpotqa, the overview task
   in ../cc_adapter. Task slack is the variable that decides whether a budget
   saves money.

## Reproduce

```bash
python build_data.py            # eval.json (10 q) + calib.json (6 q), disjoint
python sweep_all.py             # calibration -> budget rule -> eval, both platforms
python analyze_real.py          # paired tables + significance -> analysis.json
```

`runs/` is gitignored (full per-run traces: CLI event stream, CC transcript
copy, hook log, daemon dump with every delivered injection). `budgets.json`,
`analysis.json`, and this doc are committed.
