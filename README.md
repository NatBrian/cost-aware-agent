# cost-aware-agent

A background daemon + hook adapters that give CLI coding agents (Claude Code, OpenCode)
real-time visibility into how much a session is actually costing, so the agent can factor
cost into its own decisions instead of running blind.

**Advisory only.** The daemon never blocks a tool call, never aborts a session, and never
sets a hard budget ceiling. It computes real dollar cost from actual token/tool-call usage
and injects that information into the model's context. What the agent does with it — keep
going, wrap up, try a cheaper approach — is entirely the agent's call.

## Why

Coding agents today have no sense of what they're spending. A five-minute task and a
five-dollar detour look identical from inside the conversation. This gives the agent the
missing half of the trade-off: not just "can I do this" but "is this still worth it."

## How it works

A local FastAPI + SQLite daemon (`~/.cost-aware-agent/`) tracks cost per session:

- **Cost Engine** — real `$` from token counts and model pricing (vendored LiteLLM price
  map), not synthetic units. Handles Anthropic's split pricing for 5-minute vs 1-hour
  prompt-cache writes.
- **Budget Tracker** — injects current spend, tier (HIGH/MEDIUM/LOW/CRITICAL), and
  guidance text before tool calls / LLM turns.
- **Planning** — asks the agent to decompose the task into an exploration/verification
  checklist up front, tracked for the rest of the session.
- **Self-Verification** — on milestone actions (edits, test runs), asks the agent to check
  its checklist against the budget and decide: finish up, keep going, or change approach.

Two adapters, two capture mechanisms:

| | Claude Code | OpenCode |
|---|---|---|
| Mechanism | pull — daemon parses the session transcript on hook fire | push — plugin posts usage straight from hook payloads |
| Install | `cost-aware-agent install --for claude-code [--project-dir .\|--global]` | `cost-aware-agent install --for opencode [--project-dir .\|--global]` |

The daemon auto-spawns on first hook fire if it isn't already running, and fails open —
if it's ever unreachable, adapters silently no-op rather than blocking the agent.

## Install

```bash
pip install -e .
cost-aware-agent install --for claude-code --project-dir .
# or: cost-aware-agent install --for opencode --project-dir .
```

## Experiment — does cost-awareness change behavior?

`experiments/hotpotqa/` tests the core claim: does injecting a **money budget** make
the agent spend less while holding accuracy? It replicates a budget-vs-accuracy setup on
the official **HotpotQA** distractor dataset (`load_dataset("hotpot_qa", "distractor")`).

- **Harness** (`run_claude.py`) — a ReAct agent (Claude Sonnet via the `claude` CLI)
  answers multi-hop questions using two offline retrieval tools (BM25 `search`/`read`
  over the corpus). Every turn is routed through the **live daemon** (`/session/start`,
  `/tool/pre`, `/tool/post`, `/llm/usage`), so the daemon's real dollar-budget injection
  is the thing under test — not a reimplementation.
- **Arms** (`orchestrate.py`) — OFF (no injection) vs dollar budgets ($0.30 / $0.60 /
  $1.20). The daemon injects `LLM cost used: $X of $Y, Tier Z`, and the harness feeds
  real per-call token cost back via `/llm/usage` so the budget pressure is live.
- **Full traceability** — each run writes `runs/<run_id>/` with `meta.json` (dataset, git
  sha, condition), `config_snapshot.json`, `results.jsonl`, and `traces/<qid>.jsonl`
  logging **every step**: injected budget text, exact prompt, verbatim model output,
  action, observation, tokens, and cost.
- **Cheat audit** — the harness runs the CLI with `--allowedTools ""` and logs the complete
  `stream-json` event stream, capturing every tool the model invoked *by name*, plus web
  request counts and permission denials, so a run can be proven to have used only the
  offline corpus (no web lookup of dataset answers). `analyze.py` rolls this up per tier.

Run outputs (`runs/`, `results/`) are gitignored — regenerable from the committed harness
and dataset.

**Result (10 retrieval-forcing questions, 3 seeds/tier, all audit-clean).** Full tables and
method in [`experiments/hotpotqa/RESULTS.md`](experiments/hotpotqa/RESULTS.md).

The earlier null result had one cause — Sonnet answered famous-entity trivia from memory, so
the budget had nothing to trim. Fixed with a **closed-book screen** that keeps only questions
the model gets wrong without tools (all 10 kept have closed-book F1 = 0.0 → retrieval is forced).

- **Claude / Sonnet — money drops at equal accuracy.** vs OFF ($0.2484/q), a **$0.30** budget
  spends **$0.1789/q — 28% cheaper — with ΔEM 0.000** and F1 within noise. Under pressure the
  model commits earlier (2.2 vs 2.83 tool calls) instead of over-exploring; looser budgets
  ($0.60 / $1.20) exert less pressure and save ~10%. A real accuracy-vs-cost curve, and money
  is the metric.
- **OpenCode / deepseek-v4-flash-free — cross-agent, but capability-gated.** The same daemon
  computes retail cost from OpenCode's token stream and injects the Budget Tracker into
  deepseek's prompt (money/tracking path proven cross-agent). But deepseek **ignores** the
  guidance — it keeps looping (q5 hits the 20-call cap under every tier), so savings are
  near-noise (1.7–3.6%). Budget *reasoning* needs a capable model; the cross-agent *plumbing*
  is model-independent.

**Integrity:** a pilot caught the model `grep`-ing the gold answer out of `data/questions.json`
(global settings override `--allowedTools ""`); fixed with an empty sandbox CWD + hard
`--disallowedTools`. Every reported run is audit-clean (`web_requests=0`, `dirty=none`).

### Does it hold in *real* Claude Code? (`experiments/cc_adapter/`)

The HotpotQA harness fed the Budget Tracker straight into the model's **prompt** (high
salience). Real Claude Code delivers hook output as a low-salience `additionalContext`
system-reminder — so a second experiment drives a real `claude -p` (CC's own agent loop +
native tools) with the **live PreToolUse hook** injecting the budget, on a 10-file sandbox.
Full method + tables in
[`experiments/cc_adapter/EXPERIMENT_RESULTS_cc_adapter_2026-07-03_1327.md`](experiments/cc_adapter/EXPERIMENT_RESULTS_cc_adapter_2026-07-03_1327.md).

**Result: the effect is real but conditional on task slack.** On a *discretionary* task
(brief overview, budget $0.05 → CRITICAL) the model took the cheap path — `Glob` + summarize
instead of reading files — cutting **tool calls −56% / cost −9%** with coverage held. On a
*hard-requirement* task (find all bugs) it refused to trade recall for budget — same tool
count, +44% cost (injection token-tax). That is close to ideal economic judgement: spend less
when the marginal work is optional, don't cut corners when it isn't. **Limitation found:** CC
has no per-turn usage hook, so in-session spend lags (the injected tier can stay static within
one session); BATS-style *escalating* pressure isn't deliverable on CC today.

## Status

- Claude Code adapter — built, live-verified against real sessions. Planning/Self-Verification
  round-trip confirmed working end to end, including a fix for a prompt-wording bug where the
  model copied the example's placeholder text verbatim into real plan rows.
- OpenCode adapter — live-verified for cost/tool tracking (`session/start`, `tool/pre`,
  `tool/post`, `llm/usage` all confirmed against real sessions). Daemon-side Planning/
  Self-Verification handling (`/plan/seed`, `/verification/result`) is proven correct via
  direct synthetic requests, isolated from model behavior.
- Fixed an LLM-cost **double-count** on the OpenCode push path (found via live e2e test):
  OpenCode fires `message.updated` several times per assistant turn, each with the same
  message id and growing *cumulative* tokens. The push path inserted every snapshot (the
  unique `(session_id, message_id)` index never applied because the plugin sent no message id),
  so a turn was counted 2–3×. Fixed by sending `message.id` from the plugin and making
  `insert_llm_usage` a latest-wins UPSERT — verified one row per message (9 msgs → 9 rows, was
  ~2× before).
- Root cause of the Planning/Self-Verification gap on OpenCode, confirmed by disassembling the
  OpenCode 1.17.13 binary (`LLMRequestPrep.prepare`): it is **model compliance, not prompt
  delivery**. An earlier hypothesis that `experimental.chat.system.transform`'s `output.system`
  never reaches the model was **wrong**. The binary builds the system array as a single collapsed
  string `l = [ [...agentPrompt, ...providerPrompt, ...o.system, ...o.user.system].join("\n") ]`,
  fires the hook with `{system:l}`, and then consumes `l` directly:
  `[...l.map(g => ({role:"system", content:g})), ...o.messages]`. Our `output.system.push(text)`
  makes `l` length 2, below the `l.length > 2` collapse threshold, so our text is delivered to the
  model as its **own dedicated system message** — cleanly, not buried. Verified the hook name and
  `sessionID` match the real call site (a second trigger site with no `sessionID` is OpenCode's
  agent-config generator, correctly skipped by our `if(!sessionID)return`).
- Practical consequence: injection works on OpenCode exactly like Claude Code. Whether the Planning
  checklist / Self-Verification block actually gets emitted depends on model capability — cheap /
  free tiers (`big-pickle`, `deepseek-v4-flash-free`, `deepseek-chat`) tend to ignore low-salience
  system-role instructions, which is what the earlier tests hit. A capable model complies. Nothing
  to fix on our side for delivery. Cost/tool tracking is independent regardless — it uses the push
  path (`tool.execute.after`, `event`), not this hook.
