# cost-aware-agent

A background daemon + adapters that give CLI coding agents (Claude Code, OpenCode)
real-time visibility into how much a session is actually costing — in dollars — so
the agent can factor cost into its own decisions instead of running blind.

**Money is the metric.** The budget is always a dollar amount, and spend is always
the real dollar cost of LLM usage (LiteLLM retail pricing; free-tier models are
priced at the paid model's retail rate so the pressure stays real). Cost is never a
proxy — not tool-call counts, not turns, not iterations.

**Advisory only.** The daemon never blocks a tool call, never aborts a session, and
never enforces a ceiling. It injects the real number; the model decides. If the
model judges the task worth exceeding the budget, it can — the bet is that a
visible, depleting dollar budget produces the same economic judgment a human
engineer applies, not that a hard stop does.

## Why

Coding agents today have no sense of what they're spending. A five-minute task and
a five-dollar detour look identical from inside the conversation. This gives the
agent the missing half of the trade-off: not just "can I do this" but "is this
still worth it."

## How it works

A local FastAPI + SQLite daemon (`~/.cost-aware-agent/`) tracks cost per session:

- **Cost Engine** — real `$` from token counts and model pricing (vendored LiteLLM
  price map), including Anthropic's split pricing for 5-minute vs 1-hour
  prompt-cache writes. A model with no price-map entry is **never costed $0**
  (that would be an unmeasured channel — route work through an unpriced model id
  and spend vanishes): it's charged a conservative mid-tier fallback rate, the
  row is flagged `price_unknown`, and the tracker shows a warning line.
- **Subagent capture** — Claude Code Task-tool subagents write their own
  transcript files (`<session>/subagents/agent-*.jsonl`), not the parent
  transcript — and Workflow-tool agents nest one level deeper
  (`subagents/workflows/wf_*/agent-*.jsonl`). The daemon discovers the whole
  `subagents/` tree recursively and ingests it onto the parent session. On a
  real workflow-heavy session this was **71% of true spend** ($30.84 of
  $43.60) that parent-only parsing missed.
- **Budget Tracker** — injects current spend, remaining budget, measured burn
  rate, and tier (HIGH/MEDIUM/LOW/CRITICAL). The numbers come with a single
  delegation line — "decide yourself what these numbers mean" — not canned
  per-tier verdicts: the harness measures, the model judges. At spend
  milestones (each 10% budget slice crossed) it asks one accounting question:
  "what did that spend buy? If nothing new, change course or finalize."
- **Project wallet** — one number from the user, maximum: `cost-aware-agent
  init --budget 10` gives the project a dollar wallet that **depletes across
  every session** in that directory until exhausted (the session budget is a
  view over the wallet, not a per-session reset). Resolution order for a
  session's budget: explicit `/session/start` override (used by experiments to
  switch conditions without a daemon restart) → project wallet → config
  default. `cost-aware-agent budget show` prints budget/spent/remaining.
- **Injection policy (anti-tax)** — on accumulating channels (Claude Code's
  `additionalContext` persists in the conversation) the tracker is re-injected
  **only when the signal changes**: tier transition, or spend crossing another
  10% slice of the budget (`inject_mode: "on_change"`, the default). An earlier
  version injected on every tool call and measurably *increased* session cost by
  ~44% on tool-heavy tasks — the fix cut a 10-tool session from 22 injections to
  ~4. Rebuilt channels (OpenCode's system prompt is reconstructed every LLM call)
  always carry the current tracker, since nothing accumulates there.
- **Audit log** — every delivered injection is recorded in the daemon DB
  (`injections` table). `GET /session/<id>/dump` exports a session's full history:
  LLM usage rows, tool calls, every injected text, plan state — runs are
  replayable and debuggable from the daemon alone.
- **Session history (measured, not estimated)** — a new session in a project
  with prior finished sessions starts with "[PROJECT HISTORY] Your last N
  session(s) here cost $A–$B (median $M)" — real measured totals the model can
  baseline against; nothing is injected when there is no history.
- **End-of-session receipt** — `/session/stop` logs a human-readable receipt
  (total by model, budget/wallet state, tool-call breakdown, biggest single
  calls) to `daemon.log`; `cost-aware-agent receipt <session>` prints the same
  on demand.
- **Compaction survival** — Claude Code re-fires SessionStart with
  `source=compact` after compaction, which wipes every accumulated injection
  from the conversation; the daemon detects it, resets its on_change state,
  and re-delivers the tracker immediately (same for `clear`).
- **Planning / Self-Verification (dormant)** — session-start checklist and
  milestone-triggered verification prompts (BATS-style) never demonstrated
  value in any experiment and are **off by default**
  (`enable_plan_verification: false`); the code paths stay behind the flag.

Two adapters, two capture mechanisms:

| | Claude Code | OpenCode |
|---|---|---|
| Mechanism | pull — daemon parses the session transcript on hook fire | push — plugin posts usage straight from hook payloads |
| Injection channel | hook `additionalContext` (accumulating → on_change policy) | system-transform (rebuilt → always current) |
| Install | `cost-aware-agent install --for claude-code [--project-dir . \| --global]` | `cost-aware-agent install --for opencode [--project-dir . \| --global]` |

The daemon auto-spawns on first hook fire and fails open — if it's ever
unreachable, adapters silently no-op rather than blocking the agent.

## Install

```bash
pip install -e .
cost-aware-agent install --for claude-code --project-dir .
# or: cost-aware-agent install --for opencode --project-dir .
cost-aware-agent init --budget 10        # "$10 for my project" — the one number
cost-aware-agent budget show             # budget / spent / remaining
```

(Re-run `install` after upgrading — it rewrites the hook script, which the
adapters depend on for fields like `project_dir`.)

Run the unit tests with `python3 -m pytest tests/`.

## Tamper boundary (honest scope)

The daemon is an unauthenticated localhost service and the DB is a local file:
a Bash-capable agent on the same host can ultimately tamper (POST forged
usage, run `budget set`, edit SQLite). The design answer is advisory-trust
plus **tamper evidence**, not prevention: all inputs are validated at the
boundary (negative token counts clamped at the cost formula *and* the intake;
non-finite or non-positive budget overrides rejected), every accepted budget
override is logged to `daemon.log`, and every usage row / injection / override
is in the DB for post-hoc audit. Experiments additionally sandbox the agent
away from the daemon and DB.

## Known accuracy limits (documented, by design or pending upstream)

- **Claude Code spend lags the current turn.** CC exposes no per-turn usage hook
  (`anthropics/claude-code#11008`), so spend is parsed from the transcript as
  turns complete. The injected tracker says so explicitly ("cost is measured from
  completed turns"). In practice tiers do escalate live within a session (verified
  HIGH→LOW→CRITICAL in a single real `claude -p` run), but the number trails by
  up to one turn.
- **Session-title generation cost is invisible** to the transcript parser (CC
  writes no usage block for its haiku title calls). Roughly constant per session:
  negligible on expensive tasks, up to ~12% of LLM cost on very cheap ones.
- **OpenCode's usage payload has no 5m/1h cache-write split**; cache writes are
  priced at the 5-minute rate — a conservative undercount.
- **LiteLLM price-map staleness** — refresh by re-vendoring
  `cost_aware_agent/data/model_prices_and_context_window.json`. A stale map
  degrades to the conservative fallback rate + `price_unknown` flag, never to
  silent $0.

## Experiments — does cost-awareness change behavior?

Three experiments, in increasing order of realism. Full per-run traces (event
streams, transcripts, hook logs, daemon dumps) are written under each
experiment's `runs/` (gitignored, regenerable); committed results snapshots and
methodology live in each experiment's results doc.

### 1. `experiments/hotpotqa/` — prompt-level A/B (ReAct harness)

A plain-text ReAct agent (Sonnet via `claude -p`, and deepseek via `opencode run
--pure`) answers screened retrieval-forcing HotpotQA questions with BM25 tools,
budget tracker fed directly into the prompt. Full method + honest read in
[`experiments/hotpotqa/RESULTS.md`](experiments/hotpotqa/RESULTS.md).

**What it showed:** the budget consistently truncated one runaway question (OFF
over-explored it every seed, up to $1.91; a $0.30 budget stopped at 5-6 calls),
but the aggregate "28% cheaper" is **not significant at n=10** (paired t=0.85)
and is concentrated in that single question. The earlier version of this README
reported it as a robust monotonic result — that was an overclaim, corrected
after audit. `analyze.py` now prints paired t + CI by default.

**Integrity work that held up:** closed-book screening (all kept questions have
closed-book F1 = 0.0, so retrieval is forced), a caught-and-blocked cheat (the
model `grep`-ed the gold answers out of the data file until the harness moved to
an empty sandbox CWD + hard `--disallowedTools`), and per-run audit fields
(`web_requests=0`, tool-use-by-name) on every reported row.

### 2. `experiments/cc_adapter/` — production-path probe (superseded)

First test of the real Claude Code hook path. Its headline finding stands —
budget guidance steers *discretionary* work but not *hard-requirement* work —
but its ON arm was a static CRITICAL banner (spend lag + tight preset budget),
and it measured the injection tax that motivated the on_change policy. See the
correction header in its results doc.

### 3. `experiments/real_cli/` — real agents, real dataset, pre-registered budget

The production-path experiment: **real Claude Code** (Sonnet, native
Read/Grep/Glob, live production hooks) and **real OpenCode**
(deepseek-v4-flash-free, production plugin) answer HotpotQA distractor
questions from actual corpus files in a sandbox project. Bias controls fixed
from the audit: the budget is set by a **pre-registered rule** (median OFF cost
per question) computed on a **held-out calibration set**, never on the eval
questions. 120 eval runs + 24 calibration runs, all audit-clean.

Headline findings (full tables in
[`experiments/real_cli/RESULTS.md`](experiments/real_cli/RESULTS.md)):

- **Claude Code**: the on_change policy makes injection ~free (+2.8% cost, not
  significant, vs the +44% measured before), tiers escalate live in-session
  (HIGH→CRITICAL observed in real hook deliveries), and the one runaway-prone
  question is again the only one where the budget saves money. Advisory
  budgets do not cut *required* work — no saving on low-slack retrieval tasks.
- **OpenCode**: cross-agent plumbing fully proven (plugin, retail pricing,
  tiered injection, full audit trail), but deepseek ignores the guidance
  (behavior flat), and a second harness tax was found and measured: changing
  injected system-prompt text busts the provider's prompt cache (+92% cost,
  cut to +70% by byte-stable quantized tracker text; residual is structural
  when a single call spends ~20% of the budget). Absolute overhead:
  $0.00185/question.
- **The consistent picture across all three experiments**: the money savings
  live in discretionary and runaway work; capable models respect that
  boundary, weak models ignore the budget entirely, and the harness's own
  injection cost must be engineered to ~zero (now measured and mostly fixed)
  or it eats the benefit.

### 4. `experiments/e2e_verify/` — end-to-end feature acceptance (post-audit)

Not an A/B experiment: an acceptance test of the whole post-audit harness on
the real CLIs. 5 runs per agent (Claude Code 2.1.199 / Sonnet, OpenCode
1.17.13 / deepseek-v4-flash-free), each run targeting specific features;
**165/165 machine-checked assertions pass** (`verify.py`) with a clean cheat
audit. Live-proved in one sweep: exact-to-the-cent cost capture (4/5 CC runs
match CLI billing to full float precision), subagent capture (30% of the
delegation run's spend arrived via `pull-subagent` rows), wallet depletion
across sessions + [PROJECT HISTORY], spend-milestone checkpoints, tier
escalation to CRITICAL, advisory-only under 3× overspend, per-turn
`/turn/end`, receipts, and OFF-arm silence. Also caught: CC ≥ 2.1.x renamed
the subagent tool Task→Agent (deny lists must cover both), and a
session-metadata race in the OpenCode adapter (fixed + regression test).
Details: [`experiments/e2e_verify/RESULTS.md`](experiments/e2e_verify/RESULTS.md).

### 5. `experiments/swe_ab/` — real coding tasks, paired A/B (the slack benchmark)

The follow-up to `real_cli`, on a dataset that *has* discretionary slack:
**SWE-bench Lite** — real GitHub issues graded by each project's own
FAIL_TO_PASS / PASS_TO_PASS tests. 6 mechanically-screened instances × 2 seeds,
OFF vs ON (advisory money budget, wallet = 0.5× per-instance OFF median,
rule pre-registered). **Result (Claude Code / Sonnet): the budget cut cost
−29.4% paired (significant, t=2.53, 95% CI [$0.019, $0.345]), −40% total, with
success rate up (0.67→0.75) and 2→0 runaway timeouts.** Savings concentrate on
high-slack instances; low-slack ones barely move — confirming slack is the
variable that decides whether an advisory budget saves money (HotpotQA, no
slack, was +2.8% n.s.). Honest downside: 1/12 ON runs failed under budget
pressure that OFF solved. The audit also caught a real contamination —
bash `pip download` reached PyPI and one run read the fixed source (bash
network egress was not sandboxed; fixed with a subprocess proxy black-hole,
Anthropic whitelisted); the run was flagged, archived, and re-run clean.
OpenCode arm deferred (free-tier quota exhausted). Details:
[`experiments/swe_ab/EXPERIMENT_RESULTS_2026-07-03_2255.md`](experiments/swe_ab/EXPERIMENT_RESULTS_2026-07-03_2255.md).

## Status

- Claude Code adapter — live-verified: hook delivery, transcript cost pull
  (matches CLI-reported billing to ~$0.001 in test runs), on_change injection,
  in-session tier escalation.
- OpenCode adapter — live-verified: plugin loads project-scoped, usage push with
  message-id dedup (UPSERT, one row per assistant message), system-transform
  injection delivered to the model.
- Unit tests cover the cost engine (tier boundaries, cache-split pricing, price
  fallbacks), DB dedup/UPSERT, transcript parsing, prompt parsing, and the
  injection bucket policy.
