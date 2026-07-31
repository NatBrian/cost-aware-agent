# cost-aware-agent

A cost-metering harness for CLI coding agents (Claude Code, OpenCode). It tracks
the real dollar cost of a session and reports it back to the agent in real time,
so the agent can weigh whether its next action is worth the money — the way a
human engineer does.

The budget is **advisory**: the harness measures and reports, it never blocks a
call or enforces a ceiling. The bet is that a capable model, shown a real
depleting dollar budget, makes the same economic judgment a senior engineer
makes — and on real coding tasks, it does (see [Results](#results)).

---

## Motivation

AI coding agents have no economic judgment. They will improve, refactor, polish,
and iterate indefinitely — not because the work is valuable, but because they are
trained to complete tasks and be helpful. There is no training signal for
*stopping*: "this is not worth improving" is not a reward-consistent output, so
the model never learns to say it. It will spend $2.50 polishing code that was
correct at $0.30 and never notice the difference.

A senior engineer behaves differently. They stop not when time runs out, but when
they judge that **continuing produces less value than it costs** — skip the
refactor that fixes no bugs, ship once tests pass, escalate after four hours of
dead-end debugging. That is economic reasoning, and it is the input the model is
missing: not the capability (frontier models can reason about cost-value
trade-offs), but the *data* — what the work actually costs, and whether the
marginal improvement justifies the marginal spend.

This harness supplies that missing input. It is not a spend limiter — the LLM API
already has those. It is the cost side of the trade-off equation, delivered to
the agent while it works, so "is this still worth it?" becomes answerable.

The full rationale is in [`VISION.md`](VISION.md).

---

## Architecture

A local daemon meters every session and hands the agent a budget tracker through
each CLI's native extension mechanism. Adapters are thin; all accounting is
centralized in the daemon.

```
  Claude Code  ──(bash hook)──┐
   pull: daemon parses the    │        ┌──────────────────────────────────────┐
   session transcript for     │        │   cost-aware-agent daemon            │
   token usage                │  HTTP  │   FastAPI · 127.0.0.1:7331           │
                              ├───────▶│                                      │
  OpenCode     ──(TS plugin)──┘        │   cost engine   — $ from tokens ×    │
   push: plugin posts token            │                   LiteLLM prices     │
   counts from hook payloads           │   budget/tier   — spend vs wallet    │
                              ◀────────│   injection     — anti-cache-tax     │
        Budget Tracker                 │        policy     delivery           │
        injected back into             │                                      │
        the agent's context            │   SQLite ledger:                     │
                                       │   sessions · llm_usage · tool_calls  │
                                       │   · injections · wallets · plan      │
                                       └──────────────────────────────────────┘
```

**Components** (`cost_aware_agent/`):

| Module | Responsibility |
|---|---|
| `daemon.py` | FastAPI service; the single process both adapters talk to over HTTP. Endpoints for session lifecycle, per-tool hooks, usage ingest, and a full-session `dump` for audit. |
| `cost.py` | Cost engine — real USD from token counts × a vendored LiteLLM price map (incl. Anthropic 5m/1h cache-write split); tier computation; fallback pricing for unknown models. |
| `db.py` | SQLite ledger — `sessions`, `llm_usage` (message-id deduped), `tool_calls`, `injections`, `wallets`, `plan`. |
| `transcript.py` | Claude Code transcript (JSONL) parser — the pull-side usage source, including the subagent transcript tree. |
| `prompts.py` | Budget Tracker text rendering + deterministic (regex/XML) parsing of agent responses. No secondary LLM call. |
| `cli.py` | `cost-aware-agent` CLI — daemon control, adapter install (writes the hook script / copies the plugin), budget/wallet management. |
| `config.py` | Config load (`~/.cost-aware-agent/config.json`). |

**Two adapters, two capture mechanisms:**

| | Claude Code | OpenCode |
|---|---|---|
| Integration | bash hook via `--settings` (`SessionStart`, `PreToolUse`, `PostToolUse`, `Stop`, `SessionEnd`) | TypeScript plugin (`system.transform`, `tool.execute.after`, `event`) |
| Cost capture | **pull** — daemon parses the transcript as turns complete | **push** — plugin posts token counts from hook payloads |
| Injection channel | hook `additionalContext` (persists in context → re-injected only on change) | appended to the tool result (end of context → cache-safe, current every turn) |

The daemon auto-spawns on first contact and **fails open**: if it is ever
unreachable, adapters silently no-op rather than block the agent.

---

## Features

- **Real-dollar cost engine.** Cost is always USD from token counts × model
  pricing — never a proxy (tool-call counts, turns). Unknown models are charged a
  conservative fallback rate and flagged `price_unknown`, never costed $0.
- **Budget Tracker injection.** Live spend, remaining budget, measured burn rate,
  and tier (HIGH/MEDIUM/LOW/CRITICAL), delivered into the agent's context with a
  single delegation line — "decide what these numbers mean" — not canned verdicts.
- **Project wallet.** One number from the user (`init --budget 10`): a dollar
  wallet that depletes across every session in the directory until exhausted.
- **Subagent capture.** Ingests Claude Code Task/Workflow subagent transcripts
  onto the parent session — on one workflow-heavy run this was 71% of true spend
  that parent-only parsing missed.
- **Spend-milestone checkpoints.** At each 10% of budget consumed: "what did that
  spend buy? If nothing new, change course or finalize."
- **Session history.** A new session in a project starts with the measured cost of
  prior sessions ("last N sessions cost $A–$B") — real totals, never estimates.
- **End-of-session receipt.** Per-model totals, wallet state, tool breakdown, and
  the largest single calls (`cost-aware-agent receipt <id>`).
- **Injection anti-tax policy.** The harness must not inflate the cost it meters.
  On persistent channels the tracker re-injects only when the signal changes
  (tier flip or another 10% spent); on rebuilt channels it is delivered at the end
  of context so it never invalidates the provider's prompt cache. Both taxes were
  measured, then engineered away (see [Results](#results)).
- **Full audit trail.** Every usage row, tool call, and injected text is stored;
  `GET /session/<id>/dump` replays a session for debugging or cheat review.

---

## Results

The thesis — that a visible dollar budget induces economic judgment — is tested
across a ladder of experiments, culminating in a paired A/B on real coding tasks.
Full methodology and per-run traces live in each experiment's results doc.

| Experiment | Agent · task | Finding |
|---|---|---|
| **`swe_ab`** | Claude Code (Sonnet) · SWE-bench Lite (real GitHub issues, graded by project tests) | **Budget cut cost −29% (statistically significant, t=2.53), success 0.67→0.75, runaway timeouts 2→0.** |
| `swe_ab` | OpenCode (deepseek) · same | Every harness feature verified (12/12); weak model ignores the budget, so no behavioral saving — but harness overhead engineered to ≈break-even. |
| `real_cli` | Both · HotpotQA distractor (low slack) | No saving (+2.8%, not significant) — required retrieval has no discretionary work to trim. |
| `e2e_verify` | Both · feature acceptance | 165/165 machine-checked feature assertions pass, clean cheat audit. |

**The conclusion across all of them:** the money saving lives in *discretionary
and runaway* work — exactly where a human engineer exercises judgment. On
low-slack tasks a budget saves nothing (as expected). On open-ended work, a
capable model trims the waste and stops the runaways. Two conditions must both
hold: the task must have slack, and the model must be capable enough to act on
the signal.

**Honesty, up front:**
- The −29% is on tasks *with* slack; low-slack tasks show ~0 effect by design.
- A budget can also cut genuinely-needed work: 1 of 12 Claude Code runs failed
  under pressure the no-budget arm solved. Budget *size* is a real accuracy/cost
  knob.
- A weak model ignores the budget entirely — the harness still measures correctly,
  but there is no behavioral change to bank.

<details>
<summary>The cache-tax bug (found, measured, fixed)</summary>

On OpenCode the budget first made sessions **+336% more expensive** — not the
model's doing, but ours. OpenCode rebuilds the system prompt every call, so a
*changing* tracker in the prompt prefix invalidated the provider's prompt cache
from the top down every turn (cache-hit rate collapsed ~90% → ~40%, confirmed
from token logs). The fix delivers the tracker at the *end* of context (appended
to the tool result) instead of the prefix; the tail extends the cache instead of
busting it, while the model still sees current spend every turn. Validated across
the full arm with fresh wallets: **+336% → ≈break-even**, freshness preserved,
12/12 feature checks still green. General rule the harness now follows: injected
text may change often, but must go where a change does not invalidate the cache.

Experiment docs: [`swe_ab`](experiments/swe_ab/EXPERIMENT_RESULTS_2026-07-03_2255.md)
· [`real_cli`](experiments/real_cli/RESULTS.md)
· [`e2e_verify`](experiments/e2e_verify/RESULTS.md).
</details>

---

## Installation

**Requirements:** Python 3.11+, and the CLI you want to meter (Claude Code or
OpenCode) already installed. Four commands, from the project you want metered:

```bash
pip install -e .                                              # 1. the package
cost-aware-agent install --for claude-code --project-dir .    # 2. the adapter (or --for opencode)
cost-aware-agent init --budget 10 --project-dir .             # 3. a $10 project wallet
cost-aware-agent budget show --project-dir .                  # 4. verify: budget / spent / remaining
```

Then **start a new agent session** in that project — Claude Code loads hooks at
session start, so tracking begins on the next session, not the current one.

Re-run `install` after upgrading the package (it rewrites the hook script), and
run the test suite with `python3 -m pytest tests/`.

### If a coding agent is installing this

The sequence above is fully scriptable, but an installing agent should know what
`install` changes so it can verify, and one caveat that will otherwise trip it up.

`install --for claude-code`:
- writes the hook script to `~/.cost-aware-agent/hooks/claude-code.sh`;
- merges five hook entries (`SessionStart`, `PreToolUse`, `PostToolUse`, `Stop`,
  `SessionEnd`) into `<project>/.claude/settings.json` (or `~/.claude/settings.json`
  with `--global`) — idempotently, so re-running never duplicates entries.

`install --for opencode` copies the plugin to
`<project>/.opencode/plugins/cost-aware-agent.ts`.

**Verify:** `cost-aware-agent daemon status` prints `running`; `budget show`
prints the wallet; and `.claude/settings.json` contains a hook command pointing
at `claude-code.sh`.

> [!IMPORTANT]
> The session that runs `install` will **not** have tracking active — Claude Code
> reads hooks only at session start. Do not report the install as working by
> checking the current session; start a new `claude` session in the project, then
> confirm the Budget Tracker appears. (OpenCode likewise loads the plugin when a
> new `opencode` session starts.)

---

## Scope and limitations

**Tamper boundary.** The daemon is an unauthenticated localhost service and the DB
is a local file; a Bash-capable agent on the same host can ultimately tamper. The
design is advisory-trust plus **tamper-evidence**, not prevention: inputs are
validated at the boundary (negative token counts clamped, invalid budget
overrides rejected and logged), and every usage row, injection, and override is
auditable after the fact. Experiments additionally sandbox the agent away from the
daemon and DB.

**Known measurement limits** (documented; by design or pending upstream):

- Claude Code spend lags by up to one turn — no per-turn usage hook exists
  ([claude-code#11008](https://github.com/anthropics/claude-code/issues/11008));
  the tracker states this explicitly.
- Claude Code session-title generation cost is invisible to the transcript parser
  — negligible on expensive tasks, up to ~12% on very cheap ones.
- OpenCode's usage payload has no 5m/1h cache-write split; cache writes are priced
  at the 5-minute rate (a conservative undercount).
- A stale LiteLLM price map degrades to the fallback rate + `price_unknown` flag,
  never to a silent $0.

---

## Development

- Config: `~/.cost-aware-agent/config.json`
- Session inspection: `GET /session/<id>/dump` (usage rows, tool calls, every
  injected text, plan state)
- Tests: `python3 -m pytest tests/` (cost engine, DB dedup/UPSERT, transcript and
  prompt parsing, injection policy)
- Planning / self-verification code paths exist but are dormant behind
  `enable_plan_verification: false` — they never demonstrated value in any
  experiment.

---

## Repository layout — two projects, one repo

This repo holds **two separate projects**. Everything outside `research/` is the
harness described above; everything inside `research/` is an independent research
effort (RL training for cost-aware agents) that shares only the underlying idea.

| Path | Belongs to | What it is |
|---|---|---|
| `cost_aware_agent/` | **harness** | The daemon, cost engine, ledger, adapters, CLI (table above). |
| `tests/` | **harness** | Unit tests for the harness only (`python3 -m pytest tests/`). |
| `experiments/` | **harness** | The harness's own A/B experiments — `swe_ab`, `real_cli`, `e2e_verify`, `cc_adapter`, `hotpotqa` — the runs behind [Results](#results). |
| `VISION.md` | **harness** | The rationale for the harness. |
| `research/` | **research** | The CASSI research project: paper plans, literature review, and the `foundation/` training codebase. Has its own tests, configs, and experiment outputs. |

> [!IMPORTANT]
> **Research code and research experiments live in `research/` — never in
> `experiments/` or `tests/`.** Those two directories are the harness's; adding a
> research run there mixes two projects with different dependencies (the harness
> is FastAPI + SQLite on CPU; the research stack is vLLM/torch on GPUs), different
> test suites, and different data policies. Conversely, harness experiments do not
> go under `research/`.
>
> Concretely, when working on the research project:
> - training/eval code → `research/foundation/` (its own package, tests, Makefile)
> - experiment outputs and reports → `research/foundation/experiments/`
> - tests → `research/foundation/tests/` (`cd research/foundation && make test`)
> - every experiment constant → `research/foundation/configs/foundation.yaml`
>
> Note the name collision: `experiments/` at the repo root is the harness's, while
> `research/foundation/experiments/` is the research project's. They are unrelated.

**Start here for the research side:** [`research/README.md`](research/README.md) —
it indexes every research directory, with code in
[`research/foundation/`](research/foundation/).

**Where the research stands (2026-07-31).** The first end-to-end
pipeline-validation run — FOUNDATION-1
([plan](research/paper_plan_v2_1_foundation.md) ·
[report](research/foundation/experiments/reports/foundation_report.md)) — ran to
completion and passed its pre-registered gate: the RL-trained arm beat both
prompting and harness enforcement on utility at every budget (B=4: .289 vs .205
and .180, paired CIs excluding zero). A follow-up λ ablation then showed **the
gate passed for the wrong reason** — the trained agent produced better answers,
not cheaper ones, and sweeping the step-cost coefficient from 0 to 1.0 moved
stopping by 0.04 steps
([ablation](research/foundation/experiments/reports/ablation_report.md)).
Diagnostics traced this to three design errors rather than to the method: the
experiment required a larger effect than could physically exist on the dataset, a
scalar per-step price cannot express a state-dependent stopping rule, and the
objective was scaled so that even perfect play was worth +0.012 utility
([diagnostics](research/foundation/experiments/reports/pre_redesign_diagnostics.md)).

**The active plan is the redesign:**
[`research/paper_plan_v2_2_foundation.md`](research/paper_plan_v2_2_foundation.md)
(FOUNDATION-2). It re-targets the behaviour that the data says is actually there
— *abandoning unproductive work*, where 53% of all steps currently buy zero
quality — replaces the scalar price with a learned continuation value, and gates
the whole redesign behind a one-CPU-day check that can kill it cheaply.
FOUNDATION-1's plan and reports are kept, not deleted: they are the input to the
redesign and their surviving findings carry into the paper.

The harness is not a dependency of the research code (and vice versa) — the two
only share a design principle: budget information is delivered to the agent as
*facts, not advice* (`cost_aware_agent/prompts.py` →
`research/foundation/agent/prompts.py`).
