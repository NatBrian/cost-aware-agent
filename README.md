# cost-aware-agent

> **Give your coding agent a wallet.** It sees, in real dollars, what the current session is costing — and factors that into its own decisions instead of running blind.

A local daemon + adapters for **Claude Code** and **OpenCode** that meter the real dollar cost of a session and inject a live budget tracker into the agent's context. A five-minute task and a five-dollar detour no longer look identical from inside the conversation.

- 💵 **Money is the metric** — always real USD (retail LLM pricing), never a proxy like tool-call counts or turns.
- 🤝 **Advisory only** — never blocks, aborts, or enforces a ceiling. It shows the number; the model decides.
- 📉 **Proven to save** — on real coding tasks with slack, the budget cut Claude Code's cost **−29%** (significant) with *higher* success and zero runaway timeouts.

---

## Quickstart

```bash
pip install -e .

# install the adapter for your agent (per-project or --global)
cost-aware-agent install --for claude-code --project-dir .
#   or: cost-aware-agent install --for opencode --project-dir .

cost-aware-agent init --budget 10   # "$10 for this project" — the one number you set
cost-aware-agent budget show        # budget / spent / remaining
```

That's it. The daemon auto-spawns on the first hook fire, and the agent starts
seeing its spend. Nothing else to configure.

> [!TIP]
> The `--budget` you set is a **project wallet**: it depletes across *every*
> session in that directory until exhausted, not per-session. One number, set
> once. Re-run `install` after upgrading the package — it refreshes the hook script.

---

## What you get

| Feature | What it does |
|---|---|
| **Budget Tracker** | Injects live spend, remaining budget, burn rate, and tier (HIGH→CRITICAL). One line — "decide yourself what these numbers mean" — no canned verdicts. |
| **Project wallet** | `init --budget 10` gives a dollar wallet that depletes across all sessions in the directory. |
| **Real-dollar cost engine** | `$` from token counts × model pricing (vendored LiteLLM map), incl. Anthropic 5m/1h cache-write split. Unknown models are charged a fallback rate + flagged, **never $0**. |
| **Subagent capture** | Ingests Claude Code Task/Workflow subagent transcripts onto the parent session — was **71% of true spend** on one workflow-heavy run that parent-only parsing missed. |
| **Spend checkpoints** | At each 10% budget slice: "what did that spend buy? If nothing new, change course or finalize." |
| **Session history** | A new session starts with "[PROJECT HISTORY] your last N sessions cost $A–$B" — measured totals, never estimates. |
| **End-of-session receipt** | `cost-aware-agent receipt <id>` — total by model, wallet state, tool breakdown, biggest calls. |
| **Full audit trail** | Every usage row, tool call, and injected text is in the daemon DB; `GET /session/<id>/dump` replays a session for debugging. |

---

## Does it actually save money?

Short answer: **yes — when the task has discretionary slack *and* the model is
capable enough to act on the signal.** We ran a ladder of experiments from
prompt-level toy tests up to real GitHub bug-fixes. The headline:

| Experiment | Agent | Task | Result |
|---|---|---|---|
| **`swe_ab`** (headline) | Claude Code (Sonnet) | SWE-bench Lite (real issues, graded by project tests) | **−29% cost, significant** (t=2.53); success 0.67→0.75; runaway timeouts 2→0 |
| `swe_ab` | OpenCode (deepseek) | same | Plumbing 12/12 verified; weak model ignores budget → no saving, but harness cost engineered to **≈break-even** |
| `real_cli` | Both | HotpotQA distractor (low slack) | **No saving** (+2.8%, n.s.) — required retrieval has nothing to trim |
| `e2e_verify` | Both | Feature acceptance | **165/165** machine-checked feature assertions pass |

**The one lesson across all of them:** the savings live in *discretionary and
runaway* work. On tasks with no slack (fixed retrieval), an advisory budget
saves nothing. On open-ended work (debugging, "explore until good"), a capable
model trims the waste — Claude Code cut cost 29% and *stopped* the runaways that
otherwise ran to a timeout.

> [!IMPORTANT]
> **Honest caveats, up front.** (1) The 29% saving is on tasks *with* slack;
> low-slack tasks show ~0 effect — this is expected, not a bug. (2) A budget can
> also cut work that was genuinely needed: 1 of 12 Claude Code runs failed under
> pressure that the no-budget arm solved. Budget *size* is a real accuracy/cost
> knob. (3) A weak model (deepseek) ignores the budget entirely — no behavioral
> saving there, only correct measurement.

<details>
<summary><b>The cache-tax story (a real bug we found and fixed)</b></summary>

On OpenCode the budget first made sessions **+336% more expensive**. The cause
wasn't the model — it was our own injection design. OpenCode rebuilds the system
prompt every call, so a *changing* tracker in the prompt prefix invalidated the
provider's prompt cache from the top down every turn (cache-hit collapsed 90% →
~40%, verified from token logs).

The fix: deliver the tracker at the **end** of context (appended to the tool
result) instead of the system prefix. The tail extends the cache instead of
busting it, while the model still sees its current spend every turn. Result,
validated across the full arm with fresh wallets: **+336% → −24% (≈break-even)**,
freshness preserved, 12/12 feature checks still green. The budget is now free to
run on OpenCode too, not just Claude Code.

This is the general rule the harness now follows: *injected text may change
often, but it must go where a change doesn't invalidate the cache.*
</details>

<details>
<summary><b>Per-experiment detail + methodology</b></summary>

Full per-run traces (event streams, transcripts, hook logs, daemon dumps) live
under each experiment's `runs/` (gitignored, regenerable); committed results
snapshots and methodology live in each results doc.

- **`experiments/swe_ab/`** — the headline paired A/B on SWE-bench Lite. Wallet
  set by a pre-registered rule (0.5× per-instance OFF median), SWE-bench grading,
  cheat + network audit. Also caught a real contamination (a run `pip download`-ed
  the fixed package and read gold source — flagged, network-blocked, re-run clean)
  and the cache-tax fix above.
  [Results](experiments/swe_ab/EXPERIMENT_RESULTS_2026-07-03_2255.md).
- **`experiments/real_cli/`** — real Claude Code + real OpenCode on HotpotQA
  distractor, pre-registered budget on a held-out calibration set. 120 eval +
  24 calibration runs, all audit-clean. Established "no saving on low-slack work."
  [Results](experiments/real_cli/RESULTS.md).
- **`experiments/e2e_verify/`** — acceptance test of every harness feature on the
  real CLIs. 165/165 assertions; caught the CC Task→Agent tool rename and an
  OpenCode session-metadata race. [Results](experiments/e2e_verify/RESULTS.md).
- **`experiments/hotpotqa/`** & **`experiments/cc_adapter/`** — earlier
  prompt-level and first-production-path probes (superseded, corrections noted in
  their docs). Kept for provenance.
</details>

---

## How it works

A local FastAPI + SQLite daemon (`~/.cost-aware-agent/`) meters cost per session
and hands the agent a budget tracker through each CLI's native channel.

| | Claude Code | OpenCode |
|---|---|---|
| **Capture** | pull — daemon parses the session transcript on hook fire | push — plugin posts usage from hook payloads |
| **Injection channel** | hook `additionalContext` (accumulating → re-injected only on change) | tool-result append (rebuilt → current every turn, cache-safe) |
| **Install** | `install --for claude-code` | `install --for opencode` |

Two anti-tax rules keep the harness's own cost near zero (both taxes were
*measured*, then fixed): on accumulating channels the tracker re-injects **only
when the signal changes** (tier flip or another 10% spent — cut a tool-heavy
session from 22 injections to ~4, killing a measured +44% tax); on rebuilt
channels it's delivered at the **end** of context so it never busts the prompt
cache (the +336%→break-even fix). The daemon fails open — if it's ever
unreachable, adapters silently no-op rather than block the agent.

---

## Honest scope & limits

> [!NOTE]
> **Tamper boundary.** The daemon is an unauthenticated localhost service and the
> DB is a local file — a Bash-capable agent on the same host *can* ultimately
> tamper. The design is advisory-trust plus **tamper-evidence**, not prevention:
> all inputs validated at the boundary (negative tokens clamped, bad budget
> overrides rejected + logged), every row auditable after the fact. Experiments
> additionally sandbox the agent away from the daemon and DB.

Known measurement limits (documented, by design or pending upstream):

- **Claude Code spend lags by up to one turn** — CC exposes no per-turn usage
  hook ([claude-code#11008](https://github.com/anthropics/claude-code/issues/11008)); spend is parsed as turns complete. The tracker says so explicitly.
- **CC session-title cost is invisible** to the transcript parser — negligible on
  expensive tasks, up to ~12% on very cheap ones.
- **OpenCode has no 5m/1h cache-write split** — priced at the 5-minute rate (a
  conservative undercount).
- **LiteLLM price-map staleness** — a stale map degrades to the conservative
  fallback rate + `price_unknown` flag, never to silent $0.

---

## Development

```bash
python3 -m pytest tests/        # cost engine, DB dedup/UPSERT, parsing, injection policy
```

The daemon config lives at `~/.cost-aware-agent/config.json`; `GET
/session/<id>/dump` exports a full session for debugging. Planning /
self-verification code paths exist but are dormant behind
`enable_plan_verification: false` (never showed value in any experiment).
