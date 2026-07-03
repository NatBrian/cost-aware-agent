# Real-adapter experiment — does budget injection work in actual Claude Code?

> **Superseded 2026-07-03 by `../real_cli/`** — which runs real HotpotQA questions
> through real Claude Code AND real OpenCode with the production adapters, a
> held-out-calibrated budget, and the daemon's on_change injection policy (added
> in response to the +44% injection-tax finding below). Two framing corrections
> to this doc, found in audit: (1) the ON arm here was a **static CRITICAL tier**
> throughout (tight preset budget + in-session spend lag), so what this measured
> is a constant frugality nudge, not escalating depleting-budget pressure;
> (2) the overview "coverage" metric is near-tautological (module-name substring —
> the ON arm's Glob-only trajectory trivially scores 1.0 by echoing filenames).

**Run completed:** 2026-07-03 13:27 (local).
**Model:** Claude Sonnet via the real `claude -p` CLI (Claude Code runs its own
agent loop + native tools).
**Sandbox:** a synthetic 10-file toy commerce lib (`sandbox/src/*.py`) with 12
planted bugs (`sandbox/bug_manifest.json`).

## Why this experiment exists

The HotpotQA experiment (`../hotpotqa`) proved a capable model spends **~28% less**
when the Budget Tracker is placed **directly in its prompt** (high salience). But
that harness drove its own ReAct loop and fed the tracker into the prompt itself.
**Real Claude Code doesn't work that way** — it delivers hook output as a
low-salience `additionalContext` *system-reminder*, not a prompt directive. So the
28% result did not, on its own, prove the tool works in production.

This experiment closes the gap: a real `claude -p` per condition, Claude Code's own
loop and native tools, and the live **PreToolUse hook** injecting the daemon's
Budget Tracker on every tool call — the exact production path. Only the condition
(daemon `inject_enabled` / `session_budget_estimate_usd`) changes between arms.

Everything is traced three ways per run (`runs/<mode>/<tag>-s<seed>/`): CC's own
`events.jsonl` stream, a copy of CC's session `transcript.jsonl` (delivery ground
truth), and `hook.jsonl` (what the hook POSTed + injected on every fire).

## Headline result — the in-prompt effect is REAL but CONDITIONAL on task slack

Two tasks, 3 seeds each, budget `$0.05` for the ON arm (tight enough that the
injected tier reads **CRITICAL** — "Budget nearly exhausted, finalize now").

### Task A — bug review (LOW slack: must read every file to find all bugs)

| arm | cost/run | tool calls | recall | injections delivered | tier |
|---|---|---|---|---|---|
| OFF | $0.222 | 11.0 | 0.97 | 0 | — |
| ON ($0.05) | **$0.319** | 11.0 | 1.00 | 22/run | CRITICAL |
| Δ | **+44%** | 0 | +0.03 | | |

**Budget ignored.** Even at CRITICAL on every tool call, the model kept all 11
tool calls, read every file, found every bug. The only effect was the injection
token-tax (+$0.10/run). The model refused to trade recall for an advisory budget —
arguably correct: the task has no slack to cut without failing it.

### Task B — brief overview (HIGH slack: a shallow answer is valid)

| arm | cost/run | tool calls | coverage | injections delivered | tier |
|---|---|---|---|---|---|
| OFF | $0.173 | 3.0 | 0.97 | 0 | — |
| ON ($0.05) | **$0.157** | **1.33** | 1.00 | 2–4/run | CRITICAL |
| Δ | **−9%** | **−56%** | +0.03 (held) | | |

**Budget obeyed.** Under CRITICAL the model took the cheap path: `Glob` the file
list and summarize from filenames, **skipping the file reads** the OFF arm did.
Trajectory evidence:

- OFF: `Glob → Read → Read → summarize` (3 tools, read 2 files)
- ON:  `Glob → summarize` (1 tool, inferred the overview from filenames)

Coverage held (all 10 modules still named) at 56% fewer tool calls and 9% lower cost.

## Verdict

The 28% in-prompt effect **did not vanish** in real Claude Code — it is
**conditional on task slack**. The low-salience hook genuinely steers Sonnet's
*discretionary* choices (skip optional deep reads under budget pressure) but will
**not** override a *hard requirement* (find all the bugs). That is close to ideal
economic judgement: **spend less when the marginal work is optional; don't cut
corners when it isn't.** The naive "no effect" you'd conclude from the bug task
alone is a task-design artifact, not the model ignoring cost.

Caveats: toy sandbox, n=3, high determinism (identical tool counts across seeds);
the overview cost saving is modest (−9%) because the per-tool injection tax offsets
part of the behavioural gain — the dominant, unambiguous effect is on tool calls
(−56%). This is a **mechanism probe**, not a dataset-scale benchmark (that is
HotpotQA's job).

## Bugs / limitations found building this (all real, all documented)

1. **Hook log jq bug** (ours) — `injected:($ctx|select(.!=""))` emits nothing when
   the context is empty, and an empty value inside jq object construction drops the
   whole record → OFF fires logged nothing. Fixed with `if`/`then`/`else`.
2. **Headless CC won't run project hooks from an untrusted dir.** A fresh `claude
   -p` in a /tmp sandbox does NOT execute `.claude/settings.json` hooks (arbitrary-
   code-execution guard). Must pass `--settings <file>` explicitly (the harness
   generates it). Interactive/trusted projects fire project hooks normally.
3. **In-session spend lag (real limitation, not fixed).** Claude Code exposes no
   per-turn usage hook, so the daemon pulls spend from the transcript, which lags.
   Within a single `claude -p` session the injected "spent" can stay frozen at the
   first turn's cost while real cost climbs (final total is correct). Consequence:
   *escalating* pressure (HIGH→CRITICAL as you spend) can't be delivered
   in-session on CC; you get a roughly static tier set by the budget. The ON arm
   here used a tight budget so the static tier read CRITICAL throughout.

## Reproduce

```bash
python build_sandbox.py                                   # generate sandbox/
python run_cc_adapter.py --mode bugs     --tag off --budget 0    --seeds 3
python run_cc_adapter.py --mode bugs     --tag on  --budget 0.05 --seeds 3
python run_cc_adapter.py --mode overview --tag off --budget 0    --seeds 3
python run_cc_adapter.py --mode overview --tag on  --budget 0.05 --seeds 3
```

`runs/` is gitignored (per-run traces, regenerable). `results/run_summaries.json`
is the committed snapshot of the 12 runs behind the tables above.
