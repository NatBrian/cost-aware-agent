# E2E feature verification — real Claude Code + real OpenCode (2026-07-03)

**Question**: after the 2026-07-04 backlog (wallet, burn rate, checkpoint,
history, receipt, subagent capture) and the adversarial-audit fixes
(`6267267`), does every feature actually work end-to-end on the real CLIs —
and is everything logged well enough to debug and to prove the model didn't
cheat?

**Answer: yes — 165/165 machine-checked assertions pass** (`verify.json` in
the run dir), zero cheat signals, full trajectory captured for all 10 runs.

This is an **acceptance test, not an A/B statistics experiment** (that's
`../real_cli`). 5 runs per agent, each deliberately exercising specific
features; pass/fail is machine-checked from the captured artifacts.

## Setup

- **Claude Code 2.1.199**, model `sonnet`, production-path hooks
  (`hook.sh` = the production hook + a JSONL audit log of every fire)
- **OpenCode 1.17.13**, model `opencode/deepseek-v4-flash-free`, production
  plugin copied into each sandbox project
- Tasks: HotpotQA distractor questions from `../real_cli/data/eval.json`
  (closed-book-screened, corpus files in a git-init'd sandbox project;
  retrieval tools only, everything else denied)
- Wallets set through the real CLI (`cost-aware-agent budget set`), scaled
  from the real_cli calibration medians ($0.18/q claude, $0.0025/q opencode)
- Full capture per run: `task.txt`, `events.jsonl`, `transcript.jsonl` +
  `subagents/` (CC), `hook.jsonl` (CC), `daemon_dump.json`, `result.json`

## Run matrix and what each run proved

| run | q | wallet | spent | key features verified |
|---|---|---|---|---|
| cc1 (OFF) | q0 | — | $0.2140 | daemon == CLI billing **exactly** ($0.2139906); 0 injections; Stop→/turn/end ingested; SessionEnd fired naturally; receipt renders |
| cc2 (ON) | q3 | $0.50 | $0.1797 | tracker w/ "project wallet" scope + burn line + delegation line; on_change: **2 injections over 7 hook fires**; exact cost match |
| cc3 (ON) | q9 | reuse | $0.1512 | **[PROJECT HISTORY]** injected ("last 1 session(s) cost $0.18"); **wallet depletion**: budget view $0.331/$0.50 spent = cc2+cc3 combined |
| cc4 (ON) | q2 | $0.50 | $0.4158 | **subagent capture**: model delegated via Agent tool; daemon ingested 3 `pull-subagent` rows = **$0.1247 (30% of run spend)** that pre-rglob code would have missed; daemon $0.4158 vs CLI $0.4210 (gap = documented invisible title call, 1.2%) |
| cc5 (ON) | q1 | $0.06 | $0.1865 | **[BUDGET CHECKPOINT]** questions fired at spend milestones (wallet wording: "This project has spent $X"); tier HIGH→CRITICAL; **advisory-only**: 3× over budget, never blocked |
| oc1 (OFF) | q0 | — | $0.0027 | push capture, retail pricing of free-tier model, 0 injections |
| oc2 (ON) | q3 | $0.006 | $0.0051 | rebuilt-channel tracker every call, `~$` approximate figures, spend **quantized to bucket grid** (all values exact multiples of $0.0006) |
| oc3 (ON) | q9 | reuse | $0.0043 | [PROJECT HISTORY] ($0.0051 median) + depletion; tier LOW→CRITICAL as wallet exhausted |
| oc4 (ON) | q5 | $0.002 | $0.0057 | tight budget: CRITICAL reached, run completed (advisory) |
| oc5 (ON) | q7 | $0.001 | $0.0045 | over-budget from turn 1: CRITICAL, advisory-only, still answered |

Money spent: **Claude arm $1.15** (real Anthropic billing), OpenCode arm $0
billed (free tier; $0.0222 retail-measured — the number the budget ran on).

## Cheat audit (all 10 runs)

- no danger tools used (Bash/Write/Edit/web all denied and unused; the one
  Agent-tool use is cc4's by-design delegation)
- zero web requests; zero tool inputs referencing paths outside the sandbox
- zero tool inputs touching `~/.cost-aware-agent`, `127.0.0.1:7331`,
  `db.sqlite`, or `budget set` (regex scan of every tool_use input)
- no negative-token or negative-cost rows stored (intake clamps held)
- OFF runs delivered exactly 0 injections while still measuring cost

## Cost-measurement accuracy

4 of 5 CC runs: daemon spend equals `total_cost_usd` reported by the CLI to
full float precision. cc4 (subagent run) differs by $0.0052 (1.2%) — the
documented session-title haiku call, which writes no usage block into any
transcript. OpenCode has no billing baseline (free tier); its numbers are
retail-priced from pushed token counts, message-id-deduped.

## Findings (new, from this experiment)

1. **CC ≥ 2.1.x renamed the subagent tool "Task" → "Agent".** Capture was
   unaffected (transcripts still land under `subagents/` and the daemon's
   rglob ingests them), but tool allow/deny lists written against "Task"
   silently stop matching. Harness updated to deny/allow both names.
2. **OC session `cli` field raced to ''** — `message.updated` usage push
   reached the daemon before the plugin's `/session/start`, and
   `INSERT OR IGNORE` kept the empty row. Fixed in `db.insert_session`:
   empty identity columns are backfilled, non-empty never clobbered
   (+ regression test, 97/97).
3. **Byte-stability at sub-cent budgets is per-quantized-state, not
   per-session.** The rebuilt-channel tracker is byte-identical only within a
   (spend-bucket, tier, burn-step) state; a deepseek call moves 2–3 buckets,
   so nearly every delivery is a legitimate transition. This is the known
   structural residual of the rebuilt-channel cache tax, now visible in the
   verifier's grid check rather than mis-read as instability.

## Reproduce

```bash
cd experiments/e2e_verify
python3 run_e2e.py                       # ~30 min, ~$1.2 (Claude arm)
python3 verify.py runs/<stamp>           # 165 assertions, exit 0 = all pass
```

Raw artifacts for the reported run: `runs/20260703-185135/` (local,
gitignored; `verify.json` inside is the machine-readable checklist result).
