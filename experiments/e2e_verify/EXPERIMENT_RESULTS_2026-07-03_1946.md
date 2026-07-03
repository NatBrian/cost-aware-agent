# E2E Feature-Verification Experiment — Detailed Results

- **Timestamp**: 2026-07-03 19:46 (runs executed 18:51–19:15 local)
- **Run artifacts**: `experiments/e2e_verify/runs/20260703-185135/` (local, gitignored)
- **Verification snapshot**: `verify_snapshot.json` (committed) — **165 checks, 0 HARD FAIL, 0 warns**
- **Harness commit under test**: `6267267` (post-audit); experiment + fixes committed as `170c32b`
- **Agents**: Claude Code 2.1.199 (`sonnet`, production hook path) · OpenCode 1.17.13 (`opencode/deepseek-v4-flash-free`, production plugin)
- **Dataset**: HotpotQA distractor questions (`../real_cli/data/eval.json`), sandboxed corpus files, retrieval tools only
- **Money spent**: Claude arm **$1.147** (real Anthropic billing) · OpenCode arm **$0 billed** ($0.0222 retail-measured — the number the budget ran on)

## Per-run results

| run | q | question (short) | gold → answer | EM/F1 | wallet | daemon spend | CLI billing | tools | inj | session |
|---|---|---|---|---|---|---|---|---|---|---|
| cc1 OFF | q0 | Juventus vice-captain position | centre-back → "Defender (centre-back)" | 0/.67 | — | $0.2139906 | $0.2139906 **exact** | Glob,Grep×2,Read | 0 | `b524bb6e` |
| cc2 ON | q3 | Miss USA 2015 crowned by | Nia Sanchez → Nia Sanchez | 1/1 | $0.50 | $0.1797222 | $0.1797222 **exact** | Glob,Read | 2 | `d7eb6139` |
| cc3 ON | q9 | Crestfallen artwork nationality | Ukrainian → Ukrainian | 1/1 | reuse | $0.1511742 | $0.1511742 **exact** | Glob,Read×2 | 3 | `f8477173` |
| cc4 ON | q2 | Bud Light mascot rival | Stroh's → "Stroh's (Stroh Brewery Company)" | 0/.40 | $0.50 | $0.4158264 | $0.4210164 (−1.2%) | **Agent**,Glob,Read×2 | 3 | `8f3458a3` |
| cc5 ON | q1 | horror anthology sequel producer | (3 names) → 2 of 3 names | 0/.83 | **$0.06** | $0.1865187 | $0.1865187 **exact** | Glob,Read×2 | 3 | `d9391f1d` |
| oc1 OFF | q0 | (same as cc1) | centre-back → "defender" | 0/0 | — | $0.0026723 | free tier | glob,grep×3 | 0 | `ses_0d86395f0` |
| oc2 ON | q3 | (same as cc2) | Nia Sanchez → Nia Sanchez | 1/1 | $0.006 | $0.0051146 | free tier | glob,read×3 | 4 | `ses_0d8634852` |
| oc3 ON | q9 | (same as cc3) | Ukrainian → Ukrainian | 1/1 | reuse | $0.0043358 | free tier | grep,glob,read | 4 | `ses_0d8630e1e` |
| oc4 ON | q5 | Dost Mohammad Khan killer's country | Afghanistan → Afghanistan | 1/1 | **$0.002** | $0.0056595 | free tier | glob,grep,read×3 | 4 | `ses_0d862de3b` |
| oc5 ON | q7 | Winter's Tale — Mary Robinson's mother | Leontes → "Hermione" (model wrong) | 0/0 | **$0.001** | $0.0044640 | free tier | glob,read×2 | 3 | `ses_0d862825d` |

Accuracy is incidental here (feature test, not benchmark). Note oc5's gold
("Leontes") is itself dubious — Hermione IS the mother; dataset quirk, irrelevant
to the harness claims.

## End-of-session receipts (from daemon log, one per session — all 10 logged)

| session | LLM cost | budget line |
|---|---|---|
| cc1 | $0.21 / 5 calls | $1.00 (session estimate) — 21% used |
| cc2 | $0.18 / 3 calls | $0.50 (project wallet) — 36% used |
| cc3 | $0.15 / 3 calls | $0.50 (project wallet) — **66% used** (depletion: 36%→66% same wallet) |
| cc4 | $0.42 / 6 calls | $0.50 (project wallet) — 83% used |
| cc5 | $0.1865 / 3 calls | $0.0600 (project wallet) — **311% used** (advisory: never blocked) |
| oc1 | $0.00 / 4 calls | $1.00 (session estimate) — 0% used |
| oc2 | $0.0051 / 4 calls | $0.0060 (project wallet) — 85% used |
| oc3 | $0.0043 / 3 calls | $0.0060 (project wallet) — **158% used** |
| oc4 | $0.0057 / 4 calls | $0.0020 (project wallet) — **283% used** |
| oc5 | $0.0045 / 3 calls | $0.0010 (project wallet) — **446% used** |

## Feature evidence highlights

- **Exact cost capture**: cc1/cc2/cc3/cc5 daemon spend == CLI `total_cost_usd`
  to full float precision. cc4 gap $0.0052 = the documented invisible
  session-title haiku call.
- **Subagent capture (H1 fix live)**: cc4's model delegated via the Agent tool;
  daemon ingested 3 `pull-subagent` rows totaling **$0.1247 = 30% of the run's
  spend** — invisible to pre-audit parent-only parsing. Subagent transcript
  archived at `cc4/subagents/agent-acbbb60efbaa4e504.jsonl`.
- **Wallet + history**: cc3 tracker showed budget view $0.331/$0.50 (cc2's spend
  carried over); injected: `[PROJECT HISTORY] Your last 1 session(s) in this
  project cost $0.18–$0.18 (median $0.18)`. Same on oc3 (median $0.0051).
- **Checkpoints (cc5)**: `[BUDGET CHECKPOINT] This project has spent $0.0780
  across its sessions since the last check. In one sentence: what did that
  spend buy? ...` — wallet-scope wording (audit fix L8) confirmed live.
- **Tier escalation**: cc5 HIGH→CRITICAL; oc3 LOW→CRITICAL; oc4/oc5 CRITICAL.
- **on_change anti-tax (cc2)**: 7 hook fires (1 SessionStart, 2 PreToolUse,
  2 PostToolUse, 1 Stop, 1 SessionEnd) → only 2 injections delivered.
- **/turn/end split (M3 fix live)**: Stop fired per response → ingest only;
  sessions stayed `active` mid-run; SessionEnd fired naturally at CLI exit on
  all 5 CC runs and marked `ended` (0 forced closes on CC; OC closed by
  harness by design — no end event exists in OpenCode).
- **Rebuilt-channel quantization (oc runs)**: every rendered spend value an
  exact multiple of the bucket step (10% of wallet); `~$` approximate figures;
  deliveries byte-identical within a (spend-bucket, tier, burn-step) state.
- **Delivery ground truth (CC)**: injected "Budget Tracker" text present in
  Claude Code's own transcript files 2–6× per ON run — the model saw it.
- **OFF arms**: 0 injections, cost still measured identically, receipts logged.

## Cheat audit — clean, all 10 runs

- No danger tools (Bash/Write/Edit/web denied AND unused; cc4's Agent use is
  the by-design delegation).
- 0 web requests; 0 tool inputs with paths outside the sandbox.
- 0 tool inputs referencing `~/.cost-aware-agent`, `127.0.0.1:7331`,
  `db.sqlite`, or `budget set` (regex scan of every tool_use input).
- No negative-token/negative-cost rows stored (intake clamps held).

## Bugs found by this experiment (both fixed in `170c32b`)

1. **CC ≥ 2.1.x renamed the subagent tool "Task" → "Agent"** — capture
   unaffected, but allow/deny lists keyed on "Task" silently stop matching.
   Harness + verifier now cover both names.
2. **OpenCode session `cli` field raced to `''`** — the `message.updated`
   usage push reached the daemon before the plugin's `/session/start`;
   `INSERT OR IGNORE` kept the empty row. Fixed: `db.insert_session` backfills
   empty identity columns, never clobbers non-empty (+ regression test; 97/97).

## What this experiment does NOT claim

No cost-savings claim. Runs were feature-targeted, not paired A/B — "does the
budget make sessions cheaper" was answered by `../real_cli` (savings only in
discretionary/runaway work) and the remaining open question (does true cost
visibility make a model self-terminate planted waste) is backlog item 9, the
slack benchmark — not yet run.

## Reproduce / debug

```bash
cd experiments/e2e_verify
python3 run_e2e.py              # ~25 min, ~$1.2 Claude arm
python3 verify.py runs/<stamp>  # exit 0 = all assertions pass
```

Per-run debugging: each `runs/<stamp>/<run>/` holds `task.txt`,
`events.jsonl(+.stderr)`, `transcript.jsonl` + `subagents/` (CC),
`hook.jsonl` (CC, every hook fire with exact injected text),
`daemon_dump.json` (usage rows, tool calls, every injection), `result.json`.
Receipts: daemon stdout log (`../real_cli/daemon_sweep.out` for these runs).
