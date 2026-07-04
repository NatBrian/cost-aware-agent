# SWE-bench A/B — does the money budget reduce cost on real coding tasks?

- **Timestamp**: 2026-07-03 22:55 local
- **Harness commit under test**: (see `runs/*/meta.json` `git_sha`)
- **Agent**: Claude Code 2.1.199, model `sonnet`, production hook channel
- **Dataset**: SWE-bench Lite — 6 real GitHub issues, graded by each project's
  own FAIL_TO_PASS / PASS_TO_PASS tests
- **Money spent**: **$14.65 real Anthropic billing** (OFF $9.16 + ON $5.49)
- **OpenCode arm**: complete (2026-07-04, quota returned) — see the OpenCode
  section below. Plumbing fully verified (12/12 feature checks); cost result is
  the OPPOSITE of Claude Code — budget makes the weak deepseek model *more*
  expensive (+336%, all 7 clean pairs) via the rebuilt-channel cache tax, with
  no behavioral saving (success unchanged). $0 billed (free tier).

## Headline

**An advisory money budget cut Claude Code's cost ~29% on these coding tasks,
and the saving is statistically significant** — the opposite of the earlier
HotpotQA A/B (`../real_cli`), which found +2.8% (not significant). The one
thing that changed is the dataset: SWE-bench issues have *discretionary slack*
(how much to explore, how many times to re-verify), and that slack is exactly
what a dollar signal trims.

| metric | OFF | ON (budget) | delta |
|---|---|---|---|
| **paired cost/task** (n=10) | $0.6192 | $0.4369 | **−29.4%, significant** (t=2.53, 95% CI of saving [$0.019, $0.345]) |
| tools/task (paired) | 14.4 | 10.7 | −3.7 |
| total money, all 12 runs | $9.16 | $5.49 | **−40%** |
| success (timeouts = fail) | 8/12 = 0.67 | 9/12 = **0.75** | +1 |
| timeouts (900s cap) | **2** | **0** | budget kept runs bounded |
| injections delivered | 0 (gate off) | 5.1/run | gating verified |

The paired stat (n=10) is a **conservative lower bound**: the two runs with the
biggest ON savings are *excluded* from it because their OFF pair hit the 900s
wall (no clean OFF cost to pair against). Counting everything, ON is 40% cheaper
and has a *higher* success rate.

## Per-instance (all 12 runs each arm, clean)

```
instance                 seed|   OFF$ OFtl OFok OFto |    ON$ ONtl ONok ONinj|  saved$
pylint-6506              s0  |  0.695   16  F    F   |  0.461   13  F     7 | paired +0.235
pylint-6506              s1  |  1.020   27  F    F   |  0.717   24  F     8 | paired +0.303
pytest-11143             s0  |  0.306    7  T    F   |  0.295    6  T     3 | paired +0.010
pytest-11143             s1  |  0.392   11  T    F   |  0.244    3  T     2 | paired +0.148
pytest-11148             s0  |  1.653   29  T    F   |  0.948   22  T    10 | paired +0.705
pytest-11148             s1  |  2.046   31  F    T*  |  0.307    6  T     2 | unpair +1.740
sympy-21612              s0  |  0.865   24  T    F   |  0.505   14  F     8 | paired +0.360
sympy-21612              s1  |  0.926   28  F    T*  |  0.811   23  T    10 | unpair +0.115
sympy-22714              s0  |  0.314    7  T    F   |  0.296    6  T     3 | paired +0.018
sympy-22714              s1  |  0.336    9  T    F   |  0.346    8  T     4 | paired -0.011
sympy-24213              s0  |  0.303    7  T    F   |  0.254    4  T     3 | paired +0.049
sympy-24213              s1  |  0.307    7  T    F   |  0.303    7  T     3 | paired +0.004
```
`OFok/ONok` = graded success; `OFto` T* = OFF timed out at 900s (counts as
fail; ON completed the same task). `saved$` positive = ON cheaper on that pair.
11 of 12 pairs cheaper under budget; the one negative (sympy-22714 s1, −$0.011)
is noise.

**Slack is the variable.** Savings concentrate on the high-slack instances:
pytest-11148 ($1.65→$0.95 to +$1.74 vs the timed-out seed), pylint-6506
(+$0.24/+$0.30), sympy-21612 (+$0.36/+$0.12). The cheap, low-slack sympy issues
(24213, 22714) barely move — there was nothing discretionary to trim. Same
lesson as HotpotQA, now with instances that actually *have* slack.

## The honest caveat — budget can cut needed work

On **sympy-21612 s0** the budget made the model stop earlier ($0.505 vs $0.865)
and it **failed where OFF succeeded**. That is the real risk of an advisory
budget: pressure applied to a task that genuinely needed the spend. It happened
once in 12 ON runs here. The countervailing evidence: on pytest-11148 s1 and
sympy-21612 s1 the budget *prevented* a runaway that OFF let time out — so the
net success effect is positive (0.75 vs 0.67). Budget helps runaway cases more
than it hurts genuine ones, at this budget size (0.5× OFF median), but the hurt
is real and non-zero.

## Integrity — one contamination found and fixed

The audit caught a real cheat during bring-up (exactly its job):

- **pytest-11148 ON s1 (first run) downloaded the fix from PyPI.** With bash
  network open, the model ran `pip download pytest==8.0.0`, extracted the
  tarball, and grepped the *released, already-fixed* source. That success was
  contaminated. Root cause: only WebSearch/WebFetch **tools** were denied — bash
  network egress was not sandboxed.
- **Fix**: a black-hole proxy env on the agent subprocess only (the venv is
  built before the agent runs, so its install is unaffected), with Anthropic
  whitelisted via `NO_PROXY` so the CLI still reaches its own API. Verified:
  `pypi.org` → Connection refused, `api.anthropic.com` → reachable.
- The contaminated run was **flagged (`contaminated: true`), not deleted** —
  archived to `runs/_contaminated/`, excluded by the analyzer — and re-run
  clean. Clean result: $0.307, 6 tools, **0 network attempts**. (Notably the
  cheating run was *more* expensive, $1.039 — the download+grep wasted money.)

Rest of the audit, all current runs: **0 harness-tamper refs** (no
`.cost-aware-agent` / daemon-port / `db.sqlite` / `budget set` in any tool
input), **0 danger tools**, 0 negative-cost rows. Remaining `net_refs` (7) are
in pre-block runs and are benign: a `pip download` that failed/timed out, and
regex false-positives on in-venv `python -c` repro scripts (no host reached).
`outside_sandbox` path refs (19) are read-only `find /` package/wheel searches
that wandered into other host dirs — no writes, no gold access (SWE-bench gold
lives in the HF dataset, never on disk).

## Findings

1. **Slack is the variable that decides whether an advisory budget saves
   money.** Identical harness, identical prompts — HotpotQA (no slack) → +2.8%
   n.s.; SWE-bench (real discretionary slack) → −29% significant. Within this
   experiment the saving tracks per-instance slack: high on pytest-11148 /
   pylint-6506 / sympy-21612, ~0 on the cheap sympy issues. The budget trims
   discretionary exploration and re-verification, not required work.

2. **The budget prevents runaway timeouts.** OFF hit the 900s wall twice
   (pytest-11148 s1, sympy-21612 s1) — both *failed*. ON completed both cheaper
   and *succeeded*. Net success went up under budget (0.67 → 0.75), not down.

3. **But budget can also cut work that was actually needed.** sympy-21612 s0:
   ON stopped earlier ($0.505 vs $0.865) and failed where OFF succeeded. Real
   downside, 1/12 here. At this wallet size (0.5× OFF median) the runaway wins
   outnumber the over-cut losses, but the loss is not zero — budget size is a
   real accuracy/cost knob, not free.

4. **Cheating tends to cost *more*, not less.** The one contaminated run
   (pip-download the fix) spent $1.039; the honest re-run solved the same issue
   for $0.307. The detour to fetch and grep a release burned tokens. A money
   budget therefore has a mild anti-cheat pressure of its own — the expensive
   shortcut looks expensive.

5. **Bash network egress was a real sandbox hole.** Denying WebSearch/WebFetch
   *tools* does nothing about `curl`/`pip`/`git` run through bash. A model
   reached PyPI and read fixed source. Any tool-level web denial that forgets
   bash is porous. Fixed with a subprocess proxy black-hole (agent-only,
   Anthropic whitelisted). Lesson for the harness's own threat model: the daemon
   port is localhost-open and bash is network-capable — sandboxing must be at
   the process/network layer, not the tool-allowlist layer.

6. **The daemon measures money perfectly even when the CLI is killed.** A 900s
   timeout emits no `result` event, so the CLI reports no cost and no
   session_id — but every completed turn was already ingested via the Stop →
   `/turn/end` hook. Recovering the session_id from `hook.jsonl` read back the
   full $2.05 of spend from the daemon dump. Measurement does not depend on the
   agent exiting cleanly; only the *experiment's* sid plumbing did (now fixed).

7. **setuptools-scm vs. history-stripping** (methodology gotcha worth keeping):
   the venv `-e` install must run while real git history/tags are present, or
   the package version resolves to `0.1.dev1` and pytest's own `minversion`
   gate rejects the test run at grade time. Strip `.git` *after* install.

## Harness features exercised (all live-verified)

- **OFF/ON gating**: OFF arm delivered **0 injections** across all 12 runs while
  the daemon still measured every dollar (exact `total_cost_usd` match); ON
  delivered 2–10/run through the `on_change` policy.
- **Real depleting-dollar tracker**: e.g. pytest-11148 ON —
  `LLM cost used: $0.48, remaining (of project wallet): $0.45 · Burn rate: $0.48
  in last 10 min · Tier: MEDIUM · Decide yourself…` — plus `[BUDGET CHECKPOINT]`
  self-audit at spend milestones.
- **Tier escalation** HIGH→MEDIUM→LOW→CRITICAL observed within single sessions.
- **Project wallet** set through the real CLI (`budget set --project-dir`),
  wallet = 0.5× per-instance OFF median (rule pre-registered before any ON run).
- **Cost capture survives a killed CLI**: a 900s timeout emits no `result`
  event (no session_id), but the daemon had already ingested every turn — the
  session_id was recovered from `hook.jsonl` and $2.05 of spend read back from
  the daemon dump. Runner now does this recovery automatically.

## What this does and does not claim

- **Does**: on real coding tasks with discretionary slack, an advisory money
  budget significantly reduces Claude Code's spend (~29% paired, 40% total)
  without lowering the success rate — and prevents runaway timeouts.
- **Does not**: prove this for OpenCode (arm deferred on quota), prove it for
  tasks without slack (low-slack instances showed ~0 effect, as expected), or
  claim zero downside (one run failed under budget pressure that OFF solved).
- Single agent (sonnet), 6 instances, 2 seeds — a real but small A/B. The
  effect is significant at this n; a larger sweep would tighten the CI.

## OpenCode arm (deepseek-v4-flash-free, 2026-07-04)

Same 6 instances, same wallets rule (0.5× per-instance OFF median), production
plugin (rebuilt/system-transform channel). OpenCode free tier bills $0; money =
the daemon's retail price of the pushed tokens (the number the budget runs on).

**Harness plumbing: 12/12 machine checks pass** (`verify_oc.py`, `verify_oc.json`):
OFF gating (0 injections across all 12 OFF runs), ON injection delivery on the
rebuilt channel, retail cost exact (`cost_usd == daemon_spent`), tier escalation
(HIGH/MEDIUM/CRITICAL seen live), Budget Tracker block delivered, project-wallet
scope wording, `~$` bucket-grid quantization, zero harness-tamper refs, every
network attempt flagged (not silent), **subagent capture** (deepseek spawned
`task` subagents — the daemon ingested all of them, the same feature proven for
Claude Code subagents), and session tracking. Every OpenCode harness feature
works end-to-end on real coding tasks.

> **Update:** the cost penalty below was the *old* plugin (tracker in the system
> prompt). It has since been fixed — Follow-up #1, validated: delivering the
> tracker at the end of context instead flips the full-arm result from +336% to
> −8.4% (≈break-even), with per-turn freshness kept. The analysis below is the
> pre-fix diagnosis that motivated the fix.

**Cost result (pre-fix) — the opposite of Claude Code.** Full arm complete
(12 OFF + 12 ON runs). Every one of the 7 clean OFF/ON pairs is *more* expensive
under budget — the direction is unanimous:

| metric (7 clean pairs) | OFF | ON | delta |
|---|---|---|---|
| cost/task | $0.0095 | $0.0413 | **+336% (4.3×), t=−1.82, not significant** |
| tools/task | 16.1 | 13.3 | −2.8 |
| success | 0.71 | 0.71 | unchanged |
| injections/run | 0 | 13.6 | rebuilt channel |

Per-instance (paired cells): pytest-11143 $0.0044→$0.0053 and $0.0039→$0.0151;
pytest-11148 $0.0189→**$0.1540** (29 injections, the worst blowup);
sympy-24213 $0.0057→$0.0136 and $0.0045→$0.0294; sympy-22714 $0.0157→$0.0365;
sympy-21612 $0.0132→$0.0351. Retail totals: OFF $0.0975 → ON $0.4798 ($0 billed,
free tier). The −336% is not statistically significant at n=7 (t=−1.82; high
variance from the pytest-11148 outlier), but **all 7 pairs point the same way**,
so the direction is not in doubt — only its exact size.

**Mechanism, verified from the daemon token logs (not asserted):**
- It is *not* a cost bug: per-call costs sum exactly to the daemon's spend
  (e.g. pytest-11148 ON: 29 rows sum to $0.1540 = spent), so cost is
  tokens×price with no double-count.
- It is *not* the model doing more work: ON made *fewer* calls than OFF in 3 of
  4 sampled pairs (16→6, 33→16) with less output — each call is simply pricier.
- It *is* the rebuilt-channel cache tax: cache-hit rate collapses from 93–96%
  (OFF) to 23–44% (ON), and fresh input tokens multiply 3.6×–30×. Root cause:
  all 29 injections in the worst run are byte-*unique* (spend $0.0000→$0.0236,
  tier HIGH→MEDIUM→CRITICAL, burn line appearing) — each change is a new
  system-prompt prefix, so deepseek's prompt cache invalidates from that point
  and everything after is reprocessed fresh.

So the extra cost is real and correctly measured, caused by **our harness's
injection design** (a *changing* signal in OpenCode's every-call-rebuilt system
prompt) — a design weakness to fix (Follow-up #1), not a measurement bug and not
the model's fault. Behaviorally deepseek does not respond to the budget (tools
flat/down, success unchanged) — consistent with the prior HotpotQA finding — so
there is no offsetting saving, only the tax.

**Honest scale caveat:** the +336% is a worst case for the *sub-cent budget
regime*, not a steady-state number. deepseek is so cheap the wallet is ~$0.008,
so the 10%-bucket quantization step (~$0.0008) is finer than one call's spend —
nearly every call crosses a bucket and re-changes the text, maximizing the tax.
At realistic dollar-scale budgets a single call is a tiny fraction of the
wallet, the tracker text stays byte-stable for many calls, and the tax is far
smaller. The direction (budget costs more on a weak model that ignores it) holds
regardless; the magnitude is inflated by the tiny-budget scale. This is the
honest limit of the approach: **the budget saves money on a capable model that
acts on it (Claude Code, −29% significant) and costs money on a weak model that
ignores it while paying the injection tax (OpenCode, +336%).** Task slack alone
is not enough — the model must also be capable enough to act on the signal.

**Reliability caveats (external, not harness bugs):**
- deepseek is weak: it solves the easy instances (sympy/pytest small patches,
  ~$0.004–0.016) and loops/times out on the hard ones — the same instances
  Claude Code also failed. Success is identical across arms (0.71).
- The free tier hit a ~2-hour degradation window mid-experiment (timing out even
  on a trivial "reply OK"). The runner recovers spend from the daemon when the
  CLI is killed, and retries transient stream-error timeouts once; the few runs
  that still yielded nothing were re-run after the tier recovered. All 12 ON
  runs are now captured; 7 form clean (non-timeout, both-arm-valid) pairs.

**Bugs found and fixed during this arm** (all in the experiment scaffolding,
none in the daemon):
1. The Claude-Code black-hole proxy hung OpenCode at startup (it fetches
   models.dev / its gateway through the same proxy env). Split into
   platform-specific policies: proxy for CC, `PIP_NO_INDEX`-only for OC.
2. OpenCode buffers `--format json` stdout and loses it on SIGKILL → recover the
   session + spend from the daemon (plugin pushed live).
3. OpenCode's stdout tool list is unreliable (1 parsed vs 20 in the daemon) →
   audit and count tools from the daemon dump, so the cheat audit can never see
   a truncated subset (`reaudit_oc.py` backfills earlier runs).
4. deepseek free-tier stream errors → retry once on empty/zero-work timeouts.

## Follow-ups worth doing (ranked)

1. **Cut the rebuilt-channel cache tax — DONE + validated** (`ee8724c`). The
   entire OpenCode +336% was the tracker in the system prompt (prefix), rebuilt
   every call, busting deepseek's prompt cache. Fix: keep the system prefix
   byte-stable and deliver the live tracker at the *end* of context (appended to
   the tool result via `tool.execute.after`, `channel:"rebuilt"`) — the tail
   extends the cache instead of invalidating it, and the model still sees the
   current spend after every tool call (freshness preserved; rejected the
   "inject less often" alternative because the model reasons fresh every turn
   and should see the current number every turn). Claude Code path unchanged.
   **Validated across the full OpenCode arm (12 OFF + 12 Fix-1 ON runs):**

   | paired mean (n=7) | cost/task | vs OFF |
   |---|---|---|
   | OFF | $0.0082 | 1.0× |
   | old ON (tracker in system prompt) | $0.0202 | **2.5×** |
   | Fix-1 ON (tracker at end of context) | $0.0075 | **0.9×** |

   Per-instance the old-plugin multipliers (0.6×, 8.2×, 1.2×, 3.9×, 8.1×, 2.8×,
   2.7×, 2.3×, 2.4×, 6.6×) all collapse to ≈1.0× (0.6×, 1.5×, 1.0×, 1.1×, 0.6×,
   1.1×, 1.0×, 1.0×, 0.8×, 1.2×). Worst case pytest-11148: 8.1× → 0.6×.
   Aggregate paired result flips from **+336% (old) to −8.4% (Fix-1, n.s.,
   ≈break-even)**, success unchanged (0.71), 12/12 feature checks still green,
   13 injections/run retained (full per-turn freshness). **The cache tax is
   eliminated — the budget is now free to run on OpenCode too**, not just
   Claude Code.
2. **Deterministic OpenCode session capture.** Timeout recovery currently uses a
   "newest opencode session in the daemon" heuristic — safe only because runs
   are sequential. Have the plugin write its `session_id` to a known file at
   `/session/start` and read that instead. Removes the fragile heuristic.
3. **Budget-size sweep** (methodology, costs money). 0.5× median cost one Claude
   Code success (cut needed work). Try 0.3/0.5/0.7× to find the max-saving,
   min-accuracy-loss point.
4. **More n** (methodology). Claude Code n=10 is significant; OpenCode n=7 is
   directionally unanimous but not significant. Doubling instances/seeds tightens
   both CIs.

Not worth it: making deepseek respond to the budget (model limitation, not a
harness fix); the timeout-confounds-cost nuance (already excluded from pairs).

## Reproduce / debug

```bash
cd experiments/swe_ab
python3 run_swe.py --arm off --seeds 2      # ~$9, 12 runs
python3 run_swe.py --arm on  --seeds 2      # computes wallets.json, ~$5
python3 analyze_swe.py                       # paired t -> analysis.json
```

Per-run artifacts under `runs/claude/<arm>-s<seed>/<iid>/`: `task.txt`,
`events.jsonl(+.stderr)`, `transcript.jsonl`, `hook.jsonl` (every hook fire +
injected text), `daemon_dump.json` (every usage row + every delivered
injection), `agent.patch`, `grade_f2p.txt`, `grade_p2p.txt`, `result.json`.
Instance definitions + gold/test patches: `data/instances.json`. Contaminated
run preserved under `runs/_contaminated/`.
