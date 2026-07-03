# swe_ab — SWE-bench Lite A/B: does the money budget reduce cost?

The follow-up to `../real_cli` (HotpotQA), which proved the plumbing but found
no savings on low-slack retrieval (~4 required tool calls/question, nothing to
trim). This experiment asks the actual product question on a REAL dataset with
discretionary slack: **SWE-bench Lite** — real GitHub issues, graded by the
dataset's own FAIL_TO_PASS / PASS_TO_PASS tests.

## Design

- **Arms differ in exactly one thing**: daemon `inject_enabled` (+ a project
  wallet on ON). Same prompts, same sandboxes, same tools, same measurement —
  the OFF arm's hooks/plugin still POST everything to the daemon; the master
  gate (`daemon.py _gate`) swallows delivery only.
- **Wallet rule (pre-registered before any ON run)**:
  `wallet = 0.5 × per-instance OFF median cost`, set through the real CLI
  (`cost-aware-agent budget set <w> --project-dir <sandbox>`). Fresh sandbox
  per run → fresh wallet, no history bleed between pairs.
- **Platforms**: Claude Code (sonnet, production hook channel) and OpenCode
  (deepseek-v4-flash-free, production plugin channel; money = daemon retail
  price of pushed tokens — the same number the budget runs on).
- **Instances** (6, mechanically screened: `pip install -e .` on py3.11 works,
  F2P fails at base, passes with gold patch, P2P sample passes):
  pytest-11143, pytest-11148, pylint-6506, sympy-24213, sympy-22714,
  sympy-21612.
- **Sandbox**: checkout at `base_commit`, venv installed while history is
  still present (setuptools-scm needs the tags), then `.git` STRIPPED and
  re-initialized — a full clone contains the gold fix in future commits.
  Gold patch and test_patch never enter the sandbox.
- **Grading (SWE-bench semantics, harness-side after the agent exits)**:
  `agent.patch` = git diff; test files restored to base; test_patch applied;
  F2P + 3-test P2P sample run in the sandbox venv. success = all pass.
- **Audit per run**: no web tools (denied both platforms), regex scan of every
  tool input for harness tampering (`.cost-aware-agent`, `127.0.0.1:7331`,
  `db.sqlite`, `budget set`), network attempts in bash (`curl|wget|pip
  install|git fetch|https?://`), absolute paths outside the sandbox.
- **Safety**: CC runs stop past $15 cumulative (`MAX_CC_SPEND_USD`).

## Run

```bash
python3 run_swe.py --arm off --seeds 2          # both platforms, 24 runs
python3 run_swe.py --arm on  --seeds 2          # computes wallets.json first
python3 analyze_swe.py                          # paired t, success, audit
```

Per-run artifacts under `runs/<platform>/<arm>-s<seed>/<iid>/`: `task.txt`,
`events.jsonl(+.stderr)`, `transcript.jsonl` + `hook.jsonl` (CC),
`daemon_dump.json` (every usage row + every delivered injection),
`agent.patch`, `grade_f2p.txt`, `grade_p2p.txt`, `result.json`.
