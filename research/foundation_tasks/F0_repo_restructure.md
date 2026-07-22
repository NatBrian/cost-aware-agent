# F0 — Repo restructure & archive of cassi

**Goal:** a clean starting point: old code archived, fresh foundation package with a
test scaffold, project docs pointing at the foundation plan.

**Why:** the senior's point is a *robust foundation* — we don't know whether the old
cassi code embodies correct decisions, so the foundation must not inherit it.
Archiving under `research/archived/` (read-banned by CLAUDE.md) makes that
non-inheritance mechanical, the same way the git-identity rule is mechanical.

## Work items

1. **Rescue non-git data artifacts first.** The retrieval index / wiki corpus /
   downloaded datasets under `research/cassi/` are gitignored, take hours to
   rebuild, and are NOT "stale documents" — move them to a neutral shared location
   (`research/data_shared/`) BEFORE archiving, so reusing them never requires
   reading archived content. List what was moved in the commit message.
   **Also rescue the human-facing records:** `cassi/experiments/reports/` (the
   plain-language results of the first experiment run) move to
   `research/reports_run1/` — they are Brian's experimental records, not stale AI
   guidance, and must stay readable after the archive.
2. `git mv research/cassi research/archived/cassi_2026-07` (history preserved).
3. Create the new package skeleton:

```
research/foundation/
├── README.md            # 20 lines: what this is, how to run, pointer to the plan
├── configs/foundation.yaml   # single config file, single source of truth
├── agent/               # ReAct agent, step-budget harness, system prompts
├── envs/                # HotpotQA local-retrieval env wrapper
├── collect/             # trajectory collection script + JSONL schema (F2)
├── reward/              # rubric prompt, judge client, parser, reward calc (F3)
├── train/               # verl GRPO launcher + reward wiring (F5)
├── eval/                # metrics, comparisons (F6)
├── analysis/            # figures + report generation (F7)
├── scripts/             # f1_data.sh … f7_report.sh, one per stage
└── tests/               # pytest; CPU-only; mocked judge/env where needed
```

4. Python env: new `research/foundation/.venv` (or reuse machine conventions);
   `requirements-cpu.txt` / `requirements-gpu.txt` split as before.
5. Update `cost-aware-agent/CLAUDE.md`: active work = foundation; reading order =
   `paper_plan_v2_1_foundation.md` → `foundation_tasks/` → `paper_plan_v2_1.md`
   (full plan, still source of truth for the eventual paper); cassi_2026-07 is
   archived and read-banned like the rest of `research/archived/`.
6. Update the persistent memory index (cassi-paper-status) to record the pivot.
7. Seed `tests/` with a trivial passing test + CI-style `make test` target so the
   "tests green" ritual exists from day one.

## Depends on / feeds

Depends on: user review of the plan docs (this is the first coding step).
Feeds: everything.

## Done criterion

`ls research/cassi` fails; `make -C research/foundation test` passes; CLAUDE.md
describes the new layout; data artifacts exist under `research/data_shared/`;
one commit, authored by Nathanael Brian.
