# CLAUDE.md — read this first (zero context assumed)

This repository has THREE parts:

1. **`cost_aware_agent/`** — an inference-time budget/cost-metering harness for
   frontier agents (see `README.md`, `VISION.md`). Working software; not the
   current focus — but its design informs the research (see
   `cost_aware_agent/prompts.py` facts-not-advice philosophy).
2. **`research/foundation/` + the foundation docs** — the ACTIVE work: the
   simplified pipeline-validation run of the CASSI research (step-count budgets,
   HotpotQA, GRPO with prompted-judge rewards).
3. **`research/paper_plan_v2_1.md`** — the full ICLR 2027 paper plan. Still the
   source of truth for the eventual paper; the foundation run de-risks it first.

## If you are here to continue the work → START HERE

1. Read `research/paper_plan_v2_1_foundation.md` — the foundation plan (what,
   why, the three arms, the pre-registered gate). Reviewed and approved by the
   user doc-by-doc on 2026-07-22.
2. Read `research/foundation_tasks/PROGRESS.md` — the live tracker: what stage
   the implementation/experiments are in, what's next, decisions locked.
3. Read the stage docs `research/foundation_tasks/F0…F7*.md` for whatever stage
   is current.
4. `research/paper_plan_v2_1.md` (with `paper_plan_v2_simple.md` as its
   plain-language companion) only when full-paper context is needed — the
   foundation deliberately simplifies it; where they conflict, the foundation
   doc governs foundation work.
5. Verify nothing rotted: `cd research/foundation && make test` (all green);
   `git config user.name` → must print "Nathanael Brian" — if it prints
   anything else, STOP before committing.

## Hard rules

- **Never read anything under `research/archived/`** — that now includes
  `archived/cassi_2026-07` (the previous CASSI codebase, archived 2026-07-22 by
  user decision: the foundation is built fresh so its correctness never depends
  on code we no longer trust). Rescued BEFORE archiving and still readable:
  data artifacts → `research/data_shared/` (gitignored),
  first-run reports + paid-measurement records → `research/reports_run1/`.
- `research/paper_plan_v2_1.md` supersedes `paper_plan_v2.md` where they differ;
  both supersede the archived v5 plan. Do not re-litigate decisions in their
  changelogs (§14, §20) or the foundation plan's simplification table (§1).
- **Git identity:** this repo belongs to the GitHub account **NatBrian**,
  enforced by folder-scoped git config (author
  `Nathanael Brian <22826533+NatBrian@users.noreply.github.com>`, dedicated SSH
  key via host alias `github-natbrian`) — see `/home/liangsheng/brian/CLAUDE.md`.
  Never commit/push with any other identity; no AI co-author lines in commits.
- **GPUs (lab machine):** acquire before use, release after:
  `eval $(/mnt/src/zhanka/gpu_acquire.sh N)` … `/mnt/src/zhanka/gpu_release.sh`
  (N=2 collection/serving, 4–8 GRPO). Never kill GPU occupier processes.
  User `yongyue` ignores the locks and cycles servers — wait for stable-free
  windows, never race or co-locate mid-load.
- Datasets/indices/checkpoints are NOT in git (`research/data_shared/`,
  `research/foundation/data/`); scripts under `research/foundation/scripts/`
  regenerate them. The 101 reference-paper PDFs under `research/papers/` are
  gitignored and exist only on the lab machine.
- Every experiment constant lives in `research/foundation/configs/foundation.yaml`
  (single source of truth) — never hard-code one anywhere else.
