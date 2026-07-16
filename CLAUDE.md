# CLAUDE.md — read this first (zero context assumed)

This repository has TWO parts:

1. **`cost_aware_agent/`** — an inference-time budget/cost-metering harness for
   frontier agents (see `README.md`, `VISION.md`). Working software; not the
   current focus.
2. **`research/`** — the CASSI research project: an ICLR 2027 paper training LLM
   agents to know when further work stops being worth its cost. **This is the
   active work.**

## If you are here to continue the work → START HERE

1. Read `research/cassi/PROJECT_GUIDE.md` — the master plain-language explanation
   (what the research is, glossary mapping paper terms to code symbols, pipeline
   walkthrough, document map, status, the "ten commandments" invariants).
2. Read `research/cassi/HANDOFF.md` — current status, pending GPU work, exact next
   commands, post-experiment adjustment map, every decision already made.
3. Read `research/paper_plan_v2.md` — the full research plan; it is the SOURCE OF
   TRUTH for everything under `research/cassi/` (each module cites its sections).
4. Verify nothing rotted — the canonical ritual is PROJECT_GUIDE §7 (activate
   `research/cassi/.venv`, then pytest → expect "114 passed, 1 skipped"; the
   train_grpo --dry-run; the paper make; and `git config user.name` → must print
   "Nathanael Brian" — if it prints anything else, STOP before committing).
   Never "verify" by re-running scripts/p0_setup.sh (it's an installer; P0 is done).

## Hard rules

- **Never read anything under `research/archived/`** (that single directory is the whole ban; stale outputs kept for history only —
  user instruction — they will bias you).
- `research/paper_plan_v2.md` supersedes the archived `paper_plan.md` (v5). Where
  they disagree, v2 wins. Do not re-litigate decisions logged in its §14 changelog.
- **Git identity:** this repo belongs to the GitHub account **NatBrian**. On the
  lab machine this is enforced by folder-scoped git config (author
  `Nathanael Brian <22826533+NatBrian@users.noreply.github.com>`, dedicated SSH key
  via host alias `github-natbrian`) — see `/root/dataDisk/liangsheng/Brian/CLAUDE.md`.
  Never commit/push with any other identity; no AI co-author lines in commits.
- **GPUs (lab machine):** acquire before use, release after:
  `eval $(/mnt/src/zhanka/gpu_acquire.sh N)` … `/mnt/src/zhanka/gpu_release.sh`
  (N=2 collection/stopper-SFT, 4–8 GRPO). Never kill GPU occupier processes.
- Datasets/checkpoints/clones are NOT in git (`research/cassi/.gitignore`); scripts
  `p0_setup.sh`/`p1_data.sh` regenerate them. The 101 reference-paper PDFs under
  `research/papers/` are gitignored and exist only on the lab machine.
