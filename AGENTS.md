# AGENTS.md — read this first (any AI agent, zero context assumed)

This repository has TWO parts:

1. **`cost_aware_agent/`** — an inference-time budget/cost-metering harness for
   frontier agents (see `README.md`, `VISION.md`). Working software; not the
   current focus.
2. **`research/`** — the CASSI research project: an ICLR 2027 paper training LLM
   agents to know when further work stops being worth its cost. **This is the
   active work.**

## If you are here to continue the work → START HERE

1. Read `research/cassi/HANDOFF.md` — current implementation status, what is
   pending (GPU experiments), exact next commands, and every decision already made.
2. Read `research/paper_plan_v2.md` — the full research plan; it is the SOURCE OF
   TRUTH for everything under `research/cassi/` (each module cites its sections).
3. Verify nothing rotted: `cd research/cassi && python -m pytest tests/ -q`
   (expect 109+ passed, CPU-only, ~5 s).

## Hard rules

- **Never read `research/archived*` folders** (stale outputs kept for history only;
  user instruction — they will bias you).
- `research/paper_plan_v2.md` supersedes the archived `paper_plan.md` (v5). Where
  they disagree, v2 wins. Do not re-litigate decisions logged in its §14 changelog.
- **Git identity:** this repo belongs to the GitHub account **NatBrian**. On the
  lab machine this is enforced by folder-scoped git config (author
  `Nathanael Brian <22826533+NatBrian@users.noreply.github.com>`, dedicated SSH key
  via host alias `github-natbrian`) — see `/root/dataDisk/liangsheng/Brian/AGENTS.md`.
  Never commit/push with any other identity; no AI co-author lines in commits.
- **GPUs (lab machine):** acquire before use, release after:
  `eval $(/mnt/src/zhanka/gpu_acquire.sh N)` … `/mnt/src/zhanka/gpu_release.sh`
  (N=2 collection/stopper-SFT, 4–8 GRPO). Never kill GPU occupier processes.
- Datasets/checkpoints/clones are NOT in git (`research/cassi/.gitignore`); scripts
  `p0_setup.sh`/`p1_data.sh` regenerate them. The 101 reference-paper PDFs under
  `research/papers/` are gitignored and exist only on the lab machine.
