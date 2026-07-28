# foundation — CASSI simplified pipeline-validation run

Step-budget agents on HotpotQA: can GRPO with per-step rubric rewards from a
prompted Qwen3.6-27B judge teach a ReAct agent to stop itself — beating
budget-in-prompt (A1) and hard-cutoff enforcement (A2)?

- Project overview + directory map: `../README.md`
- Plan: `../paper_plan_v2_1_foundation.md` (read first)
- Stage docs + progress: `../foundation_tasks/` (`PROGRESS.md` = live tracker)
- Every constant: `configs/foundation.yaml` (single source of truth)
- Setup: `make venv` then `make test`
- Old cassi codebase: archived at `../archived/cassi_2026-07` — READ-BANNED
  (built fresh on purpose; see F0 doc)
