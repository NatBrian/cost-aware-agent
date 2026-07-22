# Experiment-start checklist (I7) — follow in order, gates between stages

Pre-flight (any session):
- [ ] `make test` → all green; `git config user.name` → "Nathanael Brian"
- [ ] GPUs: `eval $(/mnt/src/zhanka/gpu_acquire.sh N)` before / release after;
      watch for yongyue's server cycling (CLAUDE.md)
- [ ] GPU env once: create serving venv, install `requirements-gpu.txt`, PIN
      exact versions into that file
- [ ] Servers: `scripts/serve_retrieval.sh` (first start builds the 21M-line
      offsets file — slow once) · `bash scripts/serve_executor.sh` (2 GPUs)
- [ ] **Smoke (pending from I2):** 20-task batch in each harness mode:
      `.venv/bin/python -m collect.run_collection --task-file data/hotpotqa_train_300.jsonl
       --limit 20 --arm a1 --mode {none|enforce|forced_continuation} --g 1 --out <scratch>`
      → schema-valid, drafts present, sane retrievals

E-a Pilot → constants:
- [ ] `--limit 50 --arm a1 --mode forced_continuation --g 4 --budget large`
      → pilot memo (quality-vs-steps curves; propose budgets + λ; check T_max=10)
- [ ] **Brian approves memo** → freeze `episode.budgets` + `economy.lambda` in
      foundation.yaml (one commit; PROVISIONAL markers removed)

E-b Judge calibration (GATE — no RL before it passes):
- [ ] **Brian provides judge endpoint URL + model name** → `judge:` in config
- [ ] `reward.calibration.make_labeling_sheet(pilot.jsonl, sheet.csv)`
- [ ] **Brian labels 50 rows (~1h)** → `agreement(sheet, judge, cfg)` →
      mean ≥ 0.80, floor 0.70; log confusion table; else fix prompt, bump
      rubric version, repeat

E-c Baselines: `bash scripts/f4_baselines.sh` → baseline_rows.csv +
      10-line summary in experiments/reports/ · hand-skim 10 trajectories/arm

E-d Micro-run (GATE): wire verl glue in `train/grpo_runner.py:_run_real`
      against the pinned verl; 10 tasks, G=4, ~20 steps, real judge →
      rewards non-degenerate, KL sane, dashboard populates

E-e Full GRPO: 300 tasks, 1 seed, 4–8 GPUs; watch divergence + steps
      distribution + entropy → checkpoint + run stamp

E-f Eval: A3 dev-200 × 3 budgets, harness-off AND harness-on (temp 0) +
      oracle forced-continuation replay → `eval/build_rows.py` →
      sanity checks → `eval/gate_check.py --rows …` prints GO/NO-GO

E-g Report: `analysis/figures.py` + `analysis/report.py` (+ optional
      `diagnostic_rubric`) → fill TODO prose → Brian reviews → update memory ·
      tag `foundation-run-1`
