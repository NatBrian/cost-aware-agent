# CASSI — Cost-Aware Stopping Supervision, Internalized

Implementation of `research/paper_plan_v2.md` (the source of truth — read it first).
This folder is the `cassi/` repo layout defined in paper_plan_v2 §17.

**New here (human or AI)? Read [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) first** — the full
plain-language explanation: research idea, glossary (paper terms → code symbols),
pipeline walkthrough, document map, invariants.
**Status / what to run next: see [`HANDOFF.md`](HANDOFF.md).**

## What this is (plain language)

We train a small "coach" model (Qwen3.5-2B) to judge, at every step of an agent's
work, whether the *next* step is worth its dollar cost. The coach is trained on
hindsight-optimal stop labels (Snell envelope, Algorithm 1). Its value estimate is
then used as a potential-based process reward to RL-train the "worker" agent
(Qwen3.5-9B) — so the worker *internalizes* when to stop, instead of being cut off
by an external monitor.

## Layout (paper_plan_v2 §17)

```
cassi/
├── common/        # trajectory schema (§11), config loader
├── labels/        # snell.py (Algorithm 1), drafts.py (running-draft parse/stability), quality.py (F1/EM/subgoals)
├── stopper/       # features.py (§18.1 serialization), model.py (3 heads), train_sft.py, eval_regret.py
├── executor/      # react_agent.py, collect.py (forced-continuation), shaping.py (PBRS + step advantages), train_grpo.py, envs/
├── budget/        # cost.py — wraps the repo harness price map; tiers, wallets, pilot normalization
├── baselines/     # b2_probe.py … b9_direct_shaping.py (one per §5.2 row)
├── eval/          # metrics.py (regret, Pareto, matched-risk), stats.py (§5.6), overhead.py
├── analysis/      # figures/ (F1–F6), tables/ (T1–T5) — one script each, CSV in → PDF/tex out
├── paper/         # main.tex, sections/, references.bib
├── configs/       # cassi.yaml — single source of truth (§17)
├── scripts/       # phase runners P0–P9 incl. kill-switches K1/K2
└── tests/         # CPU-only tests on synthetic trajectories (run: pytest tests/ from this dir)
```

## Quick start (CPU-only checks, no GPU needed)

The dedicated venv ALREADY EXISTS with the pinned stack installed (P0 is done) —
do NOT pip-install into it or into the system python; reinstalling can silently
flip which `verl` package wins name resolution.

```bash
cd research/cassi
source .venv/bin/activate
python -m pytest tests/ -q        # expect 115 passed (canonical count: PROJECT_GUIDE §7)
```

## GPU phases

Training/collection needs GPUs. On this machine, acquire/release with:

```bash
eval $(/mnt/src/zhanka/gpu_acquire.sh 2)   # collection / stopper SFT / eval serving
eval $(/mnt/src/zhanka/gpu_acquire.sh 4)   # executor GRPO (4–8)
# ... run ...
/mnt/src/zhanka/gpu_release.sh
```
Acquire the SMALLEST N the phase needs (the machine is shared). Note: acquired locks do
not guarantee freed memory — verify with nvidia-smi (see HANDOFF; never kill foreign jobs).

Phase order and done-criteria: paper_plan_v2 §16 (P0–P11). Kill-switches K1/K2
(§12) run BEFORE the full pipeline — `scripts/p5_killswitch.sh`.
