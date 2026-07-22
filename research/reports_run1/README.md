# CASSI Experiment Reports — plain-language index

One markdown per experiment/phase, written so a reader with no context can follow.
Each report answers: What did we run? What came out? What does it mean? What happens next?

| # | Report | Phase (runbook §16) | Status |
|---|---|---|---|
| 00 | [Smoke test + wallet pilot](00_smoke_pilot.md) | P0 done-criterion + P2 pilot | ✅ PASSED (2026-07-22) |
| 01 | [Collection round 0](01_collection.md) | P2 | ✅ DONE (9,600 trajectories, 2026-07-22) |
| 02 | [Stopping labels](02_labels.md) | P3 | ✅ DONE (dial 5.0→1.0 monotone, 2026-07-22) |
| 03 | Reward model (stopper) v0 + gate | P4 | ⏸️ PAUSED by user 2026-07-22 — SFT killed pre-checkpoint; **prompted-35B side already measured** (regret 5.60, ~4 steps late; see `../stopper/round0/rmp_heldout_qa_lam1.json`); resume recipe in HANDOFF §1b |
| 04 | Kill-switch K1/K2 — GO/NO-GO | P5 / §12 | pending (needs trained stopper + 4 GPUs + K2 stub implemented) |

Environment context (who ran what, on which GPUs, the serving gotchas):
see `HANDOFF.md` machine note (2026-07-21).
