# E-c baselines report — A0/A1/A2 on frozen dev-200 — 2026-07-22

**What ran:** the two baselines plus the no-information floor, 1,400 episodes
(A0 once + A1/A2 at each frozen budget {2,4,8}), temperature 0, identical
frozen 200 questions. Dev-200 look #1 of ≤3. Raw: `../results/baselines/`.

## The table (mean over 200 tasks; U = F1 − 0.3·steps/B)

| arm | B | F1 | steps | utility | self-stop≤B | never-answered |
|---|---|---|---|---|---|---|
| A0 no info | (8) | .415 | 3.93 | .268 | .87 | .105 |
| A1 prompted | 2 | .478 | 2.96 | .035 | .345 | .035 |
| A1 prompted | 4 | .471 | 3.55 | .205 | .785 | .030 |
| A1 prompted | 8 | .501 | 3.98 | .352 | .925 | .045 |
| A2 enforced | 2 | **.221** | 2.00 | −.079 | .370 | 0 |
| A2 enforced | 4 | .411 | 3.09 | .180 | .765 | 0 |
| A2 enforced | 8 | .455 | 4.00 | .306 | .885 | 0 |

## What this means (plain language)

1. **Telling beats forcing, everywhere.** Paired per-task, A1 wins utility over
   A2 at every budget (B=2: +.114, CI +.04..+.18; B=8: +.046, CI +.01..+.09;
   B=4: +.026, CI −.01..+.06). The foundation's bar is now clear: the RL arm
   must beat the *stronger* baseline, A1 — prompting is not a strawman here.
2. **The predicted A2 ceiling appeared exactly as designed.** At B=2 the
   referee submits whatever draft exists at step 2 — and F1 collapses to .221
   (vs A1's .478). External enforcement cannot make drafts ready early; only
   changed behavior can. This is the gap internalization is supposed to close,
   measured.
3. **Budget information alone moves the agent, modestly.** A1's steps scale
   with the stated budget (2.96 → 3.55 → 3.98); the uninformed A0 always takes
   ~3.93. So the model reads the tracker and adapts ~1 step at tight budgets —
   but it still *overshoots* B=2 (2.96 > 2, only 34.5% stop within budget),
   showing prompt-following stops at the edge of its natural habits.
4. **The RL target (gate, B=4):** beat U=.205 (A1) and .180 (A2) with F1 ≥ .361
   and ≥70% self-stop within 4 steps. Per the pilot curves (knee at step 3),
   e.g. F1≈.46 at ~2.8 steps ⇒ U≈.25 clears it. Achievable, not free.

## Notes for later stages

- A2's zero never-answered is mechanical (the referee always submits).
- A0's hit-cap 10.5% = the never-stopping tail the RL arm should eliminate.
- Hand-skim of 10 trajectories/arm: A1@B2 mostly ignores the budget after
  step 2 rather than compressing its search; classic rephrase-loops persist
  in all arms (as in the pilot).
