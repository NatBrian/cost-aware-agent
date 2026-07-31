# Pre-redesign diagnostics — evidence for Brian's four proposals — 2026-07-31

Run on **existing rollouts** (λ=0 arm, round 1: 2400 episodes, 300 questions ×
8 attempts) plus the committed dev-200 evaluation. No new compute, no new GPU
time, and **dev-200 was not re-read** — the surviving row CSV was reused.

Purpose: test each proposed change against the question that actually killed the
last experiment — **does it give the agent something it controls?** — before
committing weeks to any of them.

---

## The headline finding

**We required an effect larger than the effect that could physically exist.**

| quantity | value |
|---|---|
| step headroom that exists on this dataset (D1) | **0.313 steps** |
| our pre-registered detection threshold | **0.500 steps** |
| effect we actually observed | 0.040 steps |

The ablation asked "can the cost term save ≥0.5 steps?" on a dataset where
perfect play saves **0.31**. **The test was unpassable by construction**, and no
reward design, λ value, or model could have passed it. That is a flaw in the
experiment, not only in the method — and it is the strongest possible argument
for Brian's proposal #2 (change the dataset).

It also means the honest reading of the ablation shifts slightly: the λ term
captured 13% of a very small available headroom. "Not effective" remains correct;
"the reward is incapable in principle" is **not** supported, because the ceiling
was never more than 0.31 steps.

---

## D1 — Step slack at equal quality *(the decisive one)*

For each question, among the 8 attempts that reached the group's **best** F1, how
many steps did the cheapest use versus the average? That gap is headroom a reward
could in principle find, with difficulty held constant (same question).

| metric | value |
|---|---|
| groups analysed | 271 (+29 excluded: no attempt scored above zero) |
| **mean slack** | **0.313 steps** |
| **median slack** | **0.000 steps** |
| groups with any slack | 112/271 = **41%** |
| groups with ≥1 step of slack | 39/271 = **14%** |

**More than half the questions have literally no slack.** On those, every attempt
that got the best answer needed the same number of steps. There is no "stop
sooner" behaviour available at any price.

This is the quantitative form of the confound found earlier: steps are mostly
dictated by the question, so on most questions there is nothing to optimise.

**Verdict on HotpotQA: it cannot support a cost-aware-stopping claim.** Not
because the method fails, but because the phenomenon is absent from the data.

---

## D2 — Verbosity slack *(tokens: a dimension the agent DOES control)*

| metric | value |
|---|---|
| mean characters per step | 454 (cap ≈ 2000 chars / 512 tokens) |
| SD across episodes | 248 |
| **at equal best quality, the cheapest attempt uses** | **28% fewer characters** |

**This is the strongest controllable signal in the data.** Two attempts at the
same question reaching the same F1 differ by ~28% in output length. Unlike step
count, that gap is not explained by difficulty — the question is held fixed and
the outcome is identical. It is pure verbosity.

Corroborating evidence from earlier in the run: malformed steps were caused by
the model writing ~2000-character thoughts that hit the 512-token cap. It is
visibly wasteful in a dimension it fully controls.

**Verdict: Brian's proposal #1 is right, but only for the token component.**
Tool calls and steps carry the same difficulty confound. Tokens do not.

---

## D3 — Redundant search *(controllable waste)*

| metric | value |
|---|---|
| exact duplicate queries | 134/5330 = 2.5% |
| near-duplicate (Jaccard ≥ 0.8) | 228/5330 = 4.3% |
| **combined wasted search** | **6.8% of all searches** |
| episodes containing a repeat | 120/2400 = 5.0% |

Real, fully controllable, but **small**. Eliminating every redundant search saves
~0.23 steps per episode — the same order as the total step headroom, and below
what n=50 can resolve.

**Verdict: worth rewarding, too small to be a headline.** Supports pointing the
rubric at controllable behaviours (proposal #3) but not as a primary claim.

---

## D4 — Is the 9B executor capable of responding to a budget?

| arm | B=2 | B=4 | B=8 | range |
|---|---|---|---|---|
| A0 (never told a budget) | 3.92 | 3.92 | 3.92 | 0.00 |
| A1 (told, same model) | 2.96 | 3.54 | 3.98 | **1.02** |

Merely *telling* the 9B model its budget moves it **1.02 steps** — more than
three times the total slack available at equal quality, and 25× the λ effect.

**The model reads the signal and acts on it.** Capability is not the binding
constraint at this scale.

**Verdict: proposal #4 (bigger executor) is NOT the first thing to change.** I
previously ranked it second on the strength of the earlier harness result
(Sonnet −29%, weak model no effect); this data contradicts that ranking for
*this* setup. A bigger model may still help on long-horizon tasks where the
decisions are harder — but on HotpotQA it would be solving a problem we do not
have.

---

## What this does to the proposed priority order

| | before diagnostics | after |
|---|---|---|
| Dataset → long-horizon | 1st | **1st — now proven necessary, not just suspected** |
| Bigger executor | 2nd | **4th — the 9B already responds (1.02 steps)** |
| Multi-dimensional budget | 3rd | **2nd — but tokens only; 28% headroom, difficulty-free** |
| Rubric → controllable | 4th | **3rd — real (6.8% waste) but small** |

**Revised recommendation:**

1. **Change the dataset.** D1 proves HotpotQA has 0.31 steps of headroom with a
   median of zero. Any stopping claim needs tasks with genuine discretionary
   work — GAIA, BrowseComp, or long-horizon synthetic. This is no longer a
   judgement call; the ceiling is measured.
2. **Add token cost, kept as its own dimension.** D2 shows 28% verbosity
   headroom at equal quality — the one place the agent has real, unconfounded
   freedom. Do *not* collapse tokens, tool calls and steps into a single dollar
   figure: two of the three are confounded, and merging them would hide the one
   clean signal underneath them.
3. **Repoint the rubric at controllable behaviours** (redundancy, verbosity,
   query quality) rather than "was stopping right". D3 gives 6.8% waste to aim
   at, and our own calibration showed the stop-decision bit was both the hardest
   to judge (.775, the only bit to fail) and the least actionable.
4. **Defer the bigger executor** until the tasks are long-horizon. Then retest —
   capability plausibly binds there even though it does not here.

**And set the detection threshold from measured headroom, not intuition.** The
0.5-step threshold was chosen as "~14% of a 3.6-step baseline" and turned out to
exceed the achievable maximum. On any new dataset, run D1 *first* and set the
threshold below the measured ceiling.

---

## Honest limitations of these diagnostics

- All from the **λ=0 arm, round 1**. A more-trained policy might show different
  slack; the direction is unlikely to reverse but the magnitudes could move.
- D1 defines headroom as "cheapest attempt reaching the group's best F1". If the
  best F1 is itself luck on a hard question, the cheapest such attempt may not be
  reliably reproducible — this may **overstate** achievable slack, which only
  strengthens the conclusion.
- D3's near-duplicate threshold (Jaccard ≥ 0.8) is a judgement call; a looser
  threshold would report more waste.
- D2 measures characters, not tokens. The ratio is roughly 4:1 here but varies.
- None of this touches dev-200. **1 of 3 looks remains unused.**
