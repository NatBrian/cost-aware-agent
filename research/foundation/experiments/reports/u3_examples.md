# U3 — what the behaviour actually looks like — 2026-08-05

A behavioural claim should show the behaviour. Matched control/treatment pairs on
identical questions, at the gate budget.

## Counts

| dataset | WIN (quit, nothing lost) | COST (quit, answer lost) | identical | **win:cost** |
|---|---|---|---|---|
| MuSiQue s123 | **44** | **3** (0.5%) | 282 | **14.7 : 1** |
| HotpotQA | **12** | **6** (1.0%) | 312 | 2 : 1 |

About half of all episodes are byte-identical between arms — the policy is not
uniformly hastier, it intervenes selectively. **On MuSiQue the trade is far more
favourable (14.7:1) than on HotpotQA (2:1)**, consistent with the effect being
larger where there is more doomed work to cut.

## WIN — the behaviour we claim

> **Q:** *Who is daughter to the third Rebbe of Chabad Lubavitch, named after his father?*

```
control    10 steps, F1=0.00
  1. third Rebbe of Chabad Lubavitch
  2. children of Menachem Mendel Schneersohn third Rebbe
  3. daughters of Menachem Mendel Schneersohn third Rebbe
  ... (7 more)
  -> (no answer at all)

treatment   3 steps, F1=0.00
  1. third Rebbe of Chabad Lubavitch name
  2. Menachem Mendel Schneersohn daughter named after him
  -> "Nechama Dina Schneersohn"
```

The control burned all 10 steps and produced **nothing**. The treatment reached
the same dead end in 3, committed to its best guess, and stopped. **Seven steps
saved, zero quality lost** — neither answer was right.

## COST — the honest downside

> **Q:** *Who's married to the man who produced a documentary about the singer of
> "She's Out of My Life"?*

```
control     4 steps, F1=1.00
  1. "She's Out of My Life" singer
  2. Michael Jackson documentary producer
  3. David Gest wife
  -> "Liza Minnelli"          CORRECT

treatment   3 steps, F1=0.00
  1. "She's Out of My Life" singer
  2. Michael Jackson documentary producer
  -> "David Gest is married to Lisa Marie Presley."   WRONG
```

**This is what abandonment costs.** The treatment had done the hard work — it
identified the producer correctly — and then quit one step before the lookup that
would have resolved the spouse. The control took that step and got it right.

It is rare (3/600 on MuSiQue, 6/600 on HotpotQA) but it is real, and any honest
write-up shows this case next to the win.

## REALLOC — zero clean cases, which moderates an earlier claim

**No episodes matched "treatment spent ≥2 more steps AND answered where the
control failed."** Zero, on both datasets.

That matters. U1 and T5 reported the treatment spending *more* on successful work
(+0.114 steps, +3420 tokens, both CI-excluding-zero on seed 123) and I described
this as **reallocation** — budget moved from hopeless questions to answerable ones.

The aggregate signal is real, but it does **not** correspond to visible
"spend-more-and-win" episodes. It is a diffuse shift of a fraction of a step
across many episodes, not a behaviour you could point at in a trajectory.

**Corrected wording:** the treatment spends marginally more on work the control
also succeeded at. Calling that "reallocating budget to questions it can answer"
implies a purposiveness these traces do not support.

## Reading these fairly

- Cherry-picking risk is real: these are illustrative, not evidence. The evidence
  is the aggregate, and the *counts* above are the honest summary of how often
  each category occurs.
- The WIN example's treatment answer was also wrong (F1=0). The claim is that it
  was wrong **cheaply**, not that it was better.
- ~50% of episodes are identical, so the policy change is narrow. That is a
  feature — it is not trading quality broadly for speed.

Artifacts: `scripts/u3_examples.py`.
