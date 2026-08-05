# U1 — the saving in tokens, the unit the paper is actually about — 2026-08-05

**In relative terms the token saving is 2.4–4.3× the step saving.** The headline
moves from "~6% fewer steps" to **13–33% fewer tokens** — and tokens, not steps,
are what a cost-aware claim is about.

## Why steps understate it

A step is not a billable unit, and steps are not interchangeable. Every step
re-reads the whole conversation, so on this harness **step 10 costs ~9.7× step 1**
(measured 2026-07-31). The policy abandons *doomed* episodes, and doomed episodes
are the long ones — so the steps it cuts are the expensive tail.

## Result

| dataset | Δsteps | rel. | Δtokens | rel. | **ratio** |
|---|---|---|---|---|---|
| HotpotQA (Step 1) | −0.167 ✱ | −5.6% | −764 | **−13.4%** | 2.40 |
| SimpleQA (OOD) | −0.228 ✱ | −7.6% | −2135 ✱ | **−32.9%** | 4.34 |
| MuSiQue seed 42 (r1) | −0.242 ✱ | −6.8% | −2509 | **−20.0%** | 2.97 |
| MuSiQue seed 123 (r3) | −0.292 ✱ | −8.0% | −2847 ✱ | **−22.4%** | 2.79 |

Ratio > 1 means the tokens saved outrun the steps saved.

## The mechanism is visible: it is avoided context re-reading

| dataset | Δprompt | Δcompletion | **prompt share** |
|---|---|---|---|
| HotpotQA | −699 | −65 | **91.4%** |
| SimpleQA | −1942 | −192 | **91.0%** |
| MuSiQue s42 | −2323 | −186 | **92.6%** |
| MuSiQue s123 | −2589 | −257 | **91.0%** |

**~91% of the saving is prompt tokens** — context the agent no longer has to
re-read. That is exactly what skipping a late step buys, and it is why the
relative token effect is several times the relative step effect. The agent is not
being terser (completion tokens barely move); it is having fewer conversations to
re-read.

## Honest caveat: tokens are a NOISIER estimand than steps

| dataset | Δtokens 95% CI | excludes zero? |
|---|---|---|
| HotpotQA | [−2198, +661] | **no** |
| SimpleQA | [−3878, −473] | yes |
| MuSiQue s42 | [−5055, +13] | **no** (barely) |
| MuSiQue s123 | [−5454, −235] | yes |

**Only 2 of 4 token comparisons reach significance, against 4 of 4 on steps.**
Token counts are sums of variable-length quantities, so their variance is much
higher and the same underlying effect is harder to resolve — the same problem that
made the `W` estimand unusable at S2 (n≈2289 required).

**So the two measures play different roles and both belong in the paper:**

- **Δsteps is the better-powered estimand** and is what the pre-registered gate
  was decided on. It stays the primary result.
- **Δtokens is the more meaningful unit** and shows the effect is ~3× larger than
  the step figure implies — but it is reported with its wider CIs, and the
  per-dataset token numbers must not be quoted as individually significant where
  they are not.

Pooling across seeds and datasets (U2/U5) will tighten these; the two-seed MuSiQue
pool is the natural place to state a significant token figure if one exists.

## Selectivity holds in token terms — and shows the reallocation clearly

| dataset | doomed work | successful work |
|---|---|---|
| HotpotQA | −3567 | +427 |
| SimpleQA | −4551 ✱ | +569 |
| MuSiQue s42 | −5811 ✱ | +1367 |
| MuSiQue s123 | **−7906 ✱** | **+3420 ✱** |

The pattern is the same as in steps but starker: **thousands of tokens cut from
doomed episodes, with some spent back on answerable ones.** On the fully-trained
MuSiQue seed both sides are individually significant — the clearest evidence yet
that this is *reallocation of budget*, not indiscriminate economising.

## Caveats

- **Tokens processed, as reported by the server.** With prefix caching a provider
  may bill re-read context at a discount, which would shrink the absolute dollar
  figure. It would not change the paired difference: both arms ran under the
  identical serving regime.
- No dollar conversion is given. That needs a price map and a stated serving
  regime; tokens are the honest unit until then.
- Seed 789 is outstanding.

Artifacts: `scripts/u1_token_cost.py`. No GPU used — this was recoverable only
because per-step token accounting was added on 2026-07-31; the entire
FOUNDATION-1 run recorded characters instead and cannot be re-analysed this way.
