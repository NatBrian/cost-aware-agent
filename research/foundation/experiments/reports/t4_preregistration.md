# T4 — pre-registered expectations, written before the treatment arm exists

**Committed 2026-08-02, while the MuSiQue *control* arm is at round 1 and the
treatment arm has not been trained at all.** Everything below is derived from
control-only data and from the Step 1 / T2 results already reported.

## Baseline facts (MuSiQue control, round 1, n=2400)

| | HotpotQA | MuSiQue |
|---|---|---|
| mean steps | 3.09 | **3.54** (+0.46) |
| F1 | 0.515 | 0.259 |
| failure rate | 24.9% | **53.9%** |

Within MuSiQue, **steps rise monotonically with required hops** while **failure
rate does not**:

| hops | n | steps | fail% |
|---|---|---|---|
| 2 | 1736 | 3.50 | 51.7 |
| 3 | 528 | 3.63 | **61.9** |
| 4 | 136 | **3.74** | 50.7 |

That decoupling is useful: horizon and difficulty are not the same variable here,
so the two candidate mechanisms make *different* predictions.

## Two mechanisms, two predictions

The abandonment effect concentrates on doomed work (Step 1: −0.486 on failed vs
−0.031 on succeeded; SimpleQA: −0.420 vs −0.013). So the overall Δsteps should be
roughly *failure-rate × per-doomed-saving*.

**(H-fail) The effect is driven by how much doomed work exists.**
MuSiQue has 2.2× HotpotQA's failure rate, so predict
`Δsteps ≈ 0.539 × (−0.45) ≈ −0.24`, i.e. **roughly 1.5–2× the HotpotQA −0.167**.
*Within* MuSiQue it should peak at **3-hop**, following the failure rate.

**(H-horizon) The effect is driven by how long the task is.**
MuSiQue's horizon is only 15% longer, so predict **≈ −0.19, barely above
HotpotQA**. *Within* MuSiQue it should rise **monotonically with hops** (2 < 3 < 4),
following step count.

These are distinguishable both between datasets and within MuSiQue, and the
within-dataset test is the stronger one: it holds dataset, policy and training
fixed and varies only the required horizon.

## Why this matters for the paper

The headline question is whether the effect **scales**. If it is **H-fail**, then
it scales with *how often the agent is stuck*, not with task length — which means
the right pitch is "cost-aware abandonment pays off wherever failure is common",
and the natural next benchmark is one with a high failure rate, not merely a long
horizon. If it is **H-horizon**, the long-horizon deep-research direction
(BrowseComp etc.) is the right scale-up.

I currently expect **H-fail**, because every selectivity result so far has been
about *doomed work* rather than about *length*. Recording that so the outcome is
not read as confirmation whichever way it falls.

## What would count as a null

Δsteps on MuSiQue with a CI containing zero, or |Δsteps| ≤ 0.167 (no growth). That
would say the effect does not scale on either account, and the ~6% saving on
HotpotQA is close to what this method delivers. That is a publishable limit, and
it should be reported as plainly as a positive.

## Caveat found mid-run: the health probe is out-of-domain

`probe_policy_health.sh` hardcodes `data/hotpotqa_val_50.jsonl`, so the MuSiQue
arms are health-checked on **HotpotQA** questions.

- **The gate itself remains valid.** It tests malformed-output rate (<10%) and
  cap-out rate (<15%), which are format and behaviour properties of the policy,
  not domain performance.
- **The probe's F1 must NOT be read as MuSiQue performance.** mqctrl round 1
  reports F1 0.502 on the probe while its own MuSiQue training rollouts score
  0.259. Those are different datasets, not a contradiction.

**Deliberately not changed mid-run.** Round 1 was already gated on HotpotQA;
switching rounds 2–3 to a MuSiQue probe would make the three rounds
non-comparable and would mean the arm's stopping criterion changed part-way
through. Consistency matters more here than domain-matching, given the gate
metrics are domain-independent. Logged as a known limitation instead.

*(Incidental observation, not a result: a MuSiQue-trained control scoring 0.502 on
HotpotQA val-50 is close to the HotpotQA-trained control's 0.500 at the same
round. n=40, so this is an anecdote.)*

## Amendment (2026-08-02, before the treatment arm finished): which comparison is primary

**The MuSiQue λ=0 control failed its round-2 health gate at 29.4% malformed** and
stopped at **round 1**. Its trajectory was 4.5% → 29.4%, against the HotpotQA
control's 3.6% → 6.7% → 20.5% (failed at round 3). Steps also *rose* to 4.08 as it
degraded: without cost pressure, training on the harder dataset degenerates faster
and toward **longer, malformed** episodes.

If the treatment survives to round 3, the protocol comparison would be
**round 1 control vs round 3 treatment** — three times the training on one side.
That is far too confounded to headline, and much worse than the 2-vs-3 asymmetry
on HotpotQA.

**Decision, recorded before the numbers exist: on MuSiQue the ROUND-MATCHED
comparison is primary, and the protocol comparison is secondary.** Matching at the
lower round holds training amount fixed so that only λ differs, which is the whole
point of having a control. This inverts the precedence used on HotpotQA — stated
here explicitly rather than chosen later, because choosing after seeing which one
looks better is exactly the freedom pre-registration exists to remove.

**Known cost of this choice:** the round-matched pair will be two lightly-trained
policies (one round each), so the effect may be smaller than a fully-trained
comparison would show, and a null there is weaker evidence of absence than it
would be at round 3. That is the price of an unconfounded comparison, and it will
be stated with the result.

## Unchanged commitments

- Gate budget B=2; paired per task; 10k bootstrap; same λ pair (0 / 0.568).
- The T2 regularisation confound is **re-tested here, not assumed absent**:
  ΔF1 on episodes whose step count did not change (HotpotQA +0.092, SimpleQA
  +0.086). A similar value means it reproduces again and stays a disclosed
  confound.
- H2 selectivity is re-tested here, not assumed to carry over.
