# T1 — where does the F1 gain come from? — 2026-08-02

Step 1 passed its gate, but F1 rose by **+0.080** — an effect that was not
predicted. Two explanations fit equally well:

- **(A) cost-awareness** — the policy abandons doomed work, and the economic
  reward incidentally improves how it spends the steps it keeps
- **(B) regulariser** — λ is simply a better-conditioned objective and the policy
  got better at everything; cost-awareness is then the wrong story even though the
  gate passed

Run on the 9 eval files already on disk, at the gate budget B=2, n=600.

## The answer: the quality gain is INDEPENDENT of the saving

### 1. The gain lives where the treatment spent the *same* steps

| what the treatment did | n | ΔF1 | Δsteps |
|---|---|---|---|
| spent **fewer** steps | 92 | +0.015 [−0.070, +0.097] — n.s. | −2.033 |
| spent the **same** | **472** | **+0.102 [+0.074, +0.130]** ✱ | 0.000 |
| spent **more** | 36 | −0.042 [−0.214, +0.126] — n.s. | +2.417 |

**The F1 gain is significant only where the step count did not change**, and is
*not* significant on the episodes the treatment actually shortened.

### 2. The two effects barely correlate

`corr(Δsteps, ΔF1) = −0.119`, 95% CI [−0.219, −0.018]. Statistically distinguishable
from zero, but far too small to be one phenomenon.

### 3. The gain survives on unshortened work

F1 gain restricted to episodes the treatment did **not** shorten:
**+0.092 [+0.064, +0.121]** — significant.

### 4. It is not better research

| | control | treatment | Δ |
|---|---|---|---|
| distinct titles retrieved | 3.068 | 2.828 | **−0.240** ✱ |
| malformed steps/episode | 0.243 | 0.212 | −0.032 n.s. |
| hit t_max | 0.030 | 0.025 | −0.005 n.s. |
| chars emitted/step | 440.3 | 424.9 | −15.4 n.s. |

The treatment retrieves **fewer** distinct documents and is not measurably
better-formatted. The quality gain is not explained by more research or cleaner
output.

### 5. The Pareto cell is small

Better **and** cheaper on **18 of 600 episodes (3.0%)**. Better on 149, worse on
69, tied on 382.

## What this means for the claim

**There are two separate effects, not one.**

1. **A real but modest cost-awareness effect** — −0.167 steps, concentrated on
   doomed work (H2: −0.486 on failed vs −0.031 on succeeded). This is what the
   gate tested, and it stands: Δsteps is essentially independent of ΔF1, so the
   stopping result is not an artefact of the quality result.
2. **A larger, unexplained quality effect** — +0.092 F1 on episodes whose step
   count never changed. This has nothing to do with stopping.

**The paper must not claim that cost-aware training improves answer quality.** The
quality gain is real and significant, but it is independent of the stopping
behaviour and its mechanism is unknown. Reporting them as one result would be
wrong.

This also deflates the "quality guard passed with room to spare" reading in the
S5 verdict. The guard passed — but not because cost-awareness improved quality.

## Consequence for T2

**SimpleQA is now the decisive experiment, not a formality.** It is single-hop:
no discretionary work exists, so a cost-awareness effect *cannot* appear there.

- if **ΔF1 > 0 on SimpleQA** → λ is a general regulariser, effect (2) confirmed,
  and it must be reported as a confound rather than a benefit
- if **ΔF1 ≈ 0 on SimpleQA** → the quality gain is specific to multi-hop work and
  something more interesting is happening

Either way the stopping claim survives on its own evidence — but what we say
*about quality* depends on the answer.

Artifacts: `scripts/t1_diagnose_f1.py`. No GPU used; run on existing eval files.
