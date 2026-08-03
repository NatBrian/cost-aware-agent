# T4 — MuSiQue: does the effect scale? — 2026-08-03

**Yes — it grows ~1.45×, and it scales with FAILURE RATE, not with horizon.**
The pre-registered prediction (H-fail) was correct; H-horizon is rejected.

Per the amendment committed *before* these numbers existed
(`t4_preregistration.md`), the **round-matched** comparison is primary here: the
λ=0 control failed its round-2 gate at 29.4% malformed and stopped at round 1,
so the protocol comparison would be a round-1 control against a round-3
treatment. Matching at round 1 holds training amount fixed so only λ differs.

## 1. Primary — round-matched (both arms at round 1), MuSiQue eval-600

| B | Δsteps | 95% CI | ΔF1 | 95% CI |
|---|---|---|---|---|
| **2 (gate)** | **−0.242** | **[−0.435, −0.050]** ✱ | −0.005 | [−0.027, +0.016] |
| 3 | −0.052 | [−0.217, +0.113] | +0.005 | [−0.017, +0.027] |
| 4 | −0.127 | [−0.295, +0.047] | −0.007 | [−0.033, +0.017] |

**−0.242 steps vs HotpotQA's −0.167 — a 1.45× increase.** Secondary (protocol,
r1 vs r3) gives −0.298, i.e. 1.79×; both point the same way, and the matched
figure is the one to quote because it is unconfounded by training amount.

## 2. Which mechanism? H-fail, as pre-registered

The prediction, written before the treatment arm was trained:

| | predicted between | predicted within |
|---|---|---|
| **H-fail** (failure-rate driven) | ≈ −0.24, **1.5–2×** | **peaks at 3-hop** |
| H-horizon (length driven) | ≈ −0.19, barely above | rises monotonically 2<3<4 |

Observed, round-matched at B=2:

| hops | n | Δsteps | 95% CI | control fail% |
|---|---|---|---|---|
| 2 | 311 | −0.148 | [−0.354, +0.058] | 51.7 |
| **3** | 189 | **−0.429** | **[−0.810, −0.042]** ✱ | **61.9** |
| 4 | 100 | −0.180 | [−0.800, +0.440] | 50.7 |

**The effect peaks at 3-hop and is not monotonic in hops** — it tracks the
failure rate (51.7 / 61.9 / 50.7), not step count (3.50 / 3.63 / 3.74). Between
datasets, −0.242 vs −0.167 = 1.45×, close to H-fail's 1.5–2× and above
H-horizon's "barely above". Both tests agree.

**H-fail confirmed, H-horizon rejected.** This was recorded in advance precisely
so the outcome could not be read as confirmation either way; last time a
pre-registered directional prediction (the Step-1 dose-response) came out wrong
and was reported as such.

**Caveat on the within-test:** only the 3-hop cell reaches significance; 2-hop and
4-hop CIs contain zero, and 4-hop has n=100 with a very wide CI. The *pattern*
matches H-fail, but the within-dataset evidence rests largely on one cell.

## 3. H2 selectivity — holds, third dataset

| partition | n | Δsteps | 95% CI |
|---|---|---|---|
| control **FAILED** | 324 | **−0.500** | [−0.849, −0.164] ✱ |
| control **SUCCEEDED** | 276 | +0.062 | [−0.043, +0.181] |

The saving lands on doomed work and leaves successful work alone — now replicated
on HotpotQA (−0.486 / −0.031), SimpleQA (−0.420 / −0.013) and MuSiQue
(−0.500 / +0.062). **The per-doomed-episode saving is remarkably stable at ~−0.45
to −0.50 across all three**, which is exactly why the overall effect scales with
how much doomed work a dataset contains.

## 4. The regularisation confound does NOT reproduce here

| dataset | ΔF1 on episodes whose steps did not fall |
|---|---|
| HotpotQA | +0.092 ✱ |
| SimpleQA | +0.086 ✱ |
| **MuSiQue** | **−0.005 (CI contains zero)** |

This weakens the T2 conclusion and I am correcting it rather than leaving it
overstated. **"λ is a general regulariser" was too strong.** Two readings:

- **(a)** The quality gain belongs to *that specific pair of HotpotQA-trained
  policies*, and transferred across eval datasets because the policies were the
  same. T2 varied the eval set but not the training run, so it could not have
  distinguished this.
- **(b)** The MuSiQue arms are both round-1 policies, so there may simply have
  been too little training for a regularisation effect to appear.

These are not separable with what we have. The honest statement is: **the quality
gain is a confound that CAN appear and must always be controlled for, not one that
always appears.** Either way it is not evidence that cost-awareness improves
quality.

## 5. λ stabilises training — replicated, and stronger here

| round | ctrl (λ=0) | trt (λ=0.568) |
|---|---|---|
| 1 | 4.5% | **1.6%** |
| 2 | **29.4% — FAIL** | 5.1% |
| 3 | — | **0.8% — PASS** |

The λ=0 control collapsed one round earlier than on HotpotQA and to a worse rate
(29.4% vs 20.5%), while the priced arm ended *healthier than it started* (0.8%
malformed, F1 0.651 on the probe). As it degraded, the control's steps *rose* to
4.08: without cost pressure, training on a harder dataset degenerates toward
longer, malformed episodes.

Unexplained, and now the finding I have least account for. It is also the reason
the primary comparison had to be round-matched.

## 6. What this changes for the project

**The scale-up target should be high-failure, not long-horizon.** H-fail means the
effect pays off wherever the agent is frequently stuck — so the natural next
benchmark is one with a high failure rate, and the plan's BrowseComp/deep-research
direction is justified by *difficulty*, not by episode length. That is a concrete
correction to `paper_plan_v2_2_foundation.md` §10.

**The headline configuration is now MuSiQue**, not HotpotQA: larger effect
(−0.242 vs −0.167), no quality confound, selectivity intact. T3's seeds should be
spent here.

## Limitations

- **The primary comparison is two round-1 policies.** Lightly trained on both
  sides. Unconfounded, but not a fully-trained comparison, and a null in the 2-hop
  and 4-hop cells is weaker evidence of absence than it would be at round 3.
- **One seed** (T3 addresses this).
- **The health probe ran on HotpotQA val-50** even for MuSiQue arms — valid as a
  format/behaviour gate, but its F1 readings are out-of-domain and must not be
  read as MuSiQue performance.
- Absolute MuSiQue F1 is low (~0.27) against a 2018 Wikipedia index; both arms
  face this equally so the paired comparison is unaffected.
- Sizes were matched to HotpotQA (300 train / 600 eval) by design, so a bigger
  effect cannot be attributed to more data.

Artifacts: `experiments/results/t4_musique/` (9 JSONLs + `t4_verdict.txt` +
`arm_provenance.json`) · scripts `t4_build_musique.py`, `t4_musique_run.sh`,
`t4_analyse.py` · pre-registration `t4_preregistration.md`.
