# CORRECTED RESULTS — supersedes T5_SYNTHESIS.md and foundation_v2_2_COMPLETE_REPORT.md

<!-- provenance -->
> **Run:** FOUNDATION-2 · **Plan:** `research/paper_plan_v2_2_foundation.md`
> **Lineage:** full paper plan `paper_plan_v2_1.md` → FOUNDATION-1
> (`paper_plan_v2_1_foundation.md`, completed 2026-07-29, returned a null) →
> **FOUNDATION-2** (the redesign, completed and closed 2026-08-11).
> **Status of this run:** closed. Current findings are in
> `foundation_v2_2_CORRECTED_RESULTS.md`.

**2026-08-06.** Four independent audits found that the previously reported
headline was substantially inflated and that three interpretive claims were
artefacts. Every number below was recomputed from the raw episode files.

**Read this instead of the two documents above.** They are retained unedited as a
record of what was claimed, but their headline numbers are wrong.

---

## 1. What the result actually is

The effect splits into two parts that were previously reported as one.

| dataset | **total Δsteps** | of which crashes | **well-formed Δsteps** |
|---|---|---|---|
| HotpotQA | −0.167 [−0.280, −0.057] ✱ | 37% | **−0.112 [−0.151, −0.075]** ✱ |
| SimpleQA | −0.228 [−0.362, −0.098] ✱ | 70% | **−0.072 [−0.111, −0.041]** ✱ |
| MuSiQue s42 | −0.242 [−0.435, −0.050] ✱ | 90% | −0.031 [−0.081, +0.017] |
| MuSiQue s123 | −0.292 [−0.490, −0.090] ✱ | 76% | **−0.087 [−0.143, −0.031]** ✱ |

*"Crash" = an episode that hit the 10-step cap without ever producing a parseable
answer, or that contained a malformed step.*

**Both columns are legitimate. They answer different questions:**

- **Total** answers *"how much does deployment cost?"* — a crash costs real steps
  and real tokens. **~6–8% saving.**
- **Well-formed** answers *"did the policy learn better decisions?"* — crashes are
  parser failures, not decisions. **~1–3% saving**, significant on 3 of 4.

**The error in the previous reports was reporting the first number under a claim
that requires the second.** "It learned to abandon hopeless questions" is a claim
about decision-making; 37–90% of the number supporting it was format collapse.

## 2. Cost in tokens

| dataset | total | **well-formed** |
|---|---|---|
| HotpotQA | −13.4% | **−11.0%** ✱ |
| SimpleQA | −32.9% | **−9.7%** ✱ |
| MuSiQue s42 | −20.0% | −1.8% (n.s.) |
| MuSiQue s123 | −22.4% | **−12.9%** ✱ |

Previously reported as "13–33%". **The honest range is ~10–13%** on well-formed
work, 3 of 4 significant.

## 3. Quality — the previous claim had the wrong sign

F1 divides credit by answer length, so a shorter answer scores higher even when
both contain the same fact. The treatment answers in 4.70 words, the control in
7.68. Re-scored with a length-free metric (does the answer contain the gold
string?):

| dataset | Δ F1 (length-sensitive) | **Δ contains-gold (length-free)** |
|---|---|---|
| HotpotQA | +0.080 ✱ | **−0.052 [−0.078, −0.025]** ✱ **worse** |
| SimpleQA | +0.085 ✱ | +0.006 (n.s.) |
| MuSiQue s42 | −0.005 | +0.012 (n.s.) |
| MuSiQue s123 | −0.021 | −0.005 (n.s.) |

On HotpotQA, **96% of the reported +0.080 came from 281 episodes where both arms
already contained the gold answer** — the treatment simply said it in 4.3 words
instead of 6.9. On finding the answer it lost 49 questions and won 18.

**Corrected claim: quality is unchanged on three datasets and significantly worse
on HotpotQA.** Not "a Pareto improvement".

## 4. The selectivity claim is withdrawn entirely

Previously: *"the saving concentrates on doomed work — −0.49 vs −0.03, 16×."*

Running the identical analysis on **two arms with the same λ** (no treatment
effect possible):

| comparison | "selectivity gap" |
|---|---|
| placebo — two λ=0 controls | **+0.42** |
| placebo — two λ=0.568 arms | **+0.49** |
| "real" MuSiQue s42 | +0.56 |
| "real" MuSiQue s123 | +0.83 |

**70–85% of the signature is reproduced with no treatment at all.** It is
regression to the mean from conditioning on one arm's outcome. Partitioning by the
*treatment's* own outcome instead shows no selectivity (−0.160 vs −0.170).

No fix recovers this. The claim is withdrawn.

## 5. The pre-registered gate fails on the corrected estimand

Threshold was **−0.119**. Corrected HotpotQA is **−0.112** (and −0.089
round-matched). The gate passed on the uncorrected number, which was 37% crashes.

Also: the point estimate never cleared the threshold at 95% confidence even as
originally computed — P(true effect > −0.119) = 0.20, t-test against −0.119 gives
p = 0.41.

## 6. Other corrections

| previously claimed | corrected |
|---|---|
| "~half of episodes byte-for-byte identical" | **14–23%**. The 47–52% figure was *same step count and same F1*. Answer text differs in 55–65% |
| "14.7:1 win:cost ratio" | Category-definition artefact. Full ledger: treatment **loses more answers than it gains on all four datasets**; HotpotQA McNemar **p=0.017** |
| "the effect grows on harder data, 1.45×" | Between-dataset contrasts give **p = 0.28–0.51**. No scaling result |
| "~3× larger in tokens than steps" | Ratio CI on HotpotQA is **[−6.00, +4.09]**. Uninterpretable |
| "pooled n=2300" | **1700** — MuSiQue seeds share the same 600 questions |
| "eval-600 read once" | Analysed 5–6 times, each pass generating new hypotheses |
| MuSiQue B=3, B=4 | **Both null** (−0.052, −0.127). The headline was 1 of 3 cells |
| seed 123 F1 guard | **Breaches** the pre-registered −0.02 (at −0.021). Undisclosed |
| Bonferroni | The gate (p=0.0041) **fails** correction over the reported family (α/m=0.0015) |

## 7. Novelty — the contribution is pre-empted

- **[MASH, arXiv:2510.01152](https://arxiv.org/abs/2510.01152)** (Oct 2025) —
  "Pay-Per-Search Models are Abstention Models". Same reward construction, GRPO,
  HotpotQA + multi-hop, OOD transfer. **Our claimed mechanism is their title.**
  They report **+7.6% accuracy** on multi-hop; ours degrades accuracy.
- **[OTC-PO, arXiv:2504.14870](https://arxiv.org/abs/2504.14870)** (NeurIPS 2025)
  — tool-call cost in PPO **and GRPO**, same stack, same benchmarks: **up to
  68.3% fewer tool calls** at comparable accuracy. Ours: 5.6% steps.

Both verified directly. A 12× effect-size gap against a paper that should have
been our closest baseline.

## 8. What is defensible

> A per-step price makes the policy **terser and more robust to format collapse**.
> Total cost falls ~6–8%; roughly two-thirds of that is fewer catastrophic parser
> failures, and ~1–3% is genuinely less deliberation on well-formed episodes.
> Token cost falls ~10–13% on well-formed work. Answer quality is unchanged on
> three datasets and significantly worse on one.

That is real and measured. It is **not** cost-aware abandonment, **not** selective,
**not** iso-quality, **not** scaling, and **not** novel.

## 9. What was done well, for the record

The audits confirmed: λ threading is correct and touches nothing else; the GRPO
math is sound; pairing is well-founded and refuses duplicates; the bootstrap uses
the right resampling unit; config hashes and task sets are identical across arms;
and every published number reproduces to three decimals. **The arithmetic was
honest throughout. The estimand was not.**

Attacks that failed (the claim survives these): no stock-phrase abstention hack,
no F1 gaming by padding, no answering without searching, no config tampering, and
the control — not the treatment — is the arm with more degenerate repeated queries.

## 10. The transferable lesson

The three artefacts here are **generic to agent-efficiency research**:

1. **A step cap turns crashes into apparent deliberation.** Any `steps_used`
   metric with a hard cap has this.
2. **F1 rewards terseness.** Any efficiency method that shortens outputs will
   look like it improves quality on F1/EM.
3. **Conditioning on one arm's outcome manufactures selectivity.** We reproduced
   70–85% of our own headline mechanism with a placebo.

None were caught by pre-registration, power analysis, frozen test sets, an
out-of-distribution control, a mechanism test, or two seeds — all of which this
project had. They were caught by adversarial audit.
