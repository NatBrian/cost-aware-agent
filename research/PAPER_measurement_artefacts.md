# Three Measurement Artefacts That Break Agent-Efficiency Experiments

**A case study in which a pre-registered, adequately-powered, seed-replicated
result was still an artefact.**

Draft, 2026-08-11. All numbers re-verified from raw episode files on that date.
Supporting evidence: `foundation/experiments/reports/`, raw data
`foundation/experiments/results/`.

---

## Abstract

Efficiency claims about LLM agents are usually measured in **steps** or **tokens**
saved at matched task accuracy. We show that three properties of this standard
setup can each independently manufacture a significant, replicable, wrong result.

We demonstrate them on our own experiment. A per-step cost penalty in a GRPO
reward appeared to teach a search agent to abandon hopeless questions: **−0.22
steps** and **13–33% fewer tokens** at improved accuracy, significant on four
paired comparisons across three datasets, transferring to a dataset never trained
on, replicating across two seeds, against a threshold fixed in code before the
data existed.

Every interpretive part of that was wrong.

- **37–90% of the step saving** was episodes in which the *control* never emitted
  a parseable answer and ran to the step cap. Excluding them, two of four
  comparisons reverse sign significantly.
- **The accuracy gain was a verbosity effect.** F1 divides credit by answer
  length; the treated policy answered in 4.70 words versus 7.68. On a
  length-insensitive metric accuracy is **−0.052** on the gate dataset — the
  opposite sign. 96% of the reported gain came from episodes where both arms
  already contained the gold answer.
- **The "selectivity" mechanism reproduces without any treatment.** Running the
  identical analysis between two policies trained with *identical* λ yields
  70–85% of the reported signature.

None of these were caught by pre-registration, a power analysis, a frozen test
set, an out-of-distribution control, a mechanism test, or seed replication — all
of which this experiment had. They were caught by adversarial audit.

We name each artefact, give a diagnostic that detects it, and give the correction.

---

## 1. Why this matters

Agent-efficiency research shares a measurement template: run two policies on the
same tasks, report the difference in steps or tokens, and check that accuracy did
not fall. The template looks sound. Our experiment followed it carefully and
produced a result that survived six independent robustness checks before failing
to a seventh.

The three artefacts below are not properties of our method. They are properties of
**the measurement template**, and they apply to any paper using it.

---

## 2. Setup

| | |
|---|---|
| Agent | ReAct loop: think → search local Wikipedia → read → repeat. Emitting an answer ends the episode |
| Model | Qwen3.5-9B, trained with GRPO, 3 rounds, G=8 |
| Arms | **Control** `λ=0` (steps free) vs **Treatment** `λ=0.568` (each step costs). Identical base model, data, seed, rounds, budgets. **λ is the only difference** |
| Step cap | `t_max = 10`. An episode that never emits a parseable answer runs to the cap |
| Evaluation | 600 held-out questions (500 for SimpleQA), temperature 0, paired per question, 10k bootstrap |
| Datasets | HotpotQA (in-distribution), SimpleQA (never trained on), MuSiQue (harder, 2 seeds) |

An audit confirmed the implementation is correct: λ reaches only the terminal
reward, the GRPO advantage math is sound, pairing refuses duplicate task IDs, the
bootstrap resamples questions (the right unit), config hashes and task sets are
identical across arms, and every reported number reproduces to three decimals.
**The arithmetic was never the problem.**

---

## 3. Artefact 1 — A step cap turns crashes into apparent deliberation

### The mechanism

The agent stops early only by emitting a correctly-formatted answer. If it emits
something the parser rejects, it keeps going until `t_max`. A crashed episode
therefore records `steps_used = 10` — indistinguishable, in the metric, from ten
steps of deliberation.

Crashes are rare but enormous: ~3 steps and ~1,000 tokens for a typical episode,
**10 steps and ~12,000 tokens** for a crashed one.

### What it did to our result

| dataset | reported Δsteps | **share from crashes** | Δsteps excluding crashes |
|---|---|---|---|
| HotpotQA | −0.167 [−0.28, −0.06] | **37%** | −0.112 [−0.15, −0.07] |
| SimpleQA | −0.228 [−0.36, −0.10] | **70%** | −0.072 [−0.11, −0.04] |
| MuSiQue-42 | −0.242 [−0.43, −0.05] | **90%** | −0.031 [−0.08, +0.02] *(null)* |
| MuSiQue-123 | −0.292 [−0.49, −0.09] | **76%** | −0.087 [−0.14, −0.03] |

Excluding *only* episodes where the control never terminated, and keeping
everything else, both MuSiQue comparisons **reverse sign with intervals excluding
zero** (+0.298 and +0.251): the treated policy uses *more* steps on episodes that
actually ran.

The **median Δsteps is exactly 0.0 on every dataset at every budget.** The typical
question was unaffected; the mean was a tail statistic driven by ~12 of 600
episodes.

### Diagnostic

Report the **median and a trimmed mean** beside the mean, and the **share of the
effect contributed by cap-touching episodes**. If the median is zero, say so.

### Correction

Decide what you are claiming, and measure that:

- *"deployment costs less"* → include crashes; they are real cost
- *"the policy learned better decisions"* → exclude them; a crash is not a decision

Our error was reporting the first number under the second claim.

---

## 4. Artefact 2 — F1 and EM reward terseness

### The mechanism

Token-overlap metrics divide credit by answer length. A shorter answer containing
the same information scores **higher**. Any intervention that shortens outputs —
which is what most efficiency methods do — gets an accuracy bonus for free.

In our data, `corr(answer length, F1) = −0.372`.

### What it did to our result

The treated policy answered in **4.70 words**; the control in **7.68**.

| dataset | ΔF1 *(length-sensitive)* | **Δ contains-gold** *(length-free)* |
|---|---|---|
| **HotpotQA** | **+0.080** ✱ | **−0.052 [−0.078, −0.025]** ✱ **reversed** |
| SimpleQA | +0.085 ✱ | +0.006 (n.s.) |
| MuSiQue-42 | −0.005 | +0.012 (n.s.) |
| MuSiQue-123 | −0.021 | −0.005 (n.s.) |

**96% of the +0.080 came from 281 episodes where both arms already contained the
gold answer** — the treatment simply said it in 4.3 words instead of 6.9. On
whether the answer was found at all, the treatment **lost 49 questions and won
18** (McNemar p = 0.017).

We had reported this as *"cheaper **and** better — a Pareto improvement."* The
correct statement is *"cheaper and, on the gate dataset, significantly worse."*

### Diagnostic

Re-score with **at least one length-insensitive metric** — substring containment,
gold recall, or an LLM judge — and report answer-length statistics for both arms.
If lengths differ, F1 alone cannot support an iso-quality claim.

### Correction

Treat F1/EM as **length-confounded** whenever the intervention plausibly changes
verbosity. Report containment or recall as the primary quality axis.

---

## 5. Artefact 3 — Conditioning on one arm's outcome manufactures selectivity

### The mechanism

A natural way to show an efficiency method is *targeted* rather than merely hasty:
split questions by whether the control succeeded, and show the saving concentrates
on the ones it failed.

This is regression to the mean. Conditioning on the control being extreme selects
a subgroup where the *other* arm regresses toward its own mean. **Determinism does
not rescue it** — at temperature 0 there is no sampling noise, but the idiosyncratic
policy difference plays the identical mathematical role.

### What it did to our result

We reported: saving of **−0.49 on doomed work vs −0.03 on successful work**, a
"16× selectivity ratio", and called it the central mechanism.

Running the identical analysis between **two policies with the same λ**:

| comparison | doomed | successful | **gap** |
|---|---|---|---|
| **placebo** — two λ=0 controls | −0.145 | +0.279 | **+0.42** |
| **placebo** — two λ=0.568 arms | −0.222 | +0.269 | **+0.49** |
| "real" MuSiQue-42 | −0.500 | +0.062 | +0.56 |
| "real" MuSiQue-123 | −0.663 | +0.168 | +0.83 |

**70–85% of the signature appears with no treatment effect whatsoever.**

A second, cheaper check: partition by the **treatment's** own outcome instead. If
the policy abandoned work it expected to fail, the episodes *it* failed are where
it should save. Result: −0.160 vs −0.170 — **no selectivity at all.**

### Diagnostic

Two, both cheap:

1. **Placebo calibration.** Run the split between two arms that differ only by
   seed. Report the null gap beside the real one. If the real gap does not clearly
   exceed it, there is no selectivity result.
2. **Symmetric partition.** Report the both-fail / both-succeed / disagree
   breakdown instead of conditioning on one arm.

### Correction

Never condition on a single arm's outcome without a null-calibrated baseline.
Our pre-registered criterion (`mean(Δ|failed) < mean(Δ|succeeded)`, no significance
requirement, no calibration) was **satisfied by all four placebos** — it could not
fail.

---

## 6. What survived

After all three corrections:

> A per-step price makes the policy **terser and more robust to format collapse**.
> Total cost falls ~6–8%, of which roughly two-thirds is fewer parser failures and
> **~1–3% is genuinely less deliberation**. Tokens fall ~10–13% on well-formed
> work. Accuracy is unchanged on three datasets and significantly worse on one.

The pre-registered gate, which passed at −0.167, **fails on the corrected
estimand** (−0.112 against a −0.119 threshold).

---

## 7. Why the standard safeguards did not help

| safeguard | present? | caught it? |
|---|---|---|
| Pre-registered hypothesis, threshold and analysis script, committed before data | ✔ | ✘ |
| Power analysis; threshold derived from measured headroom | ✔ | ✘ |
| Frozen held-out test set | ✔ | ✘ |
| Matched control differing in exactly one variable | ✔ | ✘ |
| Out-of-distribution replication | ✔ | ✘ |
| Mechanism test (is the saving where the theory says?) | ✔ | ✘ — *was itself the artefact* |
| Seed replication | ✔ | ✘ |
| **Adversarial audit by independent readers** | added late | **✔ all three** |

Every safeguard tests whether the *number* is trustworthy. None tests whether the
**estimand measures what the claim says**. That gap is the paper's point.

The mechanism test is the sharpest illustration: designed to guard against
"the policy just got hastier," it was the artefact with the largest apparent
effect size.

---

## 8. Recommended checklist

For any agent-efficiency claim:

1. **Report the median and trimmed mean** with the mean. State the fraction of the
   effect carried by the top 2% of episodes.
2. **Report cap-touching and malformed rates for both arms**, and the effect with
   and without them.
3. **Re-score quality on a length-insensitive metric.** Report answer lengths.
4. **Placebo-calibrate any subgroup analysis** using two arms that differ only by
   seed.
5. **Prefer symmetric partitions** (both-fail / both-succeed) to conditioning on
   one arm.
6. **Record retry counts and per-attempt token usage.** Totals that silently
   include retries cannot be decomposed afterwards — ours cannot, permanently.
7. **Pre-declare the test family** and correct for multiplicity. Our gate
   (p = 0.0041) does not survive Bonferroni over the family we reported.

---

## 9. Related work

The efficiency method we tested is not novel and we do not claim it is.
[MASH](https://arxiv.org/abs/2510.01152) (Oct 2025) trains abstention by
penalising search under GRPO on the same benchmarks;
[OTC-PO](https://arxiv.org/abs/2504.14870) (NeurIPS 2025) puts a tool-call cost in
PPO and GRPO and reports up to **68.3%** fewer calls. Length-penalty RL is a
labelled category in at least two 2025 surveys.

Our contribution is not the method but **the measurement failure**, and the prior
work's larger effect sizes make the artefacts *more* relevant, not less: a method
reporting 68% savings on a `steps_used`-style metric has not necessarily excluded
Artefact 1.

---

## 10. Limitations

- **One case study.** We show these artefacts can dominate a result; we do not
  quantify how often they do in published work.
- **Our token figures remain inflated** by retry duplication that cannot be
  corrected retrospectively — the instrumentation was added only after the fact.
  This is itself an instance of item 6 above.
- **Two seeds, not three.** A third was blocked by shared-cluster contention.
- **The surviving effect is small** (~1–3% steps) and we did not pursue it.
- The containment metric is a proxy for a human or LLM judge, which we did not run.

---

## 11. Reproducing this

| what | where |
|---|---|
| Corrected results | `foundation/experiments/reports/CORRECTED_RESULTS.md` |
| Retracted originals (kept as record) | `T5_SYNTHESIS.md`, `COMPLETE_REPORT.md` |
| Ten withdrawn claims, with what refuted each | `CORRECTED_RESULTS.md` §6, `T5_SYNTHESIS.md` §5 |
| Raw episodes | `foundation/experiments/results/{s5_eval,t2_simpleqa,t4_musique,t3_seeds}/` |
| Pre-registrations | `s3_preregistration.md`, `t4_preregistration.md` |
| Audit findings | `u2_mechanism_correction.md` and the reports above |

Every table in this document was recomputed from the raw episode files on
2026-08-11.
