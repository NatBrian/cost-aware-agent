# T2 — SimpleQA negative control — 2026-08-02

**Two effects, cleanly separated. The stopping effect is selective and transfers.
The quality effect is a generic regularisation confound.**

Evaluation only — both arms already trained on HotpotQA. SimpleQA is **out of
distribution**: single-hop short-fact lookups the policy never trained on.
n=500, B=2, temperature 0, harness off, paired.

## The power check, and the correction it forced

The plan justified SimpleQA as *"single-hop, so no discretionary work exists, so a
cost-awareness effect cannot appear."* A 50-question pilot showed that premise is
only half right: the agent spends **3.34 steps** on these single-hop questions and
leaves **62% unanswered**. Doomed work exists here too, so **a negative Δsteps
would be consistent with cost-awareness, not a falsification of it.**

The discriminating quantity is therefore **ΔF1**, not Δsteps — a single short fact
offers no multi-hop reasoning to do better. The pilot also confirmed the control
had power at all (baseline F1 0.202, 38% non-zero); had both arms scored ~0, ΔF1≈0
would have been trivially true and proved nothing.

## Result 1 — the quality gain is GENERIC (the confound)

| | control | treatment | Δ | 95% CI |
|---|---|---|---|---|
| F1 | 0.296 | 0.381 | **+0.085** | [+0.060, +0.109] |
| ΔF1 restricted to unshortened episodes | | | **+0.086** | [+0.060, +0.113] |

**HotpotQA's equivalent was +0.092.** The gain reproduces almost exactly on a
dataset where there is no multi-hop reasoning to improve and which the policy never
trained on.

**λ is acting as a general regulariser.** The quality improvement has nothing to do
with cost-awareness and must be reported as a confound, never as a benefit of the
method. This retires the "F1 rose, so the guard passed with room to spare" reading
in the S5 verdict.

## Result 2 — the stopping effect is SELECTIVE, and it transfers

The same H2 decomposition, partitioned by the control's outcome:

| partition | HotpotQA | **SimpleQA** |
|---|---|---|
| control **FAILED** (doomed) | −0.486 [−0.832, −0.151] ✱ | **−0.420 [−0.659, −0.193] ✱** |
| control **SUCCEEDED** | −0.031 [−0.093, +0.038] n.s. | **−0.013 [−0.089, +0.085] n.s.** |

**Near-identical on both datasets.** The saving lands on work that was going
nowhere and leaves successful work alone. If λ merely made the policy terser, the
reduction would be uniform across both partitions. It is not — on either dataset.

`corr(Δsteps, ΔF1)` = −0.047 on SimpleQA (−0.119 on HotpotQA): the two effects are
independent, confirming T1 on fresh, out-of-distribution data.

**This is a transfer result.** Cost-aware abandonment, trained on multi-hop
HotpotQA, generalises to single-hop SimpleQA — a task distribution the policy has
never seen. That is stronger evidence for the behaviour than the in-distribution
gate alone.

## What the paper may and may not claim

**May claim:**

1. A scalar per-step price teaches **cost-aware abandonment** — −0.167 steps at
   the gate (pre-registered, CI excluding zero).
2. The behaviour is **selective**, not general haste: it concentrates on doomed
   work and leaves successful work untouched, replicated on two datasets.
3. It **transfers out of distribution** to an unseen single-hop task.

**May not claim:**

4. That cost-aware training improves answer quality. **It does not.** The +0.09 F1
   is a generic regularisation effect that reproduces where cost-awareness cannot
   operate. It is a confound to disclose, and it means any future arm using λ needs
   a λ=0 control at matched training — comparing against an untrained or
   differently-trained baseline would silently attribute regularisation to the
   method.

## Limitations

- **The mechanism of the regularisation effect is unknown.** We know it is not
  better retrieval (T1: the treatment retrieves *fewer* distinct titles) and not
  measurably better formatting. Why λ improves answers is open.
- **One seed**, both arms. The separation is consistent across two datasets, which
  is reassuring but is not a seed replication.
- SimpleQA scoring uses our F1/EM path against a single short gold answer; the
  official benchmark uses an LLM grader. Absolute numbers are not comparable to
  published SimpleQA scores — only the *paired difference* between our two arms is
  meaningful here, and that is all it is used for.
- Both arms face the same 2018 Wikipedia index against questions that often
  postdate it. That depresses both arms equally and does not bias the paired
  comparison.

Artifacts: `experiments/results/t2_simpleqa/` (control.jsonl, treatment.jsonl,
pilot_control.jsonl, t2_verdict.txt) · scripts `t2_build_simpleqa.py`,
`t2_eval_simpleqa.sh`, `t2_analyse.py`.
