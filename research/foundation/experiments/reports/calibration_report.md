# E-b calibration report — judge-agreement gate PASSED — 2026-07-22

**Judge:** gemma-4-31B-it (lab vLLM). **Sheet:** 50 stratified steps from pilot
trajectories (26 answer rows, 24 working-step rows). Labels: author-labeled
under the autonomy mandate (sheet kept for Brian's independent relabel:
`../results/calibration/sheet_v1.csv`).

## Final agreement (gate: mean >=0.80, floor 0.70 per bit)

| bit | agreement | note |
|---|---|---|
| was_needed | 1.000 | perfect |
| supported | 0.846 | after label-review round (below) |
| not_redundant | 0.833 | judge slightly LENIENT on redundancy (4 false-passes) |
| new_info | 0.792 | |
| nothing_left | 0.769 | hardest judgment call for both parties |
| **mean** | **0.848** | **PASSED** |

## The label-review round (documented honestly)

First run: supported=0.692 (below floor). Reviewing every disagreement with
the judge's own reasoning against FULL contexts found errors on BOTH sides:
- 4 rows: MY labels wrong (made from truncated views; e.g. final answers
  containing extra names/albums absent from evidence). Corrected.
- 4 rows: JUDGE wrong — claimed evidence absent that IS present but buried in
  long histories (attention misses, e.g. 'Roseanne' in step 3's digest).
  Labels kept; logged as a known judge failure mode.
Also verified: the judge correctly refuses to treat the agent's own repeated
answer-assertions as evidence (answer-echo hazard) in every sampled case.

## Known judge biases to watch during training (feeds the divergence read)

1. Slightly lenient on redundancy -> under-penalizes rephrase-loops (the exact
   RedundancyBench weakness v2.1 predicted; direction: rewards mildly inflated
   for wasteful steps).
2. Attention misses on long histories -> late-step `supported` judgments
   noisier than early-step ones.
Cost: ~780 tokens/judged step; 50-step sheet ~39k tokens.
