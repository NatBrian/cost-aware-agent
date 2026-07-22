# F1 — Data: HotpotQA samples + local retrieval

**Goal:** frozen, hashed, small datasets and a working local search tool.

## Work items

1. **Train sample:** 300 questions from HotpotQA train. Stratify by the official
   hard/medium level field so the mix isn't accidentally all-easy (log the strata
   counts). Seeded sampling (seed in config), script re-runnable.
2. **Dev sample (the frozen eval set):** 200 questions from HotpotQA dev,
   same stratification, sampled ONCE, then frozen — every arm, budget, and later
   rerun evaluates this identical list (v2.1 §5.6 frozen-subsample rule, kept).
3. **Retrieval:** local Wikipedia index, Search-R1 recipe (E5 dense or BM25 —
   whichever artifact was rescued into `research/data_shared/` in F0; if none
   survives, build BM25 first since it needs no GPU, add E5 only if retrieval
   quality is visibly the bottleneck in the F2 pilot).
4. **Manifest:** `foundation/data/manifest.json` — counts, sampling seed, SHA256 of
   each question-list file, index provenance. Committed to git (the data itself is
   gitignored as before; the script regenerates it).
5. Basic overlap check: no question appears in both train-300 and dev-200
   (HotpotQA splits are disjoint, but the check is one line and the manifest
   should say it ran). The full v2.1 contamination protocol is deferred.

## Deliberately NOT done

No MuSiQue/NQ mix, no ALFWorld, no OOD sets, no decontamination sweep — full plan.

## Known limitations — symptom → what to refine first (recorded so mid-experiment
## debugging starts here, not from scratch)

| Limitation accepted now | Symptom if it bites | First refinement |
|---|---|---|
| Only 300 train tasks (senior's "couple hundred" directive) | F5 micro-run/full run: reward curves too noisy to see a trend; GRPO advantages dominated by a few tasks | Resample to 500 via the sampling script's count flag (dev-200 stays frozen; rerun manifest) |
| BM25 fallback retriever is weaker than E5 | F2 pilot: quality-vs-steps curves flat because search rarely finds the right passage (agent can't succeed at ANY step count) | Build/restore the E5 index before touching agent or rubric — retrieval failure masquerades as stopping failure |
| Stratification trusts HotpotQA's easy/medium/hard field | Pilot: nearly all tasks solved in ≤2 steps (budget never binds, stopping problem trivial) | Re-stratify by observed steps-to-first-correct-draft from pilot trajectories instead of the dataset's label |
| No base-model decontamination sweep | Closed-book spot-check (no search) already answers many dev questions correctly | Memorization inflates all three arms equally, so arm *differences* stay valid — but report it, and prefer step-count deltas over absolute F1 in the writeup |

## Done criterion

`scripts/f1_data.sh` runs end-to-end on a clean checkout (given the shared index);
manifest committed; a hand-run of 3 sample retrievals returns sane passages.

Depends on: F0. Feeds: F2 (collection), F4/F6 (eval).
