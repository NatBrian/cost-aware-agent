# Code audit vs the plan — 2026-07-28 (before restarting E-e)

**Why now.** The container wipe forced training to restart from base, which made
this the free moment to check the implementation against
`paper_plan_v2_1_foundation.md` and the F-docs — before spending GPU-days. The
piece most worth checking is the one with no upstream to trust: the in-house GRPO
trainer that replaced verl at E-d.

**Method.** `make test` (58 green) + `make dry-run`, then four parallel reads:
harness/collection vs plan §2–3 + F2; eval/metrics/gate vs plan §6–7 + F4/F6;
rubric/judge vs F3 (with `/home/liangsheng/PPTAgent/eval_codes` as the reference);
and trainer/advantages/rewards vs plan §4 + F5 by hand.

**Verdict: the pipeline is sound in its core math, and had four defects that
would have silently corrupted the run.** All four are fixed below. Suite now 60
green.

---

## Fixed — would have corrupted the result

**1. The gate blended three different A3 populations.** `eval/gate_check.py`
selected rows by `arm` alone. F6 deliberately collects A3 harness-OFF,
harness-ON, *and* an oracle forced-continuation replay, and all three carry
`arm="a3"` — so the headline "A3 with the harness off beats A1 and A2" would have
been computed on the average of the three. Both gate conditions (utility,
self-stop) were affected. Fixed: `GATE_MODE = {a1: none, a2: enforce, a3: none}`
pins the mode per arm, the gate now asserts each selection covers exactly
`data.dev_size` unique tasks, and it re-runs the utility and row-count guards on
its input instead of trusting the CSV. Regression test added
(`test_gate_ignores_a3_harness_on_rows`).

**2. `self_stopped` counted enforced episodes.** The definition keyed only on
`forced_stop`, but in `enforce` mode the harness is armed even on episodes it
never had to cut — so A2 was credited with "self-stopping" it does not do by
construction. Fixed: `self_stopped` now requires `mode == "none"`. This changes a
published number — see the correction to `baselines_report.md` below.

**3. Judge cache could be permanently poisoned.** `judge_client.py` wrote *every*
result to disk including the neutral 0.5 returned on parse/transport failure. One
outage on the shared judge server would have pinned those steps at reward 0
forever, unrepairable by rerun because the poisoned entry is a cache hit. Fixed:
neutral verdicts are never persisted; the failing prompt and reason are appended
to `judge_failures.jsonl` (a counter cannot tell you *which* step went neutral);
transport and parse failures are counted disjointly. Separately, the cache key now
includes the judge model and sampling params — it keyed only on rubric version,
so switching gemma-4-31B-it → Qwen3.6-27B would have silently served the old
judge's scores for the new one.

**4. Advantage groups could mix budgets.** `group_episodes` keyed on `task_id`
alone. Group advantages are z-scores *within* a group, so a group spanning two
wallets would score "3 steps of 8" against "3 steps of 2" and bake budget luck
into the advantage — exactly the confound plan §2 forbids. One round draws one
wallet per (task, group), so this was latent, not active; it is now impossible
(key is `(task_id, budget_B)`, with a test).

## Fixed — would have degraded, not corrupted

- **Silent sample dropping.** The trainer dropped over-long samples against a
  hard-coded 8192 cap while the server serves 16384. Long context correlates with
  *late steps*, so the drop was biased against precisely the over-continuation the
  reward is meant to punish. Cap now comes from `executor.max_model_len`; any
  dropping is logged loudly and >5% aborts the round rather than training on a
  biased subset.
- **No entropy curve** (plan §7 requires one). Added — and it is the metric that
  would have shown round 1's collapse before the post-round probe did.
- **The Fig-3 hacking curve had one point per round** (three for the whole run),
  because `batch_rewards` is called once per round. Now logs a row per group
  (~300/round) plus the round-level row, so judge-score-vs-realized-F1 divergence
  is visible *within* a round.
- **Bootstrap resamples**: plan §7 pre-registers 10k; every caller passed 2000.
- **`paired_delta` returned a wrong number on duplicated task ids** — the aligned
  subtraction fans out to a many-to-many join without raising, and the reported
  `n` understates the population actually averaged. Now refuses.
- **A0 was absent from the gate budget.** A0 sees no budget, so its behaviour
  cannot depend on B and plan §3 re-scores one run under all three budget
  utilities. That re-scoring was never implemented, so A0 existed only at B=8.
  Added in `build_rows.rescore_a0` (re-running A0 per budget instead would add
  sampling noise to an arm that is identical by construction).

## Open — decisions, not bugs

**The calibration gate's semantics quietly loosened.** Plan §5 and F3 both say
"≥80% per bit". The code gated on *mean* ≥0.80 with a 0.70 floor. Under the
spec's literal reading the 2026-07-22 run did **not** pass: `new_info` .792 and
`nothing_left` .769. `agreement()` now reports both readings explicitly and
`passed` follows the spec. This is re-decided in E-b anyway, since the judge
changed.

**Judge-vs-human agreement is upward-biased.** `supported` went .692 → .846 only
after a label-review round in which the human labels were revised *while reading
the judge's reasoning*. That is anchoring: the labels are no longer independent of
the judge. The sheet is preserved, so an independent relabel remains possible.
Also, F3 asks Brian to label; the sheet was author-labeled.

**Two judged quantities are exactly computable** (plan §4: anything computable
exactly is computed exactly, never judged): query redundancy (normalized-token
overlap against earlier queries) and result novelty (how many returned document
titles were already retrieved). The judge is currently asked to eyeball both from
truncated digests, and "lenient on redundancy" is the documented failure mode.
Candidate for `rubric_v2` — decided on evidence in E-b, not pre-emptively.

**The ANSWER prompt states a counterfactual under forced continuation**: it tells
the judge the agent "stopped at step t" when the pilot mode logs the answer and
keeps searching. 26 of the 50 calibration rows are answer rows, and
`nothing_left` is the weakest bit.

## Environment blockers found (not code)

- `.gpu_hold` still named GPU 6 from the old run; `e5_round.sh` greps it for
  `CUDA_VISIBLE_DEVICES`, so a round would have trained on a card we do not hold
  — violating the never-take-someone-else's-GPU rule. Must be rewritten from the
  live hold each session.
- `probe_policy_health.sh` reads `data/hotpotqa_val_50.jsonl`, and **nothing in
  the repo generates it**. The whole anti-overfitting policy rests on that
  validation slice. It must become a reproducible, seeded output of
  `scripts/f1_data.py` (F1 currently produces only train-300/dev-200).
- `requirements-gpu-pinned.txt` (vllm 0.17.1 / torch 2.10.0 / transformers 4.57.6)
  contradicts the PROGRESS log for the run that actually worked (vllm 0.25.1 /
  torch 2.11.0), and the trainer's own comments assume transformers 5. Re-pin
  honestly at rebuild.

## Deferred (logged, not fixed)

Resume/append has no config guard, so re-running a collection into an existing
file with a different arm/mode/budget silently mixes populations; `prompts.py` is
unversioned so `config_hash` is identical across prompt edits; the carried-forward
draft is logged as if the step emitted it (the judge then scores a draft that step
never produced); a missing draft line never triggers the specified retry; several
constants are hard-coded rather than config-sourced. These are real but do not
threaten the verdict; the first and third are the ones to fix next.
