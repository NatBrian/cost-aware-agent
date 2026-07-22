# Report 00 — Smoke Test + Wallet Pilot (P0 done-criterion + P2 pilot)

**Date:** 2026-07-21 → 22 · **Hardware:** 2 GPUs (retriever on GPU 6, executor LLM on GPU 7)
· **Status: ✅ BOTH PASSED**

## What we ran, in plain words

Two things, in order:

1. **Smoke test** — one single question, run through the FULL pipeline end to end:
   the agent (Qwen3.5-9B) reads a HotpotQA question, searches our local Wikipedia
   (21M passages, E5 index), thinks step by step, writes its "BEST ANSWER SO FAR"
   line at every step, and every step's dollar cost is logged. Purpose: prove that
   every pipe is connected before spending real money.
2. **Wallet pilot** — 200 questions with NO budget limit, letting the agent spend
   naturally. Purpose: measure what tasks *naturally cost*, so we can set the
   three wallet sizes (small / medium / large) used in all later training.

## Results

### Smoke test: PASS
One full 10-step episode, total cost **$0.0098**, draft line present at every
step, all fields of the trajectory format verified by the checker script.
(That one episode costing about one cent tells you the scale of this whole
project's economics — the agent's decisions are about fractions of cents,
which is exactly why they must be *learned*, not hand-tuned.)

### Wallet calibration (now frozen into the config)

| Wallet | Meaning | Value |
|---|---|---|
| small | 25% of tasks naturally cost less than this | **$0.00182** |
| medium | 75% of tasks cost less than this | **$0.00988** |
| large | 2× the 90th-percentile spend — "money is no problem" | **$0.04043** |
| median spend | the cost-normalization constant (what "1 unit of cost" means) | **$0.00349** |

**How to read this:** a typical question costs about a third of a cent. A "small"
wallet (~$0.002) forces the agent to be frugal; a "large" wallet (~$0.04) means
it can afford ~12× the typical spend. During data collection, every task group
gets a randomly drawn wallet — that's how budget-awareness gets *into the
training data* rather than being a hand-written rule.

## Problems found and fixed on the way (so nobody re-hits them)

- This machine has **no CUDA compiler (nvcc)** → the LLM server must be launched
  with two flags that avoid runtime kernel compilation (both baked into the
  script now).
- Port 8001 is occupied by an unrelated service → our server runs on 8901.
- The pinned Search-R1 retriever has **no `--port` option** (it hardcodes 8000)
  and its API **crashes unless `return_scores: true` is sent** (upstream bug —
  fixed on our side; the third-party code stays untouched).
- Two colleagues' GPU jobs cycled across all 8 cards during the evening; we
  waited for a stable free window instead of racing (twice our server got
  crowded out mid-load before we adopted the wait-for-stable rule).

## What this unlocks

The wallet numbers are **frozen** in `configs/cassi.yaml` — the runbook forbids
any later phase from re-deriving them (that would silently shift the whole
economy between phases). Next: **Report 01 — full collection round** (thousands
of episodes with randomized wallets and forced continuation), the raw material
for the stopping labels.
