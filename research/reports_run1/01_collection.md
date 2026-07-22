# Report 01 — Collection Round 0 (P2)

**Date:** 2026-07-22 · **Hardware:** 2 GPUs (torch retrieval server on GPU 6, Qwen3.5-9B on GPU 7)
· **Status: ✅ DONE — 9,600 / 9,600 trajectories**

## What we ran, in plain words

This is the project's raw-material harvest. 1,200 questions (a mix of HotpotQA,
NQ, and MuSiQue — easy 1-hop to hard 4-hop), each solved **8 times** by the
agent, under a randomly drawn wallet per question (small / medium / large).
Two special rules make this data usable for stopping labels:

1. **Forced continuation:** even when the agent says "final answer" at step 4,
   we log that moment and keep it running to step 10. Why: to learn *when
   stopping is best*, we must observe what would have happened after every
   possible stopping point. Without this, the future after early answers would
   be invisible.
2. **Running draft:** every step ends with a "BEST ANSWER SO FAR" line, so we
   can score answer quality *at every step* later by simple string comparison —
   free, no extra model calls.

## Results

| Measure | Value | What it means |
|---|---|---|
| Trajectories | **9,600** (1,200 tasks × 8) | meets the ≥8,000 requirement exactly |
| Steps per trajectory | 10.0 | forced continuation worked — nothing stopped early |
| Total simulated spend | **$143.08** | the whole harvest cost less than a dinner |
| Forced-continuation overhead | $65.80 (46%) | the price of running past the natural stop — honestly billed to the method's account (paper table T4) |
| Draft-line tokens | 2.3% of all tokens | the "answer so far" bookkeeping is cheap |
| Wallet balance | 382 / 417 / 401 (small/med/large) | balanced as designed — budget variety is in the data |
| Trajectories where the agent tried to answer | **68%** | free measurement: the base model *wants* to stop early two-thirds of the time; 32% never found an answer in 10 steps |

## Engineering problems solved on the way (the big one)

The stock retrieval server searched the 21-million-passage index on CPU:
**35 seconds per search** — the whole harvest would have taken GPU-weeks. The
GPU version of the faiss library doesn't support this machine's GPUs at all.
Fix: a small replacement server (`scripts/torch_retrieval_server.py`) that
keeps the *identical* encoder, corpus, and exact search math, but does the
search as one GPU matrix multiplication: **46 milliseconds** (≈760× faster).
Result quality was spot-checked (e.g., "who wrote the novel The Sea" → the
John Banville article, correct). This counts as a serving change, not a
science change — worth one sentence in the paper's appendix.

## What this unlocks

Everything from here to the first big decision gate is CPU-only or small-GPU:
- **Next: Report 02 — stopping labels** (the Snell backward recursion: for
  every one of the ~96,000 logged steps, compute "was stopping here better
  than continuing?", at five different price-sensitivity levels λ)
- Then the 2B reward model is trained on those labels (Report 03) and must
  pass its gate before any RL training money is spent.
