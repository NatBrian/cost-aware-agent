#!/usr/bin/env python3
"""Stage 1 of the money-budget experiment: sample a POOL of hard HotpotQA
(distractor) candidates that are *retrieval-shaped*, not trivia the model can
recite from memory.

Why this matters. The preliminary run's null result had a single root cause:
Claude Sonnet answered famous-entity HotpotQA questions straight from parametric
memory in ONE turn, so there was nothing for a money budget to bite on. The whole
validation split is already labelled `level=hard`, so difficulty labels don't
separate memorized from non-memorized. We fix this in two stages:

  build_corpus.py  (this file) — sample a large candidate POOL, biased toward
      multi-hop *bridge* questions with named-entity answers (drop yes/no
      comparison answers, where a closed-book coin-flip would pollute screening).
  screen_questions.py          — closed-book screen each candidate with the real
      model; KEEP only the ones it gets wrong from memory (those force retrieval),
      then build the pooled corpus from the kept set.

Output: data/candidates.json — each item carries its full distractor context
(titles + sentences) so the screener can build the final corpus without
reloading the dataset.
"""
import argparse
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__)) + "/data"
os.makedirs(OUT, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=60,
                    help="number of candidate questions to sample for screening")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    ds = ds.shuffle(seed=args.seed)

    candidates = []
    for ex in ds:
        if len(candidates) >= args.pool:
            break
        # Bridge questions are genuinely multi-hop (find entity A, then a fact
        # about A) so they force retrieval steps; comparison yes/no answers make
        # closed-book screening a coin-flip. Keep bridge, non-yes/no only.
        if ex["type"] != "bridge":
            continue
        if ex["answer"].strip().lower() in ("yes", "no"):
            continue
        candidates.append({
            "hotpot_id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "gold_titles": list(dict.fromkeys(ex["supporting_facts"]["title"])),
            # full distractor context (2 gold + 8 distractors) kept for corpus build
            "titles": ex["context"]["title"],
            "sentences": ex["context"]["sentences"],
        })

    json.dump(candidates, open(f"{OUT}/candidates.json", "w"), indent=2)
    print(f"pool: {len(candidates)} bridge/non-yesno candidates (seed={args.seed})")
    print("sample Q:", candidates[0]["question"][:80], "-> A:", candidates[0]["answer"])


if __name__ == "__main__":
    main()
