#!/usr/bin/env python3
"""Build a local multi-hop QA corpus from HotpotQA (distractor).

HotpotQA-distractor ships each question with 10 context paragraphs (2 gold +
8 distractors). We sample N questions and POOL every paragraph into one shared
corpus keyed by title — so the agent must retrieve the right passages among
distractors (from this and other questions), multi-hop reason, and answer.
Fully offline: no web search, no API key. Gold answers are short/yes-no →
deterministic EM/F1 grading.
"""
import json, os, sys

OUT = os.path.dirname(os.path.abspath(__file__)) + "/data"
os.makedirs(OUT, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SEED = 0

from datasets import load_dataset

ds = load_dataset("hotpot_qa", "distractor", split="validation")
ds = ds.shuffle(seed=SEED).select(range(N))

corpus = {}      # title -> paragraph text
questions = []
for i, ex in enumerate(ds):
    titles = ex["context"]["title"]
    sents = ex["context"]["sentences"]
    for t, s in zip(titles, sents):
        if t not in corpus:
            corpus[t] = "".join(s).strip()
    questions.append({
        "id": f"q{i}",
        "question": ex["question"],
        "answer": ex["answer"],
        "gold_titles": list(dict.fromkeys(ex["supporting_facts"]["title"])),
    })

json.dump(corpus, open(f"{OUT}/corpus.json", "w"))
json.dump(questions, open(f"{OUT}/questions.json", "w"), indent=2)
print(f"corpus: {len(corpus)} passages | questions: {len(questions)}")
print("sample Q:", questions[0]["question"][:80], "-> A:", questions[0]["answer"])
print("gold titles present in corpus:",
      all(t in corpus for q in questions for t in q["gold_titles"]))
