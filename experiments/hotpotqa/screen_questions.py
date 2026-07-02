#!/usr/bin/env python3
"""Stage 2: closed-book screen the candidate pool and keep only questions the
model CANNOT answer from memory — the ones that actually force retrieval, so a
money budget has something to bite on.

For each candidate we ask the SAME model (Claude Sonnet via `claude -p`) the
question with NO tools and NO corpus. If it answers correctly from parametric
memory (EM=1, or high token F1), the question is memorized trivia → DROP it.
We keep the hardest-for-memory candidates (EM=0, lowest F1 first) and take the
top N. The final pooled corpus is built ONLY from the kept questions' distractor
contexts, so every kept question is answerable from the offline corpus (HotpotQA
distractor guarantees the gold passages are present) yet not from memory.

Writes:
  data/questions.json      kept questions, reindexed q0..q{N-1} (harness input)
  data/corpus.json         pooled title->passage from kept questions
  data/screen_report.json  full closed-book result per candidate (provenance)

Usage: screen_questions.py --n 10 [--f1-keep 0.4] [--workers 4]
"""
import argparse
import json
import os
import re
import string
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL = "sonnet"


def _norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def score(pred, gold):
    np_, ng = _norm(pred), _norm(gold)
    em = float(np_ == ng)
    pt, gt = np_.split(), ng.split()
    if not pt or not gt:
        return em, float(np_ == ng)
    common = {}
    for w in pt:
        if w in gt:
            common[w] = min(pt.count(w), gt.count(w))
    nsame = sum(common.values())
    if nsame == 0:
        return em, 0.0
    prec, rec = nsame / len(pt), nsame / len(gt)
    return em, 2 * prec * rec / (prec + rec)


CLOSED_BOOK = (
    "Answer this question with a short factual answer only — a name, date, or "
    "short phrase. Output ONLY the answer on a single line, nothing else. If you "
    "are unsure, give your single best guess.\n\nQuestion: {q}"
)


def closed_book(question, tries=3):
    """One closed-book model call. Returns the model's short answer (or '')."""
    prompt = CLOSED_BOOK.format(q=question)
    last = None
    for i in range(tries):
        try:
            p = subprocess.run(
                ["claude", "-p", prompt, "--model", MODEL,
                 "--output-format", "json", "--allowedTools", ""],
                capture_output=True, text=True, timeout=120)
            d = json.loads(p.stdout)
            if d.get("is_error"):
                raise RuntimeError(str(d.get("result"))[:120])
            return (d.get("result") or "").strip().splitlines()[0].strip() \
                if d.get("result") else ""
        except Exception as e:
            last = e
            time.sleep(2 + 2 * i)
    print(f"  [screen] call failed: {last}")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="questions to keep")
    ap.add_argument("--f1-keep", type=float, default=0.4,
                    help="keep only candidates whose closed-book F1 <= this")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    candidates = json.load(open(os.path.join(DATA, "candidates.json")))
    print(f"screening {len(candidates)} candidates closed-book (workers={args.workers})...")

    def probe(idx_c):
        idx, c = idx_c
        pred = closed_book(c["question"])
        em, f1 = score(pred, c["answer"])
        print(f"  cand{idx:02d} em={em:.0f} f1={f1:.2f} "
              f"pred={pred[:34]!r} gold={c['answer'][:28]!r}", flush=True)
        return {**c, "cb_pred": pred, "cb_em": em, "cb_f1": round(f1, 3)}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        scored = list(ex.map(probe, enumerate(candidates)))

    json.dump(scored, open(os.path.join(DATA, "screen_report.json"), "w"), indent=2)

    # KEEP = model failed from memory. Rank hardest-first (lowest closed-book F1).
    kept = [s for s in scored if s["cb_em"] == 0.0 and s["cb_f1"] <= args.f1_keep]
    kept.sort(key=lambda s: s["cb_f1"])
    kept = kept[:args.n]
    if len(kept) < args.n:
        print(f"WARNING: only {len(kept)} candidates passed the screen "
              f"(wanted {args.n}). Consider a larger --pool.")

    # Build the final pooled corpus from ONLY the kept questions' contexts.
    corpus = {}
    questions = []
    for i, c in enumerate(kept):
        for t, sents in zip(c["titles"], c["sentences"]):
            if t not in corpus:
                corpus[t] = "".join(sents).strip()
        questions.append({
            "id": f"q{i}",
            "question": c["question"],
            "answer": c["answer"],
            "gold_titles": c["gold_titles"],
            "cb_pred": c["cb_pred"], "cb_f1": c["cb_f1"],  # provenance: proven not-memorized
        })

    json.dump(corpus, open(os.path.join(DATA, "corpus.json"), "w"))
    json.dump(questions, open(os.path.join(DATA, "questions.json"), "w"), indent=2)

    gold_ok = all(t in corpus for q in kept for t in q["gold_titles"])
    print(f"\nkept {len(kept)} retrieval-forcing questions | corpus {len(corpus)} passages")
    print(f"all gold titles present in corpus: {gold_ok}")
    print(f"closed-book F1 of kept: {[q['cb_f1'] for q in questions]}")


if __name__ == "__main__":
    main()
