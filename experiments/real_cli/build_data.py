#!/usr/bin/env python3
"""Build the real-CLI experiment's question sets from the HotpotQA screen.

Two DISJOINT sets, both from the same closed-book screen (../hotpotqa):
  eval.json   the 10 retrieval-forcing questions (closed-book F1 = 0.0) the
              main sweep runs on — same questions as the hotpotqa experiment
  calib.json  N additional closed-book-hard questions (F1 = 0.0, NOT in the
              eval set) used ONLY to calibrate the budget from OFF-arm cost.
              Budget must never be tuned on the eval questions themselves —
              that was an audit finding against the earlier experiment.

Each question carries its OWN 10 distractor-setting passages (2 gold + 8
distractors, straight from HotpotQA), which the harness writes into the run
sandbox as corpus/*.txt files — the real CLI agent retrieves with its native
Read/Grep/Glob over actual files.
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
HOTPOT_DATA = os.path.join(HERE, "..", "hotpotqa", "data")


def passages(cand):
    return {t: "".join(s) for t, s in zip(cand["titles"], cand["sentences"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-n", type=int, default=6)
    args = ap.parse_args()

    cands = json.load(open(os.path.join(HOTPOT_DATA, "candidates.json")))
    report = json.load(open(os.path.join(HOTPOT_DATA, "screen_report.json")))
    kept = json.load(open(os.path.join(HOTPOT_DATA, "questions.json")))

    by_question = {c["question"]: c for c in cands}
    kept_qs = {q["question"] for q in kept}

    evalset = []
    for q in kept:
        c = by_question[q["question"]]
        evalset.append({"id": q["id"], "question": q["question"], "answer": q["answer"],
                        "gold_titles": q["gold_titles"], "cb_f1": q["cb_f1"],
                        "passages": passages(c)})

    # calibration pool: screened hardest (closed-book F1 == 0), excluding eval,
    # in screen-report order (deterministic, no re-cherry-picking)
    pool = [r for r in report if r["f1"] == 0.0 and r["question"] not in kept_qs]
    calib = []
    for i, r in enumerate(pool[: args.calib_n]):
        c = cands[r["idx"]]
        assert c["question"] == r["question"]
        calib.append({"id": f"c{i}", "question": c["question"], "answer": c["answer"],
                      "gold_titles": c["gold_titles"], "cb_f1": r["f1"],
                      "passages": passages(c)})

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(evalset, open(os.path.join(HERE, "data", "eval.json"), "w"), indent=1)
    json.dump(calib, open(os.path.join(HERE, "data", "calib.json"), "w"), indent=1)
    overlap = {q["question"] for q in evalset} & {q["question"] for q in calib}
    print(f"eval: {len(evalset)} questions, calib: {len(calib)} questions, "
          f"overlap: {len(overlap)} (must be 0)")
    assert not overlap


if __name__ == "__main__":
    main()
