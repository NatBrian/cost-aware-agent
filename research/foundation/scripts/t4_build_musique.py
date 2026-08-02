"""T4a — build MuSiQue train/eval splits, matched to the HotpotQA ones.

WHY MuSiQue. Step 1's effect is −0.167 steps on a ~3-step task, roughly 6%.
Honest, but thin. The question that decides whether the result is interesting is
whether it **grows with horizon**. MuSiQue is built by composing single-hop
questions specifically to defeat shortcut reasoning: its DiRe (disconnected
reasoning) answer-F1 is **37.8 vs HotpotQA's 68.8**, and its human–model gap is
28.2 vs 9.6. So it should contain genuinely more required work per question.

SIZES ARE MATCHED TO HOTPOTQA ON PURPOSE (300 train / 600 eval). If MuSiQue got
more data, a larger effect could be data volume rather than horizon, and the
comparison would be uninterpretable.

Stratified by **hop count** (length of `question_decomposition`), which is the
variable of interest — it is what "horizon" means here, and it lets the analysis
ask whether |Δsteps| grows with hops *within* the dataset as well as between
datasets.

Only `answerable` questions are kept: an unanswerable question has no gold to
score against, and our F1 path would treat it as a guaranteed failure for both
arms, adding noise without signal.

Usage: .venv/bin/python scripts/t4_build_musique.py
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("HF_HOME", "/mnt/src/liangsheng/cassi_foundation/hf_cache")

from common import FOUNDATION_ROOT, load_config, sha256_file
from collect.sampling import stratified_sample, write_jsonl
from eval.qa_metrics import normalize

TRAIN_OUT = "data/musique_train_300.jsonl"
EVAL_OUT = "data/musique_eval_600.jsonl"


def to_rows(split) -> list[dict]:
    out = []
    for r in split:
        if not r.get("answerable", True):
            continue
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        if not q or not a:
            continue
        answers = [a] + [x for x in (r.get("answer_aliases") or []) if x]
        hops = len(r.get("question_decomposition") or []) or 2
        out.append({"id": str(r["id"]), "question": q, "answers": answers,
                    "level": f"{hops}hop", "hops": hops})
    return out


def main() -> None:
    from datasets import load_dataset
    cfg = load_config()
    d = cfg["data"]
    root = FOUNDATION_ROOT

    ds = load_dataset("dgslibisey/MuSiQue")
    train = to_rows(ds["train"])
    val = to_rows(ds["validation"])
    print(f"answerable: train {len(train)}  validation {len(val)}")
    print(f"train hops: {dict(sorted(Counter(r['hops'] for r in train).items()))}")
    print(f"val   hops: {dict(sorted(Counter(r['hops'] for r in val).items()))}")

    # sizes matched to the HotpotQA splits so the only difference is the dataset
    tr = stratified_sample(train, d["train_size"], d["sampling_seed"] + 31)
    ev = stratified_sample(val, d["eval_size"], d["sampling_seed"] + 37)

    # eval must not overlap train, by id or by normalized question
    tr_ids = {r["id"] for r in tr}
    tr_qs = {normalize(r["question"]) for r in tr}
    assert not (tr_ids & {r["id"] for r in ev}), "eval overlaps train by id"
    assert not (tr_qs & {normalize(r["question"]) for r in ev}), \
        "eval overlaps train by question text"

    for rows, out in ((tr, TRAIN_OUT), (ev, EVAL_OUT)):
        p = root / out
        write_jsonl(rows, p)
        print(f"wrote {p}  n={len(rows)}  sha256={sha256_file(p)[:16]}…  "
              f"hops={dict(sorted(Counter(r['hops'] for r in rows).items()))}")

    man_p = root / d["manifest_file"]
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    for rows, out in ((tr, TRAIN_OUT), (ev, EVAL_OUT)):
        man.setdefault("files", {})[out] = {
            "n": len(rows), "sha256": sha256_file(root / out),
            "source": "dgslibisey/MuSiQue (answerable only)",
            "purpose": "T4 horizon-scaling test; sizes matched to the HotpotQA splits",
            "hops": {str(k): v for k, v in sorted(Counter(r["hops"] for r in rows).items())},
        }
    man_p.write_text(json.dumps(man, indent=2))
    print(f"manifest updated: {man_p}")


if __name__ == "__main__":
    main()
