"""T2a — build the SimpleQA negative-control evaluation set.

WHY. Step 1 showed a +0.080 F1 gain that T1 proved is INDEPENDENT of the step
saving (significant only on episodes whose step count never changed). Two
explanations remain: λ teaches cost-awareness and separately improves quality, or
λ is simply a better-conditioned objective and the quality gain is generic.

SimpleQA discriminates them. Its questions are **single-hop, single-answer, short
fact lookups** — there is no discretionary work, so a cost-awareness effect
*cannot* appear. A quality gain there would therefore be generic by construction.

  Δsteps ≈ 0 AND ΔF1 ≈ 0  ->  the quality gain is specific to multi-hop work
  ΔF1 > 0 with CI off zero ->  λ is a general regulariser; report as a confound

Both arms are already trained; this is evaluation only, no retraining.

POWER CAVEAT, checked before the result is trusted: our agent retrieves over a
2018 Wikipedia dump, while SimpleQA is adversarially built against GPT-4 and
skews obscure. If BOTH arms score near zero, ΔF1 is trivially ~0 and the control
is uninformative rather than passed. `t2_eval_simpleqa.sh` runs a 50-question
pilot first and reports the baseline F1 so that case is caught, not claimed.

Usage: .venv/bin/python scripts/t2_build_simpleqa.py --n 500
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import FOUNDATION_ROOT, load_config, sha256_file

SRC = "../data_shared/simpleqa/simple_qa_test_set.csv"
OUT = "data/simpleqa_eval.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--pilot", type=int, default=50)
    args = ap.parse_args()
    cfg = load_config()
    root = FOUNDATION_ROOT
    src = root / SRC
    if not src.exists():
        raise SystemExit(f"missing {src}")

    rows = list(csv.DictReader(open(src)))
    print(f"SimpleQA source: {len(rows)} questions")

    # Seeded and disjoint-by-construction: SimpleQA shares no questions with
    # HotpotQA, so no decontamination against train-300 is needed. The seed is
    # offset from every other draw in this project so the sample is independent.
    rng = random.Random(cfg["data"]["sampling_seed"] + 21)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    sel = idx[: args.n]

    out = []
    for i, j in enumerate(sel):
        r = rows[j]
        q, a = (r.get("problem") or "").strip(), (r.get("answer") or "").strip()
        if not q or not a:
            continue
        out.append({"id": f"simpleqa-{j}", "question": q, "answers": [a],
                    "level": "simpleqa"})
    p = root / OUT
    with open(p, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {p}  n={len(out)}  sha256={sha256_file(p)[:16]}…")

    # a pilot slice, so the power caveat can be checked cheaply first
    pp = root / OUT.replace(".jsonl", "_pilot.jsonl")
    with open(pp, "w") as f:
        for r in out[: args.pilot]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {pp}  n={min(args.pilot, len(out))} (baseline-F1 power check)")


if __name__ == "__main__":
    main()
