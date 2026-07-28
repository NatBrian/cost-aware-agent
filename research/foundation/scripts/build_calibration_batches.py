"""Build the calibration sheet and split it into batches for blind labeling.

WHY THE DATA BLOCK ONLY: the sheet's `context` column is the *rendered rubric
prompt* — bit definitions, worked examples and tie-break rules included. Handing
that to a labeler primes it with the very instructions under test, so agreement
would partly measure "did both parties read the same wording the same way"
instead of "does the judge agree with an independent reading of the step". Each
batch therefore carries the DATA block only (question, history, drafts, budget,
this step), plus neutral bit definitions written from F3's intent without v3's
tie-breaks. Ground truth must not be defined by the prompt being evaluated.

Labelers are freshly spawned subagents with no session context: no judge output,
no rubric version history, no knowledge of this conversation.

Usage:
  .venv/bin/python scripts/build_calibration_batches.py \
      --pilot experiments/results/pilot/pilot.jsonl \
      --sheet experiments/results/calibration/sheet_v2.csv \
      --out-dir experiments/results/calibration/batches --n 150 --batch 15
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import FOUNDATION_ROOT, load_config
from reward.calibration import make_labeling_sheet
from reward.rubric import ANSWER_ANCHOR, STEP_ANCHOR

# Neutral restatements of the five bits. Deliberately NOT the rubric's wording:
# no worked examples, no "if undecided answer NO/YES" tie-breaks, no anchoring
# phrases. Faithful to F3's definitions, stripped of anything tuned for a judge.
STEP_DEFS = """You are shown one step of an agent that answers a question by searching Wikipedia.
Judge only from what is shown. You do NOT know the correct answer, and you must not
use outside knowledge of the answer to decide.

Answer three yes/no questions about THIS step (1 = yes, 0 = no):

new_info   — Did this step's RESULT add information relevant to the question that was
             not already present in the earlier history?
not_redundant — Was this step's QUERY meaningfully different from the earlier queries,
             rather than a repeat or rewording that targets the same information?
was_needed — Before this step ran, was more work still genuinely needed — i.e. was the
             draft answer missing, incomplete, or not yet supported by the history?"""

ANSWER_DEFS = """You are shown the final answer of an agent that answers a question by searching
Wikipedia, together with the history that led to it. Judge only from what is shown.
You do NOT know the correct answer, and you must not use outside knowledge of the
answer to decide.

Answer two yes/no questions (1 = yes, 0 = no):

supported    — Is this answer backed by the evidence in the history — do the retrieved
               results state it or directly imply it?
nothing_left — Was stopping at this point reasonable, i.e. would further searching be
               unlikely to change or improve this answer?"""


def data_block(context: str, action_type: str) -> str:
    anchor = ANSWER_ANCHOR if action_type == "answer" else STEP_ANCHOR
    head, sep, _ = context.partition(anchor)
    if not sep:
        raise ValueError(f"no {action_type} anchor in context")
    return head.rstrip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch", type=int, default=15)
    args = ap.parse_args()
    cfg = load_config()
    n = args.n or cfg["rubric"]["calibration"]["n_steps"]

    made = make_labeling_sheet(args.pilot, args.sheet, n=n, seed=cfg["seed"])
    print(f"sheet: {made} rows -> {args.sheet}")

    with open(args.sheet) as f:
        rows = list(csv.DictReader([l for l in f if not l.startswith("#")]))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        items = [{"sheet_id": r["sheet_id"], "action_type": r["action_type"],
                  "data": data_block(r["context"], r["action_type"])}
                 for r in chunk]
        p = out / f"batch_{i // args.batch:02d}.json"
        p.write_text(json.dumps({"step_definitions": STEP_DEFS,
                                 "answer_definitions": ANSWER_DEFS,
                                 "items": items}, indent=2))
        print(f"  {p} ({len(items)} items)")


if __name__ == "__main__":
    main()
