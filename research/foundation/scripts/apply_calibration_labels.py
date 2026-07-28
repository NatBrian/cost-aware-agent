"""Write blind subagent labels back into the calibration sheet.

Input: one or more JSON files, each a list of
  {"sheet_id": "12", "new_info": 0|1, "not_redundant": 0|1, "was_needed": 0|1}
for working steps, or
  {"sheet_id": "12", "supported": 0|1, "nothing_left": 0|1}
for answer steps.

Refuses to write a label for a bit that does not belong to the row's
action_type, and reports any sheet row left unlabelled — a silently short sheet
would shrink the agreement denominator and quietly inflate the gate.

Usage:
  .venv/bin/python scripts/apply_calibration_labels.py \
      --sheet experiments/results/calibration/sheet_v2.csv labels_*.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reward.rubric import ANSWER_BITS, STEP_BITS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--sheet", required=True)
    args = ap.parse_args()

    by_id: dict[str, dict] = {}
    for p in args.labels:
        for rec in json.loads(Path(p).read_text()):
            by_id[str(rec["sheet_id"])] = rec

    sheet = Path(args.sheet)
    with open(sheet) as f:
        header = [l for l in f if l.startswith("#")]
    with open(sheet) as f:
        rows = list(csv.DictReader([l for l in f if not l.startswith("#")]))
        fields = list(rows[0].keys())

    applied = missing = 0
    for r in rows:
        rec = by_id.get(r["sheet_id"])
        if not rec:
            missing += 1
            continue
        bits = ANSWER_BITS if r["action_type"] == "answer" else STEP_BITS
        for b in bits:
            if b not in rec:
                raise SystemExit(f"row {r['sheet_id']} ({r['action_type']}) "
                                 f"missing bit {b} in labels")
            v = int(rec[b])
            if v not in (0, 1):
                raise SystemExit(f"row {r['sheet_id']} bit {b}: {v!r} not 0/1")
            r[f"label_{b}"] = v
        applied += 1

    with open(sheet, "w", newline="") as f:
        for h in header:
            f.write(h)
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"applied labels to {applied}/{len(rows)} rows"
          + (f" — {missing} UNLABELLED" if missing else ""))
    if missing:
        raise SystemExit("refusing to proceed with an incomplete sheet: an "
                         "unlabelled row silently shrinks the agreement "
                         "denominator and inflates the gate")


if __name__ == "__main__":
    main()
