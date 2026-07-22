#!/usr/bin/env python3
"""Build a task file for cassi.executor.collect from staged dataset JSONLs.

collect.py's --tasks contract is {task_id, question, gold} per line; the P1
downloads are {id, question, answers, ...}. This helper merges N input files,
converts fields, and (optionally) samples a fixed-size subset with a fixed seed —
so collection/pilot/kill-switch task lists are reproducible and schema-correct.

Usage:
  make_task_file.py --in a.jsonl b.jsonl --out tasks.jsonl [--n 1200] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=None, help="sample size (default: all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    for p in args.inputs:
        for i, line in enumerate(p.open()):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            answers = r.get("answers") or ([r["gold"]] if "gold" in r else [])
            if not r.get("question"):
                continue
            rows.append({
                "task_id": f"{p.stem}:{r.get('id', r.get('task_id', i))}",
                "question": str(r["question"]),
                "gold": str(answers[0]) if answers else "",
                "golds": [str(a) for a in answers],   # full alias set for F1/EM scoring
                "source": p.stem,
            })
    if not rows:
        print("no usable rows", file=sys.stderr)
        return 1
    if args.n is not None and len(rows) > args.n:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.n]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{args.out}: {len(rows)} tasks from {len(args.inputs)} file(s) (seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
