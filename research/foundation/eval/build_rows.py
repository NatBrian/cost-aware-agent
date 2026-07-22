"""Episodes JSONL -> combined per-task rows CSV (feeds aggregate/gate/figures).

Usage: .venv/bin/python -m eval.build_rows --out rows.csv ep1.jsonl ep2.jsonl ...
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import load_config
from eval.metrics import check_utility_recompute, rows_from_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config()
    lam = cfg["economy"]["lambda"]
    df = pd.concat([rows_from_jsonl(p, lam) for p in args.inputs],
                   ignore_index=True)
    check_utility_recompute(df, lam)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} rows ({sorted(df.arm.unique())}) -> {args.out}")


if __name__ == "__main__":
    main()
