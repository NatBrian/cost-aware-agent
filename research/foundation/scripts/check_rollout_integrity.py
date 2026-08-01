"""Rollout integrity check — run before any round's data is trusted.

WHY THIS EXISTS (2026-08-01). Killing the parent `s4_s5_run.sh` did NOT kill its
children: `run_lambda_arm.sh` was reparented to init and kept running. When the
pipeline was relaunched, TWO complete collection trees ran concurrently and wrote
to the same `rollouts_shard*.jsonl` files. The result was 3800 lines containing
only 1915 unique (task_id, rollout, budget) triples — nearly every episode
duplicated.

Nothing downstream would have noticed. GRPO groups on (task_id, budget_B), so
duplicates silently inflate group sizes, double-count some trajectories in the
advantage baseline, and bias the update toward whichever episodes happened to be
written twice. The run would have completed and produced a number.

Checks:
  1. duplicate (task_id, rollout, budget_B) triples  -> concurrent writers
  2. episode count vs the expected tasks x G
  3. malformed JSON lines                            -> interleaved writes
  4. GRPO group sizes                                -> uneven groups

Usage:
  .venv/bin/python scripts/check_rollout_integrity.py experiments/results/train/trt_round1
  .venv/bin/python scripts/check_rollout_integrity.py --all
"""

import argparse
import collections
import glob
import json
import sys
from pathlib import Path


def check(round_dir: Path, expect: int | None = None) -> bool:
    files = sorted(glob.glob(str(round_dir / "rollouts_shard*.jsonl")))
    assembled = round_dir / "rollouts.jsonl"
    # Prefer the assembled file when it exists; otherwise read the shards. Never
    # both — that double-counts and was itself a bug in the progress heartbeat.
    srcs = [assembled] if assembled.exists() else [Path(f) for f in files]
    if not srcs:
        print(f"{round_dir.name}: no rollouts found")
        return False

    keys = collections.Counter()
    groups = collections.Counter()
    n = bad = 0
    for s in srcs:
        for line in open(s):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            n += 1
            keys[(e["task_id"], e.get("rollout", 0), e.get("budget_B"))] += 1
            groups[(e["task_id"], e.get("budget_B"))] += 1

    dup = [k for k, v in keys.items() if v > 1]
    gsz = collections.Counter(groups.values())
    ok = not dup and not bad and (expect is None or n == expect)

    print(f"{round_dir.name}: lines={n} unique={len(keys)} "
          f"duplicated={len(dup)} malformed_json={bad} "
          f"group_sizes={dict(sorted(gsz.items()))}  "
          f"{'OK' if ok else '*** FAIL ***'}")
    if dup:
        print(f"   e.g. {dup[0]} appears {keys[dup[0]]}x  "
              "-> concurrent writers; the round must be recollected")
    if bad:
        print(f"   {bad} unparseable lines -> interleaved writes")
    if expect is not None and n != expect:
        print(f"   expected {expect} episodes, found {n}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--expect", type=int, default=None,
                    help="expected episode count (tasks x G), e.g. 2400")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1] / "experiments/results/train"
    dirs = ([p for p in sorted(root.iterdir()) if p.is_dir()]
            if args.all else [Path(d) for d in args.dirs])
    if not dirs:
        ap.error("give round directories or --all")
    ok = all(check(d, args.expect) for d in dirs)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
