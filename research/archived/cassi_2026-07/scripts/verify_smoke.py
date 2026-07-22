#!/usr/bin/env python3
"""P0 done-criterion checker (paper_plan_v2 §16 P0).

Verifies, for each smoke trajectory JSONL:
  * the file parses into the §11 Trajectory schema;
  * the running-draft template line is present in EVERY step (§2.6/§18.2 —
    draft field set; EMPTY_DRAFT allowed only while no answer exists yet);
  * per-step dollar cost is logged (c > 0 on every step — draft-line tokens are
    inside c_t, so a zero-cost step means cost logging is broken).

Exit 0 iff all checks pass on all files.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # research/ -> import cassi.*

from cassi.common.schema import load_trajectories  # noqa: E402


def check(path: str) -> bool:
    trajs = list(load_trajectories(path))
    if not trajs:
        print(f"[FAIL] {path}: no trajectories")
        return False
    ok = True
    for tr in trajs:
        if not tr.steps:
            print(f"[FAIL] {path} task={tr.task_id}: zero steps")
            ok = False
            continue
        missing_draft = [s.x.step_idx for s in tr.steps if not s.draft]
        zero_cost = [s.x.step_idx for s in tr.steps if not s.c > 0.0]
        if missing_draft:
            print(f"[FAIL] {path} task={tr.task_id}: draft line missing at steps {missing_draft}")
            ok = False
        if zero_cost:
            print(f"[FAIL] {path} task={tr.task_id}: per-step cost not logged (c<=0) at steps {zero_cost}")
            ok = False
        if ok:
            print(
                f"[ ok ] {path} task={tr.task_id}: {len(tr.steps)} steps, "
                f"total ${sum(s.c for s in tr.steps):.4f}, drafts present"
            )
    return ok


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: verify_smoke.py <trajectories.jsonl> [...]", file=sys.stderr)
        return 2
    all_ok = all(check(p) for p in sys.argv[1:])
    print("P0 done-criterion:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
