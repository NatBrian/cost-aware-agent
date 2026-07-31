"""S2b — build FOUNDATION-2's frozen evaluation set (eval-600).

WHY A NEW SET. S2's power analysis says the primary estimand (paired Δsteps at
B=2) needs **n >= 479** to resolve the achievable 0.238-step effect at the
pre-registered threshold. FOUNDATION-1's dev-200 is less than half of that, so
reusing it would guarantee an unresolvable result — the exact failure that
produced the original null, where n=50 gave a ±0.220 CI against a 0.164 effect.

WHY DISJOINT rather than a superset of dev-200. FOUNDATION-2 changes the budgets
({2,4,8} -> {2,3,4}), so FOUNDATION-1's dev-200 rows are not comparable to the
new arms anyway — the reuse argument buys nothing. A disjoint draw additionally
removes any question of adaptive contamination from the three looks dev-200 has
already taken, and 7,405 HotpotQA dev questions make the draw free.

Guarantees:
  - stratified and seeded, so the file is byte-reproducible
  - no overlap with train-300 (by id AND by normalized question text)
  - no overlap with dev-200 or val-50
  - SHA256 recorded in the manifest alongside the existing sets

Usage: .venv/bin/python scripts/s2_build_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import FOUNDATION_ROOT, git_hash, load_config, sha256_file
from collect.sampling import load_jsonl, stratified_sample, strata_counts, write_jsonl
from eval.qa_metrics import normalize

EVAL_SIZE = 600          # n >= 479 required by S2 power; 600 leaves margin
EVAL_FILE = "data/hotpotqa_eval_600.jsonl"


def main() -> None:
    cfg = load_config()
    d = cfg["data"]
    root = FOUNDATION_ROOT
    src = root / d["data_dir"] / "hotpotqa_dev.jsonl"
    if not src.exists():
        raise SystemExit(f"missing source {src} — run scripts/f1_data.py first")

    dev_src = load_jsonl(src)
    used_ids, used_qs = set(), set()
    for name in ("train_file", "dev_file", "val_file"):
        p = root / d[name]
        if p.exists():
            for r in load_jsonl(p):
                used_ids.add(r["id"])
                used_qs.add(normalize(r["question"]))
    print(f"source dev pool: {len(dev_src)}")
    print(f"excluded (train-300 / dev-200 / val-50): {len(used_ids)} ids")

    pool = [r for r in dev_src
            if r["id"] not in used_ids and normalize(r["question"]) not in used_qs]
    print(f"eligible after decontamination: {len(pool)}")
    if len(pool) < EVAL_SIZE:
        raise SystemExit(f"only {len(pool)} eligible, need {EVAL_SIZE}")

    ev = stratified_sample(pool, EVAL_SIZE, d["sampling_seed"] + 7)
    out = root / EVAL_FILE
    write_jsonl(ev, out)

    # hard assertions rather than trust
    ids = {r["id"] for r in ev}
    qs = {normalize(r["question"]) for r in ev}
    assert len(ids) == EVAL_SIZE, "duplicate ids in eval set"
    assert not (ids & used_ids), "eval overlaps an existing split by id"
    assert not (qs & used_qs), "eval overlaps an existing split by question text"

    sha = sha256_file(out)
    print(f"\nwrote {out}  n={len(ev)}  sha256={sha[:16]}…")
    print(f"strata: {strata_counts(ev)}")

    man_p = root / d["manifest_file"]
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    man.setdefault("files", {})[EVAL_FILE] = {
        "n": len(ev), "sha256": sha, "seed": d["sampling_seed"] + 7,
        "purpose": "FOUNDATION-2 frozen evaluation set (S2 power: n>=479)",
        "disjoint_from": ["train_file", "dev_file", "val_file"],
        "git": git_hash(),
    }
    man_p.write_text(json.dumps(man, indent=2))
    print(f"manifest updated: {man_p}")


if __name__ == "__main__":
    main()
