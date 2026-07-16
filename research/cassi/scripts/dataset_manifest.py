#!/usr/bin/env python3
"""P1 done-criterion artifact (paper_plan_v2 §16 P1):
"✅ Done: dataset manifest with counts + split hashes committed."

Walks research/cassi/data/*.jsonl, records row counts + sha256 per file, and writes
  * data/manifest.json                       (working copy, gitignored with data/)
  * experiments/results/dataset_manifest.csv (the COMMITTED copy — results CSVs are
    the one un-ignored path under experiments/, see .gitignore)

Exit 1 if any REQUIRED split is missing, so p1_data.sh can gate on it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CASSI_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = CASSI_ROOT / "data"
RESULTS_DIR = CASSI_ROOT / "experiments" / "results"

# required splits per §5.1/§11 (gaia + browsecomp-plus may be PENDING: gated/staged later)
REQUIRED = [
    "nq_train", "hotpotqa_train", "musique_train",
    "hotpotqa_dev", "musique_dev", "hotpotqa_dev_frozen", "musique_dev_frozen",
    "bamboogle", "2wikimultihopqa_dev", "math500", "aime2025",
]
OPTIONAL = ["gaia_dev_textonly"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not DATA_DIR.exists():
        print(f"no data dir at {DATA_DIR} — run scripts/p1_data.sh first", file=sys.stderr)
        return 1
    entries = []
    for p in sorted(DATA_DIR.glob("*.jsonl")):
        n = sum(1 for line in p.open() if line.strip())
        entries.append({"split": p.stem, "file": p.name, "n_rows": n, "sha256": sha256(p)})

    have = {e["split"] for e in entries}
    missing = [r for r in REQUIRED if r not in have]

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed_frozen_subsamples": 42,
        "entries": entries,
        "missing_required": missing,
        "notes": "counts + split hashes per paper_plan_v2 §16 P1; frozen subsamples chosen once (§5.6)",
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "dataset_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "file", "n_rows", "sha256"])
        w.writeheader()
        w.writerows(entries)

    print(f"{'split':32s} {'rows':>8s}  sha256[:12]")
    for e in entries:
        print(f"{e['split']:32s} {e['n_rows']:8d}  {e['sha256'][:12]}")
    print(f"\nmanifest -> {DATA_DIR/'manifest.json'} and {RESULTS_DIR/'dataset_manifest.csv'}")
    if missing:
        print(f"MISSING required splits: {missing}", file=sys.stderr)
        return 1
    absent_opt = [o for o in OPTIONAL if o not in have]
    if absent_opt:
        print(f"note: optional/gated splits not yet staged: {absent_opt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
