"""Stratified, seeded question sampling (F1).

Pure functions so tests run on synthetic data; scripts/f1_data.py does the I/O.
"""

import json
import random
from collections import Counter
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def attach_levels(rows: list[dict], level_by_id: dict[str, str]) -> list[dict]:
    return [{**r, "level": level_by_id.get(r["id"], "unknown")} for r in rows]


def stratified_sample(rows: list[dict], n: int, seed: int,
                      field: str = "level") -> list[dict]:
    """Proportional allocation per stratum, deterministic under (rows, n, seed).

    Rounding remainders are assigned to the largest strata first; output order
    is sorted by id so file bytes are reproducible.
    """
    if n > len(rows):
        raise ValueError(f"asked for {n} of {len(rows)} rows")
    strata: dict[str, list[dict]] = {}
    for r in rows:
        strata.setdefault(r.get(field, "unknown"), []).append(r)
    sizes = {k: len(v) for k, v in strata.items()}
    total = len(rows)
    alloc = {k: (n * s) // total for k, s in sizes.items()}
    short = n - sum(alloc.values())
    for k in sorted(sizes, key=sizes.get, reverse=True)[:short]:
        alloc[k] += 1
    rng = random.Random(seed)
    out: list[dict] = []
    for k in sorted(strata):
        pool = sorted(strata[k], key=lambda r: r["id"])
        out.extend(rng.sample(pool, min(alloc[k], len(pool))))
    return sorted(out, key=lambda r: r["id"])


def strata_counts(rows: list[dict], field: str = "level") -> dict[str, int]:
    return dict(Counter(r.get(field, "unknown") for r in rows))


def assert_no_overlap(a: list[dict], b: list[dict]) -> None:
    """Fail loudly if any id or normalized question text appears in both sets."""
    ids = {r["id"] for r in a} & {r["id"] for r in b}
    if ids:
        raise AssertionError(f"id overlap between splits: {sorted(ids)[:5]}")
    norm = lambda q: " ".join(q.lower().split())
    qs = {norm(r["question"]) for r in a} & {norm(r["question"]) for r in b}
    if qs:
        raise AssertionError(f"question-text overlap between splits: {sorted(qs)[:3]}")
