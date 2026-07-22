#!/usr/bin/env python3
"""P1 decontamination pass (paper_plan_v2 §5.6 contamination protocol, step 1; §16 P1).

Checks every TRAIN prompt against every EVAL set with two detectors and drops hits:

  1. exact 13-gram overlap (word-level, normalized text) — the standard n-gram
     decontamination used by LLM training pipelines;
  2. MinHash (128 permutations over word 3-gram shingles) with LSH banding
     (16 bands x 8 rows) — near-duplicate detection at Jaccard >= ~0.5.

Outputs:
  * a JSON report (per train-set x eval-set hit counts + the dropped ids);
  * cleaned train files `<name>.decontaminated.jsonl` (only when --drop is given).

CPU-only, stdlib-only (no datasketch dependency).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

NGRAM = 13           # 13-gram exact overlap (task spec / standard practice)
SHINGLE = 3          # word shingles for MinHash
NUM_PERM = 128
BANDS, ROWS = 16, 8  # 16*8 = 128; LSH threshold ~ (1/16)^(1/8) ~ 0.71 candidate Jaccard
JACCARD_MIN = 0.5    # verified-Jaccard threshold for flagging a near-duplicate

_norm_re = re.compile(r"[^a-z0-9 ]+")


def normalize(text: str) -> list[str]:
    return _norm_re.sub(" ", text.lower()).split()


def ngrams(tokens: list[str], n: int) -> set[str]:
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _h64(s: str, seed: int) -> int:
    d = hashlib.blake2b(s.encode(), digest_size=8, salt=struct.pack("<q", seed)).digest()
    return struct.unpack("<Q", d)[0]


def minhash(shingles: set[str]) -> list[int] | None:
    if not shingles:
        return None
    return [min(_h64(s, p) for s in shingles) for p in range(NUM_PERM)]


def jaccard_est(a: list[int], b: list[int]) -> float:
    return sum(x == y for x, y in zip(a, b)) / NUM_PERM


def read_prompts(path: Path) -> list[tuple[str, str]]:
    """-> [(id, prompt_text)]; tolerant to the field names used across our JSONL files."""
    rows = []
    for i, line in enumerate(path.open()):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        text = r.get("question") or r.get("prompt") or r.get("problem") or r.get("text") or ""
        rows.append((str(r.get("id", i)), str(text)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True, type=Path, help="train prompt JSONL files")
    ap.add_argument("--eval", nargs="+", required=True, type=Path, help="eval set JSONL files")
    ap.add_argument("--report", type=Path, required=True, help="output JSON report path")
    ap.add_argument("--drop", action="store_true",
                    help="write <train>.decontaminated.jsonl with hits removed")
    args = ap.parse_args()

    # ---- index eval sets ----------------------------------------------------
    eval_ngrams: dict[str, set[str]] = {}
    lsh_buckets: dict[str, dict[tuple[int, bytes], list[int]]] = {}
    eval_sigs: dict[str, list[list[int] | None]] = {}
    for ep in args.eval:
        name = ep.stem
        grams, sigs, buckets = set(), [], defaultdict(list)
        for j, (_id, text) in enumerate(read_prompts(ep)):
            toks = normalize(text)
            grams |= ngrams(toks, NGRAM)
            sig = minhash(ngrams(toks, SHINGLE))
            sigs.append(sig)
            if sig:
                for b in range(BANDS):
                    key = (b, hashlib.blake2b(
                        struct.pack(f"<{ROWS}Q", *sig[b * ROWS : (b + 1) * ROWS]),
                        digest_size=8).digest())
                    buckets[key].append(j)
        eval_ngrams[name], eval_sigs[name], lsh_buckets[name] = grams, sigs, buckets
        print(f"[index] eval {name}: {len(sigs)} prompts, {len(grams)} distinct {NGRAM}-grams")

    # ---- scan train sets ----------------------------------------------------
    report = {"params": {"ngram": NGRAM, "shingle": SHINGLE, "num_perm": NUM_PERM,
                         "bands": BANDS, "rows": ROWS, "jaccard_min": JACCARD_MIN},
              "train_sets": {}}
    exit_dirty = False
    for tp in args.train:
        tname = tp.stem
        hits: dict[str, dict] = {}
        rows = read_prompts(tp)
        for tid, text in rows:
            toks = normalize(text)
            tg = ngrams(toks, NGRAM)
            sig = minhash(ngrams(toks, SHINGLE))
            for ename in eval_ngrams:
                overlap = tg & eval_ngrams[ename]
                near = None
                if sig:
                    cands = set()
                    for b in range(BANDS):
                        key = (b, hashlib.blake2b(
                            struct.pack(f"<{ROWS}Q", *sig[b * ROWS : (b + 1) * ROWS]),
                            digest_size=8).digest())
                        cands.update(lsh_buckets[ename].get(key, []))
                    for j in cands:
                        esig = eval_sigs[ename][j]
                        if esig and jaccard_est(sig, esig) >= JACCARD_MIN:
                            near = j
                            break
                if overlap or near is not None:
                    hits.setdefault(tid, {"id": tid, "matched_eval_sets": [], "reasons": []})
                    hits[tid]["matched_eval_sets"].append(ename)
                    if overlap:
                        hits[tid]["reasons"].append(f"{ename}:13gram({len(overlap)})")
                    if near is not None:
                        hits[tid]["reasons"].append(f"{ename}:minhash(row {near})")
        n_hit = len(hits)
        report["train_sets"][tname] = {
            "n_prompts": len(rows), "n_contaminated": n_hit,
            "contamination_rate": n_hit / len(rows) if rows else 0.0,
            "hits": sorted(hits.values(), key=lambda h: h["id"]),
        }
        print(f"[scan ] train {tname}: {n_hit}/{len(rows)} contaminated prompts")
        if n_hit:
            exit_dirty = True
        if args.drop:
            hit_ids = set(hits)
            out = tp.parent / f"{tname}.decontaminated.jsonl"
            kept = 0
            with tp.open() as fin, out.open("w") as fout:
                for i, line in enumerate(fin):
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if str(r.get("id", i)) not in hit_ids:
                        fout.write(line + "\n")
                        kept += 1
            report["train_sets"][tname]["decontaminated_file"] = str(out)
            report["train_sets"][tname]["n_kept"] = kept
            print(f"[drop ] {out.name}: kept {kept}/{len(rows)}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"[done ] report -> {args.report}")
    # non-zero when hits were found but not dropped, so the runbook can't skip it silently
    return 1 if (exit_dirty and not args.drop) else 0


if __name__ == "__main__":
    raise SystemExit(main())
