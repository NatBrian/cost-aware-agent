#!/usr/bin/env python3
"""P1 data staging (paper_plan_v2 §5.1, §11, §16 P1, §19 datasets table).

Downloads via HF `datasets` and normalizes everything to JSONL rows
{id, question, answers, source, split} under research/cassi/data/:

  train  : NQ + HotpotQA (sample 8-10K combined -> 4500+4500), MuSiQue (5000)
  dev    : HotpotQA dev, MuSiQue dev (+ FROZEN eval subsamples: 1000 / 500, seed 42,
           chosen ONCE before any method runs — §5.6; never resampled if present)
  eval   : Bamboogle (125), 2WikiMultihopQA (500-dev), MATH-500, AIME 2025,
           GAIA text-only 103-Q dev (gated — needs `huggingface-cli login` + accepted terms)

BrowseComp-Plus staging and the Search-R1 wiki index are documented in
scripts/p1_data.sh (they are corpus builds, not HF-dataset pulls).

HF dataset ids marked TODO(P1-verify) are best-known candidates tried in order;
verify against the §19 manifest re-check before submission.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

CASSI_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = CASSI_ROOT / "data"
SEED = 42  # §5.6 frozen subsamples; headline seed list [42,123,789]

sys.path.insert(0, str(CASSI_ROOT.parent))
from cassi.common.config import load_config  # noqa: E402


# ------------------------------------------------------------------ normalizers
def _aslist(a):
    if a is None:
        return []
    return [str(x) for x in a] if isinstance(a, (list, tuple)) else [str(a)]


def _std(name):  # question/answer(s) under standard-ish keys
    def f(row, i):
        q = row.get("question") or row.get("Question") or row.get("problem") or row.get("query")
        a = row.get("answers") or row.get("answer") or row.get("Answer")
        if isinstance(a, dict):  # e.g. HF answers structs {"text": [...]}
            a = a.get("text") or a.get("answer") or list(a.values())[0]
        return {"id": str(row.get("id", row.get("_id", i))), "question": str(q), "answers": _aslist(a)}

    return f


def _gaia(row, i):
    return {
        "id": str(row.get("task_id", i)),
        "question": str(row.get("Question")),
        "answers": _aslist(row.get("Final answer")),
        "level": row.get("Level"),
        "file_name": row.get("file_name", ""),
    }


# ---------------------------------------------------------------------- registry
# (out_name, candidate (hf_id, config) list, split, normalizer, sample_n or None)
def build_registry(cfg):
    n_combined = cfg["data"]["qa_train"]["nq_hotpotqa_combined"]  # 9000 (§11 / §17)
    n_musique = cfg["data"]["qa_train"]["musique"]  # 5000
    return [
        # --- training corpus: the Search-R1/OTC-PO mix (§5.1) ---
        ("nq_train", [("google-research-datasets/nq_open", None)], "train", _std("nq"), n_combined // 2),
        ("hotpotqa_train", [("hotpotqa/hotpot_qa", "fullwiki")], "train", _std("hotpotqa"), n_combined - n_combined // 2),
        ("musique_train", [("dgslibisey/MuSiQue", None)], "train", _std("musique"), n_musique),  # TODO(P1-verify) hf id
        # --- dev sets (frozen subsamples cut below) ---
        ("hotpotqa_dev", [("hotpotqa/hotpot_qa", "fullwiki")], "validation", _std("hotpotqa"), None),
        ("musique_dev", [("dgslibisey/MuSiQue", None)], "validation", _std("musique"), None),  # TODO(P1-verify)
        # --- OOD / control evals (§5.1) ---
        ("bamboogle", [("chiayewken/bamboogle", None)], "test", _std("bamboogle"), None),  # TODO(P1-verify)
        ("2wikimultihopqa_dev",
         [("framolfese/2WikiMultihopQA", None), ("xanhho/2WikiMultihopQA", None)],
         "validation", _std("2wiki"), 500),  # 500-dev per §5.1  TODO(P1-verify) hf id
        ("math500", [("HuggingFaceH4/MATH-500", None)], "test", _std("math500"), None),
        ("aime2025",
         [("math-ai/aime25", None), ("opencompass/AIME2025", "AIME2025-I"), ("yentinglin/aime_2025", None)],
         "test", _std("aime"), None),  # TODO(P1-verify) hf id
        # --- GAIA text-only dev (gated; val-used-as-test is DISCLOSED, §5.1) ---
        ("gaia_dev_textonly", [("gaia-benchmark/GAIA", "2023_all")], "validation", _gaia, None),
    ]


def load_first(candidates, split):
    from datasets import load_dataset

    errs = []
    for hf_id, conf in candidates:
        try:
            return load_dataset(hf_id, conf, split=split), hf_id
        except Exception as e:  # gated / renamed / missing split
            errs.append(f"{hf_id}({conf}): {type(e).__name__}: {e}")
    raise RuntimeError("all candidates failed:\n  " + "\n  ".join(errs))


def write_jsonl(rows, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def freeze_subsample(src: Path, dst: Path, n: int) -> None:
    """§5.6: frozen eval subsamples are chosen ONCE, before any method runs."""
    if dst.exists():
        print(f"[frozen] {dst.name} already exists — NOT resampling (frozen, §5.6)")
        return
    rows = [json.loads(l) for l in src.open()]
    random.Random(SEED).shuffle(rows)
    write_jsonl(rows[:n], dst)
    print(f"[frozen] {dst.name}: {min(n, len(rows))} tasks (seed {SEED}, chosen once)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of out_names to (re)download")
    args = ap.parse_args()

    cfg = load_config()
    registry = build_registry(cfg)
    failures = []

    for out_name, candidates, split, norm, sample_n in registry:
        if args.only and out_name not in args.only:
            continue
        out = DATA_DIR / f"{out_name}.jsonl"
        if out.exists():
            print(f"[skip] {out_name}: {out} exists")
            continue
        try:
            ds, used = load_first(candidates, split)
        except RuntimeError as e:
            tag = "PENDING (gated: huggingface-cli login + accept terms on the hub)" \
                if out_name.startswith("gaia") else "FAILED"
            print(f"[{tag}] {out_name}:\n{e}", file=sys.stderr)
            failures.append(out_name)
            continue
        rows = [norm(row, i) | {"source": used, "split": split} for i, row in enumerate(ds)]
        if out_name == "gaia_dev_textonly":
            rows = [r for r in rows if not r.get("file_name")]  # text-only subset (§5.1)
            if len(rows) != 103:
                print(f"[warn] GAIA text-only dev has {len(rows)} rows, plan expects 103 (§5.1) — verify filter")
        if sample_n is not None and len(rows) > sample_n:
            random.Random(SEED).shuffle(rows)
            rows = rows[:sample_n]
        n = write_jsonl(rows, out)
        print(f"[done] {out_name}: {n} rows <- {used} [{split}]")

    # frozen eval subsamples (§5.6 / §17 data.eval_frozen_subsamples)
    fs = cfg["data"]["eval_frozen_subsamples"]
    if (DATA_DIR / "hotpotqa_dev.jsonl").exists():
        freeze_subsample(DATA_DIR / "hotpotqa_dev.jsonl",
                         DATA_DIR / "hotpotqa_dev_frozen.jsonl", fs["hotpotqa_dev"])
    if (DATA_DIR / "musique_dev.jsonl").exists():
        freeze_subsample(DATA_DIR / "musique_dev.jsonl",
                         DATA_DIR / "musique_dev_frozen.jsonl", fs["musique_dev"])

    if failures:
        print(f"\nIncomplete: {failures} — rerun with --only {' '.join(failures)} after fixing.",
              file=sys.stderr)
        return 1
    print("\nAll HF datasets staged. ALFWorld task lists come with verl-agent (P0 clone);")
    print("wiki index + BrowseComp-Plus corpus: see scripts/p1_data.sh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
