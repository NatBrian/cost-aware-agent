"""F1 — build the frozen foundation datasets + manifest.

Usage: .venv/bin/python scripts/f1_data.py

Rebuilt 2026-07-28 after the container wipe. Two changes from the original, both
forced and both logged in the manifest:

1. Sources are built HERE from the HuggingFace HotpotQA parquet, instead of read
   from rescued `hotpotqa_*.jsonl` in data_shared. The rescued train file was
   `*.decontaminated.jsonl`; it is gone and the code that produced it lives in
   the read-banned archive, so it cannot be reproduced. Decontamination is now
   done in this script, explicitly, against the frozen dev set.
2. dev-200 is recovered BY ID from the committed baseline CSV rather than
   re-sampled, so the surviving A0/A1/A2 numbers stay comparable and the gate is
   still evaluated on exactly the questions it was pre-registered on. train-300
   is re-derived and WILL differ from the lost run's — an accepted deviation.

Also produces the val-50 slice that `scripts/probe_policy_health.sh` and the
anti-overfitting policy both depend on and that nothing previously generated.
"""

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import FOUNDATION_ROOT, config_hash, git_hash, load_config, sha256_file
from collect.sampling import (assert_no_overlap, load_jsonl, stratified_sample,
                              strata_counts, write_jsonl)
from eval.qa_metrics import normalize

HF_HOTPOT_GLOB = os.environ.get(
    "HOTPOT_PARQUET_GLOB",
    "/mnt/src/liangsheng/cassi_foundation/hotpotqa/fullwiki/*.parquet")


def _answers(rec) -> list[str]:
    a = rec.get("answer")
    return [a] if isinstance(a, str) else list(a or [])


def build_sources(out_dir: Path) -> tuple[Path, Path]:
    """HF parquet -> {id, question, answers, level} jsonl for train and dev."""
    train_p, dev_p = out_dir / "hotpotqa_train.jsonl", out_dir / "hotpotqa_dev.jsonl"
    if train_p.exists() and dev_p.exists():
        return train_p, dev_p
    files = [p for p in glob.glob(HF_HOTPOT_GLOB) if "test-" not in p]
    if not files:
        raise FileNotFoundError(
            f"no HotpotQA parquet at {HF_HOTPOT_GLOB} — the fetch job writes it")
    split: dict[str, list[dict]] = {"train": [], "dev": []}
    for p in files:
        df = pd.read_parquet(p, columns=["id", "question", "answer", "level"])
        key = "train" if "train-" in Path(p).name else "dev"
        for rec in df.to_dict("records"):
            split[key].append({"id": rec["id"], "question": rec["question"],
                               "answers": _answers(rec), "level": rec["level"]})
    write_jsonl(split["train"], train_p)
    write_jsonl(split["dev"], dev_p)
    return train_p, dev_p


def recover_dev_ids() -> list[str] | None:
    """The 200 frozen dev task ids, read back out of the committed baseline CSV.

    This is the only surviving record of which questions the pre-registered gate
    was defined on — the data/ manifest went with the wipe.
    """
    csv = FOUNDATION_ROOT / "experiments/results/baselines/baseline_rows.csv"
    if not csv.exists():
        return None
    ids = sorted(pd.read_csv(csv).task_id.astype(str).unique())
    return ids or None


def main() -> None:
    cfg = load_config()
    d = cfg["data"]
    shared = (FOUNDATION_ROOT / d["data_dir"]).resolve()
    shared.mkdir(parents=True, exist_ok=True)
    train_src_p, dev_src_p = build_sources(shared)
    train_src, dev_src = load_jsonl(train_src_p), load_jsonl(dev_src_p)

    # ---- dev-200: recover by id, else fall back to a fresh stratified draw ----
    dev_ids = recover_dev_ids()
    if dev_ids:
        by_id = {r["id"]: r for r in dev_src}
        missing = [i for i in dev_ids if i not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} frozen dev ids absent from the "
                             f"source split (e.g. {missing[:3]}) — refusing to "
                             "silently evaluate on a different dev set")
        dev = [by_id[i] for i in dev_ids]
        dev_provenance = "recovered by id from experiments/results/baselines/baseline_rows.csv"
    else:
        dev = stratified_sample(dev_src, d["dev_size"], d["sampling_seed"],
                                d["stratify_field"])
        dev_provenance = "fresh stratified sample (no baseline CSV found)"
    if len(dev) != d["dev_size"]:
        raise SystemExit(f"dev has {len(dev)} rows, want {d['dev_size']}")

    # ---- decontaminate the train pool against the frozen dev set -------------
    dev_id_set = {r["id"] for r in dev}
    dev_q = {normalize(r["question"]) for r in dev}
    pool = [r for r in train_src
            if r["id"] not in dev_id_set and normalize(r["question"]) not in dev_q]
    dropped = len(train_src) - len(pool)

    # ---- train-300, then val-50 from what train-300 did not take -------------
    train = stratified_sample(pool, d["train_size"], d["sampling_seed"],
                              d["stratify_field"])
    train_ids = {r["id"] for r in train}
    remainder = [r for r in pool if r["id"] not in train_ids]
    val = stratified_sample(remainder, d["val_size"], d["sampling_seed"] + 1,
                            d["stratify_field"])
    assert_no_overlap(train, dev)
    assert_no_overlap(val, dev)
    assert_no_overlap(val, train)

    paths = {}
    for name, rows in (("train", train), ("dev", dev), ("val", val)):
        p = FOUNDATION_ROOT / d[f"{name}_file"]
        write_jsonl(rows, p)
        paths[name] = p

    manifest = {
        "git": git_hash(),
        "config_hash": config_hash(cfg),
        "sampling_seed": d["sampling_seed"],
        "rebuilt": "2026-07-28 after container wipe",
        "sources": {"train": {"path": str(train_src_p), "n": len(train_src)},
                    "dev": {"path": str(dev_src_p), "n": len(dev_src)},
                    "levels_from": "HF hotpotqa/hotpot_qa fullwiki parquet"},
        "dev_provenance": dev_provenance,
        "decontamination": {
            "method": "drop train rows whose id or normalized question matches "
                      "the frozen dev set (the rescued *.decontaminated.jsonl "
                      "and its generator were lost; this replaces it)",
            "dropped_from_train_pool": dropped,
            "pool_after": len(pool)},
        "deviation": "train-300 is RE-DERIVED and differs from the lost run's "
                     "train split; dev-200 is identical by id, so the "
                     "pre-registered gate is unaffected",
        "outputs": {name: {"path": d[f"{name}_file"], "n": len(rows),
                           "sha256": sha256_file(paths[name]),
                           "strata": strata_counts(rows)}
                    for name, rows in (("train", train), ("dev", dev), ("val", val))},
        "retrieval_index": {"path": cfg["retrieval"]["index_dir"]},
    }
    man_path = FOUNDATION_ROOT / d["manifest_file"]
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2))
    for name in ("train", "dev", "val"):
        o = manifest["outputs"][name]
        print(f"{name}={o['n']} {o['strata']}")
    print(f"decontamination dropped {dropped} train rows; manifest -> {man_path}")


if __name__ == "__main__":
    main()
