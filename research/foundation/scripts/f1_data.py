"""F1 — build the frozen foundation datasets + manifest.

Usage: .venv/bin/python scripts/f1_data.py
Reads rescued sources in research/data_shared/, levels from the offline HF
parquet cache; writes foundation/data/{train_300,dev_200}.jsonl + manifest.json.
Re-running with the same config reproduces byte-identical outputs.
"""

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import FOUNDATION_ROOT, config_hash, git_hash, load_config, sha256_file
from collect.sampling import (assert_no_overlap, attach_levels, load_jsonl,
                              stratified_sample, strata_counts, write_jsonl)

HF_HOTPOT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--hotpotqa--hotpot_qa/snapshots/*/fullwiki/*.parquet")


def build_level_map() -> dict[str, str]:
    files = [p for p in glob.glob(HF_HOTPOT_GLOB) if "test-" not in p]
    if not files:
        raise FileNotFoundError(f"no cached HotpotQA parquet at {HF_HOTPOT_GLOB}")
    level_by_id: dict[str, str] = {}
    for p in files:
        df = pd.read_parquet(p, columns=["id", "level"])
        level_by_id.update(zip(df["id"], df["level"]))
    return level_by_id


def main() -> None:
    cfg = load_config()
    d = cfg["data"]
    level_by_id = build_level_map()

    train_src = attach_levels(load_jsonl(FOUNDATION_ROOT / d["source_train"]), level_by_id)
    dev_src = attach_levels(load_jsonl(FOUNDATION_ROOT / d["source_dev"]), level_by_id)

    train = stratified_sample(train_src, d["train_size"], d["sampling_seed"],
                              d["stratify_field"])
    dev = stratified_sample(dev_src, d["dev_size"], d["sampling_seed"],
                            d["stratify_field"])
    assert_no_overlap(train, dev)

    train_path = FOUNDATION_ROOT / d["train_file"]
    dev_path = FOUNDATION_ROOT / d["dev_file"]
    write_jsonl(train, train_path)
    write_jsonl(dev, dev_path)

    manifest = {
        "git": git_hash(),
        "config_hash": config_hash(cfg),
        "sampling_seed": d["sampling_seed"],
        "sources": {
            "train": {"path": d["source_train"], "n": len(train_src)},
            "dev": {"path": d["source_dev"], "n": len(dev_src)},
            "levels_from": "HF cache hotpotqa/hotpot_qa fullwiki parquet",
        },
        "outputs": {
            "train": {"path": d["train_file"], "n": len(train),
                      "sha256": sha256_file(train_path),
                      "strata": strata_counts(train)},
            "dev": {"path": d["dev_file"], "n": len(dev),
                    "sha256": sha256_file(dev_path),
                    "strata": strata_counts(dev)},
        },
        "overlap_check": "passed (id + normalized question text)",
        "retrieval_index": {
            "path": cfg["retrieval"]["index_dir"],
            "files": {p.name: p.stat().st_size for p in sorted(
                (FOUNDATION_ROOT / cfg["retrieval"]["index_dir"]).glob("*")) if p.is_file()},
        },
    }
    man_path = FOUNDATION_ROOT / d["manifest_file"]
    man_path.parent.mkdir(parents=True, exist_ok=True)
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"train={len(train)} {manifest['outputs']['train']['strata']}")
    print(f"dev={len(dev)} {manifest['outputs']['dev']['strata']}")
    print(f"manifest -> {man_path}")


if __name__ == "__main__":
    main()
