"""Shared plumbing: config loading and run stamping.

Every module gets constants from configs/foundation.yaml through load_config() —
nothing else may define experiment constants (foundation.yaml header rule).
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

FOUNDATION_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = FOUNDATION_ROOT / "configs" / "foundation.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    """Stable short hash of the whole config — stamped into every output row."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=FOUNDATION_ROOT, check=True,
        ).stdout.strip()
    except Exception:
        return "nogit"


def write_run_stamp(out_dir: str | Path, cfg: dict, extra: dict | None = None) -> Path:
    """Config snapshot + git hash next to every run's outputs (plan §8)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = {"git": git_hash(), "config_hash": config_hash(cfg), "config": cfg}
    if extra:
        stamp.update(extra)
    p = out_dir / "run_stamp.json"
    with open(p, "w") as f:
        json.dump(stamp, f, indent=2, default=str)
    return p


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
