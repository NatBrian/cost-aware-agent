#!/usr/bin/env python3
"""P0 helper — write third_party commit hashes + installed lib versions into
configs/cassi.yaml `pins:` (paper_plan_v2 §16 P0: "pin every commit hash into §17";
§19: reused repos are cloned and version-pinned).

Comment-preserving on purpose: a PyYAML round-trip would drop every §17 comment in
cassi.yaml, so we rewrite ONLY the value part of the `  <key>: ...` lines inside the
`pins:` block (the last top-level block of the file).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CASSI_ROOT = Path(__file__).resolve().parent.parent
CONFIG = CASSI_ROOT / "configs" / "cassi.yaml"
THIRD_PARTY = CASSI_ROOT / "third_party"

# pins key -> third_party clone dir (paper_plan_v2 §19 reuse table)
REPO_DIRS = {
    "verl": "verl",
    "verl_tool": "verl-tool",
    "verl_agent": "verl-agent",
    "search_r1": "Search-R1",
}
# pins key -> installed python package (§19 stack pins for Qwen3.5)
PKG_NAMES = {"trl": "trl", "transformers": "transformers", "vllm": "vllm"}


def git_head(repo: Path) -> str | None:
    if not (repo / ".git").exists():
        return None
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def main() -> int:
    pins: dict[str, str | None] = {}
    for key, dirname in REPO_DIRS.items():
        pins[key] = git_head(THIRD_PARTY / dirname)
    for key, pkg in PKG_NAMES.items():
        pins[key] = pkg_version(pkg)

    text = CONFIG.read_text()
    m = re.search(r"^pins:.*$", text, flags=re.M)
    if m is None:
        print("ERROR: no `pins:` block in configs/cassi.yaml", file=sys.stderr)
        return 1
    head, block = text[: m.start()], text[m.start() :]

    n_written, missing = 0, []
    for key, val in pins.items():
        if val is None:
            missing.append(key)
            continue
        # `  verl: null            # >= v0.8.0`  ->  `  verl: <hash>            # >= v0.8.0`
        pattern = rf"^(\s+{re.escape(key)}:)\s*[^#\n]*?(\s*(?:#.*)?)$"
        block, n = re.subn(pattern, rf"\g<1> {val}\g<2>", block, count=1, flags=re.M)
        if n:
            n_written += 1
            print(f"[pins] {key}: {val}")
        else:
            print(f"[pins] WARNING: key '{key}' not found in pins block", file=sys.stderr)

    CONFIG.write_text(head + block)
    print(f"[pins] wrote {n_written} pin(s) into {CONFIG}")
    if missing:
        print(
            f"[pins] NOT resolved (clone/install first, then rerun): {', '.join(missing)}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
