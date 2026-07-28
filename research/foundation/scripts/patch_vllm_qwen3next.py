"""Force vLLM's Qwen3.5 gated-delta-rule onto its Triton path (idempotent).

WHY THIS EXISTS
Qwen3.5's `ChunkGatedDeltaRule` picks its implementation in `__init__`:

    if current_platform.is_cuda() and current_platform.is_device_capability(90):
        self._forward_method = self.forward_cuda      # flashinfer gdn_prefill
    else:
        self._forward_method = self.forward_native    # vLLM's own Triton FLA

It assigns `_forward_method` directly, which BYPASSES `CustomOp.enabled()` — so
`--compilation-config '{"custom_ops":["-chunk_gated_delta_rule"]}'` has no effect
(verified 2026-07-28: the engine still died in flashinfer). Our cards are compute
capability 9.0, so the check always selects flashinfer, whose kernels JIT-compile
and therefore need nvcc. This box has no /usr/local/cuda, and neither pip nvcc
wheel is usable: the 12.9 one ships nvcc 13.2 with clashing headers, the 12.8 one
ships no nvcc at all. The Triton path self-compiles and needs none of it.

The first run reached the same fix (commit 28b2728) by hand-editing this file and
documenting it "in-file" — inside a gitignored venv, so it died in the 2026-07-28
container wipe and cost a rediscovery. This script is that fix, in git.

Usage (after any venv rebuild):  .venv/bin/python scripts/patch_vllm_qwen3next.py
"""

import sys
from pathlib import Path

MARKER = "# PATCHED-FOUNDATION: force Triton FLA path (no nvcc on this box)"
TARGET = ("        if current_platform.is_cuda() and "
          "current_platform.is_device_capability(90):")
REPLACEMENT = (f"        {MARKER}\n"
               "        if False:  # was: is_cuda() and is_device_capability(90)")


def find_qwen3_next(venv: Path) -> Path:
    hits = list(venv.glob("lib/python*/site-packages/vllm/model_executor/models/qwen3_next.py"))
    if not hits:
        raise SystemExit(f"qwen3_next.py not found under {venv} — is vllm installed?")
    return hits[0]


def main() -> None:
    venv = Path(sys.argv[1] if len(sys.argv) > 1 else ".venv-gpu3")
    path = find_qwen3_next(venv)
    src = path.read_text()
    if MARKER in src:
        print(f"already patched: {path}")
        return
    if TARGET not in src:
        raise SystemExit(
            f"expected dispatch line not found in {path}.\n"
            "vLLM's implementation changed — re-read ChunkGatedDeltaRule.__init__ "
            "and update TARGET rather than forcing this through.")
    path.write_text(src.replace(TARGET, REPLACEMENT, 1))
    print(f"patched: {path}\n  -> ChunkGatedDeltaRule now uses forward_native "
          "(vLLM Triton FLA), never flashinfer's JIT")


if __name__ == "__main__":
    main()
