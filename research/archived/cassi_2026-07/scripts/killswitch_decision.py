#!/usr/bin/env python3
"""P5 GO/NO-GO computation (paper_plan_v2 §12 kill-switch protocol; §16 P5).

Reads the K1/K2 frontier CSVs and applies the §12 thresholds:

  K1 (bridge test):  GO iff Δ-shaped GRPO beats controller-only by >= 3 points
      cost-at-iso-accuracy AND is <= B9-direct-shaping's cost at iso-accuracy.
      (1 seed — read as direction + magnitude, not significance; §12.)
  K2 (separation test): any outcome is publishable via H4 — logged, never gating.

CSV contract (produced by the p5 eval calls):
  k1_frontier.csv: arm,lambda_dial,accuracy,cost_dollars   arm in {shaped, controller_only, b9}
  k2_frontier.csv: arm,lambda_dial,accuracy,cost_dollars   arm in {single_multitask, two_model}

Iso-accuracy cost is read by linear interpolation between adjacent frontier points
(§5.3 frontier protocol); single-point arms are compared at their own accuracy.

Appends a dated decision to research/cassi/GO_NO_GO.log (append-only — §5.6
no-cherry-picking clause). Exit 0 = GO, 1 = NO-GO, 75 = inputs missing.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

CASSI_ROOT = Path(__file__).resolve().parent.parent
KS_DIR = CASSI_ROOT / "experiments" / "killswitch" / "results"
LOG = CASSI_ROOT / "GO_NO_GO.log"

K1_MIN_POINTS = 3.0  # §12: ">= 3 points cost-at-iso-accuracy over controller-only"


def read_frontier(path: Path) -> dict[str, list[tuple[float, float]]]:
    """-> arm -> [(accuracy, cost)...] sorted by accuracy."""
    arms: dict[str, list[tuple[float, float]]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            arms.setdefault(row["arm"], []).append(
                (float(row["accuracy"]), float(row["cost_dollars"])))
    for a in arms:
        arms[a].sort()
    return arms


def iso_cost(frontier: list[tuple[float, float]], target_acc: float) -> float | None:
    """Cost at target accuracy by linear interpolation between adjacent frontier
    points (§5.3). None if target is outside the frontier's accuracy range."""
    if len(frontier) == 1:
        return frontier[0][1] if abs(frontier[0][0] - target_acc) < 1e-9 else None
    for (a0, c0), (a1, c1) in zip(frontier, frontier[1:]):
        if min(a0, a1) - 1e-12 <= target_acc <= max(a0, a1) + 1e-12:
            if abs(a1 - a0) < 1e-12:
                return min(c0, c1)
            w = (target_acc - a0) / (a1 - a0)
            return c0 + w * (c1 - c0)
    return None


def main() -> int:
    k1p, k2p = KS_DIR / "k1_frontier.csv", KS_DIR / "k2_frontier.csv"
    if not k1p.exists():
        print(f"PENDING: {k1p} missing — run the K1 arms first", file=sys.stderr)
        return 75
    k1 = read_frontier(k1p)
    for arm in ("shaped", "controller_only", "b9"):
        if arm not in k1:
            print(f"PENDING: K1 arm '{arm}' missing from {k1p}", file=sys.stderr)
            return 75

    # reference operating point: shaped headline (its median-accuracy frontier point)
    shaped = k1["shaped"]
    ref_acc, ref_cost = shaped[len(shaped) // 2]

    lines = [f"K1 reference: shaped headline point accuracy={ref_acc:.4f} cost=${ref_cost:.4f}"]
    ctrl_cost = iso_cost(k1["controller_only"], ref_acc)
    if ctrl_cost is None:
        # frontiers don't overlap at ref_acc — fall back to controller's own headline point
        ctrl_pts = k1["controller_only"]
        c_acc, c_cost = ctrl_pts[len(ctrl_pts) // 2]
        shaped_at_ctrl = iso_cost(shaped, c_acc)
        lines.append(f"NOTE: controller frontier does not span shaped's accuracy — "
                     f"comparing at controller's headline accuracy {c_acc:.4f} instead")
        if shaped_at_ctrl is None:
            lines.append("K1 UNDECIDABLE: frontiers do not overlap — collect more dial points")
            verdict = "NO-GO (undecidable — frontiers disjoint; extend the λ dial sweep)"
            gain_pts = float("nan")
            beats_b9 = False
        else:
            gain_pts = (c_cost - shaped_at_ctrl) / c_cost * 100.0
            ref_acc, ref_cost = c_acc, shaped_at_ctrl
            ctrl_cost = c_cost
    if ctrl_cost is not None:
        gain_pts = (ctrl_cost - ref_cost) / ctrl_cost * 100.0
        lines.append(f"controller-only cost at iso-accuracy: ${ctrl_cost:.4f} -> "
                     f"shaped saves {gain_pts:.2f} points (threshold >= {K1_MIN_POINTS})")
        # vs B9 at iso-accuracy (B9 may be a single point)
        b9 = k1["b9"]
        b9_at_ref = iso_cost(b9, ref_acc)
        if b9_at_ref is not None:
            beats_b9 = ref_cost <= b9_at_ref
            lines.append(f"B9 cost at iso-accuracy: ${b9_at_ref:.4f} -> shaped <= B9: {beats_b9}")
        else:
            b_acc, b_cost = b9[len(b9) // 2]
            shaped_at_b9 = iso_cost(shaped, b_acc)
            beats_b9 = shaped_at_b9 is not None and shaped_at_b9 <= b_cost
            lines.append(f"B9 single point (acc={b_acc:.4f}, ${b_cost:.4f}); shaped at that "
                         f"accuracy: {shaped_at_b9} -> shaped <= B9: {beats_b9}")
        go = gain_pts >= K1_MIN_POINTS and beats_b9
        verdict = "GO" if go else "NO-GO (pivot per H2/H3 fallback framings, §6)"

    # K2 — logged, never gating (§12: "any outcome is publishable via H4")
    if k2p.exists():
        k2 = read_frontier(k2p)
        if {"single_multitask", "two_model"} <= set(k2):
            sm = k2["single_multitask"][len(k2["single_multitask"]) // 2]
            tm = k2["two_model"][len(k2["two_model"]) // 2]
            sep = iso_cost(k2["single_multitask"], tm[0])
            lines.append(f"K2: two_model (acc={tm[0]:.4f}, ${tm[1]:.4f}) vs single_multitask "
                         f"(acc={sm[0]:.4f}, ${sm[1]:.4f}); single at two_model's accuracy: {sep}")
            lines.append("K2 framing: two-model wins -> Pareto claim stands; parity/loss -> "
                         "H4 fallback (transfer + privileged-info hygiene + controllability)")
        else:
            lines.append("K2: arms incomplete — decision logged on K1 only (K2 never gates)")
    else:
        lines.append("K2: k2_frontier.csv not present yet — decision logged on K1 only")

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = [f"", f"[{stamp}] KILL-SWITCH DECISION (scripts/killswitch_decision.py)",
             *[f"  {l}" for l in lines],
             f"  VERDICT: {verdict}",
             f"  inputs: {k1p.relative_to(CASSI_ROOT)}"
             + (f", {k2p.relative_to(CASSI_ROOT)}" if k2p.exists() else "")]
    with LOG.open("a") as f:
        f.write("\n".join(entry) + "\n")
    print("\n".join(entry))
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
