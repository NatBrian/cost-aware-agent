#!/usr/bin/env python3
"""P4 gate driver (paper_plan_v2 §16 P4 done-criterion).

Loads the SFT'd three-head stopper from a train_sft output directory and runs
cassi.stopper.eval_regret.compare_p4_baselines on HELD-OUT tasks:
stopper vs (i) majority-class vs (ii) the calibrated draft-stability probe
(the module's confidence-probe stand-in), on held-out stopping regret.

Uses the SAME task-level split machinery (split_task_ids, same seed/frac) as
train_sft, so "held-out" here matches training's held-out set. One labelset per
domain (the default-λ set); pass several to gate on all domains.

Writes a JSON report with `p4_pass` — scripts/p4_stopper.sh gates on it.
GPU required (loads the 2B backbone).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CASSI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CASSI_ROOT.parent))

from cassi.common.config import load_config  # noqa: E402
from cassi.common.schema import load_trajectories  # noqa: E402
from cassi.stopper.dataset import load_labelset, split_task_ids  # noqa: E402
from cassi.stopper.eval_regret import compare_p4_baselines  # noqa: E402
from cassi.stopper.model import HFStopperPredictor, create_model  # noqa: E402


def load_stopper(stopper_dir: Path, device: str, cfg: dict):
    import torch
    from transformers import AutoTokenizer

    meta = json.loads((stopper_dir / "metrics.json").read_text())
    model = create_model(meta["base_model"], device=device)
    state = torch.load(stopper_dir / "stopper_best.pt", map_location=device)
    model.load_state_dict(state)
    tokenizer = AutoTokenizer.from_pretrained(stopper_dir / "tokenizer")
    horizon = cfg["executor"]["horizon"]
    return HFStopperPredictor(model, tokenizer, t_max_by_domain=dict(horizon),
                              max_seq=int(cfg["stopper"]["sft"]["max_seq"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stopper-dir", type=Path, required=True, help="train_sft --out dir")
    ap.add_argument("--labels", nargs="+", type=Path, required=True,
                    help="default-lambda LabelSet JSONL, one per domain")
    ap.add_argument("--trajectories", nargs="+", type=Path, required=True,
                    help="forced-continuation trajectory JSONLs (round dir files)")
    ap.add_argument("--heldout-frac", type=float, default=0.2, help="MUST match train_sft")
    ap.add_argument("--seed", type=int, default=42, help="MUST match train_sft")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config()
    stopper = load_stopper(args.stopper_dir, args.device, cfg)

    trajs = []
    for p in args.trajectories:
        trajs.extend(load_trajectories(p))
    by_domain = {}
    for t in trajs:
        by_domain.setdefault(t.domain, []).append(t)

    report = {"stopper_dir": str(args.stopper_dir), "domains": {}, "p4_pass": True}
    for lp in args.labels:
        ls = load_labelset(lp)
        dtrajs = by_domain.get(ls.domain, [])
        if not dtrajs:
            print(f"[p4_gate] no trajectories for domain {ls.domain} — skipping", file=sys.stderr)
            continue
        train_ids, hold_ids = split_task_ids([t.task_id for t in dtrajs],
                                             args.heldout_frac, args.seed)
        train = [t for t in dtrajs if t.task_id in train_ids]
        hold = [t for t in dtrajs if t.task_id in hold_ids]
        res = compare_p4_baselines(stopper, train_trajectories=train, train_labelset=ls,
                                   heldout_trajectories=hold, heldout_labelset=ls)
        report["domains"][ls.domain] = res
        report["p4_pass"] = report["p4_pass"] and res["p4_pass"]
        arms = {n: m["mean_regret"] for n, m in res["arms"].items()}
        print(f"[p4_gate] {ls.domain} (lambda={ls.lam}): held-out mean regret {arms} "
              f"-> pass={res['p4_pass']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[p4_gate] report -> {args.out}; P4 GATE {'PASS' if report['p4_pass'] else 'FAIL'}")
    return 0 if report["p4_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
