#!/usr/bin/env python3
"""Evaluate the PROMPTED reward model (RM-P, baseline B10) on the SAME held-out
split and metric as the P4 gate — so trained-vs-prompted appears side by side.

Maps B10's continue_score s∈[0,1] onto the stopper's margin convention via
Δ̂ = 2s − 1 (stop when Δ̂ ≤ 0 ⇔ s ≤ 0.5, the neutral operating point; B10's own
frontier knob θ_p sweeps this at P8). V̂ = state_value (the V̂ analog).
Unparseable judge replies fail open to CONTINUE (Δ̂ = +1), per the B10 module.

Cost control: --sample N trajectories (default 150 ≈ 1.5K judge calls ≈ <$1 at
the measured $0.0004/call). Calls run in a thread pool against the vLLM server.

Usage:
  python scripts/eval_rmp_heldout.py --round 0 --domain qa --lam 1.0 --sample 150
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cassi.baselines import b10_prompted_rm as b10
from cassi.common.config import load_config
from cassi.common.schema import load_trajectories
from cassi.executor.monitor import StopperOutput  # noqa: F401 (interface doc)
from cassi.stopper.dataset import load_labelset, split_task_ids
from cassi.stopper.eval_regret import evaluate_stopper
from cassi.stopper.features import serialize
from cassi.stopper.model import StopperPrediction


class RMPJudgePredictor:
    """B10 judge behind the stopper predict() interface (thread-pooled)."""

    name = "rmp_prompted_judge"

    def __init__(self, base_url: str, model: str, cfg: dict, workers: int = 8):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cfg = cfg
        self.workers = workers
        self.calls = 0
        self.parse_failures = 0
        self.tokens = [0, 0]

    def _complete(self, prompt: str) -> str:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(self.cfg["prompted_rm"]["max_output_tokens"]),
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }).encode(),
            headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = json.load(r)
                u = body.get("usage", {})
                self.tokens[0] += u.get("prompt_tokens", 0)
                self.tokens[1] += u.get("completion_tokens", 0)
                return body["choices"][0]["message"]["content"]
            except Exception:
                if attempt == 2:
                    return ""          # fail-open (parse_rubric -> None -> CONTINUE)
        return ""

    def _one(self, item) -> StopperPrediction:
        x, lam, meta = item
        t_max = {"qa": 10, "alfworld": 20}.get(meta.get("domain", "qa"), 10)
        sx = serialize(x, lam, tokens_max=8192, tool_calls_max=20,
                       allowance_dollars=float(meta.get("allowance_B", 0.0)),
                       t_max=t_max)
        bits = b10.parse_rubric(self._complete(b10.build_judge_prompt(sx)))
        self.calls += 1
        if bits is None:
            self.parse_failures += 1
            return StopperPrediction(action="CONTINUE", delta=1.0, v=0.0)
        s = b10.continue_score(bits)
        delta = 2.0 * s - 1.0
        return StopperPrediction(action="STOP" if delta <= 0 else "CONTINUE",
                                 delta=delta, v=b10.state_value(bits))

    def predict(self, x, lam, meta=None):
        return self._one((x, lam, meta or {}))

    def predict_batch(self, items, batch_size: int = 32):
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(self._one, items))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--domain", default="qa")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--heldout-frac", type=float, default=0.2)  # MUST match train_sft/p4_gate
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    prm = cfg["prompted_rm"]
    lam_name = str(args.lam).rstrip("0").rstrip(".")
    ls = load_labelset(root / f"experiments/labels/round{args.round}/{args.domain}_lambda{lam_name}.jsonl")
    trajs = list(load_trajectories(   # materialize: it's a generator, consumed twice below
        root / f"experiments/collect/round{args.round}/{args.domain}.jsonl"))

    _, hold_ids = split_task_ids([t.task_id for t in trajs], args.heldout_frac, args.seed)
    hold = [t for t in trajs if t.task_id in set(hold_ids)]
    rng = np.random.default_rng(args.seed)
    if len(hold) > args.sample:
        hold = [hold[i] for i in rng.choice(len(hold), args.sample, replace=False)]

    judge = RMPJudgePredictor(prm["base_url"], prm["model"], cfg, workers=args.workers)
    res = evaluate_stopper(judge, hold, ls)
    res["rmp_meta"] = {
        "model": prm["model"], "n_trajectories": len(hold), "judge_calls": judge.calls,
        "parse_failures": judge.parse_failures,
        "tokens_in": judge.tokens[0], "tokens_out": judge.tokens[1],
        "est_dollars": round(b10.bill_judge(judge.tokens[0], judge.tokens[1]).dollars, 4),
        "operating_point": "theta_p=0.5 (delta = 2*continue_score - 1)",
    }
    out = Path(args.out or root / f"experiments/stopper/round{args.round}/rmp_heldout_{args.domain}_lam{lam_name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {"metrics": res["metrics"], "rmp_meta": res["rmp_meta"]}  # "records" hold dataclasses
    out.write_text(json.dumps(serializable, indent=2))
    print(json.dumps(serializable, indent=2)[:1500])
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
