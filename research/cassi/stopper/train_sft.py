"""Algorithm 2 — stopper SFT (paper_plan_v2 §2.3, §10, §16 P4, §17 `stopper`).

    SFT M_θ on (x_t → a*_t) CE + (x_t → Δ*_t) MSE + (x_t → V*_t) MSE,
    3 epochs, lr 2e-5, batch 64, max_seq 2048,
    early stop on HELD-OUT STOPPING REGRET (not CE).          — §10 Alg.2 / §17

Loss weights per §17 `stopper.heads`: 1.0·CE(action) + 0.5·MSE(Δ̂) + 0.5·MSE(V̂).
Regret is computed each epoch with `eval_regret.evaluate_multi_lambda` on
held-out (split BY TASK, §dataset) forced-continuation trajectories.

Why a plain torch loop, not TRL: the plan's §17/§19 name "TRL v1.8 SFTTrainer +
scalar head (AutoModelForSequenceClassification, num_labels=1)" — but that recipe
covers a SINGLE scalar head only; M_θ needs three heads with a mixed CE+MSE+MSE
objective (§2.3), which TRL's SFTTrainer does not expose. This module implements
the SAME recipe (AutoModel backbone + AdamW, identical hyperparameters) as a
custom loop, without depending on TRL internals.

Run (GPU expected):
    python -m cassi.stopper.train_sft --config configs/cassi.yaml \
        --labels labels_lam0.5.jsonl labels_lam1.0.jsonl \
        --trajectories round0.jsonl --out runs/stopper_v0

If CUDA is unavailable the script fails fast at startup and points at the GPU
acquire command for this machine (§16 P0: N=2 for stopper SFT).

Import hygiene: torch/transformers are imported inside main() only — importing
this module on a CPU-only box without torch never fails.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from cassi.common.config import load_config
from cassi.common.schema import Trajectory, load_trajectories
from cassi.labels.snell import LabelSet
from cassi.stopper import dataset as ds
from cassi.stopper.eval_regret import evaluate_multi_lambda
from cassi.stopper.model import create_model, stopper_loss

GPU_ACQUIRE_HINT = (
    "CUDA is unavailable but stopper SFT expects a GPU (§16 P0/P4). On this "
    "machine, acquire GPUs first with:  eval $(/mnt/src/zhanka/gpu_acquire.sh 2)  "
    "then rerun; release with /mnt/src/zhanka/gpu_release.sh when done."
)


def filter_labelset(ls: LabelSet, task_ids: set[str]) -> LabelSet:
    out = LabelSet(lam=ls.lam, domain=ls.domain, scale_s=ls.scale_s,
                   backup_residuals=list(ls.backup_residuals))
    out.labels = [l for l in ls.labels if l.task_id in task_ids]
    out.tau_star = {k: v for k, v in ls.tau_star.items() if k[0] in task_ids}
    return out


def _batches(n: int, size: int, rng: np.random.Generator):
    order = rng.permutation(n)
    for i in range(0, n, size):
        yield order[i:i + size]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CASSI stopper SFT (Algorithm 2)")
    p.add_argument("--config", default=None, help="configs/cassi.yaml (§17)")
    p.add_argument("--labels", nargs="+", required=True,
                   help="LabelSet JSONL files, one per λ — ALL pooled (§2.3)")
    p.add_argument("--trajectories", nargs="+", required=True,
                   help="forced-continuation trajectory JSONL files (§2.1)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--heldout-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--micro-batch", type=int, default=8,
                   help="per-device batch; grads accumulate to §17 batch 64")
    p.add_argument("--patience", type=int, default=1,
                   help="epochs without held-out regret improvement before stopping")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)

    import torch                                    # lazy — GPU box only
    from transformers import AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(GPU_ACQUIRE_HINT)

    cfg = load_config(args.config)
    sft = cfg["stopper"]["sft"]
    heads = cfg["stopper"]["heads"]
    weights = (float(heads["action"]["weight"]), float(heads["delta"]["weight"]),
               float(heads["value"]["weight"]))
    epochs, lr = int(sft["epochs"]), float(sft["lr"])
    batch, max_seq = int(sft["batch"]), int(sft["max_seq"])
    accum = max(1, batch // args.micro_batch)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------------------------------------------------------------- data
    labelsets = [ds.load_labelset(p) for p in args.labels]
    trajectories: list[Trajectory] = []
    for p in args.trajectories:
        trajectories.extend(load_trajectories(p))

    ctx = ds.SerializeContext.from_config(cfg)
    examples = ds.build_examples(trajectories, labelsets, ctx=ctx)
    train_ids, hold_ids = ds.split_task_ids(
        [e.task_id for e in examples], args.heldout_frac, args.seed)
    train_ex = [e for e in examples if e.task_id in train_ids]
    hold_trajs = [t for t in trajectories if t.task_id in hold_ids]
    hold_labelsets = [filter_labelset(ls, hold_ids) for ls in labelsets]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_examples(train_ex, out_dir / "train_examples.jsonl")
    ds.save_examples([e for e in examples if e.task_id in hold_ids],
                     out_dir / "heldout_examples.jsonl")

    # --------------------------------------------------------------- model
    base_model = cfg["stopper"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = create_model(base_model, device=args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    texts = [e.text for e in train_ex]
    y_action = torch.tensor([1.0 if e.action == "STOP" else 0.0 for e in train_ex])
    y_delta = torch.tensor([e.delta_norm for e in train_ex], dtype=torch.float32)
    y_value = torch.tensor([e.v_star for e in train_ex], dtype=torch.float32)

    from cassi.stopper.model import HFStopperPredictor
    predictor = HFStopperPredictor(
        model, tokenizer, tokens_max=ctx.tokens_max, tool_calls_max=ctx.tool_calls_max,
        t_max_by_domain=ctx.t_max_by_domain, max_seq=max_seq, device=args.device)

    # ------------------------------------------------------------- training
    rng = np.random.default_rng(args.seed)
    history: list[dict] = []
    best = {"regret": float("inf"), "epoch": -1, "state": None}
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses, comps = [], []
        optim.zero_grad()
        for i, idx in enumerate(_batches(len(train_ex), args.micro_batch, rng)):
            enc = tokenizer([texts[j] for j in idx], return_tensors="pt", padding=True,
                            truncation=True, max_length=max_seq).to(args.device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            dev = out["action_logit"].device
            loss, comp = stopper_loss(out, y_action[idx].to(dev), y_delta[idx].to(dev),
                                      y_value[idx].to(dev), weights=weights)
            (loss / accum).backward()
            losses.append(loss.detach().item())
            comps.append(comp)
            if (i + 1) % accum == 0:
                optim.step()
                optim.zero_grad()
        optim.step()                                 # flush any partial accumulation
        optim.zero_grad()

        # early-stop metric: HELD-OUT STOPPING REGRET, not CE (§10 Alg.2 / §17)
        eval_out = evaluate_multi_lambda(predictor, hold_trajs, hold_labelsets)
        regret = eval_out["mean_regret"]
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_ce_action": float(np.mean([c["ce_action"] for c in comps])),
            "train_mse_delta": float(np.mean([c["mse_delta"] for c in comps])),
            "train_mse_value": float(np.mean([c["mse_value"] for c in comps])),
            "heldout_mean_regret": regret,
            "heldout_mean_f1_stop": eval_out["mean_f1_stop"],
            "seconds": time.time() - t0,
        }
        history.append(rec)
        print(json.dumps(rec))

        if regret < best["regret"]:
            best = {"regret": regret, "epoch": epoch,
                    "state": copy.deepcopy({k: v.detach().cpu()
                                            for k, v in model.state_dict().items()})}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > args.patience:
                print(f"early stop: no held-out regret improvement for {bad_epochs} epochs")
                break

    # ---------------------------------------------------------------- output
    if best["state"] is not None:
        torch.save(best["state"], out_dir / "stopper_best.pt")
    tokenizer.save_pretrained(out_dir / "tokenizer")
    summary = {
        "base_model": base_model, "best_epoch": best["epoch"],
        "best_heldout_regret": best["regret"], "epochs_run": len(history),
        "n_train_examples": len(train_ex), "n_heldout_tasks": len(hold_ids),
        "lambda_values": sorted({ls.lam for ls in labelsets}),
        "head_weights": weights, "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
