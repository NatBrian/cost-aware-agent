"""Lean GRPO trainer (F5, E-d decision 2026-07-23) — replaces verl.

WHY CUSTOM (logged): verl 0.8 resolves against our pinned torch, but its value
is colocated rollout workers with weight sync — adopting it means abandoning
our tested harness rollouts and debugging ray workers on a box with an old
driver and no nvcc (the GDN-JIT swamp again, one layer deeper). This trainer
is ~350 lines over already-tested pieces (rewards, advantages, divergence).

DESIGN — round-synced on-policy GRPO:
  round r: collect (harness --train: messages + sampled-token logprobs via the
  serving vLLM) -> judge rewards -> per-step group advantages -> clipped-ratio
  policy-gradient update (ratios vs rollout-time logprobs absorb within-round
  staleness; KL anchor to round-start policy via the same logprobs) -> save
  checkpoint -> restart server on it -> next round.
  Dr. GRPO hygiene: token loss normalized by a CONSTANT (max_gen_tokens), never
  per-sequence length. Malformed/mismatched samples dropped and counted.

Runs in .venv-gpu (torch 2.10+cu128). CPU tests cover sample building
(fake tokenizer); tensor math is exercised by --self-test on GPU and at the
E-d micro-run.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import FOUNDATION_ROOT, load_config, write_run_stamp
from reward.judge_client import JudgeClient
from train.reward_adapter import DivergenceLog, batch_rewards


# --------------------------------------------------------------------------
# sample building (pure python — CPU-testable)
# --------------------------------------------------------------------------

def group_episodes(episodes: list[dict]) -> list[list[dict]]:
    by_task: dict[str, list[dict]] = {}
    for ep in episodes:
        by_task.setdefault(ep["task_id"], []).append(ep)
    return list(by_task.values())


def _template_ids(tokenizer, msgs, **kwargs) -> list[int]:
    """Plain token-id list from apply_chat_template across transformers 4/5
    (v5 returns a BatchEncoding)."""
    out = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=True, **kwargs)
    if hasattr(out, "input_ids"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def build_samples(enriched_groups: list[list[dict]], tokenizer,
                  max_ctx: int = 8192,
                  template_kwargs: dict | None = None) -> tuple[list[dict], dict]:
    """One training sample per (episode, step): prompt token ids (chat template
    up to the step's reply), reply token ids, old logprobs, scalar advantage.

    Drops (counted): steps whose re-tokenized reply length mismatches the
    rollout logprob count (template drift), or whose context exceeds max_ctx.
    """
    samples, stats = [], {"kept": 0, "len_mismatch": 0, "too_long": 0,
                          "no_logprobs": 0}
    for group in enriched_groups:
        for ep in group:
            msgs = ep["messages"]
            for step, adv in zip(ep["steps"], ep["advantages"]):
                reply = msgs[step["asst_idx"]]["content"]
                old_lps = step.get("logprobs") or []
                if not old_lps:
                    stats["no_logprobs"] += 1
                    continue
                prompt_ids = _template_ids(tokenizer, msgs[: step["asst_idx"]],
                                           **(template_kwargs or {}))
                reply_ids = tokenizer.encode(reply, add_special_tokens=False)
                if len(reply_ids) != len(old_lps):
                    # vLLM appends EOS/thinking-off wrappers the raw text lacks;
                    # tolerate off-by-small by truncating to the shorter, else drop
                    if abs(len(reply_ids) - len(old_lps)) <= 2:
                        n = min(len(reply_ids), len(old_lps))
                        reply_ids, old_lps = reply_ids[:n], old_lps[:n]
                    else:
                        stats["len_mismatch"] += 1
                        continue
                if len(prompt_ids) + len(reply_ids) > max_ctx:
                    stats["too_long"] += 1
                    continue
                samples.append({"prompt_ids": prompt_ids, "reply_ids": reply_ids,
                                "old_logprobs": old_lps, "advantage": float(adv)})
                stats["kept"] += 1
    return samples, stats


# --------------------------------------------------------------------------
# tensor side (imported lazily; GPU env only)
# --------------------------------------------------------------------------

def grpo_microbatch_loss(model, batch, device, clip_eps: float, kl_beta: float,
                         norm_const: float, torch):
    """Clipped-ratio PG + k3 KL to rollout policy, Dr.GRPO constant norm.
    batch: list of samples. Returns (loss_tensor, metrics_dict)."""
    total = None
    m = {"ratio_mean": 0.0, "kl": 0.0, "clipped_frac": 0.0, "n_tokens": 0}
    for s in batch:
        ids = torch.tensor([s["prompt_ids"] + s["reply_ids"]], device=device)
        n_p, n_r = len(s["prompt_ids"]), len(s["reply_ids"])
        out = model(ids).logits[0, n_p - 1: n_p + n_r - 1]      # predict reply tokens
        logps = torch.log_softmax(out.float(), dim=-1)
        new_lp = logps[torch.arange(n_r), torch.tensor(s["reply_ids"], device=device)]
        old_lp = torch.tensor(s["old_logprobs"], device=device, dtype=torch.float32)
        ratio = torch.exp(new_lp - old_lp)
        adv = s["advantage"]
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        pg = -torch.minimum(unclipped, clipped)                 # per token
        # k3 KL estimator to the rollout policy (anchor). Round-1 lesson:
        # unbounded log-ratios exploded KL to 1e5 on rare outlier tokens
        # (mean 626 vs healthy 0.02); clamp keeps the estimator finite while
        # leaving the in-trust-region signal untouched.
        log_r = (old_lp - new_lp).clamp(-8.0, 8.0)
        kl = torch.exp(log_r) - 1 - log_r
        tok_loss = (pg + kl_beta * kl).sum() / norm_const       # Dr.GRPO const norm
        total = tok_loss if total is None else total + tok_loss
        with torch.no_grad():
            m["ratio_mean"] += float(ratio.mean())
            m["kl"] += float(kl.mean())
            m["clipped_frac"] += float(((ratio < 1 - clip_eps) |
                                        (ratio > 1 + clip_eps)).float().mean())
            m["n_tokens"] += n_r
    k = len(batch)
    for key in ("ratio_mean", "kl", "clipped_frac"):
        m[key] /= max(1, k)
    return total / max(1, k), m


def merge_untrained_hub_weights(ckpt_dir, hub_model: str, torch) -> None:
    """Qwen3.5-9B is multimodal; AutoModelForCausalLM trains/saves only the
    text side. Serving with the hub config needs the untouched extras
    (model.visual.*, mtp.*) — copy them from the hub shards into our save."""
    import glob as _glob
    import json as _json
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    ours = load_file(str(ckpt_dir / "model.safetensors"))
    snap = snapshot_download(hub_model, allow_patterns=["*.safetensors*"])
    added = 0
    for shard in sorted(_glob.glob(snap + "/*.safetensors")):
        with safe_open(shard, "pt") as f:
            for k in f.keys():
                if k not in ours:
                    ours[k] = f.get_tensor(k)
                    added += 1
    save_file(ours, str(ckpt_dir / "model.safetensors"),
              metadata={"format": "pt"})
    (ckpt_dir / "model.safetensors.index.json").unlink(missing_ok=True)
    print(f"[round] merged {added} untrained hub tensors into checkpoint")


def train_round(episodes_path: str, out_dir: str, cfg: dict,
                max_steps: int | None = None, mock_judge=None,
                init_from: str | None = None) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = cfg["grpo"]
    episodes = [json.loads(l) for l in open(episodes_path) if l.strip()]
    groups = group_episodes(episodes)
    judge = mock_judge or JudgeClient(
        cfg["judge"]["endpoint"], cfg["judge"]["model"],
        cfg["rubric"]["version"], FOUNDATION_ROOT / cfg["judge"]["cache_dir"],
        max_tokens=cfg["judge"]["max_tokens"])
    div = DivergenceLog()
    print(f"[round] {len(episodes)} episodes / {len(groups)} groups; judging...")
    enriched = batch_rewards(groups, judge, cfg, div, train_step=0)

    tok = AutoTokenizer.from_pretrained(cfg["executor"]["model"])
    # template rendering must match vLLM's rollout-time rendering
    samples, stats = build_samples(
        enriched, tok,
        template_kwargs={"enable_thinking": cfg["executor"]["enable_thinking"]})
    print(f"[round] samples: {stats}")
    if not samples:
        raise SystemExit("no trainable samples — check logprob capture")

    model = AutoModelForCausalLM.from_pretrained(
        init_from or cfg["executor"]["model"],
        torch_dtype=torch.bfloat16).to(device)
    print(f"[round] init from: {init_from or cfg['executor']['model']}")
    model.gradient_checkpointing_enable()
    model.train()
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=g["lr"])
    except ImportError:
        opt = torch.optim.AdamW(model.parameters(), lr=g["lr"])
        print("[round] bitsandbytes missing — fp32 AdamW states (more VRAM)")

    accum = int(cfg["grpo"].get("accum", 32))
    norm_const = float(cfg["executor"]["max_tokens_per_step"])
    import random as _rnd
    order = list(range(len(samples)))
    _rnd.Random(cfg["seed"]).shuffle(order)   # never length-sorted: biases updates
    log, n_upd = [], 0
    opt.zero_grad()
    for i, idx in enumerate(order):
        loss, m = grpo_microbatch_loss(model, [samples[idx]], device,
                                       g["clip_eps"], g["kl_beta"],
                                       norm_const, torch)
        (loss / accum).backward()
        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            n_upd += 1
            m["update"] = n_upd
            m["loss"] = float(loss.detach())
            log.append(m)
            if n_upd % 5 == 0:
                print(f"[round] upd {n_upd}: loss={m['loss']:.4f} "
                      f"ratio={m['ratio_mean']:.3f} kl={m['kl']:.4f} "
                      f"clip={m['clipped_frac']:.2f}")
            if max_steps and n_upd >= max_steps:
                break

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "checkpoint", safe_serialization=True)
    tok.save_pretrained(out / "checkpoint")
    # transformers 5 relabels the config model_type (qwen3_5_text) which vLLM's
    # registry doesn't know; restore the hub original (same class, same keys).
    import glob as _glob
    import shutil
    from huggingface_hub import snapshot_download
    snap = snapshot_download(cfg["executor"]["model"],
                             allow_patterns=["*.json", "*.txt", "*.jinja"])
    for f in _glob.glob(snap + "/*"):
        b = f.rsplit("/", 1)[1]
        if not b.endswith(".safetensors") and "safetensors.index" not in b:
            shutil.copy(f, out / "checkpoint" / b)
    merge_untrained_hub_weights(out / "checkpoint", cfg["executor"]["model"], torch)
    div.save(out / "divergence.jsonl")
    with open(out / "train_log.jsonl", "w") as f:
        for row in log:
            f.write(json.dumps(row) + "\n")
    summary = {"updates": n_upd, "samples": stats,
               "judge": getattr(judge, "stats", None) and judge.stats.as_dict(),
               "final_loss": log[-1]["loss"] if log else None,
               "mean_kl": sum(r["kl"] for r in log) / max(1, len(log))}
    with open(out / "round_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--init-from", default=None,
                    help="previous round checkpoint dir (default: base model)")
    args = ap.parse_args()
    cfg = load_config()
    write_run_stamp(args.out, cfg, {"cli": vars(args)})
    print(json.dumps(train_round(args.episodes, args.out, cfg,
                                 args.max_steps,
                                 init_from=args.init_from), indent=2))


if __name__ == "__main__":
    main()
