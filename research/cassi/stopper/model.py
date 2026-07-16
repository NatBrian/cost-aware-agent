"""M_θ — the three-headed stopping-value model (paper_plan_v2 §2.3, §18.3
value-head variant).

Backbone: `stopper.base_model` (§17, default Qwen/Qwen3.5-2B), plus three heads
on the last hidden state of the FINAL non-pad token of the serialized §18.1 input:

    action head  — binary logit, STOP vs CONTINUE           (CE, weight 1.0)
    Δ̂ head       — normalized stop margin, tanh-bounded     (MSE, weight 0.5)
    V̂ head       — UNNORMALIZED value, the shaping potential Φ (MSE, weight 0.5)

Head weights per §17 `stopper.heads`. Enforcement at inference: STOP when
Δ̂ ≤ 0 (§2.5); V̂ feeds the potential-based executor shaping (§2.4) and is never
parsed from text (§18.3).

Import hygiene: ALL torch/transformers imports live inside functions, so
`import cassi.stopper.model` (and everything that imports it) works on a
CPU-only box without torch. `MockStopper` is pure numpy and deterministic — it
backs the CPU tests here and in other modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from cassi.common.schema import StepFeatures
from cassi.labels.snell import LabelSet
from cassi.stopper.features import serialize


# ---------------------------------------------------------- predictor protocol
@dataclass
class StopperPrediction:
    action: str          # STOP | CONTINUE — derived from delta (Δ̂ ≤ 0 ⇒ STOP, §2.5)
    delta: float         # Δ̂ ∈ [−1, 1]
    v: float             # V̂ (unnormalized quality units)


def _action_from_delta(delta: float, threshold: float = 0.0) -> str:
    return "STOP" if delta <= threshold else "CONTINUE"


class MockStopper:
    """Deterministic pure-numpy stand-in for M_θ (CPU tests, other modules' tests).

    delta_fn / v_fn take (x: StepFeatures, lam: float, meta: dict) and return a
    float. `meta` carries {task_id, group_id, rollout_idx, t, domain, allowance_B}
    so mocks can look up ground-truth labels (the oracle mock) without those
    quantities ever entering x_t (§2.1 hard rule).
    """

    def __init__(self, delta_fn: Callable[[StepFeatures, float, dict], float],
                 v_fn: Callable[[StepFeatures, float, dict], float] | None = None,
                 name: str = "mock"):
        self.delta_fn = delta_fn
        self.v_fn = v_fn or (lambda x, lam, meta: 0.0)
        self.name = name

    def predict(self, x: StepFeatures, lam: float, meta: dict | None = None) -> StopperPrediction:
        meta = meta or {}
        d = float(self.delta_fn(x, lam, meta))
        return StopperPrediction(action=_action_from_delta(d), delta=d,
                                 v=float(self.v_fn(x, lam, meta)))

    # ------------------------------------------------------- canonical mocks
    @classmethod
    def oracle(cls, label_sets: list[LabelSet] | LabelSet) -> "MockStopper":
        """Predicts the TRUE (Δ*_norm, V*) from the label sets — should achieve
        ~zero stopping regret and F1 = 1 (the eval_regret consistency check)."""
        if isinstance(label_sets, LabelSet):
            label_sets = [label_sets]
        table = {}
        for ls in label_sets:
            for lab in ls.labels:
                table[(lab.task_id, lab.group_id, lab.rollout_idx, lab.t, lab.lam)] = (
                    lab.delta_norm, lab.v_star)
        def _lookup(meta, lam):
            key = (meta["task_id"], meta["group_id"], meta["rollout_idx"], meta["t"], lam)
            if key not in table:
                raise KeyError(f"oracle mock has no label for {key}")
            return table[key]
        return cls(delta_fn=lambda x, lam, meta: _lookup(meta, lam)[0],
                   v_fn=lambda x, lam, meta: _lookup(meta, lam)[1], name="oracle")

    @classmethod
    def constant(cls, delta: float, v: float = 0.0, name: str = "constant") -> "MockStopper":
        return cls(delta_fn=lambda x, lam, meta: delta,
                   v_fn=lambda x, lam, meta: v, name=name)

    @classmethod
    def majority_class(cls, label_set: LabelSet) -> "MockStopper":
        """Predicts the majority a* everywhere (P4 done-criterion baseline (i)):
        STOP-majority ⇒ Δ̂ ≡ −1 (stop at t=1); CONTINUE-majority ⇒ Δ̂ ≡ +1 (run to T)."""
        n_stop = sum(1 for lab in label_set.labels if lab.a_star == "STOP")
        stop_major = n_stop * 2 > len(label_set.labels)
        return cls.constant(-1.0 if stop_major else 1.0, name="majority_class")

    @classmethod
    def draft_stability(cls, k: int) -> "MockStopper":
        """Confidence-probe baseline (P4 done-criterion (ii)): STOP once the running
        draft has been unchanged for ≥ k steps (§11 steps_since_draft_changed);
        the threshold k is calibrated on train (eval_regret.calibrate_draft_stability)."""
        return cls(delta_fn=lambda x, lam, meta:
                   -1.0 if x.steps_since_draft_changed >= k else 1.0,
                   name=f"draft_stability_k{k}")


# ----------------------------------------------------------- torch implementation
_MODEL_CLS = None


def get_model_class():
    """Lazily define (and cache) the torch nn.Module — torch/transformers are
    imported HERE, never at module import time."""
    global _MODEL_CLS
    if _MODEL_CLS is not None:
        return _MODEL_CLS

    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class ThreeHeadStopper(nn.Module):
        """AutoModel backbone + three linear heads on the final-token hidden state
        (§2.3, §18.3 value-head variant). Δ̂ is tanh-bounded to match the
        tanh-normalized label (Alg.1 line 7); V̂ is unbounded (unnormalized Φ)."""

        def __init__(self, base_model: str, torch_dtype=None):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(
                base_model, dtype=torch_dtype or torch.float32)   # transformers v5 kwarg (§19 pin)
            h = self.backbone.config.hidden_size
            # heads in float32 for regression stability regardless of backbone dtype
            self.action_head = nn.Linear(h, 1)
            self.delta_head = nn.Linear(h, 1)
            self.value_head = nn.Linear(h, 1)

        def forward(self, input_ids, attention_mask):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state                      # (B, L, H)
            last_idx = attention_mask.sum(dim=1) - 1            # final non-pad token
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
            pooled = pooled.float()
            return {
                "action_logit": self.action_head(pooled).squeeze(-1),   # >0 ⇒ STOP
                "delta": torch.tanh(self.delta_head(pooled)).squeeze(-1),
                "value": self.value_head(pooled).squeeze(-1),
            }

    _MODEL_CLS = ThreeHeadStopper
    return _MODEL_CLS


def create_model(base_model: str, device: str = "cuda", torch_dtype=None):
    """Instantiate M_θ from `stopper.base_model` (§17) on `device`."""
    import torch
    cls = get_model_class()
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if (device.startswith("cuda")
                                         and torch.cuda.is_bf16_supported()) else torch.float32
    return cls(base_model, torch_dtype=torch_dtype).to(device)


def stopper_loss(outputs: dict, action_target, delta_target, value_target,
                 weights: tuple[float, float, float] = (1.0, 0.5, 0.5)):
    """L = w_a·CE(action) + w_Δ·MSE(Δ̂) + w_V·MSE(V̂) — §17 stopper.heads /
    Alg.2. `action_target` is 1.0 for STOP, 0.0 for CONTINUE. Returns
    (total, dict of components)."""
    import torch.nn.functional as F
    ce = F.binary_cross_entropy_with_logits(outputs["action_logit"], action_target)
    mse_d = F.mse_loss(outputs["delta"], delta_target)
    mse_v = F.mse_loss(outputs["value"], value_target)
    total = weights[0] * ce + weights[1] * mse_d + weights[2] * mse_v
    return total, {"ce_action": ce.detach().item(), "mse_delta": mse_d.detach().item(),
                   "mse_value": mse_v.detach().item()}


class HFStopperPredictor:
    """Adapts a trained ThreeHeadStopper (+tokenizer) to the predict(x, λ, meta)
    interface that eval_regret and the inference monitor (§2.5) consume.
    Serialization uses the SAME §18.1 template as training (features.serialize);
    the trajectory's own wallet arrives via meta['allowance_B'] (§2.2)."""

    def __init__(self, model, tokenizer, *, tokens_max: int = 8192,
                 tool_calls_max: int = 20, t_max_by_domain: dict | None = None,
                 max_seq: int = 2048, device: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.tokens_max = tokens_max
        self.tool_calls_max = tool_calls_max
        self.t_max_by_domain = t_max_by_domain or {"qa": 10, "alfworld": 20}
        self.max_seq = max_seq
        self._device = device
        self.name = "hf_stopper"

    def _serialize(self, x: StepFeatures, lam: float, meta: dict) -> str:
        t_max = self.t_max_by_domain.get(meta.get("domain", x.domain), 10)
        return serialize(x, lam, tokens_max=self.tokens_max,
                         tool_calls_max=self.tool_calls_max,
                         allowance_dollars=float(meta.get("allowance_B", 0.0)),
                         t_max=t_max)

    def _forward_texts(self, texts: list[str]) -> list[StopperPrediction]:
        import torch
        device = self._device or next(self.model.parameters()).device
        enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=self.max_seq).to(device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        deltas = out["delta"].float().cpu().numpy()
        values = out["value"].float().cpu().numpy()
        return [StopperPrediction(action=_action_from_delta(float(d)),
                                  delta=float(d), v=float(v))
                for d, v in zip(deltas, values)]

    def predict(self, x: StepFeatures, lam: float, meta: dict | None = None) -> StopperPrediction:
        return self._forward_texts([self._serialize(x, lam, meta or {})])[0]

    def predict_batch(self, items: list[tuple[StepFeatures, float, dict]],
                      batch_size: int = 32) -> list[StopperPrediction]:
        """Vectorized path used by eval_regret when available — predictions do not
        depend on the stopping decision, so a whole trajectory ships at once."""
        preds: list[StopperPrediction] = []
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            preds.extend(self._forward_texts(
                [self._serialize(x, lam, meta or {}) for x, lam, meta in chunk]))
        return preds
