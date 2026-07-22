"""Stopper SFT dataset — paper_plan_v2 §2.3, §10 Alg.2, §18.1.

Turns a Snell `LabelSet` (labels/snell.py, Algorithm 1) plus its source
trajectories into supervised examples for the three-headed stopping-value model:

    input   = features.serialize(x_t, λ, ...)   — the §18.1 text block
    targets = (a*_t ∈ {STOP, CONTINUE},  Δ*_t normalized ∈ [−1,1],  V*_t unnormalized)

λ-conditioning (§2.3): ONE dataset pools ALL λ label sets; λ lives inside the
serialized input, so a single stopper implements the whole cost-sensitivity dial.

Train/heldout split is BY TASK_ID (never within a task): the G rollouts of a task
share the question and correlated states — splitting within a task leaks.

Pure python + numpy; CPU-safe (no torch/transformers anywhere in this module).
JSONL IO for both the built examples and `LabelSet` objects (snell.py defines the
dataclasses but no IO; adding it there is off-limits, so it lives here).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from cassi.common.schema import Trajectory
from cassi.labels.snell import LabelSet, StepLabel
from cassi.stopper.features import serialize

DEFAULT_T_MAX = {"qa": 10, "alfworld": 20}   # §17 executor.horizon


# ------------------------------------------------------------- serialization ctx
@dataclass
class SerializeContext:
    """Nominal caps fed to features.serialize (§18.1 [BUDGET]/[PROGRESS] lines).
    allowance_dollars always comes from the trajectory's own wallet (§2.2)."""

    tokens_max: int = 8192
    tool_calls_max: int = 20
    t_max_by_domain: dict = field(default_factory=lambda: dict(DEFAULT_T_MAX))

    def t_max(self, domain: str) -> int:
        return int(self.t_max_by_domain.get(domain, max(self.t_max_by_domain.values())))

    @classmethod
    def from_config(cls, cfg: dict) -> "SerializeContext":
        return cls(t_max_by_domain=dict(cfg["executor"]["horizon"]))


# ------------------------------------------------------------------- examples
@dataclass
class StopperExample:
    """One (x_t, λ) → (a*, Δ*, V*) supervision triple (§10 Alg.2)."""

    text: str               # serialize(x_t, λ, ...) — §18.1, identical at train/inference
    action: str             # a*_t: STOP | CONTINUE
    delta_norm: float       # tanh(Δ*_t / s) ∈ [−1,1] — decision target (§2.5)
    v_star: float           # V*_t UNNORMALIZED — shaping-potential target Φ (§2.4)
    # bookkeeping (split, regret eval, QC)
    task_id: str
    group_id: str
    rollout_idx: int
    t: int
    lam: float
    domain: str
    u_t: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StopperExample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _traj_index(trajectories: list[Trajectory]) -> dict:
    idx = {}
    for tr in trajectories:
        idx[(tr.task_id, tr.group_id, tr.rollout_idx)] = tr
    return idx


def build_examples(
    trajectories: list[Trajectory],
    label_sets: list[LabelSet],
    *,
    ctx: SerializeContext | None = None,
) -> list[StopperExample]:
    """Pool ALL λ label sets into one dataset (§2.3 λ-conditioning): each labeled
    step becomes one example per λ, with that λ inside the serialized input."""
    ctx = ctx or SerializeContext()
    trajs = _traj_index(trajectories)
    out: list[StopperExample] = []
    for ls in label_sets:
        for lab in ls.labels:
            key = (lab.task_id, lab.group_id, lab.rollout_idx)
            tr = trajs.get(key)
            if tr is None:
                raise KeyError(f"label references unknown trajectory {key}")
            if not (1 <= lab.t <= len(tr)):
                raise IndexError(f"label t={lab.t} out of range for trajectory {key}")
            x = tr.steps[lab.t - 1].x
            out.append(StopperExample(
                text=serialize(
                    x, lab.lam,
                    tokens_max=ctx.tokens_max, tool_calls_max=ctx.tool_calls_max,
                    allowance_dollars=tr.allowance_B, t_max=ctx.t_max(tr.domain),
                ),
                action=lab.a_star, delta_norm=lab.delta_norm, v_star=lab.v_star,
                task_id=lab.task_id, group_id=lab.group_id, rollout_idx=lab.rollout_idx,
                t=lab.t, lam=lab.lam, domain=tr.domain, u_t=lab.u_t,
            ))
    return out


# ------------------------------------------------------------- split by task_id
def split_task_ids(task_ids: list[str], heldout_frac: float = 0.2,
                   seed: int = 42) -> tuple[set[str], set[str]]:
    """Deterministic task-level split. Sorted-then-shuffled so the result depends
    only on (task_ids, heldout_frac, seed), not input order."""
    uniq = sorted(set(task_ids))
    if not uniq:
        return set(), set()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_hold = max(1, int(round(heldout_frac * len(uniq)))) if len(uniq) > 1 else 0
    hold = {uniq[i] for i in perm[:n_hold]}
    return set(uniq) - hold, hold


def split_by_task(
    examples: list[StopperExample], heldout_frac: float = 0.2, seed: int = 42,
) -> tuple[list[StopperExample], list[StopperExample]]:
    """Train/heldout split BY TASK_ID — every example of a task lands on one side
    (the G rollouts of a task are correlated; within-task splits leak)."""
    train_ids, hold_ids = split_task_ids([e.task_id for e in examples], heldout_frac, seed)
    train = [e for e in examples if e.task_id in train_ids]
    hold = [e for e in examples if e.task_id in hold_ids]
    return train, hold


# ---------------------------------------------------------------- examples IO
def save_examples(examples: list[StopperExample], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in examples:
            f.write(json.dumps(e.to_dict()) + "\n")
    return len(examples)


def load_examples(path: str | Path) -> list[StopperExample]:
    out = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(StopperExample.from_dict(json.loads(line)))
    return out


# ---------------------------------------------------------------- LabelSet IO
# JSONL: first record is the meta header, remaining records are StepLabels.
# tau_star keys (task_id, rollout_idx) are encoded as [task_id, rollout_idx, tau] rows.
def save_labelset(ls: LabelSet, path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "kind": "labelset_meta", "lam": ls.lam, "domain": ls.domain,
        "scale_s": ls.scale_s,
        "tau_star": [[k[0], k[1], v] for k, v in ls.tau_star.items()],
        "backup_residuals": list(ls.backup_residuals),
    }
    with path.open("w") as f:
        f.write(json.dumps(meta) + "\n")
        for lab in ls.labels:
            f.write(json.dumps(asdict(lab)) + "\n")
    return len(ls.labels)


def load_labelset(path: str | Path) -> LabelSet:
    with Path(path).open() as f:
        lines = [ln for ln in (l.strip() for l in f) if ln]
    meta = json.loads(lines[0])
    if meta.get("kind") != "labelset_meta":
        raise ValueError(f"{path}: not a LabelSet JSONL (missing meta header)")
    ls = LabelSet(
        lam=meta["lam"], domain=meta["domain"], scale_s=meta["scale_s"],
        backup_residuals=list(meta.get("backup_residuals", [])),
    )
    ls.tau_star = {(t, int(r)): int(tau) for t, r, tau in meta.get("tau_star", [])}
    for line in lines[1:]:
        d = json.loads(line)
        ls.labels.append(StepLabel(**{k: v for k, v in d.items()
                                      if k in StepLabel.__dataclass_fields__}))
    return ls
