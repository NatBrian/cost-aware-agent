"""Trajectory & step schema — paper_plan_v2 §11.

JSONL per trajectory:
{task_id, allowance_B, steps[{x_t, a_t, o_t, c_t, tier_t, draft_t, q_t, answered_flag}], outcome}

Hard rule (§2.1/§11): `x_t` holds inference-available features ONLY. Ground-truth
derived quantities (q_t) and the answered_flag live on the Step, never inside x_t —
the stopper must never see them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

EMPTY_DRAFT = "EMPTY_DRAFT"

ACTION_TYPES = ("reason", "tool_call", "answer")
TIERS = ("HIGH", "MEDIUM", "LOW", "CRITICAL")
WALLET_SIZES = ("small", "medium", "large")


@dataclass
class StepFeatures:
    """x_t — everything the harness can compute at inference time (§11)."""

    # budget group
    tokens_used: int = 0
    tokens_pct: float = 0.0            # of a nominal token cap
    tool_calls: int = 0
    tool_pct: float = 0.0
    dollars: float = 0.0
    dollars_pct: float = 0.0           # of allowance_B
    burn_rate: float = 0.0             # $/step, trailing mean
    tier: str = "HIGH"                 # from % of allowance REMAINING (budget/cost.py)
    # progress group
    step_idx: int = 1                  # 1-based
    steps_since_draft_changed: int = 0
    draft_edit_distance_last3: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    retrieval_overlap_last3: float = 0.0   # in [0,1]
    n_distinct_sources: int = 0
    # draft group
    draft: str = EMPTY_DRAFT
    draft_len: int = 0
    # task group
    question: str = ""
    domain: str = "qa"                 # qa | alfworld
    # history group: last-K (action_type, obs digest ≤64 tok)
    history: list[dict] = field(default_factory=list)   # [{"t": int, "action_type": str, "obs_digest": str}]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StepFeatures":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Step:
    x: StepFeatures
    a: str                             # action type actually taken at t (ACTION_TYPES)
    o: str                             # observation digest (≤64 tok)
    c: float                           # step cost in dollars (draft-line tokens included, §2.6)
    tier: str                          # tier prevailing when c was spent — feeds m(tier_i) in U_t
    draft: str                         # running draft AFTER step t (what [DRAFT] shows at t+1)
    q: float                           # step-t quality — LABEL MACHINERY ONLY, never in x (§2.1)
    answered_flag: bool = False        # forced-continuation ANSWER event (§2.1) — free self-stop signal

    def to_dict(self) -> dict:
        d = asdict(self)
        d["x"] = self.x.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        d = dict(d)
        d["x"] = StepFeatures.from_dict(d["x"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Trajectory:
    task_id: str
    domain: str                        # qa | alfworld
    allowance_B: float                 # dollars — wallet drawn per (task, GRPO group) (§2.2)
    wallet_size: str                   # small | medium | large
    group_id: str                      # ties the G rollouts of one GRPO group together
    rollout_idx: int                   # 0..G-1 within the group
    steps: list[Step] = field(default_factory=list)
    outcome: dict = field(default_factory=dict)
    # outcome keys: {"Q_tau": float, "success": bool, "tau": int|None (ANSWER step; None if T_max),
    #                "gold": str|None, "collection_mode": "forced_continuation"|"rl",
    #                "seed": int, "iteration": int}

    def __len__(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "domain": self.domain,
            "allowance_B": self.allowance_B, "wallet_size": self.wallet_size,
            "group_id": self.group_id, "rollout_idx": self.rollout_idx,
            "steps": [s.to_dict() for s in self.steps], "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        d = dict(d)
        d["steps"] = [Step.from_dict(s) for s in d["steps"]]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def save_trajectories(trajs: list[Trajectory] | Iterator[Trajectory], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for t in trajs:
            f.write(json.dumps(t.to_dict()) + "\n")
            n += 1
    return n


def load_trajectories(path: str | Path) -> Iterator[Trajectory]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield Trajectory.from_dict(json.loads(line))
