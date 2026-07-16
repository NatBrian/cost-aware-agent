"""Stopping-regret evaluation — paper_plan_v2 §2.5 (policy), §5.3 (metric),
§16 P4 (done-criterion).

CPU-safe: works with `MockStopper` or the real `HFStopperPredictor` — anything
exposing `predict(x, lam, meta) -> StopperPrediction` (optional `predict_batch`).

Given held-out FORCED-CONTINUATION trajectories and their Snell `LabelSet`:

  (a) simulate the stopper's policy — stop at the first t with Δ̂_t ≤ 0 (§2.5,
      fixed threshold; Alg.4) — and compute STOPPING REGRET per trajectory:
      U_{τ*} − U_{τ_stopper}, the utility gap of §5.3 (NOT |t − t*|, which
      ignores magnitude — a v5 metric explicitly replaced);
  (b) STOP/CONTINUE F1 vs a* over all labeled steps;
  (c) the P4 done-criterion comparisons: the stopper must beat (i) the
      majority-class baseline and (ii) a calibrated confidence probe —
      draft-stability threshold (STOP once steps_since_draft_changed ≥ k),
      with k calibrated on TRAIN regret. "If it cannot, STOP — fix
      features/labels before touching RL" (§16 P4).

U_t comes from the labels themselves (StepLabel.u_t — same economy as Alg.1),
so no cost recomputation can drift from the label pipeline (§2.4 "one
Lagrangian, not three").
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from cassi.common.schema import Trajectory
from cassi.labels.snell import LabelSet, StepLabel
from cassi.stopper.model import MockStopper

STOP, CONTINUE = "STOP", "CONTINUE"


# ------------------------------------------------------------------- indexing
def labels_by_trajectory(labelset: LabelSet) -> dict[tuple, list[StepLabel]]:
    idx: dict[tuple, list[StepLabel]] = {}
    for lab in labelset.labels:
        idx.setdefault((lab.task_id, lab.group_id, lab.rollout_idx), []).append(lab)
    for labs in idx.values():
        labs.sort(key=lambda l: l.t)
    return idx


def tau_star_of(labels: list[StepLabel]) -> int:
    """τ* = min{t : a*_t = STOP} — matches Alg.1 (interior Δ* ≤ 0 or terminal)."""
    return min(l.t for l in labels if l.a_star == STOP)


# ------------------------------------------------------------------ simulation
@dataclass
class EpisodeRecord:
    task_id: str
    group_id: str
    rollout_idx: int
    lam: float
    tau_star: int
    tau_stopper: int
    u_tau_star: float
    u_tau_stopper: float
    regret: float                      # U_{τ*} − U_{τ_stopper}  (§5.3)


def _predict_all(stopper, traj: Trajectory, lam: float) -> list:
    items = []
    for t in range(1, len(traj) + 1):
        meta = {"task_id": traj.task_id, "group_id": traj.group_id,
                "rollout_idx": traj.rollout_idx, "t": t,
                "domain": traj.domain, "allowance_B": traj.allowance_B}
        items.append((traj.steps[t - 1].x, lam, meta))
    if hasattr(stopper, "predict_batch"):
        return stopper.predict_batch(items)
    return [stopper.predict(x, l, m) for x, l, m in items]


def simulate_episode(stopper, traj: Trajectory, labels: list[StepLabel],
                     *, threshold: float = 0.0) -> tuple[EpisodeRecord, list, list]:
    """Replay one forced-continuation trajectory under the stopper's policy
    (Alg.4: stop at the first Δ̂_t ≤ threshold, else at T). Returns the episode
    record plus per-step (predicted action, a*) pairs for F1."""
    if len(labels) != len(traj):
        raise ValueError(
            f"label/trajectory length mismatch for {traj.task_id!r}: "
            f"{len(labels)} labels vs {len(traj)} steps")
    lam = labels[0].lam
    preds = _predict_all(stopper, traj, lam)
    T = len(traj)
    tau_stop = T
    for t in range(1, T + 1):
        if preds[t - 1].delta <= threshold:
            tau_stop = t
            break
    tau_opt = tau_star_of(labels)
    u = {l.t: l.u_t for l in labels}
    rec = EpisodeRecord(
        task_id=traj.task_id, group_id=traj.group_id, rollout_idx=traj.rollout_idx,
        lam=lam, tau_star=tau_opt, tau_stopper=tau_stop,
        u_tau_star=u[tau_opt], u_tau_stopper=u[tau_stop],
        regret=u[tau_opt] - u[tau_stop],
    )
    pred_actions = [STOP if p.delta <= threshold else CONTINUE for p in preds]
    true_actions = [l.a_star for l in labels]
    return rec, pred_actions, true_actions


# -------------------------------------------------------------------- metrics
def stop_f1(pred: list[str], true: list[str]) -> dict:
    """STOP as the positive class (the rare, decision-critical one)."""
    tp = sum(1 for p, t in zip(pred, true) if p == STOP and t == STOP)
    fp = sum(1 for p, t in zip(pred, true) if p == STOP and t == CONTINUE)
    fn = sum(1 for p, t in zip(pred, true) if p == CONTINUE and t == STOP)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = sum(1 for p, t in zip(pred, true) if p == t) / len(pred) if pred else 0.0
    return {"f1_stop": f1, "precision_stop": prec, "recall_stop": rec, "accuracy": acc}


def evaluate_stopper(stopper, trajectories: list[Trajectory], labelset: LabelSet,
                     *, threshold: float = 0.0) -> dict:
    """Full evaluation of one stopper against one λ's labels: mean/median regret,
    STOP F1, stop-step stats. Trajectories without labels are skipped (they were
    not part of this label set)."""
    idx = labels_by_trajectory(labelset)
    records: list[EpisodeRecord] = []
    pred_all: list[str] = []
    true_all: list[str] = []
    for traj in trajectories:
        labels = idx.get((traj.task_id, traj.group_id, traj.rollout_idx))
        if labels is None:
            continue
        rec, pred, true = simulate_episode(stopper, traj, labels, threshold=threshold)
        records.append(rec)
        pred_all.extend(pred)
        true_all.extend(true)
    if not records:
        raise ValueError("no (trajectory, label) overlap — check task splits")
    regrets = np.array([r.regret for r in records])
    metrics = {
        "lam": labelset.lam,
        "n_trajectories": len(records),
        "n_steps": len(pred_all),
        "mean_regret": float(regrets.mean()),
        "median_regret": float(np.median(regrets)),
        "max_regret": float(regrets.max()),
        "mean_tau_stopper": float(np.mean([r.tau_stopper for r in records])),
        "mean_tau_star": float(np.mean([r.tau_star for r in records])),
        **stop_f1(pred_all, true_all),
    }
    return {"metrics": metrics, "records": records}


def evaluate_multi_lambda(stopper, trajectories: list[Trajectory],
                          labelsets: list[LabelSet], *, threshold: float = 0.0) -> dict:
    """Pooled evaluation across the λ grid (the λ-conditioned stopper serves all
    of them, §2.3). `mean_regret` — the Alg.2 early-stop metric — averages
    per-λ mean regrets equally."""
    per_lam = {}
    for ls in labelsets:
        per_lam[ls.lam] = evaluate_stopper(stopper, trajectories, ls,
                                           threshold=threshold)["metrics"]
    return {
        "per_lambda": per_lam,
        "mean_regret": float(np.mean([m["mean_regret"] for m in per_lam.values()])),
        "mean_f1_stop": float(np.mean([m["f1_stop"] for m in per_lam.values()])),
    }


# ---------------------------------------------------- P4 baseline comparisons
def calibrate_draft_stability(train_trajectories: list[Trajectory],
                              train_labelset: LabelSet,
                              *, k_grid: range | list[int] = range(1, 9)) -> int:
    """Calibrate the confidence-probe threshold k (STOP once
    steps_since_draft_changed ≥ k) by minimizing mean stopping regret on TRAIN —
    the probe gets its best shot before the P4 comparison (§16 P4 (ii))."""
    best_k, best_regret = None, np.inf
    for k in k_grid:
        m = evaluate_stopper(MockStopper.draft_stability(int(k)),
                             train_trajectories, train_labelset)["metrics"]
        if m["mean_regret"] < best_regret:
            best_k, best_regret = int(k), m["mean_regret"]
    assert best_k is not None
    return best_k


def compare_p4_baselines(stopper, *, train_trajectories: list[Trajectory],
                         train_labelset: LabelSet,
                         heldout_trajectories: list[Trajectory],
                         heldout_labelset: LabelSet,
                         threshold: float = 0.0) -> dict:
    """The §16 P4 done-criterion table: {stopper, majority_class, draft_stability}
    each evaluated on held-out regret + F1. `p4_pass` is True iff the stopper's
    held-out mean regret beats BOTH baselines."""
    k = calibrate_draft_stability(train_trajectories, train_labelset)
    arms = {
        "stopper": stopper,
        "majority_class": MockStopper.majority_class(train_labelset),
        f"draft_stability_k{k}": MockStopper.draft_stability(k),
    }
    out: dict = {"probe_k": k, "arms": {}}
    for name, arm in arms.items():
        out["arms"][name] = evaluate_stopper(
            arm, heldout_trajectories, heldout_labelset, threshold=threshold)["metrics"]
    stopper_regret = out["arms"]["stopper"]["mean_regret"]
    baseline_regrets = [m["mean_regret"] for n, m in out["arms"].items() if n != "stopper"]
    out["p4_pass"] = bool(all(stopper_regret < b for b in baseline_regrets))
    return out


# ------------------------------------------------------------------ CSV output
def write_records_csv(records: list[EpisodeRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(EpisodeRecord.__dataclass_fields__)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))
    return path
