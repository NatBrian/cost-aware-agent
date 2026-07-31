"""S1 — the gold-free predictability check. THE KILL GATE for FOUNDATION-2.

The question
------------
FOUNDATION-2's thesis is that the agent should ABANDON unproductive episodes.
We can tell an episode is doomed because we hold the gold answer. **The agent
cannot.** So the entire redesign rests on one unproven claim:

    can eventual failure be predicted from GOLD-FREE state alone?

This script answers it on rollouts already on disk. No GPU, no new collection.

Gate (pre-registered in paper_plan_v2_2_foundation.md §8):
    held-out AUC >= 0.65  ->  proceed to Step 1
    below                 ->  hopelessness is visible only in hindsight; the
                              dataset change is promoted from Step 3 to mandatory

Leakage discipline (the thing that would silently invalidate this)
-----------------------------------------------------------------
1. `draft_f1_vs_gold` is GOLD. It is never a feature. Only the *shape* of the
   draft (length, churn, emptiness) is used — never its score.
2. **The split is by task_id, never by episode.** Each task has G=8 rollouts of
   the SAME question. An episode-level split puts siblings of a test question in
   train, and the model memorises "this question is answerable" instead of
   learning "this trajectory is going nowhere". That inflates AUC and is exactly
   the failure mode that would make us build a week of machinery on nothing.
3. Features are computed from steps[:k] only — the prefix a deployed policy would
   actually have seen at its decision point.

Usage:
  .venv/bin/python scripts/s1_predictability.py \
      --rollouts experiments/results/train/lam0_round1/rollouts.jsonl \
      --out experiments/results/s1
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EMPTY = "EMPTY_DRAFT"
_WORD = re.compile(r"[a-z0-9]+")

# Pre-registered gate. Not tunable from the command line on purpose.
AUC_GATE = 0.65
SEED = 42


def _toks(s: str) -> set:
    return set(_WORD.findall((s or "").lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def features(ep: dict, k: int) -> dict | None:
    """Gold-free features from the first k steps. None if the episode is shorter.

    Every value here is computable at inference time by the policy itself.
    """
    steps = ep["steps"]
    if len(steps) < k:
        return None
    pre = steps[:k]
    q_toks = _toks(ep["question"])

    queries = [s["query_or_answer"] for s in pre if s["action_type"] == "search"]
    qsets = [_toks(q) for q in queries]
    obs = [s.get("obs_digest") or "" for s in pre]
    drafts = [s.get("draft") or EMPTY for s in pre]

    # --- retrieval productivity ------------------------------------------
    empty_obs = sum(1 for o in obs if not o or "No results found" in o)
    obs_lens = [len(o) for o in obs]
    # distinct retrieved titles: "[n] Title: body"
    titles = set()
    for o in obs:
        titles.update(re.findall(r"\[\d+\]\s*([^:]{1,80}):", o))

    # --- query redundancy (the agent spinning) ---------------------------
    exact_rep = len(queries) - len(set(queries))
    near_rep = 0
    for i in range(len(qsets)):
        for j in range(i):
            if _jaccard(qsets[i], qsets[j]) >= 0.8:
                near_rep += 1
                break

    # --- draft dynamics ---------------------------------------------------
    non_empty = [d for d in drafts if d != EMPTY]
    draft_changes = sum(1 for a, b in zip(drafts, drafts[1:]) if a != b)
    last_draft = drafts[-1]
    draft_stable = float(len(drafts) >= 2 and drafts[-1] == drafts[-2]
                         and drafts[-1] != EMPTY)

    # --- grounding: does the draft appear in what we retrieved? ----------
    obs_toks = _toks(" ".join(obs))
    draft_toks = _toks(last_draft) if last_draft != EMPTY else set()
    draft_in_obs = (len(draft_toks & obs_toks) / len(draft_toks)) if draft_toks else 0.0

    # --- question coverage: are we retrieving about the right entities? ---
    q_cov = (len(q_toks & obs_toks) / len(q_toks)) if q_toks else 0.0

    # --- model confidence -------------------------------------------------
    lps = [np.mean(s["logprobs"]) for s in pre if s.get("logprobs")]
    lp_min = [np.min(s["logprobs"]) for s in pre if s.get("logprobs")]

    malformed = sum(1 for s in pre if s["action_type"] == "malformed")

    return {
        "k": k,
        "budget_B": ep["budget_B"],
        "budget_frac_used": k / max(ep["budget_B"], 1),
        # retrieval productivity
        "empty_obs_frac": empty_obs / k,
        "obs_len_mean": float(np.mean(obs_lens)),
        "obs_len_last": float(obs_lens[-1]),
        "distinct_titles": len(titles),
        "titles_per_step": len(titles) / k,
        # redundancy
        "exact_repeat": exact_rep,
        "near_repeat": near_rep,
        # draft dynamics
        "has_draft": float(bool(non_empty)),
        "draft_len": len(last_draft) if last_draft != EMPTY else 0,
        "draft_changes": draft_changes,
        "draft_stable": draft_stable,
        "steps_since_draft": k - (max(i for i, d in enumerate(drafts)
                                      if d != EMPTY) + 1) if non_empty else k,
        # grounding / coverage
        "draft_in_obs": draft_in_obs,
        "q_coverage": q_cov,
        # effort
        "raw_len_mean": float(np.mean([s.get("raw_len", 0) for s in pre])),
        "query_len_mean": float(np.mean([len(q) for q in queries])) if queries else 0.0,
        "malformed": malformed,
        # confidence
        "logprob_mean": float(np.mean(lps)) if lps else 0.0,
        "logprob_min": float(np.min(lp_min)) if lp_min else 0.0,
        "logprob_last": float(lps[-1]) if lps else 0.0,
    }


def build(rows: list[dict], k: int):
    X, y, g = [], [], []
    for ep in rows:
        f = features(ep, k)
        if f is None:
            continue
        X.append(f)
        y.append(int(ep["final_f1"] > 0))          # eventual success
        g.append(ep["task_id"])
    if not X:
        return None, None, None, []
    names = sorted(X[0])
    return (np.array([[x[n] for n in names] for x in X], float),
            np.array(y), np.array(g), names)


def evaluate(X, y, groups, names, seed=SEED):
    """Grouped split by task_id — never by episode (see module docstring)."""
    gss = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=seed)
    out = {"logreg": [], "gbm": []}
    imp = np.zeros(len(names))
    for tr, te in gss.split(X, y, groups):
        if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
            continue
        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000, C=1.0))
        lr.fit(X[tr], y[tr])
        out["logreg"].append(roc_auc_score(y[te], lr.predict_proba(X[te])[:, 1]))

        gb = GradientBoostingClassifier(random_state=seed, n_estimators=200,
                                        max_depth=3, learning_rate=0.05)
        gb.fit(X[tr], y[tr])
        out["gbm"].append(roc_auc_score(y[te], gb.predict_proba(X[te])[:, 1]))
        imp += gb.feature_importances_
    return out, imp / max(gss.get_n_splits(), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-k", type=int, default=5)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.rollouts) if l.strip()]
    print(f"episodes: {len(rows)}   tasks: {len({r['task_id'] for r in rows})}")
    succ = np.mean([r["final_f1"] > 0 for r in rows])
    print(f"base rate P(eventual success) = {succ:.3f}  "
          f"(a trivial classifier scores AUC 0.500)\n")

    report = {"rollouts": args.rollouts, "n_episodes": len(rows),
              "base_rate": float(succ), "gate": AUC_GATE, "by_k": {}}

    print(f"{'k':>3}{'n':>7}{'pos':>7}{'logreg AUC':>13}{'GBM AUC':>13}{'verdict':>10}")
    best = 0.0
    for k in range(1, args.max_k + 1):
        X, y, g, names = build(rows, k)
        if X is None or len(set(y)) < 2:
            print(f"{k:>3}   -- insufficient data --")
            continue
        aucs, imp = evaluate(X, y, g, names)
        lr_m, gb_m = float(np.mean(aucs["logreg"])), float(np.mean(aucs["gbm"]))
        lr_s, gb_s = float(np.std(aucs["logreg"])), float(np.std(aucs["gbm"]))
        m = max(lr_m, gb_m)
        best = max(best, m)
        print(f"{k:>3}{len(y):>7}{int(y.sum()):>7}"
              f"{lr_m:>9.3f}±{lr_s:.3f}{gb_m:>8.3f}±{gb_s:.3f}"
              f"{('PASS' if m >= AUC_GATE else 'below'):>10}")
        top = sorted(zip(names, imp), key=lambda t: -t[1])[:6]
        report["by_k"][k] = {
            "n": int(len(y)), "n_positive": int(y.sum()),
            "logreg_auc_mean": lr_m, "logreg_auc_std": lr_s,
            "gbm_auc_mean": gb_m, "gbm_auc_std": gb_s,
            "top_features": [{"name": n, "importance": float(v)} for n, v in top],
        }

    report["best_auc"] = best
    report["verdict"] = "PASS" if best >= AUC_GATE else "FAIL"
    (outdir / "s1_predictability.json").write_text(json.dumps(report, indent=2))

    print(f"\nbest held-out AUC across k: {best:.3f}   gate {AUC_GATE}")
    print(f"VERDICT: {report['verdict']}")
    if best < AUC_GATE:
        print("\nHopelessness is not predictable from gold-free state on this "
              "data. Per the plan, the dataset change is promoted from Step 3 to "
              "MANDATORY and Step 1 does not run as specified.")
    for k, d in report["by_k"].items():
        print(f"\n  k={k} top features: "
              + ", ".join(f"{f['name']}({f['importance']:.2f})"
                          for f in d["top_features"]))
    sys.exit(0 if best >= AUC_GATE else 1)


if __name__ == "__main__":
    main()
