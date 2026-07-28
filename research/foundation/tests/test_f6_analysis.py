"""I6 tests: figures render from synthetic CSVs; report fills numbers;
diagnostic parses."""

import json

import pandas as pd

from analysis.diagnostic_rubric import DIMENSIONS, diagnose_episode
from analysis.figures import fig1_frontier, fig2_internalization, fig3_divergence
from analysis.report import build_report
from eval.metrics import episode_row

LAM = 0.5


def synthetic_rows() -> pd.DataFrame:
    eps = []
    for i in range(15):
        for arm, f1, steps in (("a0", 0.50, 9), ("a1", 0.55, 8),
                               ("a2", 0.55, 6), ("a3", 0.62, 4)):
            for B in (3, 6, 10):
                eps.append({"task_id": f"t{i}", "arm": arm,
                            "mode": "enforce" if arm == "a2" else "none",
                            "budget_B": B, "rollout": 0, "final_f1": f1,
                            "final_em": 0.0, "steps_used": min(steps, B),
                            "answered_at": min(steps, B), "forced_stop": arm == "a2",
                            "config_hash": "x"})
    return pd.DataFrame([episode_row(e, LAM) for e in eps])


def cfg():
    return {"gate": {"budget": "medium", "min_self_stop": 0.70,
                     "f1_margin": 0.05, "bootstrap_resamples": 100},
            "economy": {"lambda": LAM},
            "data": {"dev_size": 15},
            "episode": {"budgets": {"small": 3, "medium": 6, "large": 10}}}


def test_figures_render(tmp_path):
    rows = synthetic_rows()
    p1 = fig1_frontier(rows, tmp_path, resamples=200)
    p2 = fig2_internalization(rows, tmp_path, resamples=200)
    div = tmp_path / "div.jsonl"
    div.write_text("".join(json.dumps({"step": s, "judge_score_mean": 0.5 + s / 100,
                                       "f1_mean": 0.5 + s / 200}) + "\n"
                           for s in range(0, 50, 10)))
    p3 = fig3_divergence(div, tmp_path)
    for p in (p1, p2, p3):
        assert p.exists() and p.stat().st_size > 1000


def test_report_contains_verdict_and_numbers():
    text = build_report(synthetic_rows(), cfg(), "2026-07-22",
                        divergence=[{"judge_score_mean": .5, "f1_mean": .4},
                                    {"judge_score_mean": .6, "f1_mean": .5}])
    assert "Gate verdict" in text and ("**GO**" in text or "**NO-GO**" in text)
    assert "| a3 |" in text and "Paired a3−a1" in text
    assert "divergence: judge 0.500→0.600" in text


def test_diagnostic_parses_and_handles_garbage():
    ep = {"question": "Who?", "final_answer": "X",
          "steps": [{"t": 1, "action_type": "search", "query_or_answer": "q",
                     "obs_digest": "o", "draft": "d", "draft_f1_vs_gold": 0.0}]}
    good = json.dumps({"reasoning": "fine", "score": 4})
    out = diagnose_episode(ep, lambda prompt: good)
    assert set(out) == set(DIMENSIONS)
    assert all(v["score"] == 4 for v in out.values())
    out_bad = diagnose_episode(ep, lambda prompt: "no json here")
    assert all(v["score"] is None for v in out_bad.values())
