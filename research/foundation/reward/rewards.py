"""Reward computation (F3/F5): judge bits -> per-step rewards; terminal economy;
returns-to-go for step-level credit assignment.

Economy (one formula everywhere, from config):
  r_t      = alpha * (weighted_bits - 0.5)          per step, judge-graded
  R_final  = F1(answer, gold) - lambda * (steps_used / B) + fmt_w * format_ok
  R_t(rtg) = sum_{t'>=t} r_t' + R_final             (credit for step t)
"""

from reward.rubric import (ANSWER_BITS, STEP_BITS, render_answer_prompt,
                           render_step_prompt, step_reward)


def format_ok(ep: dict) -> float:
    """1.0 iff no malformed steps and a non-empty final answer."""
    clean = all(s["action_type"] != "malformed" for s in ep["steps"])
    return float(clean and bool(ep["final_answer"].strip()))


def terminal_reward(ep: dict, lam: float, fmt_w: float) -> float:
    return (ep["final_f1"]
            - lam * (ep["steps_used"] / max(1, ep["budget_B"]))
            + fmt_w * format_ok(ep))


def judge_episode_steps(ep: dict, judge) -> list[dict]:
    """One bits-dict per step (working steps get STEP_BITS, ANSWER steps get
    ANSWER_BITS; malformed steps get all-zero bits — worst score, by design:
    a malformed step did nothing and cost a step)."""
    out = []
    for i, s in enumerate(ep["steps"]):
        if s["action_type"] == "search":
            out.append(judge.judge(render_step_prompt(ep, i), STEP_BITS))
        elif s["action_type"] == "answer":
            out.append(judge.judge(render_answer_prompt(ep, i), ANSWER_BITS))
        else:
            out.append({b: 0 for b in STEP_BITS})
    return out


def step_rewards(ep: dict, bits_per_step: list[dict], rubric_cfg: dict) -> list[float]:
    alpha = rubric_cfg["alpha"]
    rs = []
    for s, bits in zip(ep["steps"], bits_per_step):
        weights = (rubric_cfg["answer_bits"] if s["action_type"] == "answer"
                   else rubric_cfg["step_bits"])
        if s["action_type"] == "malformed":
            rs.append(step_reward({b: 0 for b in weights}, weights, alpha))
        else:
            rs.append(step_reward(bits, weights, alpha))
    return rs


def returns_to_go(rs: list[float], r_final: float) -> list[float]:
    """R_t = sum_{t'>=t} r_t' + R_final (worked example: plan §4 discussion)."""
    out, acc = [], r_final
    for r in reversed(rs):
        acc += r
        out.append(acc)
    return list(reversed(out))


def training_lambda(cfg: dict) -> float:
    """The λ the REWARD is computed with, which is not necessarily the λ results
    are SCORED with.

    The λ ablation trains arms at λ ∈ {0, 0.3, 1.0} but must score them all on
    one fixed yardstick, or "which arm has higher utility" just measures which
    yardstick was used. `economy.train_lambda` moves for the experiment;
    `economy.lambda` stays put for evaluation and reporting. Absent the key,
    they are the same number and nothing changes. (2026-07-29)
    """
    return float(cfg["economy"].get("train_lambda", cfg["economy"]["lambda"]))


def episode_rewards(ep: dict, judge, cfg: dict) -> dict:
    """Everything F5 needs for one trajectory."""
    bits = judge_episode_steps(ep, judge)
    rs = step_rewards(ep, bits, cfg["rubric"])
    r_final = terminal_reward(ep, training_lambda(cfg),
                              cfg["reward"]["format_weight"])
    return {"bits": bits, "step_rewards": rs, "r_final": r_final,
            "returns_to_go": returns_to_go(rs, r_final)}
