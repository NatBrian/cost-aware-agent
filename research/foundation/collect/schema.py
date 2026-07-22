"""Episode JSONL schema validator (F2) — every stage's contract with the next."""

REQUIRED_EPISODE = {
    "task_id": str, "arm": str, "mode": str, "budget_B": int, "seed": int,
    "config_hash": str, "steps": list, "forced_stop": bool,
    "final_answer": str, "final_f1": float, "final_em": float,
    "steps_used": int, "total_steps_run": int,
}
REQUIRED_STEP = {
    "t": int, "action_type": str, "query_or_answer": str, "obs_digest": str,
    "draft": str, "draft_f1_vs_gold": float,
}
ARMS = ("a0", "a1", "a2", "a3")
ACTIONS = ("search", "answer", "malformed")


def validate_episode(ep: dict) -> None:
    """Raises ValueError with a precise message on first violation."""
    for key, typ in REQUIRED_EPISODE.items():
        if key not in ep:
            raise ValueError(f"episode missing key: {key}")
        if not isinstance(ep[key], typ):
            raise ValueError(f"episode[{key}] is {type(ep[key]).__name__}, want {typ.__name__}")
    if "answered_at" not in ep:
        raise ValueError("episode missing key: answered_at")
    if ep["arm"] not in ARMS:
        raise ValueError(f"unknown arm: {ep['arm']}")
    if not ep["steps"]:
        raise ValueError("episode has no steps")
    if not (0.0 <= ep["final_f1"] <= 1.0):
        raise ValueError(f"final_f1 out of range: {ep['final_f1']}")
    for i, s in enumerate(ep["steps"]):
        for key, typ in REQUIRED_STEP.items():
            if key not in s:
                raise ValueError(f"step {i} missing key: {key}")
            if not isinstance(s[key], typ):
                raise ValueError(f"step {i}[{key}] wrong type")
        if s["action_type"] not in ACTIONS:
            raise ValueError(f"step {i} unknown action_type: {s['action_type']}")
        if s["t"] != i + 1:
            raise ValueError(f"step {i} has t={s['t']}, want {i + 1}")
    if ep["steps_used"] > ep["total_steps_run"]:
        raise ValueError("steps_used > total_steps_run")
