"""FastAPI daemon — the process both CLI adapters talk to over HTTP.

One deliberate simplification for Claude Code: verification-block detection
happens server-side during transcript ingestion (the daemon already has the
full turn text there), instead of the bash adapter scanning JSONL text
itself with jq. /verification/result stays live for OpenCode's push-based
adapter, which hands text over directly.
"""

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from cost_aware_agent import cost, db, prompts, transcript
from cost_aware_agent.config import load_config

app = FastAPI()
_config = load_config()


@app.on_event("startup")
def _startup():
    db.init_db()


# --- shared helpers ---

def _seed_checklist_if_new(conn, session_id: str, text: str) -> bool:
    """Parses a <checklist> block out of free-form model text and seeds
    `plan` on first sighting only. Shared by CC's transcript pull and
    OpenCode's push-based /plan/seed — the plan table must be seeded from
    exactly one source per session, so a session that already has plan rows
    is always a no-op here regardless of caller."""
    if len(db.get_plan(conn, session_id)) > 0:
        return False
    checklist_items = prompts.parse_checklist(text)
    if not checklist_items:
        return False
    for clue_type, clue_text in checklist_items:
        db.insert_plan_item(conn, session_id, clue_type, clue_text)
    return True


def _ingest_transcript(conn, session_id: str, transcript_path: Optional[str]) -> None:
    if not transcript_path:
        return
    seen_rows = conn.execute(
        "SELECT message_id FROM llm_usage WHERE session_id = ? AND message_id IS NOT NULL",
        (session_id,),
    ).fetchall()
    seen = {r["message_id"] for r in seen_rows}
    turns = transcript.parse_new_assistant_turns(transcript_path, seen)
    plan_seeded = len(db.get_plan(conn, session_id)) > 0
    for turn in turns:
        if not plan_seeded:
            plan_seeded = _seed_checklist_if_new(conn, session_id, turn["text"])

        model = turn["model"]
        usage = turn["usage"]
        cache_1h = usage["cache_creation_1h_tokens"]
        # clamp: a malformed payload with 1h > total must not yield negative 5m
        # tokens (which would *reduce* computed cost).
        cache_5m = max(0, usage["cache_creation_tokens"] - cache_1h)
        c = cost.cost_llm_usage(
            model,
            usage["input_tokens"], usage["output_tokens"],
            usage["cache_read_tokens"], cache_5m, cache_1h,
        )
        db.insert_llm_usage(
            conn, session_id, model,
            usage["input_tokens"], usage["output_tokens"],
            usage["cache_read_tokens"], usage["cache_creation_tokens"],
            c, source="pull", message_id=turn["message_id"],
            cache_creation_1h_tokens=cache_1h,
        )
        if transcript.has_verification_block(turn["text"]):
            _handle_verification(conn, session_id, turn["text"])


def _handle_verification(conn, session_id: str, raw_response: str) -> Optional[str]:
    items, decision = prompts.parse_verification(raw_response)
    if decision is None:
        return None  # malformed — log-and-skip per parsing contract, no retry/block
    for item_id, status in items:
        db.update_plan_status(conn, session_id, item_id, status)

    spent, _ = db.spent_usd(conn, session_id)
    tier, _ = cost.compute_tier(spent, _config["session_budget_estimate_usd"])

    session = db.get_session(conn, session_id)
    current_streak = session["low_tier_continue_streak"] if session else 0
    if decision == "CONTINUE" and tier in ("LOW", "CRITICAL"):
        new_streak = current_streak + 1
    else:
        new_streak = 0
    db.set_streak(conn, session_id, new_streak)

    if new_streak >= _config["streak_warning_threshold"]:
        return prompts.streak_fact(new_streak)
    return None


def _gate(context):
    """Injection master switch. When inject_enabled is False (the A/B OFF arm),
    swallow every additionalContext so the model sees nothing, while all db
    recording upstream of this call still happens — identical measurement, no
    injection."""
    return context if _config.get("inject_enabled", True) else None


def _tracker_context(conn, session_id: str) -> str:
    # Budget is ALWAYS money: the real dollar cost of LLM API calls this session
    # versus the session's dollar budget. There is no tool-call budget mode —
    # cost is money, not a count of actions.
    spent, tool_calls_used = db.spent_usd(conn, session_id)
    plan_rows = db.get_plan(conn, session_id)
    tier, _ = cost.compute_tier(spent, _config["session_budget_estimate_usd"])
    return prompts.render_budget_tracker(
        spent, _config["session_budget_estimate_usd"], tool_calls_used, tier, plan_rows,
    )


# --- request/response models ---

class SessionStartReq(BaseModel):
    session_id: str
    cli: str
    task: str = ""
    model: str = ""
    transcript_path: Optional[str] = None


class ToolPreReq(BaseModel):
    session_id: str
    tool_name: str
    transcript_path: Optional[str] = None


class ToolPostReq(BaseModel):
    session_id: str
    tool_name: str
    tool_input: dict = {}
    tool_result: str = ""
    transcript_path: Optional[str] = None


class VerificationResultReq(BaseModel):
    session_id: str
    raw_response: str


class PlanSeedReq(BaseModel):
    session_id: str
    raw_response: str


class LlmUsageReq(BaseModel):
    session_id: str
    model: str
    usage: dict
    message_id: Optional[str] = None


class SessionStopReq(BaseModel):
    session_id: str
    transcript_path: Optional[str] = None


# --- endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/session/start")
def session_start(req: SessionStartReq):
    with db.get_conn() as conn:
        db.insert_session(conn, req.session_id, req.cli, req.task, req.model)
        _ingest_transcript(conn, req.session_id, req.transcript_path)
    return {"additionalContext": _gate(prompts.PLANNING_PROMPT)}


@app.post("/tool/pre")
def tool_pre(req: ToolPreReq):
    with db.get_conn() as conn:
        db.insert_session(conn, req.session_id, "", "", "")
        _ingest_transcript(conn, req.session_id, req.transcript_path)
        context = _tracker_context(conn, req.session_id)
    return {"additionalContext": _gate(context)}


@app.post("/tool/post")
def tool_post(req: ToolPostReq):
    with db.get_conn() as conn:
        db.insert_session(conn, req.session_id, "", "", "")
        _ingest_transcript(conn, req.session_id, req.transcript_path)
        db.insert_tool_call(
            conn, req.session_id, req.tool_name,
            str(req.tool_input)[:2000], (req.tool_result or "")[:500],
            cost.cost_tool_call(_config),
        )
        milestone = prompts.is_milestone(req.tool_name, req.tool_input, _config)
    context = prompts.SELF_VERIFICATION_PROMPT if milestone else None
    return {"additionalContext": _gate(context)}


@app.post("/verification/result")
def verification_result(req: VerificationResultReq):
    with db.get_conn() as conn:
        db.insert_session(conn, req.session_id, "", "", "")
        context = _handle_verification(conn, req.session_id, req.raw_response)
    # Must go through _gate like every other endpoint: when inject_enabled is
    # False (the A/B OFF arm) this streak nudge must be suppressed too, else the
    # OFF control leaks injected [STREAK] text on the OpenCode push path.
    return {"additionalContext": _gate(context)}


@app.post("/plan/seed")
def plan_seed(req: PlanSeedReq):
    with db.get_conn() as conn:
        db.insert_session(conn, req.session_id, "", "", "")
        _seed_checklist_if_new(conn, req.session_id, req.raw_response)
    return {"additionalContext": None}


@app.post("/llm/usage")
def llm_usage(req: LlmUsageReq):
    u = req.usage
    cache_creation_total = u.get("cache_creation_tokens", 0)
    # OpenCode's message.updated payload does not expose a 1h/5m cache-write
    # split the way Claude Code's transcript does (confirmed: its usage
    # shape is a flat {read, write}). Fall back to the 5m rate for the whole
    # amount — conservative (undercounts if OpenCode also defaults to 1h
    # caching), not a guess in the expensive direction.
    cache_1h = u.get("cache_creation_1h_tokens", 0)
    cache_5m = max(0, cache_creation_total - cache_1h)
    c = cost.cost_llm_usage(
        req.model,
        u.get("input_tokens", 0), u.get("output_tokens", 0),
        u.get("cache_read_tokens", 0), cache_5m, cache_1h,
    )
    with db.get_conn() as conn:
        db.insert_session(conn, req.session_id, "", "", req.model)
        db.insert_llm_usage(
            conn, req.session_id, req.model,
            u.get("input_tokens", 0), u.get("output_tokens", 0),
            u.get("cache_read_tokens", 0), cache_creation_total,
            c, source="push", message_id=req.message_id,
            cache_creation_1h_tokens=cache_1h,
        )
    return {"additionalContext": None}


@app.post("/session/stop")
def session_stop(req: SessionStopReq):
    with db.get_conn() as conn:
        db.insert_session(conn, req.session_id, "", "", "")
        _ingest_transcript(conn, req.session_id, req.transcript_path)
        db.mark_session_ended(conn, req.session_id)
    return {"additionalContext": None}


@app.get("/status/{session_id}")
def status(session_id: str):
    with db.get_conn() as conn:
        spent, _ = db.spent_usd(conn, session_id)
        tier, _ = cost.compute_tier(spent, _config["session_budget_estimate_usd"])
        plan_rows = db.get_plan(conn, session_id)
        plan = [
            {"id": r["id"], "type": r["clue_type"], "text": r["clue_text"], "status": r["status"]}
            for r in plan_rows
        ]
    return {"spent_usd": spent, "tier": tier, "plan": plan}
