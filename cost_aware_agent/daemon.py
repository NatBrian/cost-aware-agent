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
    plan_seeded = len(db.get_plan(conn, session_id)) > 0
    # Task-tool subagents write SEPARATE transcripts (see
    # transcript.subagent_transcript_paths) — skipping them leaves all subagent
    # LLM spend unmeasured. Their spend lands on the parent session; only the
    # parent's own text drives checklist seeding / verification parsing.
    paths = [(transcript_path, True)] + [
        (p, False) for p in transcript.subagent_transcript_paths(transcript_path)
    ]
    for path, is_parent in paths:
        turns = transcript.parse_new_assistant_turns(path, seen)
        for turn in turns:
            seen.add(turn["message_id"])
            if is_parent and not plan_seeded:
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
            # zero-token rows (CC synthetic '<synthetic>' entries) spent nothing —
            # flagging them would put a permanent spurious warning in the tracker.
            total_tokens = (usage["input_tokens"] + usage["output_tokens"]
                            + usage["cache_read_tokens"] + usage["cache_creation_tokens"])
            _, price_unknown = cost.resolve_price(model)
            price_unknown = price_unknown and total_tokens > 0
            db.insert_llm_usage(
                conn, session_id, model,
                usage["input_tokens"], usage["output_tokens"],
                usage["cache_read_tokens"], usage["cache_creation_tokens"],
                c, source="pull" if is_parent else "pull-subagent",
                message_id=turn["message_id"],
                cache_creation_1h_tokens=cache_1h,
                price_unknown=price_unknown,
            )
            if is_parent and transcript.has_verification_block(turn["text"]):
                _handle_verification(conn, session_id, turn["text"])


def _handle_verification(conn, session_id: str, raw_response: str) -> Optional[str]:
    items, decision = prompts.parse_verification(raw_response)
    if decision is None:
        return None  # malformed — log-and-skip per parsing contract, no retry/block
    for item_id, status in items:
        db.update_plan_status(conn, session_id, item_id, status)

    spent, budget, _scope = _budget_view(conn, session_id)
    tier, _ = cost.compute_tier(spent, budget)

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


def _deliver(conn, session_id: str, endpoint: str, context):
    """Gate + audit-log in one step. Every context actually handed to an
    adapter lands in the `injections` table, so a run can be replayed/debugged
    from the daemon DB alone (which text, which endpoint, when) without relying
    on client-side logs."""
    delivered = _gate(context)
    if delivered:
        db.insert_injection(conn, session_id, endpoint, delivered)
    return delivered


def _budget_view(conn, session_id: str) -> tuple[float, float, str]:
    """(spent, budget, scope_label) for the session's budget pressure.

    Resolution order:
      1. per-session override sent on /session/start (experiments switch
         conditions without a daemon restart)
      2. project wallet — ONE user number ("$10 for my project") depleting
         across every session of the project; spend/budget are project-level
      3. config default (per-session estimate)
    scope_label feeds the tracker text so the model knows whether the number
    is this session's estimate or the project's remaining wallet."""
    session = db.get_session(conn, session_id)
    if session is not None:
        if session["budget_override_usd"] is not None:
            spent, _ = db.spent_usd(conn, session_id)
            return spent, float(session["budget_override_usd"]), "session estimate"
        wallet = db.get_wallet(conn, session["project_dir"])
        if wallet is not None:
            spent = db.wallet_spent_usd(conn, session["project_dir"])
            return spent, float(wallet["budget_usd"]), "project wallet"
    spent, _ = db.spent_usd(conn, session_id)
    return spent, _config["session_budget_estimate_usd"], "session estimate"


def _spend_bucket(spent: float, budget: float) -> int:
    """Which inject_spend_bucket_pct slice of the budget spend sits in. Past
    100% the cadence coarsens to one bucket per HALF-budget overspent: the
    session still gets periodic over-budget reminders, but not one per tiny
    slice (a $0.10 budget would otherwise re-inject every extra cent —
    reintroducing the per-call token tax exactly when spend is already high)."""
    if budget <= 0:
        return 0
    pct_spent = (spent / budget) * 100
    pct = max(1, int(_config.get("inject_spend_bucket_pct", 10)))
    if pct_spent <= 100:
        return int(pct_spent / pct)
    return int(100 / pct) + int((pct_spent - 100) / 50)


def _tracker_context(conn, session_id: str, channel: str = "accumulating",
                     lagging: bool = False) -> Optional[str]:
    # Budget is ALWAYS money: the real dollar cost of LLM API calls this session
    # versus the session's dollar budget. There is no tool-call budget mode —
    # cost is money, not a count of actions.
    _, tool_calls_used = db.spent_usd(conn, session_id)
    plan_rows = db.get_plan(conn, session_id)
    spent, budget, scope = _budget_view(conn, session_id)
    tier, _ = cost.compute_tier(spent, budget)
    # flips at most once per session (0 -> 1), so it does not violate the
    # rebuilt channel's byte-stability requirement
    price_unknown = db.session_has_unknown_priced_usage(conn, session_id)

    # Injection policy — both branches exist to stop the harness taxing the
    # session it meters (both taxes were MEASURED, not hypothetical):
    #
    # Accumulating channels (Claude Code additionalContext — every injection
    # persists in the conversation and taxes all later turns; +44% session cost
    # when fired per-tool-call): inject only when the signal CHANGED — new
    # tier, or spend crossed another budget slice.
    #
    # Rebuilt channels (OpenCode's system array is reconstructed per LLM call):
    # the tracker must be PRESENT every call (suppression would delete the
    # budget from context), but its BYTES must not change per call — an exact
    # running total invalidates the provider's prompt cache from that point,
    # which on deepseek halved the cache-hit rate and cost +92%. So spend is
    # quantized to the current injection bucket's floor and the per-call tool
    # counter is omitted: text mutates only on bucket/tier transitions.
    if channel == "rebuilt":
        bucket = _spend_bucket(spent, budget)
        pct = max(1, int(_config.get("inject_spend_bucket_pct", 10)))
        spent_display = (bucket * pct / 100.0) * budget if budget > 0 else spent
        return prompts.render_budget_tracker(
            spent_display, budget, None, tier, plan_rows, lagging=lagging,
            approximate=True, price_unknown=price_unknown, scope=scope,
        )

    if _config.get("inject_mode", "on_change") == "on_change":
        bucket = _spend_bucket(spent, budget)
        last_tier, last_bucket = db.get_inject_state(conn, session_id)
        if tier == last_tier and bucket == last_bucket:
            return None
        db.set_inject_state(conn, session_id, tier, bucket)

    return prompts.render_budget_tracker(
        spent, budget, tool_calls_used, tier, plan_rows, lagging=lagging,
        price_unknown=price_unknown, scope=scope,
    )


# --- request/response models ---

class SessionStartReq(BaseModel):
    session_id: str
    cli: str
    task: str = ""
    model: str = ""
    transcript_path: Optional[str] = None
    # links the session to a project wallet (adapters send the project cwd)
    project_dir: Optional[str] = None
    # explicit per-session budget — beats wallet and config; lets experiments
    # switch conditions per session instead of restarting the daemon per config
    budget_usd: Optional[float] = None


class ToolPreReq(BaseModel):
    session_id: str
    tool_name: str
    transcript_path: Optional[str] = None
    # "accumulating" (default): injected text persists in the conversation
    # (Claude Code additionalContext) — subject to on_change suppression.
    # "rebuilt": the channel is reconstructed every LLM call (OpenCode system
    # transform) — always receives the current tracker.
    channel: str = "accumulating"


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
        db.set_session_budget_source(conn, req.session_id, req.project_dir, req.budget_usd)
        _ingest_transcript(conn, req.session_id, req.transcript_path)
        context = _deliver(conn, req.session_id, "session/start", prompts.PLANNING_PROMPT)
    return {"additionalContext": context}


@app.post("/tool/pre")
def tool_pre(req: ToolPreReq):
    with db.get_conn() as conn:
        db.insert_session(conn, req.session_id, "", "", "")
        _ingest_transcript(conn, req.session_id, req.transcript_path)
        # transcript_path == pull-based capture: spend lags the current turn
        # (Claude Code has no per-turn usage hook), so the tracker must say so
        # rather than present a stale number as current.
        context = _tracker_context(conn, req.session_id, channel=req.channel,
                                   lagging=bool(req.transcript_path))
        context = _deliver(conn, req.session_id, "tool/pre", context)
    return {"additionalContext": context}


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
        context = _deliver(conn, req.session_id, "tool/post", context)
    return {"additionalContext": context}


@app.post("/verification/result")
def verification_result(req: VerificationResultReq):
    with db.get_conn() as conn:
        db.insert_session(conn, req.session_id, "", "", "")
        context = _handle_verification(conn, req.session_id, req.raw_response)
        # Must gate like every other endpoint: when inject_enabled is False (the
        # A/B OFF arm) this streak nudge must be suppressed too, else the OFF
        # control leaks injected [STREAK] text on the OpenCode push path.
        context = _deliver(conn, req.session_id, "verification/result", context)
    return {"additionalContext": context}


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
    total_tokens = (u.get("input_tokens", 0) + u.get("output_tokens", 0)
                    + u.get("cache_read_tokens", 0) + cache_creation_total)
    _, price_unknown = cost.resolve_price(req.model)
    with db.get_conn() as conn:
        db.insert_session(conn, req.session_id, "", "", req.model)
        db.insert_llm_usage(
            conn, req.session_id, req.model,
            u.get("input_tokens", 0), u.get("output_tokens", 0),
            u.get("cache_read_tokens", 0), cache_creation_total,
            c, source="push", message_id=req.message_id,
            cache_creation_1h_tokens=cache_1h,
            price_unknown=price_unknown and total_tokens > 0,
        )
    return {"additionalContext": None}


@app.post("/session/stop")
def session_stop(req: SessionStopReq):
    with db.get_conn() as conn:
        db.insert_session(conn, req.session_id, "", "", "")
        _ingest_transcript(conn, req.session_id, req.transcript_path)
        db.mark_session_ended(conn, req.session_id)
    return {"additionalContext": None}


@app.get("/session/{session_id}/dump")
def session_dump(session_id: str):
    """Full per-session export for experiment archival / debugging: every
    llm_usage row, tool call, delivered injection, plan item, and the session
    row itself. This is the 'all events retrievable' contract."""
    with db.get_conn() as conn:
        def rows(sql):
            return [dict(r) for r in conn.execute(sql, (session_id,)).fetchall()]
        session = db.get_session(conn, session_id)
        session_spent, _ = db.spent_usd(conn, session_id)
        view_spent, budget, scope = _budget_view(conn, session_id)
        tier, _ = cost.compute_tier(view_spent, budget)
        dump = {
            "session": dict(session) if session else None,
            # this session's own spend — stable meaning regardless of wallet
            "spent_usd": session_spent,
            # what the budget pressure was computed from (wallet-level when a
            # project wallet is active, session-level otherwise)
            "budget_view": {"spent_usd": view_spent, "budget_usd": budget,
                            "scope": scope},
            "tier": tier,
            "llm_usage": rows("SELECT * FROM llm_usage WHERE session_id = ? ORDER BY id"),
            "tool_calls": rows("SELECT * FROM tool_calls WHERE session_id = ? ORDER BY id"),
            "injections": rows("SELECT * FROM injections WHERE session_id = ? ORDER BY id"),
            "plan": rows("SELECT * FROM plan WHERE session_id = ? ORDER BY id"),
        }
    return dump


@app.get("/status/{session_id}")
def status(session_id: str):
    with db.get_conn() as conn:
        session_spent, _ = db.spent_usd(conn, session_id)
        view_spent, budget, scope = _budget_view(conn, session_id)
        tier, _ = cost.compute_tier(view_spent, budget)
        plan_rows = db.get_plan(conn, session_id)
        plan = [
            {"id": r["id"], "type": r["clue_type"], "text": r["clue_text"], "status": r["status"]}
            for r in plan_rows
        ]
    return {"spent_usd": session_spent,
            "budget_view": {"spent_usd": view_spent, "budget_usd": budget,
                            "scope": scope},
            "tier": tier, "plan": plan}
