"""SQLite layer. One schema note worth flagging: `llm_usage.message_id` is
needed to dedupe Claude Code transcript parsing — one model turn writes
multiple JSONL lines sharing a single `message.id`, all with an identical,
already-complete usage block. Deduping by anything else double- or
triple-counts that turn's cost.
"""

import sqlite3
import time
from contextlib import contextmanager

from cost_aware_agent.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    cli TEXT,
    task TEXT,
    model TEXT,
    start_time INTEGER,
    state TEXT DEFAULT 'active',
    low_tier_continue_streak INTEGER DEFAULT 0,
    created_at INTEGER
);

CREATE TABLE IF NOT EXISTS llm_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(session_id),
    message_id TEXT,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_creation_tokens INTEGER,      -- total cache-write tokens (5m + 1h)
    cache_creation_1h_tokens INTEGER DEFAULT 0,  -- portion billed at the 1h ephemeral rate; see cost.py
    cost_usd REAL,
    source TEXT,
    created_at INTEGER
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_llm_usage_session_message
    ON llm_usage(session_id, message_id) WHERE message_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(session_id),
    tool_name TEXT,
    tool_input TEXT,
    tool_result_excerpt TEXT,
    cost_usd REAL,
    created_at INTEGER
);

CREATE TABLE IF NOT EXISTS plan (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(session_id),
    clue_type TEXT,
    clue_text TEXT,
    status TEXT DEFAULT 'pending',
    created_at INTEGER,
    updated_at INTEGER
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = _connect()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def now() -> int:
    return int(time.time())


# --- sessions ---

def insert_session(conn, session_id: str, cli: str, task: str, model: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sessions "
        "(session_id, cli, task, model, start_time, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, cli, task, model, now(), now()),
    )


def get_session(conn, session_id: str):
    return conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()


def mark_session_ended(conn, session_id: str) -> None:
    conn.execute(
        "UPDATE sessions SET state = 'ended' WHERE session_id = ?", (session_id,)
    )


def set_streak(conn, session_id: str, value: int) -> None:
    conn.execute(
        "UPDATE sessions SET low_tier_continue_streak = ? WHERE session_id = ?",
        (value, session_id),
    )


# --- llm_usage ---

def insert_llm_usage(
    conn, session_id: str, model: str, input_tokens: int, output_tokens: int,
    cache_read_tokens: int, cache_creation_tokens: int, cost_usd: float,
    source: str, message_id: str | None = None, cache_creation_1h_tokens: int = 0,
) -> None:
    conn.execute(
        "INSERT INTO llm_usage "
        "(session_id, message_id, model, input_tokens, output_tokens, "
        " cache_read_tokens, cache_creation_tokens, cache_creation_1h_tokens, cost_usd, source, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        # OpenCode's push path fires message.updated repeatedly per assistant turn,
        # each carrying the SAME message id with growing *cumulative* token counts.
        # Latest-wins UPSERT keeps one row per message at its final totals instead of
        # summing every streamed snapshot (which double/triple-counted the turn). The
        # partial unique index (session_id, message_id) only exists WHERE message_id
        # IS NOT NULL, so message_id-less rows still plain-insert.
        "ON CONFLICT(session_id, message_id) WHERE message_id IS NOT NULL "
        "DO UPDATE SET model=excluded.model, input_tokens=excluded.input_tokens, "
        " output_tokens=excluded.output_tokens, cache_read_tokens=excluded.cache_read_tokens, "
        " cache_creation_tokens=excluded.cache_creation_tokens, "
        " cache_creation_1h_tokens=excluded.cache_creation_1h_tokens, cost_usd=excluded.cost_usd",
        (session_id, message_id, model, input_tokens, output_tokens,
         cache_read_tokens, cache_creation_tokens, cache_creation_1h_tokens, cost_usd, source, now()),
    )


# --- tool_calls ---

def insert_tool_call(
    conn, session_id: str, tool_name: str, tool_input: str,
    tool_result_excerpt: str, cost_usd: float,
) -> None:
    conn.execute(
        "INSERT INTO tool_calls "
        "(session_id, tool_name, tool_input, tool_result_excerpt, cost_usd, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, tool_name, tool_input, tool_result_excerpt, cost_usd, now()),
    )


# --- plan ---

def insert_plan_item(conn, session_id: str, clue_type: str, clue_text: str) -> int:
    cur = conn.execute(
        "INSERT INTO plan (session_id, clue_type, clue_text, status, created_at, updated_at) "
        "VALUES (?, ?, ?, 'pending', ?, ?)",
        (session_id, clue_type, clue_text, now(), now()),
    )
    return cur.lastrowid


def get_plan(conn, session_id: str):
    return conn.execute(
        "SELECT * FROM plan WHERE session_id = ? ORDER BY id", (session_id,)
    ).fetchall()


def update_plan_status(conn, session_id: str, item_id: int, status: str) -> None:
    conn.execute(
        "UPDATE plan SET status = ?, updated_at = ? WHERE session_id = ? AND id = ?",
        (status, now(), session_id, item_id),
    )


# --- aggregation ---

def spent_usd(conn, session_id: str) -> tuple[float, int]:
    """Returns (spent_usd, tool_calls count). Money-only budget: spend is the
    REAL LLM dollar cost of this session, NOT a synthetic per-tool-call charge.
    The old behaviour added SUM(tool_calls.cost) (a fixed tool_call_price_usd
    per call) into the total, which contradicted the money-only design and, on
    small budgets, folded a large non-LLM term into the injected 'LLM cost used'
    pressure. tool_calls are still recorded (and their count is returned for the
    tracker's 'Tool calls used: N' line) but do not contribute to spend."""
    llm = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM llm_usage WHERE session_id = ?",
        (session_id,),
    ).fetchone()["s"]
    tool_call_count = conn.execute(
        "SELECT COUNT(*) AS c FROM tool_calls WHERE session_id = ?",
        (session_id,),
    ).fetchone()["c"]
    return float(llm), tool_call_count
