"""SQLite layer tests — UPSERT dedup (the OpenCode double-count bug),
money-only spend, injection-state tracking. All against a temp DB."""
import pytest

import cost_aware_agent.db as db


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "db.sqlite")
    db.init_db()
    with db.get_conn() as c:
        yield c


def test_upsert_dedup_cumulative_push(conn):
    """OpenCode fires message.updated repeatedly per turn: same message id,
    growing CUMULATIVE tokens. Latest snapshot must win — summing snapshots
    double/triple-counted the turn (the shipped bug)."""
    db.insert_session(conn, "s1", "opencode", "", "m")
    db.insert_llm_usage(conn, "s1", "m", 100, 10, 0, 0, 0.001,
                        source="push", message_id="msg1")
    db.insert_llm_usage(conn, "s1", "m", 300, 30, 0, 0, 0.003,
                        source="push", message_id="msg1")
    rows = conn.execute("SELECT * FROM llm_usage WHERE session_id='s1'").fetchall()
    assert len(rows) == 1
    assert rows[0]["input_tokens"] == 300
    spent, _ = db.spent_usd(conn, "s1")
    assert spent == pytest.approx(0.003)  # latest, NOT 0.001 + 0.003


def test_null_message_id_rows_all_kept(conn):
    # partial unique index only applies WHERE message_id IS NOT NULL
    db.insert_session(conn, "s2", "x", "", "m")
    db.insert_llm_usage(conn, "s2", "m", 1, 1, 0, 0, 0.01, source="pull")
    db.insert_llm_usage(conn, "s2", "m", 1, 1, 0, 0, 0.02, source="pull")
    spent, _ = db.spent_usd(conn, "s2")
    assert spent == pytest.approx(0.03)


def test_spend_is_money_only_tool_calls_do_not_count(conn):
    """The core invariant: budget spend = real LLM dollars. A synthetic
    per-tool-call fee must never leak into spend (it did once)."""
    db.insert_session(conn, "s3", "x", "", "m")
    db.insert_llm_usage(conn, "s3", "m", 0, 0, 0, 0, 0.10, source="pull",
                        message_id="a")
    db.insert_tool_call(conn, "s3", "Read", "{}", "", 0.001)
    db.insert_tool_call(conn, "s3", "Grep", "{}", "", 0.001)
    spent, tool_count = db.spent_usd(conn, "s3")
    assert spent == pytest.approx(0.10)   # not 0.102
    assert tool_count == 2


def test_inject_state_roundtrip(conn):
    db.insert_session(conn, "s4", "x", "", "m")
    assert db.get_inject_state(conn, "s4") == (None, None, None)
    db.set_inject_state(conn, "s4", "MEDIUM", 3, 0.35)
    assert db.get_inject_state(conn, "s4") == ("MEDIUM", 3, 0.35)


def test_inject_state_missing_session(conn):
    assert db.get_inject_state(conn, "nope") == (None, None, None)


def test_insert_session_is_idempotent(conn):
    db.insert_session(conn, "s5", "claude-code", "task", "m")
    db.insert_session(conn, "s5", "", "", "")  # hook refire must not clobber
    row = db.get_session(conn, "s5")
    assert row["cli"] == "claude-code"
