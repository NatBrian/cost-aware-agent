"""Burn rate — measured trailing spend in the tracker (backlog item 3).
Measurement only: sum of llm_usage rows inside the window; no prediction."""
import pytest

import cost_aware_agent.db as db
from cost_aware_agent import prompts


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "db.sqlite")
    db.init_db()
    with db.get_conn() as c:
        yield c


def _spend_at(conn, session_id, cost_usd, message_id, created_at):
    db.insert_llm_usage(conn, session_id, "claude-sonnet-5", 100, 10, 0, 0,
                        cost_usd, source="pull", message_id=message_id)
    conn.execute("UPDATE llm_usage SET created_at = ? WHERE message_id = ?",
                 (created_at, message_id))


def test_recent_spend_only_counts_rows_inside_window(conn):
    db.insert_session(conn, "s1", "claude-code", "", "")
    t = db.now()
    _spend_at(conn, "s1", 0.50, "old", t - 700)    # outside a 600s window
    _spend_at(conn, "s1", 0.20, "new", t - 60)     # inside
    assert db.recent_spend_usd(conn, "s1", 600) == pytest.approx(0.20)
    # total spend still counts both — burn is a view, not a replacement
    spent, _ = db.spent_usd(conn, "s1")
    assert spent == pytest.approx(0.70)


def test_recent_spend_scoped_to_session(conn):
    db.insert_session(conn, "s1", "claude-code", "", "")
    db.insert_session(conn, "s2", "claude-code", "", "")
    t = db.now()
    _spend_at(conn, "s1", 0.30, "a", t)
    _spend_at(conn, "s2", 0.90, "b", t)
    assert db.recent_spend_usd(conn, "s1", 600) == pytest.approx(0.30)


def test_tracker_burn_line_rendering():
    with_burn = prompts.render_budget_tracker(
        0.5, 1.0, 3, "MEDIUM", [], burn_usd=0.12, burn_window_secs=600)
    assert "Burn rate: $0.12 spent in the last 10 min" in with_burn
    # zero burn (or feature disabled) renders no line — no noise at idle
    no_burn = prompts.render_budget_tracker(
        0.5, 1.0, 3, "MEDIUM", [], burn_usd=0.0, burn_window_secs=600)
    assert "Burn rate" not in no_burn
    omitted = prompts.render_budget_tracker(0.5, 1.0, 3, "MEDIUM", [])
    assert "Burn rate" not in omitted


def test_tracker_burn_line_approximate_for_rebuilt_channel():
    text = prompts.render_budget_tracker(
        0.5, 1.0, None, "MEDIUM", [], approximate=True,
        burn_usd=0.10, burn_window_secs=600)
    assert "Burn rate: ~$0.10" in text
