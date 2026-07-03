"""End-of-session receipt (backlog item 8) and measured session history
injection (backlog item 6)."""
import pytest
from fastapi.testclient import TestClient

import cost_aware_agent.db as db
from cost_aware_agent import daemon, prompts


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "db.sqlite")
    monkeypatch.setitem(daemon._config, "inject_enabled", True)
    monkeypatch.setitem(daemon._config, "enable_plan_verification", False)
    c = TestClient(daemon.app)
    db.init_db()
    return c


PROJ = "/home/user/proj"


def _run_session(client, sid, input_tokens):
    client.post("/session/start", json={
        "session_id": sid, "cli": "claude-code", "project_dir": PROJ})
    client.post("/llm/usage", json={
        "session_id": sid, "model": "claude-sonnet-5", "message_id": f"{sid}-m",
        "usage": {"input_tokens": input_tokens, "output_tokens": 0}})
    client.post("/session/stop", json={"session_id": sid})


# --- item 6: session history ---

def test_first_session_gets_no_history_line(client):
    r = client.post("/session/start", json={
        "session_id": "h0", "cli": "claude-code", "project_dir": PROJ})
    assert r.json()["additionalContext"] is None  # no history -> no guess


def test_later_sessions_get_measured_history(client):
    _run_session(client, "h1", 100_000)  # $0.30
    _run_session(client, "h2", 200_000)  # $0.60
    r = client.post("/session/start", json={
        "session_id": "h3", "cli": "claude-code", "project_dir": PROJ})
    ctx = r.json()["additionalContext"]
    assert ctx is not None and "[PROJECT HISTORY]" in ctx
    assert "2 session(s)" in ctx
    assert "$0.30" in ctx and "$0.60" in ctx


def test_history_excludes_other_projects_and_live_sessions(client):
    _run_session(client, "h1", 100_000)
    # a still-active session must not count (its cost is not final)
    client.post("/session/start", json={
        "session_id": "live", "cli": "claude-code", "project_dir": PROJ})
    client.post("/llm/usage", json={
        "session_id": "live", "model": "claude-sonnet-5", "message_id": "live-m",
        "usage": {"input_tokens": 999_000, "output_tokens": 0}})
    r = client.post("/session/start", json={
        "session_id": "h9", "cli": "claude-code", "project_dir": PROJ})
    ctx = r.json()["additionalContext"]
    assert "1 session(s)" in ctx


def test_history_fact_none_on_empty():
    assert prompts.session_history_fact([]) is None


# --- item 8: receipt ---

def test_receipt_renders_from_dump(client):
    _run_session(client, "r1", 100_000)
    dump = client.get("/session/r1/dump").json()
    receipt = prompts.render_receipt(dump)
    assert "receipt — session r1" in receipt
    assert "LLM cost: $0.30 across 1 calls" in receipt
    assert "claude-sonnet-5: $0.30 (1 calls)" in receipt


def test_receipt_shows_wallet_remaining(client):
    with db.get_conn() as conn:
        db.set_wallet(conn, PROJ, 1.0)
    _run_session(client, "r2", 100_000)  # $0.30 of the $1 wallet
    receipt = prompts.render_receipt(client.get("/session/r2/dump").json())
    assert "project wallet" in receipt
    assert "wallet remaining: $0.70" in receipt


def test_session_stop_logs_receipt(client, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="cost_aware_agent.receipt"):
        _run_session(client, "r3", 100_000)
    assert any("receipt — session r3" in r.getMessage() for r in caplog.records)
