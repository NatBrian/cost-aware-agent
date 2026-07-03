"""Session lifecycle over HTTP: compaction re-injection (backlog item 7) and
the enable_plan_verification kill-switch (backlog item 10)."""
import pytest
from fastapi.testclient import TestClient

import cost_aware_agent.db as db
from cost_aware_agent import daemon, prompts


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "db.sqlite")
    monkeypatch.setitem(daemon._config, "inject_enabled", True)
    monkeypatch.setitem(daemon._config, "inject_mode", "on_change")
    monkeypatch.setitem(daemon._config, "inject_spend_bucket_pct", 10)
    monkeypatch.setitem(daemon._config, "enable_plan_verification", False)
    c = TestClient(daemon.app)
    db.init_db()
    return c


def _spend(client, sid, cost_proxy_tokens):
    client.post("/llm/usage", json={
        "session_id": sid, "model": "claude-sonnet-5",
        "message_id": f"m-{cost_proxy_tokens}",
        "usage": {"input_tokens": cost_proxy_tokens, "output_tokens": 0}})


def test_compact_refire_redelivers_tracker(client):
    """Compaction wipes every accumulated injection from the conversation —
    the SessionStart(source=compact) refire must hand the tracker back."""
    client.post("/session/start", json={
        "session_id": "s1", "cli": "claude-code", "budget_usd": 1.0})
    _spend(client, "s1", 100_000)  # $0.30 at sonnet input rate
    # normal delivery consumes the on_change transition
    first = client.post("/tool/pre", json={"session_id": "s1", "tool_name": "Read"})
    assert first.json()["additionalContext"] is not None
    # same state again -> suppressed
    again = client.post("/tool/pre", json={"session_id": "s1", "tool_name": "Read"})
    assert again.json()["additionalContext"] is None
    # compaction refire -> tracker delivered in the SessionStart response itself
    compact = client.post("/session/start", json={
        "session_id": "s1", "cli": "claude-code", "source": "compact"})
    ctx = compact.json()["additionalContext"]
    assert ctx is not None and "Budget Tracker" in ctx
    # and the budget override survived the refire (no clobber)
    with db.get_conn() as conn:
        assert db.get_session(conn, "s1")["budget_override_usd"] == 1.0


def test_startup_source_does_not_get_planning_prompt_when_disabled(client):
    r = client.post("/session/start", json={
        "session_id": "s2", "cli": "claude-code", "source": "startup"})
    assert r.json()["additionalContext"] is None


def test_plan_verification_machinery_dormant_by_default(client):
    """Item 10: checklist/verification/streak never demonstrated value —
    dormant unless enable_plan_verification is set."""
    client.post("/session/start", json={"session_id": "s3", "cli": "claude-code"})
    # milestone tool -> no SELF_VERIFICATION prompt
    r = client.post("/tool/post", json={
        "session_id": "s3", "tool_name": "Edit", "tool_input": {}})
    assert r.json()["additionalContext"] is None
    # plan/seed -> no plan rows
    client.post("/plan/seed", json={
        "session_id": "s3",
        "raw_response": '<checklist><item type="exploration">x</item></checklist>'})
    with db.get_conn() as conn:
        assert db.get_plan(conn, "s3") == []
    # verification/result -> no streak text, no crash
    r = client.post("/verification/result", json={
        "session_id": "s3", "raw_response": "<decision>CONTINUE</decision>"})
    assert r.json()["additionalContext"] is None


def test_plan_verification_reenabled_by_flag(client, monkeypatch):
    monkeypatch.setitem(daemon._config, "enable_plan_verification", True)
    r = client.post("/session/start", json={"session_id": "s4", "cli": "claude-code"})
    assert r.json()["additionalContext"] == prompts.PLANNING_PROMPT
    r = client.post("/tool/post", json={
        "session_id": "s4", "tool_name": "Edit", "tool_input": {}})
    assert r.json()["additionalContext"] == prompts.SELF_VERIFICATION_PROMPT
