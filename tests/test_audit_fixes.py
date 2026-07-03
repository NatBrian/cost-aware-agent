"""Fixes from the 2026-07-04 adversarial audit. Each test names its finding.

H1  workflow agent transcripts (subagents/workflows/wf_*/) were unmeasured
H2  negative tokens / bogus budget overrides could erase or neuter spend
M3  CC 'Stop' fires per response, not per session — /turn/end split
M4  wallet key normalization (CLI resolved, daemon stored raw cwd)
L5  history unbounded ("last N" actually meant "all ever")
L6  burn rate stamped rows with ingest time, not occurrence time
L7  negative/zero wallet amounts accepted
L8  wallet-scope checkpoint blamed one session for project-wide spend
L9  receipt silent about unknown-priced calls
"""
import json

import pytest
from fastapi.testclient import TestClient

import cost_aware_agent.db as db
from cost_aware_agent import cost, daemon, prompts, transcript


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


def _assistant_line(message_id, model="claude-sonnet-5", input_tokens=1000,
                    output_tokens=100, timestamp=None):
    obj = {
        "type": "assistant",
        "message": {
            "id": message_id, "model": model,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "content": [{"type": "text", "text": "ok"}],
        },
    }
    if timestamp:
        obj["timestamp"] = timestamp
    return json.dumps(obj)


# --- H1: workflow agents under subagents/workflows/wf_*/ ---

def test_workflow_agent_transcripts_discovered(tmp_path):
    parent = tmp_path / "sess.jsonl"
    parent.write_text(_assistant_line("m1") + "\n")
    task_dir = tmp_path / "sess" / "subagents"
    wf_dir = task_dir / "workflows" / "wf_abc123-9x"
    wf_dir.mkdir(parents=True)
    (task_dir / "agent-task1.jsonl").write_text(_assistant_line("t1") + "\n")
    (wf_dir / "agent-wf1.jsonl").write_text(_assistant_line("w1") + "\n")
    # meta.json files sit alongside and must NOT match
    (task_dir / "agent-task1.meta.json").write_text("{}")
    paths = transcript.subagent_transcript_paths(str(parent))
    assert len(paths) == 2
    assert any("workflows" in p for p in paths)
    assert not any(p.endswith("meta.json") for p in paths)


# --- H2: spend-erasure and budget-neutering inputs ---

def test_negative_tokens_never_produce_negative_cost():
    assert cost.cost_llm_usage("claude-sonnet-5", -1_000_000, -50_000, -10, -10, -10) == 0.0
    # mixed: negative fields clamp to 0, positive fields still bill
    c = cost.cost_llm_usage("claude-sonnet-5", 1000, -50_000, 0, 0, 0)
    assert c == pytest.approx(1000 * 3e-06)


def test_llm_usage_endpoint_clamps_forged_negative_push(client):
    client.post("/session/start", json={"session_id": "n1", "cli": "opencode"})
    client.post("/llm/usage", json={
        "session_id": "n1", "model": "claude-sonnet-5", "message_id": "forged",
        "usage": {"input_tokens": -9_000_000, "output_tokens": -9_000_000}})
    with db.get_conn() as conn:
        spent, _ = db.spent_usd(conn, "n1")
        row = conn.execute("SELECT input_tokens FROM llm_usage "
                           "WHERE session_id='n1'").fetchone()
    assert spent == 0.0          # not negative — spend cannot be erased
    assert row["input_tokens"] == 0  # stored clamped too


@pytest.mark.parametrize("bad", [-5.0, 0.0, float("nan"), float("inf")])
def test_invalid_budget_override_rejected(client, bad):
    # NaN/inf cannot ride standard JSON, so exercise the endpoint function
    # directly — the guard must hold for any caller, not just HTTP
    daemon.session_start(daemon.SessionStartReq(
        session_id=f"b-{bad}", cli="claude-code", budget_usd=bad))
    with db.get_conn() as conn:
        row = db.get_session(conn, f"b-{bad}")
    assert row["budget_override_usd"] is None


def test_valid_budget_override_still_works(client):
    client.post("/session/start", json={
        "session_id": "b-ok", "cli": "claude-code", "budget_usd": 2.5})
    with db.get_conn() as conn:
        assert db.get_session(conn, "b-ok")["budget_override_usd"] == 2.5


# --- M3: Stop is per-turn, SessionEnd is per-session ---

def test_turn_end_ingests_but_does_not_end_session(client, tmp_path):
    p = tmp_path / "sess.jsonl"
    p.write_text(_assistant_line("m1") + "\n")
    client.post("/session/start", json={"session_id": "t1", "cli": "claude-code"})
    r = client.post("/turn/end", json={"session_id": "t1",
                                       "transcript_path": str(p)})
    assert r.status_code == 200
    with db.get_conn() as conn:
        assert db.get_session(conn, "t1")["state"] == "active"  # NOT ended
        spent, _ = db.spent_usd(conn, "t1")
    assert spent > 0  # but the turn's usage WAS ingested


def test_mid_conversation_session_not_counted_as_history(client, tmp_path):
    """A live interactive session that has fired Stop (now /turn/end) several
    times must not appear in [PROJECT HISTORY] with a partial cost."""
    proj = str(tmp_path / "proj")
    p = tmp_path / "sess.jsonl"
    p.write_text(_assistant_line("m1") + "\n")
    client.post("/session/start", json={
        "session_id": "live", "cli": "claude-code", "project_dir": proj})
    client.post("/turn/end", json={"session_id": "live", "transcript_path": str(p)})
    r = client.post("/session/start", json={
        "session_id": "newcomer", "cli": "claude-code", "project_dir": proj})
    assert r.json()["additionalContext"] is None  # no finished sessions yet


def test_abandoned_session_eventually_counts_as_history(client):
    """SIGKILLed sessions never fire SessionEnd; after the idle threshold their
    measured spend still becomes history (finished-for-accounting)."""
    with db.get_conn() as conn:
        db.insert_session(conn, "dead", "claude-code", "", "")
        conn.execute("UPDATE sessions SET project_dir='/p' WHERE session_id='dead'")
        db.insert_llm_usage(conn, "dead", "claude-sonnet-5", 0, 0, 0, 0, 0.42,
                            source="pull", message_id="d1",
                            created_at=db.now() - db._ABANDONED_AFTER_SECS - 60)
        costs = db.project_session_costs(conn, "/p")
    assert costs == [pytest.approx(0.42)]


# --- M4 / L7: wallet key normalization + amount validation ---

def test_daemon_normalizes_project_dir_to_match_cli(client, tmp_path):
    real = tmp_path / "realproj"
    real.mkdir()
    link = tmp_path / "linkproj"
    link.symlink_to(real)
    with db.get_conn() as conn:
        db.set_wallet(conn, db.normalize_project_dir(str(real)), 10.0)
    # hook sends the SYMLINK path; wallet was set on the real path
    client.post("/session/start", json={
        "session_id": "sym", "cli": "claude-code", "project_dir": str(link)})
    with db.get_conn() as conn:
        spent, budget, scope = daemon._budget_view(conn, "sym")
    assert (budget, scope) == (10.0, "project wallet")


def test_cli_rejects_nonpositive_wallet(tmp_path, monkeypatch, capsys):
    import sys

    from cost_aware_agent import cli
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "db.sqlite")
    monkeypatch.setattr(sys, "argv",
                        ["cost-aware-agent", "budget", "set", "-5", "--project-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1
    assert "invalid budget amount" in capsys.readouterr().err


# --- L5: history bounded to the most recent finished sessions ---

def test_history_caps_at_limit_most_recent(client):
    with db.get_conn() as conn:
        for i in range(db._HISTORY_LIMIT + 3):
            sid = f"h{i}"
            db.insert_session(conn, sid, "claude-code", "", "")
            conn.execute("UPDATE sessions SET project_dir='/p', state='ended' "
                         "WHERE session_id=?", (sid,))
            db.insert_llm_usage(conn, sid, "claude-sonnet-5", 0, 0, 0, 0,
                                float(i + 1), source="pull", message_id=f"{sid}m",
                                created_at=db.now() - 1000 + i)
        costs = db.project_session_costs(conn, "/p")
    assert len(costs) == db._HISTORY_LIMIT
    # the OLDEST 3 fell off; the newest survive, oldest-first
    assert costs[0] == 4.0 and costs[-1] == 13.0


# --- L6: occurrence time, not ingest time ---

def test_burn_rate_uses_transcript_timestamp_not_ingest_time(client, tmp_path):
    """Batch-ingesting an old backlog must not spike the burn window."""
    from datetime import datetime, timedelta, timezone
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z")
    p = tmp_path / "sess.jsonl"
    p.write_text(_assistant_line("old-turn", input_tokens=1_000_000,
                                 timestamp=old) + "\n")
    client.post("/session/start", json={"session_id": "burn1", "cli": "claude-code"})
    client.post("/turn/end", json={"session_id": "burn1", "transcript_path": str(p)})
    with db.get_conn() as conn:
        spent, _ = db.spent_usd(conn, "burn1")
        burn = db.recent_spend_usd(conn, "burn1", 600)
    assert spent > 0        # total spend counted
    assert burn == 0.0      # but it did not happen in the last 10 minutes


# --- L8: wallet-scope checkpoint wording ---

def test_wallet_checkpoint_does_not_blame_single_session():
    q = prompts.spend_audit_question(1.0, 10.0, scope="project wallet")
    assert "This project has spent $1.00" in q
    assert "You have spent" not in q
    q = prompts.spend_audit_question(0.25, 1.0)  # session scope unchanged
    assert "You have spent $0.25" in q


# --- L9: receipt discloses unknown-priced calls ---

def test_receipt_warns_about_unknown_priced_calls(client):
    client.post("/session/start", json={"session_id": "u1", "cli": "opencode"})
    client.post("/llm/usage", json={
        "session_id": "u1", "model": "mystery-model", "message_id": "u1m",
        "usage": {"input_tokens": 1000, "output_tokens": 100}})
    client.post("/session/stop", json={"session_id": "u1"})
    receipt = prompts.render_receipt(client.get("/session/u1/dump").json())
    assert "no known price" in receipt
