"""The security model: no channel through which LLM spend can flow uncounted.

Covers backlog item 1 (2026-07-03 handoff §7):
  a. unknown-model pricing never $0 + price_unknown flag + tracker warning
  b. Task-tool subagent transcripts (separate files, NOT in the parent
     transcript — verified against a real CC session 2026-07-03) are ingested
  c. multi-model sessions price each row by its own model id
"""
import json

import pytest

import cost_aware_agent.db as db
from cost_aware_agent import cost, daemon, prompts, transcript


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "db.sqlite")
    db.init_db()
    with db.get_conn() as c:
        yield c


def _assistant_line(message_id, model, input_tokens=1000, output_tokens=100,
                    text="ok"):
    return json.dumps({
        "type": "assistant",
        "message": {
            "id": message_id,
            "model": model,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "content": [{"type": "text", "text": text}],
        },
    })


def _write_session_transcripts(tmp_path, parent_lines, subagent_lines=None):
    """Real CC layout: <dir>/<session-uuid>.jsonl for the parent and
    <dir>/<session-uuid>/subagents/agent-*.jsonl for Task-tool children."""
    parent = tmp_path / "sess-uuid.jsonl"
    parent.write_text("\n".join(parent_lines) + "\n")
    if subagent_lines is not None:
        sub_dir = tmp_path / "sess-uuid" / "subagents"
        sub_dir.mkdir(parents=True)
        (sub_dir / "agent-abc123.jsonl").write_text("\n".join(subagent_lines) + "\n")
    return str(parent)


# --- b: subagent transcript discovery + ingestion ---

def test_subagent_paths_discovered(tmp_path):
    p = _write_session_transcripts(
        tmp_path, [_assistant_line("m1", "claude-sonnet-5")],
        [_assistant_line("sub1", "claude-haiku-4-5-20251001")],
    )
    paths = transcript.subagent_transcript_paths(p)
    assert len(paths) == 1
    assert paths[0].endswith("agent-abc123.jsonl")


def test_subagent_paths_empty_when_no_dir(tmp_path):
    p = _write_session_transcripts(tmp_path, [_assistant_line("m1", "claude-sonnet-5")])
    assert transcript.subagent_transcript_paths(p) == []


def test_ingest_captures_subagent_spend(conn, tmp_path):
    """Task-tool children write their own transcript files; parent-only parsing
    misses 100% of their spend. Both must land on the parent session."""
    p = _write_session_transcripts(
        tmp_path, [_assistant_line("m1", "claude-sonnet-5", 1000, 100)],
        [_assistant_line("sub1", "claude-sonnet-5", 2000, 200)],
    )
    db.insert_session(conn, "s1", "claude-code", "", "")
    daemon._ingest_transcript(conn, "s1", p)
    rows = conn.execute(
        "SELECT source, input_tokens FROM llm_usage WHERE session_id='s1' ORDER BY id"
    ).fetchall()
    assert [(r["source"], r["input_tokens"]) for r in rows] == [
        ("pull", 1000), ("pull-subagent", 2000)]
    spent, _ = db.spent_usd(conn, "s1")
    sonnet = cost.price_for_model("claude-sonnet-5")
    expected = (3000 * sonnet["input_cost_per_token"]
                + 300 * sonnet["output_cost_per_token"])
    assert spent == pytest.approx(expected)


def test_ingest_is_idempotent_across_refires(conn, tmp_path):
    # hooks re-fire constantly; parent + subagent turns must not double-count
    p = _write_session_transcripts(
        tmp_path, [_assistant_line("m1", "claude-sonnet-5")],
        [_assistant_line("sub1", "claude-sonnet-5")],
    )
    db.insert_session(conn, "s2", "claude-code", "", "")
    daemon._ingest_transcript(conn, "s2", p)
    daemon._ingest_transcript(conn, "s2", p)
    n = conn.execute(
        "SELECT COUNT(*) AS c FROM llm_usage WHERE session_id='s2'").fetchone()["c"]
    assert n == 2


# --- a: price_unknown flag end-to-end ---

def test_ingest_flags_unknown_model_and_charges_fallback(conn, tmp_path):
    p = _write_session_transcripts(
        tmp_path, [_assistant_line("m1", "mystery-model-9000", 1000, 100)])
    db.insert_session(conn, "s3", "claude-code", "", "")
    daemon._ingest_transcript(conn, "s3", p)
    row = conn.execute(
        "SELECT price_unknown, cost_usd FROM llm_usage WHERE session_id='s3'"
    ).fetchone()
    assert row["price_unknown"] == 1
    assert row["cost_usd"] > 0.0
    assert db.session_has_unknown_priced_usage(conn, "s3") is True


def test_synthetic_zero_token_rows_not_flagged(conn, tmp_path):
    # CC writes '<synthetic>' assistant entries with zero usage — nothing was
    # spent, so they must not trip a permanent tracker warning
    p = _write_session_transcripts(
        tmp_path, [_assistant_line("m1", "<synthetic>", 0, 0)])
    db.insert_session(conn, "s4", "claude-code", "", "")
    daemon._ingest_transcript(conn, "s4", p)
    assert db.session_has_unknown_priced_usage(conn, "s4") is False


def test_known_model_not_flagged(conn, tmp_path):
    p = _write_session_transcripts(
        tmp_path, [_assistant_line("m1", "claude-sonnet-5")])
    db.insert_session(conn, "s5", "claude-code", "", "")
    daemon._ingest_transcript(conn, "s5", p)
    assert db.session_has_unknown_priced_usage(conn, "s5") is False


def test_tracker_renders_price_unknown_warning():
    with_warn = prompts.render_budget_tracker(
        0.5, 1.0, 3, "MEDIUM", [], price_unknown=True)
    without = prompts.render_budget_tracker(
        0.5, 1.0, 3, "MEDIUM", [], price_unknown=False)
    assert prompts.PRICE_UNKNOWN_NOTE in with_warn
    assert prompts.PRICE_UNKNOWN_NOTE not in without


# --- c: multi-model sessions price each row by its own model ---

def test_multi_model_session_prices_each_row_by_its_model(conn, tmp_path):
    p = _write_session_transcripts(tmp_path, [
        _assistant_line("m1", "claude-sonnet-5", 1000, 0),
        _assistant_line("m2", "claude-haiku-4-5-20251001", 1000, 0),
    ])
    db.insert_session(conn, "s6", "claude-code", "", "")
    daemon._ingest_transcript(conn, "s6", p)
    sonnet = cost.price_for_model("claude-sonnet-5")["input_cost_per_token"]
    haiku = cost.price_for_model("claude-haiku-4-5-20251001")["input_cost_per_token"]
    assert sonnet != haiku  # the test is vacuous if the rates ever converge
    spent, _ = db.spent_usd(conn, "s6")
    assert spent == pytest.approx(1000 * sonnet + 1000 * haiku)
