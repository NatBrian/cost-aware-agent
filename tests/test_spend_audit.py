"""Spend-milestone self-audit (backlog item 4): when spend crosses another
budget slice on an accumulating channel, the tracker asks the model to account
for the delta. The harness states a measured number and asks ONE question —
no rule-based dead-end detection."""
import pytest

import cost_aware_agent.db as db
from cost_aware_agent import daemon, prompts


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "db.sqlite")
    # pin the policy knobs: daemon._config is the developer's real config file,
    # loaded at import — tests must not depend on its local values
    monkeypatch.setitem(daemon._config, "inject_mode", "on_change")
    monkeypatch.setitem(daemon._config, "inject_spend_bucket_pct", 10)
    db.init_db()
    with db.get_conn() as c:
        yield c


def _set_spend(conn, session_id, total, message_id):
    db.insert_llm_usage(conn, session_id, "claude-sonnet-5", 0, 0, 0, 0,
                        total, source="push", message_id=message_id)


def test_audit_question_fires_on_bucket_crossing(conn):
    db.insert_session(conn, "s1", "claude-code", "", "")
    db.set_session_budget_source(conn, "s1", None, 1.0)
    # first tracker delivery establishes the baseline — no checkpoint yet
    _set_spend(conn, "s1", 0.05, "m1")
    first = daemon._tracker_context(conn, "s1")
    assert first is not None and "BUDGET CHECKPOINT" not in first
    # spend crosses into the next 10% slice -> checkpoint with the exact delta
    _set_spend(conn, "s1", 0.12, "m2")
    second = daemon._tracker_context(conn, "s1")
    assert second is not None
    assert "[BUDGET CHECKPOINT] You have spent $0.12 since the last check" in second


def test_no_audit_question_on_tier_relabel_without_spend(conn):
    """A tier change without a bucket increase (e.g. budget crossing a tier
    boundary inside one bucket) re-injects the tracker but asks nothing —
    there is no new spend to account for."""
    db.insert_session(conn, "s2", "claude-code", "", "")
    db.set_session_budget_source(conn, "s2", None, 1.0)
    _set_spend(conn, "s2", 0.65, "m1")   # bucket 6, MEDIUM
    first = daemon._tracker_context(conn, "s2")
    assert first is not None
    _set_spend(conn, "s2", 0.005, "m2")  # 0.655: still bucket 6, still MEDIUM
    assert daemon._tracker_context(conn, "s2") is None  # suppressed, unchanged


def test_audit_delta_is_since_last_delivery_not_since_zero(conn):
    db.insert_session(conn, "s3", "claude-code", "", "")
    db.set_session_budget_source(conn, "s3", None, 1.0)
    _set_spend(conn, "s3", 0.15, "m1")
    daemon._tracker_context(conn, "s3")          # baseline at 0.15
    _set_spend(conn, "s3", 0.10, "m2")           # total 0.25 -> bucket 2
    text = daemon._tracker_context(conn, "s3")
    assert "$0.10 since the last check" in text  # delta, not the 0.25 total


def test_question_text_shape():
    q = prompts.spend_audit_question(0.25, 1.0)
    assert q.startswith("[BUDGET CHECKPOINT]")
    assert "$0.25" in q
    # sub-cent budgets get the adaptive precision the tracker uses
    q = prompts.spend_audit_question(0.0004, 0.0025)
    assert "$0.0004" in q
