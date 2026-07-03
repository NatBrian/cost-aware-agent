"""Project wallet — ONE user number ("$10 for my project") depleting across
sessions, plus the /session/start budget override (backlog item 2)."""
import pytest

import cost_aware_agent.db as db
from cost_aware_agent import daemon, prompts


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "db.sqlite")
    db.init_db()
    with db.get_conn() as c:
        yield c


PROJ = "/home/user/myproject"


def _spend(conn, session_id, cost_usd, message_id):
    db.insert_llm_usage(conn, session_id, "claude-sonnet-5", 100, 10, 0, 0,
                        cost_usd, source="pull", message_id=message_id)


def test_wallet_set_and_update(conn):
    db.set_wallet(conn, PROJ, 10.0)
    assert db.get_wallet(conn, PROJ)["budget_usd"] == 10.0
    db.set_wallet(conn, PROJ, 25.0)  # re-set replaces, no duplicate row
    assert db.get_wallet(conn, PROJ)["budget_usd"] == 25.0
    assert db.get_wallet(conn, "/elsewhere") is None
    assert db.get_wallet(conn, None) is None


def test_wallet_depletes_across_sessions(conn):
    """The core wallet semantic: spend accumulates over every session of the
    project — a new session does NOT reset the budget."""
    db.set_wallet(conn, PROJ, 10.0)
    for sid, c in [("s1", 3.0), ("s2", 4.0)]:
        db.insert_session(conn, sid, "claude-code", "", "")
        db.set_session_budget_source(conn, sid, PROJ, None)
        _spend(conn, sid, c, f"{sid}-m1")
    assert db.wallet_spent_usd(conn, PROJ) == pytest.approx(7.0)
    # a third session starts with only $3 of the wallet left
    db.insert_session(conn, "s3", "claude-code", "", "")
    db.set_session_budget_source(conn, "s3", PROJ, None)
    spent, budget, scope = daemon._budget_view(conn, "s3")
    assert (spent, budget, scope) == (pytest.approx(7.0), 10.0, "project wallet")


def test_other_projects_do_not_deplete_the_wallet(conn):
    db.set_wallet(conn, PROJ, 10.0)
    db.insert_session(conn, "mine", "claude-code", "", "")
    db.set_session_budget_source(conn, "mine", PROJ, None)
    db.insert_session(conn, "other", "claude-code", "", "")
    db.set_session_budget_source(conn, "other", "/somewhere/else", None)
    _spend(conn, "other", 99.0, "o-m1")
    assert db.wallet_spent_usd(conn, PROJ) == 0.0


def test_budget_view_resolution_order(conn):
    """override > wallet > config default."""
    db.set_wallet(conn, PROJ, 10.0)
    db.insert_session(conn, "s1", "claude-code", "", "")
    db.set_session_budget_source(conn, "s1", PROJ, 2.5)
    _spend(conn, "s1", 1.0, "m1")
    # override wins over the wallet, and spend is SESSION spend
    assert daemon._budget_view(conn, "s1") == (
        pytest.approx(1.0), 2.5, "session estimate")
    # no override -> wallet
    db.insert_session(conn, "s2", "claude-code", "", "")
    db.set_session_budget_source(conn, "s2", PROJ, None)
    spent, budget, scope = daemon._budget_view(conn, "s2")
    assert (budget, scope) == (10.0, "project wallet")
    # no override, no wallet -> config default
    db.insert_session(conn, "s3", "claude-code", "", "")
    spent, budget, scope = daemon._budget_view(conn, "s3")
    assert budget == daemon._config["session_budget_estimate_usd"]
    assert scope == "session estimate"


def test_session_start_refire_does_not_clobber(conn):
    """CC re-fires SessionStart (source=compact after compaction) with no
    budget fields — the wallet link and override must survive."""
    db.insert_session(conn, "s1", "claude-code", "", "")
    db.set_session_budget_source(conn, "s1", PROJ, 2.5)
    db.set_session_budget_source(conn, "s1", None, None)  # the refire
    row = db.get_session(conn, "s1")
    assert row["project_dir"] == PROJ
    assert row["budget_override_usd"] == 2.5


def test_tracker_labels_wallet_scope():
    text = prompts.render_budget_tracker(7.0, 10.0, 3, "LOW", [],
                                         scope="project wallet")
    assert "remaining (of project wallet): $3.00" in text
    default = prompts.render_budget_tracker(0.1, 1.0, 3, "HIGH", [])
    assert "remaining (of session estimate)" in default
