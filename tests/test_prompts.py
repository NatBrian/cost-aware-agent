"""Prompt parsing/rendering tests — deterministic tag parsing (no LLM),
milestone rule, tracker rendering incl. the stale-spend lag note."""
from cost_aware_agent import prompts


def test_parse_checklist():
    text = ('blah <checklist>\n'
            '<item type="exploration">find the config loader</item>\n'
            '<item type="verification">tests pass after fix</item>\n'
            '</checklist> blah')
    assert prompts.parse_checklist(text) == [
        ("exploration", "find the config loader"),
        ("verification", "tests pass after fix"),
    ]


def test_parse_checklist_absent():
    assert prompts.parse_checklist("no checklist here") == []


def test_parse_verification():
    text = ('<verification>\n<item id="1" status="satisfied"/>\n'
            '<item id="2" status="contradicted"/>\n'
            '<decision>CONTINUE</decision>\n<reason>r</reason>\n</verification>')
    items, decision = prompts.parse_verification(text)
    assert items == [(1, "satisfied"), (2, "contradicted")]
    assert decision == "CONTINUE"


def test_parse_verification_malformed_gives_none_decision():
    items, decision = prompts.parse_verification("<verification>garbage</verification>")
    assert items == []
    assert decision is None  # caller must log-and-skip, never block


CFG = {"milestone_tool_patterns": ["Edit", "Write", "edit", "write"],
       "milestone_bash_test_patterns": ["test", "pytest"]}


def test_milestone_file_edit():
    assert prompts.is_milestone("Edit", {}, CFG)
    assert not prompts.is_milestone("Read", {}, CFG)


def test_milestone_bash_test_command_only():
    assert prompts.is_milestone("Bash", {"command": "pytest -x"}, CFG)
    assert not prompts.is_milestone("Bash", {"command": "ls -la"}, CFG)


def test_tracker_lag_note_toggle():
    with_note = prompts.render_budget_tracker(0.1, 1.0, 2, "HIGH", [], lagging=True)
    without = prompts.render_budget_tracker(0.1, 1.0, 2, "HIGH", [], lagging=False)
    assert prompts.LAG_NOTE in with_note
    assert prompts.LAG_NOTE not in without
    # both must stay well-formed blocks
    assert with_note.startswith("Budget Tracker <budget>")
    assert with_note.rstrip().endswith("</budget>")


def test_tracker_shows_money_and_tier():
    t = prompts.render_budget_tracker(0.45, 0.30, 7, "CRITICAL", [])
    assert "LLM cost used: $0.45" in t
    assert "remaining (of session estimate): $0.00" in t  # floors at 0
    assert "Tier: CRITICAL" in t


def test_tracker_rebuilt_channel_stability_features():
    # tool counter omitted + '~' approx marker: rebuilt channels re-send the
    # tracker every LLM call, so per-call-changing bytes bust the provider's
    # prompt cache (measured +92% cost on deepseek before this)
    t = prompts.render_budget_tracker(0.10, 1.0, None, "HIGH", [], approximate=True)
    assert "Tool calls used" not in t
    assert "~$0.10" in t


def test_tracker_subcent_budget_gets_decimals():
    # a $0.0025 retail budget at :.2f reads "$0.00 of $0.00" — useless numbers
    t = prompts.render_budget_tracker(0.0015, 0.0025, 3, "LOW", [])
    assert "$0.0015" in t
    assert "$0.0010" in t
