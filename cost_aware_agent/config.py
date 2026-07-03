"""Config loading.

Single file at ~/.cost-aware-agent/config.json, created with defaults on
first daemon start if missing. Nothing tunable should be hardcoded elsewhere.
"""

import json
from pathlib import Path

HOME_DIR = Path.home() / ".cost-aware-agent"
CONFIG_PATH = HOME_DIR / "config.json"
DB_PATH = HOME_DIR / "db.sqlite"

DEFAULTS = {
    # Master switch for advisory injection. When False the daemon still records
    # llm_usage/tool_calls (so cost is measured identically) but returns no
    # additionalContext anywhere — the OFF arm of the budget-awareness A/B, giving
    # an injection-vs-none comparison with the exact same measurement path.
    "inject_enabled": True,
    # The budget is ALWAYS money: LLM API dollar cost for the session versus this
    # dollar estimate. Tiers (HIGH/MEDIUM/LOW/CRITICAL) are computed from spend as
    # a fraction of this budget.
    "session_budget_estimate_usd": 1.00,
    # Money-only budget: spend = real LLM dollars, so tool calls carry no
    # synthetic price. Kept as a config knob (recorded on each tool_call row) but
    # default 0 so it never contaminates the injected 'LLM cost used' pressure.
    "tool_call_price_usd": 0.0,
    # Injection policy for ACCUMULATING context channels (Claude Code's
    # additionalContext persists in the conversation, so every injection is a
    # permanent token tax on all later turns — measured +44% session cost when
    # the tracker fired on all 22 hook calls of a tool-heavy task).
    #   on_change  re-inject only when the tier changes or spend crosses another
    #              inject_spend_bucket_pct slice of the budget (default)
    #   every      inject on every /tool/pre (for stateless harnesses that
    #              rebuild the prompt each step and need the tracker present)
    # Rebuilt channels (OpenCode's system-transform array is reconstructed on
    # every LLM call, nothing accumulates) always receive the current tracker
    # regardless of this setting — suppressing there would DELETE the budget
    # from context, not save tokens.
    "inject_mode": "on_change",
    "inject_spend_bucket_pct": 10,
    "streak_warning_threshold": 3,
    "context_mask_threshold_chars": 640000,
    "milestone_tool_patterns": ["Edit", "Write", "edit", "write"],
    "milestone_bash_test_patterns": [
        "test", "pytest", "npm test", "cargo test", "jest", "mocha",
    ],
}


def load_config() -> dict:
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULTS, indent=2) + "\n")
        return dict(DEFAULTS)
    on_disk = json.loads(CONFIG_PATH.read_text())
    # missing keys fall back to defaults rather than KeyError at call sites
    merged = dict(DEFAULTS)
    merged.update(on_disk)
    return merged
