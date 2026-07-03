"""Claude Code transcript JSONL parser. Schema confirmed against a real
transcript, 2026-07-01.

Dedup key is `message.id`, NOT `uuid`: one model turn (one requestId) writes
multiple JSONL lines (one per content block), all sharing the same
message.id and an identical, already-complete usage block. Deduping by uuid
would count that turn's cost 2-3x.
"""

import json
from pathlib import Path

_VERIFICATION_RE_MARKER = "<verification>"


def parse_new_assistant_turns(transcript_path: str, seen_message_ids: set[str]) -> list[dict]:
    """Reads the transcript, returns one dict per assistant message.id not
    already in seen_message_ids: {message_id, model, usage, text}.
    Silently skips malformed lines — best-effort, advisory-only extends to
    the harness's own parsing too."""
    path = Path(transcript_path)
    if not path.exists():
        return []

    turns: dict[str, dict] = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "assistant":
            continue
        message = obj.get("message") or {}
        message_id = message.get("id")
        if not message_id or message_id in seen_message_ids or message_id in turns:
            continue
        usage = message.get("usage") or {}
        content = message.get("content") or []
        text = "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        cache_creation_total = usage.get("cache_creation_input_tokens", 0)
        # 1h vs 5m ephemeral cache writes bill at different rates (see cost.py).
        # Fall back to treating the whole amount as 5m (the conservative,
        # Anthropic-documented default) if the split isn't present in this
        # payload — matches the pre-fix behavior rather than guessing high.
        cache_breakdown = usage.get("cache_creation") or {}
        cache_creation_1h = cache_breakdown.get("ephemeral_1h_input_tokens", 0)
        turns[message_id] = {
            "message_id": message_id,
            "model": message.get("model"),
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                "cache_creation_tokens": cache_creation_total,
                "cache_creation_1h_tokens": cache_creation_1h,
            },
            "text": text,
        }
    return list(turns.values())


def has_verification_block(text: str) -> bool:
    return _VERIFICATION_RE_MARKER in text


def subagent_transcript_paths(transcript_path: str) -> list[str]:
    """Task-tool subagents do NOT write into the parent transcript (verified
    2026-07-03 against a real session: 0 sidechain assistant lines in the
    parent, subagent message ids absent from it). Current Claude Code writes
    each subagent to <transcript_dir>/<session-uuid>/subagents/agent-*.jsonl.
    Without ingesting these, all Task-tool LLM spend is invisible — an
    unmeasured channel that voids the budget. (Older CC versions wrote
    sidechain turns inline in the parent file; those parse via the normal
    path — type == 'assistant' — and message-id dedup makes overlap safe.)"""
    parent = Path(transcript_path)
    subagents_dir = parent.parent / parent.stem / "subagents"
    if not subagents_dir.is_dir():
        return []
    return sorted(str(p) for p in subagents_dir.glob("agent-*.jsonl"))
