"""Claude Code transcript parser tests — dedup by message.id (one turn writes
multiple JSONL lines), malformed-line tolerance, cache-split extraction."""
import json

from cost_aware_agent import transcript


def _line(msg_id, text="", usage=None, typ="assistant"):
    usage = usage or {"input_tokens": 10, "output_tokens": 5,
                      "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}
    return json.dumps({
        "type": typ,
        "message": {"id": msg_id, "model": "claude-sonnet-5", "usage": usage,
                    "content": [{"type": "text", "text": text}]},
    })


def test_multiline_turn_counted_once(tmp_path):
    # one requestId writes 2-3 lines (text block + tool_use blocks), all with
    # the same message.id and identical usage — must yield exactly ONE turn
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join([_line("m1", "part a"), _line("m1", "part b"),
                            _line("m2", "other")]) + "\n")
    turns = transcript.parse_new_assistant_turns(str(p), set())
    assert sorted(t["message_id"] for t in turns) == ["m1", "m2"]


def test_seen_ids_skipped(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(_line("m1") + "\n" + _line("m2") + "\n")
    turns = transcript.parse_new_assistant_turns(str(p), {"m1"})
    assert [t["message_id"] for t in turns] == ["m2"]


def test_malformed_and_foreign_lines_ignored(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("not json at all\n"
                 + json.dumps({"type": "ai-title", "aiTitle": "x"}) + "\n"
                 + _line("m1") + "\n")
    turns = transcript.parse_new_assistant_turns(str(p), set())
    assert len(turns) == 1


def test_cache_1h_split_extracted(tmp_path):
    usage = {"input_tokens": 1, "output_tokens": 1,
             "cache_read_input_tokens": 2,
             "cache_creation_input_tokens": 100,
             "cache_creation": {"ephemeral_1h_input_tokens": 80,
                                "ephemeral_5m_input_tokens": 20}}
    p = tmp_path / "t.jsonl"
    p.write_text(_line("m1", usage=usage) + "\n")
    turn = transcript.parse_new_assistant_turns(str(p), set())[0]
    assert turn["usage"]["cache_creation_tokens"] == 100
    assert turn["usage"]["cache_creation_1h_tokens"] == 80


def test_missing_file_returns_empty():
    assert transcript.parse_new_assistant_turns("/nonexistent/x.jsonl", set()) == []
