"""I1 tests: sampling determinism/stratification/overlap + retrieval client."""

import pytest

from collect.sampling import (assert_no_overlap, attach_levels,
                              stratified_sample, strata_counts)
from envs.retrieval_client import RetrievalClient, RetrievalError


def make_rows(n, level_cycle=("easy", "medium", "hard")):
    return [{"id": f"q{i:04d}", "question": f"question number {i}?",
             "level": level_cycle[i % len(level_cycle)]} for i in range(n)]


def test_sampling_is_deterministic():
    rows = make_rows(1000)
    a = stratified_sample(rows, 100, seed=42)
    b = stratified_sample(rows, 100, seed=42)
    assert a == b
    c = stratified_sample(rows, 100, seed=43)
    assert a != c


def test_sampling_is_proportional():
    rows = make_rows(900)  # exactly 300 per level
    out = stratified_sample(rows, 90, seed=1)
    assert strata_counts(out) == {"easy": 30, "medium": 30, "hard": 30}


def test_sampling_allocates_remainders_to_largest_stratum():
    rows = [{"id": f"a{i}", "question": f"a {i}?", "level": "easy"} for i in range(70)]
    rows += [{"id": f"b{i}", "question": f"b {i}?", "level": "hard"} for i in range(30)]
    out = stratified_sample(rows, 10, seed=1)
    assert strata_counts(out) == {"easy": 7, "hard": 3}


def test_sampling_rejects_oversized_request():
    with pytest.raises(ValueError):
        stratified_sample(make_rows(10), 11, seed=1)


def test_attach_levels_marks_unknown():
    rows = [{"id": "x", "question": "q?"}]
    assert attach_levels(rows, {})[0]["level"] == "unknown"


def test_overlap_check_catches_shared_id_and_text():
    a = [{"id": "1", "question": "Who is X?"}]
    with pytest.raises(AssertionError):
        assert_no_overlap(a, [{"id": "1", "question": "other?"}])
    with pytest.raises(AssertionError):
        assert_no_overlap(a, [{"id": "2", "question": "  who IS x?  "}])
    assert_no_overlap(a, [{"id": "2", "question": "Who is Y?"}])  # passes


class _FakeResp:
    def __init__(self, status=200, results=None):
        self.status_code = status
        self.text = "err"
        self._results = results or []

    def json(self):
        return {"results": self._results}


def test_retrieval_client_parses_hits(monkeypatch):
    hits = [{"title": "Paris", "text": "Capital of France."}]
    monkeypatch.setattr("envs.retrieval_client.requests.post",
                        lambda *a, **k: _FakeResp(results=hits))
    c = RetrievalClient("http://x:1", top_k=3)
    out = c.search("capital of France?")
    assert out == [{"title": "Paris", "text": "Capital of France."}]
    assert "Paris" in c.format_observation(out)


def test_retrieval_client_raises_on_http_error(monkeypatch):
    monkeypatch.setattr("envs.retrieval_client.requests.post",
                        lambda *a, **k: _FakeResp(status=500))
    with pytest.raises(RetrievalError):
        RetrievalClient("http://x:1").search("q?")


def test_format_observation_empty():
    assert RetrievalClient("http://x:1").format_observation([]) == "No results found."
