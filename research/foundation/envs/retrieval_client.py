"""HTTP client for the local retrieval server (F1/F2).

The agent's one tool. Server: scripts/serve_retrieval.py (E5+FAISS over the
rescued searchr1 index). Tests mock `requests.post`.
"""

import requests


class RetrievalError(RuntimeError):
    pass


class RetrievalClient:
    def __init__(self, endpoint: str, top_k: int = 3, timeout: float = 30.0):
        self.endpoint = endpoint.rstrip("/")
        self.top_k = top_k
        self.timeout = timeout

    def search(self, query: str) -> list[dict]:
        """Returns [{"title": str, "text": str, "score": float}, ...], <= top_k.

        `score` is the retriever's similarity for that hit. It is carried through
        (and recorded per step) because S1 showed retrieval productivity predicts
        eventual failure — the whole first run discarded it here. Older servers
        that do not send a score yield 0.0 rather than a KeyError.
        """
        try:
            resp = requests.post(f"{self.endpoint}/search",
                                 json={"query": query, "top_k": self.top_k},
                                 timeout=self.timeout)
        except requests.RequestException as e:
            raise RetrievalError(f"retrieval server unreachable: {e}") from e
        if resp.status_code != 200:
            raise RetrievalError(f"retrieval server HTTP {resp.status_code}: {resp.text[:200]}")
        hits = resp.json().get("results", [])
        return [{"title": h.get("title", ""), "text": h.get("text", ""),
                 "score": float(h.get("score", 0.0))} for h in hits]

    def format_observation(self, hits: list[dict], max_chars: int = 2000) -> str:
        """Render hits as the observation string the agent reads."""
        if not hits:
            return "No results found."
        parts = [f"[{i+1}] {h['title']}: {h['text']}" for i, h in enumerate(hits)]
        return "\n".join(parts)[:max_chars]
