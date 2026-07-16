"""Search-R1 retrieval env wrapper — paper_plan_v2 §5.1 (primary training domain:
NQ + HotpotQA + MuSiQue with the Search-R1 local Wikipedia retriever, run via
verl-tool at training time, §19), §16 P1 (index build).

This wrapper speaks the standard Search-R1 local retrieval server HTTP protocol:

    POST {retriever_url}   body: {"queries": ["..."], "topk": k}
    response: {"result": [[{"document": {"id": ..., "contents": ...}, "score": ...}, ...]]}

(`PeterGriffinJin/Search-R1` retrieval_server.py; verl-tool reuses the same
server). Tool costing follows §17 cost_model.tool_costs via
`cassi.budget.cost.tool_cost` — "retrieval_local" for the local index (default)
or "web_search" (per query + per result) when pointed at a web-search bridge.

All network (`requests`) imports are lazy — CPU test environments import this
module without them. GRPO training does NOT go through this class's HTTP path
step-by-step; verl-tool owns the training-time rollout loop (§19). This wrapper
serves label collection (P2/P7) and inference/monitor evaluation (§2.5).
"""

from __future__ import annotations

from cassi.budget.cost import tool_cost
from cassi.executor.envs.base import AgentEnv
from cassi.labels.quality import qa_quality

_RETRIEVER_HELP = (
    "Could not reach the Search-R1 retrieval server at {url!r}: {err}\n"
    "Start the local retriever first (paper_plan_v2 §16 P1 — Search-R1 recipe: "
    "local Wikipedia dump + E5/BM25 index):\n"
    "  bash scripts/p1_data.sh        # downloads the wiki dump, builds the index,\n"
    "                                 # and launches Search-R1's retrieval_server.py\n"
    "or manually, from a Search-R1 checkout:\n"
    "  python search_r1/search/retrieval_server.py --index_path <index> "
    "--corpus_path <corpus> --topk 3 --retriever_name e5\n"
    "The server must accept POST {{'queries': [...], 'topk': k}} on /retrieve."
)


class SearchR1QAEnv(AgentEnv):
    """QA search env over the Search-R1 local retriever (or a web-search bridge).

    Quality reading (§2.1): gold answer available in the task dict; q_t is read
    by scoring the running draft with `qa_quality` at collection time ONLY.
    """

    domain = "qa"

    def __init__(
        self,
        *,
        retriever_url: str = "http://127.0.0.1:8000/retrieve",
        topk: int = 3,
        tool_type: str = "retrieval_local",   # or "web_search" (§17 tool_costs)
        timeout: float = 30.0,
        quality_metric: str = "f1",           # §17 label.quality_scoring.qa
    ):
        if tool_type not in ("retrieval_local", "web_search"):
            raise ValueError(f"tool_type must be retrieval_local|web_search, got {tool_type!r}")
        self.retriever_url = retriever_url
        self.topk = topk
        self.tool_type = tool_type
        self.timeout = timeout
        self.quality_metric = quality_metric
        self._task: dict = {}

    def reset(self, task: dict) -> str:
        self._task = task
        return f"Question: {task.get('question', '')}"

    def tools(self) -> str:
        return ("search[query]: retrieve the top passages for the query from the "
                "local Wikipedia index (Search-R1 retriever).")

    # -- retrieval --------------------------------------------------------------
    def _retrieve(self, query: str) -> list[dict]:
        import requests  # lazy import — CPU import safety

        try:
            resp = requests.post(
                self.retriever_url,
                json={"queries": [query], "topk": self.topk},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise NotImplementedError(
                _RETRIEVER_HELP.format(url=self.retriever_url, err=e)
            ) from e
        payload = resp.json()
        rows = payload.get("result", payload.get("results", [[]]))
        return rows[0] if rows else []

    @staticmethod
    def _doc_fields(hit: dict) -> tuple[str, str]:
        """Normalize the few shapes Search-R1-line servers return."""
        doc = hit.get("document", hit)
        docid = str(doc.get("id", doc.get("docid", doc.get("_id", ""))))
        contents = str(doc.get("contents", doc.get("content", doc.get("text", ""))))
        return docid, contents

    def step(self, tool: str, arg: str) -> tuple[str, bool, dict]:
        if tool != "search":
            return f"Unknown tool '{tool}'. Available: search[query].", False, {"tool_cost": 0.0}
        hits = self._retrieve(arg)
        docids, snippets = [], []
        for i, h in enumerate(hits):
            docid, contents = self._doc_fields(h)
            docids.append(docid or f"hit{i}")
            snippets.append(f"[{docid or f'hit{i}'}] {contents}")
        obs = " | ".join(snippets) if snippets else "No results found."
        cost = tool_cost(self.tool_type, n_results=len(hits))
        return obs, False, {"tool_cost": cost, "docids": docids}

    # -- collection-time quality (§2.1) ------------------------------------------
    def step_quality(self, draft: str, task: dict, info: dict) -> float:
        gold = task.get("gold")
        if gold is None:
            return 0.0
        return qa_quality(draft, gold, metric=self.quality_metric)
