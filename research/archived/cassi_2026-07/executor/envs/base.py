"""Minimal environment interface for the shared ReAct scaffold — paper_plan_v2
§2.6 (method plumbing), §19 Build table ("cost/draft/monitor hooks around the
agent loop" live in `cassi/executor/envs/`).

Contract (duck-typed; `AgentEnv` documents it):
  * reset(task) -> initial observation string
  * step(tool, arg) -> (observation, done, info)
      info keys the scaffold understands:
        - "tool_cost": dollars for this tool call (from cassi.budget.cost.tool_cost),
          added into c_t;
        - "docids": iterable of retrieved doc ids (feeds retrieval-overlap /
          n_distinct_sources in x_t, §11);
        - domain-specific quality signals (e.g. ALFWorld subgoals_done /
          subgoals_total) read ONLY by `step_quality` at collection time.
  * tools() -> tool description string for the system prompt
  * step_quality(draft, task, info) -> q_t — LABEL MACHINERY ONLY (§2.1):
      QA reads the gold answer from the task dict and scores the running draft
      with `qa_quality` at collection time; ALFWorld reads subgoal completion
      from env info with `alfworld_quality`. Never called at inference; never
      enters x_t.

`MockSearchEnv` is a deterministic dict-backed corpus for CPU tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cassi.budget.cost import tool_cost
from cassi.labels.quality import alfworld_quality, qa_quality


class AgentEnv(ABC):
    """Abstract base — see module docstring for the full contract."""

    domain: str = "qa"

    @abstractmethod
    def reset(self, task: dict) -> str:
        """Start an episode; returns the initial observation."""

    @abstractmethod
    def step(self, tool: str, arg: str) -> tuple[str, bool, dict]:
        """Execute one tool call; returns (observation, done, info)."""

    @abstractmethod
    def tools(self) -> str:
        """Tool description block for the agent's system prompt."""

    def step_quality(self, draft: str, task: dict, info: dict) -> float:
        """q_t for the step whose running draft is `draft` (§2.1, per-domain).
        Collection-time only — uses ground truth (QA) or env-privileged state
        (ALFWorld). Default: QA scoring against task['gold']."""
        gold = task.get("gold")
        if gold is None:
            return 0.0
        return qa_quality(draft, gold, metric=getattr(self, "quality_metric", "f1"))


def alfworld_step_quality(info: dict) -> float:
    """Shared ALFWorld quality reading (§2.1): subgoal-completion fraction from
    env info. Accepts either explicit {subgoals_done, subgoals_total} counts or
    a precomputed 'goal_condition_success_rate' fraction (TextWorld convention)."""
    if "subgoals_done" in info and "subgoals_total" in info:
        return alfworld_quality(int(info["subgoals_done"]), int(info["subgoals_total"]))
    if "goal_condition_success_rate" in info:
        return max(0.0, min(1.0, float(info["goal_condition_success_rate"])))
    return 0.0


class MockSearchEnv(AgentEnv):
    """Dict-backed corpus with deterministic search — CPU tests only.

    search[query] ranks documents by the number of query terms they contain
    (ties broken by doc id), returns the top-k, and charges
    tool_cost('retrieval_local') per query (§17 cost_model.tool_costs)."""

    domain = "qa"

    def __init__(self, corpus: dict[str, str], *, topk: int = 3,
                 quality_metric: str = "f1"):
        self.corpus = dict(corpus)
        self.topk = topk
        self.quality_metric = quality_metric
        self._task: dict = {}

    def reset(self, task: dict) -> str:
        self._task = task
        return f"Question: {task.get('question', '')}"

    def tools(self) -> str:
        return ("search[query]: search a local document corpus; returns the "
                "top matching passages with their doc ids.")

    def step(self, tool: str, arg: str) -> tuple[str, bool, dict]:
        if tool != "search":
            return f"Unknown tool '{tool}'. Available: search[query].", False, {"tool_cost": 0.0}
        terms = [w for w in arg.lower().split() if w]
        scored = []
        for docid in sorted(self.corpus):
            text = self.corpus[docid].lower()
            score = sum(1 for w in terms if w in text)
            if score > 0:
                scored.append((-score, docid))
        scored.sort()
        hits = [docid for _, docid in scored[: self.topk]]
        obs = (" | ".join(f"[{d}] {self.corpus[d]}" for d in hits)
               if hits else "No results found.")
        info = {"tool_cost": tool_cost("retrieval_local"), "docids": hits}
        return obs, False, info
