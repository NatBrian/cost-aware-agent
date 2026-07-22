"""ReAct agent scaffold — the shared agent loop for ALL methods (paper_plan_v2 §2.6:
"used by CASSI and every baseline alike, so it's a constant, not an advantage").

Implements:
  * the reason → tool-call → observe loop driven by an abstract `LLMClient`
    (the real vLLM-backed client lives in `vllm_client.py`; `ScriptedLLMClient`
    here serves CPU tests);
  * the mandatory running-draft template line at every step (§2.6, §18.2) —
    parsed with `cassi.labels.drafts.parse_draft`, tokens charged into c_t;
  * per-step `StepFeatures` x_t (§11): budget arithmetic via `cassi.budget.cost`,
    draft-stability via `cassi.labels.drafts`, history digests ≤64 tokens
    (~256 chars);
  * the TWO rollout modes of §2.1:
      - "rl": the episode ends when the executor emits the ANSWER action (or at
        T_max) — the policy experiences real termination economics (P6);
      - "forced_continuation": ANSWER is logged on the step (answered_flag=True,
        the free self-stop measurement) and the episode force-continues to T_max
        so continuation values are observable from every state (P2/P7 label
        collection — without this, labels are censored by the policy's own
        stopping choices).
  * optional monitor hook (§2.5 / §10 Alg.4): a monitor may stop the episode at
    inference time; during TRAINING rollouts the monitor is never passed —
    "exploration must be free, economics reach the policy only through rewards"
    (§2.1). The agent itself never enforces the budget for the same reason.

Emits a `cassi.common.schema.Trajectory` (§11 JSONL schema).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from cassi.budget.cost import tier_from_remaining, token_cost
from cassi.common.schema import EMPTY_DRAFT, Step, StepFeatures, Trajectory
from cassi.labels.drafts import (
    DRAFT_LINE_RE,
    DRAFT_TEMPLATE_INSTRUCTION,
    draft_stability_features,
    parse_draft,
    retrieval_overlap,
)

# ≤64-token observation digests (§11 history group); ~4 chars/token heuristic.
DIGEST_CHARS = 256
# Nominal caps used only for the tokens_pct / tool_pct features (§11) — NOT
# enforcement (the agent never truncates itself; §2.1).
DEFAULT_TOKENS_MAX = 32_768
HISTORY_KEEP = 8          # x_t stores the last-K history entries (serialize shows 3)

ROLLOUT_MODES = ("rl", "forced_continuation")

# ACTION: tool[argument] — last occurrence wins (models sometimes restate).
ACTION_RE = re.compile(r"ACTION:\s*([A-Za-z_]\w*)\s*\[(.*?)\]", re.DOTALL)

SYSTEM_TEMPLATE = (
    "You are a careful research agent solving a task step by step.\n"
    "Available tools:\n{tools}\n\n"
    "At each step, respond in EXACTLY this format:\n"
    "THOUGHT: <one short paragraph of reasoning>\n"
    "ACTION: <tool>[<argument>]   (or  answer[<your final answer>]  to finish)\n"
    "{draft_instruction}"
)


# --------------------------------------------------------------------- clients
@runtime_checkable
class LLMClient(Protocol):
    """Abstract chat client. `messages` are OpenAI-style [{'role', 'content'}]."""

    def generate(self, messages: list[dict], max_tokens: int) -> str: ...


class ScriptedLLMClient:
    """Returns pre-canned step outputs in order — CPU tests only (§16 P0 done-criterion
    is exercised without any GPU/vLLM dependency).

    `reset()` rewinds to the first output; `ReactAgent.run` calls it at episode
    start when present, so one client instance can drive many episodes
    deterministically. When outputs run out, the last one repeats (so
    forced-continuation episodes can always reach T_max)."""

    def __init__(self, outputs: list[str], *, repeat_last: bool = True):
        if not outputs:
            raise ValueError("ScriptedLLMClient needs at least one output")
        self.outputs = list(outputs)
        self.repeat_last = repeat_last
        self._i = 0
        self.calls: list[list[dict]] = []      # message log, for test assertions

    def reset(self) -> None:
        self._i = 0

    def generate(self, messages: list[dict], max_tokens: int) -> str:
        self.calls.append(messages)
        if self._i >= len(self.outputs):
            if not self.repeat_last:
                raise IndexError("ScriptedLLMClient exhausted its outputs")
            return self.outputs[-1]
        out = self.outputs[self._i]
        self._i += 1
        return out


# --------------------------------------------------------------------- monitor
@runtime_checkable
class MonitorProtocol(Protocol):
    """§2.5/§10 Alg.4 inference hook — see `cassi.executor.monitor.StopperMonitor`.

    `should_stop` is called on the PRE-ACTION state x_t; a non-None return is the
    stop reason ("monitor" = Δ̂≤threshold, "budget" = wallet exhausted)."""

    def should_stop(self, x: StepFeatures, allowance_dollars: float) -> str | None: ...


# ---------------------------------------------------------------------- result
@dataclass
class EpisodeResult:
    trajectory: Trajectory
    final_answer: str
    stopped_by: str                 # answer | t_max | monitor | budget | env_done
    self_terminated: bool           # executor emitted ANSWER before any monitor fired (§2.5)
    step_infos: list[dict] = field(default_factory=list)   # raw env infos, for collection-time
                                                           # quality scoring (never into x_t)
    draft_line_tokens: int = 0      # running-draft template tokens (feeds T4 accounting, §2.6)
    output_tokens_total: int = 0


# ----------------------------------------------------------------------- utils
def approx_tokens(text: str) -> int:
    """Deterministic ~4-chars/token approximation. Used for CPU-side budget
    arithmetic so tests need no tokenizer; real serving can substitute exact
    usage counts without changing any interface (costs stay comparable because
    every method runs through this same scaffold, §2.6)."""
    return max(1, (len(text) + 3) // 4)


def _digest(text: str) -> str:
    return " ".join((text or "").split())[:DIGEST_CHARS]


# ----------------------------------------------------------------------- agent
class ReactAgent:
    """The shared ReAct scaffold (§2.6). One instance is reusable across episodes."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_tokens_per_step: int = 512,
        tokens_max: int = DEFAULT_TOKENS_MAX,
    ):
        self.llm = llm
        self.max_tokens_per_step = max_tokens_per_step
        self.tokens_max = tokens_max

    # -- x_t construction (§11) ------------------------------------------------
    def _features(
        self, *, t: int, t_max: int, question: str, domain: str,
        tokens_used: int, tool_calls: int, dollars: float, allowance: float,
        draft_history: list[str], docid_sets: list[set], all_docids: set,
        history: list[dict],
    ) -> StepFeatures:
        stab = draft_stability_features(draft_history)
        draft = draft_history[-1] if draft_history else EMPTY_DRAFT
        return StepFeatures(
            tokens_used=tokens_used,
            tokens_pct=min(1.0, tokens_used / self.tokens_max),
            tool_calls=tool_calls,
            tool_pct=min(1.0, tool_calls / max(1, t_max)),
            dollars=dollars,
            dollars_pct=(dollars / allowance) if allowance > 0 else 1.0,
            burn_rate=(dollars / (t - 1)) if t > 1 else 0.0,
            tier=tier_from_remaining(dollars, allowance),
            step_idx=t,
            steps_since_draft_changed=stab["steps_since_draft_changed"],
            draft_edit_distance_last3=stab["draft_edit_distance_last3"],
            retrieval_overlap_last3=retrieval_overlap(docid_sets[-4:]),
            n_distinct_sources=len(all_docids),
            draft=draft,
            draft_len=len(draft) if draft != EMPTY_DRAFT else 0,
            question=question,
            domain=domain,
            history=history[-HISTORY_KEEP:],
        )

    # -- the loop ---------------------------------------------------------------
    def run(
        self,
        task: dict,
        env,
        *,
        mode: str = "rl",
        t_max: int = 10,
        allowance_dollars: float,
        wallet_size: str = "medium",
        group_id: str = "",
        rollout_idx: int = 0,
        monitor: MonitorProtocol | None = None,
        seed: int | None = None,
        iteration: int = 0,
    ) -> EpisodeResult:
        """One episode. `task` needs at least {'task_id', 'question'}; QA tasks
        carry 'gold' (read at collection scoring only, never by the agent).

        §2.1: in "rl" mode ANSWER ends the episode; in "forced_continuation"
        mode ANSWER sets answered_flag on the step and the loop continues to
        T_max (or env termination). The monitor, when provided, is an
        INFERENCE-time device only — never pass one during training rollouts."""
        if mode not in ROLLOUT_MODES:
            raise ValueError(f"mode must be one of {ROLLOUT_MODES}, got {mode!r}")
        if hasattr(self.llm, "reset"):
            self.llm.reset()

        question = task.get("question", task.get("goal", ""))
        domain = getattr(env, "domain", "qa")
        obs0 = env.reset(task)
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_TEMPLATE.format(
                tools=env.tools(), draft_instruction=DRAFT_TEMPLATE_INSTRUCTION)},
            {"role": "user", "content": obs0},
        ]

        steps: list[Step] = []
        step_infos: list[dict] = []
        history: list[dict] = []
        draft_history: list[str] = []
        docid_sets: list[set] = []
        all_docids: set = set()
        tokens_used = 0
        tool_calls = 0
        dollars = 0.0
        tau: int | None = None
        final_answer = ""
        format_hits = 0
        draft_line_tokens = 0
        output_tokens_total = 0
        stopped_by: str | None = None

        for t in range(1, t_max + 1):
            x = self._features(
                t=t, t_max=t_max, question=question, domain=domain,
                tokens_used=tokens_used, tool_calls=tool_calls,
                dollars=dollars, allowance=allowance_dollars,
                draft_history=draft_history, docid_sets=docid_sets,
                all_docids=all_docids, history=history,
            )
            if monitor is not None:
                reason = monitor.should_stop(x, allowance_dollars)
                if reason is not None:
                    stopped_by = reason
                    break

            output = self.llm.generate(messages, self.max_tokens_per_step)
            in_tok = sum(approx_tokens(m["content"]) for m in messages)
            out_tok = approx_tokens(output)
            cost = token_cost(in_tok, out_tok)
            tokens_used += in_tok + out_tok
            output_tokens_total += out_tok

            # draft template line (§18.2) — tokens charged into c_t (§2.6)
            m = DRAFT_LINE_RE.search(output or "")
            if m:
                format_hits += 1
                draft_line_tokens += approx_tokens(m.group(0))
            draft = parse_draft(output)

            # action
            am = ACTION_RE.findall(output or "")
            tool, arg = (am[-1][0].lower(), am[-1][1].strip()) if am else ("", "")
            done_env = False
            info: dict = {}
            is_answer = tool == "answer"
            if is_answer:
                a_type = "answer"
                answer_text = arg or (draft if draft != EMPTY_DRAFT else "")
                if draft == EMPTY_DRAFT and answer_text:
                    draft = answer_text
                obs = (
                    "(answer recorded; the episode continues — keep improving "
                    "your answer if further work is worth its cost)"
                    if mode == "forced_continuation" else "(final answer emitted)"
                )
            elif tool:
                a_type = "tool_call"
                tool_calls += 1
                obs, done_env, info = env.step(tool, arg)
                cost += float(info.get("tool_cost", 0.0))
                if info.get("docids"):
                    ids = set(info["docids"])
                    docid_sets.append(ids)
                    all_docids |= ids
            else:
                a_type = "reason"
                obs = "(no tool call this step)"

            dollars += cost
            draft_history.append(draft)
            steps.append(Step(
                x=x, a=a_type, o=_digest(obs), c=cost, tier=x.tier,
                draft=draft, q=0.0, answered_flag=is_answer,
            ))
            step_infos.append(info)
            history.append({"t": t, "action_type": a_type, "obs_digest": _digest(obs)})
            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": f"OBSERVATION: {obs}"})

            if is_answer:
                if tau is None:
                    tau = t
                    final_answer = answer_text
                if mode == "rl":
                    stopped_by = "answer"
                    break
            if done_env:
                stopped_by = "env_done"
                break
        if stopped_by is None:
            stopped_by = "t_max"

        if not final_answer:
            last_draft = draft_history[-1] if draft_history else EMPTY_DRAFT
            final_answer = last_draft if last_draft != EMPTY_DRAFT else ""

        traj = Trajectory(
            task_id=str(task.get("task_id", "")), domain=domain,
            allowance_B=float(allowance_dollars), wallet_size=wallet_size,
            group_id=group_id, rollout_idx=rollout_idx, steps=steps,
            outcome={
                "Q_tau": None, "success": None, "tau": tau,
                "gold": task.get("gold"), "collection_mode": mode,
                "seed": seed, "iteration": iteration,
                "final_answer": final_answer,
                "format_score": (format_hits / len(steps)) if steps else 0.0,
                "stopped_by": stopped_by,
            },
        )
        result = EpisodeResult(
            trajectory=traj, final_answer=final_answer, stopped_by=stopped_by,
            self_terminated=(stopped_by == "answer"),
            step_infos=step_infos, draft_line_tokens=draft_line_tokens,
            output_tokens_total=output_tokens_total,
        )
        if monitor is not None and hasattr(monitor, "record_episode"):
            monitor.record_episode(result)
        return result
