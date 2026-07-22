"""Environment wrappers for the shared agent scaffold — paper_plan_v2 §5.1, §19.

CPU-safe exports only; `searchr1_qa` (verl-tool / Search-R1 retriever) and
`alfworld` (verl-agent) keep their heavy imports lazy and are imported by
module path where needed:

    from cassi.executor.envs.searchr1_qa import SearchR1QAEnv
    from cassi.executor.envs.alfworld import ALFWorldEnv
"""

from cassi.executor.envs.base import AgentEnv, MockSearchEnv, alfworld_step_quality

__all__ = ["AgentEnv", "MockSearchEnv", "alfworld_step_quality"]
