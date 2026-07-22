"""ALFWorld env wrapper — paper_plan_v2 §5.1 (second training domain), §19.

IMPORTANT scope note (§19 reuse table): verl-agent (`langfengQ/verl-agent`,
GiGPO's official harness) owns the TRAINING-time rollout path for ALFWorld —
GRPO/GiGPO training runs through its batched env workers, not this class.
This wrapper exists for label COLLECTION (P2/P7 forced-continuation rollouts)
and INFERENCE/monitor evaluation (§2.5), where a single-episode text interface
around the ALFWorld TextWorld env is all that is needed.

Quality reading (§2.1): ALFWorld q_t = subgoal-completion fraction read directly
from env info (zero additional label cost) — `goal_condition_success_rate` in
the TextWorld info dict, or explicit subgoals_done/subgoals_total when a
harness provides counts. Extracted per step into `info` so collection scoring
(`step_quality`) never touches ground truth beyond env-privileged state.

All alfworld/textworld imports are lazy — CPU test environments must import
this module cleanly.
"""

from __future__ import annotations

from cassi.executor.envs.base import AgentEnv, alfworld_step_quality

_ALFWORLD_HELP = (
    "ALFWorld is not installed (or its data is missing): {err}\n"
    "Install the GPU/agent stack (paper_plan_v2 §16 P0):\n"
    "  pip install -r research/cassi/requirements-gpu.txt   # includes alfworld\n"
    "  alfworld-download                                     # task data\n"
    "and clone verl-agent (langfengQ/verl-agent) for the GiGPO training harness "
    "(scripts/p0_setup.sh). Set ALFWORLD_DATA if the data lives off-default."
)


class ALFWorldEnv(AgentEnv):
    """Single-episode text wrapper around the ALFWorld TextWorld env
    (`AlfredTWEnv`, train split by default)."""

    domain = "alfworld"

    def __init__(self, *, config_path: str | None = None, split: str = "train"):
        self.split = split
        self._env = None
        self._config = self._load_config(config_path)
        self._last_admissible: list[str] = []

    # -- lazy plumbing ------------------------------------------------------------
    @staticmethod
    def _load_config(config_path: str | None) -> dict:
        try:
            import alfworld.agents.environment  # noqa: F401 — presence check
            import yaml
        except ImportError as e:
            raise NotImplementedError(_ALFWORLD_HELP.format(err=e)) from e
        if config_path is None:
            import importlib.resources as res
            import os
            config_path = os.environ.get("ALFWORLD_CONFIG", "")
            if not config_path:
                # alfworld ships a base config; fall back to its packaged one
                config_path = str(res.files("alfworld") / "configs" / "base_config.yaml")
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _get_env(self):
        if self._env is None:
            from alfworld.agents.environment import get_environment  # lazy
            env_type = self._config["env"]["type"]      # AlfredTWEnv for text
            self._env = get_environment(env_type)(self._config, train_eval=self.split)
            self._env = self._env.init_env(batch_size=1)
        return self._env

    # -- interface ----------------------------------------------------------------
    def reset(self, task: dict) -> str:
        """ALFWorld draws its own next task from the split; `task` supplies the
        bookkeeping id (task_id) used in the trajectory schema."""
        env = self._get_env()
        obs, infos = env.reset()
        self._last_admissible = list(infos.get("admissible_commands", [[]])[0])
        return str(obs[0])

    def tools(self) -> str:
        return ("act[command]: execute one ALFWorld text command (e.g. 'go to "
                "shelf 1', 'take mug 2 from desk 1', 'open drawer 3'). Only "
                "admissible household commands succeed.")

    def step(self, tool: str, arg: str) -> tuple[str, bool, dict]:
        if tool != "act":
            return f"Unknown tool '{tool}'. Available: act[command].", False, {"tool_cost": 0.0}
        env = self._get_env()
        obs, _scores, dones, infos = env.step([arg])
        self._last_admissible = list(infos.get("admissible_commands", [[]])[0])
        info = {
            "tool_cost": 0.0,          # ALFWorld actions carry no external fee;
                                       # token costs are charged by the scaffold
            "won": bool(infos.get("won", [False])[0]),
            "goal_condition_success_rate": float(
                infos.get("goal_condition_success_rate", [0.0])[0]
            ),
            "admissible_commands": self._last_admissible[:20],
        }
        return str(obs[0]), bool(dones[0]), info

    # -- collection-time quality (§2.1: env subgoal fraction, zero label cost) -----
    def step_quality(self, draft: str, task: dict, info: dict) -> float:
        return alfworld_step_quality(info)
