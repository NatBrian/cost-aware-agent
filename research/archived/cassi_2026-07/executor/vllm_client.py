"""vLLM-backed LLMClient — paper_plan_v2 §19 stack (vLLM ≥ 0.17 serving Qwen3.5,
OpenAI-compatible endpoint) and §16 P0 (enable_thinking=False via chat-template
kwargs; the Qwen3.5 template strips <think> from history, so multi-turn is
token-in-token-out on the server side).

ALL vllm/openai imports live inside functions — this module must import cleanly
on CPU-only test environments (the CPU tests use `ScriptedLLMClient` instead).
"""

from __future__ import annotations

_INSTALL_HINT = (
    "The OpenAI client is required to talk to the vLLM server. Install the GPU "
    "stack: `pip install -r research/cassi/requirements-gpu.txt` (see also "
    "scripts/p0_setup.sh, paper_plan_v2 §16 P0). Then start vLLM, e.g.:\n"
    "  eval $(/mnt/src/zhanka/gpu_acquire.sh 1)\n"
    "  vllm serve Qwen/Qwen3.5-9B --port 8001\n"
    "and release GPUs afterwards with /mnt/src/zhanka/gpu_release.sh."
)


class VLLMClient:
    """OpenAI-compatible client for a local vLLM server (satisfies the
    `react_agent.LLMClient` protocol)."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8001/v1",
        model: str = "Qwen/Qwen3.5-9B",
        temperature: float = 1.0,
        enable_thinking: bool = False,       # §17 executor.enable_thinking
        timeout: float = 300.0,
        api_key: str = "EMPTY",
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.timeout = timeout
        self.api_key = api_key
        self._client = None                  # lazy — CPU import safety

    @classmethod
    def from_config(cls, cfg: dict, *, base_url: str | None = None,
                    temperature: float | None = None) -> "VLLMClient":
        """Model name + enable_thinking from configs/cassi.yaml `executor` (§17);
        rollout temperature defaults to executor.grpo.rollout_temp."""
        ex = cfg["executor"]
        return cls(
            base_url=base_url or "http://127.0.0.1:8001/v1",
            model=ex["base_model"],
            temperature=(temperature if temperature is not None
                         else float(ex["grpo"]["rollout_temp"])),
            enable_thinking=bool(ex.get("enable_thinking", False)),
        )

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI   # lazy import (CPU safety)
            except ImportError as e:
                raise RuntimeError(_INSTALL_HINT) from e
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                                  timeout=self.timeout)
        return self._client

    def generate(self, messages: list[dict], max_tokens: int) -> str:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
            # enable_thinking=False through the chat template (§16 P0 / §19 pins)
            extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
        )
        return resp.choices[0].message.content or ""
