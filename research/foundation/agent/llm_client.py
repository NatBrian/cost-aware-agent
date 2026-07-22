"""OpenAI-compatible chat client for the vLLM executor server (F2)."""

import requests


class LLMError(RuntimeError):
    pass


class OpenAIChat:
    def __init__(self, endpoint: str, model: str, max_tokens: int = 512,
                 timeout: float = 120.0, extra_body: dict | None = None):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_body = extra_body or {}

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        body = {"model": self.model, "messages": messages,
                "temperature": temperature, "max_tokens": self.max_tokens,
                **self.extra_body}
        try:
            resp = requests.post(f"{self.endpoint}/chat/completions", json=body,
                                 timeout=self.timeout)
        except requests.RequestException as e:
            raise LLMError(f"LLM server unreachable: {e}") from e
        if resp.status_code != 200:
            raise LLMError(f"LLM server HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()["choices"][0]["message"]["content"]
