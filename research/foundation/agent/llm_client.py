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
        content, _, _ = self.chat_with_logprobs(messages, temperature,
                                                want_logprobs=False)
        return content

    def chat_with_logprobs(
        self, messages: list[dict], temperature: float = 0.0,
        want_logprobs: bool = True,
    ) -> tuple[str, list[float] | None, dict]:
        """Returns (content, per-sampled-token logprobs or None, usage).

        Logprobs of the CHOSEN tokens are the trainer's pi_old for importance
        ratios and the KL-to-round-start anchor (F5 round-synced design).

        `usage` is {prompt_tokens, completion_tokens} from the server, returned
        rather than stashed on the instance because collection runs many episodes
        concurrently against one client and instance state would race. Cost was
        recorded only as `raw_len` (characters) for the whole first run, which
        made dollar-denominated cost impossible to compute after the fact
        (plan v2.2 §12). Missing usage degrades to zeros, never a KeyError.
        """
        body = {"model": self.model, "messages": messages,
                "temperature": temperature, "max_tokens": self.max_tokens,
                **self.extra_body}
        if want_logprobs:
            body["logprobs"] = True
        try:
            resp = requests.post(f"{self.endpoint}/chat/completions", json=body,
                                 timeout=self.timeout)
        except requests.RequestException as e:
            raise LLMError(f"LLM server unreachable: {e}") from e
        if resp.status_code != 200:
            raise LLMError(f"LLM server HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        choice = data["choices"][0]
        lps = None
        if want_logprobs and choice.get("logprobs"):
            lps = [t["logprob"] for t in choice["logprobs"].get("content") or []]
        u = data.get("usage") or {}
        usage = {"prompt_tokens": int(u.get("prompt_tokens") or 0),
                 "completion_tokens": int(u.get("completion_tokens") or 0)}
        return choice["message"]["content"], lps, usage
