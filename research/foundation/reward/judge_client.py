"""Judge client (F3): prompted 27B via OpenAI-compatible endpoint.

Guarantees: strict JSON parse with one reprompt; NEUTRAL bits (score 0.5) on
final failure — parser failures must read as "no opinion", never "bad step";
every failure logged. Disk cache keyed (rubric_version, prompt hash) so reruns
are free and identical states get identical rewards. Every call counted for
the overhead report.
"""

import hashlib
import json
import re
from pathlib import Path

import requests

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class JudgeStats:
    def __init__(self):
        self.calls = 0
        self.cache_hits = 0
        self.parse_failures = 0
        self.transport_failures = 0
        self.total_tokens = 0

    def as_dict(self):
        return dict(calls=self.calls, cache_hits=self.cache_hits,
                    parse_failures=self.parse_failures,
                    transport_failures=self.transport_failures,
                    total_tokens=self.total_tokens)


def neutral_bits(bit_names: tuple[str, ...]) -> dict:
    """No-opinion result: every bit 0.5 -> score 0.5 -> reward exactly 0."""
    return {b: 0.5 for b in bit_names} | {"_neutral": True}


class JudgeClient:
    def __init__(self, endpoint: str, model: str, rubric_version: str,
                 cache_dir: str | Path, temperature: float = 0.0,
                 max_tokens: int = 512, parse_retries: int = 1,
                 timeout: float = 120.0):
        if "PLACEHOLDER" in endpoint.upper():
            raise ValueError(
                "judge endpoint is still the config placeholder — set "
                "judge.endpoint/judge.model in configs/foundation.yaml "
                "(Brian provides the URL before calibration; F3 doc)")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.rubric_version = rubric_version
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parse_retries = parse_retries
        self.timeout = timeout
        self.stats = JudgeStats()

    # -- cache -------------------------------------------------------------
    def _cache_path(self, prompt: str) -> Path:
        key = hashlib.sha256(f"{self.rubric_version}\n{prompt}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    # -- transport ---------------------------------------------------------
    def _complete(self, prompt: str) -> str:
        """POST with transport retries (connection resets happen under
        concurrent load on the shared judge server — round-2 lesson)."""
        import time
        last_err: Exception | None = None
        for wait in (0, 3, 10, 30):
            if wait:
                time.sleep(wait)
            try:
                resp = requests.post(
                    f"{self.endpoint}/chat/completions",
                    json={"model": self.model,
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": self.temperature,
                          "max_tokens": self.max_tokens},
                    timeout=self.timeout)
            except requests.RequestException as e:
                last_err = e
                continue
            if resp.status_code == 200:
                break
            last_err = RuntimeError(
                f"judge HTTP {resp.status_code}: {resp.text[:200]}")
        else:
            raise last_err or RuntimeError("judge unreachable")
        data = resp.json()
        usage = data.get("usage") or {}
        self.stats.total_tokens += int(usage.get("total_tokens") or 0)
        return data["choices"][0]["message"]["content"]

    # -- parsing -----------------------------------------------------------
    @staticmethod
    def _parse(text: str, bit_names: tuple[str, ...]) -> dict | None:
        m = _JSON_RE.search(text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        bits = {}
        for b in bit_names:
            v = obj.get(b)
            if v not in (0, 1, True, False):
                return None
            bits[b] = int(v)
        bits["reasoning"] = str(obj.get("reasoning", ""))[:500]
        return bits

    # -- public ------------------------------------------------------------
    def judge(self, prompt: str, bit_names: tuple[str, ...]) -> dict:
        """Returns {bit: 0|1, ...} (or 0.5s with _neutral=True on failure)."""
        cpath = self._cache_path(prompt)
        if cpath.exists():
            self.stats.cache_hits += 1
            return json.loads(cpath.read_text())
        attempt_prompt = prompt
        bits = None
        for _ in range(1 + self.parse_retries):
            self.stats.calls += 1
            try:
                text = self._complete(attempt_prompt)
            except Exception:
                # judge unreachable after all transport retries: neutral, never
                # crash the reward pipeline (round-2 lesson)
                self.stats.transport_failures += 1
                bits = None
                break
            bits = self._parse(text, bit_names)
            if bits is not None:
                break
            attempt_prompt = (prompt + "\n\nYour previous reply was not valid "
                              "JSON in the required schema. Reply with ONLY the JSON.")
        if bits is None:
            self.stats.parse_failures += 1
            bits = neutral_bits(bit_names)
        cpath.write_text(json.dumps(bits))
        return bits
