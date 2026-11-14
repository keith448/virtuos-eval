"""Model adapters. Standard library only, so the package installs cleanly."""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request

LETTERS = ("A", "B", "C")
RETRYABLE = {429, 500, 502, 503, 504}


def parse_letter(text: str) -> str | None:
    """Return the chosen A, B or C from a model response, or None."""
    text = (text or "").strip()
    match = re.match(r"^\s*\(?([ABC])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([ABC])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([ABC])\b", text)
    return match.group(1).upper() if match else None


class RetryPolicy:
    """Retry budget for HTTP calls.

    The providers read from hosted APIs, so a run of a hundred items will meet a
    429 or a dropped connection eventually. Retry those; never retry a 400.
    """

    def __init__(self, retries: int = 4, base_delay: float = 1.0, timeout: float = 60.0):
        self.retries = max(0, int(retries))
        self.base_delay = max(0.0, float(base_delay))
        self.timeout = max(1.0, float(timeout))

    def delay(self, attempt: int, exc: Exception | None = None) -> float:
        if isinstance(exc, urllib.error.HTTPError) and exc.headers:
            retry_after = exc.headers.get("Retry-After")
            if retry_after and str(retry_after).isdigit():
                return min(float(retry_after), 60.0)
        return min(self.base_delay * (2**attempt), 30.0) + random.uniform(0, 0.25)


def retry_call(fn, policy: RetryPolicy):
    """Call fn, retrying transient failures. Non-retryable errors pass through."""
    last: Exception | None = None
    for attempt in range(policy.retries + 1):
        try:
            return fn()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in RETRYABLE or attempt == policy.retries:
                raise
            time.sleep(policy.delay(attempt, exc))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
            if attempt == policy.retries:
                raise
            time.sleep(policy.delay(attempt))
    raise last  # pragma: no cover


class Provider:
    name = "provider"

    def complete(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class EchoProvider(Provider):
    """Always answers the same letter. Use to check the plumbing."""

    name = "echo"

    def __init__(self, letter: str = "A"):
        self.letter = letter.upper()

    def complete(self, prompt: str) -> str:
        return self.letter


class RandomProvider(Provider):
    """A chance baseline, reproducible by seed."""

    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def complete(self, prompt: str) -> str:
        return self._rng.choice(LETTERS)


def _options(prompt: str) -> dict[str, str]:
    found = {}
    for line in prompt.splitlines():
        match = re.match(r"^([ABC])\. (.*)$", line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


class LongestProvider(Provider):
    """Picks the longest option. Measures how much the length cue is worth."""

    name = "longest"

    def complete(self, prompt: str) -> str:
        options = _options(prompt)
        if not options:
            return "A"
        return max(options, key=lambda letter: len(options[letter]))


class ShortestProvider(Provider):
    """Picks the shortest option, as a counter-baseline."""

    name = "shortest"

    def complete(self, prompt: str) -> str:
        options = _options(prompt)
        if not options:
            return "A"
        return min(options, key=lambda letter: len(options[letter]))


def _post_json(url: str, payload: dict, headers: dict, policy: RetryPolicy) -> dict:
    data = json.dumps(payload).encode("utf-8")

    def send():
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=policy.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        return retry_call(send, policy)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"request to {url} failed: {exc}") from exc


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        retries: int = 4,
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.policy = RetryPolicy(retries=retries, timeout=timeout)
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 8,
        }
        data = _post_json(
            f"{self.base_url}/chat/completions",
            payload,
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            self.policy,
        )
        return data["choices"][0]["message"]["content"]


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        retries: int = 4,
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.policy = RetryPolicy(retries=retries, timeout=timeout)
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": 8,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = _post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            self.policy,
        )
        return data["content"][0]["text"]


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(
        self,
        model: str,
        host: str | None = None,
        *,
        retries: int = 4,
        timeout: float = 60.0,
    ):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.policy = RetryPolicy(retries=retries, timeout=timeout)

    def complete(self, prompt: str) -> str:
        data = _post_json(
            f"{self.host}/api/chat",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0},
            },
            {"Content-Type": "application/json"},
            self.policy,
        )
        return data["message"]["content"]


def get_provider(
    name: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    host: str | None = None,
    seed: int = 0,
    letter: str = "A",
    retries: int = 4,
    timeout: float = 60.0,
) -> Provider:
    name = name.lower()
    if name == "echo":
        return EchoProvider(letter)
    if name == "random":
        return RandomProvider(seed)
    if name == "longest":
        return LongestProvider()
    if name == "shortest":
        return ShortestProvider()
    if name == "openai":
        if not model:
            raise RuntimeError("--model is required for openai")
        return OpenAIProvider(model, api_key, base_url, retries=retries, timeout=timeout)
    if name == "anthropic":
        if not model:
            raise RuntimeError("--model is required for anthropic")
        return AnthropicProvider(model, api_key, retries=retries, timeout=timeout)
    if name == "ollama":
        if not model:
            raise RuntimeError("--model is required for ollama")
        return OllamaProvider(model, host, retries=retries, timeout=timeout)
    raise RuntimeError(f"unknown provider {name!r}; use echo, random, longest, shortest, openai, anthropic or ollama")
