"""Model providers.

Three of them: Ollama (what the live demo runs on, against the GPU), any
OpenAI-compatible endpoint, and a scripted provider that replays a fixed list of
turns. The scripted one is not a convenience — it is what lets the whole platform,
graders included, be tested in CI with no model, no GPU and no network, which is
the difference between a test suite that runs on every push and one that does not.

Malformed tool-call arguments are kept, not raised. A model that emits
`{"amount_cents": "forty-nine ninety-nine"}` has told you something worth
measuring, and a provider that throws on it turns a gradeable failure into a
crashed run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    # Set when the model emitted something that was not a JSON object.
    parse_error: str | None = None
    id: str | None = None


@dataclass
class Completion:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    # Non-fatal transport or decoding trouble, recorded so a run can be marked
    # errored rather than silently scored as a bad answer.
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Provider(Protocol):
    name: str
    model: str

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Completion: ...

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float: ...


def _parse_arguments(value: Any) -> tuple[dict[str, Any], str, str | None]:
    """Normalise whatever the model called arguments into a dict.

    Returns (arguments, raw, parse_error). A non-object stays out of `arguments`
    so a grader sees empty arguments plus an explanation, rather than a dict with
    invented keys.
    """
    if isinstance(value, dict):
        return value, json.dumps(value, sort_keys=True), None
    if value is None:
        return {}, "", None
    raw = value if isinstance(value, str) else repr(value)
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError) as exc:
        return {}, raw, "arguments were not valid JSON: " + str(exc)
    if not isinstance(decoded, dict):
        return {}, raw, "arguments decoded to " + type(decoded).__name__ + ", not an object"
    return decoded, raw, None


class OllamaProvider:
    """Ollama's /api/chat, with tool calling.

    temperature defaults to 0 and a seed is always sent. That does not make a run
    bit-identical — batching and GPU non-determinism still move things — which is
    why the runner repeats cases and the report carries intervals instead of
    pretending one run is the answer.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.5:9b",
        temperature: float = 0.0,
        seed: int = 7,
        timeout_s: float = 180.0,
        num_ctx: int = 8192,
        think: bool | None = None,
    ) -> None:
        self.name = "ollama"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        # None leaves the model's own default alone. Setting it False matters for
        # a reviewer: asked for a one-line JSON verdict, this model deliberated
        # for 4456 generated tokens before answering. With thinking off it
        # answered the same question correctly in 6. Nothing about a structured
        # approve/reject verdict justifies the first number, and a reviewer that
        # costs more than the action it reviews will not be deployed.
        self.think = think
        self._client = httpx.Client(timeout=timeout_s)

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        # Local inference on hardware that is already paid for. Reported as zero
        # on purpose: the cost panel for a local run should read zero, not an
        # invented dollar figure.
        return 0.0

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
            },
        }
        if tools:
            payload["tools"] = tools
        if self.think is not None:
            payload["think"] = self.think

        started = time.monotonic()
        try:
            response = self._client.post(self.base_url + "/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            return Completion(
                latency_ms=int((time.monotonic() - started) * 1000),
                error=type(exc).__name__ + ": " + str(exc),
            )
        latency_ms = int((time.monotonic() - started) * 1000)

        message = body.get("message") or {}
        calls: list[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            fn = raw_call.get("function") or {}
            args, raw, err = _parse_arguments(fn.get("arguments"))
            calls.append(
                ToolCall(
                    name=fn.get("name") or "",
                    arguments=args,
                    raw_arguments=raw,
                    parse_error=err,
                    id=raw_call.get("id"),
                )
            )

        return Completion(
            text=message.get("content") or "",
            tool_calls=calls,
            prompt_tokens=int(body.get("prompt_eval_count") or 0),
            completion_tokens=int(body.get("eval_count") or 0),
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        self._client.close()


class OpenAICompatProvider:
    """Any /v1/chat/completions endpoint.

    Prices are passed in rather than looked up from a table baked into the repo:
    a stale price table silently turns every cost number in the dashboard into a
    wrong one, and nobody notices because it still looks like money.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.0,
        price_in_per_mtok: float = 0.0,
        price_out_per_mtok: float = 0.0,
        timeout_s: float = 180.0,
    ) -> None:
        self.name = "openai-compatible"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.price_in_per_mtok = price_in_per_mtok
        self.price_out_per_mtok = price_out_per_mtok
        headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        self._client = httpx.Client(timeout=timeout_s, headers=headers)

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.price_in_per_mtok
            + completion_tokens * self.price_out_per_mtok
        ) / 1_000_000

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        started = time.monotonic()
        try:
            response = self._client.post(self.base_url + "/v1/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            return Completion(
                latency_ms=int((time.monotonic() - started) * 1000),
                error=type(exc).__name__ + ": " + str(exc),
            )
        latency_ms = int((time.monotonic() - started) * 1000)

        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls: list[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            fn = raw_call.get("function") or {}
            args, raw, err = _parse_arguments(fn.get("arguments"))
            calls.append(
                ToolCall(
                    name=fn.get("name") or "",
                    arguments=args,
                    raw_arguments=raw,
                    parse_error=err,
                    id=raw_call.get("id"),
                )
            )

        usage = body.get("usage") or {}
        return Completion(
            text=message.get("content") or "",
            tool_calls=calls,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        self._client.close()


class ScriptedProvider:
    """Replays a fixed list of Completions, in order.

    Used by the tests to drive the agent and every grader down an exact path. Once
    the script runs out it returns a plain reply, so a test that mis-counts turns
    fails on its assertion rather than on an IndexError.
    """

    def __init__(self, script: list[Completion], model: str = "scripted") -> None:
        self.name = "scripted"
        self.model = model
        self.script = list(script)
        self.calls: list[list[dict[str, Any]]] = []
        self._index = 0

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.0

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Completion:
        self.calls.append(list(messages))
        if self._index >= len(self.script):
            return Completion(text="(script exhausted)")
        completion = self.script[self._index]
        self._index += 1
        return completion

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self.script)


def tool_turn(name: str, **arguments: Any) -> Completion:
    """Shorthand for a scripted turn that calls one tool."""
    return Completion(
        tool_calls=[
            ToolCall(
                name=name,
                arguments=dict(arguments),
                raw_arguments=json.dumps(arguments, sort_keys=True),
            )
        ],
        prompt_tokens=100,
        completion_tokens=20,
        latency_ms=10,
    )


def text_turn(text: str) -> Completion:
    """Shorthand for a scripted turn that just replies."""
    return Completion(text=text, prompt_tokens=100, completion_tokens=30, latency_ms=10)
