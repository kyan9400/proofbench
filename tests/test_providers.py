"""The provider layer, tested against a mock transport.

What the provider sends and how it reads a reply back decides whether every
number above it means anything, so the request body and the awkward reply shapes
are pinned here. No network: httpx's MockTransport answers in-process.
"""

from __future__ import annotations

import json

import httpx
import pytest

from proofbench.providers import (
    OllamaProvider,
    OpenAICompatProvider,
    ScriptedProvider,
    _parse_arguments,
    text_turn,
)


def ollama_with(handler, **kwargs) -> tuple[OllamaProvider, list[dict]]:
    """An OllamaProvider whose transport records the bodies it was given."""
    sent: list[dict] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return handler(request)

    provider = OllamaProvider(**kwargs)
    provider._client = httpx.Client(transport=httpx.MockTransport(record))
    return provider, sent


def ok_reply(content: str = "hello", **extra) -> httpx.Response:
    body = {
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 120,
        "eval_count": 8,
    }
    body.update(extra)
    return httpx.Response(200, json=body)


# ---------------------------------------------------------------- the request


def test_temperature_and_seed_are_always_sent():
    """So that a result can be traced to the sampling it was produced under.

    Not for determinism: temperature 0 gets close on this model - two runs out of
    three matched byte for byte - but the third diverged on 9 of 24 cases, so the
    recorded sampling is a provenance record rather than a guarantee.
    """
    provider, sent = ollama_with(lambda r: ok_reply(), temperature=0.0, seed=7)
    provider.complete([{"role": "user", "content": "hi"}], [])
    assert sent[0]["options"]["temperature"] == 0.0
    assert sent[0]["options"]["seed"] == 7
    assert sent[0]["stream"] is False


def test_think_is_omitted_unless_asked_for():
    """Omitted means the model keeps its own default, which is not the same as
    sending think=false."""
    provider, sent = ollama_with(lambda r: ok_reply())
    provider.complete([{"role": "user", "content": "hi"}], [])
    assert "think" not in sent[0]


@pytest.mark.parametrize("value", [True, False])
def test_think_is_sent_when_set(value: bool):
    provider, sent = ollama_with(lambda r: ok_reply(), think=value)
    provider.complete([{"role": "user", "content": "hi"}], [])
    assert sent[0]["think"] is value


def test_tools_are_only_sent_when_there_are_some():
    provider, sent = ollama_with(lambda r: ok_reply())
    provider.complete([{"role": "user", "content": "hi"}], [])
    assert "tools" not in sent[0]
    provider.complete([{"role": "user", "content": "hi"}], [{"type": "function"}])
    assert sent[1]["tools"] == [{"type": "function"}]


# ---------------------------------------------------------------- the reply


def test_token_counts_and_text_are_read_back():
    provider, _ = ollama_with(lambda r: ok_reply("the answer"))
    completion = provider.complete([], [])
    assert completion.text == "the answer"
    assert completion.prompt_tokens == 120
    assert completion.completion_tokens == 8
    assert completion.total_tokens == 128
    assert completion.error is None


def test_a_thinking_field_does_not_become_the_reply():
    """This model returns its deliberation separately. Folding it into the reply
    would put reasoning in front of the customer and break every reply grader."""
    provider, _ = ollama_with(
        lambda r: httpx.Response(200, json={
            "message": {
                "role": "assistant",
                "content": "Refunded 49.99.",
                "thinking": "Let me check the window. 8 days. Inside. Refund it.",
            },
            "prompt_eval_count": 10, "eval_count": 2,
        })
    )
    completion = provider.complete([], [])
    assert completion.text == "Refunded 49.99."
    assert "Let me check" not in completion.text


def test_tool_calls_are_read_with_arguments_as_an_object():
    provider, _ = ollama_with(
        lambda r: httpx.Response(200, json={
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "issue_refund",
                                  "arguments": {"order_id": "A-1001", "amount_cents": 4999}}}
                ],
            },
        })
    )
    completion = provider.complete([], [])
    assert len(completion.tool_calls) == 1
    call = completion.tool_calls[0]
    assert call.name == "issue_refund"
    assert call.arguments == {"order_id": "A-1001", "amount_cents": 4999}
    assert call.parse_error is None


def test_an_unreachable_endpoint_is_an_error_not_an_exception():
    """The runner needs this to mark the attempt errored rather than wrong."""
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider, _ = ollama_with(refuse)
    completion = provider.complete([], [])
    assert completion.error is not None
    assert "connection refused" in completion.error
    assert completion.text == ""


def test_an_http_error_status_is_an_error_too():
    provider, _ = ollama_with(lambda r: httpx.Response(500, text="boom"))
    completion = provider.complete([], [])
    assert completion.error is not None
    assert completion.text == ""


def test_latency_is_recorded_even_on_failure():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    provider, _ = ollama_with(refuse)
    completion = provider.complete([], [])
    assert completion.latency_ms >= 0
    assert "timed out" in completion.error


# ---------------------------------------------------------------- arguments


def test_arguments_given_as_a_json_string_are_decoded():
    args, raw, err = _parse_arguments('{"order_id": "A-1001"}')
    assert args == {"order_id": "A-1001"}
    assert err is None
    assert raw == '{"order_id": "A-1001"}'


def test_arguments_that_are_not_json_are_kept_verbatim_with_an_explanation():
    """Kept rather than raised: a model that emits this has told you something
    worth grading, and a provider that throws turns it into a crashed run."""
    args, raw, err = _parse_arguments("order_id=A-1001, amount=49.99")
    assert args == {}
    assert raw == "order_id=A-1001, amount=49.99"
    assert "not valid JSON" in err


def test_arguments_that_decode_to_a_non_object_are_rejected():
    args, raw, err = _parse_arguments("[1, 2, 3]")
    assert args == {}
    assert "not an object" in err


def test_absent_arguments_are_an_empty_object_not_an_error():
    args, raw, err = _parse_arguments(None)
    assert args == {}
    assert err is None


# ---------------------------------------------------------------- cost


def test_local_inference_reports_zero_cost_rather_than_a_guess():
    provider, _ = ollama_with(lambda r: ok_reply())
    assert provider.cost_usd(1_000_000, 1_000_000) == 0.0


def test_hosted_cost_uses_the_prices_it_was_given():
    """Prices are passed in, never read from a table in the repo: a stale table
    turns every cost figure into a wrong one that still looks like money."""
    provider = OpenAICompatProvider(
        base_url="https://example", model="m",
        price_in_per_mtok=3.0, price_out_per_mtok=15.0,
    )
    assert provider.cost_usd(1_000_000, 0) == pytest.approx(3.0)
    assert provider.cost_usd(0, 1_000_000) == pytest.approx(15.0)
    assert provider.cost_usd(500_000, 100_000) == pytest.approx(1.5 + 1.5)
    assert provider.cost_usd(0, 0) == 0.0


# ---------------------------------------------------------------- scripted


def test_the_scripted_provider_runs_out_rather_than_raising():
    """So a test that mis-counts turns fails on its assertion, not an IndexError."""
    provider = ScriptedProvider([text_turn("one")])
    assert provider.complete([], []).text == "one"
    assert provider.exhausted
    assert "exhausted" in provider.complete([], []).text


def test_the_scripted_provider_records_what_it_was_asked():
    provider = ScriptedProvider([text_turn("one")])
    provider.complete([{"role": "user", "content": "hello"}], [])
    assert provider.calls[0][0]["content"] == "hello"
