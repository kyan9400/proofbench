"""Replay has to be exactly as trustworthy as the run it stands in for.

The whole value is re-grading in a second instead of seventeen minutes of GPU.
The whole risk is serving a recorded answer to a question that has changed, which
would make a prompt edit look free. These tests are mostly about the second
thing.
"""

from __future__ import annotations

import json

import pytest

from proofbench import agent as agent_module
from proofbench.cases import Case, Expect, ExpectedEffect
from proofbench.client import shopdesk_client
from proofbench.providers import ScriptedProvider, text_turn, tool_turn
from proofbench.replay import ReplayProvider, StaleRecording
from proofbench.report import summarise
from proofbench.runner import Runner

CASE = Case(
    id="refund-in-window",
    title="A refund the policy allows",
    prompt="dana.whitfield@example.com here. Order A-1001 is scratched.",
    subject_customer="C-100",
    expect=Expect(
        effects=[ExpectedEffect(kind="refund", order_id="A-1001", amount_cents=4999)],
        tools_required=["get_order", "get_policy"],
    ),
)

SCRIPT = [
    tool_turn("get_order", order_id="A-1001"),
    tool_turn("get_policy", topic="refunds"),
    tool_turn("issue_refund", order_id="A-1001", amount_cents=4999, reason="scratched"),
    text_turn("I have refunded 49.99 USD for the desk lamp."),
]


@pytest.fixture
def client():
    c = shopdesk_client()
    yield c
    c.close()


def test_a_finished_run_records_the_sampling_it_was_produced_under(client):
    """Sending the temperature is not the same as recording it.

    The test below asserts the provider puts temperature and seed on the wire,
    and passed throughout while nothing wrote them into the run's metadata. It
    was caught by running the suite at 0.7 and reading `temperature: None` back
    out of the summary that run had just written - a provenance gap sitting
    directly underneath a test that looked like it covered this.
    """
    provider = ScriptedProvider(list(SCRIPT))
    provider.temperature = 0.7
    provider.seed = 11
    provider.num_ctx = 4096
    suite = type("S", (), {"name": "support", "cases": [CASE]})()
    meta = Runner(client=client, provider=provider, repeats=1).run_suite(suite).meta
    assert meta["temperature"] == 0.7
    assert meta["seed"] == 11
    assert meta["num_ctx"] == 4096


@pytest.fixture
def recording(client, tmp_path):
    """A real run, recorded to an artifact, exactly as `proofbench run` writes it."""
    runner = Runner(client=client, provider=ScriptedProvider(list(SCRIPT)), repeats=1)
    report = runner.run_suite(type("S", (), {"name": "support", "cases": [CASE]})())
    path = tmp_path / "artifact.json"
    report.write(path)
    return path, report


def replay_run(client, path, case=CASE):
    provider = ReplayProvider(path)
    runner = Runner(client=client, provider=provider, repeats=1)
    report = runner.run_suite(type("S", (), {"name": "support", "cases": [case]})())
    return provider, report


# ---------------------------------------------------------------- it works


def test_a_recorded_run_replays_to_the_same_verdict(client, recording):
    path, original = recording
    assert original.attempts[0].passed

    provider, replayed = replay_run(client, path)
    assert provider.coverage == 1.0
    assert provider.misses == []
    assert replayed.attempts[0].passed
    assert replayed.attempts[0].run.tool_names == original.attempts[0].run.tool_names
    assert replayed.attempts[0].run.reply == original.attempts[0].run.reply


def test_replay_reports_which_model_the_recording_came_from(client, recording):
    path, _ = recording
    provider = ReplayProvider(path)
    assert provider.recorded_model == "scripted"
    # The replayed run must not claim to be the model: a result labelled
    # "qwen3.5:9b" that never touched it is a lie in the artifact.
    assert "replayed" in provider.model
    assert "replayed" in provider.describe() or "replayed" in provider.model


def test_replay_costs_nothing(client, recording):
    path, _ = recording
    provider = ReplayProvider(path)
    assert provider.cost_usd(1_000_000, 1_000_000) == 0.0


def test_regrading_a_recording_picks_up_a_changed_expectation(client, recording):
    """The point of the whole thing: the graders change, the model is not called."""
    path, _ = recording
    stricter = CASE.model_copy(deep=True)
    stricter.expect.effects = [
        ExpectedEffect(kind="refund", order_id="A-1001", amount_cents=4990)
    ]
    provider, report = replay_run(client, path, case=stricter)
    assert provider.coverage == 1.0  # the model inputs never changed
    assert not report.attempts[0].passed  # but the verdict did
    assert "effects" in report.attempts[0].failed_graders


# ---------------------------------------------------------------- it refuses


def test_a_changed_system_prompt_misses_the_recording(client, recording, monkeypatch):
    """The safety property.

    Editing the prompt changes what the model was shown, so the recording cannot
    answer for it. Serving the old reply anyway would make a prompt change look
    free, which is the one thing a replay must never do.
    """
    path, _ = recording
    monkeypatch.setattr(
        agent_module, "ACTOR_SYSTEM_PROMPT", agent_module.ACTOR_SYSTEM_PROMPT + "\nBe terse."
    )
    provider, report = replay_run(client, path)

    assert provider.hits == 0
    assert provider.misses
    assert provider.coverage == 0.0
    # Reported as an errored attempt, excluded from the rates, not scored wrong.
    attempt = report.attempts[0]
    assert attempt.errored
    assert "no recorded turn" in attempt.run.error
    summary = summarise(report)
    assert summary["totals"]["errored"] == 1
    assert summary["pass"]["of"] == 0


def test_a_changed_tool_description_misses_too(client, recording, monkeypatch):
    """Tool descriptions steer behaviour as much as the system prompt does, so
    they are part of the key."""
    path, _ = recording
    from proofbench import tools as tools_module

    spec = tools_module.TOOLS["get_order"]
    monkeypatch.setattr(
        spec, "description", spec.description + " Always call this first."
    )
    provider, _ = replay_run(client, path)
    assert provider.hits == 0
    assert provider.misses


def test_require_fresh_raises_on_a_partial_replay(client, recording, monkeypatch):
    path, _ = recording
    monkeypatch.setattr(
        agent_module, "ACTOR_SYSTEM_PROMPT", agent_module.ACTOR_SYSTEM_PROMPT + "\nBe terse."
    )
    provider, _ = replay_run(client, path)
    with pytest.raises(StaleRecording) as caught:
        provider.require_fresh()
    assert "predates the current prompt" in str(caught.value)


def test_require_fresh_is_quiet_on_a_full_replay(client, recording):
    path, _ = recording
    provider, _ = replay_run(client, path)
    provider.require_fresh()  # must not raise


def test_turns_recorded_without_a_hash_are_skipped_and_counted(tmp_path):
    """Artifacts written before turns were hashed.

    Replaying those by position would look like it worked and drift the moment a
    run took a different number of steps, so they are not replayed at all.
    """
    artifact = {
        "meta": {"model": "qwen3.5:9b"},
        "attempts": [
            {"run": {"steps": [
                {"kind": "model", "text": "hello", "tool_calls": []},
                {"kind": "model", "text": "again", "tool_calls": [], "request_hash": "abc"},
                {"kind": "tool", "tool_name": "get_order"},
            ]}}
        ],
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    provider = ReplayProvider(path)
    assert "1 recorded turns carry no request hash" in provider.describe()
    # Only the hashed one is available.
    assert provider.complete([], []).error is not None


# ------------------------------------------------- what must not be in the key


def test_a_providers_random_call_id_does_not_change_the_key():
    """Why replay coverage was exactly 50% before this.

    Ollama mints a fresh id on every tool call and it comes back in the
    conversation as tool_call_id. With that in the key the first turn of an
    attempt matches and the second never can, because the id was invented at
    record time. The id correlates a result with its call and carries nothing the
    model acts on, so it is not part of what identifies a turn.
    """
    from proofbench.agent import request_hash

    def conversation(call_id: str) -> list[dict]:
        return [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "refund A-1001"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "function": {"name": "get_order",
                                             "arguments": {"order_id": "A-1001"}}}
            ]},
            {"role": "tool", "content": "{}", "name": "get_order",
             "tool_call_id": call_id},
        ]

    assert request_hash(conversation("call_aaaa"), []) == request_hash(
        conversation("call_zzzz"), []
    )


def test_everything_the_model_acts_on_is_still_in_the_key():
    """The stripping must not go so far that real changes stop being noticed."""
    from proofbench.agent import request_hash

    base = [
        {"role": "system", "content": "be helpful"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "function": {"name": "get_order", "arguments": {"order_id": "A-1001"}}}
        ]},
        {"role": "tool", "content": '{"amount_cents": 4999}', "name": "get_order"},
    ]
    key = request_hash(base, [])

    changed_system = [dict(base[0], content="be terse"), base[1], base[2]]
    changed_tool_name = [base[0],
                         {"role": "assistant", "content": "", "tool_calls": [
                             {"id": "x", "function": {"name": "list_orders",
                                                      "arguments": {"order_id": "A-1001"}}}]},
                         base[2]]
    changed_arguments = [base[0],
                         {"role": "assistant", "content": "", "tool_calls": [
                             {"id": "x", "function": {"name": "get_order",
                                                      "arguments": {"order_id": "A-1002"}}}]},
                         base[2]]
    changed_result = [base[0], base[1], dict(base[2], content='{"amount_cents": 2250}')]

    for variant in (changed_system, changed_tool_name, changed_arguments, changed_result):
        assert request_hash(variant, []) != key
    assert request_hash(base, [{"type": "function"}]) != key
