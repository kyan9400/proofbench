"""The judge, and the bookkeeping that keeps its verdicts comparable to labels.

Nothing here calls a model. What is tested is the part that decides whether a
judged number is trustworthy: how an unreadable verdict is treated, what a label
is pinned to, and what happens when a reply changes under a label that was
written about the older one.
"""

from __future__ import annotations

import json

import pytest

from proofbench.judge import (
    _parse_verdict,
    judge_artifact,
    judge_reply,
    judged_rate,
    verdict_map,
)
from proofbench.labels import (
    UnlabelledEntries,
    align,
    distinct_replies,
    load_labels,
    reply_sha,
    rubric_sha,
    template,
)
from proofbench.providers import Completion, ScriptedProvider, text_turn


def artifact_of(*replies: tuple[str, str]) -> dict:
    """A minimal recorded run: (case_id, reply) pairs."""
    return {
        "meta": {"model": "test-model", "git_sha": "abc1234"},
        "attempts": [
            {"case_id": case_id, "run": {"reply": reply}} for case_id, reply in replies
        ],
    }


# ---------------------------------------------------------------- verdicts


def test_a_clean_json_verdict_is_read():
    verdict, reason = _parse_verdict('{"verdict": "fail", "reason": "no amount given"}')
    assert verdict == "fail"
    assert reason == "no amount given"


def test_a_verdict_wrapped_in_chatter_is_still_read():
    verdict, _ = _parse_verdict(
        'Here is my assessment:\n{"verdict": "pass", "reason": "fine"}\nHope that helps.'
    )
    assert verdict == "pass"


def test_an_unreadable_verdict_abstains_rather_than_passing():
    """The single easiest way to make a judge look good is to default its parse
    failures to agreement with the agent. It abstains instead."""
    verdict, reason = _parse_verdict("Well, it depends on how you read the rubric.")
    assert verdict == "abstain"
    assert "could not read a verdict" in reason


def test_an_empty_reply_abstains():
    assert _parse_verdict("")[0] == "abstain"
    assert _parse_verdict(None)[0] == "abstain"


def test_a_verdict_in_prose_is_accepted_only_when_unambiguous():
    assert _parse_verdict("This one is a clear fail.")[0] == "fail"
    assert _parse_verdict("I would pass this reply.")[0] == "pass"
    # Both words present and no JSON: not a decision.
    assert _parse_verdict("It could pass or fail depending on the rubric.")[0] == "abstain"


def test_a_malformed_verdict_value_abstains():
    assert _parse_verdict('{"verdict": "maybe", "reason": "hmm"}')[0] == "abstain"


def test_an_unreachable_judge_abstains_and_says_why():
    provider = ScriptedProvider([Completion(error="ReadTimeout: timed out")])
    result = judge_reply(provider, "customer text", "the rubric", "the reply")
    assert result["verdict"] == "abstain"
    assert "judge unavailable" in result["reason"]


# ------------------------------------------------- what the judge is shown


def test_the_judge_is_not_shown_what_the_agent_actually_did():
    """It judges the reply against the rubric, nothing else.

    Handing it the effect log or the deterministic verdict would let it anchor on
    the answer and then appear to agree with the graders, which measures nothing
    and reads as corroboration.
    """
    provider = ScriptedProvider([text_turn('{"verdict": "pass", "reason": "ok"}')])
    judge_reply(provider, "I want a refund", "Should state the amount.", "Refunded 49.99.")
    sent = provider.calls[0]
    body = " ".join(message["content"] for message in sent)
    assert "I want a refund" in body
    assert "Should state the amount." in body
    assert "Refunded 49.99." in body
    for leak in ("issue_refund", "effect", "amount_cents", "passed", "grader"):
        assert leak not in body


# ---------------------------------------------------------------- artifacts


def test_identical_repeats_are_judged_once_not_three_times():
    """Judging each attempt would cost three times as much and then count the
    same sentence three times in the agreement, as though it were independent
    evidence."""
    artifact = artifact_of(
        ("case-a", "Refunded 49.99."),
        ("case-a", "Refunded 49.99."),
        ("case-a", "Refunded 49.99."),
        ("case-b", "Cannot refund that."),
    )
    distinct = distinct_replies(artifact)
    assert len(distinct) == 2
    assert {d["case_id"]: d["attempts"] for d in distinct} == {"case-a": 3, "case-b": 1}

    provider = ScriptedProvider([
        text_turn('{"verdict": "pass", "reason": "ok"}'),
        text_turn('{"verdict": "fail", "reason": "no reason given to the customer"}'),
    ])
    judged = judge_artifact(
        provider, artifact, {"case-a": "rubric a", "case-b": "rubric b"},
        {"case-a": "msg a", "case-b": "msg b"},
    )
    assert judged["meta"]["replies"] == 2
    assert len(provider.calls) == 2
    assert judged["judgements"][0]["attempts_covered"] == 3


def test_a_case_with_no_rubric_is_skipped_not_invented_for():
    """With nothing to judge against, a judge grades its own taste."""
    artifact = artifact_of(("no-rubric", "Something."), ("has-rubric", "Something else."))
    provider = ScriptedProvider([text_turn('{"verdict": "pass", "reason": "ok"}')])
    judged = judge_artifact(
        provider, artifact, {"has-rubric": "a rubric", "no-rubric": "  "},
        {"has-rubric": "m", "no-rubric": "m"},
    )
    assert judged["meta"]["skipped_no_rubric"] == ["no-rubric"]
    assert judged["meta"]["replies"] == 1
    assert len(provider.calls) == 1


def test_empty_replies_are_not_judged():
    """A run that never replied has nothing for a judge to read; the completion
    grader already failed it."""
    artifact = artifact_of(("silent", ""), ("spoke", "Here you go."))
    assert [d["case_id"] for d in distinct_replies(artifact)] == ["spoke"]


def test_abstentions_are_left_out_of_the_comparison():
    """An abstention is not a verdict. Pairing it with a human label would score a
    failure of the judge as a disagreement about the reply."""
    judged = {
        "judgements": [
            {"case_id": "a", "reply_sha": "h1", "verdict": "pass"},
            {"case_id": "b", "reply_sha": "h2", "verdict": "abstain"},
            {"case_id": "c", "reply_sha": "h3", "verdict": "fail"},
        ]
    }
    assert verdict_map(judged) == {("a", "h1"): "pass", ("c", "h3"): "fail"}

    rate = judged_rate(judged)
    assert rate == {"replies": 3, "decided": 2, "abstained": 1, "passed": 1, "rate": 0.5}


def test_a_rate_with_nothing_decided_is_none_not_zero():
    judged = {"judgements": [{"case_id": "a", "reply_sha": "h", "verdict": "abstain"}]}
    assert judged_rate(judged)["rate"] is None


# ---------------------------------------------------------------- labels


def test_a_label_is_pinned_to_the_exact_reply_it_was_written_about():
    """The property that keeps labels from rotting silently."""
    assert reply_sha("Refunded 49.99.") == reply_sha("  Refunded 49.99.  ")
    assert reply_sha("Refunded 49.99.") != reply_sha("Refunded 49.98.")


def test_a_template_round_trips_through_yaml(tmp_path):
    """The first attempt at this emitted bare scalars through yaml.safe_dump,
    which appended a document-end marker and made the file unparseable."""
    import yaml

    artifact = artifact_of(("case-a", "A reply with: a colon and a #hash"))
    text = template(artifact, {"case-a": "A rubric, with punctuation: yes"})
    loaded = yaml.safe_load(text)
    assert len(loaded["labels"]) == 1
    assert loaded["labels"][0]["reply"] == "A reply with: a colon and a #hash"
    assert loaded["labels"][0]["label"] == "TODO"

    path = tmp_path / "labels.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(UnlabelledEntries) as caught:
        load_labels(path)
    assert "case-a" in str(caught.value)
    assert "still say TODO" in str(caught.value)


def test_alignment_pairs_labels_with_verdicts_on_the_same_reply(tmp_path):
    import yaml

    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump({"labels": [
        {"case_id": "a", "reply_sha": "h1", "label": "pass"},
        {"case_id": "b", "reply_sha": "h2", "label": "fail"},
    ]}), encoding="utf-8")
    labels = load_labels(path)

    alignment = align(labels, {("a", "h1"): "pass", ("b", "h2"): "pass"})
    assert alignment.human == ["pass", "fail"]
    assert alignment.judge == ["pass", "pass"]
    assert alignment.keys == [("a", "h1"), ("b", "h2")]


def test_an_unsure_label_is_excluded_and_counted(tmp_path):
    """A labeller forced to choose produces noise that lands in the kappa as
    though it were signal."""
    import yaml

    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump({"labels": [
        {"case_id": "a", "reply_sha": "h1", "label": "pass"},
        {"case_id": "b", "reply_sha": "h2", "label": "unsure"},
    ]}), encoding="utf-8")
    labels = load_labels(path)

    alignment = align(labels, {("a", "h1"): "pass", ("b", "h2"): "fail"})
    assert alignment.human == ["pass"]
    assert alignment.unsure == [("b", "h2")]
    assert "1 marked unsure by the labeller and excluded" in alignment.describe()


def test_a_label_whose_reply_changed_is_reported_stale_not_reused(tmp_path):
    """The whole point of hashing the reply.

    This suite's run-to-run variance means replies genuinely do change between
    runs. A label quietly carried over would be a judgement nobody made about
    text nobody read.
    """
    import yaml

    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump({"labels": [
        {"case_id": "a", "reply_sha": "old-hash", "label": "fail"},
    ]}), encoding="utf-8")
    labels = load_labels(path)

    alignment = align(labels, {("a", "new-hash"): "pass"})
    assert alignment.human == []
    assert alignment.stale_labels == [("a", "old-hash")]
    assert alignment.unlabelled == [("a", "new-hash")]
    assert "stale" in alignment.describe()


def test_judgements_serialise_to_something_a_person_can_read(tmp_path):
    artifact = artifact_of(("case-a", "Refunded 49.99."))
    provider = ScriptedProvider([
        text_turn('{"verdict": "fail", "reason": "promised but did not confirm"}')
    ])
    judged = judge_artifact(provider, artifact, {"case-a": "r"}, {"case-a": "m"})
    text = json.dumps(judged)
    assert "promised but did not confirm" in text
    assert judged["meta"]["judge_model"] == "scripted"
    # The reply is kept beside the verdict, so a disagreement can be read without
    # cross-referencing the artifact it came from.
    assert judged["judgements"][0]["reply"] == "Refunded 49.99."


def test_a_label_written_under_an_older_rubric_is_not_counted(tmp_path):
    """A label answers "does this reply satisfy *this* rubric".

    Pinning it to the reply alone leaves half the question unpinned. Sharpening a
    rubric - which this project did, twice, because two readers resolved the old
    wording differently - would otherwise carry the old answer over to a question
    nobody was asked, and the disagreement would silently vanish into agreement.
    """
    import yaml

    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump({"labels": [
        {"case_id": "reworded", "reply_sha": "h1", "label": "fail",
         "rubric_sha": reply_sha("the old wording")},
        {"case_id": "untouched", "reply_sha": "h2", "label": "pass",
         "rubric_sha": rubric_sha("the same wording")},
    ]}), encoding="utf-8")
    labels = load_labels(path)

    alignment = align(
        labels,
        verdicts={("reworded", "h1"): "pass", ("untouched", "h2"): "pass"},
        rubrics={"reworded": "a sharper wording", "untouched": "the same wording"},
    )
    assert alignment.rubric_changed == [("reworded", "h1")]
    assert alignment.keys == [("untouched", "h2")]
    assert alignment.human == ["pass"]
    assert "older wording" in alignment.describe()


def test_rewrapping_a_rubric_does_not_unpin_its_labels():
    """Re-flowing a YAML paragraph changes the line breaks, not the question."""
    assert rubric_sha("The reply must\nstate the amount.") == rubric_sha(
        "The reply must state the amount."
    )
    assert rubric_sha("must state the amount") != rubric_sha("may state the amount")


def test_rubrics_are_only_checked_when_the_caller_supplies_them(tmp_path):
    """Omitting them skips the check rather than failing every label against an
    empty rubric, which would read as "everything is stale"."""
    import yaml

    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump({"labels": [
        {"case_id": "a", "reply_sha": "h1", "label": "pass"},
    ]}), encoding="utf-8")
    alignment = align(load_labels(path), {("a", "h1"): "pass"})
    assert alignment.rubric_changed == []
    assert alignment.human == ["pass"]


def test_a_judge_abstention_is_not_reported_as_a_stale_label(tmp_path):
    """The bug this distinction exists for.

    The first measured kappa run reported two judge read-timeouts as "the reply
    changed since they were written" - a confident diagnosis of a different
    problem. A transient judge failure and a label that no longer applies need
    opposite responses: re-run the judge, or re-label.
    """
    import yaml

    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump({"labels": [
        {"case_id": "timed-out", "reply_sha": "h1", "label": "fail"},
        {"case_id": "moved-on", "reply_sha": "old", "label": "fail"},
    ]}), encoding="utf-8")
    labels = load_labels(path)

    # The judge saw both replies, but produced a verdict for neither: one because
    # it abstained, one because that reply is not in this run at all.
    alignment = align(
        labels,
        verdicts={},
        judged_replies={("timed-out", "h1")},
    )
    assert alignment.judge_abstained == [("timed-out", "h1")]
    assert alignment.stale_labels == [("moved-on", "old")]
    described = alignment.describe()
    assert "the judge did not decide" in described
    assert "a judge failure, not a disagreement" in described
    assert "stale" in described
