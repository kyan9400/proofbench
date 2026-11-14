"""A panel of raters, and the arithmetic that turns its votes into a reference.

The reference labels are the one input to the judge's kappa that no code can
check, so everything around them is pinned here: when a split panel counts as a
label and when it does not, that excluding the hard replies is never the only
figure available, that the panel's agreement with itself is computed correctly,
and that saved verdicts and pooled runs cannot quietly flatter the judge.
"""

from __future__ import annotations

import json

import pytest
import yaml

from proofbench.agreement import agreement, bootstrap_kappa
from proofbench.cli import main
from proofbench.judge import stale_verdict_keys
from proofbench.labels import align, load_labels, rubric_sha
from proofbench.panel import (
    Panel,
    PanelItem,
    consensus,
    default_threshold,
    fleiss_kappa,
    leave_one_out,
    load_panel,
    majority_threshold,
    split_counts,
    summarise_panel,
    to_labels,
    vote_table,
)

P, F, U = "pass", "fail", "unsure"


def panel_of(*rows: tuple[str, list[str]]) -> Panel:
    return Panel(items=[
        PanelItem(case_id=case_id, reply_sha="sha-" + case_id, votes=votes)
        for case_id, votes in rows
    ])


# ------------------------------------------------------------ the reference rule


def test_four_of_five_is_a_label_and_three_of_five_is_not():
    """The rule that was fixed before anyone voted.

    A reply the panel splits three to two on is a reply the rubric does not
    settle. Calling that a label would put the panel's coin-flips into the kappa
    as though they were ground truth.
    """
    assert default_threshold(5) == 4
    assert consensus([P, P, P, P, F], 4) == P
    assert consensus([F, F, F, F, U], 4) == F
    assert consensus([P, P, P, F, F], 4) == U
    assert consensus([F, F, F, U, U], 4) == U


def test_the_default_threshold_is_four_in_five_rounded_up():
    assert [default_threshold(n) for n in (2, 3, 4, 5, 7, 10)] == [2, 3, 4, 4, 6, 8]


def test_unsure_votes_never_add_up_to_a_label():
    """Five raters who all could not decide have not decided "pass"."""
    assert consensus([U, U, U, U, U], 4) == U
    assert consensus([U, U, U, U, P], 3) == U


def test_a_threshold_that_is_not_a_majority_is_refused():
    """With two of five, pass and fail could both qualify, and the label would
    depend on which the code happened to check first."""
    with pytest.raises(ValueError, match="not a majority"):
        consensus([P, P, F, F, U], 2)
    assert majority_threshold(5) == 3
    assert majority_threshold(4) == 3


def test_contested_replies_become_unsure_labels_with_the_split_written_down():
    labels = to_labels(panel_of(
        ("settled", [P, P, P, P, P]),
        ("one-out", [F, F, F, F, P]),
        ("contested", [P, P, P, F, F]),
    ))
    by_case = {entry.case_id: entry for entry in labels.labels}
    assert by_case["settled"].label == P
    assert by_case["one-out"].label == F
    assert by_case["contested"].label == U
    assert "3 pass, 2 fail" in by_case["contested"].note
    assert "contested" in by_case["contested"].note
    assert "1 pass, 4 fail" in by_case["one-out"].note


def test_the_majority_rule_keeps_the_replies_the_strict_rule_drops():
    """Dropping contested replies is the move that once inflated this project's
    kappa to 1.0, so it must always be possible to put them back and look."""
    panel = panel_of(("contested", [P, P, P, F, F]), ("settled", [F, F, F, F, F]))
    strict = {e.case_id: e.label for e in to_labels(panel).labels}
    loose = {e.case_id: e.label for e in to_labels(panel, majority_threshold(5)).labels}
    assert strict == {"contested": U, "settled": F}
    assert loose == {"contested": P, "settled": F}


def test_a_rater_who_skipped_an_item_is_not_a_vote_of_unsure():
    """A hole in a sheet is missing data. Padding it would invent a vote."""
    panel = panel_of(("a", [P, P, P, P, P]), ("b", [P, P, P, P]))
    with pytest.raises(ValueError, match="different numbers of votes"):
        _ = panel.raters


def test_a_panel_file_with_the_same_reply_twice_is_refused(tmp_path):
    path = tmp_path / "panel.yaml"
    row = {"case_id": "a", "reply_sha": "h", "votes": [P, P, F]}
    path.write_text(yaml.safe_dump({"items": [row, row]}), encoding="utf-8")
    with pytest.raises(ValueError, match="appears twice"):
        load_panel(path)


def test_a_vote_outside_pass_fail_unsure_is_refused(tmp_path):
    path = tmp_path / "panel.yaml"
    path.write_text(yaml.safe_dump({"items": [
        {"case_id": "a", "reply_sha": "h", "votes": [P, "maybe", F]},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_panel(path)


# ---------------------------------------------------------------- Fleiss' kappa


def test_fleiss_kappa_on_the_textbook_table():
    """Fleiss (1971), as reproduced in the standard worked example: 10 items, 14
    raters, 5 categories, kappa = 0.210."""
    table = [
        [0, 0, 0, 0, 14],
        [0, 2, 6, 4, 2],
        [0, 0, 3, 5, 6],
        [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0],
        [3, 2, 6, 3, 0],
        [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0],
        [0, 2, 2, 3, 7],
    ]
    assert fleiss_kappa(table) == pytest.approx(0.210, abs=0.0005)


def test_fleiss_kappa_is_one_when_every_rater_agrees_on_every_item():
    assert fleiss_kappa([[5, 0, 0], [0, 5, 0], [5, 0, 0]]) == pytest.approx(1.0)


def test_fleiss_kappa_is_undefined_not_zero_when_one_category_was_ever_used():
    """Every rater said pass to everything: chance agreement is already total.
    Zero would read as "no better than chance"; the truth is "cannot say"."""
    assert fleiss_kappa([[5, 0, 0], [5, 0, 0]]) is None
    assert fleiss_kappa([]) is None
    assert fleiss_kappa([[1, 0, 0]]) is None


def test_fleiss_kappa_refuses_a_table_with_uneven_raters():
    with pytest.raises(ValueError, match="same number of raters"):
        fleiss_kappa([[5, 0, 0], [3, 1, 0]])


def test_the_vote_table_counts_unsure_as_its_own_category():
    """Folding "unsure" into pass or fail would make a hesitant panel look like
    an agreeing one."""
    panel = panel_of(("a", [P, P, F, U, U]))
    assert vote_table(panel) == [[2, 1, 2]]


def test_splits_are_counted_by_how_settled_each_reply_was():
    panel = panel_of(
        ("a", [P, P, P, P, P]),
        ("b", [P, P, P, P, F]),
        ("c", [P, P, P, F, F]),
        ("d", [P, P, F, U, U]),
    )
    assert split_counts(panel) == {"unanimous": 1, "one_dissent": 1, "split": 2}


# ------------------------------------------------------------- rater against rest


def test_each_rater_is_scored_against_the_majority_of_the_others():
    """Rater 3 is the odd one out on the second reply and should be the only one
    whose agreement with the rest is not perfect."""
    panel = panel_of(
        ("a", [P, P, P, P, P]),
        ("b", [F, F, P, F, F]),
        ("c", [F, F, F, F, F]),
        ("d", [P, P, P, P, P]),
    )
    rows = leave_one_out(panel)
    assert [row["rater"] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[2]["observed_agreement"] == pytest.approx(0.75)
    for index in (0, 1, 3, 4):
        assert rows[index]["observed_agreement"] == pytest.approx(1.0)
        assert rows[index]["kappa"] == pytest.approx(1.0)


def test_a_rater_who_says_unsure_is_counted_as_unscored_not_as_agreeing():
    """Otherwise the surest way to look agreeable is never to commit."""
    panel = panel_of(("a", [U, P, P, P, P]), ("b", [U, F, F, F, F]), ("c", [P, P, P, P, P]))
    first = leave_one_out(panel)[0]
    assert first["said_unsure"] == 2
    assert first["n"] == 1


def test_replies_the_others_could_not_settle_are_left_out_of_a_raters_score():
    panel = panel_of(("split", [P, P, P, F, F]), ("clear", [P, P, P, P, P]))
    # For rater 1 the others are P,P,F,F: no strict majority, so it is skipped.
    first = leave_one_out(panel)[0]
    assert first["others_had_no_majority"] == 1
    assert first["n"] == 1


def test_the_panel_summary_reports_what_was_excluded_and_why():
    panel = panel_of(
        ("settled-pass", [P, P, P, P, P]),
        ("settled-fail", [F, F, F, F, P]),
        ("contested", [F, F, F, U, U]),
    )
    summary = summarise_panel(panel)
    assert summary["raters"] == 5
    assert summary["threshold"] == 4
    assert summary["labelled"] == {"pass": 1, "fail": 1}
    assert [c["case_id"] for c in summary["contested"]] == ["contested"]
    assert summary["contested"][0]["split"] == "3 fail, 2 unsure"


# ------------------------------------------------------ pooling runs, by the case


def test_resampling_by_case_with_one_reply_per_case_changes_nothing():
    """Clusters of one are items. The published intervals from single runs must
    not move because pooled runs became possible."""
    human = [P, F, P, F, P, P, F, P, P, P]
    judge = [P, F, P, P, P, P, F, P, F, P]
    plain = bootstrap_kappa(human, judge, draws=400, seed=3)
    by_case = bootstrap_kappa(
        human, judge, draws=400, seed=3, clusters=[str(i) for i in range(len(human))]
    )
    assert plain == by_case


def test_pooled_replies_to_one_case_are_resampled_together():
    """Two replies to a case share a prompt and a rubric. Resampling them one by
    one counts the same rubric twice and narrows the interval with evidence that
    is not there."""
    # The mechanism first. Two cases, each with one failing and one passing reply.
    # Drawn case by case, every resample holds both classes, so kappa is always
    # defined. Drawn reply by reply, an eighth of resamples land on one class.
    human = [F, P, F, P]
    by_reply = bootstrap_kappa(human, list(human), draws=2000, seed=5)
    by_case = bootstrap_kappa(
        human, list(human), draws=2000, seed=5, clusters=["a", "a", "b", "b"]
    )
    assert by_reply is not None and by_case is not None
    assert by_case[2] == 0.0
    assert by_reply[2] > 0.05

    # And what it does to an interval. Twenty cases answered twice; the judge
    # misses three failing cases, both replies each time. Reply by reply that is
    # six independent misses in forty; in truth it is three.
    human, judge, cases = [], [], []
    for case in range(20):
        for _ in range(2):
            cases.append("case-" + str(case))
            human.append(F if case < 6 else P)
            judge.append(F if 3 <= case < 6 else P)
    narrow = bootstrap_kappa(human, judge, draws=2000, seed=5)
    honest = bootstrap_kappa(human, judge, draws=2000, seed=5, clusters=cases)
    assert narrow is not None and honest is not None
    assert (honest[1] - honest[0]) > (narrow[1] - narrow[0]) + 0.2
    assert agreement(human, judge, draws=200, clusters=cases)["clusters"] == 20


def test_the_zero_disagreement_bound_counts_cases_not_replies():
    """Forty agreeing replies from twenty cases are twenty trials of the rubrics,
    not forty. Counting replies would tighten the bound for free."""
    human = [P, F] * 20
    cases = [str(i // 2) for i in range(40)]
    pooled = agreement(human, list(human), draws=50, clusters=cases)
    single = agreement(human, list(human), draws=50)
    assert pooled["max_disagreement_rate_if_none_seen"] == pytest.approx(1 - 0.05 ** (1 / 20))
    assert single["max_disagreement_rate_if_none_seen"] == pytest.approx(1 - 0.05 ** (1 / 40))
    assert (
        pooled["max_disagreement_rate_if_none_seen"]
        > single["max_disagreement_rate_if_none_seen"]
    )


def test_one_cluster_id_is_needed_per_item():
    with pytest.raises(ValueError, match="one cluster id"):
        bootstrap_kappa([P, F], [P, F], draws=10, clusters=["only-one"])


# --------------------------------------------- saved verdicts, and what they answer


def judgements_of(*rows: tuple[str, str, str, str]) -> dict:
    return {"judgements": [
        {"case_id": case_id, "reply_sha": sha, "rubric_sha": pinned, "verdict": verdict}
        for case_id, sha, pinned, verdict in rows
    ]}


def test_a_verdict_given_under_an_older_rubric_is_stale():
    """The judge side of the label-pinning bug. A saved verdict answers one
    wording of the rubric; reuse it after the rubric is reworded and it is being
    compared with a label that answers a different question."""
    rubrics = {"same": "must state the amount", "reworded": "must state the amount twice"}
    judged = judgements_of(
        ("same", "h1", rubric_sha("must state the amount"), "pass"),
        ("reworded", "h2", rubric_sha("must state the amount"), "pass"),
    )
    assert stale_verdict_keys(judged, rubrics) == {("reworded", "h2")}


def test_a_verdict_with_no_rubric_hash_cannot_be_attributed_and_is_stale():
    judged = {"judgements": [{"case_id": "a", "reply_sha": "h", "verdict": "pass"}]}
    assert stale_verdict_keys(judged, {"a": "any rubric"}) == {("a", "h")}


def test_a_stale_verdict_is_reported_as_stale_not_as_a_judge_that_abstained(tmp_path):
    """Each exclusion has its own remedy. "The judge did not decide" says re-run
    the judge and hope; this one says the verdict exists and is about something
    else."""
    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump({"labels": [
        {"case_id": "a", "reply_sha": "h1", "label": "fail", "rubric_sha": rubric_sha("new")},
        {"case_id": "b", "reply_sha": "h2", "label": "pass", "rubric_sha": rubric_sha("kept")},
    ]}), encoding="utf-8")
    alignment = align(
        load_labels(path),
        verdicts={("a", "h1"): "pass", ("b", "h2"): "pass"},
        rubrics={"a": "new", "b": "kept"},
        stale_verdicts={("a", "h1")},
    )
    assert alignment.verdict_stale == [("a", "h1")]
    assert alignment.judge_abstained == []
    assert alignment.keys == [("b", "h2")]
    assert "older wording of the rubric" in alignment.describe()
    assert "re-judge" in alignment.describe()


# ------------------------------------------------------------- the two commands


@pytest.fixture
def workspace(tmp_path):
    """A one-case suite, a panel that split on nothing, and saved verdicts."""
    suite = tmp_path / "suite"
    suite.mkdir()
    rubric = "The reply must state the amount refunded."
    (suite / "cases.yaml").write_text(yaml.safe_dump({
        "suite": "tiny",
        "cases": [
            {"id": "case-a", "title": "a", "prompt": "refund please", "rubric": rubric},
            {"id": "case-b", "title": "b", "prompt": "refund please", "rubric": rubric},
        ],
    }), encoding="utf-8")
    pinned = rubric_sha(rubric)
    votes = tmp_path / "votes.yaml"
    votes.write_text(yaml.safe_dump({"items": [
        {"case_id": "case-a", "reply_sha": "ha", "rubric_sha": pinned, "votes": [P, P, P, P, F]},
        {"case_id": "case-b", "reply_sha": "hb", "rubric_sha": pinned, "votes": [F, F, F, F, F]},
    ]}), encoding="utf-8")
    judgements = tmp_path / "judgements.json"
    judgements.write_text(json.dumps(judgements_of(
        ("case-a", "ha", pinned, "pass"), ("case-b", "hb", pinned, "pass"),
    )), encoding="utf-8")
    return {"suite": suite, "votes": votes, "judgements": judgements, "tmp": tmp_path,
            "rubric": rubric}


def test_panel_then_agree_runs_end_to_end_without_a_model(workspace, capsys):
    """Re-labelling or adding a rater must not cost another half hour of GPU."""
    labels = workspace["tmp"] / "labels.yaml"
    assert main(["panel", str(workspace["votes"]), "--labels-out", str(labels)]) == 0
    printed = capsys.readouterr().out
    assert "5 raters on 2 replies" in printed
    assert "1 pass, 1 fail, 0 contested" in printed

    out = workspace["tmp"] / "agreement.json"
    code = main([
        "agree", str(workspace["judgements"]), "--labels", str(labels),
        "--suite", str(workspace["suite"]), "--bootstrap-draws", "50", "--out", str(out),
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "2 replies labelled and judged" in printed
    assert "case-b" in printed and "labels fail, judge pass" in printed
    result = json.loads(out.read_text(encoding="utf-8"))["agreement"]
    assert result["n"] == 2
    assert result["confusion"] == {"tp": 0, "fp": 0, "fn": 1, "tn": 1}


def test_agree_refuses_to_use_a_verdict_given_under_another_rubric(workspace, capsys):
    labels = workspace["tmp"] / "labels.yaml"
    main(["panel", str(workspace["votes"]), "--labels-out", str(labels)])
    old = json.loads(workspace["judgements"].read_text(encoding="utf-8"))
    for record in old["judgements"]:
        record["rubric_sha"] = rubric_sha("an earlier wording")
    workspace["judgements"].write_text(json.dumps(old), encoding="utf-8")
    capsys.readouterr()

    code = main([
        "agree", str(workspace["judgements"]), "--labels", str(labels),
        "--suite", str(workspace["suite"]), "--bootstrap-draws", "50",
    ])
    printed = capsys.readouterr().out
    assert code == 2
    assert "older wording of the rubric" in printed
    assert "nothing to compare" in printed


def test_agree_will_not_pool_two_different_verdicts_on_one_reply(workspace, capsys):
    labels = workspace["tmp"] / "labels.yaml"
    main(["panel", str(workspace["votes"]), "--labels-out", str(labels)])
    other = workspace["tmp"] / "other.json"
    flipped = json.loads(workspace["judgements"].read_text(encoding="utf-8"))
    flipped["judgements"][0]["verdict"] = "fail"
    other.write_text(json.dumps(flipped), encoding="utf-8")
    capsys.readouterr()

    code = main([
        "agree", str(workspace["judgements"]), str(other), "--labels", str(labels),
        "--suite", str(workspace["suite"]),
    ])
    assert code == 2
    assert "two different verdicts" in capsys.readouterr().out


def test_panel_will_not_overwrite_labels_without_being_told_to(workspace, capsys):
    labels = workspace["tmp"] / "labels.yaml"
    assert main(["panel", str(workspace["votes"]), "--labels-out", str(labels)]) == 0
    assert main(["panel", str(workspace["votes"]), "--labels-out", str(labels)]) == 2
    assert main(["panel", str(workspace["votes"]), "--labels-out", str(labels), "--force"]) == 0
