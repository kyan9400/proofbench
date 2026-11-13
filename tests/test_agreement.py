"""The statistics that decide whether a judged number is worth printing.

Worked by hand where possible, because this is the layer that turns "a model
agreed with me" into a published claim, and a quietly wrong kappa would make the
judge look trustworthy in exactly the cases where it is not.
"""

from __future__ import annotations

import pytest

from proofbench.agreement import (
    agreement,
    bootstrap_kappa,
    cohen_kappa,
    format_agreement,
    interpret,
)

P = "pass"
F = "fail"


# ---------------------------------------------------------------- kappa


def test_perfect_agreement_is_one():
    human = [P, F, P, F]
    assert cohen_kappa(human, list(human)) == pytest.approx(1.0)


def test_a_worked_example():
    """n=10, two failures each, agreeing on one of them.

    po = 8/10 = 0.80
    pe = 0.8*0.8 + 0.2*0.2 = 0.68
    k  = (0.80 - 0.68) / (1 - 0.68) = 0.375
    """
    human = [F, F, P, P, P, P, P, P, P, P]
    judge = [F, P, F, P, P, P, P, P, P, P]
    assert cohen_kappa(human, judge) == pytest.approx(0.375)


def test_systematic_disagreement_goes_negative():
    human = [P, P, F, F]
    judge = [F, F, P, P]
    kappa = cohen_kappa(human, judge)
    assert kappa is not None and kappa < 0


def test_kappa_is_undefined_when_both_raters_used_one_class():
    """Not zero. Zero reads as "no better than chance"; the truth is that this
    set cannot say anything at all, and the two deserve different words."""
    assert cohen_kappa([P, P, P, P], [P, P, P, P]) is None
    assert cohen_kappa([F, F, F], [F, F, F]) is None


def test_kappa_of_an_empty_set_is_undefined():
    assert cohen_kappa([], []) is None


def test_raters_must_have_rated_the_same_number_of_things():
    with pytest.raises(ValueError):
        cohen_kappa([P, F], [P])


# ---------------------------------------------------------------- prevalence


def test_the_prevalence_problem_is_visible_in_the_report():
    """Why observed agreement is never published on its own.

    A judge that calls everything a pass, on a set that is 95% passes, agrees 95%
    of the time and is worthless. Kappa says 0. The marginals are what let a
    reader see which situation they are in.
    """
    human = [F] + [P] * 19
    judge = [P] * 20
    result = agreement(human, judge, draws=500)

    assert result["observed_agreement"] == pytest.approx(0.95)
    assert result["kappa"] == pytest.approx(0.0)
    assert result["human_fail_rate"] == pytest.approx(0.05)
    assert result["judge_fail_rate"] == 0.0
    assert result["judge_recall_on_failures"] == 0.0  # caught none of them

    text = format_agreement(result)
    assert "95.0%" in text
    assert "one rater used a single class" in text


def test_a_judge_that_fails_everything_is_also_caught():
    human = [P] * 19 + [F]
    judge = [F] * 20
    result = agreement(human, judge, draws=500)
    assert result["judge_recall_on_failures"] == 1.0  # caught the one real failure
    assert result["judge_precision_on_failures"] == pytest.approx(1 / 20)
    assert result["observed_agreement"] == pytest.approx(0.05)


# ---------------------------------------------------------------- confusion


def test_the_confusion_matrix_treats_failure_as_the_positive_class():
    """The judge's job is catching bad replies, so recall means "of the replies a
    person called bad, how many did it catch"."""
    human = [F, F, P, P]
    judge = [F, P, F, P]
    result = agreement(human, judge, draws=500)
    assert result["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert result["judge_precision_on_failures"] == pytest.approx(0.5)
    assert result["judge_recall_on_failures"] == pytest.approx(0.5)
    assert result["disagreements"] == [1, 2]


def test_precision_and_recall_are_none_rather_than_zero_when_undefined():
    """No predicted failures means precision has no denominator. Reporting 0.0
    would claim the judge was wrong about something it never said."""
    result = agreement([P, P, P], [P, P, P], draws=100)
    assert result["judge_precision_on_failures"] is None
    assert result["judge_recall_on_failures"] is None


# ---------------------------------------------------------------- bootstrap


def test_the_bootstrap_interval_brackets_the_point_estimate():
    human = [F, F, F, P, P, P, P, P, F, P] * 3
    judge = [F, F, P, P, P, P, F, P, F, P] * 3
    kappa = cohen_kappa(human, judge)
    interval = bootstrap_kappa(human, judge, draws=2000)
    assert interval is not None
    low, high, undefined_share = interval
    assert low <= kappa <= high
    assert undefined_share < 0.05  # plenty of disagreement here, so draws survive


def test_the_bootstrap_is_reproducible():
    human = [F, P, F, P, F, P, P, P]
    judge = [F, P, P, P, F, P, F, P]
    assert bootstrap_kappa(human, judge, draws=500, seed=7) == bootstrap_kappa(
        human, judge, draws=500, seed=7
    )


def test_a_degenerate_interval_is_flagged_rather_than_read_as_precision():
    """One real failure, a judge that passes everything.

    Every resample that keeps the failure gives kappa exactly 0; every resample
    that loses it has no kappa. So the interval is (0.0, 0.0) - zero width, which
    reads as enormous confidence - while a third of the draws said nothing. Both
    facts have to reach the reader.
    """
    human = [F] + [P] * 9
    judge = [P] * 10
    interval = bootstrap_kappa(human, judge, draws=2000)
    assert interval is not None
    low, high, undefined_share = interval
    assert (low, high) == (0.0, 0.0)
    assert 0.2 < undefined_share < 0.5

    text = interpret(agreement(human, judge, draws=2000))
    assert "degenerate set rather than a precise estimate" in text
    assert "resamples had no kappa" in text


def test_the_interval_is_withheld_when_most_resamples_are_undefined():
    """With no disagreement at all and one class, there is nothing to resample."""
    assert bootstrap_kappa([P] * 8, [P] * 8, draws=1000) is None


def test_a_small_sample_is_called_out_rather_than_dressed_up():
    human = [F, F, P, P, P, P, P, P, P, P]
    judge = [F, P, F, P, P, P, P, P, P, P]
    result = agreement(human, judge, draws=2000)
    text = interpret(result)
    assert "10 labelled replies" in text
    # 0.375 lands in the "fair" band, but the interval at n=10 is far too wide to
    # lean on it, and the wording has to say so.
    assert "fair" in text
    assert "not worth leaning on" in text


def test_undefined_kappa_is_explained_not_printed_as_a_number():
    result = agreement([P, P, P], [P, P, P], draws=100)
    text = interpret(result)
    assert "undefined" in text
    assert "cannot say whether the judge is any good" in text
