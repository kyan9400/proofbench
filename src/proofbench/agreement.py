"""How much to trust a judge.

A judged pass rate on its own is a number with no error bars and no provenance:
it tells you what one model thought of another model's prose. The only thing that
makes it worth reading is a measurement of how often that judge agrees with a
person on the same items.

Three numbers are reported together, deliberately, because any one of them alone
misleads:

* **Observed agreement** is what people actually want to know, and it flatters
  the judge whenever one class dominates. A judge that says "pass" to everything
  agrees 90% of the time on a set that is 90% passes, while being useless.

* **Cohen's kappa** corrects for agreement by chance, and in exchange becomes
  unstable and hard to read at exactly those lopsided margins - the prevalence
  problem. A kappa of 0.3 on a 90/10 split is not the same evidence as a kappa of
  0.3 on a 50/50 split.

* **The marginals** - how often each rater said "fail" - are what let a reader
  tell those two situations apart, so they are never omitted.

Plus a bootstrap interval, because these sets are small. With a couple of dozen
labelled items a point estimate of kappa is not a measurement, and reporting one
bare would be the same mistake the rest of this project is built to avoid.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

Label = Literal["pass", "fail"]

# "fail" is the positive class throughout: the judge's job is to catch bad
# replies, so recall here means "of the replies a person judged bad, how many did
# the judge catch".
POSITIVE: Label = "fail"


def cohen_kappa(human: Sequence[Label], judge: Sequence[Label]) -> float | None:
    """Cohen's kappa for two raters on the same items.

    Returns None when it is undefined, which happens whenever both raters used
    only one class: chance agreement is then 1, the denominator is 0, and there
    is no answer. Returning 0.0 there would read as "no better than chance" when
    the truth is "this set cannot tell you anything".
    """
    if len(human) != len(judge):
        raise ValueError("raters disagree on how many items there are")
    n = len(human)
    if n == 0:
        return None

    classes = ("pass", "fail")
    observed = sum(1 for a, b in zip(human, judge, strict=True) if a == b) / n
    expected = sum(
        (human.count(c) / n) * (judge.count(c) / n) for c in classes
    )
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def _confusion(human: Sequence[Label], judge: Sequence[Label]) -> dict[str, int]:
    tp = sum(1 for h, j in zip(human, judge, strict=True) if h == POSITIVE and j == POSITIVE)
    fp = sum(1 for h, j in zip(human, judge, strict=True) if h != POSITIVE and j == POSITIVE)
    fn = sum(1 for h, j in zip(human, judge, strict=True) if h == POSITIVE and j != POSITIVE)
    tn = sum(1 for h, j in zip(human, judge, strict=True) if h != POSITIVE and j != POSITIVE)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def bootstrap_kappa(
    human: Sequence[Label],
    judge: Sequence[Label],
    draws: int = 5000,
    seed: int = 7,
) -> tuple[float, float, float] | None:
    """A percentile interval for kappa, by resampling the labelled items.

    Returns (low, high, undefined_share). The third number is not decoration: a
    resample that lands on a single class has no kappa and is dropped, and with
    only one or two disagreements a large share of draws drop out. That share has
    to travel with the interval, because without it the interval looks like
    evidence it is not.

    A degenerate case shows why. One real failure, a judge that calls everything
    a pass: every resample that keeps the failure gives kappa exactly 0, and
    every resample that loses it is undefined. The interval comes back (0.0, 0.0)
    - zero width, which reads as enormous confidence - while a third of the draws
    said nothing at all. The share is what stops that being published as
    precision.

    None is returned only when most draws were undefined, where no interval
    computed from the survivors would mean anything.
    """
    n = len(human)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(draws):
        index = rng.integers(0, n, size=n)
        resampled_human = [human[i] for i in index]
        resampled_judge = [judge[i] for i in index]
        value = cohen_kappa(resampled_human, resampled_judge)
        if value is not None:
            values.append(value)
    undefined_share = 1 - (len(values) / draws)
    if len(values) < draws * 0.5:
        return None
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
        undefined_share,
    )


def disagreement_upper_bound(n: int, confidence: float = 0.95) -> float | None:
    """With no disagreements at all, the largest true disagreement rate still
    consistent with what was seen.

    This is the number that matters when a judge agrees with a person on every
    item. A bootstrap cannot express uncertainty there - every resample of a
    perfect set is perfect, so the interval comes back zero-width and reads as
    certainty. The exact one-sided bound says the useful thing instead: nought
    disagreements in 21 replies is consistent with a true disagreement rate as
    high as 13%.

    Clopper-Pearson, which for zero events reduces to 1 - alpha^(1/n).
    """
    if n <= 0:
        return None
    return 1 - (1 - confidence) ** (1 / n)


def agreement(
    human: Sequence[Label], judge: Sequence[Label], draws: int = 5000
) -> dict[str, Any]:
    """Everything needed to decide whether to believe a judged number."""
    if len(human) != len(judge):
        raise ValueError("raters disagree on how many items there are")
    n = len(human)
    human = list(human)
    judge = list(judge)

    observed = (sum(1 for a, b in zip(human, judge, strict=True) if a == b) / n) if n else 0.0
    kappa = cohen_kappa(human, judge)
    confusion = _confusion(human, judge)

    tp, fp, fn = confusion["tp"], confusion["fp"], confusion["fn"]
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    return {
        "n": n,
        "observed_agreement": observed,
        "kappa": kappa,
        "kappa_ci95": bootstrap_kappa(human, judge, draws=draws) if n else None,
        "human_fail_rate": (human.count("fail") / n) if n else 0.0,
        "judge_fail_rate": (judge.count("fail") / n) if n else 0.0,
        "confusion": confusion,
        "judge_precision_on_failures": precision,
        "judge_recall_on_failures": recall,
        "disagreements": [
            i for i, (h, j) in enumerate(zip(human, judge, strict=True)) if h != j
        ],
        # Only meaningful when nothing disagreed, which is exactly the case the
        # bootstrap cannot describe.
        "max_disagreement_rate_if_none_seen": (
            disagreement_upper_bound(n)
            if n and not any(h != j for h, j in zip(human, judge, strict=True))
            else None
        ),
    }


def interpret(result: dict[str, Any]) -> str:
    """A one-line reading, stated conservatively.

    The bands are Landis and Koch's, which are a convention rather than a law.
    They are reported as such, and the interval is what decides whether the band
    means anything at this sample size.
    """
    kappa = result["kappa"]
    n = result["n"]
    if kappa is None:
        return (
            "kappa is undefined here: both raters used a single class, so this set "
            "cannot say whether the judge is any good"
        )

    band = (
        "poor" if kappa < 0.0
        else "slight" if kappa < 0.21
        else "fair" if kappa < 0.41
        else "moderate" if kappa < 0.61
        else "substantial" if kappa < 0.81
        else "almost perfect"
    )
    interval = result["kappa_ci95"]
    text = (
        "kappa " + str(round(kappa, 3)) + " (" + band + ", by the usual bands)"
        + " on " + str(n) + " labelled replies"
    )
    if interval is None:
        return text + "; the interval is not reportable at this sample size"

    low, high, undefined_share = interval
    text += "; 95% " + str(round(low, 3)) + " to " + str(round(high, 3))

    # A zero-width interval here means the resamples were degenerate, not that
    # the estimate is precise, and it must never read as the second thing. When
    # the cause is perfect agreement there is a better number to give.
    bound = result.get("max_disagreement_rate_if_none_seen")
    if math.isclose(low, high) and bound is not None:
        text += (
            " - but nothing disagreed, so the interval is zero-width by "
            "construction and says nothing. Nought disagreements in " + str(n)
            + " replies is consistent with a true disagreement rate as high as "
            + str(round(bound * 100, 1)) + "%"
        )
    elif math.isclose(low, high):
        text += (
            " - zero width because every usable resample gave the same value, "
            "which is a degenerate set rather than a precise estimate"
        )
    elif low < 0.21 < high or (high - low) > 0.5:
        text += " - wide enough that the band is not worth leaning on"

    if undefined_share > 0.1:
        text += (
            "; " + str(round(undefined_share * 100)) + "% of resamples had no kappa "
            "at all, so there is less here than the interval suggests"
        )
    return text


def format_agreement(result: dict[str, Any], width: int = 78) -> str:
    lines = ["=" * width, "judge against hand labels", "=" * width]
    lines.append(interpret(result))
    lines.append("-" * width)
    lines.append(
        "observed agreement   " + str(round(result["observed_agreement"] * 100, 1)) + "%"
        + "   on " + str(result["n"]) + " replies"
    )
    lines.append(
        "called it a failure  person " + str(round(result["human_fail_rate"] * 100, 1)) + "%"
        + "   judge " + str(round(result["judge_fail_rate"] * 100, 1)) + "%"
    )
    confusion = result["confusion"]
    lines.append(
        "on the failures      caught " + str(confusion["tp"])
        + "   missed " + str(confusion["fn"])
        + "   invented " + str(confusion["fp"])
    )
    precision = result["judge_precision_on_failures"]
    recall = result["judge_recall_on_failures"]
    lines.append(
        "                     precision "
        + ("-" if precision is None else str(round(precision * 100, 1)) + "%")
        + "   recall "
        + ("-" if recall is None else str(round(recall * 100, 1)) + "%")
    )
    if result["human_fail_rate"] in (0.0, 1.0) or result["judge_fail_rate"] in (0.0, 1.0):
        lines.append(
            "note: one rater used a single class, so agreement here is not evidence "
            "the judge works"
        )
    return "\n".join(lines)
