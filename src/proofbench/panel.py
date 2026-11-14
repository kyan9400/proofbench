"""A panel of raters, instead of one rater who also wrote the rubrics.

The first reference labels in this project were written by the same hand that
wrote the suite and the rubrics, and then two of them were revised after reading
the judge's reasoning. Both are ways of manufacturing agreement: a single rater's
blind spots are invisible, and a rater who has seen the verdicts cannot
un-see them. The second of those produced a kappa of exactly 1.0 that measured
nothing.

A panel fixes what one rater cannot:

**Raters are blind and work alone.** Each is shown exactly what the judge is
shown - the customer's message, the rubric and the reply - under an opaque id and
in a different order from every other rater. No case name, no effect log, no
deterministic verdict, no earlier label, no judge verdict, no other rater.

**The reference is whatever most of the panel says, and the rule is fixed before
anyone votes.** A reply is labelled pass or fail only when at least four raters
in five agree (`default_threshold`). Anything short of that is *contested* and is
excluded from the agreement - and reported, with the split, because a reply the
panel cannot settle is evidence about the rubric rather than about the judge.

**The panel's agreement with itself is measured.** Fleiss' kappa across the
raters says how settled the reference is, and each rater is scored against the
other four, which gives the judge something to be compared with: a judge that
agrees with the panel as well as the panel's own members do is doing as well as
this reference can show.

Excluding contested items is the same move that once inflated this project's
kappa to 1.0 - dropping the items most likely to disagree - so it is never
reported alone. `to_labels` takes the threshold as an argument so the agreement
can be re-computed under a simple majority, where almost nothing is excluded, and
the two figures are published side by side.

What a panel does not fix: if every rater is a model, their agreement is not a
person's. It is a stronger reference than one author, and a weaker one than
people. Say which it was.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .agreement import cohen_kappa
from .labels import LabelEntry, LabelSet

Vote = Literal["pass", "fail", "unsure"]
CATEGORIES: tuple[Vote, ...] = ("pass", "fail", "unsure")


class PanelItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    reply_sha: str
    # The wording the raters were shown, by hash, for the same reason a label
    # carries one: a vote is an answer to one particular rubric.
    rubric_sha: str = ""
    # One vote per rater, in rater order. A rater who did not rate an item is a
    # hole in the sheet, not a vote of "unsure", so every item must carry the
    # same number of votes; `Panel` enforces it.
    votes: list[Vote]
    reasons: list[str] = Field(default_factory=list)
    rubric: str = ""
    reply: str = ""


class Panel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: dict[str, Any] = Field(default_factory=dict)
    items: list[PanelItem] = Field(default_factory=list)

    @property
    def raters(self) -> int:
        sizes = {len(item.votes) for item in self.items}
        if not sizes:
            return 0
        if len(sizes) > 1:
            raise ValueError(
                "items carry different numbers of votes (" + ", ".join(map(str, sorted(sizes)))
                + "): a rater skipped something, which is not the same as voting unsure"
            )
        return sizes.pop()


def load_panel(path: str | Path) -> Panel:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    panel = Panel.model_validate(raw)
    seen: set[tuple[str, str]] = set()
    for item in panel.items:
        key = (item.case_id, item.reply_sha)
        if key in seen:
            raise ValueError("the same reply appears twice in " + str(path) + ": " + item.case_id)
        seen.add(key)
    if panel.items and panel.raters < 2:
        raise ValueError("a panel needs at least two raters; " + str(path) + " has one")
    return panel


def default_threshold(raters: int) -> int:
    """At least four in five, rounded up: 4 of 5, 3 of 3, 6 of 7."""
    return math.ceil(0.8 * raters)


def majority_threshold(raters: int) -> int:
    """The smallest number that is more than half."""
    return raters // 2 + 1


def consensus(votes: Sequence[Vote], threshold: int) -> Vote:
    """The panel's label for one reply, or "unsure" when it has none.

    "unsure" is used for a contested reply on purpose: it is the label a single
    careful rater gives when the rubric does not settle the question, and a panel
    that splits has found exactly that.
    """
    if threshold * 2 <= len(votes):
        # Otherwise pass and fail could both reach it and the answer would depend
        # on which was checked first.
        raise ValueError(
            "a threshold of " + str(threshold) + " is not a majority of "
            + str(len(votes)) + " raters"
        )
    counts = Counter(votes)
    for label in ("pass", "fail"):
        if counts[label] >= threshold:
            return label  # type: ignore[return-value]
    return "unsure"


def describe_split(votes: Sequence[Vote]) -> str:
    counts = Counter(votes)
    parts = [str(counts[c]) + " " + c for c in CATEGORIES if counts[c]]
    return ", ".join(parts)


def to_labels(panel: Panel, threshold: int | None = None) -> LabelSet:
    """The panel's votes reduced to one label per reply, in the label format the
    rest of the pipeline already reads."""
    raters = panel.raters
    limit = default_threshold(raters) if threshold is None else threshold
    entries: list[LabelEntry] = []
    for item in panel.items:
        label = consensus(item.votes, limit)
        split = describe_split(item.votes)
        note = (
            "panel of " + str(raters) + ": " + split
            if label != "unsure"
            else "contested, so excluded - panel of " + str(raters) + ": " + split
        )
        entries.append(
            LabelEntry(
                case_id=item.case_id,
                reply_sha=item.reply_sha,
                label=label,
                note=note,
                rubric_sha=item.rubric_sha,
                rubric=item.rubric,
                reply=item.reply,
            )
        )
    return LabelSet(labels=entries)


def fleiss_kappa(table: Sequence[Sequence[int]]) -> float | None:
    """Fleiss' kappa: agreement among several raters, corrected for chance.

    `table` has one row per item and one column per category, each cell the
    number of raters who chose that category for that item. Every row must sum to
    the same number of raters.

    Returns None when it is undefined - no items, a single rater, or every rater
    using one category throughout, where chance agreement is already 1 and there
    is nothing left to correct. Same rule as Cohen's kappa here, for the same
    reason: 0.0 would read as "no better than chance" when the truth is "this set
    cannot say".
    """
    rows = [list(row) for row in table]
    if not rows:
        return None
    raters = sum(rows[0])
    if raters < 2:
        return None
    if any(sum(row) != raters for row in rows):
        raise ValueError("every item must be rated by the same number of raters")

    items = len(rows)
    per_item = [
        (sum(cell * cell for cell in row) - raters) / (raters * (raters - 1)) for row in rows
    ]
    observed = sum(per_item) / items
    categories = len(rows[0])
    shares = [sum(row[j] for row in rows) / (items * raters) for j in range(categories)]
    expected = sum(share * share for share in shares)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def vote_table(panel: Panel) -> list[list[int]]:
    return [[Counter(item.votes)[c] for c in CATEGORIES] for item in panel.items]


def split_counts(panel: Panel) -> dict[str, int]:
    """How settled the panel was, reply by reply."""
    raters = panel.raters
    out = {"unanimous": 0, "one_dissent": 0, "split": 0}
    for item in panel.items:
        top = max(Counter(item.votes).values())
        if top == raters:
            out["unanimous"] += 1
        elif top == raters - 1:
            out["one_dissent"] += 1
        else:
            out["split"] += 1
    return out


def leave_one_out(panel: Panel) -> list[dict[str, Any]]:
    """Each rater against the majority of the others.

    This is what gives the judge's figure a scale. A judge compared with a panel
    is being asked to match a reference its own members do not match perfectly,
    and how well they do match it is the ceiling worth knowing.

    A reply is used for a rater only when the others reach a strict majority on
    pass or fail, and only when the rater in question voted pass or fail. Both
    exclusions are counted, because a rater who says "unsure" a great deal would
    otherwise look agreeable by never being scored.
    """
    raters = panel.raters
    out: list[dict[str, Any]] = []
    for rater in range(raters):
        mine: list[str] = []
        theirs: list[str] = []
        unsure = 0
        no_majority = 0
        for item in panel.items:
            others = [vote for index, vote in enumerate(item.votes) if index != rater]
            reference = consensus(others, majority_threshold(len(others)))
            if reference == "unsure":
                no_majority += 1
                continue
            if item.votes[rater] == "unsure":
                unsure += 1
                continue
            mine.append(item.votes[rater])
            theirs.append(reference)
        n = len(mine)
        agreed = sum(1 for a, b in zip(mine, theirs, strict=True) if a == b)
        out.append(
            {
                "rater": rater + 1,
                "n": n,
                "kappa": cohen_kappa(theirs, mine),  # type: ignore[arg-type]
                "observed_agreement": (agreed / n) if n else None,
                "said_unsure": unsure,
                "others_had_no_majority": no_majority,
            }
        )
    return out


def summarise_panel(panel: Panel, threshold: int | None = None) -> dict[str, Any]:
    raters = panel.raters
    limit = default_threshold(raters) if threshold is None else threshold
    labels = to_labels(panel, limit)
    decided = Counter(entry.label for entry in labels.labels)
    contested = [
        {
            "case_id": item.case_id,
            "reply_sha": item.reply_sha,
            "split": describe_split(item.votes),
        }
        for item in panel.items
        if consensus(item.votes, limit) == "unsure"
    ]
    return {
        "raters": raters,
        "replies": len(panel.items),
        "threshold": limit,
        "fleiss_kappa": fleiss_kappa(vote_table(panel)),
        "splits": split_counts(panel),
        "labelled": {"pass": decided["pass"], "fail": decided["fail"]},
        "contested": contested,
        "leave_one_out": leave_one_out(panel),
    }


def format_panel(summary: dict[str, Any], width: int = 78) -> str:
    raters = summary["raters"]
    lines = ["=" * width, "the rater panel", "=" * width]
    kappa = summary["fleiss_kappa"]
    lines.append(
        str(raters) + " raters on " + str(summary["replies"]) + " replies; Fleiss' kappa "
        + ("undefined (the panel used a single category)" if kappa is None
           else str(round(kappa, 3)))
        + " across pass / fail / unsure"
    )
    splits = summary["splits"]
    lines.append(
        "unanimous " + str(splits["unanimous"]) + "   one dissent " + str(splits["one_dissent"])
        + "   split further " + str(splits["split"])
    )
    labelled = summary["labelled"]
    lines.append(
        "reference: at least " + str(summary["threshold"]) + " of " + str(raters)
        + " must agree  ->  " + str(labelled["pass"]) + " pass, " + str(labelled["fail"])
        + " fail, " + str(len(summary["contested"])) + " contested and excluded"
    )
    lines.append("-" * width)
    lines.append("each rater against the majority of the others")
    for row in summary["leave_one_out"]:
        kappa_text = "-" if row["kappa"] is None else str(round(row["kappa"], 3))
        agree_text = (
            "-" if row["observed_agreement"] is None
            else str(round(row["observed_agreement"] * 100, 1)) + "%"
        )
        lines.append(
            "  rater " + str(row["rater"]) + "   kappa " + kappa_text.ljust(6)
            + " agreement " + agree_text.ljust(7) + " on " + str(row["n"])
            + ("" if not row["said_unsure"] else "   said unsure " + str(row["said_unsure"]))
        )
    if summary["contested"]:
        lines.append("-" * width)
        lines.append("contested replies (a split panel is evidence about the rubric)")
        for item in summary["contested"]:
            lines.append("  " + item["case_id"].ljust(40) + item["split"])
    return "\n".join(lines)


def dump_labels(labels: LabelSet, header: str = "") -> str:
    body = yaml.safe_dump(
        {"labels": [entry.model_dump() for entry in labels.labels]},
        sort_keys=False, width=88, allow_unicode=True, default_flow_style=False,
    )
    return header + body
