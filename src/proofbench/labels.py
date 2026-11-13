"""Hand labels, and keeping them honest.

A judge is only worth reading next to a measurement of how often it agrees with a
person. That means somebody has to sit down and label replies, and it means the
labels have to be pinned to the exact replies they were written about.

So every label carries a hash of the reply it judged. When a later run produces a
different reply - a new model, a changed prompt, or just the run-to-run variance
this suite has plenty of - the hash stops matching and the label is reported as
stale rather than quietly applied to text nobody ever read. Labels are the one
part of an eval that cannot be regenerated, which is exactly why they are the
part most likely to drift out of date without anyone noticing.

`unsure` is a real option. A labeller forced to choose on a reply they cannot
call is a labeller producing noise, and that noise would land in the kappa as
though it were signal. Unsure labels are excluded from agreement and counted
separately, so the reader can see how much of the set the author could not
decide.

The limitation worth stating plainly: on this project the labeller is the author
of the suite and of the rubrics. That makes the kappa a measure of agreement
between the judge and one interested party, not between the judge and an
independent rater, and it should be read as the weaker of the two claims.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Verdict = Literal["pass", "fail", "unsure"]


def reply_sha(reply: str) -> str:
    """Identify a reply by its text.

    Whitespace is normalised at the edges only: a model that changes its
    indentation has not changed what it said, but one that changes a word has.
    """
    return hashlib.sha256((reply or "").strip().encode("utf-8")).hexdigest()[:16]


class LabelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    reply_sha: str
    label: Verdict
    note: str = ""
    # Carried for the labeller's benefit and for reading a diff later. Never used
    # for matching, so editing them cannot change which reply a label applies to.
    rubric: str = ""
    reply: str = ""


class LabelSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: list[LabelEntry] = Field(default_factory=list)

    def by_key(self) -> dict[tuple[str, str], LabelEntry]:
        return {(entry.case_id, entry.reply_sha): entry for entry in self.labels}

    @property
    def decided(self) -> list[LabelEntry]:
        return [entry for entry in self.labels if entry.label != "unsure"]


class UnlabelledEntries(Exception):
    """Raised when a template was only partly filled in."""


def load_labels(path: str | Path) -> LabelSet:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if isinstance(raw, list):
        raw = {"labels": raw}

    # Caught here, by name, rather than left to a validation error listing a
    # Literal's allowed values. A half-filled label file is the likeliest mistake
    # in this whole flow and the message should say which rows are missing.
    outstanding = [
        str(entry.get("case_id"))
        for entry in raw.get("labels", [])
        if isinstance(entry, dict) and entry.get("label") in {"TODO", None, ""}
    ]
    if outstanding:
        raise UnlabelledEntries(
            str(len(outstanding)) + " entries in " + str(path) + " still say TODO: "
            + ", ".join(outstanding[:8])
            + ("..." if len(outstanding) > 8 else "")
        )
    return LabelSet.model_validate(raw)


def distinct_replies(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry per distinct (case, reply) in a recorded run.

    Repeats of a case come back identical on this model, so labelling every
    attempt would mean labelling the same sentence three times and then counting
    it three times in the kappa - inflating the sample with copies and making the
    interval look narrower than the evidence is.
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for attempt in artifact.get("attempts", []):
        reply = (attempt.get("run", {}).get("reply") or "").strip()
        if not reply:
            continue
        key = (attempt["case_id"], reply_sha(reply))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "case_id": attempt["case_id"],
                "reply_sha": key[1],
                "reply": reply,
                "attempts": 1,
            }
        )
    # Count how many attempts each distinct reply stood for, so a reader can see
    # that one label covered three identical runs.
    for entry in out:
        entry["attempts"] = sum(
            1
            for attempt in artifact.get("attempts", [])
            if attempt["case_id"] == entry["case_id"]
            and reply_sha(attempt.get("run", {}).get("reply") or "") == entry["reply_sha"]
        )
    return out


def template(artifact: dict[str, Any], rubrics: dict[str, str]) -> str:
    """A YAML file with every distinct reply and a blank label, ready to fill in.

    The reply and its rubric are written in full so that labelling needs nothing
    but this file - no cross-referencing an artifact while deciding, which is how
    labelling gets rushed.
    """
    entries = distinct_replies(artifact)
    meta = artifact.get("meta", {})
    header = "\n".join(
        [
            "# Hand labels for the judge.",
            "#",
            "# One entry per distinct reply in " + str(meta.get("model", "?"))
            + " run " + str(meta.get("git_sha", "?")) + ".",
            "# Replace every `label: TODO` with pass, fail or unsure.",
            "#",
            "# Judge the REPLY against the RUBRIC, and nothing else. Whether the agent",
            "# refunded the right amount is decided by the deterministic graders and is",
            "# not what this measures. unsure is a real answer: it is excluded from the",
            "# agreement rather than guessed at.",
            "#",
            "# reply_sha pins each label to the exact text it was written about. If a",
            "# later run says something different the label goes stale and is reported,",
            "# not silently reused.",
            "",
        ]
    )
    payload = {
        "labels": [
            {
                "case_id": entry["case_id"],
                "reply_sha": entry["reply_sha"],
                # Not left blank and not defaulted to a verdict: a blank would
                # fail validation with an unhelpful message, and defaulting to
                # "unsure" would let a forgotten entry quietly leave the sample.
                "label": "TODO",
                "note": "",
                "rubric": rubrics.get(entry["case_id"], ""),
                "reply": entry["reply"],
            }
            for entry in entries
        ]
    }
    body = yaml.safe_dump(
        payload, sort_keys=False, width=88, allow_unicode=True, default_flow_style=False
    )
    return header + body


class Alignment(BaseModel):
    """Labels matched against a set of judgements, with everything that failed to
    match reported rather than dropped."""

    model_config = ConfigDict(extra="forbid")

    human: list[str] = Field(default_factory=list)
    judge: list[str] = Field(default_factory=list)
    keys: list[tuple[str, str]] = Field(default_factory=list)
    unsure: list[tuple[str, str]] = Field(default_factory=list)
    # A label whose reply is nowhere in this judgement set: the run said something
    # different, so the label no longer applies to anything.
    stale_labels: list[tuple[str, str]] = Field(default_factory=list)
    # A label whose reply *was* judged, but the judge produced no verdict. Kept
    # apart from stale on purpose: the first version lumped them together and
    # reported two judge timeouts as "the reply changed since they were written",
    # which is a confident diagnosis of the wrong problem.
    judge_abstained: list[tuple[str, str]] = Field(default_factory=list)
    unlabelled: list[tuple[str, str]] = Field(default_factory=list)

    def describe(self) -> str:
        parts = [str(len(self.human)) + " replies labelled and judged"]
        if self.unsure:
            parts.append(str(len(self.unsure)) + " marked unsure by the labeller and excluded")
        if self.judge_abstained:
            parts.append(
                str(len(self.judge_abstained)) + " the judge did not decide (excluded; "
                "a judge failure, not a disagreement)"
            )
        if self.stale_labels:
            parts.append(
                str(len(self.stale_labels)) + " labels are stale (the reply changed "
                "since they were written)"
            )
        if self.unlabelled:
            parts.append(str(len(self.unlabelled)) + " judged replies have no label")
        return "; ".join(parts)


def align(
    labels: LabelSet,
    verdicts: dict[tuple[str, str], str],
    judged_replies: set[tuple[str, str]] | None = None,
) -> Alignment:
    """Pair each label with the judge's verdict on the same reply.

    `verdicts` holds decided verdicts only. `judged_replies` is every reply the
    judge was actually shown, including the ones it abstained on, and is what
    makes it possible to tell a stale label from a judge that failed. Without it
    the two are indistinguishable, and the wrong one gets reported.
    """
    result = Alignment()
    label_keys = labels.by_key()
    seen = judged_replies if judged_replies is not None else set(verdicts)

    for key, entry in label_keys.items():
        if key not in seen:
            result.stale_labels.append(key)
            continue
        if entry.label == "unsure":
            result.unsure.append(key)
            continue
        verdict = verdicts.get(key)
        if verdict is None:
            result.judge_abstained.append(key)
            continue
        result.keys.append(key)
        result.human.append(entry.label)
        result.judge.append(verdict)

    for key in seen:
        if key not in label_keys:
            result.unlabelled.append(key)
    return result
