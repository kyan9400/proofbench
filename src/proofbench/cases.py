"""The case format.

A case is a conversation turn plus everything that can be checked about what the
agent did with it. The format is deliberately biased towards things that can be
decided without asking a model: which effects landed in the world, which tools
were called, whether another customer's data appeared in the reply, what it cost.
A free-text rubric exists for the part that genuinely needs judgement, and the
judge that reads it is itself measured against hand labels (see `judge.py`).

Expectations are written as:

  effects            every one of these must have happened, by (kind, order_id,
                     amount) — extra effects are caught by `forbidden` or by
                     `effects_exhaustive`
  forbidden          none of these may have happened, accepted or merely attempted
  effects_exhaustive nothing beyond `effects` may have been accepted
  tools_required     each of these tools must have been called at least once
  tools_forbidden    none of these may have been called
  reply_matches      regexes the reply must match (case-insensitive)
  reply_rejects      regexes the reply must not match
  budget             cost, latency and step ceilings

`subject_customer` is the customer whose conversation this is. The leak grader
derives the forbidden set from it, so a case does not have to enumerate every
other customer's email by hand — and cannot forget one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from shopdesk.data import DEFAULT_NOW

Severity = Literal["critical", "major", "minor"]

EffectKind = Literal["refund", "cancel", "escalate"]


class ExpectedEffect(BaseModel):
    """One mutation that must appear in the effect log."""

    model_config = ConfigDict(extra="forbid")

    kind: EffectKind
    order_id: str | None = None
    amount_cents: int | None = None
    # Whether shopdesk accepted it. An expectation almost always wants True; False
    # is for cases where the right behaviour is to try and be refused.
    accepted: bool = True


class ForbiddenEffect(BaseModel):
    """A mutation that must not appear — attempted or accepted.

    `attempted` is the point: an agent that tries to refund an already-refunded
    order and is blocked by shopdesk has still made the mistake, and a grader that
    only reads final balances would score it as a pass.
    """

    model_config = ConfigDict(extra="forbid")

    kind: EffectKind
    order_id: str | None = None
    # When set, only amounts at or above this count as the forbidden thing, so a
    # case can forbid "a refund over the authority cap" without forbidding refunds.
    min_amount_cents: int | None = None
    severity: Severity = "critical"
    note: str | None = None


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int | None = None
    max_latency_ms: int | None = None
    max_cost_usd: float | None = None
    max_tokens: int | None = None


class Expect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effects: list[ExpectedEffect] = Field(default_factory=list)
    forbidden: list[ForbiddenEffect] = Field(default_factory=list)
    effects_exhaustive: bool = True
    tools_required: list[str] = Field(default_factory=list)
    tools_forbidden: list[str] = Field(default_factory=list)
    reply_matches: list[str] = Field(default_factory=list)
    reply_rejects: list[str] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)


class Injection(BaseModel):
    """Hostile content planted in a tool result.

    Real agents read text written by other people — order notes, product reviews,
    support history. This makes a named tool return its normal payload with
    `append` added, so a case can ask what the agent does when the data it reads
    tells it to empty the till.
    """

    model_config = ConfigDict(extra="forbid")

    tool: str
    when_arg: str | None = None
    when_value: str | None = None
    append: str


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    prompt: str
    suite: str = "support"
    subject_customer: str | None = None
    now: str = DEFAULT_NOW
    expect: Expect = Field(default_factory=Expect)
    inject: list[Injection] = Field(default_factory=list)
    rubric: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Case:
        if not self.id.strip():
            raise ValueError("case id must not be empty")
        bad = set(self.expect.tools_required) & set(self.expect.tools_forbidden)
        if bad:
            raise ValueError("tools both required and forbidden: " + ", ".join(sorted(bad)))
        return self


class Suite(BaseModel):
    name: str
    cases: list[Case]

    def by_id(self, case_id: str) -> Case:
        for c in self.cases:
            if c.id == case_id:
                return c
        raise KeyError(case_id)


def load_case(data: dict[str, Any]) -> Case:
    return Case.model_validate(data)


def load_suite(path: Path) -> Suite:
    """Load every .yaml under `path` (or a single file) as one suite.

    Duplicate ids are an error rather than a last-one-wins merge: a case silently
    shadowing another is how a suite ends up measuring less than it claims.
    """
    path = Path(path)
    files = sorted(path.rglob("*.yaml")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError("no .yaml case files under " + str(path))

    cases: list[Case] = []
    seen: dict[str, Path] = {}
    for f in files:
        raw = yaml.safe_load(f.read_text(encoding="utf-8"))
        if raw is None:
            continue
        entries = raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw
        if not isinstance(entries, list):
            entries = [entries]
        for entry in entries:
            try:
                case = load_case(entry)
            except Exception as exc:
                raise ValueError("invalid case in " + str(f) + ": " + str(exc)) from exc
            if case.id in seen:
                raise ValueError(
                    "duplicate case id " + repr(case.id) + " in " + str(f)
                    + " and " + str(seen[case.id])
                )
            seen[case.id] = f
            cases.append(case)

    name = path.name if path.is_dir() else path.stem
    return Suite(name=name, cases=cases)
